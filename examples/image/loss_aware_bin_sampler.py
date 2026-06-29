"""
Loss-aware adaptive bin sampler for flow-matching training-time timestep t.

Splits t in [0,1] into K equal-width bins. Unlike the REINFORCE-based
AdaptiveBinSampler (adaptive_bin_sampler.py), this sampler has NO learnable
parameters and NO optimizer. Instead it periodically *measures* the current
flow-matching loss in every bin (at the bin center) and builds a categorical
sampling distribution that puts more mass on the bins whose loss is high:

    score  = zscore(loss_ema)
    p_loss = softmax(score / temperature)
    p      = (1 - uniform_mix) * p_loss + uniform_mix / K

The only learned state is an EMA of the per-bin loss (loss_ema). `uniform_mix`
alone guarantees the exploration floor (~uniform_mix/K per bin); there is no
min/max-prob clamp.

Each batch samples ONE bin (1-batch-1-bin), then draws batch_size uniform t
inside that bin. During warmup (global_step < warmup_steps) sampling is fully
uniform regardless of last_probs.

This file is self-contained and does not touch adaptive_bin_sampler.py or the
single-distribution samplers in timestep_sampler.py.
"""

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_fm_loss_per_bin(k, samples, noise, path, model):
    """Per-bin flow-matching loss measured at each bin center.

    For bin i the eval timestep is t = (i + 0.5) / K. x_t is built from the SAME
    (samples, noise) pair used by the current training batch and the loss is
    (pred - dx_t)^2.mean(). Returns a tensor of shape [K]. Gradients are not
    needed (sampler loss evaluation only), hence torch.no_grad().
    """
    bs = samples.shape[0]
    device = samples.device
    losses = []
    for i in range(k):
        center = (i + 0.5) / k
        t_eval = torch.full((bs,), float(center), device=device)
        ps = path.sample(t=t_eval, x_0=noise, x_1=samples)
        pred = model(ps.x_t, t_eval, extra={})
        losses.append((pred - ps.dx_t).pow(2).mean())
    return torch.stack(losses)


class LossAwareBinSampler:
    """Loss-aware (parameter-free) categorical distribution over K timestep bins."""

    def __init__(self, k=10, temperature=0.25, uniform_mix=0.05,
                 loss_ema_beta=0.8, update_sampler_every=500,
                 warmup_steps=1000, device="cuda"):
        self.k = k
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.loss_ema_beta = loss_ema_beta
        self.update_sampler_every = update_sampler_every
        self.warmup_steps = warmup_steps
        self.device = device

        # Learned state (no gradients / no optimizer).
        self.loss_ema = torch.zeros(k, device=device)
        self.initialized = False
        self.num_sampler_updates = 0
        self.last_probs = torch.full((k,), 1.0 / k, device=device)
        self.last_bin_losses = torch.zeros(k, device=device)

        self._last_bin = None  # bin chosen by the most recent sample_t()

    # -- distribution helpers -------------------------------------------------
    def probs(self):
        return self.last_probs

    def entropy(self):
        p = self.last_probs
        return -(p * torch.log(p + 1e-8)).sum()

    def bin_prob_dict(self):
        p = self.last_probs.detach().cpu()
        return {f"sampler/bin_prob_{i}": float(p[i]) for i in range(self.k)}

    def bin_loss_dict(self):
        bl = self.last_bin_losses.detach().cpu()
        ema = self.loss_ema.detach().cpu()
        d = {f"sampler/bin_loss_{i}": float(bl[i]) for i in range(self.k)}
        d |= {f"sampler/bin_loss_ema_{i}": float(ema[i]) for i in range(self.k)}
        return d

    # -- sampling -------------------------------------------------------------
    def sample_t(self, batch_size, device, global_step=0):
        """Pick ONE bin, then draw batch_size uniform t inside it.

        During warmup (global_step < warmup_steps) the bin is drawn from a fully
        uniform distribution; afterwards from last_probs.
        """
        if global_step < self.warmup_steps:
            p = torch.full((self.k,), 1.0 / self.k, device=self.device)
        else:
            p = self.last_probs
        bin_idx = int(torch.multinomial(p, 1).item())
        self._last_bin = bin_idx
        lo = bin_idx / self.k
        u = torch.rand(batch_size, device=device)
        return (lo + u / self.k).clamp(1e-6, 1 - 1e-6)

    # -- loss-aware update ----------------------------------------------------
    def update(self, bin_losses):
        """Update loss_ema and last_probs from freshly measured per-bin losses.

        `bin_losses` is a tensor of shape [K] (see eval_fm_loss_per_bin). On the
        very first update loss_ema is initialized to the measured losses; later
        updates apply the EMA. Returns a logging dict (no reward/advantage —
        those do not exist for the loss-aware sampler).
        """
        bin_losses = bin_losses.detach().to(self.device)

        if not self.initialized:
            self.loss_ema = bin_losses.clone()
            self.initialized = True
        else:
            self.loss_ema = (self.loss_ema_beta * self.loss_ema
                             + (1.0 - self.loss_ema_beta) * bin_losses)
        self.last_bin_losses = bin_losses.clone()

        score = (self.loss_ema - self.loss_ema.mean()) / (self.loss_ema.std() + 1e-8)
        p_loss = torch.softmax(score / self.temperature, dim=0)
        p = (1.0 - self.uniform_mix) * p_loss + self.uniform_mix / self.k
        self.last_probs = p
        self.num_sampler_updates += 1

        entropy = -(p * torch.log(p + 1e-8)).sum()
        return {
            "entropy": float(entropy.item()),
            "max_prob": float(p.max().item()),
            "min_prob": float(p.min().item()),
            "temperature": self.temperature,
            "uniform_mix": self.uniform_mix,
            "loss_ema_beta": self.loss_ema_beta,
        }

    # -- checkpoint I/O -------------------------------------------------------
    def state_dict(self):
        return {
            "type": "loss_aware_bin",
            "loss_ema": self.loss_ema.detach().cpu(),
            "initialized": self.initialized,
            "num_sampler_updates": self.num_sampler_updates,
            "last_probs": self.last_probs.detach().cpu(),
            "last_bin_losses": self.last_bin_losses.detach().cpu(),
        }

    def load_state_dict(self, sd):
        sd_type = sd.get("type")
        if sd_type != "loss_aware_bin":
            logger.warning(
                "sampler_state type mismatch (got %r, expected 'loss_aware_bin') "
                "– skipping sampler restore", sd_type
            )
            return
        self.loss_ema = sd["loss_ema"].to(self.device)
        self.initialized = bool(sd["initialized"])
        self.num_sampler_updates = int(sd["num_sampler_updates"])
        self.last_probs = sd["last_probs"].to(self.device)
        self.last_bin_losses = sd["last_bin_losses"].to(self.device)
