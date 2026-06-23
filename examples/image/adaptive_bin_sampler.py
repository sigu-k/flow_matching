"""
Adaptive bin sampler for flow-matching training-time timestep t.

Splits t in [0,1] into K equal-width bins and learns a categorical distribution
over bins via REINFORCE. Each batch samples ONE bin (1-batch-1-bin), then draws
batch_size uniform t inside that bin. The bin distribution is updated only every
`update_sampler_every` optimizer steps, using a reward defined as the drop in
flow-matching loss at a fixed set of evaluation timesteps S, measured before and
after the UNet update:

    reward = eval_loss_before - eval_loss_after   (>0 means the chosen bin helped)

This file is self-contained and does not touch the existing single-distribution
samplers in timestep_sampler.py.
"""

import torch


@torch.no_grad()
def eval_fm_loss_at_times(times, samples, noise, path, model):
    """Mean flow-matching loss over a fixed set of evaluation timesteps S.

    For each tau in `times`, build x_t from the SAME (samples, noise) pair and
    measure (pred - dx_t)^2.mean(), then average over the times. Used only to
    produce the sampler reward, so gradients are not needed.
    """
    bs = samples.shape[0]
    losses = []
    for tau in times:
        t_eval = torch.full((bs,), float(tau), device=samples.device)
        ps = path.sample(t=t_eval, x_0=noise, x_1=samples)
        pred = model(ps.x_t, t_eval, extra={})
        losses.append((pred - ps.dx_t).pow(2).mean())
    return torch.stack(losses).mean()


class AdaptiveBinSampler:
    """Learns a categorical distribution over K timestep bins via REINFORCE."""

    def __init__(self, k=10, sampler_lr=1e-3, baseline_beta=0.9,
                 entropy_coef=0.01, update_sampler_every=40,
                 eval_times=(0.1, 0.5, 0.9), reward_scale=100.0, device="cuda"):
        self.k = k
        self.bin_logits = torch.nn.Parameter(torch.zeros(k, device=device))
        self.optimizer = torch.optim.Adam([self.bin_logits], lr=sampler_lr)
        self.baseline = 0.0
        self.baseline_beta = baseline_beta
        self.entropy_coef = entropy_coef
        self.update_sampler_every = update_sampler_every
        self.eval_times = list(eval_times)
        self.reward_scale = reward_scale
        self.device = device
        self._last_bin = None  # bin chosen by the most recent sample_t()

    # -- distribution helpers -------------------------------------------------
    def probs(self):
        return torch.softmax(self.bin_logits, dim=0)

    def entropy(self):
        p = self.probs()
        return -(p * torch.log(p + 1e-8)).sum()

    def bin_prob_dict(self):
        p = self.probs().detach().cpu()
        return {f"sampler/bin_prob_{i}": float(p[i]) for i in range(self.k)}

    # -- sampling -------------------------------------------------------------
    def sample_t(self, batch_size, device):
        """Pick ONE bin (categorical), then draw batch_size uniform t inside it."""
        p = self.probs().detach()
        bin_idx = int(torch.multinomial(p, 1).item())
        self._last_bin = bin_idx
        lo = bin_idx / self.k
        u = torch.rand(batch_size, device=device)
        return (lo + u / self.k).clamp(1e-6, 1 - 1e-6)

    # -- policy-gradient update ----------------------------------------------
    def update(self, eval_loss_before, eval_loss_after):
        """One REINFORCE step for the bin distribution. Returns a logging dict."""
        assert self._last_bin is not None, "sample_t() must be called before update()"
        reward = float(eval_loss_before) - float(eval_loss_after)
        self.baseline = (self.baseline_beta * self.baseline
                         + (1.0 - self.baseline_beta) * reward)
        advantage = reward - self.baseline

        p = self.probs()
        log_prob = torch.log(p[self._last_bin] + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        # advantage is a python float (detached by construction); gradient flows
        # only through log_prob and entropy (i.e. through bin_logits). reward_scale
        # amplifies the (typically tiny) advantage so the bins actually move.
        scaled_advantage = self.reward_scale * advantage
        sampler_loss = -scaled_advantage * log_prob - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        sampler_loss.backward()
        self.optimizer.step()

        return {
            "reward": reward,
            "advantage": advantage,
            "scaled_advantage": scaled_advantage,
            "reward_scale": self.reward_scale,
            "entropy": float(entropy.item()),
            "selected_bin": self._last_bin,
            "eval_loss_before": float(eval_loss_before),
            "eval_loss_after": float(eval_loss_after),
        }

    # -- checkpoint I/O -------------------------------------------------------
    def state_dict(self):
        return {
            "bin_logits": self.bin_logits.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.bin_logits.copy_(sd["bin_logits"].to(self.bin_logits.device))
        self.optimizer.load_state_dict(sd["optimizer"])
        self.baseline = sd.get("baseline", 0.0)
