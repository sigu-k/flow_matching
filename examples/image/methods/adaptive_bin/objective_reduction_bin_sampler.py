"""
Objective-reduction adaptive bin sampler for flow-matching training-time t.

Splits t in [0,1] into K equal-width bins. A bin is scored by how much updating
the UNet *on that bin* reduces the flow-matching objective, measured on a FIXED
eval set at the K bin centers S = {(i+0.5)/K}. To stabilise that (noisy) reward
and let the sampling distribution move more, this version adds:

  * fixed eval samples / noise (reward measured against a frozen reference set,
    not the current training batch);
  * a delta queue of recent per-eval-time loss reductions;
  * an SNR-based selected eval subset (reward uses only the eval times whose
    reduction signal is most consistent: score_j = |mean_j| / (std_j + eps));
  * a reward_scale knob (so softmax(reward_scale * reward_ema / T) can spread).

    relative_reward = mean(before[S] - after[S]) / (mean(before[S]) + eps)
    reward_ema[b]   = beta * reward_ema[b] + (1 - beta) * relative_reward
    p_reward        = softmax(reward_scale * reward_ema / temperature)
    p               = (1 - uniform_mix) * p_reward + uniform_mix / K

There are NO learnable parameters / NO optimizer / NO bin_logits. `uniform_mix`
guarantees the exploration floor (each bin >= uniform_mix/K). During warmup
(global_step < warmup_steps) sampling is fully uniform, but reward_ema / the
queue are still updated. 1-batch-1-bin is preserved.

Self-contained: does not touch adaptive_bin_sampler.py, loss_aware_bin_sampler.py
or timestep_sampler.py. See methods/adaptive_bin/2.md for the spec.
"""

import logging
from collections import deque

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_fm_loss_at_bin_centers(eval_samples, eval_noise, path, model, k):
    """Mean flow-matching loss at each of the K bin centers over a FIXED eval set.

    `eval_samples` / `eval_noise` are lists of equally-shaped batches (already on
    device, samples in [-1,1]). For bin j the eval timestep is tau = (j+0.5)/K;
    x_t is built from each fixed (samples, noise) batch and the loss is
    (pred - dx_t)^2.mean(), averaged over ALL eval images (sample-weighted).
    Returns a tensor of shape [K]. `model` should be the RAW UNet (not EMA). No
    gradients are needed, hence torch.no_grad().
    """
    device = eval_samples[0].device
    totals = torch.zeros(k, device=device)
    n_total = 0
    for samples, noise in zip(eval_samples, eval_noise):
        bs = samples.shape[0]
        for j in range(k):
            tau = (j + 0.5) / k
            t_eval = torch.full((bs,), float(tau), device=device)
            ps = path.sample(t=t_eval, x_0=noise, x_1=samples)
            pred = model(ps.x_t, t_eval, extra={})
            totals[j] += (pred - ps.dx_t).pow(2).mean() * bs
        n_total += bs
    return totals / max(n_total, 1)


class ObjectiveReductionBinSampler:
    """Objective-reduction categorical distribution over K timestep bins."""

    def __init__(self, k=10, reward_ema_beta=0.9, temperature=0.5,
                 uniform_mix=0.1, update_sampler_every=40, warmup_steps=1000,
                 importance_gamma=0.5, reward_scale=100.0, queue_size=20,
                 selected_eval_count=3, selection_method="snr_topk",
                 sampler_eval_batches=4, eps=1e-8, device="cuda"):
        self.k = k
        self.reward_ema_beta = reward_ema_beta
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.update_sampler_every = update_sampler_every
        self.warmup_steps = warmup_steps
        self.importance_gamma = importance_gamma
        self.reward_scale = reward_scale
        self.queue_size = queue_size
        self.selected_eval_count = selected_eval_count
        self.selection_method = selection_method
        self.sampler_eval_batches = sampler_eval_batches
        self.eps = eps
        self.device = device

        # State (no gradients / no optimizer).
        self.reward_ema = torch.zeros(k, device=device)
        self.last_probs = self._compute_probs()  # uniform while reward_ema == 0
        self.delta_queue = deque(maxlen=queue_size)  # each entry: tensor [K]
        self.selected_bin = None
        self.selected_prob = 1.0 / k
        self.num_sampler_updates = 0
        self.round_robin_idx = 0
        self.last_reward = 0.0
        self.last_eval_loss_before = 0.0
        self.last_eval_loss_after = 0.0
        self.last_importance_weight = 1.0

    # -- distribution helpers -------------------------------------------------
    def _compute_probs(self):
        scaled = self.reward_scale * self.reward_ema
        p_reward = torch.softmax(scaled / self.temperature, dim=0)
        probs = (1.0 - self.uniform_mix) * p_reward + self.uniform_mix / self.k
        return probs

    def probs(self):
        return self.last_probs

    def entropy(self):
        p = self.last_probs
        return -(p * torch.log(p + 1e-8)).sum()

    def bin_prob_dict(self):
        p = self.last_probs.detach().cpu()
        return {f"sampler/bin_prob_{i}": float(p[i]) for i in range(self.k)}

    def reward_ema_dict(self):
        r = self.reward_ema.detach().cpu()
        return {f"sampler/reward_ema_{i}": float(r[i]) for i in range(self.k)}

    def scaled_reward_ema_dict(self):
        r = (self.reward_scale * self.reward_ema).detach().cpu()
        return {f"sampler/scaled_reward_ema_{i}": float(r[i]) for i in range(self.k)}

    # -- sampling -------------------------------------------------------------
    def sample_t(self, batch_size, device, global_step=0, force_bin=None):
        """Pick ONE bin, then draw batch_size uniform t inside it.

        force_bin overrides the choice (round-robin forced exploration on
        sampler-update steps). Otherwise the bin is drawn uniformly during
        warmup (global_step < warmup_steps) and from last_probs afterwards.
        Records selected_bin and selected_prob (q_b under the sampling
        distribution actually in effect: 1/K during warmup, last_probs[b]
        afterwards) for importance weighting.
        """
        in_warmup = global_step < self.warmup_steps
        if force_bin is not None:
            bin_idx = int(force_bin)
        elif in_warmup:
            p = torch.full((self.k,), 1.0 / self.k, device=self.device)
            bin_idx = int(torch.multinomial(p, 1).item())
        else:
            bin_idx = int(torch.multinomial(self.last_probs, 1).item())

        q_b = 1.0 / self.k if in_warmup else float(self.last_probs[bin_idx].item())

        self.selected_bin = bin_idx
        self.selected_prob = q_b

        lo = bin_idx / self.k
        u = torch.rand(batch_size, device=device)
        return (lo + u / self.k).clamp(1e-6, 1 - 1e-6)

    def importance_weight(self):
        """Partial importance weight (1 / (K * q_b)) ** gamma for the loss.

        With q_b = 1/K (uniform warmup sampling) this is exactly 1.
        """
        w = (1.0 / (self.k * self.selected_prob)) ** self.importance_gamma
        self.last_importance_weight = float(w)
        return float(w)

    # -- eval-time subset selection -------------------------------------------
    def _select_eval_subset(self):
        """Pick the eval-time subset S used to compute the reward.

        Returns (reward_indices, log_indices, scores[K], using_all):
          * reward_indices : indices the reward is averaged over (all K when the
            queue is too short for a stable std, else the SNR top-k).
          * log_indices    : exactly selected_eval_count indices for W&B keys.
          * scores         : per-eval-time SNR score (zeros while using_all).
          * using_all      : True while the queue has < 2 entries.
        Selection uses the queue as it stands BEFORE the current delta is
        appended (causal: a step's own delta never drives its own selection).
        """
        K = self.k
        n = min(self.selected_eval_count, K)
        if len(self.delta_queue) < 2:
            scores = torch.zeros(K, device=self.device)
            return list(range(K)), list(range(n)), scores, True

        Q = torch.stack(list(self.delta_queue), dim=0)  # [Q, K]
        mean = Q.mean(dim=0)
        std = Q.std(dim=0)
        scores = mean.abs() / (std + self.eps)
        top = torch.topk(scores, n).indices
        idx = sorted(int(i) for i in top.tolist())
        return idx, idx, scores, False

    # -- objective-reduction update -------------------------------------------
    def update(self, before_losses, after_losses):
        """Update reward_ema for the selected bin from a before/after eval pair.

        before/after_losses are tensors [K] (per-eval-time FM loss on the fixed
        eval set). The reward is the relative reduction averaged over the
        selected eval subset S; only the forced bin's reward_ema changes; the
        per-eval-time delta is then queued and round_robin_idx advances. Returns
        a logging dict.
        """
        b = self.selected_bin
        if b is None:
            raise RuntimeError("update() called before sample_t()")

        before_losses = before_losses.detach()
        after_losses = after_losses.detach()

        reward_indices, log_indices, scores, using_all = self._select_eval_subset()
        S = torch.tensor(reward_indices, device=self.device)
        selected_delta_mean = (before_losses[S] - after_losses[S]).mean()
        selected_before_mean = before_losses[S].mean()
        relative_reward = float(
            (selected_delta_mean / (selected_before_mean + self.eps)).item()
        )

        self.reward_ema[b] = (self.reward_ema_beta * self.reward_ema[b]
                              + (1.0 - self.reward_ema_beta) * relative_reward)
        self.last_probs = self._compute_probs()

        delta = (before_losses - after_losses)
        self.delta_queue.append(delta)  # deque(maxlen) auto-trims to queue_size

        self.last_reward = relative_reward
        self.last_eval_loss_before = float(selected_before_mean.item())
        self.last_eval_loss_after = float(after_losses[S].mean().item())
        self.num_sampler_updates += 1
        self.round_robin_idx = (self.round_robin_idx + 1) % self.k

        return {
            "reward": relative_reward,
            "relative_reward": relative_reward,
            "selected_delta_mean": float(selected_delta_mean.item()),
            "selected_before_mean": float(selected_before_mean.item()),
            "eval_loss_before": float(selected_before_mean.item()),
            "eval_loss_after": float(after_losses[S].mean().item()),
            "eval_loss_mean_before": float(before_losses.mean().item()),
            "eval_loss_mean_after": float(after_losses.mean().item()),
            "delta": delta.detach().cpu(),               # [K]
            "scores": scores.detach().cpu(),             # [K]
            "log_indices": log_indices,                  # len selected_eval_count
            "using_all": int(using_all),
            "queue_len": len(self.delta_queue),
        }

    # -- checkpoint I/O -------------------------------------------------------
    def state_dict(self):
        return {
            "type": "objective_reduction_bin",
            "k": self.k,
            "reward_ema": self.reward_ema.detach().cpu(),
            "last_probs": self.last_probs.detach().cpu(),
            "delta_queue": [d.detach().cpu() for d in self.delta_queue],
            "selected_bin": self.selected_bin,
            "selected_prob": self.selected_prob,
            "num_sampler_updates": self.num_sampler_updates,
            "round_robin_idx": self.round_robin_idx,
            "last_reward": self.last_reward,
            "last_eval_loss_before": self.last_eval_loss_before,
            "last_eval_loss_after": self.last_eval_loss_after,
            "last_importance_weight": self.last_importance_weight,
            # hyperparameters (restored so a resume reproduces the run)
            "reward_ema_beta": self.reward_ema_beta,
            "temperature": self.temperature,
            "uniform_mix": self.uniform_mix,
            "update_sampler_every": self.update_sampler_every,
            "warmup_steps": self.warmup_steps,
            "importance_gamma": self.importance_gamma,
            "reward_scale": self.reward_scale,
            "queue_size": self.queue_size,
            "selected_eval_count": self.selected_eval_count,
            "selection_method": self.selection_method,
            "sampler_eval_batches": self.sampler_eval_batches,
        }

    def load_state_dict(self, sd):
        sd_type = sd.get("type")
        if sd_type != "objective_reduction_bin":
            logger.warning(
                "sampler_state type mismatch (got %r, expected "
                "'objective_reduction_bin') – skipping sampler restore", sd_type
            )
            return
        if sd.get("k") != self.k:
            logger.warning(
                "sampler_state k mismatch (got %r, expected %d) – skipping "
                "sampler restore", sd.get("k"), self.k
            )
            return

        self.reward_ema = sd["reward_ema"].to(self.device)
        self.selected_bin = sd["selected_bin"]
        self.selected_prob = sd["selected_prob"]
        self.num_sampler_updates = int(sd["num_sampler_updates"])
        self.round_robin_idx = int(sd["round_robin_idx"])
        self.last_reward = float(sd["last_reward"])
        self.last_eval_loss_before = float(sd["last_eval_loss_before"])
        self.last_eval_loss_after = float(sd["last_eval_loss_after"])
        self.last_importance_weight = float(sd["last_importance_weight"])

        # New fields: default-initialise when absent (old checkpoints).
        self.queue_size = int(sd.get("queue_size", self.queue_size))
        self.selected_eval_count = int(sd.get("selected_eval_count", self.selected_eval_count))
        self.selection_method = sd.get("selection_method", self.selection_method)
        self.reward_scale = float(sd.get("reward_scale", self.reward_scale))
        self.sampler_eval_batches = int(sd.get("sampler_eval_batches", self.sampler_eval_batches))

        saved_queue = sd.get("delta_queue", [])
        self.delta_queue = deque(
            (d.to(self.device) for d in saved_queue), maxlen=self.queue_size
        )

        # last_probs may be stale vs restored hyperparameters; recompute.
        self.last_probs = self._compute_probs()
