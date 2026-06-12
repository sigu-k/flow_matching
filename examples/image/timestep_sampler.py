"""
Timestep samplers for flow matching training.

Convention: t=0 → noise, t=1 → data  (x_t = (1-t)·noise + t·data)
All functions return shape [batch_size], values clipped to (1e-6, 1-1e-6).
"""

import torch


def sample_uniform(batch_size: int, device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def sample_logit_normal(batch_size: int, mu: float, sigma: float, device) -> torch.Tensor:
    """LN(μ, σ²): u ~ N(μ, σ²), t = sigmoid(u)."""
    u = torch.randn(batch_size, device=device) * sigma + mu
    return torch.sigmoid(u).clamp(1e-6, 1 - 1e-6)


def sample_mode(batch_size: int, s: float, device) -> torch.Tensor:
    """Mode/Beta(s+1, s+1) sampler.
    s < 0: U-shaped (peaks at boundaries)
    s > 0: bell-shaped (peaks at centre)
    """
    alpha = s + 1.0
    if alpha <= 0:
        raise ValueError(f"s must be > -1, got s={s}")
    concentration = torch.tensor(alpha, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(concentration, concentration)
    return dist.sample((batch_size,)).clamp(1e-6, 1 - 1e-6)
