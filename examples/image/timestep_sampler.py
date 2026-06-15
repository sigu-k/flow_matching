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


def sample_linear_decreasing(batch_size: int, floor: float, device) -> torch.Tensor:
    """t=0 で peak、t=1 で floor まで線形減少する分布からサンプリング。

    p(t) = (2 - floor) - 2·(1 - floor)·t
    実装: floor·Uniform + (1-floor)·Triangular_decreasing の混合。
    Triangular_decreasing(t) = 2(1-t) からは t = 1 - sqrt(u) でサンプリング。
    """
    if not (0.0 <= floor < 1.0):
        raise ValueError(f"floor must be in [0, 1), got floor={floor}")
    u = torch.rand(batch_size, device=device)
    mask = torch.rand(batch_size, device=device) < floor
    triangular = 1.0 - torch.sqrt(u)
    uniform = u
    return torch.where(mask, uniform, triangular).clamp(1e-6, 1 - 1e-6)


def sample_linear_increasing(batch_size: int, floor: float, device) -> torch.Tensor:
    """t=0 で floor、t=1 で peak まで線形増加する分布からサンプリング。

    sample_linear_decreasing の鏡像(data 側寄り)。
    p(t) = floor + 2·(1 - floor)·t
    実装: floor·Uniform + (1-floor)·Triangular_increasing の混合。
    Triangular_increasing(t) = 2t からは t = sqrt(u) でサンプリング。
    """
    if not (0.0 <= floor < 1.0):
        raise ValueError(f"floor must be in [0, 1), got floor={floor}")
    u = torch.rand(batch_size, device=device)
    mask = torch.rand(batch_size, device=device) < floor
    triangular = torch.sqrt(u)
    uniform = u
    return torch.where(mask, uniform, triangular).clamp(1e-6, 1 - 1e-6)
