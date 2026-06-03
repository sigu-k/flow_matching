# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""Shared timestep-density utilities.

A single density rho(t) = 1 + A * shape(t) is used both for the
*training* timestep distribution and for the *inference* step placement,
so that the two stay perfectly coupled (the point of the experiment).

The placement is produced by inverse-transform sampling of rho over a
fine grid. See handoff_to_claude_code for the verified reference shapes.
"""
import math

import torch

# Each entry returns shape(t); rho(t) = 1 + A * shape(t).
_SHAPES = {
    "uniform": lambda t: torch.zeros_like(t),
    "center": lambda t: torch.sin(math.pi * t),
    "both": lambda t: 1.0 - torch.sin(math.pi * t),
    "data": lambda t: 1.0 - t,
    "noise": lambda t: t,
}

DIST_CHOICES = list(_SHAPES.keys())


def shape_fn(t: torch.Tensor, dist: str, A: float = 1.0) -> torch.Tensor:
    """Return rho(t) = 1 + A * shape(t) for the given distribution."""
    return 1.0 + A * _SHAPES[dist](t)


def build_cdf(dist, A=1.0, grid_size=10000, device=None, dtype=torch.float64):
    """Return (grid, cdf) for inverse-transform sampling of rho(t)."""
    grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    rho = shape_fn(grid, dist, A=A)
    cdf = torch.cumsum(rho, dim=0)
    cdf = cdf / cdf[-1].clone()
    return grid, cdf


def sample_timesteps(n, dist, A=1.0, device=None, generator=None):
    """Training side: draw n timesteps ~ rho(t) via inverse transform."""
    grid, cdf = build_cdf(dist, A=A, device=device)
    u = torch.rand(n, device=device, dtype=cdf.dtype, generator=generator)
    idx = torch.searchsorted(cdf, u).clamp(max=grid.shape[0] - 1)
    return grid[idx].to(torch.float32)


def sampling_timesteps(num_steps, dist, A=1.0, device=None):
    """Inference side: num_steps step times placed by rho(t).

    The two endpoints are pinned to exactly 0.0 and 1.0 so the ODE is
    always integrated over the full [0, 1] interval.
    """
    grid, cdf = build_cdf(dist, A=A, device=device)
    targets = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=cdf.dtype)
    idx = torch.searchsorted(cdf, targets).clamp(max=grid.shape[0] - 1)
    t = grid[idx].clone()
    t[0] = 0.0
    t[-1] = 1.0
    return t
