#!/usr/bin/env python3
"""
Visualise the 4 timestep samplers used in Phase 0 / Phase 1:
  Uniform, LN(μ=-0.8, σ=1.0), Mode(s=-0.5), Mode(s=+1.0)

Output: outputs/sampler_distributions.png
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from timestep_sampler import sample_logit_normal, sample_mode, sample_uniform

N_SAMPLES = 100_000
N_BINS = 100
DEVICE = "cpu"
OUT_PATH = Path("outputs/sampler_distributions.png")

SAMPLERS = [
    {
        "label": "Uniform",
        "fn": lambda: sample_uniform(N_SAMPLES, DEVICE),
        "pdf": lambda t: np.ones_like(t),
        "color": "#4878CF",
    },
    {
        "label": "LN (μ=−0.8, σ=1.0)",
        "fn": lambda: sample_logit_normal(N_SAMPLES, mu=-0.8, sigma=1.0, device=DEVICE),
        "pdf": lambda t: (
            (1.0 / (1.0 * math.sqrt(2 * math.pi)))
            * np.exp(-0.5 * ((np.log(t / (1 - t)) - (-0.8)) / 1.0) ** 2)
            / (t * (1 - t) + 1e-12)
        ),
        "color": "#D65F5F",
    },
    {
        "label": "Mode Beta (s=−0.5)  [U-shape]",
        "fn": lambda: sample_mode(N_SAMPLES, s=-0.5, device=DEVICE),
        "pdf": lambda t: (
            t ** (-0.5) * (1 - t) ** (-0.5)
            / float(torch.distributions.Beta(
                torch.tensor(0.5), torch.tensor(0.5)
            ).log_prob(torch.tensor(0.5)).exp() * math.pi)
            # B(0.5,0.5) = π
        ),
        "color": "#6ACC65",
    },
    {
        "label": "Mode Beta (s=+1.0)  [bell]",
        "fn": lambda: sample_mode(N_SAMPLES, s=1.0, device=DEVICE),
        "pdf": lambda t: 6.0 * t * (1 - t),  # Beta(2,2): 6t(1-t)
        "color": "#B47CC7",
    },
]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Timestep distributions  ·  t=0 (noise) ← ——— → t=1 (data)",
        fontsize=13, y=1.01,
    )

    t_grid = np.linspace(1e-4, 1 - 1e-4, 500)

    for ax, cfg in zip(axes.flat, SAMPLERS):
        samples = cfg["fn"]().numpy()

        # Histogram (density)
        counts, edges = np.histogram(samples, bins=N_BINS, range=(0, 1), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=edges[1] - edges[0],
               alpha=0.55, color=cfg["color"], label="samples")

        # Analytical PDF
        pdf_vals = cfg["pdf"](t_grid)
        ax.plot(t_grid, pdf_vals, color=cfg["color"], linewidth=2, label="PDF")

        ax.set_title(cfg["label"], fontsize=11)
        ax.set_xlabel("t  (t=0: noise  —  t=1: data)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
