#!/usr/bin/env python3
"""
Plot a hand-picked set of timestep distributions in a single figure.

Reuses the sampler functions and analytical PDFs from visualize_samplers.py,
but lets each panel carry its own Uniform-mixture weight α so distributions
that differ only in α (e.g. LN α=0 vs LN α=0.3) can be compared side by side.

Panels produced here:
  1. Uniform
  2. LN (μ=+0.8, σ=1.0, α=0)
  3. LN (μ=+0.8, σ=1.0, α=0.3)
  4. LI (floor=0.3)            [LinearIncreasing, data-side]

Uniform mixture:  p_new(t) = α·1 + (1-α)·p_orig(t)   (density ≥ α everywhere)

Usage:
  python plot_selected_distributions.py
  python plot_selected_distributions.py --out_path outputs/selected.png --dpi 200
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from timestep_sampler import (
    sample_linear_increasing,
    sample_logit_normal,
    sample_uniform,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Uniform / LN(μ=+0.8) α=0 / LN(μ=+0.8) α=0.3 / LI(f=0.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--n_bins", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--out_path", type=str, default=None,
                        help="Output PNG path. If omitted, a timestamped name under outputs/.")
    return parser.parse_args()


def uniform_mix_fn(base_fn, alpha, n):
    """Sampler with Uniform mixture: draw Uniform w.p. α, base sampler w.p. (1-α)."""
    if alpha == 0.0:
        return base_fn

    def _fn():
        mask = torch.bernoulli(torch.full((n,), alpha)).bool()
        return torch.where(mask, torch.rand(n), base_fn())
    return _fn


def uniform_mix_pdf(base_pdf, alpha):
    if alpha == 0.0:
        return base_pdf
    return lambda t: alpha + (1.0 - alpha) * base_pdf(t)


def ln_pdf(t, mu, sigma):
    return (
        (1.0 / (sigma * math.sqrt(2 * math.pi)))
        * np.exp(-0.5 * ((np.log(t / (1 - t)) - mu) / sigma) ** 2)
        / (t * (1 - t) + 1e-12)
    )


def build_panels(args):
    n, dev = args.n_samples, args.device
    mu, sigma = 0.8, 1.0
    lif = 0.3

    return [
        {
            "label": "Uniform",
            "fn": lambda: sample_uniform(n, dev),
            "pdf": lambda t: np.ones_like(t),
            "color": "#4878CF",
            "alpha": 0.0,
        },
        {
            "label": f"LN (μ={mu:+.2f}, σ={sigma:.2f}, α=0.00)",
            "fn": uniform_mix_fn(lambda: sample_logit_normal(n, mu=mu, sigma=sigma, device=dev), 0.0, n),
            "pdf": uniform_mix_pdf(lambda t: ln_pdf(t, mu, sigma), 0.0),
            "color": "#D65F5F",
            "alpha": 0.0,
        },
        {
            "label": f"LN (μ={mu:+.2f}, σ={sigma:.2f}, α=0.30)",
            "fn": uniform_mix_fn(lambda: sample_logit_normal(n, mu=mu, sigma=sigma, device=dev), 0.3, n),
            "pdf": uniform_mix_pdf(lambda t: ln_pdf(t, mu, sigma), 0.3),
            "color": "#AB4642",
            "alpha": 0.3,
        },
        {
            "label": f"LI (floor={lif:.2f})  [data-side]",
            "fn": lambda: sample_linear_increasing(n, floor=lif, device=dev),
            "pdf": lambda t, _lf=lif: _lf + 2.0 * (1.0 - _lf) * t,
            "color": "#8C613C",
            "alpha": 0.0,
        },
    ]


def resolve_out_path(args):
    if args.out_path is not None:
        return Path(args.out_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/selected_distributions_{ts}.png")


def main():
    args = parse_args()
    panels = build_panels(args)
    out_path = resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Timestep distributions  ·  t=0 (noise) ← ——— → t=1 (data)",
        fontsize=12, y=1.01,
    )

    t_grid = np.linspace(1e-4, 1 - 1e-4, 500)

    for ax, cfg in zip(axes.flat, panels):
        samples = cfg["fn"]().numpy()

        counts, edges = np.histogram(samples, bins=args.n_bins, range=(0, 1), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=edges[1] - edges[0],
               alpha=0.55, color=cfg["color"], label="samples")

        ax.plot(t_grid, cfg["pdf"](t_grid), color=cfg["color"], linewidth=2, label="PDF")

        if cfg["alpha"] > 0.0:
            ax.axhline(cfg["alpha"], color="gray", linewidth=1.0, linestyle="--",
                       label=f"floor (α={cfg['alpha']:.2f})")

        ax.set_title(cfg["label"], fontsize=11)
        ax.set_xlabel("t  (t=0: noise  —  t=1: data)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
