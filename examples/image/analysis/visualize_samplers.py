#!/usr/bin/env python3
"""
Visualise the 4 timestep samplers used in Phase 0 / Phase 1:
  Uniform, LN(μ, σ), Mode(s1), Mode(s2)

Uniform-mixture mode (--alpha):
  p_new(t) = α · 1 + (1-α) · p_original(t)
  Guarantees density ≥ α everywhere.  α=0 → original, α=1 → pure Uniform.

Usage examples:
  python visualize_samplers.py
  python visualize_samplers.py --n_samples 200000 --ln_mu -1.0 --ln_sigma 1.5
  python visualize_samplers.py --mode_s1 -1.0 --mode_s2 2.0 --dpi 200
  python visualize_samplers.py --alpha 0.3
  python visualize_samplers.py --alpha 0.2 --ln_mu -1.2
  python visualize_samplers.py --out_path outputs/custom_name.png  # exact path (no timestamp added)
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

from timestep_sampler import sample_logit_normal, sample_mode, sample_uniform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise timestep sampler distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=100_000,
                        help="Number of samples drawn per sampler")
    parser.add_argument("--n_bins", type=int, default=100,
                        help="Number of histogram bins")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu / cuda)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI")
    parser.add_argument("--ln_mu", type=float, default=-0.8,
                        help="Logit-Normal μ parameter")
    parser.add_argument("--ln_sigma", type=float, default=1.0,
                        help="Logit-Normal σ parameter")
    parser.add_argument("--mode_s1", type=float, default=-0.5,
                        help="Mode-Beta s parameter for first mode sampler (U-shape when s<0)")
    parser.add_argument("--mode_s2", type=float, default=1.0,
                        help="Mode-Beta s parameter for second mode sampler (bell when s>0)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Uniform mixture weight α ∈ [0, 1]. "
                             "p_new(t) = α·1 + (1-α)·p_orig(t). "
                             "0 = no mixing, 1 = pure Uniform.")
    parser.add_argument("--out_path", type=str, default=None,
                        help="Output PNG path. If omitted, a timestamped name is used under outputs/")
    args = parser.parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        parser.error(f"--alpha must be in [0, 1], got {args.alpha}")
    return args


def build_samplers(args):
    n, dev = args.n_samples, args.device
    mu, sigma = args.ln_mu, args.ln_sigma
    s1, s2 = args.mode_s1, args.mode_s2

    # alpha/beta for Beta distribution: alpha = beta = 1 / (1 + |s|) when s != 0
    def _beta_ab(s):
        # sample_mode uses Beta(alpha, beta) where mode = (alpha-1)/(alpha+beta-2)
        # Simplified: alpha = beta = 0.5 + s  (matches original s=-0.5 → Beta(0,0) edge / s=1 → Beta(2,2))
        # Actual formula from timestep_sampler: alpha = 1 + s, beta = 1 + s  (when s > -1)
        a = 1.0 + s
        b = 1.0 + s
        return max(a, 1e-6), max(b, 1e-6)

    a1, b1 = _beta_ab(s1)
    a2, b2 = _beta_ab(s2)

    import scipy.special as sc  # for beta function normalisation in analytical PDF

    def _beta_pdf(t, a, b):
        log_norm = sc.betaln(a, b)
        return np.exp((a - 1) * np.log(np.clip(t, 1e-12, 1)) +
                      (b - 1) * np.log(np.clip(1 - t, 1e-12, 1)) - log_norm)

    return [
        {
            "label": "Uniform",
            "fn": lambda: sample_uniform(n, dev),
            "pdf": lambda t: np.ones_like(t),
            "color": "#4878CF",
        },
        {
            "label": f"LN (μ={mu:+.2f}, σ={sigma:.2f})",
            "fn": lambda: sample_logit_normal(n, mu=mu, sigma=sigma, device=dev),
            "pdf": lambda t: (
                (1.0 / (sigma * math.sqrt(2 * math.pi)))
                * np.exp(-0.5 * ((np.log(t / (1 - t)) - mu) / sigma) ** 2)
                / (t * (1 - t) + 1e-12)
            ),
            "color": "#D65F5F",
        },
        {
            "label": f"Mode Beta (s={s1:+.2f})  [{'U-shape' if s1 < 0 else 'bell'}]",
            "fn": lambda: sample_mode(n, s=s1, device=dev),
            "pdf": lambda t, _a=a1, _b=b1: _beta_pdf(t, _a, _b),
            "color": "#6ACC65",
        },
        {
            "label": f"Mode Beta (s={s2:+.2f})  [{'U-shape' if s2 < 0 else 'bell'}]",
            "fn": lambda: sample_mode(n, s=s2, device=dev),
            "pdf": lambda t, _a=a2, _b=b2: _beta_pdf(t, _a, _b),
            "color": "#B47CC7",
        },
    ]


def apply_uniform_mix(samplers, alpha, n):
    """Wraps each sampler's fn and pdf with Uniform mixture in-place.

    Sampling: for each draw, choose Uniform with prob α, original with prob (1-α).
    PDF:      p_new(t) = α + (1-α) · p_orig(t)
    """
    if alpha == 0.0:
        return samplers

    mixed = []
    for cfg in samplers:
        orig_fn = cfg["fn"]
        orig_pdf = cfg["pdf"]

        def _make_fn(fn):
            def _fn():
                mask = torch.bernoulli(torch.full((n,), alpha)).bool()
                return torch.where(mask, torch.rand(n), fn())
            return _fn

        def _make_pdf(pdf):
            return lambda t: alpha + (1.0 - alpha) * pdf(t)

        mixed.append({**cfg, "fn": _make_fn(orig_fn), "pdf": _make_pdf(orig_pdf)})
    return mixed


def resolve_out_path(args):
    if args.out_path is not None:
        return Path(args.out_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alpha_tag = f"_a{args.alpha:.2f}" if args.alpha > 0.0 else ""
    return Path(f"outputs/sampler_distributions_{ts}{alpha_tag}.png")


def main():
    args = parse_args()
    samplers = build_samplers(args)
    samplers = apply_uniform_mix(samplers, args.alpha, args.n_samples)
    out_path = resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_note = f"  ·  Uniform mix α={args.alpha:.2f}  (floor={args.alpha:.2f})" if args.alpha > 0.0 else ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Timestep distributions  ·  t=0 (noise) ← ——— → t=1 (data){alpha_note}",
        fontsize=12, y=1.01,
    )

    t_grid = np.linspace(1e-4, 1 - 1e-4, 500)

    for ax, cfg in zip(axes.flat, samplers):
        samples = cfg["fn"]().numpy()

        counts, edges = np.histogram(samples, bins=args.n_bins, range=(0, 1), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=edges[1] - edges[0],
               alpha=0.55, color=cfg["color"], label="samples")

        pdf_vals = cfg["pdf"](t_grid)
        ax.plot(t_grid, pdf_vals, color=cfg["color"], linewidth=2, label="PDF")

        if args.alpha > 0.0:
            ax.axhline(args.alpha, color="gray", linewidth=1.0, linestyle="--",
                       label=f"floor (α={args.alpha:.2f})")

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
