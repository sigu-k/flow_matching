#!/usr/bin/env python3
"""Statistical analysis of 5x5 FID sweep results.

Reads fid_results/<train>__<sample>/fid.json and answers:
  1. Which train_dist is best / most robust?
  2. Which sampling_dist is best / most robust?
  3. Does matching train/inference dist (diagonal) help?
  4. Which factor (train vs inference) explains more variance?
  5. Top/bottom cell rankings.

Outputs:
  - fid_results/analysis.md
  - fid_results/barplot_train.png
  - fid_results/barplot_sample.png

Usage:
    python analyze_fid.py [--fid_dir fid_results]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DISTS = ["uniform", "center", "both", "data", "noise"]


def load_grid(fid_dir: Path) -> np.ndarray:
    grid = np.full((5, 5), np.nan)
    for r, train in enumerate(DISTS):
        for c, sample in enumerate(DISTS):
            p = fid_dir / f"{train}__{sample}" / "fid.json"
            if p.exists():
                grid[r, c] = json.load(open(p))["fid"]
    return grid


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def row_stats(grid: np.ndarray) -> list[dict]:
    """Per-train_dist statistics (rows)."""
    stats = []
    for r, name in enumerate(DISTS):
        row = grid[r][~np.isnan(grid[r])]
        stats.append({
            "name": name,
            "mean": float(np.mean(row)) if len(row) else math.nan,
            "min": float(np.min(row)) if len(row) else math.nan,
            "std": float(np.std(row)) if len(row) else math.nan,
            "best_sample": DISTS[int(np.nanargmin(grid[r]))] if len(row) else "—",
            "n": len(row),
        })
    return sorted(stats, key=lambda x: x["mean"])


def col_stats(grid: np.ndarray) -> list[dict]:
    """Per-sampling_dist statistics (columns)."""
    stats = []
    for c, name in enumerate(DISTS):
        col = grid[:, c][~np.isnan(grid[:, c])]
        stats.append({
            "name": name,
            "mean": float(np.mean(col)) if len(col) else math.nan,
            "min": float(np.min(col)) if len(col) else math.nan,
            "std": float(np.std(col)) if len(col) else math.nan,
            "best_train": DISTS[int(np.nanargmin(grid[:, c]))] if len(col) else "—",
            "n": len(col),
        })
    return sorted(stats, key=lambda x: x["mean"])


def diagonal_analysis(grid: np.ndarray) -> dict:
    """Diagonal (train==sample) vs off-diagonal comparison."""
    diag, offdiag = [], []
    diag_cells, offdiag_cells = [], []
    for r in range(5):
        for c in range(5):
            v = grid[r, c]
            if math.isnan(v):
                continue
            if r == c:
                diag.append(v)
                diag_cells.append((DISTS[r], v))
            else:
                offdiag.append(v)
                offdiag_cells.append((DISTS[r], DISTS[c], v))
    return {
        "diag_mean": float(np.mean(diag)) if diag else math.nan,
        "diag_min": float(np.min(diag)) if diag else math.nan,
        "diag_cells": diag_cells,
        "offdiag_mean": float(np.mean(offdiag)) if offdiag else math.nan,
        "offdiag_min": float(np.min(offdiag)) if offdiag else math.nan,
        "offdiag_best": min(offdiag_cells, key=lambda x: x[2]) if offdiag_cells else None,
        "n_diag": len(diag),
        "n_offdiag": len(offdiag),
    }


def variance_decomposition(grid: np.ndarray) -> dict:
    """One-way ANOVA-style variance attributed to train_dist vs sampling_dist.

    Uses sum-of-squares decomposition on the complete cells only.
    SS_total = SS_row + SS_col + SS_residual
    """
    valid = [(r, c, grid[r, c]) for r in range(5) for c in range(5)
             if not math.isnan(grid[r, c])]
    if not valid:
        return {}

    vals = np.array([v for _, _, v in valid])
    grand_mean = vals.mean()

    row_means = {}
    for r in range(5):
        rv = [v for rr, _, v in valid if rr == r]
        row_means[r] = np.mean(rv) if rv else grand_mean

    col_means = {}
    for c in range(5):
        cv = [v for _, cc, v in valid if cc == c]
        col_means[c] = np.mean(cv) if cv else grand_mean

    ss_total = sum((v - grand_mean) ** 2 for _, _, v in valid)
    ss_row = sum((row_means[r] - grand_mean) ** 2 for r, _, _ in valid)
    ss_col = sum((col_means[c] - grand_mean) ** 2 for _, c, _ in valid)
    ss_resid = ss_total - ss_row - ss_col

    return {
        "grand_mean": float(grand_mean),
        "ss_total": float(ss_total),
        "ss_train": float(ss_row),
        "ss_sample": float(ss_col),
        "ss_residual": float(ss_resid),
        "pct_train": float(ss_row / ss_total * 100) if ss_total else 0,
        "pct_sample": float(ss_col / ss_total * 100) if ss_total else 0,
        "pct_residual": float(ss_resid / ss_total * 100) if ss_total else 0,
    }


def top_bottom(grid: np.ndarray, k: int = 5) -> tuple[list, list]:
    cells = []
    for r, tr in enumerate(DISTS):
        for c, sa in enumerate(DISTS):
            v = grid[r, c]
            if not math.isnan(v):
                cells.append((v, tr, sa))
    cells.sort()
    return cells[:k], cells[-k:][::-1]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(v, precision=3):
    return f"{v:.{precision}f}" if not math.isnan(v) else "—"


def write_markdown(grid, fid_dir: Path) -> None:
    rstat = row_stats(grid)
    cstat = col_stats(grid)
    diag = diagonal_analysis(grid)
    vd = variance_decomposition(grid)
    top, bot = top_bottom(grid)

    lines = ["# 5×5 FID Sweep — Statistical Analysis", ""]

    # ---- 1. train_dist comparison ----
    lines += [
        "## 1. Training distribution (train_dist) ranking",
        "",
        "Sorted by mean FID across all 5 sampling strategies (lower = better).",
        "",
        "| Rank | train_dist | mean FID | min FID | std | best sampling_dist |",
        "|------|------------|----------|---------|-----|--------------------|",
    ]
    for i, s in enumerate(rstat, 1):
        lines.append(
            f"| {i} | **{s['name']}** | {_fmt(s['mean'])} | {_fmt(s['min'])} "
            f"| {_fmt(s['std'])} | {s['best_sample']} |"
        )

    lines += [
        "",
        "> **Robustness**: lower std means the model performs consistently "
        "regardless of inference step placement.",
        "",
    ]

    # ---- 2. sampling_dist comparison ----
    lines += [
        "## 2. Inference step placement (sampling_dist) ranking",
        "",
        "Sorted by mean FID across all 5 training distributions.",
        "",
        "| Rank | sampling_dist | mean FID | min FID | std | best train_dist |",
        "|------|---------------|----------|---------|-----|-----------------|",
    ]
    for i, s in enumerate(cstat, 1):
        lines.append(
            f"| {i} | **{s['name']}** | {_fmt(s['mean'])} | {_fmt(s['min'])} "
            f"| {_fmt(s['std'])} | {s['best_train']} |"
        )
    lines.append("")

    # ---- 3. Diagonal analysis ----
    lines += [
        "## 3. Diagonal analysis — does matching train/inference dist help?",
        "",
        f"| | mean FID | min FID | N |",
        f"|---|---|---|---|",
        f"| Diagonal (train == sample) | {_fmt(diag['diag_mean'])} | {_fmt(diag['diag_min'])} | {diag['n_diag']} |",
        f"| Off-diagonal (train ≠ sample) | {_fmt(diag['offdiag_mean'])} | {_fmt(diag['offdiag_min'])} | {diag['n_offdiag']} |",
        "",
    ]
    diff = diag["offdiag_mean"] - diag["diag_mean"]
    if not math.isnan(diff):
        direction = "better" if diff > 0 else "worse"
        lines.append(
            f"Diagonal mean is **{abs(diff):.3f} FID {direction}** than off-diagonal mean."
        )
    if diag["offdiag_best"]:
        tr, sa, v = diag["offdiag_best"]
        lines.append(
            f"Best off-diagonal cell: train=**{tr}** × sample=**{sa}** (FID={v:.3f})"
        )
    lines.append("")

    # ---- 4. Variance decomposition ----
    if vd:
        lines += [
            "## 4. Variance decomposition — which factor matters more?",
            "",
            "One-way SS decomposition on all valid cells:",
            "",
            f"| Factor | SS | % of total |",
            f"|--------|----|------------|",
            f"| train_dist | {vd['ss_train']:.4f} | **{vd['pct_train']:.1f}%** |",
            f"| sampling_dist | {vd['ss_sample']:.4f} | **{vd['pct_sample']:.1f}%** |",
            f"| Residual (interaction) | {vd['ss_residual']:.4f} | {vd['pct_residual']:.1f}% |",
            f"| Total | {vd['ss_total']:.4f} | 100% |",
            "",
        ]
        dominant = "train_dist" if vd["pct_train"] > vd["pct_sample"] else "sampling_dist"
        lines.append(
            f"**{dominant}** explains more variance in FID "
            f"({max(vd['pct_train'], vd['pct_sample']):.1f}% vs "
            f"{min(vd['pct_train'], vd['pct_sample']):.1f}%)."
        )
        lines.append("")

    # ---- 5. Rankings ----
    lines += [
        "## 5. Best and worst cells",
        "",
        "**Top 5 (lowest FID):**",
        "",
        "| Rank | train_dist | sampling_dist | FID |",
        "|------|------------|---------------|-----|",
    ]
    for i, (v, tr, sa) in enumerate(top, 1):
        lines.append(f"| {i} | {tr} | {sa} | **{v:.3f}** |")

    lines += [
        "",
        "**Bottom 5 (highest FID):**",
        "",
        "| Rank | train_dist | sampling_dist | FID |",
        "|------|------------|---------------|-----|",
    ]
    for i, (v, tr, sa) in enumerate(bot, 1):
        lines.append(f"| {i} | {tr} | {sa} | {v:.3f} |")
    lines.append("")

    # ---- filled count ----
    filled = int(np.sum(~np.isnan(grid)))
    lines.append(f"*Based on {filled}/25 cells.*")

    out = fid_dir / "analysis.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved {out}")


def write_barplot(stats: list[dict], xlabel: str, title: str, path: Path) -> None:
    names = [s["name"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    mins = [s["min"] for s in stats]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x - width / 2, means, width, label="mean FID", color="steelblue",
                  yerr=stds, capsize=4, alpha=0.85)
    ax.bar(x + width / 2, mins, width, label="min FID", color="seagreen", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("FID", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fid_dir",
        required=True,
        help="Root directory containing <train>__<sample>/fid.json cells. e.g. fid_results/nfe_50",
    )
    args = parser.parse_args()

    fid_dir = Path(args.fid_dir)
    if not fid_dir.exists():
        print(f"ERROR: {fid_dir} does not exist.")
        raise SystemExit(1)

    grid = load_grid(fid_dir)
    filled = int(np.sum(~np.isnan(grid)))
    print(f"Loaded {filled}/25 cells from {fid_dir}")
    if filled == 0:
        print("No fid.json files found.")
        raise SystemExit(1)

    write_markdown(grid, fid_dir)

    rstat = row_stats(grid)
    write_barplot(
        rstat,
        xlabel="train_dist",
        title="FID by training distribution (mean ± std  /  min)",
        path=fid_dir / "barplot_train.png",
    )

    cstat = col_stats(grid)
    write_barplot(
        cstat,
        xlabel="sampling_dist",
        title="FID by inference step placement (mean ± std  /  min)",
        path=fid_dir / "barplot_sample.png",
    )


if __name__ == "__main__":
    main()
