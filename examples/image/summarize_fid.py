#!/usr/bin/env python3
"""Summarize 5x5 FID sweep results.

Reads fid_results/<train>__<sample>/fid.json for all 25 cells and outputs:
  - fid_results/summary.csv
  - fid_results/summary.md
  - fid_results/heatmap.png

Usage:
    python summarize_fid.py [--fid_dir fid_results]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

DISTS = ["uniform", "center", "both", "data", "noise"]


def load_grid(fid_dir: Path) -> tuple[np.ndarray, int | None]:
    """Return (5, 5) float array and NFE read from the first available fid.json."""
    grid = np.full((5, 5), np.nan)
    nfe = None
    for r, train in enumerate(DISTS):
        for c, sample in enumerate(DISTS):
            p = fid_dir / f"{train}__{sample}" / "fid.json"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                grid[r, c] = data["fid"]
                if nfe is None:
                    nfe = data.get("nfe")
    return grid, nfe


def write_csv(grid: np.ndarray, path: Path) -> None:
    lines = ["train_dist,sample_dist,fid"]
    for r, train in enumerate(DISTS):
        for c, sample in enumerate(DISTS):
            v = grid[r, c]
            cell = f"{v:.4f}" if not math.isnan(v) else ""
            lines.append(f"{train},{sample},{cell}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


def write_markdown(grid: np.ndarray, path: Path) -> None:
    col_w = 9

    def fmt(v):
        return f"{v:.2f}".center(col_w) if not math.isnan(v) else "-".center(col_w)

    header_label = "train↓ / infer→"
    header = f"| {header_label:<16} |" + "".join(f" {d.center(col_w)} |" for d in DISTS)
    sep = f"| {'-'*16} |" + "".join(f" {'-'*col_w} |" for _ in DISTS)
    rows = [header, sep]
    for r, train in enumerate(DISTS):
        row = f"| {train:<16} |" + "".join(f" {fmt(grid[r, c])} |" for c in range(5))
        rows.append(row)

    # append best-per-column and best-per-row summaries
    rows.append("")
    rows.append("**Best sampling_dist per train_dist** (row min):")
    for r, train in enumerate(DISTS):
        row = grid[r]
        if not np.all(np.isnan(row)):
            best_c = int(np.nanargmin(row))
            rows.append(f"  {train}: {DISTS[best_c]} (FID={grid[r, best_c]:.2f})")

    rows.append("")
    rows.append("**Best train_dist per sampling_dist** (col min):")
    for c, sample in enumerate(DISTS):
        col = grid[:, c]
        if not np.all(np.isnan(col)):
            best_r = int(np.nanargmin(col))
            rows.append(f"  {sample}: {DISTS[best_r]} (FID={grid[best_r, c]:.2f})")

    path.write_text("\n".join(rows) + "\n")
    print(f"Saved {path}")


def write_heatmap(grid: np.ndarray, path: Path, nfe: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # mask NaN for colormap
    masked = np.ma.masked_invalid(grid)
    vmin = np.nanmin(grid) if not np.all(np.isnan(grid)) else 0
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1

    cmap = plt.cm.RdYlGn_r  # low FID = green (good)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(DISTS, fontsize=10)
    ax.set_yticklabels(DISTS, fontsize=10)
    ax.set_xlabel("Inference step placement (sampling_dist)", fontsize=11)
    ax.set_ylabel("Training timestep dist (train_dist)", fontsize=11)
    nfe_str = str(nfe) if nfe is not None else "?"
    ax.set_title(f"5×5 FID Sweep — CIFAR-10 (Euler, NFE={nfe_str})", fontsize=12, pad=10)

    # annotate cells
    thresh = vmin + (vmax - vmin) * 0.5
    for r in range(5):
        for c in range(5):
            v = grid[r, c]
            if math.isnan(v):
                text = "—"
                color = "grey"
            else:
                text = f"{v:.2f}"
                color = "white" if v > thresh else "black"
            ax.text(c, r, text, ha="center", va="center", fontsize=9, color=color)

    # mark global minimum
    if not np.all(np.isnan(grid)):
        best_r, best_c = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.add_patch(plt.Rectangle(
            (best_c - 0.48, best_r - 0.48), 0.96, 0.96,
            fill=False, edgecolor="blue", linewidth=2.5, label="global best"
        ))
        ax.legend(loc="upper right", fontsize=8)

    plt.colorbar(im, ax=ax, label="FID")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


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
        print(f"ERROR: {fid_dir} does not exist. Run the sweep first.")
        raise SystemExit(1)

    grid, nfe = load_grid(fid_dir)

    filled = int(np.sum(~np.isnan(grid)))
    print(f"Loaded {filled}/25 cells from {fid_dir}")
    if filled == 0:
        print("No fid.json files found. Exiting.")
        raise SystemExit(1)

    write_csv(grid, fid_dir / "summary.csv")
    write_markdown(grid, fid_dir / "summary.md")
    write_heatmap(grid, fid_dir / "heatmap.png", nfe=nfe)


if __name__ == "__main__":
    main()
