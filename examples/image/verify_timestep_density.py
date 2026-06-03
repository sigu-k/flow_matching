# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""Verify timestep-density placement against the handoff reference shapes.

Run from examples/image/:  python verify_timestep_density.py
"""
import numpy as np
import torch

from training.timestep_density import (
    DIST_CHOICES,
    sample_timesteps,
    sampling_timesteps,
)

# Reference from handoff_to_claude_code (検算済みの期待される形).
EXPECTED_TRAIN = {
    "uniform": [9, 9, 9, 9, 10, 10, 10, 9, 10, 10],
    "center": [7, 8, 10, 11, 11, 12, 11, 10, 8, 7],
    "both": [13, 11, 9, 8, 7, 7, 8, 9, 11, 13],
    "data": [12, 12, 11, 11, 10, 9, 9, 8, 7, 7],
    "noise": [7, 7, 8, 8, 9, 10, 10, 11, 12, 12],
}
EXPECTED_INFER = {
    "uniform": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    "center": [4, 4, 5, 6, 6, 6, 6, 5, 4, 4],
    "both": [7, 6, 4, 4, 4, 4, 4, 4, 6, 7],
    "data": [7, 6, 6, 5, 5, 5, 4, 4, 4, 4],
    "noise": [4, 4, 4, 4, 5, 5, 5, 6, 6, 7],
}


def hist10(t):
    counts, _ = np.histogram(t.cpu().numpy(), bins=10, range=(0.0, 1.0))
    return [int(c) for c in counts]


def main():
    n = 200000
    g = torch.Generator().manual_seed(0)

    print("== Training side (n=200k, percent per 10 bins) ==")
    print("   (stochastic: shape must match; exact counts are RNG-dependent)")
    for d in DIST_CHOICES:
        t = sample_timesteps(n, d, generator=g)
        counts = hist10(t)
        pct = [round(c / n * 100) for c in counts]
        exp = EXPECTED_TRAIN[d]
        ok = "all bins>0" if all(c > 0 for c in counts) else "!! ZERO BIN !!"
        print(f"  {d:7s} got={pct}")
        print(f"  {'':7s} exp={exp}  ({ok})")

    print("\n== Inference side (num_steps=50, counts, endpoints fixed) ==")
    all_match = True
    for d in DIST_CHOICES:
        t = sampling_timesteps(50, d)
        got = hist10(t)
        exp = EXPECTED_INFER[d]
        match = got == exp
        all_match &= match
        ends_ok = float(t[0]) == 0.0 and float(t[-1]) == 1.0
        flag = "OK" if (match and ends_ok) else "MISMATCH"
        print(f"  {d:7s} got={got}")
        print(f"  {'':7s} exp={exp}  endpoints=({float(t[0]):.3f},{float(t[-1]):.3f}) [{flag}]")

    print()
    if all_match:
        print("RESULT: inference placement matches the reference EXACTLY. ✓")
    else:
        print("RESULT: inference placement MISMATCH — investigate. ✗")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
