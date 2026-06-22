#!/usr/bin/env python3
"""
Post-hoc EMA reconstruction.

Reconstructs an exponential-EMA model with ANY decay from the fp16 weight
snapshots dumped by train_phase1_posthoc.py, then (optionally) evaluates FID.
Training is never re-run; reconstruction is a weighted sum of stored weights.

Snapshots are read from checkpoints/<ckpt-base>/<sampler>/snapshots/. --ckpt-base
defaults to phase1_posthoc but can point at any tree that dumped snapshots via
posthoc_snapshot.save_snapshot (e.g. phase1_twophase, phase1_bin_adaptive), since
the snapshot/manifest format is shared.

Run from: flow_matching/examples/image/
  # list available snapshots for a sampler
  python reconstruct_ema.py --sampler ln_mu-0.8 --list

  # reconstruct decay=0.9999 at the final snapshot and save the model
  python reconstruct_ema.py --sampler ln_mu-0.8 --decay 0.9999 --out recon_d9999.pt

  # reconstruct and compute FID (writes posthoc_results/<sampler>.json)
  python reconstruct_ema.py --sampler ln_mu-0.8 --decay 0.9999 --fid

  # reconstruct from an existing two-phase run (no posthoc run needed)
  python reconstruct_ema.py --ckpt-base phase1_twophase \
      --sampler ln_mu-0.8__uniform__e60 --decay 0.9999 --fid

All paths are repo-relative (derived from __file__), so no server-specific
absolute path appears and the tool runs unchanged on any checkout.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from models.model_configs import MODEL_CONFIGS
from models.unet import UNetModel
from posthoc_snapshot import load_manifest, snapshot_dir

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = _REPO_ROOT / "checkpoints"
DEFAULT_CKPT_BASE = "phase1_posthoc"
RESULTS_DIR = Path(__file__).resolve().parent / "posthoc_results"

# FID defaults mirror train_phase1_posthoc.py for comparability.
FID_SAMPLES = 50_000
FID_NFE = 50
FID_BATCH = 250
FID_SEED = 0


def get_args():
    p = argparse.ArgumentParser("Post-hoc EMA reconstruction")
    p.add_argument("--sampler", required=True,
                   help="sampler_id, i.e. the dir under checkpoints/<ckpt-base>/")
    p.add_argument("--ckpt-base", type=str, default=DEFAULT_CKPT_BASE,
                   help="checkpoints/ subtree that holds the run (default "
                        f"{DEFAULT_CKPT_BASE!r}). Snapshots are read from "
                        "checkpoints/<ckpt-base>/<sampler>/snapshots/. Use e.g. "
                        "phase1_twophase or phase1_bin_adaptive to reconstruct those.")
    p.add_argument("--decay", type=float, default=0.9999,
                   help="Exponential EMA decay to reconstruct (per optimizer step).")
    p.add_argument("--until-step", type=int, default=None,
                   help="Reconstruct the EMA as of this global_step "
                        "(default: the last available snapshot).")
    p.add_argument("--list", action="store_true",
                   help="Just list available snapshots and exit.")
    p.add_argument("--out", type=str, default=None,
                   help="Path to save the reconstructed weights (state_dict, fp32).")
    p.add_argument("--fid", action="store_true",
                   help="Generate samples and compute FID for the reconstructed model.")
    p.add_argument("--fid-samples", type=int, default=FID_SAMPLES)
    p.add_argument("--data-path", type=str, default="./data/image_generation")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default=None,
                   help="Model key for MODEL_CONFIGS (default: from manifest).")
    return p.parse_args()


def compute_weights(snapshots, decay: float, until_step: int):
    """Trapezoidal approximation of the exponential EMA kernel over snapshot steps.

    Online EMA at step T weights step t by (1-d)*d^(T-t). We only have weights at
    snapshot steps s_i, so each snapshot carries the kernel mass of the interval it
    represents (midpoint rule), then we renormalise.
    """
    used = [s for s in snapshots if s["step"] <= until_step]
    if not used:
        raise ValueError(f"No snapshots at or before step {until_step}.")
    steps = [s["step"] for s in used]

    # Midpoint interval width Δ_i around each snapshot step.
    widths = []
    for i, st in enumerate(steps):
        lo = steps[i - 1] if i > 0 else st
        hi = steps[i + 1] if i < len(steps) - 1 else st
        widths.append((hi - lo) / 2 if hi != lo else 1.0)

    # log-domain for numerical stability: log(d^(T-t) * Δ) = (T-t)*log d + log Δ
    log_d = torch.log(torch.tensor(decay, dtype=torch.float64))
    log_w = torch.tensor(
        [(until_step - st) * float(log_d) + torch.log(torch.tensor(w)).item()
         for st, w in zip(steps, widths)],
        dtype=torch.float64,
    )
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    w = w / w.sum()
    return used, w


def reconstruct(snap_dir: Path, used, weights) -> dict:
    """Weighted sum of fp16 snapshot weights, accumulated in fp32."""
    acc = None
    for s, wi in zip(used, weights.tolist()):
        state = torch.load(snap_dir / s["file"], map_location="cpu", weights_only=False)
        sd = state["weights"]
        if acc is None:
            acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sd.items()}
            int_keys = {k for k, v in sd.items() if not v.is_floating_point()}
        for k, v in sd.items():
            if k in int_keys:
                acc[k] = v.clone()  # ints (e.g. num_batches_tracked): take as-is
            else:
                acc[k].add_(v.to(torch.float32) * wi)
    return acc


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )
    args = get_args()

    snap_dir = snapshot_dir(CKPT_ROOT / args.ckpt_base / args.sampler)
    manifest = load_manifest(snap_dir)
    snaps = manifest["snapshots"]
    if not snaps:
        raise SystemExit(f"No snapshots found in {snap_dir}")

    if args.list:
        logger.info(f"{len(snaps)} snapshots in {snap_dir}:")
        for s in snaps:
            logger.info(f"  step={s['step']:>8}  epoch={s['epoch']:>4}  "
                        f"num_updates={s['num_updates']:>8}  {s['file']}")
        return

    until_step = args.until_step if args.until_step is not None else snaps[-1]["step"]
    used, weights = compute_weights(snaps, args.decay, until_step)
    logger.info(f"Reconstructing decay={args.decay} at step {until_step} "
                f"from {len(used)} snapshots "
                f"(top weight {weights.max():.4f} @ step {used[weights.argmax()]['step']}).")

    acc = reconstruct(snap_dir, used, weights)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(acc, out)
        logger.info(f"Saved reconstructed weights: {out}")

    if not args.fid:
        return

    # FID — reuse the exact generation/eval path from training.
    from phase1_utils import _compute_fid, _generate_images

    model_key = args.model or manifest.get("model", "cifar10")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNetModel(**MODEL_CONFIGS[model_key])
    model.load_state_dict(acc)
    model.to(device).eval()

    t0 = time.time()
    logger.info(f"Generating {args.fid_samples} samples (nfe={FID_NFE}) ...")
    imgs = _generate_images(model, device, args.fid_samples, FID_BATCH, FID_NFE, FID_SEED)
    fid = _compute_fid(imgs, args.data_path, device)
    elapsed = time.time() - t0
    logger.info(f"Reconstructed-EMA FID: {fid:.3f}  ({elapsed:.0f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the original filename for the default tree; namespace others so runs
    # that share a sampler_id across trees don't clobber each other.
    res_name = (f"{args.sampler}.json" if args.ckpt_base == DEFAULT_CKPT_BASE
                else f"{args.ckpt_base}__{args.sampler}.json")
    res_path = RESULTS_DIR / res_name
    history = json.loads(res_path.read_text()) if res_path.exists() else []
    history.append({
        "sampler_id": args.sampler,
        "ckpt_base": args.ckpt_base,
        "decay": args.decay,
        "until_step": until_step,
        "n_snapshots_used": len(used),
        "fid": fid,
        "ode_method": "euler", "nfe": FID_NFE,
        "fid_samples": args.fid_samples, "batch_size": FID_BATCH, "seed": FID_SEED,
        "elapsed_sec": round(elapsed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    res_path.write_text(json.dumps(history, indent=2))
    logger.info(f"Appended result to {res_path}")


if __name__ == "__main__":
    main()
