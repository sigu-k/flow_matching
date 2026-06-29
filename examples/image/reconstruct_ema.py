#!/usr/bin/env python3
"""
Post-hoc EMA reconstruction.

Reconstructs an exponential-EMA model with ANY decay from the fp16 weight
snapshots dumped by train_phase1_posthoc.py, then (optionally) evaluates FID.
Training is never re-run; reconstruction is a weighted sum of stored weights.

For a given decay, FID is traced across epochs (every --eval-every epochs, default
20) so the FID-vs-epoch curve is observable, not just the final value. Generation
noise is fixed (seed=0, identical x0 every call) and cuDNN is pinned deterministic,
so FID differences reflect the weights only.

Snapshots are read from checkpoints/<ckpt-base>/<sampler>/snapshots/. --ckpt-base
defaults to phase1_posthoc but can point at any tree that dumped snapshots via
posthoc_snapshot.save_snapshot (e.g. phase1_twophase, phase1_bin_adaptive), since
the snapshot/manifest format is shared.

Run from: flow_matching/examples/image/
  # list available snapshots for a sampler
  python reconstruct_ema.py --sampler ln_mu-0.8 --list

  # reconstruct decay=0.9999 at the final snapshot and save the model
  python reconstruct_ema.py --sampler ln_mu-0.8 --decay 0.9999 --out recon_d9999.pt

  # trace FID vs epoch (every 20 epochs) for decay=0.999 -> posthoc_results/<sampler>.json
  python reconstruct_ema.py --sampler uniform --decay 0.999 --fid

  # only the final epoch's FID (no trajectory)
  python reconstruct_ema.py --sampler uniform --decay 0.999 --fid --eval-every 0

  # reconstruct from an existing two-phase run (no posthoc run needed)
  python reconstruct_ema.py --ckpt-base phase1_twophase \
      --sampler ln_mu-0.8__uniform__e60 --decay 0.9999 --fid

All paths are repo-relative (derived from __file__), so no server-specific
absolute path appears and the tool runs unchanged on any checkout.
"""

import argparse
import gc
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

# New project for post-hoc EMA decay sweeps (separate from the training runs).
WANDB_PROJECT = "phase1-posthoc-ema"

# FID defaults mirror train_phase1_posthoc.py for comparability.
FID_SAMPLES = 50_000
FID_NFE = 50
FID_BATCH = 250
FID_SEED = 0
# Trace FID every N epochs across the snapshots (mirrors train_phase1_static.py
# EVAL_EVERY=20) so the FID-vs-epoch curve for a given decay can be observed.
EVAL_EVERY_EPOCHS = 20


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
                   help="Cap the trajectory at this global_step "
                        "(default: the last available snapshot).")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY_EPOCHS,
                   help="Compute FID at every N-th epoch that has a snapshot, to "
                        "trace the FID-vs-epoch curve for this decay (default "
                        f"{EVAL_EVERY_EPOCHS}). Use 0 to evaluate only the final "
                        "snapshot.")
    p.add_argument("--list", action="store_true",
                   help="Just list available snapshots and exit.")
    p.add_argument("--out", type=str, default=None,
                   help="Path to save the reconstructed weights (state_dict, fp32).")
    p.add_argument("--fid", action=argparse.BooleanOptionalAction, default=True,
                   help="Generate samples and compute FID (default: on; --no-fid to skip).")
    p.add_argument("--fid-samples", type=int, default=FID_SAMPLES)
    p.add_argument("--data-path", type=str, default="./data/image_generation")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default=None,
                   help="Model key for MODEL_CONFIGS (default: from manifest).")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Log the FID result to wandb (default: on; --no-wandb to skip).")
    p.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                   help=f"wandb project to log into (default {WANDB_PROJECT!r}).")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="wandb run name (default: <sampler>__d<decay>).")
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

    final_step = snaps[-1]["step"]
    until_step = args.until_step if args.until_step is not None else final_step

    # Choose the snapshots at which to evaluate FID, to trace the FID-vs-epoch
    # curve for THIS decay. Mirrors train_phase1_static.py's EVAL_EVERY=20:
    # evaluate every args.eval_every epochs, and always include the last
    # snapshot at/below until_step. args.eval_every <= 0 -> only that last point.
    candidates = [s for s in snaps if s["step"] <= until_step]
    if not candidates:
        raise SystemExit(f"No snapshots at or before step {until_step}.")
    if args.eval_every and args.eval_every > 0:
        eval_points = [s for s in candidates if s["epoch"] % args.eval_every == 0]
        if candidates[-1] not in eval_points:
            eval_points.append(candidates[-1])
    else:
        eval_points = [candidates[-1]]
    logger.info(
        f"decay={args.decay}: tracing FID at {len(eval_points)} epoch(s): "
        f"{[s['epoch'] for s in eval_points]}"
    )

    if args.out:
        # Save the reconstruction at the final evaluated epoch.
        s_last = eval_points[-1]
        used, weights = compute_weights(snaps, args.decay, s_last["step"])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reconstruct(snap_dir, used, weights), out)
        logger.info(f"Saved reconstructed weights (epoch {s_last['epoch']}): {out}")

    if not args.fid:
        if args.wandb:
            logger.warning("--wandb has no effect without --fid; skipping wandb.")
        return

    # FID — reuse the exact generation/eval path from training.
    from phase1_utils import _compute_fid, _generate_images

    # Determinism: _generate_images reseeds the RNG to FID_SEED at the start of
    # every call, so the initial ODE noise x0 is IDENTICAL across all decays and
    # epochs — FID differences reflect the weights, not the sampling noise. We
    # also pin cuDNN to deterministic conv algorithms (training uses
    # benchmark=True, which can pick different kernels run-to-run) so a rerun of
    # the same (decay, epoch) reproduces the same FID.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.sampler}__d{args.decay}",
            config={
                "sampler_id": args.sampler,
                "ckpt_base": args.ckpt_base,
                "decay": args.decay,
                "until_step": until_step,
                "eval_every": args.eval_every,
                "ode_method": "euler",
                "nfe": FID_NFE,
                "fid_samples": args.fid_samples,
                "fid_batch": FID_BATCH,
                "seed": FID_SEED,
                "cudnn_deterministic": True,
            },
        )
        logger.info(f"wandb run: {run.project}/{run.id}")

    model_key = args.model or manifest.get("model", "cifar10")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNetModel(**MODEL_CONFIGS[model_key])
    model.to(device).eval()

    trajectory = []
    for s in eval_points:
        used, weights = compute_weights(snaps, args.decay, s["step"])
        logger.info(
            f"[epoch {s['epoch']}] reconstruct decay={args.decay} at step {s['step']} "
            f"from {len(used)} snapshots "
            f"(top weight {weights.max():.4f} @ step {used[weights.argmax()]['step']})"
        )
        model.load_state_dict(reconstruct(snap_dir, used, weights))
        model.eval()

        t0 = time.time()
        logger.info(f"[epoch {s['epoch']}] generating {args.fid_samples} samples (nfe={FID_NFE}) ...")
        imgs = _generate_images(model, device, args.fid_samples, FID_BATCH, FID_NFE, FID_SEED)
        fid = _compute_fid(imgs, args.data_path, device)
        del imgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - t0
        logger.info(f"[epoch {s['epoch']}] FID={fid:.3f}  ({elapsed:.0f}s)")

        trajectory.append({
            "epoch": s["epoch"],
            "step": s["step"],
            "fid": fid,
            "n_snapshots_used": len(used),
            "top_weight": float(weights.max()),
            "elapsed_sec": round(elapsed),
        })
        if run is not None:
            # x=epoch so the FID-vs-epoch curve is plottable in wandb.
            run.log({"fid": fid, "epoch": s["epoch"], "step": s["step"]})

    best = min(trajectory, key=lambda p: p["fid"])
    logger.info(
        f"decay={args.decay}: best FID={best['fid']:.3f} @ epoch {best['epoch']} "
        f"(over {len(trajectory)} eval points)"
    )

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
        "eval_every": args.eval_every,
        "ode_method": "euler", "nfe": FID_NFE,
        "fid_samples": args.fid_samples, "batch_size": FID_BATCH, "seed": FID_SEED,
        "cudnn_deterministic": True,
        "trajectory": trajectory,
        "best_fid": best["fid"], "best_epoch": best["epoch"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    res_path.write_text(json.dumps(history, indent=2))
    logger.info(f"Appended {len(trajectory)}-point trajectory to {res_path}")

    if run is not None:
        run.summary["best_fid"] = best["fid"]
        run.summary["best_epoch"] = best["epoch"]
        run.finish()


if __name__ == "__main__":
    main()
