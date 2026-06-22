#!/usr/bin/env python3
"""
Phase 1 Adaptive-bin: CIFAR-10 Flow Matching with a *learned* training-time
timestep distribution.

Sibling of train_phase1_twophase.py / _posthoc.py / _static.py (same UNet, EMA,
optimizer, FID/snapshot plumbing via phase1_utils). The ONLY conceptual change
is the training-time timestep sampler ρ(t):

  * t in [0,1] is split into K equal-width bins (default K=10).
  * A categorical distribution over bins, p = softmax(bin_logits), is LEARNED.
  * Each batch samples ONE bin, then draws batch_size uniform t inside it
    (1-batch-1-bin). The UNet is updated once with the usual FM loss.
  * Every `update_sampler_every` steps the bin distribution is updated by
    REINFORCE: reward = FM-loss at fixed eval timesteps S BEFORE the UNet step
    minus the same loss AFTER the step (drop in loss = the chosen bin helped).

All outputs go to a SEPARATE tree checkpoints/phase1_bin_adaptive/ and a SEPARATE
wandb project, so two-phase / static / posthoc runs are never touched.

Run from: flow_matching/examples/image/
  python train_phase1_bin_adaptive.py --bin-k 10
  python train_phase1_bin_adaptive.py --dry-run            # smoke-test
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import wandb

from adaptive_bin_sampler import AdaptiveBinSampler, eval_fm_loss_at_times
from flow_matching.path import CondOTProbPath
from models.ema import EMA
from models.model_configs import MODEL_CONFIGS
from models.unet import UNetModel
from phase1_utils import (
    compute_per_timestep_loss,
    evaluate_fid,
    load_checkpoint,
    load_fid_history,
    save_checkpoint,
    save_fid_history,
)
from posthoc_snapshot import save_snapshot, snapshot_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (shared with the phase1_* siblings)
# ---------------------------------------------------------------------------
DATA_PATH = "./data/image_generation"
WANDB_PROJECT = "phase1-bin-adaptive"
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Separate tree from phase1_static / _posthoc / _twophase so those are untouched.
CKPT_BASE = _REPO_ROOT / "checkpoints" / "phase1_bin_adaptive"
MODEL_KEY = "cifar10"

SNAPSHOT_EVERY = 2

LR = 1e-4
WARMUP_STEPS = 10_000
BATCH_SIZE = 64
EPOCHS = 180
EMA_DECAY = 0.99995

EVAL_EVERY = 20
FID_SAMPLES = 50_000
FID_NFE = 50
FID_BATCH = 250
FID_SEED = 0

KEEP_EPOCHS = {60, 120, 180}
KEEP_RECENT_N = 3

# Adaptive-bin sampler defaults (spec)
BIN_K = 10
SAMPLER_LR = 1e-3
BASELINE_BETA = 0.9
ENTROPY_COEF = 0.01
UPDATE_SAMPLER_EVERY = 40
EVAL_TIMES = "0.1,0.5,0.9"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Phase 1 Adaptive-bin: learned timestep distribution")
    # adaptive-bin sampler hyperparameters
    p.add_argument("--bin-k", type=int, default=BIN_K,
                   help=f"Number of equal-width timestep bins (default {BIN_K}).")
    p.add_argument("--sampler-lr", type=float, default=SAMPLER_LR,
                   help=f"Adam LR for bin_logits (default {SAMPLER_LR}).")
    p.add_argument("--baseline-beta", type=float, default=BASELINE_BETA,
                   help=f"EMA factor for the REINFORCE baseline (default {BASELINE_BETA}).")
    p.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF,
                   help=f"Entropy bonus weight (default {ENTROPY_COEF}).")
    p.add_argument("--update-sampler-every", type=int, default=UPDATE_SAMPLER_EVERY,
                   help=f"Update the bin distribution every N UNet steps (default {UPDATE_SAMPLER_EVERY}).")
    p.add_argument("--eval-times", type=str, default=EVAL_TIMES,
                   help=f"Comma-separated eval timesteps S for the reward (default '{EVAL_TIMES}').")
    # run management (mirrors the phase1_* siblings)
    p.add_argument("--sampler-id", type=str, default=None,
                   help="Override the auto-generated run id / checkpoint dir name.")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, eval every 2, 2000 FID samples (smoke-test)")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--keep-epochs", type=str, default="60,120,180",
                   help="Comma-separated epochs whose periodic checkpoints are never pruned.")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="Cadence (epochs) for full resume checkpoints. Default = --eval-every.")
    p.add_argument("--keep-recent-n", type=int, default=KEEP_RECENT_N,
                   help=f"Keep the most recent N periodic checkpoints (default {KEEP_RECENT_N}).")
    p.add_argument("--keep-all", action="store_true",
                   help="Never prune periodic checkpoints.")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=None,
                   help=f"Save a weights-only snapshot every N epochs (default {SNAPSHOT_EVERY}).")
    p.add_argument("--fid-samples", type=int, default=None)
    p.add_argument("--data-path", type=str, default=DATA_PATH)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true",
                   help="Run FID eval for the epoch stored in latest.pt, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(ema_model, raw_model, dataloader, optimizer, scheduler,
                    device, epoch, global_step, run, bin_sampler):
    ema_model.train(True)
    path = CondOTProbPath()
    total_loss = 0.0
    n_batches = 0

    for samples, _ in dataloader:
        samples = samples.to(device) * 2.0 - 1.0
        noise = torch.randn_like(samples)

        # 1-batch-1-bin: pick a bin, draw batch_size uniform t inside it.
        t = bin_sampler.sample_t(samples.shape[0], device)

        # Sampler is updated only every `update_sampler_every` UNet steps. On those
        # steps we measure the FM loss at the fixed eval timesteps S right BEFORE
        # the UNet update so we can compare against AFTER (reward = before - after).
        # Reward eval uses the RAW UNet (not the EMA model) so the effect of this
        # single optimizer step is visible instead of being washed out by EMA.
        do_sampler_update = (global_step + 1) % bin_sampler.update_sampler_every == 0
        eval_before = None
        if do_sampler_update:
            eval_before = eval_fm_loss_at_times(
                bin_sampler.eval_times, samples, noise, path, raw_model
            ).item()

        ps = path.sample(t=t, x_0=noise, x_1=samples)
        pred = ema_model(ps.x_t, t, extra={})
        loss = (pred - ps.dx_t).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_ema()

        global_step += 1
        total_loss += loss.item()
        n_batches += 1

        # Sampler (REINFORCE) update, using the post-step eval loss for the reward.
        if do_sampler_update:
            eval_after = eval_fm_loss_at_times(
                bin_sampler.eval_times, samples, noise, path, raw_model
            ).item()
            stats = bin_sampler.update(eval_before, eval_after)
            if run is not None:
                run.log(
                    bin_sampler.bin_prob_dict()
                    | {f"sampler/{k}": v for k, v in stats.items()}
                    | {"sampler/update_flag": 1, "global_step": global_step}
                )

        if run is not None and global_step % 100 == 0:
            payload = {
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
            payload |= bin_sampler.bin_prob_dict()
            payload["sampler/entropy"] = float(bin_sampler.entropy().item())
            payload["sampler/selected_bin"] = bin_sampler._last_bin
            payload["sampler/update_flag"] = 0
            run.log(payload)

        if global_step % 1000 == 0:
            logger.info(
                f"  step={global_step}  loss={loss.item():.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}  steps={global_step}")
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    args = get_args()

    eval_times = [float(x) for x in args.eval_times.split(",") if x.strip()]
    sampler_id = args.sampler_id or f"adaptive_bin_K{args.bin_k}"
    ckpt_dir = CKPT_BASE / sampler_id

    logger.info(
        f"Phase 1 Adaptive-bin  id={sampler_id}\n"
        f"  K={args.bin_k}  sampler_lr={args.sampler_lr}  baseline_beta={args.baseline_beta}\n"
        f"  entropy_coef={args.entropy_coef}  update_sampler_every={args.update_sampler_every}\n"
        f"  eval_times={eval_times}\n"
        f"  ckpt_dir={ckpt_dir}"
    )

    # Apply --dry-run defaults
    if args.dry_run:
        args.max_epochs = args.max_epochs or 2
        args.eval_every = args.eval_every or 2
        args.fid_samples = args.fid_samples or 2_000
        args.snapshot_every = args.snapshot_every or 1
    else:
        args.max_epochs = args.max_epochs or EPOCHS
        args.eval_every = args.eval_every or EVAL_EVERY
        args.fid_samples = args.fid_samples or FID_SAMPLES
        args.snapshot_every = args.snapshot_every or SNAPSHOT_EVERY

    keep_epochs = {int(x) for x in args.keep_epochs.split(",") if x.strip()}
    ckpt_every = args.ckpt_every or args.eval_every
    keep_recent_n = 10**9 if args.keep_all else args.keep_recent_n

    snap_dir = snapshot_dir(ckpt_dir)

    logger.info(
        f"max_epochs={args.max_epochs}  eval_every={args.eval_every}  dry_run={args.dry_run}"
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Model
    logger.info("Building UNet (111M params) + EMA …")
    unet = UNetModel(**MODEL_CONFIGS["cifar10"])
    ema_model = EMA(model=unet, decay=EMA_DECAY)
    ema_model.to(device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"UNet trainable params: {n_params/1e6:.1f}M")

    # Optimizer (UNet)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Adaptive-bin sampler (its own Adam optimizer + EMA baseline, kept separate
    # from the UNet optimizer; restored from checkpoint by load_checkpoint below).
    bin_sampler = AdaptiveBinSampler(
        k=args.bin_k,
        sampler_lr=args.sampler_lr,
        baseline_beta=args.baseline_beta,
        entropy_coef=args.entropy_coef,
        update_sampler_every=args.update_sampler_every,
        eval_times=eval_times,
        device=device,
    )

    # Dataset
    logger.info(f"Loading CIFAR-10 from {args.data_path} …")
    from training.data_transform import get_train_transform
    transform = get_train_transform()
    dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset)} samples  {len(dataloader)} batches/epoch")

    # Resume (also restores the bin sampler if present in the checkpoint).
    start_epoch, global_step, wandb_run_id = load_checkpoint(
        ckpt_dir, args, ema_model, optimizer, scheduler, bin_sampler=bin_sampler
    )

    # wandb
    if wandb_run_id is not None:
        run = wandb.init(project=WANDB_PROJECT, id=wandb_run_id, resume="must")
        logger.info(f"Resumed wandb run: {wandb_run_id}")
    else:
        run = wandb.init(
            project=WANDB_PROJECT,
            config={
                "sampler_id": sampler_id,
                "mode": "adaptive_bin",
                "bin_k": args.bin_k,
                "sampler_lr": args.sampler_lr,
                "baseline_beta": args.baseline_beta,
                "entropy_coef": args.entropy_coef,
                "update_sampler_every": args.update_sampler_every,
                "eval_times": eval_times,
                "max_epochs": args.max_epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "warmup_steps": WARMUP_STEPS,
                "ema_decay": EMA_DECAY,
                "fid_nfe": FID_NFE,
                "fid_samples": FID_SAMPLES,
                "fid_batch": FID_BATCH,
                "precision": "fp32",
                "optimizer": "AdamW",
                "optimizer_betas": [0.9, 0.95],
                "unet_model_channels": 128,
                "unet_channel_mult": [2, 2, 2],
                "unet_num_res_blocks": 4,
            },
        )
        wandb_run_id = run.id
        logger.info(f"New wandb run: {wandb_run_id}")

    fid_history = [] if args.no_resume else load_fid_history(ckpt_dir)

    # --eval-only: run FID for the checkpoint epoch and exit
    if args.eval_only:
        epoch_1indexed = start_epoch  # start_epoch = last completed epoch
        logger.info(f"--eval-only: evaluating epoch {epoch_1indexed} …")
        per_ts = compute_per_timestep_loss(unet, dataloader, device)
        logger.info(f"Per-timestep loss: {per_ts}")

        existing = next((e for e in fid_history if e["epoch"] == epoch_1indexed), None)
        if existing is None:
            existing = {"epoch": epoch_1indexed, "global_step": global_step}
            fid_history.append(existing)
        existing["per_timestep_loss"] = per_ts
        save_fid_history(ckpt_dir, fid_history)

        if run is not None:
            run.log(
                {f"per_t_loss/{k}": v for k, v in per_ts.items()}
                | {"epoch": epoch_1indexed, "global_step": global_step}
            )

        try:
            fid_result = evaluate_fid(
                ema_model, device, args.data_path,
                epoch_1indexed, global_step, fid_history,
                n_samples=args.fid_samples or FID_SAMPLES,
                fid_batch=FID_BATCH,
                fid_nfe=FID_NFE,
                fid_seed=FID_SEED,
            )
        except Exception:
            logger.exception(f"FID evaluation failed at epoch {epoch_1indexed} – per_t_loss already saved")
            fid_result = None

        if fid_result is not None:
            existing["fid_ema"] = fid_result["fid_ema"]
            existing["fid_raw"] = fid_result["fid_raw"]
            save_fid_history(ckpt_dir, fid_history)
            logger.info(
                f"FID@epoch{epoch_1indexed}: EMA={fid_result['fid_ema']:.3f}"
                f"  raw={fid_result['fid_raw']:.3f}"
            )
            if run is not None:
                run.log({
                    "eval/fid_ema": fid_result["fid_ema"],
                    "eval/fid_raw": fid_result["fid_raw"],
                    "epoch": epoch_1indexed,
                    "global_step": global_step,
                })

        if run is not None:
            run.finish()
        return

    # Training loop
    logger.info(f"Training epochs {start_epoch+1} → {args.max_epochs} …")
    total_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_1indexed = epoch + 1

        epoch_start = time.time()
        avg_loss, global_step = train_one_epoch(
            ema_model, unet, dataloader, optimizer, scheduler,
            device, epoch, global_step, run, bin_sampler,
        )
        epoch_sec = time.time() - epoch_start

        if run is not None:
            run.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch_1indexed,
                "global_step": global_step,
                "epoch_time_sec": epoch_sec,
            })

        save_checkpoint(
            ckpt_dir, ema_model, optimizer, scheduler,
            epoch, global_step, wandb_run_id,
            ckpt_every, keep_epochs, keep_recent_n,
            sampler_state=bin_sampler.state_dict(),
        )

        # Post-hoc EMA snapshot: raw weights only, fp16, never pruned.
        if (epoch_1indexed % args.snapshot_every == 0
                or epoch_1indexed == args.max_epochs):
            save_snapshot(
                snap_dir, unet, global_step,
                num_updates=ema_model.num_updates.item(),
                epoch_1indexed=epoch_1indexed,
                sampler_id=sampler_id,
                model_key=MODEL_KEY,
            )

        is_eval_epoch = (
            epoch_1indexed % args.eval_every == 0
            or epoch_1indexed == args.max_epochs
        )

        if is_eval_epoch:
            logger.info(f"Computing per-timestep loss at epoch {epoch_1indexed} …")
            per_ts = compute_per_timestep_loss(unet, dataloader, device)
            logger.info(f"Per-timestep loss: {per_ts}")

            existing = next((e for e in fid_history if e["epoch"] == epoch_1indexed), None)
            if existing is None:
                existing = {"epoch": epoch_1indexed, "global_step": global_step}
                fid_history.append(existing)
            existing["per_timestep_loss"] = per_ts
            save_fid_history(ckpt_dir, fid_history)

            if run is not None:
                run.log(
                    {f"per_t_loss/{k}": v for k, v in per_ts.items()}
                    | {"epoch": epoch_1indexed, "global_step": global_step}
                )

            try:
                fid_result = evaluate_fid(
                    ema_model, device, args.data_path,
                    epoch_1indexed, global_step, fid_history,
                    n_samples=args.fid_samples,
                    fid_batch=FID_BATCH,
                    fid_nfe=FID_NFE,
                    fid_seed=FID_SEED,
                )
            except Exception:
                logger.exception(f"FID evaluation failed at epoch {epoch_1indexed} – per_t_loss already saved")
                fid_result = None

            if fid_result is not None:
                existing["fid_ema"] = fid_result["fid_ema"]
                existing["fid_raw"] = fid_result["fid_raw"]
                save_fid_history(ckpt_dir, fid_history)
                logger.info(
                    f"FID@epoch{epoch_1indexed}: EMA={fid_result['fid_ema']:.3f}"
                    f"  raw={fid_result['fid_raw']:.3f}"
                )

                if run is not None:
                    run.log({
                        "eval/fid_ema": fid_result["fid_ema"],
                        "eval/fid_raw": fid_result["fid_raw"],
                        "epoch": epoch_1indexed,
                        "global_step": global_step,
                    })

    total_sec = time.time() - total_start
    logger.info(f"Training complete. Total time: {total_sec/3600:.1f}h")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
