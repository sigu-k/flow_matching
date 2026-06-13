#!/usr/bin/env python3
"""
Phase 1 Static: CIFAR-10 Flow Matching with configurable timestep sampler.
150-epoch runs for LN(μ=-0.8), Mode(s=-0.5), Mode(s=+1.0).

Run from: flow_matching/examples/image/
  source ~/work/srv11/setup_env.sh
  python train_phase1_static.py --config configs/phase1_static/ln_mu-0.8.yaml
  python train_phase1_static.py --config configs/phase1_static/ln_mu-0.8.yaml --dry-run
  python train_phase1_static.py --config configs/phase1_static/ln_mu-0.8.yaml --no-resume
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import wandb
import yaml

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
from timestep_sampler import sample_logit_normal, sample_mode, sample_uniform
from training.data_transform import get_train_transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (Phase 1 specific)
# ---------------------------------------------------------------------------
DATA_PATH = "./data/image_generation"
WANDB_PROJECT = "phase1-static-baselines"
CKPT_BASE = Path(os.path.expanduser("~/work/srv11/checkpoints/phase1_static"))

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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Phase 1 Static: configurable timestep sampler")
    p.add_argument("--config", required=True,
                   help="Path to YAML config, e.g. configs/phase1_static/ln_mu-0.8.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, eval every 2, 2000 FID samples (smoke-test)")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--fid-samples", type=int, default=None)
    p.add_argument("--data-path", type=str, default=DATA_PATH)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true",
                   help="Run FID eval for the epoch stored in latest.pt, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------
def build_sampler(cfg: dict):
    """Return a callable (batch_size, device) -> Tensor from sampler config dict."""
    stype = cfg["type"]
    if stype == "uniform":
        return lambda bs, dev: sample_uniform(bs, dev)
    elif stype == "logit_normal":
        mu = float(cfg["mu"])
        sigma = float(cfg["sigma"])
        return lambda bs, dev: sample_logit_normal(bs, mu=mu, sigma=sigma, device=dev)
    elif stype == "mode":
        s = float(cfg["s"])
        return lambda bs, dev: sample_mode(bs, s=s, device=dev)
    else:
        raise ValueError(f"Unknown sampler type: {stype!r}")


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(ema_model, dataloader, optimizer, scheduler,
                    device, epoch, global_step, run, sample_t):
    ema_model.train(True)
    path = CondOTProbPath()
    total_loss = 0.0
    n_batches = 0

    for samples, _ in dataloader:
        samples = samples.to(device) * 2.0 - 1.0
        noise = torch.randn_like(samples)

        t = sample_t(samples.shape[0], device)

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

        if run is not None and global_step % 100 == 0:
            run.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            })

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

    # Load YAML config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sampler_id = cfg.get("sampler_id") or Path(args.config).stem
    sampler_cfg = cfg["sampler"]
    sample_t = build_sampler(sampler_cfg)

    ckpt_dir = CKPT_BASE / sampler_id

    logger.info(f"Phase 1 Static  sampler={sampler_id}  ckpt_dir={ckpt_dir}")

    # Apply --dry-run defaults
    if args.dry_run:
        args.max_epochs = args.max_epochs or 2
        args.eval_every = args.eval_every or 2
        args.fid_samples = args.fid_samples or 2_000
    else:
        args.max_epochs = args.max_epochs or EPOCHS
        args.eval_every = args.eval_every or EVAL_EVERY
        args.fid_samples = args.fid_samples or FID_SAMPLES

    logger.info(
        f"max_epochs={args.max_epochs}  eval_every={args.eval_every}"
        f"  dry_run={args.dry_run}"
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

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Dataset
    logger.info(f"Loading CIFAR-10 from {args.data_path} …")
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

    # Resume
    start_epoch, global_step, wandb_run_id = load_checkpoint(
        ckpt_dir, args, ema_model, optimizer, scheduler
    )

    # wandb
    wandb_sampler_cfg = {k: v for k, v in sampler_cfg.items() if v is not None}
    if wandb_run_id is not None:
        run = wandb.init(project=WANDB_PROJECT, id=wandb_run_id, resume="must")
        logger.info(f"Resumed wandb run: {wandb_run_id}")
    else:
        run = wandb.init(
            project=WANDB_PROJECT,
            config={
                "sampler_id": sampler_id,
                "sampler": wandb_sampler_cfg,
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

    fid_history = load_fid_history(ckpt_dir)

    # --eval-only: run FID for the checkpoint epoch and exit
    if args.eval_only:
        epoch_1indexed = start_epoch  # start_epoch = last completed epoch
        logger.info(f"--eval-only: evaluating epoch {epoch_1indexed} …")
        per_ts = compute_per_timestep_loss(unet, dataloader, device)
        logger.info(f"Per-timestep loss: {per_ts}")

        # Persist per_t_loss immediately (upsert) so it survives even if FID fails
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
        epoch_start = time.time()
        avg_loss, global_step = train_one_epoch(
            ema_model, dataloader, optimizer, scheduler,
            device, epoch, global_step, run, sample_t,
        )
        epoch_sec = time.time() - epoch_start

        if run is not None:
            run.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_time_sec": epoch_sec,
            })

        save_checkpoint(
            ckpt_dir, ema_model, optimizer, scheduler,
            epoch, global_step, wandb_run_id,
            args.eval_every, KEEP_EPOCHS, KEEP_RECENT_N,
        )

        epoch_1indexed = epoch + 1
        is_eval_epoch = (
            epoch_1indexed % args.eval_every == 0
            or epoch_1indexed == args.max_epochs
        )

        if is_eval_epoch:
            logger.info(f"Computing per-timestep loss at epoch {epoch_1indexed} …")
            per_ts = compute_per_timestep_loss(unet, dataloader, device)
            logger.info(f"Per-timestep loss: {per_ts}")

            # Persist per_t_loss immediately (upsert) so it survives even if FID fails
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
