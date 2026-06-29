#!/usr/bin/env python3
"""
Phase 1 Objective-reduction bin: CIFAR-10 Flow Matching with an
objective-reduction training-time timestep distribution rho(t).

Sibling of train_phase1_bin_loss_aware.py (same UNet, EMA, optimizer and FID
plumbing via phase1_utils). The conceptual change is the sampler:
instead of scoring a bin by its raw per-bin loss, every `update_sampler_every`
steps we FORCE a round-robin bin, measure the FM loss over the K bin centers on
a FIXED eval set with the RAW UNet *before* and *after* that step, and use the
relative reduction over an SNR-selected eval subset as a reward. A per-bin EMA
of that reward (scaled by reward_scale before softmax) drives the categorical
distribution. See objective_reduction_bin_sampler.py and the spec (2.md).

Lives under methods/adaptive_bin/ per the repo's per-method directory
convention; the import header below puts examples/image on sys.path so the
shared assets (phase1_utils, models, training) resolve
regardless of cwd or server. All paths are derived from __file__ (no server
name baked in).

All outputs go to a SEPARATE tree checkpoints/phase1_bin_objective_reduction/
so the adaptive-bin / loss-aware / two-phase / static / posthoc runs are never
touched. The wandb project is shared with the adaptive-bin runs (config "mode"
disambiguates them).

Run from anywhere:
  python methods/adaptive_bin/train_phase1_bin_objective_reduction.py --bin-k 10
  python methods/adaptive_bin/train_phase1_bin_objective_reduction.py --dry-run
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

# --- import header: put examples/image on sys.path for shared assets ---------
_IMAGE_DIR = Path(__file__).resolve().parents[2]   # examples/image
_REPO_ROOT = Path(__file__).resolve().parents[4]   # flow_matching (repo root)
if str(_IMAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_IMAGE_DIR))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import wandb

# Local (same directory as this script; script dir is already on sys.path[0]).
from objective_reduction_bin_sampler import (
    ObjectiveReductionBinSampler,
    eval_fm_loss_at_bin_centers,
)
# Shared assets under examples/image (resolved via the import header above).
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
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (shared with the phase1_* siblings)
# ---------------------------------------------------------------------------
# Absolute (not cwd-relative) so the script finds the shared CIFAR-10 tree even
# when launched from methods/adaptive_bin/; derived from __file__ (no server name).
DATA_PATH = str(_IMAGE_DIR / "data" / "image_generation")
# Shared wandb project with adaptive-bin runs; config "mode" disambiguates.
WANDB_PROJECT = "phase1-bin-adaptive"
# Separate tree from the other phase1_* runs.
CKPT_BASE = _REPO_ROOT / "checkpoints" / "phase1_bin_objective_reduction"

LR = 1e-4
WARMUP_STEPS = 10_000  # LR-scheduler warmup (UNet); distinct from sampler warmup
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

# Objective-reduction bin sampler defaults (spec 2.md).
BIN_K = 10
REWARD_EMA_BETA = 0.9
TEMPERATURE = 0.5
UNIFORM_MIX = 0.1
UPDATE_SAMPLER_EVERY = 40
SAMPLER_WARMUP_STEPS = 1000
IMPORTANCE_GAMMA = 0.5
REWARD_SCALE = 100.0
SAMPLER_EVAL_BATCHES = 4
QUEUE_SIZE = 20
SELECTED_EVAL_COUNT = 3
SELECTION_METHOD = "snr_topk"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        "Phase 1 Objective-reduction bin: objective-reduction timestep distribution"
    )
    # objective-reduction bin sampler hyperparameters
    p.add_argument("--bin-k", type=int, default=BIN_K,
                   help=f"Number of equal-width timestep bins (default {BIN_K}).")
    p.add_argument("--reward-ema-beta", type=float, default=REWARD_EMA_BETA,
                   help=f"EMA factor for the per-bin reward (default {REWARD_EMA_BETA}).")
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help=f"Softmax temperature on reward_scale*reward_ema (default {TEMPERATURE}).")
    p.add_argument("--uniform-mix", type=float, default=UNIFORM_MIX,
                   help=f"Uniform-mixing weight; sets the exploration floor (default {UNIFORM_MIX}).")
    p.add_argument("--update-sampler-every", type=int, default=UPDATE_SAMPLER_EVERY,
                   help=f"Run a forced round-robin reward-eval step every N UNet steps "
                        f"(default {UPDATE_SAMPLER_EVERY}).")
    p.add_argument("--warmup-steps", type=int, default=SAMPLER_WARMUP_STEPS,
                   help=f"Sampler warmup: uniform sampling while global_step < this "
                        f"(default {SAMPLER_WARMUP_STEPS}).")
    p.add_argument("--importance-gamma", type=float, default=IMPORTANCE_GAMMA,
                   help=f"Exponent for partial importance weighting (default {IMPORTANCE_GAMMA}).")
    p.add_argument("--reward-scale", type=float, default=REWARD_SCALE,
                   help=f"Multiply reward_ema before softmax to widen prob swings (default {REWARD_SCALE}).")
    p.add_argument("--sampler-eval-batches", type=int, default=SAMPLER_EVAL_BATCHES,
                   help=f"Number of fixed eval batches for reward measurement (default {SAMPLER_EVAL_BATCHES}).")
    p.add_argument("--queue-size", type=int, default=QUEUE_SIZE,
                   help=f"Length of the per-eval-time loss-reduction (delta) queue (default {QUEUE_SIZE}).")
    p.add_argument("--selected-eval-count", type=int, default=SELECTED_EVAL_COUNT,
                   help=f"Number of eval times in the reward subset S (default {SELECTED_EVAL_COUNT}).")
    p.add_argument("--selection-method", type=str, default=SELECTION_METHOD,
                   help=f"Eval-subset selection method (default {SELECTION_METHOD}).")
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
def _eval_on_fixed_set(raw_model, eval_samples, eval_noise, path, k):
    """before/after reward eval on the FIXED eval set, RAW UNet in eval mode.

    Toggling eval mode (dropout off) makes the before/after comparison clean;
    raw_model is the online UNet so .eval()/.train() does NOT swap EMA weights.
    """
    was_training = raw_model.training
    raw_model.eval()
    losses = eval_fm_loss_at_bin_centers(eval_samples, eval_noise, path, raw_model, k)
    if was_training:
        raw_model.train()
    return losses


def train_one_epoch(ema_model, raw_model, dataloader, optimizer, scheduler,
                    device, epoch, global_step, run, bin_sampler,
                    eval_samples, eval_noise):
    ema_model.train(True)
    path = CondOTProbPath()
    k = bin_sampler.k
    total_loss = 0.0
    n_batches = 0

    for samples, _ in dataloader:
        samples = samples.to(device) * 2.0 - 1.0
        noise = torch.randn_like(samples)

        # Decide up-front whether this is a sampler-update (reward-eval) step so
        # we can force the round-robin bin and measure before-losses first.
        do_sampler_update = (global_step + 1) % bin_sampler.update_sampler_every == 0
        force_bin = bin_sampler.round_robin_idx if do_sampler_update else None

        # 1-batch-1-bin: pick a bin, draw batch_size uniform t inside it.
        t = bin_sampler.sample_t(samples.shape[0], device,
                                 global_step=global_step, force_bin=force_bin)

        # Reward eval (before) on the FIXED eval set (not the training batch).
        before_losses = None
        if do_sampler_update:
            before_losses = _eval_on_fixed_set(
                raw_model, eval_samples, eval_noise, path, k)

        # Training step uses the CURRENT training batch (with importance weight).
        ps = path.sample(t=t, x_0=noise, x_1=samples)
        pred = ema_model(ps.x_t, t, extra={})
        base_loss = (pred - ps.dx_t).pow(2).mean()

        # Partial importance weighting on the TRAINING loss only (q_b = 1/K
        # during warmup -> weight == 1). Reward eval is never reweighted.
        iw = bin_sampler.importance_weight()
        loss = iw * base_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_ema()

        global_step += 1
        total_loss += loss.item()
        n_batches += 1

        if do_sampler_update:
            after_losses = _eval_on_fixed_set(
                raw_model, eval_samples, eval_noise, path, k)

            rr_used = bin_sampler.round_robin_idx  # forced bin (== selected_bin)
            stats = bin_sampler.update(before_losses, after_losses)

            if run is not None:
                n_sel = bin_sampler.selected_eval_count
                delta = stats["delta"]
                scores = stats["scores"]
                log_idx = stats["log_indices"]
                payload = bin_sampler.bin_prob_dict()
                payload |= bin_sampler.reward_ema_dict()
                payload |= bin_sampler.scaled_reward_ema_dict()
                payload |= {f"sampler/delta_{i}": float(delta[i]) for i in range(k)}
                payload |= {f"sampler/selection_score_{i}": float(scores[i]) for i in range(k)}
                payload |= {f"sampler/selected_eval_idx_{i}": int(log_idx[i]) for i in range(n_sel)}
                payload |= {f"sampler/selected_eval_time_{i}": (int(log_idx[i]) + 0.5) / k
                            for i in range(n_sel)}
                payload |= {
                    "sampler/selected_bin": bin_sampler.selected_bin,
                    "sampler/q_selected": bin_sampler.selected_prob,
                    "sampler/reward": stats["reward"],
                    "sampler/relative_reward": stats["relative_reward"],
                    "sampler/eval_loss_before": stats["eval_loss_before"],
                    "sampler/eval_loss_after": stats["eval_loss_after"],
                    "sampler/eval_loss_mean_before": stats["eval_loss_mean_before"],
                    "sampler/eval_loss_mean_after": stats["eval_loss_mean_after"],
                    "sampler/selected_delta_mean": stats["selected_delta_mean"],
                    "sampler/selected_before_mean": stats["selected_before_mean"],
                    "sampler/selected_eval_count": n_sel,
                    "sampler/selection_method": bin_sampler.selection_method,
                    "sampler/using_all_eval_times": stats["using_all"],
                    "sampler/queue_len": stats["queue_len"],
                    "sampler/queue_size": bin_sampler.queue_size,
                    "sampler/reward_scale": bin_sampler.reward_scale,
                    "sampler/importance_weight": iw,
                    "sampler/base_train_loss": base_loss.item(),
                    "sampler/weighted_train_loss": loss.item(),
                    "sampler/entropy": float(bin_sampler.entropy().item()),
                    "sampler/max_prob": float(bin_sampler.last_probs.max().item()),
                    "sampler/min_prob": float(bin_sampler.last_probs.min().item()),
                    "sampler/update_flag": 1,
                    "sampler/round_robin_idx": rr_used,
                    "sampler/num_sampler_updates": bin_sampler.num_sampler_updates,
                    "global_step": global_step,
                }
                run.log(payload)

        if run is not None and global_step % 100 == 0:
            payload = {
                "train/loss": loss.item(),
                "train/base_loss": base_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
            payload |= bin_sampler.bin_prob_dict()
            payload |= bin_sampler.reward_ema_dict()
            payload |= bin_sampler.scaled_reward_ema_dict()
            payload |= {
                "sampler/selected_bin": bin_sampler.selected_bin,
                "sampler/q_selected": bin_sampler.selected_prob,
                "sampler/importance_weight": iw,
                "sampler/reward_scale": bin_sampler.reward_scale,
                "sampler/queue_len": len(bin_sampler.delta_queue),
                "sampler/entropy": float(bin_sampler.entropy().item()),
                "sampler/max_prob": float(bin_sampler.last_probs.max().item()),
                "sampler/min_prob": float(bin_sampler.last_probs.min().item()),
                "sampler/update_flag": 0,
                "global_step": global_step,
            }
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

    sampler_id = args.sampler_id or (
        f"objred_K{args.bin_k}_T{args.temperature}"
        f"_mix{args.uniform_mix}_beta{args.reward_ema_beta}_g{args.importance_gamma}"
        f"_rs{args.reward_scale:g}_eval{args.sampler_eval_batches}"
        f"_Q{args.queue_size}_S{args.selected_eval_count}"
    )
    # Keep --dry-run smoke-test artifacts out of the production tree so a real
    # run never resumes a 2-epoch dry-run checkpoint (only when the id is auto).
    if args.dry_run and args.sampler_id is None:
        sampler_id += "_dryrun"
    ckpt_dir = CKPT_BASE / sampler_id

    logger.info(
        f"Phase 1 Objective-reduction bin  id={sampler_id}\n"
        f"  K={args.bin_k}  temperature={args.temperature}  uniform_mix={args.uniform_mix}\n"
        f"  reward_ema_beta={args.reward_ema_beta}  update_sampler_every={args.update_sampler_every}\n"
        f"  warmup_steps={args.warmup_steps}  importance_gamma={args.importance_gamma}\n"
        f"  reward_scale={args.reward_scale}  sampler_eval_batches={args.sampler_eval_batches}\n"
        f"  queue_size={args.queue_size}  selected_eval_count={args.selected_eval_count}"
        f"  selection_method={args.selection_method}\n"
        f"  ckpt_dir={ckpt_dir}"
    )

    # Apply --dry-run defaults
    if args.dry_run:
        args.max_epochs = args.max_epochs or 2
        args.eval_every = args.eval_every or 2
        args.fid_samples = args.fid_samples or 2_000
    else:
        args.max_epochs = args.max_epochs or EPOCHS
        args.eval_every = args.eval_every or EVAL_EVERY
        args.fid_samples = args.fid_samples or FID_SAMPLES

    keep_epochs = {int(x) for x in args.keep_epochs.split(",") if x.strip()}
    ckpt_every = args.ckpt_every or args.eval_every
    keep_recent_n = 10**9 if args.keep_all else args.keep_recent_n

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

    # Objective-reduction bin sampler (parameter-free; only a per-bin reward
    # EMA, restored from checkpoint by load_checkpoint below if present).
    bin_sampler = ObjectiveReductionBinSampler(
        k=args.bin_k,
        reward_ema_beta=args.reward_ema_beta,
        temperature=args.temperature,
        uniform_mix=args.uniform_mix,
        update_sampler_every=args.update_sampler_every,
        warmup_steps=args.warmup_steps,
        importance_gamma=args.importance_gamma,
        reward_scale=args.reward_scale,
        queue_size=args.queue_size,
        selected_eval_count=args.selected_eval_count,
        selection_method=args.selection_method,
        sampler_eval_batches=args.sampler_eval_batches,
        device=device,
    )

    eval_times = [(i + 0.5) / args.bin_k for i in range(args.bin_k)]

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

    # Fixed eval set for reward measurement: take sampler_eval_batches batches
    # ONCE (in [-1,1]) plus a frozen randn noise tensor. Reward eval always uses
    # this reference set; the training loss never does. Built after the RNG is
    # seeded so it is reproducible across resumes.
    eval_samples, eval_noise = [], []
    _eval_it = iter(dataloader)
    for _ in range(args.sampler_eval_batches):
        _s, _ = next(_eval_it)
        _s = _s.to(device) * 2.0 - 1.0
        eval_samples.append(_s)
        eval_noise.append(torch.randn_like(_s))
    del _eval_it
    n_eval_imgs = sum(s.shape[0] for s in eval_samples)
    logger.info(
        f"Fixed eval set: {args.sampler_eval_batches} batches "
        f"({n_eval_imgs} images) for reward measurement"
    )

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
                "mode": "objective_reduction_bin",
                "bin_k": args.bin_k,
                "reward_ema_beta": args.reward_ema_beta,
                "temperature": args.temperature,
                "uniform_mix": args.uniform_mix,
                "update_sampler_every": args.update_sampler_every,
                "warmup_steps": args.warmup_steps,
                "importance_gamma": args.importance_gamma,
                "reward_scale": args.reward_scale,
                "sampler_eval_batches": args.sampler_eval_batches,
                "queue_size": args.queue_size,
                "selected_eval_count": args.selected_eval_count,
                "selection_method": args.selection_method,
                "eval_times": eval_times,
                "checkpoint_dir": str(ckpt_dir),
                "max_epochs": args.max_epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "lr_warmup_steps": WARMUP_STEPS,
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

    # Keep W&B's internal Step from mixing with the training global_step.
    if run is not None:
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")

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
            eval_samples, eval_noise,
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
