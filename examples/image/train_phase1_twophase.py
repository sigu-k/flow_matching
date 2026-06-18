#!/usr/bin/env python3
"""
Phase 1 Two-phase: CIFAR-10 Flow Matching with a timestep sampler that switches
between two distributions at a fixed epoch.

Copy of train_phase1_posthoc.py. The ONLY behavioural change is that the
training-time timestep distribution ρ(t) is split into two phases:

    epoch 1 .. change_epoch      -> phase-1 sampler
    epoch change_epoch+1 .. end  -> phase-2 sampler

Each phase is an existing single-distribution YAML config (the same files used by
train_phase1_static / _posthoc) plus its own uniform-mixture alpha. Like posthoc,
it ALSO dumps fp16 weights-only snapshots so any EMA decay can be reconstructed
offline (reconstruct_ema.py). All outputs go to a SEPARATE directory tree:
checkpoints/phase1_twophase/, so single-phase runs are never touched.

Run from: flow_matching/examples/image/
  # first=LN(mu=-0.8, alpha=0), second=uniform, switch after epoch 60
  python train_phase1_twophase.py \
      --phase1-config configs/phase1_static/ln_mu-0.8.yaml --phase1-alpha 0 \
      --phase2-config configs/phase1_static/uniform.yaml   --phase2-alpha 0 \
      --change-epoch 60
"""
"""
① 新規実行(source = e60)

  PHASE1_CONFIG=configs/phase1_static/ln_mu+0.8.yaml ; PHASE1_ALPHA=0
  PHASE2_CONFIG=configs/phase1_static/uniform.yaml   ; PHASE2_ALPHA=0
  CHANGE_EPOCH=60
  CKPT_EPOCHS=20,40,60     # fork 元として残す(40→e40/e50, 60→e70, 20→e30)
  INIT_FROM=               # 空 = 新規
  bash scripts/run_phase1_twophase.sh
  → 出力 checkpoints/phase1_twophase/ln_mu+0.8__uniform__e60/、ログ logs/...

  ---
  ② 再実行 / 再開(中断・クラッシュ後)

  ①と全く同じ設定のまま、もう一度実行するだけ。

  bash scripts/run_phase1_twophase.sh
  - 自分の dir の latest.pt(毎 epoch 保存)から自動再開。wandb も同じ run を継続。
  - INIT_FROM が入っていても、自分の latest.pt があればそちらを優先(fork 元に巻き戻らない)。
  - 最初からやり直すときだけ --no-resume を足す(※同じ dir を上書きするので注意)。

  ---
  ③ fork(source の checkpoint から派生)

  CHANGE_EPOCH と INIT_FROM を変えるだけ。INIT_FROM は source の epoch ≤ 新 CHANGE_EPOCH。

  # 例: e50 を作る(epoch40 から → 41〜50 を phase1 再計算 → 51〜 phase2)
  PHASE1_CONFIG=configs/phase1_static/ln_mu+0.8.yaml ; PHASE1_ALPHA=0
  PHASE2_CONFIG=configs/phase1_static/uniform.yaml   ; PHASE2_ALPHA=0
  CHANGE_EPOCH=50
  CKPT_EPOCHS=                # fork 先では基本不要(空)
  INIT_FROM=../../checkpoints/phase1_twophase/ln_mu+0.8__uniform__e60/ckpt_epoch040.pt
  bash scripts/run_phase1_twophase.sh

  各 run の設定値:

  ┌──────────┬──────────────┬──────────────────────────────────────┐
  │ 作る run │ CHANGE_EPOCH │ INIT_FROM(末尾 .../ckpt_epochNNN.pt) │
  ├──────────┼──────────────┼──────────────────────────────────────┤
  │ e30      │ 30           │ ..._e60/ckpt_epoch020.pt             │
  ├──────────┼──────────────┼──────────────────────────────────────┤
  │ e40      │ 40           │ ..._e60/ckpt_epoch040.pt             │
  ├──────────┼──────────────┼──────────────────────────────────────┤
  │ e50      │ 50           │ ..._e60/ckpt_epoch040.pt             │
  ├──────────┼──────────────┼──────────────────────────────────────┤
  │ e70      │ 70           │ ..._e60/ckpt_epoch060.pt             │
  └──────────┴──────────────┴──────────────────────────────────────┘

  (共通 prefix: ../../checkpoints/phase1_twophase/ln_mu+0.8__uniform)

  fork 後の挙動:
  - 新規 dir ..._e30/, ..._e40/ … に出力(source には触れない)
  - wandb は新規 run
  - fork 後の再実行は②と同じ(自分の latest.pt から再開)

  ---
  直接実行版(sh を使わない場合の雛形)

  cd <repo>/examples/image
  export PYTHONPATH="<repo>:$PYTHONPATH" ; mkdir -p logs
  nohup python train_phase1_twophase.py \
      --phase1-config configs/phase1_static/ln_mu+0.8.yaml --phase1-alpha 0 \
      --phase2-config configs/phase1_static/uniform.yaml   --phase2-alpha 0 \
      --change-epoch 60 \
      --ckpt-epochs 20,40,60 \           # ①新規のとき
      >> logs/run_lnp08_uniform_e60.log 2>&1 &
  echo "PID=$!"
  - ②再開:上の --ckpt-epochs ... 込みの同じコマンドを再実行。
  - ③fork:--ckpt-epochs を外し --change-epoch <N> と --init-from <...ckpt_epochNNN.pt> を指定。

  要点:① と ② はコマンドが同一(再開は自動)、③ は CHANGE_EPOCH + INIT_FROM を変えるだけです。
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
from posthoc_snapshot import save_snapshot, snapshot_dir
from timestep_sampler import (
    sample_linear_decreasing,
    sample_linear_increasing,
    sample_logit_normal,
    sample_mode,
    sample_uniform,
)
from training.data_transform import get_train_transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (Phase 1 specific)
# ---------------------------------------------------------------------------
DATA_PATH = "./data/image_generation"
WANDB_PROJECT = "phase1-twophase"
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Separate tree from phase1_static / phase1_posthoc so those runs are never touched.
CKPT_BASE = _REPO_ROOT / "checkpoints" / "phase1_twophase"
MODEL_KEY = "cifar10"

# Save a weights-only snapshot every N epochs for post-hoc EMA reconstruction.
# Reconstructable EMA windows must be >> this interval (1 epoch = len(dataloader) steps).
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Phase 1 Two-phase: timestep sampler switching at change_epoch")
    p.add_argument("--phase1-config", required=True,
                   help="YAML config for the FIRST phase (epochs 1..change_epoch).")
    p.add_argument("--phase2-config", required=True,
                   help="YAML config for the SECOND phase (epochs change_epoch+1..end).")
    p.add_argument("--change-epoch", type=int, required=True,
                   help="Last epoch (1-indexed, inclusive) trained with the phase-1 sampler. "
                        "Phase 2 starts at change_epoch+1.")
    p.add_argument("--phase1-alpha", type=float, default=0.0,
                   help="Uniform mixture weight for phase 1. "
                        "p(t) = a*Uniform + (1-a)*p_phase1(t). 0 = no mixing.")
    p.add_argument("--phase2-alpha", type=float, default=0.0,
                   help="Uniform mixture weight for phase 2. 0 = no mixing.")
    p.add_argument("--sampler-id", type=str, default=None,
                   help="Override the auto-generated run id / checkpoint dir name.")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, eval every 2, 2000 FID samples (smoke-test)")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="Fork: initialise weights+optimizer+scheduler+RNG from this "
                        "external checkpoint (e.g. another run's ckpt_epoch040.pt), "
                        "continuing the epoch count from it but starting a FRESH wandb "
                        "run and writing to THIS run's own dir. Ignored once this run "
                        "has its own latest.pt (so restarts resume normally).")
    p.add_argument("--keep-epochs", type=str, default="60,120,180",
                   help="Comma-separated epochs whose periodic checkpoints are never "
                        "pruned. Set this on the SOURCE run so a fork point survives, "
                        "e.g. --keep-epochs 40,60,120,180.")
    p.add_argument("--ckpt-epochs", type=str, default=None,
                   help="Comma-separated EXACT epochs to save a full resume checkpoint at, "
                        "kept forever and nothing else (e.g. 30,40,50,60,70 for planned "
                        "fork points). Overrides the --ckpt-every / --keep-* cadence. "
                        "latest.pt is still written every epoch for crash-resume.")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="Cadence (epochs) for full resume checkpoints, INDEPENDENT of "
                        "FID --eval-every. Default = --eval-every. Use a small value "
                        "(e.g. 10) to leave fork points at fine granularity. "
                        "Ignored if --ckpt-epochs is set.")
    p.add_argument("--keep-recent-n", type=int, default=KEEP_RECENT_N,
                   help=f"Keep the most recent N periodic checkpoints (default {KEEP_RECENT_N}).")
    p.add_argument("--keep-all", action="store_true",
                   help="Never prune periodic checkpoints, so EVERY --ckpt-every epoch "
                        "stays forkable later. Costs ~1.8GB per checkpoint.")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="Save a weights-only snapshot every N epochs "
                        f"(default {SNAPSHOT_EVERY}; in --dry-run forced to 1).")
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
    elif stype == "linear_decreasing":
        floor = float(cfg["floor"])
        return lambda bs, dev: sample_linear_decreasing(bs, floor=floor, device=dev)
    elif stype == "linear_increasing":
        floor = float(cfg["floor"])
        return lambda bs, dev: sample_linear_increasing(bs, floor=floor, device=dev)
    else:
        raise ValueError(f"Unknown sampler type: {stype!r}")


# ---------------------------------------------------------------------------
# Uniform mixture wrapper
# ---------------------------------------------------------------------------
def wrap_with_uniform_mix(sampler_fn, alpha: float):
    """p_new(t) = α · Uniform(0,1) + (1-α) · p_orig(t)"""
    if alpha == 0.0:
        return sampler_fn

    def _mixed(bs, dev):
        mask = torch.bernoulli(torch.full((bs,), alpha, device=dev)).bool()
        return torch.where(mask, torch.rand(bs, device=dev), sampler_fn(bs, dev))

    return _mixed


# ---------------------------------------------------------------------------
# Phase config loader
# ---------------------------------------------------------------------------
def load_phase(config_path: str, alpha: float):
    """Load a single-distribution YAML config and return its sampler + metadata.

    Returns (sample_fn, sampler_cfg, tag) where tag is a short id encoding the
    distribution and (if non-zero) the uniform-mixture alpha, used to build the
    combined run id / checkpoint dir name.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sampler_cfg = cfg["sampler"]
    base_id = cfg.get("sampler_id") or Path(config_path).stem
    fn = build_sampler(sampler_cfg)
    fn = wrap_with_uniform_mix(fn, alpha)
    tag = base_id if alpha == 0.0 else f"{base_id}_a{alpha:g}"
    return fn, sampler_cfg, tag


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

    # Build the two phase samplers from their YAML configs.
    sample_t_p1, sampler_cfg_p1, tag_p1 = load_phase(args.phase1_config, args.phase1_alpha)
    sample_t_p2, sampler_cfg_p2, tag_p2 = load_phase(args.phase2_config, args.phase2_alpha)
    change_epoch = args.change_epoch

    sampler_id = args.sampler_id or f"{tag_p1}__{tag_p2}__e{change_epoch}"
    ckpt_dir = CKPT_BASE / sampler_id

    logger.info(
        f"Phase 1 Two-phase  id={sampler_id}\n"
        f"  phase1 (epoch 1..{change_epoch}): {sampler_cfg_p1}  alpha={args.phase1_alpha}\n"
        f"  phase2 (epoch {change_epoch+1}..): {sampler_cfg_p2}  alpha={args.phase2_alpha}\n"
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
    ckpt_epochs = ({int(x) for x in args.ckpt_epochs.split(",") if x.strip()}
                   if args.ckpt_epochs else None)
    if ckpt_epochs is not None:
        logger.info(f"Full resume checkpoints saved ONLY at epochs {sorted(ckpt_epochs)} "
                    "(kept forever); latest.pt still written every epoch.")

    if not (1 <= change_epoch < args.max_epochs):
        logger.warning(
            f"change_epoch={change_epoch} outside [1, max_epochs-1]={args.max_epochs-1}: "
            "one of the two phases will never run."
        )

    snap_dir = snapshot_dir(ckpt_dir)

    logger.info(
        f"max_epochs={args.max_epochs}  eval_every={args.eval_every}"
        f"  change_epoch={change_epoch}  dry_run={args.dry_run}"
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

    # Resume / fork.
    # --init-from forks from an external checkpoint (full state, but a FRESH wandb
    # run) and is honoured only on first launch: once this run has its own
    # latest.pt, restarts resume from it normally.
    own_latest = (ckpt_dir / "latest.pt").exists()
    do_fork = (args.init_from and not args.no_resume
               and not args.resume_from and not own_latest)
    if do_fork:
        if not Path(args.init_from).exists():
            raise FileNotFoundError(f"--init-from not found: {args.init_from}")
        args.resume_from = args.init_from          # reuse the loader for full state
        start_epoch, global_step, _ = load_checkpoint(
            ckpt_dir, args, ema_model, optimizer, scheduler
        )
        args.resume_from = None
        wandb_run_id = None                        # fresh wandb run; don't inherit source
        logger.info(
            f"Forked from {args.init_from}: continuing at epoch {start_epoch} "
            f"(phase selected by this run's change_epoch={change_epoch})"
        )
        if start_epoch > change_epoch:
            logger.warning(
                f"fork point (epoch {start_epoch-1}) is past change_epoch={change_epoch}: "
                "phase 1 never runs in this fork — check you forked at epoch <= change_epoch."
            )
    else:
        start_epoch, global_step, wandb_run_id = load_checkpoint(
            ckpt_dir, args, ema_model, optimizer, scheduler
        )

    # wandb
    wandb_cfg_p1 = {k: v for k, v in sampler_cfg_p1.items() if v is not None}
    wandb_cfg_p2 = {k: v for k, v in sampler_cfg_p2.items() if v is not None}
    if wandb_run_id is not None:
        run = wandb.init(project=WANDB_PROJECT, id=wandb_run_id, resume="must")
        logger.info(f"Resumed wandb run: {wandb_run_id}")
    else:
        run = wandb.init(
            project=WANDB_PROJECT,
            config={
                "sampler_id": sampler_id,
                "phase1_sampler": wandb_cfg_p1,
                "phase1_alpha": args.phase1_alpha,
                "phase2_sampler": wandb_cfg_p2,
                "phase2_alpha": args.phase2_alpha,
                "change_epoch": change_epoch,
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
        epoch_1indexed = epoch + 1

        # Phase selection: epochs 1..change_epoch use phase 1, the rest phase 2.
        if epoch_1indexed <= change_epoch:
            sample_t = sample_t_p1
            phase = 1
        else:
            sample_t = sample_t_p2
            phase = 2
        if epoch_1indexed == 1 or epoch_1indexed == change_epoch + 1:
            logger.info(f"Epoch {epoch_1indexed}: using phase-{phase} timestep sampler")

        epoch_start = time.time()
        avg_loss, global_step = train_one_epoch(
            ema_model, dataloader, optimizer, scheduler,
            device, epoch, global_step, run, sample_t,
        )
        epoch_sec = time.time() - epoch_start

        if run is not None:
            run.log({
                "train/epoch_loss": avg_loss,
                "train/phase": phase,
                "epoch": epoch_1indexed,
                "global_step": global_step,
                "epoch_time_sec": epoch_sec,
            })

        if ckpt_epochs is not None:
            # Explicit mode: write latest.pt every epoch, but a kept-forever named
            # checkpoint ONLY at the requested epochs. eval_every=1 forces the named
            # save at those epochs; the huge keep_recent_n + keep set means nothing
            # else is ever written, so exactly {ckpt_epochs} survive.
            save_checkpoint(
                ckpt_dir, ema_model, optimizer, scheduler,
                epoch, global_step, wandb_run_id,
                1 if epoch_1indexed in ckpt_epochs else 10**9,
                ckpt_epochs, 10**9,
            )
        else:
            save_checkpoint(
                ckpt_dir, ema_model, optimizer, scheduler,
                epoch, global_step, wandb_run_id,
                ckpt_every, keep_epochs, keep_recent_n,
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
