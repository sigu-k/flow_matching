#!/usr/bin/env python3
"""
Phase 0: CIFAR-10 Flow Matching – Uniform Timestep Sampling
300-epoch run to identify FID saturation epoch.

Run from: flow_matching/examples/image/
  source ~/work/srv11/setup_env.sh
  python train_phase0_uniform.py [--dry-run] [--no-resume] [--resume-from PATH]
"""

import argparse
import gc
import json
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

from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.ema import EMA
from models.model_configs import MODEL_CONFIGS
from models.unet import UNetModel
from training.data_transform import get_train_transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – match Phase 0 spec exactly
# ---------------------------------------------------------------------------
CKPT_DIR = Path(os.path.expanduser("~/work/srv11/checkpoints/phase0_v2"))
DATA_PATH = "./data/image_generation"
WANDB_PROJECT = "phase0_v2"

LR = 1e-4
WARMUP_STEPS = 10_000
BATCH_SIZE = 128
EPOCHS = 300
EMA_DECAY = 0.99995

EVAL_EVERY = 20         # epochs between FID evaluations
FID_SAMPLES = 50_000
FID_NFE = 50            # Euler steps (time_grid has NFE+1 points)
FID_BATCH = 250
FID_SEED = 0

KEEP_RECENT_N = 3       # how many recent periodic checkpoints to keep
KEEP_EPOCHS = {100, 200, 300}  # long-term snapshots never pruned


# ---------------------------------------------------------------------------
# Model wrapper for unconditional fp32 ODE generation
# ---------------------------------------------------------------------------
class _UnconditionalWrapper(ModelWrapper):
    """
    Adapter between ODESolver (scalar t) and UNetModel (batch t).
    Unconditional: always passes extra={}.  No autocast – stays fp32.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # torchdiffeq passes t as a 0-dim scalar tensor; UNet needs batch-sized t
        t_batch = torch.full(
            (x.shape[0],), float(t), device=x.device, dtype=torch.float32
        )
        with torch.no_grad(), torch.cuda.amp.autocast():
            # AMP for inference only: keeps fp32 training but uses fp16 for UNet
            # forward kernels, matching existing eval_loop.py behaviour.
            out = self.model(x, t_batch, extra={})
        return out.to(torch.float32)  # Euler integration stays in fp32


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Phase 0: Uniform timestep FM training")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 2 epochs with eval every 2 (smoke-test)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override epoch count (default: 300, or 2 with --dry-run)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoints and start fresh")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from a specific checkpoint path")
    p.add_argument("--eval-every", type=int, default=None,
                   help="FID eval frequency in epochs (default: 20, or 2 with --dry-run)")
    p.add_argument("--fid-samples", type=int, default=None,
                   help="Number of samples for FID (default: 2000 for dry-run, 50000 for production)")
    p.add_argument("--data-path", type=str, default=DATA_PATH)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _atomic_save(obj, path: Path):
    """Write to .tmp then rename – prevents corrupt checkpoint on crash."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _make_state(model, optimizer, scheduler, epoch, global_step, wandb_run_id):
    assert model.training, "Call save_checkpoint in train mode"
    return {
        # Model weights
        "ema_state": model.state_dict(),   # raw weights + shadow params + num_updates
        # Optimizer & schedule
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        # Progress
        "epoch": epoch,
        "global_step": global_step,
        # wandb run continuity
        "wandb_run_id": wandb_run_id,
        # Full RNG state for exact resume
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    wandb_run_id, eval_every):
    """
    Always saves latest.pt (atomic overwrite).
    Every eval_every epochs, also saves ckpt_epoch{NNN:03d}.pt and prunes old ones.
    epoch is 0-indexed; checkpoint filenames use 1-indexed epoch numbers.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = _make_state(model, optimizer, scheduler, epoch, global_step, wandb_run_id)

    # latest.pt – saved every epoch
    _atomic_save(state, CKPT_DIR / "latest.pt")

    # periodic checkpoint
    epoch_1indexed = epoch + 1
    if epoch_1indexed % eval_every == 0:
        name = f"ckpt_epoch{epoch_1indexed:03d}.pt"
        _atomic_save(state, CKPT_DIR / name)
        logger.info(f"Saved periodic checkpoint: {name}")
        _prune_checkpoints(epoch_1indexed)


def _epoch_from_name(p: Path) -> int:
    return int(p.stem.replace("ckpt_epoch", ""))


def _prune_checkpoints(current_epoch_1indexed: int):
    """Keeps KEEP_RECENT_N newest periodic checkpoints + KEEP_EPOCHS snapshots."""
    all_ckpts = sorted(CKPT_DIR.glob("ckpt_epoch*.pt"),
                       key=_epoch_from_name)
    keep = set()
    for p in all_ckpts:
        if _epoch_from_name(p) in KEEP_EPOCHS:
            keep.add(p)
    recent = all_ckpts[-KEEP_RECENT_N:]
    keep.update(recent)
    for p in all_ckpts:
        if p not in keep:
            p.unlink()
            logger.info(f"Pruned old checkpoint: {p.name}")


def load_checkpoint(args, model, optimizer, scheduler):
    """
    Returns (start_epoch, global_step, wandb_run_id).
    Restores all state including full RNG.
    """
    if args.no_resume:
        logger.info("--no-resume: starting from scratch")
        return 0, 0, None

    ckpt_path = None
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from not found: {ckpt_path}")
    elif (CKPT_DIR / "latest.pt").exists():
        ckpt_path = CKPT_DIR / "latest.pt"

    if ckpt_path is None:
        logger.info("No checkpoint found – starting from scratch")
        return 0, 0, None

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["ema_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch_cpu"])
    if ckpt["rng_torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_torch_cuda"])

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    wandb_run_id = ckpt.get("wandb_run_id")

    logger.info(
        f"Resuming from epoch {start_epoch}, "
        f"global_step {global_step}, wandb run_id={wandb_run_id}"
    )
    return start_epoch, global_step, wandb_run_id


# ---------------------------------------------------------------------------
# FID history helpers
# ---------------------------------------------------------------------------
FID_HISTORY_PATH = CKPT_DIR / "fid_history.json"


def load_fid_history():
    if FID_HISTORY_PATH.exists():
        with open(FID_HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_fid_history(history):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FID_HISTORY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.rename(FID_HISTORY_PATH)


# ---------------------------------------------------------------------------
# Per-timestep loss (10 bins)
# ---------------------------------------------------------------------------
def compute_per_timestep_loss(raw_unet, dataloader, device, n_batches=100):
    """
    Runs raw_unet in eval mode on n_batches samples.
    Returns dict {t_0_1: float, t_1_2: float, ..., t_9_10: float}.
    """
    raw_unet.eval()
    path = CondOTProbPath()
    bin_losses = [[] for _ in range(10)]

    with torch.no_grad():
        for i, (samples, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            samples = samples.to(device) * 2.0 - 1.0
            noise = torch.randn_like(samples)
            t = torch.rand(samples.shape[0], device=device)
            ps = path.sample(t=t, x_0=noise, x_1=samples)
            pred = raw_unet(ps.x_t, t, extra={})
            loss_per = (pred - ps.dx_t).pow(2).mean(dim=(1, 2, 3))
            for j in range(samples.shape[0]):
                b = min(int(t[j].item() * 10), 9)
                bin_losses[b].append(loss_per[j].item())

    raw_unet.train()
    return {
        f"t_{i}_{i+1}": float(np.mean(v)) if v else float("nan")
        for i, v in enumerate(bin_losses)
    }


# ---------------------------------------------------------------------------
# FID evaluation
# ---------------------------------------------------------------------------
class _ImageDataset(torch.utils.data.Dataset):
    """Wraps uint8 CHW tensor for torch-fidelity."""
    def __init__(self, t):
        self.t = t
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        return self.t[i]


def _generate_images(model_module, device, n_samples, batch_size, nfe, seed):
    """
    Generate n_samples CIFAR-10 images with 50-NFE Euler ODE (fp32, seed fixed).
    model_module: a raw nn.Module (UNetModel in eval mode, or EMA with EMA weights active).
    Returns: uint8 CHW tensor, shape (n_samples, 3, 32, 32).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    wrapper = _UnconditionalWrapper(model=model_module)
    solver = ODESolver(velocity_model=wrapper)
    # nfe+1 grid points → nfe Euler steps
    time_grid = torch.linspace(0.0, 1.0, nfe + 1, device=device)

    all_images = []
    n_done = 0
    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        x0 = torch.randn(bs, 3, 32, 32, dtype=torch.float32, device=device)
        x1 = solver.sample(
            x_init=x0,
            time_grid=time_grid,
            method="euler",
            step_size=None,
            return_intermediates=False,
        )
        # [-1, 1] → [0, 255] uint8
        x1 = torch.clamp(x1 * 0.5 + 0.5, 0.0, 1.0)
        x1 = (x1 * 255).to(torch.uint8).cpu()
        all_images.append(x1)
        n_done += bs
        if n_done % 5000 < batch_size:
            logger.info(f"  Generated {n_done}/{n_samples}")

    return torch.cat(all_images, dim=0)[:n_samples]


def _compute_fid(images_uint8, data_path, device):
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1="cifar10-train",
        input2=_ImageDataset(images_uint8),
        fid=True,
        datasets_root=data_path,
        cuda=str(device).startswith("cuda"),
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def evaluate_fid(ema_model, device, data_path, epoch_1indexed, global_step,
                 fid_history, n_samples=FID_SAMPLES):
    """
    Evaluate FID with both EMA and raw weights.
    Skips (returns None) if epoch already in history.
    epoch_1indexed: 1-based epoch number for history dedup and logging.
    """
    for entry in fid_history:
        if entry["epoch"] == epoch_1indexed:
            logger.info(f"Epoch {epoch_1indexed} already in FID history – skipping")
            return None

    logger.info(f"=== FID evaluation at epoch {epoch_1indexed} ===")

    # --- Raw weights FID (model in train mode → raw weights in model.model) ---
    assert ema_model.training, "evaluate_fid expects EMA model in train mode"
    ema_model.model.eval()
    t0 = time.time()
    logger.info("Generating with RAW weights ...")
    raw_imgs = _generate_images(ema_model.model, device,
                                n_samples, FID_BATCH, FID_NFE, FID_SEED)
    fid_raw = _compute_fid(raw_imgs, data_path, device)
    del raw_imgs
    gc.collect()
    torch.cuda.empty_cache()
    ema_model.model.train()
    logger.info(f"Raw FID: {fid_raw:.3f}  ({time.time()-t0:.0f}s)")

    # --- EMA weights FID ---
    # train(False) backs up raw weights and swaps shadow → model
    ema_model.train(False)
    t0 = time.time()
    logger.info("Generating with EMA weights ...")
    ema_imgs = _generate_images(ema_model.model, device,
                                n_samples, FID_BATCH, FID_NFE, FID_SEED)
    fid_ema = _compute_fid(ema_imgs, data_path, device)
    del ema_imgs
    gc.collect()
    torch.cuda.empty_cache()
    ema_model.train(True)   # restore raw weights
    logger.info(f"EMA FID:  {fid_ema:.3f}  ({time.time()-t0:.0f}s)")

    return {"fid_ema": fid_ema, "fid_raw": fid_raw}


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(ema_model, dataloader, optimizer, scheduler,
                    device, epoch, global_step, run):
    """
    fp32, unconditional, uniform t ~ U(0,1).
    Returns (avg_loss, new_global_step).
    """
    ema_model.train(True)
    path = CondOTProbPath()
    total_loss = 0.0
    n_batches = 0

    for samples, _ in dataloader:
        # Rescale [0,1] → [-1,1]
        samples = samples.to(device) * 2.0 - 1.0
        noise = torch.randn_like(samples)

        # Uniform timestep sampling
        t = torch.rand(samples.shape[0], device=device)

        # Straight-line interpolation: x_t = (1-t)*noise + t*data
        ps = path.sample(t=t, x_0=noise, x_1=samples)

        # CFM loss (fp32, no autocast)
        pred = ema_model(ps.x_t, t, extra={})
        loss = (pred - ps.dx_t).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()       # per-step LR update
        ema_model.update_ema() # update shadow params

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

    # Apply --dry-run defaults
    if args.dry_run:
        args.max_epochs = args.max_epochs or 2
        args.eval_every = args.eval_every or 2
        args.fid_samples = args.fid_samples or 2_000   # fast smoke-test
    else:
        args.max_epochs = args.max_epochs or EPOCHS
        args.eval_every = args.eval_every or EVAL_EVERY
        args.fid_samples = args.fid_samples or FID_SAMPLES

    logger.info(
        f"Phase 0 – max_epochs={args.max_epochs}  eval_every={args.eval_every}"
        f"  dry_run={args.dry_run}"
    )

    # ── Reproducibility ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info("Building UNet (111M params) + EMA …")
    unet = UNetModel(**MODEL_CONFIGS["cifar10"])
    ema_model = EMA(model=unet, decay=EMA_DECAY)
    ema_model.to(device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"UNet trainable params: {n_params/1e6:.1f}M")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)

    # ── LR schedule: linear warmup over WARMUP_STEPS, then constant ─────────
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── Dataset ─────────────────────────────────────────────────────────────
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

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch, global_step, wandb_run_id = load_checkpoint(
        args, ema_model, optimizer, scheduler
    )

    # ── wandb ───────────────────────────────────────────────────────────────
    if wandb_run_id is not None:
        run = wandb.init(
            project=WANDB_PROJECT,
            id=wandb_run_id,
            resume="must",
        )
        logger.info(f"Resumed wandb run: {wandb_run_id}")
    else:
        run = wandb.init(
            project=WANDB_PROJECT,
            config={
                "max_epochs": args.max_epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "warmup_steps": WARMUP_STEPS,
                "ema_decay": EMA_DECAY,
                "timestep_dist": "uniform",
                "fid_nfe": FID_NFE,
                "fid_samples": FID_SAMPLES,
                "fid_batch": FID_BATCH,
                "precision": "fp32",
                "optimizer": "Adam",
                "optimizer_betas": [0.9, 0.999],
                "weight_decay": 0,
                "unet_model_channels": 128,
                "unet_channel_mult": [2, 2, 2],
                "unet_num_res_blocks": 4,
            },
        )
        wandb_run_id = run.id
        logger.info(f"New wandb run: {wandb_run_id}")

    # ── FID history ─────────────────────────────────────────────────────────
    fid_history = load_fid_history()

    # ── Training loop ───────────────────────────────────────────────────────
    logger.info(f"Training epochs {start_epoch+1} → {args.max_epochs} …")
    total_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        avg_loss, global_step = train_one_epoch(
            ema_model, dataloader, optimizer, scheduler,
            device, epoch, global_step, run
        )
        epoch_sec = time.time() - epoch_start

        if run is not None:
            run.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_time_sec": epoch_sec,
            })

        # Save latest.pt (every epoch) + periodic checkpoint (every eval_every)
        save_checkpoint(
            ema_model, optimizer, scheduler,
            epoch, global_step, wandb_run_id,
            args.eval_every,
        )

        # FID evaluation + per-timestep loss every eval_every epochs
        epoch_1indexed = epoch + 1
        is_eval_epoch = (
            epoch_1indexed % args.eval_every == 0
            or epoch_1indexed == args.max_epochs
        )

        if is_eval_epoch:
            # Per-timestep loss
            logger.info(f"Computing per-timestep loss at epoch {epoch_1indexed} …")
            per_ts = compute_per_timestep_loss(unet, dataloader, device)
            if run is not None:
                run.log(
                    {f"per_t_loss/{k}": v for k, v in per_ts.items()}
                    | {"epoch": epoch_1indexed, "global_step": global_step}
                )
            logger.info(f"Per-timestep loss: {per_ts}")

            # FID
            fid_result = evaluate_fid(
                ema_model, device, args.data_path,
                epoch_1indexed, global_step, fid_history,
                n_samples=args.fid_samples,
            )

            if fid_result is not None:
                entry = {
                    "epoch": epoch_1indexed,
                    "global_step": global_step,
                    "fid_ema": fid_result["fid_ema"],
                    "fid_raw": fid_result["fid_raw"],
                    "per_timestep_loss": per_ts,
                }
                fid_history.append(entry)
                save_fid_history(fid_history)
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
