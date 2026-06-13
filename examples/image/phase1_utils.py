"""
Shared utilities for Phase 0 / Phase 1 training.
Checkpoint I/O, FID evaluation, per-timestep loss.

All functions take ckpt_dir and other formerly-global constants as parameters
so they can be reused across training scripts without modification.
"""

import gc
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ODE generation wrapper
# ---------------------------------------------------------------------------
class UnconditionalWrapper(ModelWrapper):
    """Adapter: ODESolver (scalar t) → UNetModel (batch t), fp16 AMP inference."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_batch = torch.full(
            (x.shape[0],), float(t), device=x.device, dtype=torch.float32
        )
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.model(x, t_batch, extra={})
        return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _make_state(model, optimizer, scheduler, epoch, global_step, wandb_run_id):
    assert model.training
    return {
        "ema_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "wandb_run_id": wandb_run_id,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
    }


def _epoch_from_name(p: Path) -> int:
    return int(p.stem.replace("ckpt_epoch", ""))


def _prune_checkpoints(ckpt_dir: Path, current_epoch_1indexed: int,
                       keep_epochs: set, keep_recent_n: int):
    all_ckpts = sorted(ckpt_dir.glob("ckpt_epoch*.pt"), key=_epoch_from_name)
    keep = {p for p in all_ckpts if _epoch_from_name(p) in keep_epochs}
    keep.update(all_ckpts[-keep_recent_n:])
    for p in all_ckpts:
        if p not in keep:
            p.unlink()
            logger.info(f"Pruned old checkpoint: {p.name}")


def save_checkpoint(ckpt_dir: Path, model, optimizer, scheduler,
                    epoch, global_step, wandb_run_id, eval_every,
                    keep_epochs: set, keep_recent_n: int = 3):
    """Save latest.pt every epoch; save + prune periodic checkpoints every eval_every."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = _make_state(model, optimizer, scheduler, epoch, global_step, wandb_run_id)
    atomic_save(state, ckpt_dir / "latest.pt")

    epoch_1indexed = epoch + 1
    if epoch_1indexed % eval_every == 0:
        name = f"ckpt_epoch{epoch_1indexed:03d}.pt"
        atomic_save(state, ckpt_dir / name)
        logger.info(f"Saved periodic checkpoint: {name}")
        _prune_checkpoints(ckpt_dir, epoch_1indexed, keep_epochs, keep_recent_n)


def load_checkpoint(ckpt_dir: Path, args, model, optimizer, scheduler):
    """Returns (start_epoch, global_step, wandb_run_id)."""
    if args.no_resume:
        logger.info("--no-resume: starting from scratch")
        return 0, 0, None

    ckpt_path = None
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from not found: {ckpt_path}")
    elif (ckpt_dir / "latest.pt").exists():
        ckpt_path = ckpt_dir / "latest.pt"

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
# FID history
# ---------------------------------------------------------------------------
def load_fid_history(ckpt_dir: Path):
    path = ckpt_dir / "fid_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_fid_history(ckpt_dir: Path, history):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "fid_history.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Per-timestep loss (10 bins: bin_00..bin_09)
# ---------------------------------------------------------------------------
def compute_per_timestep_loss(raw_unet, dataloader, device, n_batches=100):
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
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i]


def _generate_images(model_module, device, n_samples, batch_size, nfe, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    wrapper = UnconditionalWrapper(model=model_module)
    solver = ODESolver(velocity_model=wrapper)
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
                 fid_history, n_samples, fid_batch, fid_nfe, fid_seed):
    """Evaluate FID with EMA and raw weights. Skips if fid_ema already recorded for this epoch."""
    for entry in fid_history:
        if entry["epoch"] == epoch_1indexed and "fid_ema" in entry:
            logger.info(f"Epoch {epoch_1indexed} already has FID in history – skipping")
            return None

    logger.info(f"=== FID evaluation at epoch {epoch_1indexed} ===")
    assert ema_model.training

    # Raw weights
    ema_model.model.eval()
    t0 = time.time()
    logger.info("Generating with RAW weights ...")
    raw_imgs = _generate_images(ema_model.model, device, n_samples, fid_batch, fid_nfe, fid_seed)
    fid_raw = _compute_fid(raw_imgs, data_path, device)
    del raw_imgs
    gc.collect()
    torch.cuda.empty_cache()
    ema_model.model.train()
    logger.info(f"Raw FID: {fid_raw:.3f}  ({time.time()-t0:.0f}s)")

    # EMA weights
    ema_model.train(False)
    t0 = time.time()
    logger.info("Generating with EMA weights ...")
    ema_imgs = _generate_images(ema_model.model, device, n_samples, fid_batch, fid_nfe, fid_seed)
    fid_ema = _compute_fid(ema_imgs, data_path, device)
    del ema_imgs
    gc.collect()
    torch.cuda.empty_cache()
    ema_model.train(True)
    logger.info(f"EMA FID:  {fid_ema:.3f}  ({time.time()-t0:.0f}s)")

    return {"fid_ema": fid_ema, "fid_raw": fid_raw}
