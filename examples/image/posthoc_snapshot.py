"""
Post-hoc EMA snapshot I/O.

Stores *weights-only*, fp16 snapshots of the raw model at regular intervals so
that any EMA decay / averaging profile can be reconstructed offline after
training (see reconstruct_ema.py). Kept completely separate from the resume
checkpoints written by phase1_utils.save_checkpoint:

  - resume checkpoints  : full state (optimizer/scheduler/RNG), pruned, latest.pt
  - post-hoc snapshots  : raw weights only, fp16, never pruned, this module

All paths are derived from the snapshot directory passed in (which the caller
builds repo-relative), so nothing here contains a server-specific absolute path.
"""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def snapshot_dir(ckpt_dir: Path) -> Path:
    """snapshots/ subdir next to the resume checkpoints for this sampler."""
    return Path(ckpt_dir) / "snapshots"


def _manifest_path(snap_dir: Path) -> Path:
    return Path(snap_dir) / MANIFEST_NAME


def load_manifest(snap_dir: Path) -> dict:
    p = _manifest_path(snap_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"snapshots": []}


def _atomic_write_json(obj: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.rename(path)


def _weights_only_fp16(model) -> dict:
    """Raw model state_dict; float tensors -> fp16, others (ints) untouched."""
    sd = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        sd[k] = v.to(torch.float16) if v.is_floating_point() else v
    return sd


def save_snapshot(snap_dir: Path, model, global_step: int, num_updates: int,
                  epoch_1indexed: int, sampler_id: str, model_key: str) -> None:
    """Save one fp16 weights-only snapshot and (idempotently) record it in the manifest.

    Safe across resume: if a snapshot for ``global_step`` is already recorded,
    this is a no-op so the manifest never gets duplicate entries.
    """
    snap_dir = Path(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(snap_dir)
    if any(s["step"] == global_step for s in manifest["snapshots"]):
        logger.info(f"Snapshot for step {global_step} already exists – skipping")
        return

    fname = f"snap_step{global_step:08d}.pt"
    state = {
        "weights": _weights_only_fp16(model),
        "step": global_step,
        "num_updates": num_updates,
        "epoch": epoch_1indexed,
    }
    tmp = snap_dir / (fname + ".tmp")
    torch.save(state, tmp)
    tmp.rename(snap_dir / fname)

    manifest.setdefault("model", model_key)
    manifest.setdefault("sampler_id", sampler_id)
    manifest["snapshots"].append({
        "step": global_step,
        "num_updates": num_updates,
        "epoch": epoch_1indexed,
        "file": fname,  # relative to snap_dir; never an absolute path
    })
    manifest["snapshots"].sort(key=lambda s: s["step"])
    _atomic_write_json(manifest, _manifest_path(snap_dir))
    logger.info(f"Saved post-hoc snapshot: {fname}  (num_updates={num_updates})")
