#!/usr/bin/env bash
# Background launcher: uniform training WITH post-hoc EMA snapshots.
#
# Trains train_phase1_posthoc.py on the uniform timestep sampler. It dumps fp16
# weights-only snapshots every SNAPSHOT_EVERY epochs into
# checkpoints/phase1_posthoc/uniform/snapshots/, so afterwards any EMA decay can
# be reconstructed offline with reconstruct_ema.py (no re-training):
#
#   python reconstruct_ema.py --sampler uniform --decay 0.999   --fid
#   python reconstruct_ema.py --sampler uniform --decay 0.9999  --fid
#   python reconstruct_ema.py --sampler uniform --decay 0.99995 --fid
#
# Portable: all paths derive from this script's location (no server name
# hardcoded), so `git pull` + run works on any server. Logs go to
# examples/image/logs/. Resumable: re-running after an interruption picks up
# from checkpoints/.../latest.pt automatically (snapshots are idempotent).
#
# Usage (from anywhere):
#   bash examples/image/run_phase1_posthoc_uniform.sh
#   SNAPSHOT_EVERY=20 bash examples/image/run_phase1_posthoc_uniform.sh  # fewer snapshots (~half disk)
#   bash examples/image/run_phase1_posthoc_uniform.sh --device cuda:1     # extra args pass through
#   tail -f <printed log path>
set -euo pipefail

# --- locate self / repo root (portable; no srv11/srv21 in paths) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../examples/image
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # repo root

SAMPLER_ID="uniform"
CONFIG="configs/phase1_static/uniform.yaml"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"  # snapshot interval in epochs (default 10)

LOG_DIR="$SCRIPT_DIR/logs/phase1_posthoc"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SAMPLER_ID}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"   # online for production; override if needed

cd "$SCRIPT_DIR"

# setsid + nohup: fully detached, survives terminal close. Production hypers
# (180 epochs, eval every 20, 50k FID) are the script defaults. --alpha 0.0
# keeps the sampler pure uniform (uniform-mix on uniform is a no-op anyway).
setsid nohup python -u train_phase1_posthoc.py \
    --config "$CONFIG" \
    --alpha 0.0 \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --device cuda \
    "$@" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "Launched post-hoc uniform run"
echo "  PID            : $PID"
echo "  sampler_id     : $SAMPLER_ID"
echo "  config         : $CONFIG"
echo "  snapshot_every : $SNAPSHOT_EVERY epoch(s)"
echo "  WANDB_MODE     : $WANDB_MODE"
echo "  log            : $LOG"
echo "  ckpt dir       : $REPO_ROOT/checkpoints/phase1_posthoc/$SAMPLER_ID"
echo "  snapshots      : $REPO_ROOT/checkpoints/phase1_posthoc/$SAMPLER_ID/snapshots"
echo
echo "Follow:  tail -f \"$LOG\""
echo "Stop:    kill $PID"
echo
echo "After training, reconstruct any EMA decay + FID (no re-training):"
echo "  python reconstruct_ema.py --sampler $SAMPLER_ID --decay 0.999   --fid"
echo "  python reconstruct_ema.py --sampler $SAMPLER_ID --decay 0.9999  --fid"
