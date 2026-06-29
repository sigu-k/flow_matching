#!/usr/bin/env bash
# Background launcher: Phase 1 static training at batch size 128.
#
# Trains train_phase1_static.py on a configurable timestep sampler with
# --batch-size 128 (default in the script is 64). On the A100 80GB MIG 3g.40gb
# slice (~42 GB), batch 128 peaks at ~21.6 GB reserved — about half the slice,
# leaving comfortable headroom. Batch 256 would peak near the 42 GB ceiling and
# is unsafe, so 128 is the recommended step up from 64.
#
# NOTE: effective batch doubles vs the 64 baseline; adjust LR offline if you
# want to match the 64-run training dynamics.
#
# Portable: all paths derive from this script's location (no server name
# hardcoded), so `git pull` + run works on any server. Logs go to
# examples/image/logs/. Resumable: re-running after an interruption picks up
# from checkpoints/.../latest.pt automatically.
#
# Usage (from anywhere):
#   bash examples/image/run_phase1_static_bs128.sh
#   SAMPLER_ID=ln_mu+0.8 CONFIG=configs/phase1_static/ln_mu+0.8.yaml \
#       bash examples/image/run_phase1_static_bs128.sh
#   BATCH_SIZE=192 bash examples/image/run_phase1_static_bs128.sh   # push further
#   bash examples/image/run_phase1_static_bs128.sh --device cuda:1  # extra args pass through
#   tail -f <printed log path>
set -euo pipefail

# --- locate self / repo root (portable; no srv11/srv21 in paths) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../examples/image
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # repo root

SAMPLER_ID="${SAMPLER_ID:-uniform}"
CONFIG="${CONFIG:-configs/phase1_static/uniform.yaml}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ALPHA="${ALPHA:-0.0}"

LOG_DIR="$SCRIPT_DIR/logs/phase1_static"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SAMPLER_ID}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"   # online for production; override if needed

cd "$SCRIPT_DIR"

# setsid + nohup: fully detached, survives terminal close. Production hypers
# (180 epochs, eval every 20, 50k FID) are the script defaults.
setsid nohup python -u train_phase1_static.py \
    --config "$CONFIG" \
    --batch-size "$BATCH_SIZE" \
    --alpha "$ALPHA" \
    --device cuda \
    "$@" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "Launched Phase 1 static run (batch size $BATCH_SIZE)"
echo "  PID         : $PID"
echo "  sampler_id  : $SAMPLER_ID"
echo "  config      : $CONFIG"
echo "  batch_size  : $BATCH_SIZE"
echo "  alpha       : $ALPHA"
echo "  WANDB_MODE  : $WANDB_MODE"
echo "  log         : $LOG"
echo "  ckpt dir    : $REPO_ROOT/checkpoints/phase1_static/$SAMPLER_ID"
echo
echo "Follow:  tail -f \"$LOG\""
echo "Stop:    kill $PID"
