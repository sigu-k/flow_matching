#!/usr/bin/env bash
# Background launcher for the loss-aware bin Phase-1 run (production).
#
# Portable: all paths are derived from this script's location (no server name
# hardcoded), so `git pull` + run works on any server. Logs go to
# examples/image/logs/ (next to this script). Resumable: NO --no-resume, so re-running this after
# an interruption picks up from checkpoints/.../latest.pt automatically.
#
# Usage (from anywhere):
#   bash examples/image/run_phase1_bin_loss_aware.sh
#   bash examples/image/run_phase1_bin_loss_aware.sh --device cuda:1   # extra args pass through
#   tail -f <printed log path>
set -euo pipefail

# --- locate self / repo root (portable; no srv11/srv21 in paths) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../examples/image
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # repo root

SAMPLER_ID="lossaware_K10_T0.25_mix0.05_beta0.8"
LOG_DIR="$SCRIPT_DIR/logs/phase1_bin_loss_aware"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SAMPLER_ID}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"   # online for production; override if needed

cd "$SCRIPT_DIR"

# setsid + nohup: fully detached, survives terminal close. Production hypers
# (180 epochs, eval every 20, 50k FID) are the script defaults — no flags needed.
setsid nohup python -u train_phase1_bin_loss_aware.py \
    --device cuda \
    "$@" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "Launched loss-aware bin run"
echo "  PID        : $PID"
echo "  sampler_id : $SAMPLER_ID"
echo "  WANDB_MODE : $WANDB_MODE"
echo "  log        : $LOG"
echo "  ckpt dir   : $REPO_ROOT/checkpoints/phase1_bin_loss_aware/$SAMPLER_ID"
echo
echo "Follow:  tail -f \"$LOG\""
echo "Stop:    kill $PID"
