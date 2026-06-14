#!/bin/bash
# Phase 1: ln_mu-0.8 with Uniform mixture (alpha=0.3 default)
# Usage: bash scripts/run_phase1_ln_mix.sh
# Run from anywhere inside the repo — paths are resolved automatically.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT/examples/image"

LOGDIR=./logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train_phase1_ln_mix_$(date +%Y%m%d_%H%M%S).log"

nohup python train_phase1_static.py \
    --config configs/phase1_static/ln_mu-0.8.yaml \
    --no-resume \
    >> "$LOGFILE" 2>&1 &

echo "PID=$!  Log: $LOGFILE"
