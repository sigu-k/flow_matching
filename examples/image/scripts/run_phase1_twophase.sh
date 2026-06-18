#!/bin/bash
# Phase 1 Two-phase: switch the training-time timestep distribution at change_epoch.
#
#   epoch 1 .. CHANGE_EPOCH      -> phase 1
#   epoch CHANGE_EPOCH+1 .. end  -> phase 2
#
# Edit the three settings below, then: bash scripts/run_phase1_twophase.sh
# Run from anywhere inside the repo — paths are resolved automatically.

# ---------------------------------------------------------------------------
# 3 settings to specify
# ---------------------------------------------------------------------------
# Phase 1 timestep distribution (epochs 1..CHANGE_EPOCH)
PHASE1_CONFIG=configs/phase1_static/ln_mu-0.8.yaml
PHASE1_ALPHA=0
# Phase 2 timestep distribution (epochs CHANGE_EPOCH+1..end)
PHASE2_CONFIG=configs/phase1_static/uniform.yaml
PHASE2_ALPHA=0
# Epoch (inclusive, 1-indexed) at which phase 1 ends; phase 2 starts at +1.
CHANGE_EPOCH=60
# ---------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT/examples/image"

LOGDIR=./logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train_phase1_twophase_$(date +%Y%m%d_%H%M%S).log"

nohup python train_phase1_twophase.py \
    --phase1-config "$PHASE1_CONFIG" --phase1-alpha "$PHASE1_ALPHA" \
    --phase2-config "$PHASE2_CONFIG" --phase2-alpha "$PHASE2_ALPHA" \
    --change-epoch "$CHANGE_EPOCH" \
    >> "$LOGFILE" 2>&1 &

echo "PID=$!  Log: $LOGFILE"
