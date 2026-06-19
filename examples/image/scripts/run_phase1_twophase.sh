#!/bin/bash
# Phase 1 Two-phase: switch the training-time timestep distribution at change_epoch.
#
#   epoch 1 .. CHANGE_EPOCH      -> phase 1
#   epoch CHANGE_EPOCH+1 .. end  -> phase 2
#
# Two ways to set the run, both work:
#   (1) Edit the defaults below, then:  bash scripts/run_phase1_twophase.sh
#   (2) Override per-run via env vars (no file edit), e.g.:
#         PHASE1_CONFIG=configs/phase1_static/ln_mu+0.8.yaml \
#         CHANGE_EPOCH=60 INIT_FROM= \
#         bash scripts/run_phase1_twophase.sh
# Every setting below uses ${VAR:-default}, so an env var of the same name wins;
# unset vars fall back to the default. Run from anywhere inside the repo.

# ---------------------------------------------------------------------------
# Settings (override any of these via an env var of the same name)
# ---------------------------------------------------------------------------
# Phase 1 timestep distribution (epochs 1..CHANGE_EPOCH)
PHASE1_CONFIG=${PHASE1_CONFIG:-configs/phase1_static/ln_mu-0.8.yaml}
PHASE1_ALPHA=${PHASE1_ALPHA:-0}
# Phase 2 timestep distribution (epochs CHANGE_EPOCH+1..end)
PHASE2_CONFIG=${PHASE2_CONFIG:-configs/phase1_static/uniform.yaml}
PHASE2_ALPHA=${PHASE2_ALPHA:-0}
# Epoch (inclusive, 1-indexed) at which phase 1 ends; phase 2 starts at +1.
CHANGE_EPOCH=${CHANGE_EPOCH:-60}

# --- fork points: which epochs to keep as full resume checkpoints -----------
# Epochs 1..F of any run with the SAME phase-1 dist are identical, so a later run
# can reuse the source's epoch-F state (INIT_FROM) and only recompute F+1.. .
# CKPT_EPOCHS lists the EXACT epochs to keep (kept forever, nothing else saved;
# ~1.8GB each). Planned forks at 30,40,50,60,70 only need {20,40,60} retained —
# 30/50/70 are reached by forking from 20/40/60 and recomputing a few epochs.
CKPT_EPOCHS=${CKPT_EPOCHS:-20,40,60}

# Then, to launch a fork, point INIT_FROM at a kept checkpoint (epoch <= CHANGE_EPOCH), e.g.
#   INIT_FROM=../../checkpoints/phase1_twophase/<source_id>/ckpt_epoch040.pt
# Leave INIT_FROM empty to train from scratch.
INIT_FROM=${INIT_FROM:-}

# Log file label. Re-runs with the SAME label APPEND to logs/run_<LABEL>.log,
# so one run stays in one file. Leave empty to auto-derive from the configs +
# change_epoch (matches the checkpoint dir id). Set a short name if you prefer,
# e.g. LABEL=lnp08_uniform_e60 -> logs/run_lnp08_uniform_e60.log
LABEL=${LABEL:-}
# ---------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT/examples/image"

LOGDIR=./logs
mkdir -p "$LOGDIR"
# Deterministic, human-readable log name -> re-runs append to the same file.
if [ -z "$LABEL" ]; then
    P1=$(basename "$PHASE1_CONFIG" .yaml)
    P2=$(basename "$PHASE2_CONFIG" .yaml)
    LABEL="${P1}__${P2}__e${CHANGE_EPOCH}"
fi
LOGFILE="$LOGDIR/run_${LABEL}.log"

EXTRA_ARGS=""
[ -n "$CKPT_EPOCHS" ] && EXTRA_ARGS="$EXTRA_ARGS --ckpt-epochs $CKPT_EPOCHS"
[ -n "$INIT_FROM" ]   && EXTRA_ARGS="$EXTRA_ARGS --init-from $INIT_FROM"

# Echo the resolved settings so a wrong override is obvious before the job starts.
echo "Resolved settings:"
echo "  PHASE1_CONFIG=$PHASE1_CONFIG  PHASE1_ALPHA=$PHASE1_ALPHA"
echo "  PHASE2_CONFIG=$PHASE2_CONFIG  PHASE2_ALPHA=$PHASE2_ALPHA"
echo "  CHANGE_EPOCH=$CHANGE_EPOCH  CKPT_EPOCHS=${CKPT_EPOCHS:-<none>}  INIT_FROM=${INIT_FROM:-<none, fresh>}"
echo "  LOGFILE=$LOGFILE"

nohup python train_phase1_twophase.py \
    --phase1-config "$PHASE1_CONFIG" --phase1-alpha "$PHASE1_ALPHA" \
    --phase2-config "$PHASE2_CONFIG" --phase2-alpha "$PHASE2_ALPHA" \
    --change-epoch "$CHANGE_EPOCH" \
    $EXTRA_ARGS \
    >> "$LOGFILE" 2>&1 &

echo "PID=$!  Log: $LOGFILE"
