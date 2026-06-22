#!/bin/bash
# Phase 1 Adaptive-bin: train CIFAR-10 Flow Matching with a LEARNED training-time
# timestep distribution (t split into K bins; the per-bin sampling probability is
# learned by REINFORCE). See train_phase1_bin_adaptive.py for the mechanism.
#
# Two ways to set the run, both work:
#   (1) Edit the defaults below, then:  bash scripts/run_phase1_bin_adaptive.sh
#   (2) Override per-run via env vars (no file edit), e.g.:
#         BIN_K=10 ENTROPY_COEF=0.02 UPDATE_SAMPLER_EVERY=40 \
#         bash scripts/run_phase1_bin_adaptive.sh
# Every setting below uses ${VAR:-default}, so an env var of the same name wins;
# unset vars fall back to the default. Run from anywhere inside the repo.

# ---------------------------------------------------------------------------
# Settings (override any of these via an env var of the same name)
# ---------------------------------------------------------------------------
# --- adaptive-bin sampler ---
BIN_K=${BIN_K:-10}                                # number of timestep bins
SAMPLER_LR=${SAMPLER_LR:-1e-3}                    # Adam lr for bin_logits
BASELINE_BETA=${BASELINE_BETA:-0.9}              # EMA factor for the reward baseline
ENTROPY_COEF=${ENTROPY_COEF:-0.01}               # entropy bonus weight
UPDATE_SAMPLER_EVERY=${UPDATE_SAMPLER_EVERY:-40}  # update bins every N UNet steps
EVAL_TIMES=${EVAL_TIMES:-0.1,0.5,0.9}            # eval timesteps S for the reward

# --- run management ---
# Empty = full run (180 epochs). DRY_RUN=1 -> 2 epochs / FID 2000 (smoke-test).
DRY_RUN=${DRY_RUN:-}
MAX_EPOCHS=${MAX_EPOCHS:-}                         # override epoch count (optional)
EVAL_EVERY=${EVAL_EVERY:-}                         # override FID/eval cadence (optional)
# Resume from an explicit checkpoint instead of this run's own latest.pt (optional).
RESUME_FROM=${RESUME_FROM:-}
# Start over, ignoring any existing latest.pt (DANGER: overwrites the same dir).
NO_RESUME=${NO_RESUME:-}
# Override the run id / checkpoint dir name (default adaptive_bin_K<BIN_K>).
SAMPLER_ID=${SAMPLER_ID:-}
# Log file label. Re-runs with the SAME label APPEND to logs/run_<LABEL>.log.
# Leave empty to auto-derive from SAMPLER_ID (matches the checkpoint dir id).
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
    LABEL="${SAMPLER_ID:-adaptive_bin_K${BIN_K}}"
fi
LOGFILE="$LOGDIR/run_${LABEL}.log"

EXTRA_ARGS=""
[ -n "$MAX_EPOCHS" ]  && EXTRA_ARGS="$EXTRA_ARGS --max-epochs $MAX_EPOCHS"
[ -n "$EVAL_EVERY" ]  && EXTRA_ARGS="$EXTRA_ARGS --eval-every $EVAL_EVERY"
[ -n "$SAMPLER_ID" ]  && EXTRA_ARGS="$EXTRA_ARGS --sampler-id $SAMPLER_ID"
[ -n "$RESUME_FROM" ] && EXTRA_ARGS="$EXTRA_ARGS --resume-from $RESUME_FROM"
[ -n "$DRY_RUN" ]     && EXTRA_ARGS="$EXTRA_ARGS --dry-run"
[ -n "$NO_RESUME" ]   && EXTRA_ARGS="$EXTRA_ARGS --no-resume"

# Echo the resolved settings so a wrong override is obvious before the job starts.
echo "Resolved settings:"
echo "  BIN_K=$BIN_K  SAMPLER_LR=$SAMPLER_LR  BASELINE_BETA=$BASELINE_BETA"
echo "  ENTROPY_COEF=$ENTROPY_COEF  UPDATE_SAMPLER_EVERY=$UPDATE_SAMPLER_EVERY  EVAL_TIMES=$EVAL_TIMES"
echo "  DRY_RUN=${DRY_RUN:-<no>}  MAX_EPOCHS=${MAX_EPOCHS:-<default>}  EVAL_EVERY=${EVAL_EVERY:-<default>}"
echo "  SAMPLER_ID=${SAMPLER_ID:-adaptive_bin_K${BIN_K}}  RESUME_FROM=${RESUME_FROM:-<own latest.pt>}"
echo "  LOGFILE=$LOGFILE"

nohup python train_phase1_bin_adaptive.py \
    --bin-k "$BIN_K" \
    --sampler-lr "$SAMPLER_LR" \
    --baseline-beta "$BASELINE_BETA" \
    --entropy-coef "$ENTROPY_COEF" \
    --update-sampler-every "$UPDATE_SAMPLER_EVERY" \
    --eval-times "$EVAL_TIMES" \
    $EXTRA_ARGS \
    >> "$LOGFILE" 2>&1 &

echo "PID=$!  Log: $LOGFILE"
