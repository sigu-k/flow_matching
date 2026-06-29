#!/usr/bin/env bash
# Objective-reduction bin run. Logs go to logs/<dirname>/ per the repo
# per-method convention. Pass any train args through, e.g.:
#   bash methods/adaptive_bin/run.sh --bin-k 10
#   bash methods/adaptive_bin/run.sh --dry-run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="$(basename "$HERE")"
LOG_DIR="$HERE/../../logs/$METHOD"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "[run.sh] method=$METHOD log=$LOG_FILE"

python "$HERE/train_phase1_bin_objective_reduction.py" "$@" 2>&1 | tee "$LOG_FILE"
