#!/usr/bin/env bash
# 新規手法の実行スクリプト雛形。
# methods/<method_name>/run.sh にコピーして使う。ログは logs/<method_name>/ に固定。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="$(basename "$HERE")"
LOG_DIR="$HERE/../../logs/$METHOD"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "[run.sh] method=$METHOD log=$LOG_FILE"

python "$HERE/train.py" "$@" 2>&1 | tee "$LOG_FILE"
