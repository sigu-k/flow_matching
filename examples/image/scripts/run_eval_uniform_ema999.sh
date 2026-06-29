#!/bin/bash
# run_eval_uniform_ema999.sh
# uniform で学習したモデルの推論 + FID 計算を EMA(decay=0.999)で実行する。
#
# EMA decay について:
#   models/ema.py の EMA は decay=0.999 がデフォルトで、output_uniform は
#   use_ema=true で学習済み(args.json 参照)。つまり checkpoint.pth に保存済みの
#   EMA がそのまま decay=0.999 の重み。したがって --use_ema を渡すだけでよく、
#   reconstruct_ema.py(スナップショットからの後付け再構成)は不要・不可
#   (uniform 学習はスナップショットを出力していない)。
#
# Usage:
#   bash scripts/run_eval_uniform_ema999.sh [SAMPLE_DIST] [NFE]
#     SAMPLE_DIST : 推論時ステップ配置 (uniform/center/both/data/noise) 既定 uniform
#     NFE         : ステップ数 既定 50

set -eo pipefail

# --- リポジトリルートを __file__ 相対で導出(サーバ名をハードコードしない) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../examples/image/scripts
IMAGE_DIR="$(dirname "$SCRIPT_DIR")"                          # .../examples/image
REPO_ROOT="$(cd "$IMAGE_DIR/../.." && pwd)"                   # flow_matching ルート
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cd "$IMAGE_DIR"

SAMPLE_DIST="${1:-uniform}"
NFE="${2:-50}"
TRAIN_DIST="uniform"

# uniform 学習済みチェックポイント(このサーバでの実体パス)
CKPT="${REPO_ROOT}/checkpoints/examples/image/output_${TRAIN_DIST}/checkpoint.pth"
OUT="fid_results/ema0.999/${TRAIN_DIST}__${SAMPLE_DIST}"

# パッケージ再インストール(コンテナリセット対策)
pip install torchdiffeq torchmetrics[image] torch-fidelity --break-system-packages -q

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi
if [ -f "${OUT}/fid.json" ]; then
    echo "[SKIP] ${OUT}/fid.json already exists"
    exit 0
fi

mkdir -p "$OUT"
echo "=== eval uniform (EMA decay=0.999) | sample=${SAMPLE_DIST} nfe=${NFE} | $(date) ==="
echo "ckpt: $CKPT"

python train.py \
    --eval_only \
    --compute_fid \
    --dataset cifar10 \
    --use_ema \
    --cfg_scale 0.0 \
    --class_drop_prob 1.0 \
    --timestep_dist "${TRAIN_DIST}" \
    --sampling_dist "${SAMPLE_DIST}" \
    --nfe "${NFE}" \
    --batch_size 250 \
    --seed 0 \
    --fid_samples 50000 \
    --resume "${CKPT}" \
    --output_dir "${OUT}" \
    --data_path ./data/image_generation \
    --ode_method euler

echo "=== Done: $(date)  -> ${OUT} ==="
