#!/bin/bash
# run_fid_sweep.sh — 5x5 FID sweep (Euler, seed=0)
# Usage: bash run_fid_sweep.sh [NFE]   (default: 50)
# Skips cells where fid_results/nfe_<N>/<train>__<sample>/fid.json already exists.

set -eo pipefail
export PYTHONPATH=/home/jovyan/work/srv21/flow_matching${PYTHONPATH:+:$PYTHONPATH}

NFE=${1:-50}
BASE_DIR="fid_results/nfe_${NFE}"

# パッケージ再インストール(コンテナリセット対策)
pip install torchdiffeq torchmetrics[image] torch-fidelity --break-system-packages -q
cd "$(dirname "$0")"

LOG="${BASE_DIR}/sweep.log"
mkdir -p "${BASE_DIR}"
exec > >(tee -a "$LOG") 2>&1

DISTS=(uniform center both data noise)

echo "========================================"
echo "5x5 FID sweep started: $(date)"
echo "NFE=${NFE}  output=${BASE_DIR}"
echo "========================================"

TOTAL=0
SKIPPED=0

for TRAIN in "${DISTS[@]}"; do
    for SAMPLE in "${DISTS[@]}"; do
        OUT="${BASE_DIR}/${TRAIN}__${SAMPLE}"
        TOTAL=$((TOTAL + 1))

        if [ -f "${OUT}/fid.json" ]; then
            echo "[SKIP] ${TRAIN} x ${SAMPLE} — fid.json exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "-------- [${TOTAL}/25] train=${TRAIN}  sample=${SAMPLE}  $(date) --------"
        python train.py \
            --eval_only \
            --compute_fid \
            --dataset cifar10 \
            --use_ema \
            --cfg_scale 0.0 \
            --class_drop_prob 1.0 \
            --timestep_dist "${TRAIN}" \
            --sampling_dist "${SAMPLE}" \
            --nfe "${NFE}" \
            --batch_size 250 \
            --seed 0 \
            --resume "output_${TRAIN}/checkpoint.pth" \
            --output_dir "${OUT}" \
            --data_path ./data/image_generation \
            --ode_method euler
        echo "-------- Done: ${TRAIN} x ${SAMPLE}  $(date) --------"
    done
done

echo ""
echo "========================================"
echo "All done: $(date)  (skipped ${SKIPPED} cells)"
echo "========================================"

# Auto-summarize when all 25 cells are present
FILLED=$(find "${BASE_DIR}" -name "fid.json" | wc -l)
if [ "$FILLED" -ge 25 ]; then
    echo "Running summarize_fid.py and analyze_fid.py ..."
    python summarize_fid.py --fid_dir "${BASE_DIR}"
    python analyze_fid.py --fid_dir "${BASE_DIR}"
fi
