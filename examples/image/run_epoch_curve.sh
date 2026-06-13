#!/bin/bash
# run_epoch_curve.sh — epoch-wise FID sweep (5 train dists x 10 epochs, sampling=uniform)
# Usage: bash run_epoch_curve.sh
# Skips cells where fid_results/epoch_curve/<dist>__epoch<N>/fid.json already exists.

set -eo pipefail
export PYTHONPATH=/home/jovyan/work/srv21/flow_matching${PYTHONPATH:+:$PYTHONPATH}

BASE_DIR="fid_results/epoch_curve"
NFE=50

pip install torchdiffeq torchmetrics[image] torch-fidelity --break-system-packages -q
cd "$(dirname "$0")"

mkdir -p "${BASE_DIR}"
LOG="${BASE_DIR}/sweep.log"
exec > >(tee -a "$LOG") 2>&1

DISTS=(uniform noise data center both)
EPOCHS=(20 40 60 80 100)
TOTAL_CELLS=$(( ${#DISTS[@]} * ${#EPOCHS[@]} ))

echo "========================================"
echo "Epoch-curve FID sweep started: $(date)"
echo "NFE=${NFE}  sampling=uniform  output=${BASE_DIR}"
echo "Total cells: ${TOTAL_CELLS}"
echo "========================================"

CELL=0
SKIPPED=0

for DIST in "${DISTS[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        CELL=$((CELL + 1))
        CKPT_IDX=$((EPOCH - 1))
        CKPT="output_${DIST}/checkpoint-${CKPT_IDX}.pth"
        OUT="${BASE_DIR}/${DIST}__epoch${EPOCH}"

        if [ -f "${OUT}/fid.json" ]; then
            echo "[SKIP] ${DIST} epoch=${EPOCH} — fid.json exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        if [ ! -f "${CKPT}" ]; then
            echo "[WARN] checkpoint not found, skipping: ${CKPT}"
            continue
        fi

        echo ""
        echo "-------- [${CELL}/${TOTAL_CELLS}] dist=${DIST}  epoch=${EPOCH}  $(date) --------"
        python train.py \
            --eval_only \
            --compute_fid \
            --dataset cifar10 \
            --use_ema \
            --cfg_scale 0.0 \
            --class_drop_prob 1.0 \
            --timestep_dist "${DIST}" \
            --sampling_dist uniform \
            --nfe "${NFE}" \
            --batch_size 250 \
            --seed 0 \
            --resume "${CKPT}" \
            --output_dir "${OUT}" \
            --data_path ./data/image_generation \
            --ode_method euler
        echo "-------- Done: ${DIST} epoch=${EPOCH}  $(date) --------"
    done
done

echo ""
echo "========================================"
echo "All done: $(date)  (skipped ${SKIPPED}/${TOTAL_CELLS} cells)"
echo "========================================"

FILLED=$(find "${BASE_DIR}" -name "fid.json" | wc -l)
echo "fid.json count: ${FILLED}/${TOTAL_CELLS}"
