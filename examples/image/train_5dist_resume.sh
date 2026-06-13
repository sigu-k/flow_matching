#!/bin/bash
set -e   # エラーで即停止(連鎖事故を防ぐ)
export PYTHONPATH=/home/jovyan/work/srv21/flow_matching:$PYTHONPATH
cd ~/work/srv21/flow_matching/examples/image
for dist in uniform center both data noise; do
    OUT=./output_${dist}
    if [ -f "${OUT}/checkpoint.pth" ]; then
        RESUME="--resume ${OUT}/checkpoint.pth"
        echo "===== Resuming $dist from checkpoint ====="
    else
        RESUME=""
        echo "===== Training $dist from scratch ====="
    fi
    python train.py \
        --dataset=cifar10 --batch_size=64 --accum_iter=1 --epochs=100 \
        --class_drop_prob=1.0 --cfg_scale=0.0 --use_ema \
        --timestep_dist=$dist --eval_frequency=10 \
        --output_dir=${OUT} ${RESUME}
    echo "===== Done $dist ====="
done
echo "All training done!"
