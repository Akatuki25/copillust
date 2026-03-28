#!/bin/bash
# Run all technique experiments sequentially
set -e

CONFIGS_DIR="pose_estimation/models/configs/experiments/techniques"
TRAIN_DIR="experiments/train/techniques"

source .venv/bin/activate

for config in t1_ema t2_dropout t3_grad_accum t4_coarse_dropout t5_freeze_backbone t6_wider_sigma t7_cosine_restart; do
    echo "============================================"
    echo "Running: $config"
    echo "============================================"
    python -m pose_estimation.training.trainer \
        --config "$CONFIGS_DIR/${config}.py" \
        --work-dir "$TRAIN_DIR/$config" \
        --device mps
    echo "$config done."
    echo ""
done

echo "All experiments completed."
