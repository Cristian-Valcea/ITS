#!/bin/bash

# 🔧 PHASE 2 CORRECTIVE TRAINING LAUNCHER
echo "🔧 LAUNCHING PHASE 2 CORRECTIVE TRAINING"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p train_runs/phase2_corrective

echo "🚀 Starting corrective training with carrot-and-smaller-stick approach..."
echo "Parameters:"
echo "  Early-exit tax: 3.0 (reduced from 5.0)"
echo "  Time bonus: +0.02 per step ≥60"
echo "  Completion bonus: +2.0 for episodes ≥80"
echo "  Threshold: 70 steps (reduced from 80)"
echo ""

# Launch training
python3 train_diagnostic.py \
    --config config/phase2_corrective.yaml \
    --resume_from "train_runs/phase2_full/model_100k_final.zip" \
    --total_timesteps 20000 \
    --save_path "train_runs/phase2_corrective/model_20k_corrective.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --early_exit_tax 3.0 \
    --early_exit_threshold 70 \
    --time_bonus 0.02 \
    --time_bonus_threshold 60 \
    --completion_bonus 2.0 \
    --completion_threshold 80 \
    2>&1 | tee train_runs/phase2_corrective/training.log

echo ""
echo "✅ CORRECTIVE TRAINING COMPLETE!"
echo "📁 Log: train_runs/phase2_corrective/training.log"
echo "📁 Model: train_runs/phase2_corrective/model_20k_corrective.zip"