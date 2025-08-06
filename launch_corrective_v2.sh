#!/bin/bash

# 🚀 CORRECTIVE V2 - STRONGER INCENTIVES
echo "🚀 LAUNCHING CORRECTIVE V2 WITH STRONGER INCENTIVES"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p train_runs/corrective_v2

echo "🎯 CORRECTIVE V2 INCENTIVE STRUCTURE:"
echo "====================================="
echo "  Early-exit tax: -5.0 for <60 steps (was -3.0)"
echo "  Secondary tax:  -3.0 for 60-69 steps (NEW)"
echo "  Time bonus:     +0.05 per step ≥50 (was +0.02 ≥60)"
echo "  Completion:     +1.0 at 60 steps (NEW)"
echo "  Major bonus:    +3.0 at 80 steps (was +2.0)"
echo "  Max position:   0.8 (was 1.0) - prevents YOLO"
echo ""
echo "🎯 SUCCESS GATES:"
echo "  5K:  ep_len≥70, tax<40%, rew≥-15"
echo "  10K: ep_len≥80, tax<25%, rew≥-15"
echo ""
echo "🚨 ABORT CONDITIONS:"
echo "  ep_len<60 OR tax>80% OR rew<-25"
echo ""

echo "🚀 Starting Corrective V2 training (10K steps, ~25 min)..."
echo "Resuming from: train_runs/phase2_full/model_100k_final.zip"
echo ""

# Launch Corrective V2 training
python3 train_diagnostic.py \
    --config config/corrective_v2.yaml \
    --resume_from "train_runs/phase2_full/model_100k_final.zip" \
    --total_timesteps 10000 \
    --save_path "train_runs/corrective_v2/model_10k_v2.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 500 \
    --eval_interval 500 \
    --early_exit_tax 5.0 \
    --early_exit_threshold 60 \
    --secondary_tax 3.0 \
    --secondary_threshold 70 \
    --time_bonus 0.05 \
    --time_bonus_threshold 50 \
    --completion_bonus 1.0 \
    --completion_threshold 60 \
    --major_completion_bonus 3.0 \
    --major_threshold 80 \
    --max_position_ratio 0.8 \
    2>&1 | tee train_runs/corrective_v2/training.log

echo ""
echo "✅ CORRECTIVE V2 TRAINING COMPLETE!"
echo "📁 Log: train_runs/corrective_v2/training.log"
echo "📁 Model: train_runs/corrective_v2/model_10k_v2.zip"
echo ""
echo "🔍 Running analysis..."

# Run analysis
python3 scripts/corrective_analysis.py train_runs/corrective_v2/training.log

echo ""
echo "🎯 CORRECTIVE V2 ANALYSIS COMPLETE!"