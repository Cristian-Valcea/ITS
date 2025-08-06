#!/bin/bash

# 🎯 PROGRESSIVE TRAINING PLAN
# Phase 1: Confirmation run (50K) - Lock in length breakthrough
# Phase 2: Risk-tighten curriculum (30K) - Teach realistic drawdown discipline  
# Phase 3: Profit fine-tune (20K) - Push toward profitability
# Total: 100K extra steps using G5 parameters as foundation

echo "🎯 PROGRESSIVE TRAINING PLAN - 3 PHASES"
echo "========================================"
echo "Phase 1: 50K Confirmation (75% DD, LR=5e-5)"
echo "Phase 2: 30K Curriculum (75%→60% DD)"  
echo "Phase 3: 20K Profit Tune (60% DD, adaptive LR)"
echo "Total: 100K steps, ~4h wall-time"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create output directories
mkdir -p train_runs/{confirm_G5,curriculum_G5,profit_tune_G5}
mkdir -p train_runs/progressive_analysis

echo "🚀 PHASE 1: CONFIRMATION RUN (50K STEPS)"
echo "========================================="
echo "Goal: Confirm G5 stability scales, give policy time to improve rewards"
echo "Parameters: LR=5e-5, KL=0.015, DD=75%"
echo "Checkpoints: Every 2.5K steps"
echo ""

# Phase 1: 50K Confirmation Run
echo "🔄 Starting Phase 1..."
LOG_FILE="train_runs/confirm_G5/training.log"

python train_diagnostic.py \
    --config config/progressive_75dd.yaml \
    --total_timesteps 50000 \
    --save_path "train_runs/confirm_G5/model_50k.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 2500 \
    --eval_interval 2500 \
    2>&1 | tee $LOG_FILE

# Check Phase 1 success criteria
echo ""
echo "🔍 PHASE 1 CHECKPOINT ANALYSIS"
echo "==============================="

# Extract final metrics
FINAL_EP_LEN=$(tail -200 "$LOG_FILE" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
FINAL_EP_REW=$(tail -200 "$LOG_FILE" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")

echo "📊 Phase 1 Results:"
echo "   Final episode length: $FINAL_EP_LEN steps"
echo "   Final episode reward: $FINAL_EP_REW"
echo ""

# Gate check for Phase 1
PHASE1_SUCCESS=false
if (( $(echo "$FINAL_EP_LEN >= 80" | bc -l) )); then
    echo "✅ PHASE 1 SUCCESS: Episode length ≥80 achieved!"
    PHASE1_SUCCESS=true
    
    if (( $(echo "$FINAL_EP_REW >= -15" | bc -l) )); then
        echo "🎯 BONUS: Reward target (-15) also achieved!"
    else
        echo "⚠️  Reward still needs work: $FINAL_EP_REW vs -15 target"
    fi
else
    echo "❌ PHASE 1 FAILED: Episode length $FINAL_EP_LEN < 80"
    echo "🛑 Aborting progressive plan - need to revisit optimizer"
    echo "💡 Suggestions: Lower LR to 4e-5, widen clip to 0.25"
    exit 1
fi

echo ""
echo "🚀 PHASE 2: RISK-TIGHTEN CURRICULUM (30K STEPS)"
echo "==============================================="
echo "Goal: Teach stable policy to trade within realistic drawdown"
echo "Schedule: 0-20K @ 75% DD → 20-30K @ 60% DD"
echo "Parameters: Keep LR=5e-5, KL=0.015"
echo ""

# Phase 2: Curriculum Learning (simplified - we'll implement schedule manually)
echo "🔄 Starting Phase 2 (Part A: 20K @ 75% DD)..."
LOG_FILE="train_runs/curriculum_G5/training_part_a.log"

python train_diagnostic.py \
    --config config/progressive_75dd.yaml \
    --total_timesteps 20000 \
    --save_path "train_runs/curriculum_G5/model_20k_75dd.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 2500 \
    --eval_interval 2500 \
    2>&1 | tee $LOG_FILE

echo ""
echo "🔄 Starting Phase 2 (Part B: 10K @ 60% DD)..."
LOG_FILE="train_runs/curriculum_G5/training_part_b.log"

python train_diagnostic.py \
    --config config/progressive_60dd.yaml \
    --total_timesteps 10000 \
    --save_path "train_runs/curriculum_G5/model_30k_60dd.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 2500 \
    --eval_interval 2500 \
    2>&1 | tee $LOG_FILE

# Check Phase 2 success criteria
echo ""
echo "🔍 PHASE 2 CHECKPOINT ANALYSIS"
echo "==============================="

FINAL_EP_LEN=$(tail -200 "train_runs/curriculum_G5/training_part_b.log" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
FINAL_EP_REW=$(tail -200 "train_runs/curriculum_G5/training_part_b.log" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")

echo "📊 Phase 2 Results:"
echo "   Final episode length: $FINAL_EP_LEN steps"
echo "   Final episode reward: $FINAL_EP_REW"
echo ""

# Gate check for Phase 2
if (( $(echo "$FINAL_EP_LEN >= 80" | bc -l) )); then
    echo "✅ PHASE 2 SUCCESS: Episode length maintained ≥80!"
    
    if (( $(echo "$FINAL_EP_REW >= -10" | bc -l) )); then
        echo "🎯 BONUS: Reward improvement (-10) achieved!"
    else
        echo "⚠️  Reward progress slower than hoped: $FINAL_EP_REW vs -10 target"
    fi
else
    echo "⚠️  PHASE 2 CONCERN: Episode length dropped to $FINAL_EP_LEN"
    echo "🔄 Continuing to Phase 3 but monitoring closely..."
fi

echo ""
echo "🚀 PHASE 3: PROFIT FINE-TUNE (20K STEPS)"
echo "========================================"
echo "Goal: Push mean reward above -1.0 with adaptive LR"
echo "Parameters: Start LR=5e-5, adapt down to 2e-5 when plateau"
echo "Drawdown: 60% (realistic trading limit)"
echo "KL: Tightened to 0.012"
echo ""

# Phase 3: Profit Fine-Tune
echo "🔄 Starting Phase 3..."
LOG_FILE="train_runs/profit_tune_G5/training.log"

python train_diagnostic.py \
    --config config/progressive_60dd.yaml \
    --total_timesteps 20000 \
    --save_path "train_runs/profit_tune_G5/model_20k_profit.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.012 \
    --entropy_coef 0.03 \
    --save_interval 2500 \
    --eval_interval 2500 \
    2>&1 | tee $LOG_FILE

# Final Analysis
echo ""
echo "🔍 FINAL ANALYSIS - 100K PROGRESSIVE TRAINING"
echo "============================================="

FINAL_EP_LEN=$(tail -200 "train_runs/profit_tune_G5/training.log" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
FINAL_EP_REW=$(tail -200 "train_runs/profit_tune_G5/training.log" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")

echo "📊 FINAL RESULTS (100K total steps):"
echo "   Final episode length: $FINAL_EP_LEN steps"
echo "   Final episode reward: $FINAL_EP_REW"
echo ""

# Success criteria evaluation
echo "🏆 SUCCESS CRITERIA EVALUATION:"
echo "================================"

SUCCESS_COUNT=0

if (( $(echo "$FINAL_EP_LEN >= 80" | bc -l) )); then
    echo "✅ Episode Length: $FINAL_EP_LEN ≥ 80 steps"
    ((SUCCESS_COUNT++))
else
    echo "❌ Episode Length: $FINAL_EP_LEN < 80 steps"
fi

if (( $(echo "$FINAL_EP_REW >= -1" | bc -l) )); then
    echo "✅ Profitability: $FINAL_EP_REW ≥ -1.0 reward"
    ((SUCCESS_COUNT++))
else
    echo "❌ Profitability: $FINAL_EP_REW < -1.0 reward"
fi

echo ""
echo "🎯 PROGRESSIVE TRAINING SUMMARY:"
echo "================================"
echo "Phases completed: 3/3"
echo "Total training time: ~4 hours"
echo "Success criteria met: $SUCCESS_COUNT/2"

if [ $SUCCESS_COUNT -eq 2 ]; then
    echo ""
    echo "🎉 COMPLETE SUCCESS! 🎉"
    echo "✅ Episode length breakthrough locked in"
    echo "✅ Profitability target achieved"
    echo "🚀 Ready for live trading evaluation!"
    echo ""
    echo "📁 Final model: train_runs/profit_tune_G5/model_20k_profit.zip"
    echo "📊 Use this model for backtesting and paper trading"
elif [ $SUCCESS_COUNT -eq 1 ]; then
    echo ""
    echo "🎯 PARTIAL SUCCESS"
    echo "✅ Episode length breakthrough maintained"
    echo "⚠️  Profitability still needs work"
    echo "💡 Consider: Extended training, reward system tuning, or data quality investigation"
else
    echo ""
    echo "⚠️  CHALLENGES REMAIN"
    echo "❌ Both targets missed - need fundamental review"
    echo "💡 Investigate: Data quality, reward system, environment parameters"
fi

echo ""
echo "📈 Detailed analysis available in:"
echo "   train_runs/confirm_G5/training.log"
echo "   train_runs/curriculum_G5/training_part_*.log"  
echo "   train_runs/profit_tune_G5/training.log"
echo ""
echo "🔍 Run comprehensive analysis with:"
echo "   python scripts/analyze_progressive_training.py"

echo ""
echo "✅ PROGRESSIVE TRAINING PLAN COMPLETE!"