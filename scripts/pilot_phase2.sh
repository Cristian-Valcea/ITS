#!/bin/bash

# 🎯 CONTROLLED PHASE 2 PILOT
# 20K steps: 10K @ 75% DD warmup + 10K @ 65% DD with early-exit tax
# Monitor live: abort if ep_len < 50 or ep_rew < -20

echo "🎯 CONTROLLED PHASE 2 PILOT"
echo "============================"
echo "Goal: Test early-exit tax effectiveness"
echo "Schedule: 10K @ 75% DD → 10K @ 65% DD"
echo "Early-exit tax: -5.0 for episodes < 80 steps"
echo "Abort conditions: ep_len < 50 OR ep_rew < -20"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p train_runs/pilot_phase2
mkdir -p train_runs/pilot_analysis

# Wait for Phase 1 to complete if still running
echo "⏳ Waiting for Phase 1 to complete..."
while pgrep -f "confirm_G5" > /dev/null; do
    echo "   Phase 1 still running, waiting 30 seconds..."
    sleep 30
done

echo "✅ Phase 1 completed, starting pilot"
echo ""

# Check if Phase 1 model exists
if [ ! -f "train_runs/confirm_G5/model_50k.zip" ]; then
    echo "❌ Phase 1 model not found: train_runs/confirm_G5/model_50k.zip"
    echo "🛑 Cannot proceed with pilot"
    exit 1
fi

echo "🚀 PILOT PART A: 10K STEPS @ 75% DD WITH EARLY-EXIT TAX"
echo "======================================================="
echo "Goal: Warm-up with same DD limit but early-exit penalty"
echo "Parameters: LR=5e-5, KL=0.015, DD=75%, Early-exit tax=5.0"
echo ""

# Part A: 10K steps at 75% DD with early-exit tax
echo "🔄 Starting Pilot Part A..."
LOG_FILE_A="train_runs/pilot_phase2/training_part_a.log"

python train_diagnostic.py \
    --config config/pilot_phase2.yaml \
    --total_timesteps 10000 \
    --save_path "train_runs/pilot_phase2/model_10k_75dd.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 1000 \
    --eval_interval 1000 \
    2>&1 | tee $LOG_FILE_A &

PILOT_A_PID=$!

# Monitor Part A progress
echo "📊 Monitoring Pilot Part A progress..."
ABORT_PILOT=false
CHECK_INTERVAL=60  # Check every minute

while kill -0 $PILOT_A_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
    
    # Check latest metrics
    if [ -f "$LOG_FILE_A" ]; then
        LATEST_EP_LEN=$(grep -E "ep_len_mean.*\|" "$LOG_FILE_A" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
        LATEST_EP_REW=$(grep -E "ep_rew_mean.*\|" "$LOG_FILE_A" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")
        LATEST_STEPS=$(grep -E "total_timesteps.*\|" "$LOG_FILE_A" | tail -1 | sed 's/.*|\s*\([0-9]*\).*/\1/' || echo "0")
        
        if [ ! -z "$LATEST_EP_LEN" ] && [ ! -z "$LATEST_EP_REW" ] && [ ! -z "$LATEST_STEPS" ]; then
            echo "   Step $LATEST_STEPS: ep_len=$LATEST_EP_LEN, ep_rew=$LATEST_EP_REW"
            
            # Check abort conditions
            if (( $(echo "$LATEST_EP_LEN < 50" | bc -l) )); then
                echo "🚨 ABORT CONDITION: Episode length $LATEST_EP_LEN < 50"
                ABORT_PILOT=true
                break
            fi
            
            if (( $(echo "$LATEST_EP_REW < -20" | bc -l) )); then
                echo "🚨 ABORT CONDITION: Episode reward $LATEST_EP_REW < -20"
                ABORT_PILOT=true
                break
            fi
        fi
    fi
done

# Handle abort or completion
if [ "$ABORT_PILOT" = true ]; then
    echo "🛑 ABORTING PILOT - Conditions not met"
    kill $PILOT_A_PID 2>/dev/null
    
    echo ""
    echo "🔍 PILOT FAILURE ANALYSIS"
    echo "========================="
    echo "Running correlation analysis on Phase 1 logs..."
    
    # Run correlation analysis
    python -c "
import pandas as pd
import numpy as np
import re

# Parse Phase 1 log
with open('train_runs/confirm_G5/training.log', 'r') as f:
    content = f.read()

# Extract episode data
ep_lens = [float(x) for x in re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)]
ep_rews = [float(x) for x in re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)]

if len(ep_lens) > 1 and len(ep_rews) > 1:
    # Align lengths
    min_len = min(len(ep_lens), len(ep_rews))
    ep_lens = ep_lens[:min_len]
    ep_rews = ep_rews[:min_len]
    
    # Calculate correlation
    correlation = np.corrcoef(ep_lens, ep_rews)[0,1]
    
    print(f'📊 Episode Length vs Reward Correlation: {correlation:.3f}')
    
    if correlation < -0.3:
        print('❌ NEGATIVE CORRELATION: Reward function favors early exits')
        print('💡 Recommendation: Reshape rewards (add per-step time bonus)')
    elif abs(correlation) < 0.3:
        print('⚠️  WEAK CORRELATION: Optimizer may be unstable')
        print('💡 Recommendation: Lower LR to 4e-5 and repeat mini-grid')
    else:
        print('✅ POSITIVE CORRELATION: Good reward-length relationship')
        print('💡 Recommendation: Investigate other factors (data quality, environment)')
else:
    print('❌ Insufficient data for correlation analysis')
"
    
    exit 1
fi

# Wait for Part A to complete
wait $PILOT_A_PID

echo ""
echo "🔍 PILOT PART A ANALYSIS"
echo "========================"

# Extract final Part A metrics
FINAL_EP_LEN_A=$(tail -200 "$LOG_FILE_A" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
FINAL_EP_REW_A=$(tail -200 "$LOG_FILE_A" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")

echo "📊 Part A Results (10K @ 75% DD + early-exit tax):"
echo "   Final episode length: $FINAL_EP_LEN_A steps"
echo "   Final episode reward: $FINAL_EP_REW_A"

# Check if we should proceed to Part B
PROCEED_TO_B=true
if (( $(echo "$FINAL_EP_LEN_A < 60" | bc -l) )); then
    echo "⚠️  Episode length still concerning: $FINAL_EP_LEN_A < 60"
    PROCEED_TO_B=false
fi

if (( $(echo "$FINAL_EP_REW_A < -18" | bc -l) )); then
    echo "⚠️  Reward regressed: $FINAL_EP_REW_A < -18"
    PROCEED_TO_B=false
fi

if [ "$PROCEED_TO_B" = false ]; then
    echo "🛑 Not proceeding to Part B - Part A results insufficient"
    exit 1
fi

echo ""
echo "🚀 PILOT PART B: 10K STEPS @ 65% DD WITH EARLY-EXIT TAX"
echo "======================================================="
echo "Goal: Test tighter drawdown with early-exit penalty"
echo "Parameters: LR=5e-5, KL=0.015, DD=65%, Early-exit tax=5.0"
echo ""

# Create 65% DD config
CONFIG_65DD="config/pilot_phase2_65dd.yaml"
cp config/pilot_phase2.yaml $CONFIG_65DD
sed -i "s/max_drawdown_pct: 0.75/max_drawdown_pct: 0.65/" $CONFIG_65DD

# Part B: 10K steps at 65% DD with early-exit tax
echo "🔄 Starting Pilot Part B..."
LOG_FILE_B="train_runs/pilot_phase2/training_part_b.log"

python train_diagnostic.py \
    --config $CONFIG_65DD \
    --total_timesteps 10000 \
    --save_path "train_runs/pilot_phase2/model_20k_65dd.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 1000 \
    --eval_interval 1000 \
    2>&1 | tee $LOG_FILE_B &

PILOT_B_PID=$!

# Monitor Part B progress
echo "📊 Monitoring Pilot Part B progress..."

while kill -0 $PILOT_B_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
    
    # Check latest metrics
    if [ -f "$LOG_FILE_B" ]; then
        LATEST_EP_LEN=$(grep -E "ep_len_mean.*\|" "$LOG_FILE_B" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
        LATEST_EP_REW=$(grep -E "ep_rew_mean.*\|" "$LOG_FILE_B" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")
        LATEST_STEPS=$(grep -E "total_timesteps.*\|" "$LOG_FILE_B" | tail -1 | sed 's/.*|\s*\([0-9]*\).*/\1/' || echo "0")
        
        if [ ! -z "$LATEST_EP_LEN" ] && [ ! -z "$LATEST_EP_REW" ] && [ ! -z "$LATEST_STEPS" ]; then
            echo "   Step $LATEST_STEPS: ep_len=$LATEST_EP_LEN, ep_rew=$LATEST_EP_REW"
            
            # Check abort conditions
            if (( $(echo "$LATEST_EP_LEN < 50" | bc -l) )); then
                echo "🚨 ABORT CONDITION: Episode length $LATEST_EP_LEN < 50"
                kill $PILOT_B_PID 2>/dev/null
                break
            fi
            
            if (( $(echo "$LATEST_EP_REW < -20" | bc -l) )); then
                echo "🚨 ABORT CONDITION: Episode reward $LATEST_EP_REW < -20"
                kill $PILOT_B_PID 2>/dev/null
                break
            fi
        fi
    fi
done

# Wait for Part B to complete
wait $PILOT_B_PID

# Clean up temp config
rm -f $CONFIG_65DD

echo ""
echo "🔍 PILOT PHASE 2 FINAL ANALYSIS"
echo "==============================="

# Extract final Part B metrics
FINAL_EP_LEN_B=$(tail -200 "$LOG_FILE_B" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' || echo "0")
FINAL_EP_REW_B=$(tail -200 "$LOG_FILE_B" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' || echo "0")

echo "📊 PILOT RESULTS SUMMARY:"
echo "========================="
echo "Part A (75% DD): ep_len=$FINAL_EP_LEN_A, ep_rew=$FINAL_EP_REW_A"
echo "Part B (65% DD): ep_len=$FINAL_EP_LEN_B, ep_rew=$FINAL_EP_REW_B"
echo ""

# Success evaluation
SUCCESS_COUNT=0

if (( $(echo "$FINAL_EP_LEN_B >= 70" | bc -l) )); then
    echo "✅ Episode Length: $FINAL_EP_LEN_B ≥ 70 steps (good progress toward 80)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Episode Length: $FINAL_EP_LEN_B < 70 steps"
fi

if (( $(echo "$FINAL_EP_REW_B >= -18" | bc -l) )); then
    echo "✅ Reward Stability: $FINAL_EP_REW_B ≥ -18 (maintained progress)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Reward Regression: $FINAL_EP_REW_B < -18"
fi

echo ""
echo "🎯 PILOT ASSESSMENT:"
echo "===================="

if [ $SUCCESS_COUNT -eq 2 ]; then
    echo "🎉 PILOT SUCCESS!"
    echo "✅ Early-exit tax is effective"
    echo "✅ Tighter drawdown limits are manageable"
    echo "🚀 Recommendation: Proceed with full Phase 2 curriculum"
    echo ""
    echo "📁 Pilot models saved:"
    echo "   train_runs/pilot_phase2/model_10k_75dd.zip"
    echo "   train_runs/pilot_phase2/model_20k_65dd.zip"
elif [ $SUCCESS_COUNT -eq 1 ]; then
    echo "🎯 PILOT PARTIAL SUCCESS"
    echo "⚠️  Some concerns remain but progress is evident"
    echo "💡 Recommendation: Proceed cautiously with modified Phase 2"
    echo "   - Consider gentler drawdown transition (75% → 70% → 65%)"
    echo "   - Monitor closely for regression"
else
    echo "❌ PILOT FAILED"
    echo "🔄 Early-exit tax insufficient to maintain episode length"
    echo "💡 Recommendations:"
    echo "   1. Increase early-exit tax to 10.0"
    echo "   2. Add per-step time bonus (+0.01 per step)"
    echo "   3. Investigate reward system fundamental issues"
fi

echo ""
echo "📊 Detailed logs available in:"
echo "   train_runs/pilot_phase2/training_part_a.log"
echo "   train_runs/pilot_phase2/training_part_b.log"
echo ""
echo "✅ PILOT PHASE 2 COMPLETE!"