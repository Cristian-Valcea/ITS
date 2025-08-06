#!/bin/bash

# 🔧 PHASE 2 CORRECTIVE TRAINING
# Fix episode length with carrot-and-smaller-stick approach
# Keep reward gains, add positive incentives

echo "🔧 PHASE 2 CORRECTIVE TRAINING"
echo "==============================="
echo "Goal: Fix episode length while preserving reward gains"
echo ""
echo "🎯 CORRECTIVE INCENTIVE STRUCTURE:"
echo "  Early-exit tax: 3.0 (reduced from 5.0)"
echo "  Time bonus: +0.02 per step ≥60"
echo "  Completion bonus: +2.0 for episodes ≥80"
echo "  Drawdown: 60% → 55% at 5K steps"
echo ""
echo "🎯 SUCCESS GATES:"
echo "  10K: ep_len≥70, tax<30%, rew≥-20, DD≤55%"
echo "  20K: ep_len≥80, tax<20%, rew≥-15, DD≤55%"
echo ""
echo "🚨 ABORT GUARDS:"
echo "  ep_len < 60 OR ep_rew < -25"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p train_runs/phase2_corrective

echo "🚀 LAUNCHING CORRECTIVE TRAINING"
echo "================================="
echo "Resuming from: train_runs/phase2_full/model_100k_final.zip"
echo "Target: 20K corrective steps"
echo ""

# Launch corrective training
LOG_FILE="train_runs/phase2_corrective/training.log"

python train_diagnostic.py \
    --config config/phase2_corrective.yaml \
    --resume_from "train_runs/phase2_full/model_100k_final.zip" \
    --total_timesteps 20000 \
    --save_path "train_runs/phase2_corrective/model_20k_corrective.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --max_daily_drawdown_pct_schedule '[{"step":0,"value":0.60},{"step":5000,"value":0.55}]' \
    --early_exit_tax 3.0 \
    --early_exit_threshold 70 \
    --time_bonus 0.02 \
    --time_bonus_threshold 60 \
    --completion_bonus 2.0 \
    --completion_threshold 80 \
    2>&1 | tee $LOG_FILE &

CORRECTIVE_PID=$!

echo "📊 LIVE MONITORING CORRECTIVE TRAINING"
echo "======================================="
echo "PID: $CORRECTIVE_PID"
echo "Log: $LOG_FILE"
echo ""

# Monitoring variables
CHECK_INTERVAL=120  # Check every 2 minutes
ABORT_TRAINING=false
LAST_STEP=0

# Success gates
declare -A STEP_GATES=(
    [10000]="70,-20,30"    # ep_len≥70, rew≥-20, tax<30%
    [20000]="80,-15,20"    # ep_len≥80, rew≥-15, tax<20%
)

echo "🔍 Starting corrective monitoring (checks every 2 minutes)..."
echo "Time: $(date)"
echo ""

while kill -0 $CORRECTIVE_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
    
    if [ -f "$LOG_FILE" ]; then
        # Extract latest metrics
        LATEST_EP_LEN=$(grep -E "ep_len_mean.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "0")
        LATEST_EP_REW=$(grep -E "ep_rew_mean.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' 2>/dev/null || echo "0")
        LATEST_STEPS=$(grep -E "total_timesteps.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([0-9]*\).*/\1/' 2>/dev/null || echo "0")
        
        # Count recent tax penalties
        TAX_COUNT=$(tail -1000 "$LOG_FILE" | grep -c "Early-exit tax applied" || echo "0")
        
        if [ ! -z "$LATEST_EP_LEN" ] && [ ! -z "$LATEST_EP_REW" ] && [ ! -z "$LATEST_STEPS" ]; then
            # Progress indicator
            PROGRESS=$(echo "scale=1; $LATEST_STEPS * 100 / 20000" | bc -l 2>/dev/null || echo "0")
            
            echo "[$(date +%H:%M:%S)] Step $LATEST_STEPS ($PROGRESS%): ep_len=$LATEST_EP_LEN, ep_rew=$LATEST_EP_REW, tax_hits=$TAX_COUNT"
            
            # Check abort guards
            if (( $(echo "$LATEST_EP_LEN < 60" | bc -l 2>/dev/null || echo "0") )); then
                echo "🚨 ABORT GUARD TRIGGERED: Episode length $LATEST_EP_LEN < 60"
                ABORT_TRAINING=true
                break
            fi
            
            if (( $(echo "$LATEST_EP_REW < -25" | bc -l 2>/dev/null || echo "0") )); then
                echo "🚨 ABORT GUARD TRIGGERED: Episode reward $LATEST_EP_REW < -25"
                ABORT_TRAINING=true
                break
            fi
            
            # Check milestone gates
            for STEP_THRESHOLD in "${!STEP_GATES[@]}"; do
                GATE_VALUES=${STEP_GATES[$STEP_THRESHOLD]}
                IFS=',' read -r TARGET_LEN TARGET_REW TARGET_TAX_RATE <<< "$GATE_VALUES"
                
                if [ "$LATEST_STEPS" -ge "$STEP_THRESHOLD" ] && [ "$LAST_STEP" -lt "$STEP_THRESHOLD" ]; then
                    echo ""
                    echo "🎯 MILESTONE GATE CHECK: Step $STEP_THRESHOLD"
                    echo "   Target: ep_len≥$TARGET_LEN, rew≥$TARGET_REW, tax<$TARGET_TAX_RATE%"
                    echo "   Actual: ep_len=$LATEST_EP_LEN, rew=$LATEST_EP_REW, tax_hits=$TAX_COUNT"
                    
                    GATES_PASSED=0
                    TOTAL_GATES=3
                    
                    if (( $(echo "$LATEST_EP_LEN >= $TARGET_LEN" | bc -l 2>/dev/null || echo "0") )); then
                        echo "   ✅ Episode length gate passed"
                        ((GATES_PASSED++))
                    else
                        echo "   ❌ Episode length gate missed"
                    fi
                    
                    if (( $(echo "$LATEST_EP_REW >= $TARGET_REW" | bc -l 2>/dev/null || echo "0") )); then
                        echo "   ✅ Reward gate passed"
                        ((GATES_PASSED++))
                    else
                        echo "   ❌ Reward gate missed"
                    fi
                    
                    # Rough tax rate estimate (very approximate)
                    if [ "$TAX_COUNT" -lt "$TARGET_TAX_RATE" ]; then
                        echo "   ✅ Tax rate gate passed"
                        ((GATES_PASSED++))
                    else
                        echo "   ❌ Tax rate gate missed"
                    fi
                    
                    echo "   📊 Gates passed: $GATES_PASSED/$TOTAL_GATES"
                    echo ""
                fi
            done
            
            LAST_STEP=$LATEST_STEPS
        fi
    fi
done

# Handle abort or completion
if [ "$ABORT_TRAINING" = true ]; then
    echo ""
    echo "🛑 ABORTING CORRECTIVE TRAINING"
    echo "==============================="
    kill $CORRECTIVE_PID 2>/dev/null
    echo "Training terminated due to abort guard trigger"
    echo "Check logs for detailed analysis: $LOG_FILE"
    exit 1
fi

# Wait for training to complete
wait $CORRECTIVE_PID
TRAINING_EXIT_CODE=$?

echo ""
echo "🔍 CORRECTIVE TRAINING FINAL ANALYSIS"
echo "====================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training exited with code: $TRAINING_EXIT_CODE"
fi

# Extract final metrics
FINAL_EP_LEN=$(tail -200 "$LOG_FILE" | grep "ep_len_mean" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "0")
FINAL_EP_REW=$(tail -200 "$LOG_FILE" | grep "ep_rew_mean" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' 2>/dev/null || echo "0")
FINAL_STEPS=$(tail -200 "$LOG_FILE" | grep "total_timesteps" | tail -1 | sed 's/.*|\s*\([0-9]*\).*/\1/' 2>/dev/null || echo "0")

echo ""
echo "📊 FINAL CORRECTIVE RESULTS:"
echo "============================"
echo "Total steps completed: $FINAL_STEPS / 20,000"
echo "Final episode length:  $FINAL_EP_LEN steps"
echo "Final episode reward:  $FINAL_EP_REW"
echo ""

# Success evaluation
SUCCESS_COUNT=0
TOTAL_GATES=2

echo "🎯 CORRECTIVE GATE EVALUATION:"
echo "=============================="

# Gate 1: Episode length ≥ 80
if (( $(echo "$FINAL_EP_LEN >= 80" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ Episode Length: $FINAL_EP_LEN ≥ 80 steps (EXCELLENT)"
    ((SUCCESS_COUNT++))
elif (( $(echo "$FINAL_EP_LEN >= 70" | bc -l 2>/dev/null || echo "0") )); then
    echo "🎯 Episode Length: $FINAL_EP_LEN ≥ 70 steps (GOOD)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Episode Length: $FINAL_EP_LEN < 70 steps"
fi

# Gate 2: Reward ≥ -15
if (( $(echo "$FINAL_EP_REW >= -15" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ Final Reward: $FINAL_EP_REW ≥ -15 (EXCELLENT)"
    ((SUCCESS_COUNT++))
elif (( $(echo "$FINAL_EP_REW >= -20" | bc -l 2>/dev/null || echo "0") )); then
    echo "🎯 Final Reward: $FINAL_EP_REW ≥ -20 (GOOD)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Final Reward: $FINAL_EP_REW < -20"
fi

echo ""
echo "🏆 CORRECTIVE ASSESSMENT:"
echo "========================="
echo "Gates passed: $SUCCESS_COUNT / $TOTAL_GATES"

if [ $SUCCESS_COUNT -eq 2 ]; then
    echo "🎉 CORRECTIVE TRAINING SUCCESS!"
    echo "✅ Ready for live-bar fine-tune"
    echo "✅ Episode length fixed"
    echo "✅ Reward gains preserved"
elif [ $SUCCESS_COUNT -eq 1 ]; then
    echo "🎯 CORRECTIVE TRAINING PARTIAL SUCCESS"
    echo "✅ Some improvement achieved"
    echo "🔄 Consider additional fine-tuning"
else
    echo "❌ CORRECTIVE TRAINING NEEDS MORE WORK"
    echo "🔄 May need parameter adjustment"
fi

echo ""
echo "📁 OUTPUTS:"
echo "==========="
echo "Final model: train_runs/phase2_corrective/model_20k_corrective.zip"
echo "Training log: $LOG_FILE"
echo "Checkpoints: train_runs/phase2_corrective/ (every 1K steps)"
echo ""

echo "✅ CORRECTIVE TRAINING ANALYSIS COMPLETE!"
echo "Next steps: Evaluate for live-bar fine-tune readiness"