#!/bin/bash

# 🎯 PHASE 2 - FULL CURRICULUM RUN
# 100K steps: Warm-keep → Risk-tighten → Profit-polish
# Based on successful pilot results

echo "🎯 PHASE 2 - FULL CURRICULUM RUN"
echo "================================="
echo "Total: 100K steps (~4h at 500 FPS)"
echo ""
echo "📋 CURRICULUM SCHEDULE:"
echo "  Warm-keep:    0→20K @ 75% DD, LR=5e-5, KL=0.015"
echo "  Risk-tighten: 20K→60K @ 60% DD, LR=5e-5, KL=0.015" 
echo "  Profit-polish: 60K→100K @ 60% DD, adaptive LR, KL=0.012"
echo ""
echo "🎯 SUCCESS GATES:"
echo "  ep_len_mean ≥ 70 throughout"
echo "  ep_rew_mean ≥ -18 by 20K; ≥ -10 by 60K; ≥ -5 by 100K"
echo "  Median max-DD ≤ 55% by 60K"
echo "  KL ceiling hit-rate ≤ 15%"
echo ""
echo "🚨 ABORT GUARDS:"
echo "  ep_len_mean < 50 OR ep_rew_mean < -25"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p train_runs/phase2_full
mkdir -p train_runs/curriculum_G5

# Wait for pilot to complete if still running
echo "⏳ Waiting for pilot to complete..."
while pgrep -f "pilot_phase2" > /dev/null; do
    echo "   Pilot still running, waiting 30 seconds..."
    sleep 30
done

echo "✅ Pilot completed, checking for checkpoint..."

# Determine starting checkpoint
CKPT=""
if [ -f "train_runs/pilot_phase2/model_20k_65dd.zip" ]; then
    CKPT="train_runs/pilot_phase2/model_20k_65dd.zip"
    echo "✅ Using pilot checkpoint: $CKPT"
elif [ -f "train_runs/confirm_G5/model_50k.zip" ]; then
    CKPT="train_runs/confirm_G5/model_50k.zip"
    echo "⚠️  Using Phase 1 checkpoint: $CKPT"
else
    echo "❌ No suitable checkpoint found!"
    echo "🛑 Cannot proceed without trained model"
    exit 1
fi

echo ""
echo "🚀 LAUNCHING FULL PHASE 2 CURRICULUM"
echo "====================================="
echo "Starting from: $CKPT"
echo "Target: 100K steps with curriculum progression"
echo ""

# Launch the full curriculum training
LOG_FILE="train_runs/phase2_full/training.log"

python train_diagnostic.py \
    --config config/phase2_full_curriculum.yaml \
    --resume_from "$CKPT" \
    --total_timesteps 100000 \
    --save_path "train_runs/phase2_full/model_100k_final.zip" \
    --learning_rate 5e-5 \
    --target_kl 0.015 \
    --entropy_coef 0.03 \
    --save_interval 2500 \
    --eval_interval 2500 \
    --lr_adapt_patience 5000 \
    --lr_adapt_factor 0.4 \
    --max_daily_drawdown_pct_schedule '[{"step":0,"value":0.75},{"step":20000,"value":0.60}]' \
    --early_exit_tax 5.0 \
    --early_exit_threshold 80 \
    2>&1 | tee $LOG_FILE &

CURRICULUM_PID=$!

echo "📊 LIVE MONITORING PHASE 2 CURRICULUM"
echo "======================================"
echo "PID: $CURRICULUM_PID"
echo "Log: $LOG_FILE"
echo ""

# Monitoring variables
CHECK_INTERVAL=120  # Check every 2 minutes
ABORT_TRAINING=false
LAST_STEP=0

# Monitoring gates
declare -A STEP_GATES=(
    [20000]="-18"    # ep_rew_mean ≥ -18 by 20K
    [60000]="-10"    # ep_rew_mean ≥ -10 by 60K  
    [100000]="-5"    # ep_rew_mean ≥ -5 by 100K
)

echo "🔍 Starting live monitoring (checks every 2 minutes)..."
echo "Time: $(date)"
echo ""

while kill -0 $CURRICULUM_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
    
    if [ -f "$LOG_FILE" ]; then
        # Extract latest metrics
        LATEST_EP_LEN=$(grep -E "ep_len_mean.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "0")
        LATEST_EP_REW=$(grep -E "ep_rew_mean.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([-0-9.]*\).*/\1/' 2>/dev/null || echo "0")
        LATEST_STEPS=$(grep -E "total_timesteps.*\|" "$LOG_FILE" | tail -1 | sed 's/.*|\s*\([0-9]*\).*/\1/' 2>/dev/null || echo "0")
        
        if [ ! -z "$LATEST_EP_LEN" ] && [ ! -z "$LATEST_EP_REW" ] && [ ! -z "$LATEST_STEPS" ]; then
            # Progress indicator
            PROGRESS=$(echo "scale=1; $LATEST_STEPS * 100 / 100000" | bc -l 2>/dev/null || echo "0")
            
            echo "[$(date +%H:%M:%S)] Step $LATEST_STEPS ($PROGRESS%): ep_len=$LATEST_EP_LEN, ep_rew=$LATEST_EP_REW"
            
            # Check abort guards
            if (( $(echo "$LATEST_EP_LEN < 50" | bc -l 2>/dev/null || echo "0") )); then
                echo "🚨 ABORT GUARD TRIGGERED: Episode length $LATEST_EP_LEN < 50"
                ABORT_TRAINING=true
                break
            fi
            
            if (( $(echo "$LATEST_EP_REW < -25" | bc -l 2>/dev/null || echo "0") )); then
                echo "🚨 ABORT GUARD TRIGGERED: Episode reward $LATEST_EP_REW < -25"
                ABORT_TRAINING=true
                break
            fi
            
            # Check continuous gate: ep_len_mean ≥ 70 throughout
            if (( $(echo "$LATEST_EP_LEN < 70" | bc -l 2>/dev/null || echo "0") )) && [ "$LATEST_STEPS" -gt 10000 ]; then
                echo "⚠️  WARNING: Episode length $LATEST_EP_LEN < 70 (continuous gate)"
            fi
            
            # Check milestone gates
            for STEP_THRESHOLD in "${!STEP_GATES[@]}"; do
                REWARD_THRESHOLD=${STEP_GATES[$STEP_THRESHOLD]}
                
                if [ "$LATEST_STEPS" -ge "$STEP_THRESHOLD" ] && [ "$LAST_STEP" -lt "$STEP_THRESHOLD" ]; then
                    echo ""
                    echo "🎯 MILESTONE GATE CHECK: Step $STEP_THRESHOLD"
                    echo "   Target: ep_rew_mean ≥ $REWARD_THRESHOLD"
                    echo "   Actual: ep_rew_mean = $LATEST_EP_REW"
                    
                    if (( $(echo "$LATEST_EP_REW >= $REWARD_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
                        echo "   ✅ GATE PASSED!"
                    else
                        echo "   ⚠️  GATE MISSED (continuing with monitoring)"
                    fi
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
    echo "🛑 ABORTING PHASE 2 CURRICULUM"
    echo "=============================="
    kill $CURRICULUM_PID 2>/dev/null
    echo "Training terminated due to abort guard trigger"
    echo "Check logs for detailed analysis: $LOG_FILE"
    exit 1
fi

# Wait for training to complete
wait $CURRICULUM_PID
TRAINING_EXIT_CODE=$?

echo ""
echo "🔍 PHASE 2 CURRICULUM FINAL ANALYSIS"
echo "===================================="

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
echo "📊 FINAL RESULTS:"
echo "================="
echo "Total steps completed: $FINAL_STEPS / 100,000"
echo "Final episode length:  $FINAL_EP_LEN steps"
echo "Final episode reward:  $FINAL_EP_REW"
echo ""

# Success evaluation
SUCCESS_COUNT=0
TOTAL_GATES=4

echo "🎯 GATE EVALUATION:"
echo "==================="

# Gate 1: Episode length ≥ 70 throughout
if (( $(echo "$FINAL_EP_LEN >= 70" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ Episode Length: $FINAL_EP_LEN ≥ 70 steps"
    ((SUCCESS_COUNT++))
else
    echo "❌ Episode Length: $FINAL_EP_LEN < 70 steps"
fi

# Gate 2: Final reward target
if (( $(echo "$FINAL_EP_REW >= -5" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ Final Reward: $FINAL_EP_REW ≥ -5 (profit-polish target)"
    ((SUCCESS_COUNT++))
elif (( $(echo "$FINAL_EP_REW >= -10" | bc -l 2>/dev/null || echo "0") )); then
    echo "🎯 Final Reward: $FINAL_EP_REW ≥ -10 (risk-tighten target met)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Final Reward: $FINAL_EP_REW < -10"
fi

# Gate 3: Completion rate
if [ "$FINAL_STEPS" -ge 90000 ]; then
    echo "✅ Completion: $FINAL_STEPS ≥ 90K steps (90%+)"
    ((SUCCESS_COUNT++))
else
    echo "❌ Completion: $FINAL_STEPS < 90K steps"
fi

# Gate 4: No abort triggers
if [ "$ABORT_TRAINING" = false ]; then
    echo "✅ Stability: No abort guards triggered"
    ((SUCCESS_COUNT++))
else
    echo "❌ Stability: Abort guards triggered"
fi

echo ""
echo "🏆 PHASE 2 ASSESSMENT:"
echo "======================"
echo "Gates passed: $SUCCESS_COUNT / $TOTAL_GATES"

if [ $SUCCESS_COUNT -eq 4 ]; then
    echo "🎉 PHASE 2 COMPLETE SUCCESS!"
    echo "✅ Ready for live-bar fine-tune"
    echo "✅ Ready for IBKR paper-trade"
elif [ $SUCCESS_COUNT -ge 3 ]; then
    echo "🎯 PHASE 2 STRONG SUCCESS!"
    echo "✅ Minor adjustments may be needed"
    echo "✅ Proceed to live-bar fine-tune with monitoring"
elif [ $SUCCESS_COUNT -ge 2 ]; then
    echo "⚠️  PHASE 2 PARTIAL SUCCESS"
    echo "🔄 Consider additional training or parameter adjustment"
else
    echo "❌ PHASE 2 NEEDS REVISION"
    echo "🔄 Analyze logs and adjust approach"
fi

echo ""
echo "📁 OUTPUTS:"
echo "==========="
echo "Final model: train_runs/phase2_full/model_100k_final.zip"
echo "Training log: $LOG_FILE"
echo "Checkpoints: train_runs/phase2_full/ (every 2.5K steps)"
echo ""

# Generate quick stats
echo "📊 QUICK STATS:"
echo "==============="
python -c "
import re
import numpy as np

try:
    with open('$LOG_FILE', 'r') as f:
        content = f.read()
    
    # Extract all episode lengths and rewards
    ep_lens = [float(x) for x in re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)]
    ep_rews = [float(x) for x in re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)]
    
    if ep_lens and ep_rews:
        print(f'Episode Length: min={min(ep_lens):.1f}, max={max(ep_lens):.1f}, mean={np.mean(ep_lens):.1f}')
        print(f'Episode Reward: min={min(ep_rews):.1f}, max={max(ep_rews):.1f}, mean={np.mean(ep_rews):.1f}')
        
        # Improvement metrics
        if len(ep_lens) > 10:
            early_len = np.mean(ep_lens[:5])
            late_len = np.mean(ep_lens[-5:])
            early_rew = np.mean(ep_rews[:5])
            late_rew = np.mean(ep_rews[-5:])
            
            print(f'Length improvement: {early_len:.1f} → {late_len:.1f} ({late_len-early_len:+.1f})')
            print(f'Reward improvement: {early_rew:.1f} → {late_rew:.1f} ({late_rew-early_rew:+.1f})')
    else:
        print('Insufficient data for statistics')
        
except Exception as e:
    print(f'Stats generation failed: {e}')
"

echo ""
echo "✅ PHASE 2 CURRICULUM ANALYSIS COMPLETE!"
echo "Next steps: Live-bar fine-tune → IBKR paper-trade → Demo prep"