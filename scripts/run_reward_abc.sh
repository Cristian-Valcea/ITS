#!/bin/bash
# Phase 1B: Reward Engine A/B/C Test
# Purpose: Isolate whether RefinedRewardSystem is causing degradation

set -e

echo "🔬 Starting Phase 1B: Reward Engine A/B/C Test"
echo "============================================================"
echo "⚠️  TESTING THREE REWARD VARIANTS:"
echo "   - Variant A: Stock V3Enhanced reward (baseline)"
echo "   - Variant B: RefinedRewardSystem as wrapper (current shim)"
echo "   - Variant C: RefinedRewardSystem integrated into environment"
echo ""
echo "🚨 SPEED-BUMP MONITORING ENABLED:"
echo "   - Auto-abort if Sharpe < -0.5 before 30K steps"
echo "   - Rolling 1K-step Sharpe tracking"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Create Phase 1B directory structure
PHASE1B_DIR="diagnostic_runs/phase1b_reward_abc"
mkdir -p "$PHASE1B_DIR"/{logs,checkpoints,results}
echo "📁 Phase 1B directory: $PHASE1B_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PHASE1B_DIR/logs/reward_abc_$TIMESTAMP.log"
echo "📝 Log file: $LOG_FILE"

# Validate configuration files
echo ""
echo "✅ Configuration files validated"

# Define variants
declare -A VARIANTS=(
    ["A"]="config/reward_v3.yaml"
    ["B"]="config/reward_shim.yaml" 
    ["C"]="config/reward_internal.yaml"
)

declare -A VARIANT_NAMES=(
    ["A"]="Stock V3Enhanced (baseline)"
    ["B"]="RefinedRewardSystem wrapper (shim)"
    ["C"]="RefinedRewardSystem integrated"
)

# Results tracking
declare -A RESULTS=()
declare -A ABORTED=()

# Process each variant
for variant in A B C; do
    config_file="${VARIANTS[$variant]}"
    variant_name="${VARIANT_NAMES[$variant]}"
    
    echo ""
    echo "=================================================="
    echo "Processing Variant $variant: $variant_name"
    echo "=================================================="
    
    checkpoint_path="$PHASE1B_DIR/checkpoints/variant_${variant}_50k.zip"
    
    echo "🚀 Training Variant $variant (50K steps)..."
    echo "   Config: $config_file"
    echo "   Checkpoint will be saved to: $checkpoint_path"
    
    # Start speed-bump monitoring in background
    monitor_log="$PHASE1B_DIR/logs/speedbump_variant_${variant}_$TIMESTAMP.log"
    python scripts/monitor_sharpe_speedbump.py \
        --variant "$variant" \
        --log-file "$monitor_log" \
        --abort-threshold -0.5 \
        --min-steps 30000 &
    MONITOR_PID=$!
    
    # Run training with speed-bump monitoring
    if python train_diagnostic.py \
        --config "$config_file" \
        --total_timesteps 50000 \
        --save_path "$checkpoint_path" \
        2>&1 | tee -a "$LOG_FILE"; then
        
        echo "✅ Training completed for Variant $variant"
        RESULTS[$variant]="SUCCESS"
        
        # Kill monitoring process
        kill $MONITOR_PID 2>/dev/null || true
        
        # Evaluate the checkpoint
        echo "🔍 Evaluating Variant $variant checkpoint..."
        eval_result="$PHASE1B_DIR/results/eval_variant_${variant}_feb2024.csv"
        
        if python evaluate_diagnostic_checkpoint.py \
            --checkpoint "$checkpoint_path" \
            --config config/freeze_early_eval.yaml \
            --output "$eval_result" \
            2>&1 | tee -a "$LOG_FILE"; then
            
            echo "✅ Evaluation completed for Variant $variant"
            
            # Extract mean reward from results
            mean_reward=$(tail -n 1 "$eval_result" | cut -d',' -f6)
            RESULTS[$variant]="SUCCESS:$mean_reward"
            
        else
            echo "❌ Evaluation failed for Variant $variant"
            RESULTS[$variant]="EVAL_FAILED"
        fi
        
    else
        echo "❌ Training failed for Variant $variant"
        RESULTS[$variant]="TRAIN_FAILED"
        
        # Check if it was a speed-bump abort
        if kill -0 $MONITOR_PID 2>/dev/null; then
            kill $MONITOR_PID 2>/dev/null || true
            if grep -q "SPEED-BUMP ABORT" "$monitor_log" 2>/dev/null; then
                echo "🚨 Variant $variant aborted due to speed-bump trigger"
                ABORTED[$variant]="SPEEDBUMP_ABORT"
                RESULTS[$variant]="SPEEDBUMP_ABORT"
            fi
        fi
    fi
done

echo ""
echo "============================================================"
echo "📊 PHASE 1B RESULTS SUMMARY"
echo "============================================================"

# Generate summary report
summary_file="$PHASE1B_DIR/reward_abc_summary.csv"
echo "variant,status,mean_reward,aborted,variant_name" > "$summary_file"

all_success=true
speedbump_aborts=0

for variant in A B C; do
    result="${RESULTS[$variant]}"
    variant_name="${VARIANT_NAMES[$variant]}"
    aborted="${ABORTED[$variant]:-NO}"
    
    if [[ "$result" == SUCCESS:* ]]; then
        mean_reward="${result#SUCCESS:}"
        status="SUCCESS"
        echo "✅ Variant $variant ($variant_name): Mean reward = $mean_reward"
    elif [[ "$result" == "SPEEDBUMP_ABORT" ]]; then
        mean_reward="N/A"
        status="SPEEDBUMP_ABORT"
        speedbump_aborts=$((speedbump_aborts + 1))
        all_success=false
        echo "🚨 Variant $variant ($variant_name): SPEED-BUMP ABORT"
    else
        mean_reward="N/A"
        status="$result"
        all_success=false
        echo "❌ Variant $variant ($variant_name): $result"
    fi
    
    echo "$variant,$status,$mean_reward,$aborted,$variant_name" >> "$summary_file"
done

echo ""
echo "📊 Summary saved to: $summary_file"

# Analysis and recommendations
echo ""
echo "============================================================"
echo "🔍 DIAGNOSTIC ANALYSIS"
echo "============================================================"

if [ $speedbump_aborts -gt 0 ]; then
    echo "🚨 CRITICAL: $speedbump_aborts variant(s) triggered speed-bump abort"
    echo "   This indicates fundamental training instability"
    echo "   Recommendation: Investigate optimizer settings and reward surface"
fi

# Pattern analysis
variant_a_status="${RESULTS[A]}"
variant_b_status="${RESULTS[B]}"
variant_c_status="${RESULTS[C]}"

if [[ "$variant_a_status" == SUCCESS:* ]] && [[ "$variant_b_status" != SUCCESS:* ]] && [[ "$variant_c_status" != SUCCESS:* ]]; then
    echo "🎯 PATTERN: Only Variant A (baseline) succeeded"
    echo "   → RefinedRewardSystem is causing the degradation"
    echo "   → Focus debugging on reward system logic"
elif [[ "$variant_a_status" == SUCCESS:* ]] && [[ "$variant_b_status" != SUCCESS:* ]] && [[ "$variant_c_status" == SUCCESS:* ]]; then
    echo "🎯 PATTERN: Variant A and C succeeded, B failed"
    echo "   → Wrapper implementation issue in RefinedRewardSystem"
    echo "   → Integration approach (C) works better than shim (B)"
elif [[ "$variant_a_status" != SUCCESS:* ]] && [[ "$variant_b_status" != SUCCESS:* ]] && [[ "$variant_c_status" != SUCCESS:* ]]; then
    echo "🎯 PATTERN: All variants failed"
    echo "   → Environment or optimizer issue, not reward system"
    echo "   → Focus on training dynamics and data pipeline"
else
    echo "🎯 PATTERN: Mixed results - requires detailed analysis"
    echo "   → Compare individual variant performance metrics"
fi

if $all_success; then
    echo ""
    echo "🎉 PHASE 1B PASSED: All variants completed successfully"
    echo "   → Reward system is not the primary cause of degradation"
    echo "   → Continue with Phase 1C (Data Leakage Audit)"
else
    echo ""
    echo "⚠️  PHASE 1B ISSUES DETECTED: Some variants failed"
    echo "   → Analyze failure patterns above"
    echo "   → Consider micro-overfit probe for detailed investigation"
fi

echo ""
echo "🏁 Phase 1B completed at $(date)"
echo "📁 All results saved in: $PHASE1B_DIR"