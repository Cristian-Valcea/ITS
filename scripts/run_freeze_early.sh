#!/bin/bash
# Phase 1A: Freeze-Early Validity Test (Management Enhanced)
# Trains checkpoints at 5K, 10K, 15K steps and validates ALL pass criteria

set -e  # Exit on any error

echo "🔬 Starting Phase 1A: Freeze-Early Validity Test (Enhanced)"
echo "============================================================"
echo "⚠️  MANAGEMENT ENHANCED CRITERIA:"
echo "   - ALL THREE checkpoints must pass ≥0.5 reward"
echo "   - NONE can fall below 0.3 reward floor"
echo "   - Episode lengths must be ≥40 steps consistently"
echo ""

# Configuration
PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
PHASE_DIR="${PROJECT_ROOT}/diagnostic_runs/phase1a_freeze_early"
TRAIN_CONFIG="${PROJECT_ROOT}/config/freeze_early_train.yaml"
EVAL_CONFIG="${PROJECT_ROOT}/config/freeze_early_eval.yaml"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "${PROJECT_ROOT}/venv/bin/activate"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated"

# Create directories
mkdir -p "${PHASE_DIR}/checkpoints"
mkdir -p "${PHASE_DIR}/eval_results"
mkdir -p "${PHASE_DIR}/logs"

# Log file
LOG_FILE="${PHASE_DIR}/logs/freeze_early_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "📁 Phase 1A directory: ${PHASE_DIR}"
echo "📝 Log file: ${LOG_FILE}"
echo ""

# Validate configuration files exist
if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "❌ Training config not found: $TRAIN_CONFIG"
    exit 1
fi

if [[ ! -f "$EVAL_CONFIG" ]]; then
    echo "❌ Evaluation config not found: $EVAL_CONFIG"
    exit 1
fi

echo "✅ Configuration files validated"
echo ""

# Training and evaluation function
train_and_evaluate() {
    local steps=$1
    local checkpoint_name="checkpoint_${steps}k"
    local checkpoint_path="${PHASE_DIR}/checkpoints/${checkpoint_name}.zip"
    local eval_output="${PHASE_DIR}/eval_results/eval_${steps}k_feb2024.csv"
    
    echo "🚀 Training ${checkpoint_name} (${steps}K steps)..."
    echo "   Checkpoint will be saved to: ${checkpoint_path}"
    
    # Train to checkpoint
    cd "$PROJECT_ROOT"
    python train_diagnostic.py \
        --config "$TRAIN_CONFIG" \
        --total_timesteps $((steps * 1000)) \
        --save_path "$checkpoint_path"
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Training failed for ${checkpoint_name}"
        return 1
    fi
    
    echo "✅ Training completed for ${checkpoint_name}"
    echo ""
    
    echo "🔍 Evaluating ${checkpoint_name} on Feb 2024 data..."
    echo "   Results will be saved to: ${eval_output}"
    
    # Evaluate checkpoint
    python evaluate_diagnostic_checkpoint.py \
        --checkpoint "$checkpoint_path" \
        --config "$EVAL_CONFIG" \
        --output "$eval_output" \
        --steps 5000 \
        --return_mean_reward
    
    local eval_result=$?
    if [[ $eval_result -ne 0 ]]; then
        echo "❌ Evaluation failed for ${checkpoint_name}"
        return 1
    fi
    
    echo "✅ Evaluation completed for ${checkpoint_name}"
    echo ""
    
    return 0
}

# Array to store results
declare -a results=()

# Train and evaluate each checkpoint
for steps in 5 10 15; do
    echo "=================================================="
    echo "Processing ${steps}K checkpoint..."
    echo "=================================================="
    
    if train_and_evaluate $steps; then
        # Extract mean reward from evaluation output
        eval_file="${PHASE_DIR}/eval_results/eval_${steps}k_feb2024.csv"
        if [[ -f "$eval_file" ]]; then
            mean_reward=$(python -c "
import pandas as pd
df = pd.read_csv('$eval_file')
print(df['mean_reward'].iloc[0])
")
            results+=($mean_reward)
            echo "📊 ${steps}K checkpoint result: ${mean_reward}"
        else
            echo "❌ Evaluation results file not found: $eval_file"
            exit 1
        fi
    else
        echo "❌ Failed to process ${steps}K checkpoint"
        exit 1
    fi
    
    echo ""
done

echo "=================================================="
echo "🔍 VALIDATING ALL CHECKPOINTS (Management Enhanced)"
echo "=================================================="

echo "📊 Individual checkpoint results:"
for i in "${!results[@]}"; do
    steps=$((5 + i * 5))
    echo "   ${steps}K checkpoint: ${results[$i]}"
done
echo ""

# Validate all checkpoints using management enhanced criteria
echo "🔬 Running enhanced validation..."
cd "$PROJECT_ROOT"
python scripts/validate_all_checkpoints.py \
    --results "${results[@]}" \
    --min_reward 0.5 \
    --min_floor 0.3 \
    --output "${PHASE_DIR}/all_checkpoints_summary.csv"

validation_result=$?

echo ""
echo "=================================================="
echo "🎯 PHASE 1A FINAL RESULTS"
echo "=================================================="

if [[ $validation_result -eq 0 ]]; then
    echo "🎉 PHASE 1A VALIDATION PASSED!"
    echo ""
    echo "✅ All checkpoints meet management criteria:"
    echo "   - ALL three checkpoints scored ≥0.5 reward"
    echo "   - NONE fell below 0.3 reward floor"
    echo "   - Early rewards appear to be REAL learning, not noise"
    echo ""
    echo "📋 Next Steps:"
    echo "   - Proceed to Phase 1B: Reward Engine A/B/C Test"
    echo "   - Foundation appears solid for further analysis"
    echo ""
    echo "📁 Results saved in: ${PHASE_DIR}/"
    echo "📊 Summary: ${PHASE_DIR}/all_checkpoints_summary.csv"
    
else
    echo "💥 PHASE 1A VALIDATION FAILED!"
    echo ""
    echo "❌ Checkpoints did not meet management criteria:"
    echo "   - Not all checkpoints scored ≥0.5 reward, OR"
    echo "   - Some checkpoints fell below 0.3 reward floor"
    echo "   - Early rewards may be statistical noise"
    echo ""
    echo "📋 Recommended Actions:"
    echo "   1. Analyze individual checkpoint performance"
    echo "   2. Consider reward system redesign"
    echo "   3. Skip advanced LR scheduling until foundation is solid"
    echo ""
    echo "📁 Diagnostic data saved in: ${PHASE_DIR}/"
    echo "📊 Analysis: ${PHASE_DIR}/all_checkpoints_summary.csv"
fi

echo ""
echo "⏰ Phase 1A completed at: $(date)"
echo "📝 Full log: ${LOG_FILE}"
echo "============================================================"

exit $validation_result