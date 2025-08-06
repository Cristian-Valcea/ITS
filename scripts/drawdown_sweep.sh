#!/bin/bash

# 🔧 DRAWDOWN SWEEP EXPERIMENT
# Test different drawdown limits to see if 30% is too restrictive
# Run 15K steps each: DD = {0.30, 0.40, 0.50, 0.75}

echo "🔧 Starting Drawdown Sweep Experiment"
echo "=============================================="
echo "Testing drawdown limits: 30%, 40%, 50%, 75%"
echo "Each variant: 15K steps"
echo "Goal: Find if longer runway helps episode survival"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p diagnostic_runs/drawdown_sweep
mkdir -p diagnostic_runs/drawdown_sweep/logs
mkdir -p diagnostic_runs/drawdown_sweep/checkpoints
mkdir -p diagnostic_runs/drawdown_sweep/results

# Create summary file
echo "drawdown_limit,status,mean_reward,mean_episode_length,max_episode_length,total_episodes,aborted" > diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv

# Test each drawdown limit
for DD in 0.30 0.40 0.50 0.75; do
    echo ""
    echo "=================================================="
    echo "Testing Drawdown Limit: ${DD} ($(echo "$DD * 100" | bc -l | cut -d. -f1)%)"
    echo "=================================================="
    
    # Create temporary config with modified drawdown limit
    CONFIG_FILE="config/drawdown_sweep_${DD}.yaml"
    cp config/reward_shim.yaml $CONFIG_FILE
    
    # Update drawdown limit in config
    sed -i "s/max_drawdown_pct: 0.30/max_drawdown_pct: ${DD}/" $CONFIG_FILE
    
    echo "🚀 Training with ${DD} drawdown limit (15K steps)..."
    echo "   Config: $CONFIG_FILE"
    echo "   Checkpoint: diagnostic_runs/drawdown_sweep/checkpoints/dd_${DD}_15k.zip"
    
    # Run training
    LOG_FILE="diagnostic_runs/drawdown_sweep/logs/dd_${DD}_15k.log"
    
    python train_diagnostic.py \
        --config $CONFIG_FILE \
        --total_timesteps 15000 \
        --save_path "diagnostic_runs/drawdown_sweep/checkpoints/dd_${DD}_15k.zip" \
        2>&1 | tee -a $LOG_FILE
    
    TRAIN_STATUS=$?
    
    if [ $TRAIN_STATUS -eq 0 ]; then
        echo "✅ Training completed successfully"
        
        # Run evaluation
        echo "🔍 Evaluating model..."
        
        python evaluate_checkpoint.py \
            --checkpoint "diagnostic_runs/drawdown_sweep/checkpoints/dd_${DD}_15k.zip" \
            --config $CONFIG_FILE \
            --eval-episodes 10 \
            --output-csv "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" \
            --output-json "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}_detailed.json"
        
        EVAL_STATUS=$?
        
        if [ $EVAL_STATUS -eq 0 ]; then
            echo "✅ Evaluation completed"
            
            # Extract metrics from evaluation
            if [ -f "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" ]; then
                MEAN_REWARD=$(tail -n 1 "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" | cut -d',' -f6)
                MEAN_LENGTH=$(tail -n 1 "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" | cut -d',' -f11)
                MAX_LENGTH=$(tail -n 1 "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" | cut -d',' -f14)
                TOTAL_EPISODES=$(tail -n 1 "diagnostic_runs/drawdown_sweep/results/eval_dd_${DD}.csv" | cut -d',' -f4)
                
                echo "${DD},SUCCESS,${MEAN_REWARD},${MEAN_LENGTH},${MAX_LENGTH},${TOTAL_EPISODES},NO" >> diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv
                
                echo "📊 Results for DD=${DD}:"
                echo "   Mean Reward: $MEAN_REWARD"
                echo "   Mean Episode Length: $MEAN_LENGTH"
                echo "   Max Episode Length: $MAX_LENGTH"
                echo "   Total Episodes: $TOTAL_EPISODES"
            else
                echo "${DD},EVAL_FAILED,,,,,NO" >> diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv
            fi
        else
            echo "❌ Evaluation failed"
            echo "${DD},EVAL_FAILED,,,,,NO" >> diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv
        fi
    else
        echo "❌ Training failed"
        echo "${DD},TRAIN_FAILED,,,,,YES" >> diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv
    fi
    
    # Clean up temporary config
    rm -f $CONFIG_FILE
    
    echo "Completed drawdown limit: ${DD}"
done

echo ""
echo "🎯 DRAWDOWN SWEEP COMPLETE!"
echo "=============================================="
echo "📊 Summary results:"
cat diagnostic_runs/drawdown_sweep/drawdown_sweep_summary.csv

echo ""
echo "📈 Next steps:"
echo "1. Analyze results to find optimal drawdown limit"
echo "2. If episode lengths improve, proceed to optimizer mini-grid"
echo "3. If no improvement, investigate data quality issues"

echo ""
echo "✅ All results saved to: diagnostic_runs/drawdown_sweep/"