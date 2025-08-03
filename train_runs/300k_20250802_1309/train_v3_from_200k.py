#!/usr/bin/env python3
"""
🎯 TRAINING WITH V3 ENVIRONMENT FROM 200K MODEL
Continue training the 200K model using the improved V3 environment
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def create_v3_training_environment(data_adapter, db_config):
    """Create V3 environment with proper data loading"""
    
    print('📥 Loading training data for V3 environment...')
    
    # Load training data
    training_data = data_adapter.load_training_data(
        start_date='2024-02-01',
        end_date='2025-08-01',
        symbols=['NVDA', 'MSFT'],
        bar_size='1min',
        data_split='train'
    )
    
    print(f'✅ Training data loaded:')
    print(f'   📊 NVDA features: {training_data["nvda_features"].shape}')
    print(f'   📊 MSFT features: {training_data["msft_features"].shape}')
    print(f'   📊 Trading days: {len(training_data["trading_days"])}')
    
    # Prepare data for V3 environment
    # V3 expects combined feature data and price data
    nvda_features = training_data['nvda_features']
    msft_features = training_data['msft_features']
    nvda_prices = training_data['nvda_prices']
    msft_prices = training_data['msft_prices']
    
    # Combine features: [NVDA features, MSFT features]
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    
    # For V3, we need to create a price series that represents the portfolio
    # We'll use NVDA prices as the primary price series for now
    price_series = nvda_prices
    
    print(f'📊 Combined features shape: {combined_features.shape}')
    print(f'📊 Price series length: {len(price_series)}')
    
    return combined_features, price_series, training_data

def setup_v3_environment(combined_features, price_series):
    """Setup the V3 environment with proper parameters"""
    
    from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
    
    # V3 environment with improved reward system
    env = DualTickerTradingEnvV3(
        processed_feature_data=combined_features,
        price_data=price_series,
        initial_capital=100000,
        lookback_window=50,
        max_episode_steps=1000,
        max_daily_drawdown_pct=0.02,
        max_position_size=500,
        transaction_cost_pct=0.0001,  # Additional friction (1bp)
        
        # V3 reward parameters (calibrated)
        base_impact_bp=68.0,          # Calibrated impact strength
        impact_exponent=0.5,          # sqrt scaling
        risk_free_rate_annual=0.05,   # 5% risk-free rate
        
        log_trades=False,
        verbose=True
    )
    
    print('✅ V3 environment created with improved reward system')
    print('   🎯 Key V3 features:')
    print('   - Risk-free baseline to prevent cost-blind trading')
    print('   - Embedded impact costs with Kyle lambda model')
    print('   - Hold bonus for doing nothing when no alpha')
    print('   - Action change penalties to reduce overtrading')
    print('   - Ticket costs and downside penalties')
    
    return env

def load_200k_model_for_v3(env):
    """Load the 200K model and adapt it for V3 environment"""
    
    from sb3_contrib import RecurrentPPO
    
    model_path = 'models/dual_ticker_200k_final_20250731_134508.zip'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"200K model not found at: {model_path}")
    
    print(f'🤖 Loading 200K model: {model_path}')
    
    # Load model with new environment
    model = RecurrentPPO.load(model_path, env=env)
    
    print('✅ 200K model loaded and adapted for V3 environment')
    print(f'   📊 Model has been trained for 200,000 steps')
    print(f'   🔄 Will continue training with V3 reward system')
    
    return model

def setup_training_callbacks():
    """Setup training callbacks for monitoring"""
    
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList
    )
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'train_runs/v3_from_200k_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f'{run_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{run_dir}/logs', exist_ok=True)
    
    # Checkpoint callback - save every 25k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f'{run_dir}/checkpoints',
        name_prefix='v3_model',
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_callback])
    
    print(f'📁 Training run directory: {run_dir}')
    print(f'💾 Checkpoints will be saved every 25,000 steps')
    
    return callbacks, run_dir

def train_v3_model(model, env, total_steps=100000):
    """Train the model with V3 environment"""
    
    print(f'\n🚀 Starting V3 training...')
    print(f'   🎯 Total steps: {total_steps:,}')
    print(f'   🧠 Algorithm: RecurrentPPO')
    print(f'   🌟 Environment: DualTickerTradingEnvV3')
    
    # Setup callbacks
    callbacks, run_dir = setup_training_callbacks()
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from 200K steps
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = f'{run_dir}/v3_model_final_{total_steps}steps.zip'
        model.save(final_model_path)
        
        print(f'\n✅ V3 TRAINING COMPLETED!')
        print(f'   ⏱️ Training time: {training_time/3600:.1f} hours')
        print(f'   📊 Total steps trained: {200000 + total_steps:,}')
        print(f'   💾 Final model saved: {final_model_path}')
        print(f'   📁 Run directory: {run_dir}')
        
        return model, run_dir
        
    except Exception as e:
        print(f'❌ Training failed: {e}')
        traceback.print_exc()
        return None, run_dir

def create_training_summary(run_dir, training_steps, start_model_steps=200000):
    """Create a summary of the training run"""
    
    summary_path = f'{run_dir}/TRAINING_SUMMARY.md'
    
    with open(summary_path, 'w') as f:
        f.write(f"# 🎯 V3 Training Summary\n\n")
        f.write(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Starting Model**: 200K steps (dual_ticker_200k_final_20250731_134508.zip)\n")
        f.write(f"**Environment**: DualTickerTradingEnvV3\n")
        f.write(f"**Additional Steps**: {training_steps:,}\n")
        f.write(f"**Total Steps**: {start_model_steps + training_steps:,}\n\n")
        
        f.write("## 🌟 V3 Environment Improvements\n\n")
        f.write("### Key Features:\n")
        f.write("- **Risk-free baseline**: Prevents cost-blind trading\n")
        f.write("- **Embedded impact costs**: Kyle lambda model with 68bp calibrated impact\n")
        f.write("- **Hold bonus**: Incentivizes doing nothing when no alpha signal\n")
        f.write("- **Action change penalties**: Reduces overtrading behavior\n")
        f.write("- **Ticket costs**: Fixed costs per trade ($25)\n")
        f.write("- **Downside penalties**: Risk management through semi-variance\n")
        f.write("- **Position decay**: Penalties for holding during low-alpha periods\n\n")
        
        f.write("### Reward Formula:\n")
        f.write("```\n")
        f.write("reward = risk_free_nav_change\n")
        f.write("       - embedded_impact\n") 
        f.write("       - downside_penalty\n")
        f.write("       + kelly_bonus\n")
        f.write("       - position_decay_penalty\n")
        f.write("       - turnover_penalty\n")
        f.write("       - size_penalty\n")
        f.write("       + hold_bonus\n")
        f.write("       - action_change_penalty\n")
        f.write("       - ticket_cost\n")
        f.write("```\n\n")
        
        f.write("## 🎯 Expected Improvements\n\n")
        f.write("Based on the V3 reward system design, we expect:\n")
        f.write("1. **Reduced overtrading** - Hold bonus and action change penalties\n")
        f.write("2. **Better profitability** - Risk-free baseline and proper cost modeling\n")
        f.write("3. **Improved risk management** - Downside penalties and position limits\n")
        f.write("4. **More realistic trading** - Embedded impact costs and ticket fees\n\n")
        
        f.write("## 📁 Files\n\n")
        f.write("- **Checkpoints**: `checkpoints/` - Model saves every 25K steps\n")
        f.write("- **Final Model**: `v3_model_final_*steps.zip`\n")
        f.write("- **Training Logs**: `logs/`\n")
        f.write("- **This Summary**: `TRAINING_SUMMARY.md`\n\n")
        
        f.write("## 🚀 Next Steps\n\n")
        f.write("After training completion:\n")
        f.write("1. **Evaluate performance** on validation data\n")
        f.write("2. **Compare with original models** (200K and 300K)\n")
        f.write("3. **Analyze trading behavior** changes\n")
        f.write("4. **Test profitability** improvements\n")
    
    print(f'📋 Training summary created: {summary_path}')

def main():
    try:
        print("🎯 V3 ENVIRONMENT TRAINING FROM 200K MODEL")
        print("=" * 60)
        print("🌟 Using improved V3 environment with better reward system")
        print()
        
        # Import after path setup
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from secrets_helper import SecretsHelper
        
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        # Create data adapter
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        # Create V3 environment
        combined_features, price_series, training_data = create_v3_training_environment(
            data_adapter, db_config
        )
        
        env = setup_v3_environment(combined_features, price_series)
        
        # Load 200K model
        model = load_200k_model_for_v3(env)
        
        # Train with V3 environment
        training_steps = 100000  # Train for additional 100K steps
        trained_model, run_dir = train_v3_model(model, env, training_steps)
        
        if trained_model is not None:
            # Create training summary
            create_training_summary(run_dir, training_steps)
            
            print(f'\n🎉 SUCCESS!')
            print(f'   📊 Model trained from 200K to {200000 + training_steps:,} steps')
            print(f'   🌟 Using improved V3 environment')
            print(f'   📁 Results in: {run_dir}')
            print(f'\n🚀 Next: Evaluate the V3-trained model performance!')
            
            return 0
        else:
            print(f'❌ Training failed')
            return 1
        
    except Exception as e:
        print(f"❌ V3 training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)