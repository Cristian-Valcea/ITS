#!/usr/bin/env python3
"""
Direct 300K Training - Load base model directly and continue training
"""

import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 STARTING DIRECT 300K MODEL TRAINING")
        print("=" * 50)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 50000,  # Start with smaller chunk
            'save_dir': 'train_runs/300k_20250802_1309/models'
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,}")
        
        # Setup environment
        print("📊 Setting up environment...")
        
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        # Load training data
        print('📥 Loading training data from TimescaleDB...')
        training_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2025-08-01', 
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        print(f'✅ Loaded {len(training_data["trading_days"])} trading periods')
        
        # Combine features
        nvda_features = training_data["nvda_features"]
        msft_features = training_data["msft_features"]
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        price_data = training_data["nvda_prices"]
        
        print(f'📐 Combined features shape: {combined_features.shape}')
        
        # Create environment
        env = DualTickerTradingEnvV3(
            processed_feature_data=combined_features,
            price_data=price_data,
            initial_capital=100000.0,
            max_episode_steps=1000,
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,  # Minimal logging
            verbose=False
        )
        
        print('✅ Environment initialized')
        
        # Load base model directly
        print('🤖 Loading base model directly...')
        try:
            # Try RecurrentPPO first
            model = RecurrentPPO.load(config['base_model_path'], env=env)
            print('✅ Loaded as RecurrentPPO')
        except Exception as e:
            print(f'⚠️ RecurrentPPO failed: {e}')
            try:
                # Fallback to regular PPO
                model = PPO.load(config['base_model_path'], env=env)
                print('✅ Loaded as PPO')
            except Exception as e2:
                print(f'❌ Both PPO attempts failed: {e2}')
                raise
        
        # Simple callback
        class DirectCallback(BaseCallback):
            def __init__(self, save_freq=10000, save_path="train_runs/300k_20250802_1309/models"):
                super().__init__()
                self.save_freq = save_freq
                self.save_path = save_path
                os.makedirs(save_path, exist_ok=True)
                
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = f"{self.save_path}/direct_checkpoint_{self.n_calls}_steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Direct checkpoint saved: {checkpoint_path}")
                
                if self.n_calls % 2000 == 0:
                    print(f"📈 Direct progress: {self.n_calls:,}/{config['training_steps']:,} steps")
                
                return True
        
        # Create callback
        callback = DirectCallback()
        
        print('📈 Starting direct training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train directly
        model.learn(
            total_timesteps=config['training_steps'],
            callback=callback,
            log_interval=20,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        print(f'✅ Direct training completed in {training_time:.1f} seconds')
        
        # Save final model
        os.makedirs(config['save_dir'], exist_ok=True)
        final_model_path = f"{config['save_dir']}/direct_300k_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.save(final_model_path)
        print(f'💾 Direct model saved: {final_model_path}')
        
        print('🎉 DIRECT 300K TRAINING COMPLETED!')
        
    except Exception as e:
        print(f"❌ Direct training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)