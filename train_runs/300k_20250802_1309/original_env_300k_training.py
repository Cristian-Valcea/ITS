#!/usr/bin/env python3
"""
300K Training with Original Environment - The one that worked for 201K
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
        print("🚀 STARTING 300K TRAINING WITH ORIGINAL ENVIRONMENT")
        print("=" * 60)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv  # ORIGINAL ENV
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 99000,  # 201K -> 300K
            'save_dir': 'train_runs/300k_20250802_1309/models'
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,}")
        print("🔧 Using ORIGINAL DualTickerTradingEnv (not V3)")
        
        # Setup environment
        print("📊 Setting up ORIGINAL environment...")
        
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
        print(f'📊 NVDA features shape: {training_data["nvda_features"].shape}')
        print(f'📊 MSFT features shape: {training_data["msft_features"].shape}')
        
        # Create ORIGINAL environment (the one that worked for 201K)
        env = DualTickerTradingEnv(
            nvda_data=training_data['nvda_features'],
            msft_data=training_data['msft_features'], 
            nvda_prices=training_data['nvda_prices'],
            msft_prices=training_data['msft_prices'],
            trading_days=training_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=1000,
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        print('✅ ORIGINAL Environment initialized')
        
        # Load base model directly
        print('🤖 Loading 201K base model...')
        model = RecurrentPPO.load(config['base_model_path'], env=env)
        print('✅ 201K model loaded successfully')
        
        # Simple callback
        class OriginalCallback(BaseCallback):
            def __init__(self, save_freq=25000, save_path="train_runs/300k_20250802_1309/models"):
                super().__init__()
                self.save_freq = save_freq
                self.save_path = save_path
                os.makedirs(save_path, exist_ok=True)
                
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = f"{self.save_path}/original_checkpoint_{self.n_calls}_steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Original checkpoint saved: {checkpoint_path}")
                
                if self.n_calls % 5000 == 0:
                    print(f"📈 Original progress: {self.n_calls:,}/{config['training_steps']:,} steps")
                
                return True
        
        # Create callback
        callback = OriginalCallback()
        
        print('📈 Starting training with ORIGINAL environment...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train with original environment
        model.learn(
            total_timesteps=config['training_steps'],
            callback=callback,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from 201K
        )
        
        training_time = time.time() - start_time
        print(f'✅ ORIGINAL training completed in {training_time:.1f} seconds')
        
        # Save final model
        os.makedirs(config['save_dir'], exist_ok=True)
        final_model_path = f"{config['save_dir']}/original_300k_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.save(final_model_path)
        print(f'💾 ORIGINAL 300K model saved: {final_model_path}')
        
        print('🎉 ORIGINAL ENVIRONMENT 300K TRAINING COMPLETED!')
        
    except Exception as e:
        print(f"❌ ORIGINAL training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)