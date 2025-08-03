#!/usr/bin/env python3
"""
Robust 300K Training Script with Better Error Handling
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
        print("🚀 STARTING ROBUST 300K MODEL TRAINING")
        print("=" * 50)
        
        # Import after path setup
        from src.training.dual_ticker_model_adapter import DualTickerModelAdapter
        from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        import torch
        
        # Configuration
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 99000,  # 201K -> 300K
            'save_dir': 'train_runs/300k_20250802_1309/models',
            'checkpoint_freq': 25000,
            'log_freq': 1000
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,}")
        
        # Initialize model adapter
        print("🔧 Initializing model adapter...")
        adapter = DualTickerModelAdapter(config['base_model_path'])
        print("✅ Model adapter initialized")
        
        # Setup environment with V3
        print("📊 Setting up V3 environment...")
        
        # Database config for live data
        from secrets_helper import SecretsHelper
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        # Load training data from TimescaleDB
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
        
        # Combine features for V3 environment (expects single processed_feature_data array)
        # V3 format: [N, 26] where 26 = 12 NVDA + 1 NVDA_pos + 12 MSFT + 1 MSFT_pos
        nvda_features = training_data["nvda_features"]  # [N, 12]
        msft_features = training_data["msft_features"]  # [N, 12]
        
        # Combine features: [NVDA_12_features, MSFT_12_features] = [N, 24]
        # Note: V3 environment will add position features internally
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        
        # Use NVDA prices as primary price data (V3 environment expects single price series)
        price_data = training_data["nvda_prices"]
        
        print(f'📐 Combined features shape: {combined_features.shape}')
        
        # Create V3 environment
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
            log_trades=False,  # Reduce logging for stability
            verbose=False
        )
        print("✅ V3 Environment initialized")
        
        # Create adapted model
        print("🤖 Creating adapted model...")
        model = adapter.prepare_dual_ticker_model(env)
        print("✅ Adapted model created")
        
        # Simple callback for checkpoints
        class CheckpointCallback(BaseCallback):
            def __init__(self, save_freq, save_path, verbose=1):
                super().__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
                
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = f"{self.save_path}/checkpoint_{self.n_calls}_steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                return True
        
        # Setup callbacks
        os.makedirs(config['save_dir'], exist_ok=True)
        callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=config['save_dir']
        )
        
        # Start training
        print("📈 Starting training...")
        print(f"⏰ Start time: {datetime.now()}")
        
        start_time = time.time()
        
        # Train with progress tracking
        model.learn(
            total_timesteps=config['training_steps'],
            callback=callback,
            log_interval=config['log_freq'] // 100,  # Less frequent logging
            progress_bar=True,
            reset_num_timesteps=False  # Continue from base model steps
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Save final model
        final_model_path = f"{config['save_dir']}/dual_ticker_300k_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        print("🎉 300K TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)