#!/usr/bin/env python3
"""
Conservative 300K Training Script - Smaller batches, frequent saves
"""

import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime
import gc
import torch

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 STARTING CONSERVATIVE 300K MODEL TRAINING")
        print("=" * 50)
        
        # Import after path setup
        from src.training.dual_ticker_model_adapter import DualTickerModelAdapter
        from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from secrets_helper import SecretsHelper
        
        # Conservative configuration
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 99000,  # 201K -> 300K
            'save_dir': 'train_runs/300k_20250802_1309/models',
            'checkpoint_freq': 5000,  # More frequent saves
            'batch_size': 32,  # Smaller batch size
            'n_steps': 1024,  # Smaller rollout buffer
            'log_freq': 500
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,}")
        print(f"🔧 Conservative settings: batch_size={config['batch_size']}, n_steps={config['n_steps']}")
        
        # Initialize model adapter
        print("🔧 Initializing model adapter...")
        adapter = DualTickerModelAdapter(config['base_model_path'])
        print("✅ Model adapter initialized")
        
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
        print('📥 Loading training data...')
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
        
        # Create environment with conservative settings
        env = DualTickerTradingEnvV3(
            processed_feature_data=combined_features,
            price_data=price_data,
            initial_capital=100000.0,
            max_episode_steps=500,  # Shorter episodes
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        print("✅ Environment initialized")
        
        # Create adapted model with conservative settings
        print("🤖 Creating adapted model...")
        model = adapter.prepare_dual_ticker_model(env)
        
        # Override with conservative hyperparameters
        model.n_steps = config['n_steps']
        model.batch_size = config['batch_size']
        
        print("✅ Adapted model created")
        
        # Conservative callback with frequent saves and memory cleanup
        class ConservativeCallback(BaseCallback):
            def __init__(self, save_freq, save_path, verbose=1):
                super().__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
                self.last_save_step = 0
                
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = f"{self.save_path}/checkpoint_{self.n_calls}_steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    self.last_save_step = self.n_calls
                
                # Progress update every 1000 steps
                if self.n_calls % 1000 == 0:
                    print(f"📈 Progress: {self.n_calls:,}/99,000 steps ({self.n_calls/990:.1f}%)")
                
                return True
        
        # Setup callbacks
        os.makedirs(config['save_dir'], exist_ok=True)
        callback = ConservativeCallback(
            save_freq=config['checkpoint_freq'],
            save_path=config['save_dir']
        )
        
        # Start training with conservative approach
        print("📈 Starting conservative training...")
        print(f"⏰ Start time: {datetime.now()}")
        
        start_time = time.time()
        
        # Train in smaller chunks to avoid memory issues
        chunk_size = 10000  # Train 10K steps at a time
        total_trained = 0
        
        while total_trained < config['training_steps']:
            remaining_steps = config['training_steps'] - total_trained
            current_chunk = min(chunk_size, remaining_steps)
            
            print(f"🔄 Training chunk: {current_chunk:,} steps (total: {total_trained:,}/{config['training_steps']:,})")
            
            try:
                model.learn(
                    total_timesteps=current_chunk,
                    callback=callback,
                    log_interval=10,
                    progress_bar=True,
                    reset_num_timesteps=False
                )
                
                total_trained += current_chunk
                
                # Save after each chunk
                chunk_save_path = f"{config['save_dir']}/chunk_{total_trained}_steps"
                model.save(chunk_save_path)
                print(f"💾 Chunk completed, saved: {chunk_save_path}")
                
                # Memory cleanup between chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Chunk training failed: {e}")
                # Save current progress before failing
                emergency_save = f"{config['save_dir']}/emergency_save_{total_trained}_steps"
                model.save(emergency_save)
                print(f"💾 Emergency save: {emergency_save}")
                raise
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Save final model
        final_model_path = f"{config['save_dir']}/dual_ticker_300k_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        print("🎉 CONSERVATIVE 300K TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)