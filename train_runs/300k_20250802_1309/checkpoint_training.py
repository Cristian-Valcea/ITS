#!/usr/bin/env python3
"""
Checkpoint-based Training - Train in small chunks to avoid freezing
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
        print("🚀 STARTING CHECKPOINT-BASED TRAINING")
        print("=" * 60)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'chunk_size': 1800,  # Train 1800 steps, stop before 2000 freeze point
            'save_dir': 'train_runs/300k_20250802_1309/checkpoints',
            'current_chunk': 1  # Start with chunk 1
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Chunk size: {config['chunk_size']:,} steps")
        print(f"📁 Checkpoint dir: {config['save_dir']}")
        
        # Create checkpoint directory
        os.makedirs(config['save_dir'], exist_ok=True)
        
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
        
        # Create environment
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
        
        print('✅ Environment initialized')
        
        # Load base model
        print('🤖 Loading base model...')
        model = RecurrentPPO.load(config['base_model_path'], env=env)
        initial_steps = model.num_timesteps
        print(f'✅ Model loaded - current steps: {initial_steps:,}')
        
        # Checkpoint callback
        class CheckpointCallback(BaseCallback):
            def __init__(self, chunk_size, save_dir, chunk_num):
                super().__init__()
                self.chunk_size = chunk_size
                self.save_dir = save_dir
                self.chunk_num = chunk_num
                self.start_steps = 0
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                print(f"📈 Starting chunk {self.chunk_num} from step {self.start_steps:,}")
                
            def _on_step(self) -> bool:
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                if steps_in_chunk % 200 == 0:
                    print(f"   Chunk progress: {steps_in_chunk:,}/{self.chunk_size:,} steps")
                
                # Stop before reaching chunk size to avoid freeze
                if steps_in_chunk >= self.chunk_size:
                    print(f"🛑 Chunk {self.chunk_num} completed: {steps_in_chunk:,} steps")
                    
                    # Save checkpoint
                    checkpoint_path = f"{self.save_dir}/checkpoint_chunk{self.chunk_num}_{self.model.num_timesteps}steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                    
                    # Stop training
                    return False
                
                return True
        
        # Create callback
        callback = CheckpointCallback(
            chunk_size=config['chunk_size'],
            save_dir=config['save_dir'],
            chunk_num=config['current_chunk']
        )
        
        print(f'📈 Starting chunk {config["current_chunk"]} training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train one chunk
        model.learn(
            total_timesteps=config['chunk_size'] + initial_steps,  # Total target
            callback=callback,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        final_steps = model.num_timesteps
        steps_trained = final_steps - initial_steps
        
        print(f'✅ Chunk training completed in {training_time:.1f} seconds')
        print(f'📊 Steps trained: {steps_trained:,}')
        print(f'📊 Total steps: {final_steps:,}')
        
        # Final checkpoint save
        final_checkpoint = f"{config['save_dir']}/chunk{config['current_chunk']}_final_{final_steps}steps"
        model.save(final_checkpoint)
        print(f'💾 Final checkpoint: {final_checkpoint}')
        
        print('🎉 CHUNK TRAINING COMPLETED SUCCESSFULLY!')
        print(f'💡 Next: Run again to train chunk {config["current_chunk"] + 1}')
        
    except Exception as e:
        print(f"❌ Chunk training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)