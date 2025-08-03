#!/usr/bin/env python3
"""
Chunk 4 Training - 3,000 steps from chunk 3 checkpoint
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
        print("🚀 STARTING CHUNK 4 TRAINING - 3,000 STEPS")
        print("=" * 60)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration
        config = {
            'checkpoint_path': 'train_runs/300k_20250802_1309/checkpoints/chunk3_final_206848steps.zip',
            'chunk_size': 3000,  # Larger chunk - 3,000 steps
            'save_dir': 'train_runs/300k_20250802_1309/checkpoints',
            'current_chunk': 4
        }
        
        print(f"📦 Checkpoint: {config['checkpoint_path']}")
        print(f"🎯 Chunk size: {config['chunk_size']:,} steps")
        print(f"📁 Checkpoint dir: {config['save_dir']}")
        
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
        
        # Load checkpoint from chunk 3
        print(f'🤖 Loading checkpoint from chunk 3...')
        model = RecurrentPPO.load(config['checkpoint_path'], env=env)
        initial_steps = model.num_timesteps
        print(f'✅ Checkpoint loaded - current steps: {initial_steps:,}')
        
        # Checkpoint callback
        class ChunkCallback(BaseCallback):
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
                
                # Progress updates every 300 steps
                if steps_in_chunk % 300 == 0 and steps_in_chunk > 0:
                    print(f"   Chunk progress: {steps_in_chunk:,}/{self.chunk_size:,} steps")
                
                # Stop when chunk is completed
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
        callback = ChunkCallback(
            chunk_size=config['chunk_size'],
            save_dir=config['save_dir'],
            chunk_num=config['current_chunk']
        )
        
        print(f'📈 Starting chunk {config["current_chunk"]} training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train chunk 4
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
        
        print('🎉 CHUNK 4 TRAINING COMPLETED SUCCESSFULLY!')
        print(f'💡 Progress: 201K → {final_steps:,} steps')
        print(f'💡 Total training so far: {final_steps - 201200:,} steps')
        print(f'💡 Remaining to 300K: {300000 - final_steps:,} steps')
        print(f'💡 Next: Run chunk 5 to continue toward 300K')
        
    except Exception as e:
        print(f"❌ Chunk 4 training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)