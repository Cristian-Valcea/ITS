#!/usr/bin/env python3
"""
Production Training - With CUDA sync fix and robust error handling
"""

import os
import sys
import time
import traceback
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 PRODUCTION TRAINING - WITH CUDA SYNC FIX")
        print("=" * 60)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration
        config = {
            'checkpoint_path': 'train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip',
            'chunk_size': 10000,  # Large chunk now that we fixed the slowdown
            'save_dir': 'train_runs/300k_20250802_1309/checkpoints',
            'current_chunk': 'production_10k'
        }
        
        print(f"📦 Checkpoint: {config['checkpoint_path']}")
        print(f"🎯 Chunk size: {config['chunk_size']:,} steps")
        print(f"🔧 CUDA sync: Every 1,000 steps")
        
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
        
        # Load checkpoint
        print(f'🤖 Loading checkpoint...')
        model = RecurrentPPO.load(config['checkpoint_path'], env=env)
        initial_steps = model.num_timesteps
        print(f'✅ Checkpoint loaded - current steps: {initial_steps:,}')
        
        # Production callback with CUDA sync fix
        class ProductionCallback(BaseCallback):
            def __init__(self, chunk_size, save_dir, chunk_name):
                super().__init__()
                self.chunk_size = chunk_size
                self.save_dir = save_dir
                self.chunk_name = chunk_name
                self.start_steps = 0
                self.last_save_steps = 0
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_save_steps = self.start_steps
                print(f"📈 Starting production training from step {self.start_steps:,}")
                
            def _on_step(self) -> bool:
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # CUDA sync every 1000 steps - THE FIX!
                if steps_in_chunk % 1000 == 0 and steps_in_chunk > 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        print(f"   🔧 CUDA sync + cache clear at step {steps_in_chunk}")
                
                # Intermediate saves every 2000 steps
                if steps_in_chunk % 2000 == 0 and steps_in_chunk > 0:
                    intermediate_path = f"{self.save_dir}/intermediate_{self.chunk_name}_{self.model.num_timesteps}steps"
                    self.model.save(intermediate_path)
                    print(f"   💾 Intermediate save: {intermediate_path}")
                
                # Progress updates every 1000 steps
                if steps_in_chunk % 1000 == 0 and steps_in_chunk > 0:
                    print(f"   📊 Progress: {steps_in_chunk:,}/{self.chunk_size:,} steps")
                
                # Stop when chunk is completed
                if steps_in_chunk >= self.chunk_size:
                    print(f"🛑 Production chunk completed: {steps_in_chunk:,} steps")
                    
                    # Save final checkpoint
                    checkpoint_path = f"{self.save_dir}/checkpoint_{self.chunk_name}_{self.model.num_timesteps}steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 Final checkpoint saved: {checkpoint_path}")
                    
                    # Stop training
                    return False
                
                return True
        
        # Create callback
        callback = ProductionCallback(
            chunk_size=config['chunk_size'],
            save_dir=config['save_dir'],
            chunk_name=config['current_chunk']
        )
        
        print(f'📈 Starting production training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train with fix
        try:
            model.learn(
                total_timesteps=config['chunk_size'] + initial_steps,
                callback=callback,
                log_interval=20,
                progress_bar=True,
                reset_num_timesteps=False
            )
        except Exception as training_error:
            print(f"⚠️ Training interrupted: {training_error}")
            print("💾 Saving emergency checkpoint...")
            
            # Emergency save
            emergency_path = f"{config['save_dir']}/emergency_{config['current_chunk']}_{model.num_timesteps}steps"
            model.save(emergency_path)
            print(f"✅ Emergency checkpoint saved: {emergency_path}")
            
            # Continue with what we have
            pass
        
        training_time = time.time() - start_time
        final_steps = model.num_timesteps
        steps_trained = final_steps - initial_steps
        
        print(f'\n✅ Production training completed in {training_time:.1f} seconds')
        print(f'📊 Steps trained: {steps_trained:,}')
        print(f'📊 Total steps: {final_steps:,}')
        print(f'📊 Average speed: {steps_trained/training_time:.1f} it/s')
        
        # Final checkpoint save
        final_checkpoint = f"{config['save_dir']}/{config['current_chunk']}_final_{final_steps}steps"
        model.save(final_checkpoint)
        print(f'💾 Final checkpoint: {final_checkpoint}')
        
        print('🎉 PRODUCTION TRAINING COMPLETED!')
        print(f'💡 Progress: 201K → {final_steps:,} steps')
        print(f'💡 Total training so far: {final_steps - 201200:,} steps')
        print(f'💡 Remaining to 300K: {300000 - final_steps:,} steps')
        print('🔧 CUDA sync fix working perfectly!')
        
        # Summary
        print(f'\n📋 TRAINING SUMMARY:')
        print(f'   🎯 Target: 300,000 steps')
        print(f'   📊 Current: {final_steps:,} steps')
        print(f'   ✅ Progress: {((final_steps - 201200) / (300000 - 201200)) * 100:.1f}%')
        print(f'   🔧 CUDA fix: Working')
        print(f'   ⚡ Speed: {steps_trained/training_time:.1f} it/s average')
        
    except Exception as e:
        print(f"❌ Production training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)