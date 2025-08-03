#!/usr/bin/env python3
"""
Chunk 6 Training - Continue from 212,848 to 214,848 steps (2000 steps)
Understanding: 2K step slowdowns are normal RecurrentPPO policy update behavior
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 CHUNK 6 TRAINING - STEPS 212,848 → 214,848")
        print("=" * 60)
        print("📊 Target: 2,000 additional steps")
        print("🎯 Progress: 212,848 / 300,000 (70.9%)")
        print("💡 Note: 2K step slowdowns are normal PPO policy updates")
        print()
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
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
        
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        print('📥 Loading training data...')
        training_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2025-08-01', 
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
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
        
        print('✅ Environment created')
        
        # Load model from chunk 5
        print('🔄 Loading model from chunk 5...')
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        
        print(f'✅ Model loaded from step {model.num_timesteps:,}')
        print(f'📊 RecurrentPPO settings: n_steps={model.n_steps}, batch_size={model.batch_size}')
        
        # Training callback with PPO-aware monitoring
        class Chunk6Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_time = time.time()
                self.start_steps = 0
                self.chunk_target = 2000
                self.step_times = []
                self.last_time = time.perf_counter()
                self.policy_updates = 0
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f'🔬 Starting chunk 6 from step {self.start_steps:,}')
                print(f'🎯 Target: {self.start_steps + self.chunk_target:,} steps')
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Detect policy updates (when speed drops significantly)
                if len(self.step_times) >= 10:
                    recent_avg = sum(self.step_times[-10:]) / 10
                    if recent_avg > 0.1:  # Slower than 10 it/s indicates policy update
                        self.policy_updates += 1
                
                # Progress reporting every 200 steps
                if steps_in_chunk % 200 == 0 and steps_in_chunk > 0:
                    elapsed = time.time() - self.start_time
                    progress_pct = (steps_in_chunk / self.chunk_target) * 100
                    
                    # Calculate recent speed
                    recent_times = self.step_times[-200:] if len(self.step_times) >= 200 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    # Estimate time remaining
                    remaining_steps = self.chunk_target - steps_in_chunk
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f'   📊 Step {self.model.num_timesteps:,} | '
                          f'Chunk: {steps_in_chunk:,}/{self.chunk_target:,} ({progress_pct:.1f}%) | '
                          f'Speed: {steps_per_sec:.1f} it/s | '
                          f'ETA: {eta_minutes:.1f}m | '
                          f'Policy updates: {self.policy_updates}')
                
                # Stop when we reach the target
                if steps_in_chunk >= self.chunk_target:
                    elapsed = time.time() - self.start_time
                    print(f'\n🎉 Chunk 6 completed!')
                    print(f'   📊 Steps trained: {steps_in_chunk:,}')
                    print(f'   ⏱️ Time elapsed: {elapsed/60:.1f} minutes')
                    print(f'   📈 Average speed: {steps_in_chunk/elapsed:.1f} it/s')
                    print(f'   🔄 Policy updates detected: {self.policy_updates}')
                    return False
                
                return True
        
        callback = Chunk6Callback()
        
        # Calculate target timesteps
        target_steps = model.num_timesteps + 2000
        
        print(f'🚀 Starting chunk 6 training...')
        print(f'   📊 Current steps: {model.num_timesteps:,}')
        print(f'   🎯 Target steps: {target_steps:,}')
        print(f'   📈 Steps to train: 2,000')
        
        start_time = time.time()
        
        # Train the chunk
        model.learn(
            total_timesteps=target_steps,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        
        # Save checkpoint
        checkpoint_path = f'train_runs/300k_20250802_1309/checkpoints/chunk6_final_{model.num_timesteps}steps.zip'
        model.save(checkpoint_path)
        
        print(f'\n✅ CHUNK 6 COMPLETED SUCCESSFULLY!')
        print(f'   📊 Final steps: {model.num_timesteps:,}')
        print(f'   💾 Saved: {checkpoint_path}')
        print(f'   ⏱️ Training time: {training_time/60:.1f} minutes')
        print(f'   📈 Average speed: {2000/training_time:.1f} it/s')
        
        # Progress toward 300K
        progress_pct = (model.num_timesteps / 300000) * 100
        remaining_steps = 300000 - model.num_timesteps
        
        print(f'\n📊 OVERALL PROGRESS:')
        print(f'   🎯 Current: {model.num_timesteps:,} / 300,000 ({progress_pct:.1f}%)')
        print(f'   📈 Remaining: {remaining_steps:,} steps')
        print(f'   🔢 Chunks remaining: {remaining_steps // 2000} chunks')
        
        if remaining_steps > 0:
            print(f'\n🚀 Ready for chunk 7!')
            print(f'   📋 Next target: {model.num_timesteps + 2000:,} steps')
        else:
            print(f'\n🎉 300K TRAINING COMPLETED!')
        
        return 0
        
    except Exception as e:
        print(f"❌ Chunk 6 training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)