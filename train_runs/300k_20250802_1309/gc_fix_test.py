#!/usr/bin/env python3
"""
GC Fix Test - Disable automatic garbage collection during training
"""

import os
import sys
import time
import traceback
import gc
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🧪 GC FIX TEST - DISABLE AUTOMATIC GARBAGE COLLECTION")
        print("=" * 60)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # DISABLE AUTOMATIC GARBAGE COLLECTION
        print("🗑️ Disabling automatic garbage collection...")
        gc.disable()
        print(f"   ✅ GC disabled. Current thresholds: {gc.get_threshold()}")
        
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
        
        # Load model
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        initial_steps = model.num_timesteps
        
        # GC Fix callback
        class GCFixCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                self.gc_collections = 0
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f"🔬 Starting GC fix test from step {self.start_steps:,}")
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Manual GC every 500 steps instead of automatic
                if steps_in_chunk % 500 == 0 and steps_in_chunk > 0:
                    print(f"   🗑️ Manual GC at step {steps_in_chunk}")
                    collected = gc.collect()
                    self.gc_collections += 1
                    print(f"      Collected {collected} objects")
                
                # Log every 100 steps with focus on the danger zone
                if steps_in_chunk % 100 == 0 and steps_in_chunk > 0:
                    recent_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    # Flag the danger zone
                    status = "🔴" if 1900 <= steps_in_chunk <= 2300 else "🟢"
                    print(f"   {status} Step {steps_in_chunk:,}: {steps_per_sec:.1f} it/s")
                
                # Stop at 2500 steps
                if steps_in_chunk >= 2500:
                    print(f"🛑 GC fix test completed: {steps_in_chunk:,} steps")
                    print(f"🗑️ Total manual GC collections: {self.gc_collections}")
                    return False
                
                return True
        
        callback = GCFixCallback()
        
        print(f'🔬 Starting GC fix test (2500 steps)...')
        start_time = time.time()
        
        # Train with disabled GC
        model.learn(
            total_timesteps=2500 + initial_steps,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        steps_trained = 2500
        
        print(f'\n✅ GC fix test completed in {training_time:.1f} seconds')
        print(f'📊 Average speed: {steps_trained/training_time:.1f} it/s')
        
        # Re-enable GC
        print("🗑️ Re-enabling garbage collection...")
        gc.enable()
        
        print('\n🎯 RESULT:')
        if steps_trained/training_time > 300:
            print('✅ GC FIX WORKED! No slowdown detected.')
        else:
            print('❌ GC fix did not solve the issue.')
        
    except Exception as e:
        print(f"❌ GC fix test failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        
        # Make sure to re-enable GC
        gc.enable()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)