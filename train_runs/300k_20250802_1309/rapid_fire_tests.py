#!/usr/bin/env python3
"""
Rapid-Fire Tests - From the expert checklist
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def test_disable_logging():
    """Test with all logging disabled"""
    print("\n🧪 RAPID TEST 1: DISABLE ALL LOGGING")
    print("=" * 50)
    
    try:
        # Set environment variables to disable logging
        os.environ['DISABLE_LOGGING'] = '1'
        os.environ['PYTHONPATH'] = '/home/cristian/IntradayTrading/ITS'
        
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
        
        # Create environment with ZERO logging
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
        
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        initial_steps = model.num_timesteps
        
        # Ultra minimal callback
        class UltraMinimalCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Only check for the critical slowdown point
                if steps_in_chunk == 2100:
                    recent_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    print(f"   🎯 CRITICAL STEP 2100: {steps_per_sec:.1f} it/s")
                
                if steps_in_chunk >= 2300:
                    return False
                
                return True
        
        callback = UltraMinimalCallback()
        
        start_time = time.time()
        
        # Train with ZERO logging
        model.learn(
            total_timesteps=2300 + initial_steps,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        steps_trained = 2300
        avg_speed = steps_trained / training_time
        
        print(f'✅ No-logging test: {avg_speed:.1f} it/s average')
        
        return avg_speed > 250  # Good if > 250 it/s
        
    except Exception as e:
        print(f"❌ No-logging test failed: {e}")
        return False

def test_single_env():
    """Test with single environment (N_ENVS=1)"""
    print("\n🧪 RAPID TEST 2: SINGLE ENVIRONMENT")
    print("=" * 50)
    
    try:
        # This is already single env in our setup, but let's be explicit
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
        
        # Create single environment
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
        
        print(f'✅ Single environment created')
        
        # Load model but create new one with explicit single env
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        
        # Check if it's actually single env
        print(f"   📋 Environment type: {type(model.env).__name__}")
        
        # Quick test - just run a few steps to see the pattern
        class QuickTestCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Focus on the critical area
                if 2050 <= steps_in_chunk <= 2150 and steps_in_chunk % 10 == 0:
                    recent_times = self.step_times[-10:] if len(self.step_times) >= 10 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    print(f"   📊 Step {steps_in_chunk}: {steps_per_sec:.1f} it/s")
                
                if steps_in_chunk >= 2200:
                    return False
                
                return True
        
        callback = QuickTestCallback()
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=2200 + model.num_timesteps,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        
        print(f'✅ Single env test completed in {training_time:.1f}s')
        
        return True
        
    except Exception as e:
        print(f"❌ Single env test failed: {e}")
        return False

def main():
    print("🚀 RAPID-FIRE TESTS - ISOLATE THE 2K STEP ISSUE")
    print("=" * 60)
    print("Based on expert checklist rapid-fire experiments...")
    
    tests = [
        ("Disable All Logging", test_disable_logging),
        ("Single Environment", test_single_env),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED - Issue may be resolved")
            else:
                print(f"❌ {test_name}: FAILED - Issue persists")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 RAPID-FIRE TEST SUMMARY")
    print(f"{'='*60}")
    
    working_tests = []
    for test_name, result in results.items():
        status = "✅ WORKING" if result else "❌ STILL BROKEN" if result is False else "💥 CRASHED"
        print(f"   {status} {test_name}")
        
        if result:
            working_tests.append(test_name)
    
    if working_tests:
        print(f"\n🎯 POTENTIAL SOLUTIONS:")
        for test in working_tests:
            print(f"   • {test}")
    else:
        print(f"\n🤔 All tests still show the issue. Need deeper investigation.")
        print(f"💡 This suggests the issue is in the core training loop or environment.")

if __name__ == "__main__":
    main()