#!/usr/bin/env python3
"""
Systematic Diagnostic - Test each potential cause from the expert checklist
"""

import os
import sys
import time
import traceback
import gc
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def test_vecnormalize_issue():
    """Test #2: VecNormalize rollover at 2048 steps"""
    print("\n🧪 TEST #2: VecNormalize Buffer Rollover")
    print("=" * 50)
    
    try:
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import VecNormalize
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
        
        # Check if VecNormalize is being used
        print("🔍 Checking for VecNormalize usage...")
        
        # Load model to see if it uses VecNormalize
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        
        # Check the environment wrapper stack
        current_env = model.env
        vec_normalize_found = False
        
        while hasattr(current_env, 'env'):
            print(f"   📋 Found wrapper: {type(current_env).__name__}")
            if isinstance(current_env, VecNormalize):
                vec_normalize_found = True
                print(f"   🎯 VecNormalize found! Buffer size: {getattr(current_env, 'buffer_size', 'unknown')}")
                break
            current_env = current_env.env
        
        if not vec_normalize_found:
            print("   ✅ No VecNormalize found - this is not the issue")
            return False
        else:
            print("   ⚠️ VecNormalize found - could be the 2048 buffer rollover issue!")
            return True
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_disk_io_issue():
    """Test #3: Disk I/O checkpoint writes"""
    print("\n🧪 TEST #3: Disk I/O Checkpoint Writes")
    print("=" * 50)
    
    try:
        # Check if there are checkpoint writes happening around 2K steps
        checkpoint_dir = 'train_runs/300k_20250802_1309/checkpoints'
        
        print(f"🔍 Checking checkpoint directory: {checkpoint_dir}")
        
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            print(f"   📁 Found {len(files)} checkpoint files")
            
            # Look for patterns in checkpoint naming that might indicate 2K step saves
            for file in files:
                if '2000' in file or '2048' in file:
                    print(f"   ⚠️ Found potential 2K checkpoint: {file}")
                    return True
            
            print("   ✅ No 2K step checkpoints found - likely not disk I/O issue")
            return False
        else:
            print("   ✅ No checkpoint directory - not disk I/O issue")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_gc_issue():
    """Test #5: Python GC / object churn"""
    print("\n🧪 TEST #5: Python Garbage Collection")
    print("=" * 50)
    
    try:
        import gc
        
        print("🔍 Checking GC settings and stats...")
        
        # Get current GC stats
        gc_stats = gc.get_stats()
        print(f"   📊 GC generations: {len(gc_stats)}")
        
        for i, stats in enumerate(gc_stats):
            print(f"   📋 Generation {i}: {stats}")
        
        # Check GC thresholds
        thresholds = gc.get_threshold()
        print(f"   🎯 GC thresholds: {thresholds}")
        
        # The default threshold for gen0 is 700 objects
        # If we're creating lots of small objects, this could trigger frequently
        if thresholds[0] < 1000:
            print("   ⚠️ Low GC threshold - could cause frequent collections!")
            return True
        else:
            print("   ✅ GC thresholds seem reasonable")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_minimal_training():
    """Test with minimal settings to isolate the issue"""
    print("\n🧪 MINIMAL TEST: Isolate the 2K step slowdown")
    print("=" * 50)
    
    try:
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
        
        # Create environment with minimal logging
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
            log_trades=False,  # Minimal logging
            verbose=False      # No verbose output
        )
        
        print('✅ Environment created with minimal logging')
        
        # Load model
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        initial_steps = model.num_timesteps
        
        # Minimal callback - just timing
        class MinimalCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f"🔬 Starting minimal test from step {self.start_steps:,}")
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Only log every 100 steps to minimize overhead
                if steps_in_chunk % 100 == 0 and steps_in_chunk > 0:
                    recent_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    # Flag if we're in the danger zone
                    status = "🔴" if 1900 <= steps_in_chunk <= 2200 else "🟢"
                    print(f"   {status} Step {steps_in_chunk:,}: {steps_per_sec:.1f} it/s")
                
                # Stop at 2500 steps to capture the slowdown
                if steps_in_chunk >= 2500:
                    print(f"🛑 Minimal test completed: {steps_in_chunk:,} steps")
                    return False
                
                return True
        
        callback = MinimalCallback()
        
        print(f'🔬 Starting minimal test (2500 steps)...')
        start_time = time.time()
        
        # Train with minimal settings
        model.learn(
            total_timesteps=2500 + initial_steps,
            callback=callback,
            log_interval=None,  # No logging
            progress_bar=False,  # No progress bar
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        steps_trained = 2500
        
        print(f'\n✅ Minimal test completed in {training_time:.1f} seconds')
        print(f'📊 Average speed: {steps_trained/training_time:.1f} it/s')
        
        return True
        
    except Exception as e:
        print(f"   ❌ Minimal test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 SYSTEMATIC DIAGNOSTIC - FINDING THE REAL 2K STEP ISSUE")
    print("=" * 70)
    print("You're right - the CUDA sync didn't actually fix it.")
    print("Let's systematically test the other potential causes...")
    
    # Test each potential cause
    tests = [
        ("VecNormalize Buffer Rollover", test_vecnormalize_issue),
        ("Disk I/O Checkpoint Writes", test_disk_io_issue),
        ("Python Garbage Collection", test_gc_issue),
        ("Minimal Training Test", test_minimal_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING: {test_name}")
        print(f"{'='*70}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"⚠️ {test_name}: POTENTIAL ISSUE FOUND")
            else:
                print(f"✅ {test_name}: Not the issue")
                
        except Exception as e:
            print(f"❌ {test_name}: Test failed - {e}")
            results[test_name] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    
    potential_issues = []
    for test_name, result in results.items():
        status = "⚠️ SUSPECT" if result else "✅ CLEAR" if result is False else "❌ FAILED"
        print(f"   {status} {test_name}")
        
        if result:
            potential_issues.append(test_name)
    
    if potential_issues:
        print(f"\n🎯 LIKELY CULPRITS:")
        for issue in potential_issues:
            print(f"   • {issue}")
    else:
        print(f"\n🤔 No obvious culprits found. May need deeper investigation.")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Focus on the suspect items above")
    print(f"   2. Try the rapid-fire experiments from the checklist")
    print(f"   3. Profile the exact moment of slowdown")

if __name__ == "__main__":
    main()