#!/usr/bin/env python3
"""
Monitor Bypass Test - Explicitly prevent Monitor wrapper from being applied
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def test_without_monitor_explicit():
    """Test by explicitly preventing Monitor wrapper"""
    print("🧪 EXPLICIT MONITOR BYPASS TEST")
    print("=" * 50)
    
    try:
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
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
        base_env = DualTickerTradingEnv(
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
        
        # HACK: Monkey patch to prevent Monitor wrapper
        import stable_baselines3.common.monitor
        original_monitor = stable_baselines3.common.monitor.Monitor
        
        class NoOpMonitor:
            """No-op Monitor that just returns the original environment"""
            def __init__(self, env, *args, **kwargs):
                print("   🚫 Monitor wrapper bypassed!")
                return env
        
        # Replace Monitor with no-op
        stable_baselines3.common.monitor.Monitor = lambda env, *args, **kwargs: env
        
        try:
            # Now load the model - it should NOT apply Monitor wrapper
            print('🤖 Loading model with Monitor bypass...')
            model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=base_env)
            
            print(f'✅ Model loaded without Monitor wrapper')
            print(f'   📋 Environment type: {type(model.env).__name__}')
            
            # Test callback
            class BypassTestCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.start_steps = 0
                    self.step_times = []
                    self.last_time = time.perf_counter()
                    
                def _on_training_start(self):
                    self.start_steps = self.model.num_timesteps
                    self.last_time = time.perf_counter()
                    print(f'🔬 Starting bypass test from step {self.start_steps:,}')
                    
                def _on_step(self) -> bool:
                    current_time = time.perf_counter()
                    step_time = current_time - self.last_time
                    self.step_times.append(step_time)
                    self.last_time = current_time
                    
                    steps_in_chunk = self.model.num_timesteps - self.start_steps
                    
                    # Focus on the critical area
                    if steps_in_chunk % 100 == 0 and steps_in_chunk > 0:
                        recent_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
                        avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                        
                        status = "🔴" if 1900 <= steps_in_chunk <= 2300 else "🟢"
                        print(f'   {status} Step {steps_in_chunk:,}: {steps_per_sec:.1f} it/s')
                    
                    if steps_in_chunk >= 2500:
                        return False
                    
                    return True
            
            callback = BypassTestCallback()
            
            start_time = time.time()
            
            model.learn(
                total_timesteps=2500 + model.num_timesteps,
                callback=callback,
                log_interval=None,
                progress_bar=False,
                reset_num_timesteps=False
            )
            
            training_time = time.time() - start_time
            avg_speed = 2500 / training_time
            
            print(f'\n✅ Bypass test completed in {training_time:.1f}s')
            print(f'📊 Average speed: {avg_speed:.1f} it/s')
            
            return avg_speed > 250
            
        finally:
            # Restore original Monitor
            stable_baselines3.common.monitor.Monitor = original_monitor
        
    except Exception as e:
        print(f"❌ Bypass test failed: {e}")
        traceback.print_exc()
        return False

def test_check_monitor_buffers():
    """Check if Monitor wrapper is accumulating large buffers"""
    print("\n🧪 MONITOR BUFFER CHECK")
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
        
        # Create environment
        base_env = DualTickerTradingEnv(
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
        
        # Load model normally (with Monitor)
        print('🤖 Loading model with Monitor...')
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=base_env)
        
        # Find the Monitor wrapper
        current_env = model.env
        monitor_env = None
        
        # Search through wrapper stack
        for _ in range(10):  # Safety limit
            if hasattr(current_env, 'episode_returns'):
                monitor_env = current_env
                break
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'envs'):
                current_env = current_env.envs[0] if current_env.envs else None
                if current_env is None:
                    break
            else:
                break
        
        if monitor_env:
            print(f'✅ Found Monitor wrapper: {type(monitor_env).__name__}')
            
            # Check buffer sizes
            if hasattr(monitor_env, 'episode_returns'):
                print(f'   📊 Episode returns buffer: {len(monitor_env.episode_returns)} items')
            if hasattr(monitor_env, 'episode_lengths'):
                print(f'   📊 Episode lengths buffer: {len(monitor_env.episode_lengths)} items')
            if hasattr(monitor_env, 'episode_times'):
                print(f'   📊 Episode times buffer: {len(monitor_env.episode_times)} items')
            
            # Check if buffers are large
            total_buffer_size = 0
            if hasattr(monitor_env, 'episode_returns'):
                total_buffer_size += len(monitor_env.episode_returns)
            if hasattr(monitor_env, 'episode_lengths'):
                total_buffer_size += len(monitor_env.episode_lengths)
            if hasattr(monitor_env, 'episode_times'):
                total_buffer_size += len(monitor_env.episode_times)
            
            print(f'   📊 Total buffer items: {total_buffer_size}')
            
            if total_buffer_size > 1000:
                print(f'   ⚠️ Large Monitor buffers detected! This could cause slowdowns.')
                return True
            else:
                print(f'   ✅ Monitor buffers are reasonable size.')
                return False
        else:
            print('   ❌ No Monitor wrapper found')
            return False
        
    except Exception as e:
        print(f"❌ Buffer check failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 MONITOR BYPASS INVESTIGATION")
    print("=" * 60)
    
    # First check if Monitor buffers are large
    large_buffers = test_check_monitor_buffers()
    
    if large_buffers:
        print("\n⚠️ Large Monitor buffers detected!")
    
    # Test without Monitor wrapper
    success = test_without_monitor_explicit()
    
    if success:
        print("\n🎉 SUCCESS! Bypassing Monitor wrapper solved the issue!")
        print("💡 Solution: The Monitor wrapper is causing the 2K step slowdown")
        print("🔧 Fix: Either disable Monitor or clear its buffers periodically")
    else:
        print("\n🤔 Monitor bypass didn't solve the issue completely.")
        print("💡 The issue might be in the Monitor wrapper but not just buffer size")

if __name__ == "__main__":
    main()