#!/usr/bin/env python3
"""
Wrapper Investigation - Check what's happening with environment wrappers
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def investigate_wrappers():
    """Investigate the environment wrapper stack"""
    print("🔍 INVESTIGATING ENVIRONMENT WRAPPERS")
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
        
        # Create base environment
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
        
        print(f'✅ Base environment created: {type(base_env).__name__}')
        
        # Load model and see what wrappers get applied
        print('🤖 Loading model and checking wrapper stack...')
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=base_env)
        
        # Investigate the wrapper stack
        print('\n📋 WRAPPER STACK ANALYSIS:')
        current_env = model.env
        wrapper_count = 0
        
        while hasattr(current_env, 'env') or hasattr(current_env, 'envs'):
            wrapper_count += 1
            print(f'   {wrapper_count}. {type(current_env).__name__}')
            
            # Check for specific attributes that might cause issues
            if hasattr(current_env, 'buffer_size'):
                print(f'      📊 Buffer size: {current_env.buffer_size}')
            if hasattr(current_env, 'gamma'):
                print(f'      📊 Gamma: {current_env.gamma}')
            if hasattr(current_env, 'epsilon'):
                print(f'      📊 Epsilon: {current_env.epsilon}')
            if hasattr(current_env, 'clip_obs'):
                print(f'      📊 Clip obs: {current_env.clip_obs}')
            if hasattr(current_env, 'clip_reward'):
                print(f'      📊 Clip reward: {current_env.clip_reward}')
            
            # Move to next wrapper
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'envs'):
                current_env = current_env.envs[0] if current_env.envs else None
                break
            else:
                break
                
            if wrapper_count > 10:  # Safety break
                print('      ⚠️ Too many wrappers, stopping investigation')
                break
        
        print(f'   {wrapper_count + 1}. {type(current_env).__name__} (base)')
        
        # Check if Monitor wrapper is causing issues
        print('\n🔍 CHECKING FOR MONITOR WRAPPER ISSUES:')
        
        # Look for Monitor wrapper specifically
        current_env = model.env
        monitor_found = False
        
        while hasattr(current_env, 'env'):
            if 'Monitor' in type(current_env).__name__:
                monitor_found = True
                print(f'   📋 Monitor wrapper found: {type(current_env).__name__}')
                
                # Check Monitor attributes
                if hasattr(current_env, 'episode_returns'):
                    print(f'      📊 Episode returns length: {len(current_env.episode_returns)}')
                if hasattr(current_env, 'episode_lengths'):
                    print(f'      📊 Episode lengths length: {len(current_env.episode_lengths)}')
                if hasattr(current_env, 'episode_times'):
                    print(f'      📊 Episode times length: {len(current_env.episode_times)}')
                
                # This could be the issue! Monitor might be accumulating data
                break
            
            current_env = current_env.env
        
        if not monitor_found:
            print('   ✅ No Monitor wrapper found')
        
        return monitor_found
        
    except Exception as e:
        print(f"❌ Wrapper investigation failed: {e}")
        traceback.print_exc()
        return False

def test_without_monitor():
    """Test training without Monitor wrapper"""
    print("\n🧪 TEST WITHOUT MONITOR WRAPPER")
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
        
        # Create environment WITHOUT Monitor wrapper
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
        
        # Manually wrap in DummyVecEnv WITHOUT Monitor
        vec_env = DummyVecEnv([lambda: base_env])
        
        print(f'✅ Environment created WITHOUT Monitor wrapper')
        print(f'   📋 Wrapper stack: DummyVecEnv -> {type(base_env).__name__}')
        
        # Create model with this clean environment
        model = RecurrentPPO(
            'MlpLstmPolicy',
            vec_env,
            verbose=0,
            device='cuda'
        )
        
        # Load weights from existing model
        print('🔄 Loading weights from existing checkpoint...')
        existing_model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip')
        
        # Copy the policy weights
        model.policy.load_state_dict(existing_model.policy.state_dict())
        model.num_timesteps = existing_model.num_timesteps
        
        print(f'✅ Weights loaded, starting from step {model.num_timesteps:,}')
        
        # Test callback
        class NoMonitorTestCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f'🔬 Starting no-monitor test from step {self.start_steps:,}')
                
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
        
        callback = NoMonitorTestCallback()
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=2500,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        avg_speed = 2500 / training_time
        
        print(f'\n✅ No-monitor test completed in {training_time:.1f}s')
        print(f'📊 Average speed: {avg_speed:.1f} it/s')
        
        return avg_speed > 250
        
    except Exception as e:
        print(f"❌ No-monitor test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 WRAPPER INVESTIGATION - FINDING THE ENVIRONMENT ISSUE")
    print("=" * 60)
    
    # First investigate the wrapper stack
    monitor_found = investigate_wrappers()
    
    if monitor_found:
        print("\n⚠️ Monitor wrapper found - this could be accumulating data and causing slowdowns!")
        
        # Test without Monitor wrapper
        success = test_without_monitor()
        
        if success:
            print("\n🎉 SUCCESS! The Monitor wrapper was causing the issue!")
            print("💡 Solution: Train without Monitor wrapper or clear its buffers periodically")
        else:
            print("\n🤔 Monitor wrapper wasn't the issue. Need to investigate further.")
    else:
        print("\n🤔 No obvious wrapper issues found. The problem might be deeper.")

if __name__ == "__main__":
    main()