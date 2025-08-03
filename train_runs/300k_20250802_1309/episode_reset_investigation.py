#!/usr/bin/env python3
"""
Episode Reset Investigation - Check if episodes are resetting around step 2100
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def investigate_episode_resets():
    """Check if episodes are resetting around the problematic step count"""
    print("🔍 EPISODE RESET INVESTIGATION")
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
        env = DualTickerTradingEnv(
            nvda_data=training_data['nvda_features'],
            msft_data=training_data['msft_features'], 
            nvda_prices=training_data['nvda_prices'],
            msft_prices=training_data['msft_prices'],
            trading_days=training_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=1000,  # This is key!
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        print(f'✅ Environment created with max_episode_steps: {env.max_steps}')
        
        # Load model
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        
        # Episode tracking callback
        class EpisodeTrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.episode_count = 0
                self.episode_starts = []
                self.episode_ends = []
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f'🔬 Starting episode tracking from step {self.start_steps:,}')
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Check for episode resets by monitoring the environment
                # In SB3, episode resets happen automatically when done=True
                
                # Log every 50 steps with episode info
                if steps_in_chunk % 50 == 0 and steps_in_chunk > 0:
                    recent_times = self.step_times[-50:] if len(self.step_times) >= 50 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    # Try to get episode info from the environment
                    try:
                        # Access the base environment
                        base_env = self.model.env.envs[0] if hasattr(self.model.env, 'envs') else self.model.env
                        if hasattr(base_env, 'env'):
                            base_env = base_env.env  # Unwrap Monitor if present
                        
                        current_episode_step = getattr(base_env, 'current_step', 'unknown')
                        max_episode_steps = getattr(base_env, 'max_steps', 'unknown')
                        
                        status = "🔴" if 1900 <= steps_in_chunk <= 2300 else "🟢"
                        print(f'   {status} Global step {steps_in_chunk:,}: {steps_per_sec:.1f} it/s | '
                              f'Episode step: {current_episode_step}/{max_episode_steps}')
                        
                        # Check if we're near episode boundary
                        if isinstance(current_episode_step, int) and isinstance(max_episode_steps, int):
                            if current_episode_step > max_episode_steps - 50:
                                print(f'      ⚠️ Near episode end! Step {current_episode_step}/{max_episode_steps}')
                        
                    except Exception as e:
                        status = "🔴" if 1900 <= steps_in_chunk <= 2300 else "🟢"
                        print(f'   {status} Global step {steps_in_chunk:,}: {steps_per_sec:.1f} it/s | Episode info unavailable')
                
                if steps_in_chunk >= 2500:
                    return False
                
                return True
        
        callback = EpisodeTrackingCallback()
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=2500 + model.num_timesteps,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        
        print(f'\n✅ Episode tracking completed in {training_time:.1f}s')
        
        return True
        
    except Exception as e:
        print(f"❌ Episode tracking failed: {e}")
        traceback.print_exc()
        return False

def check_episode_length_settings():
    """Check what the actual episode length settings are"""
    print("\n🔍 EPISODE LENGTH SETTINGS CHECK")
    print("=" * 50)
    
    try:
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
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
        
        # Create environment with default settings
        env = DualTickerTradingEnv(
            nvda_data=training_data['nvda_features'],
            msft_data=training_data['msft_features'], 
            nvda_prices=training_data['nvda_prices'],
            msft_prices=training_data['msft_prices'],
            trading_days=training_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=1000,  # This is the key parameter!
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        print(f'📊 Environment settings:')
        print(f'   📋 max_episode_steps: {env.max_steps}')
        print(f'   📋 Total data length: {len(training_data["trading_days"])}')
        print(f'   📋 NVDA data length: {len(training_data["nvda_features"])}')
        print(f'   📋 MSFT data length: {len(training_data["msft_features"])}')
        
        # Calculate how many episodes would fit in our training
        total_training_steps = 212848  # Current step count
        episodes_so_far = total_training_steps // env.max_steps
        steps_in_current_episode = total_training_steps % env.max_steps
        
        print(f'\n📊 Episode analysis:')
        print(f'   📋 Episodes completed so far: {episodes_so_far}')
        print(f'   📋 Steps in current episode: {steps_in_current_episode}')
        print(f'   📋 Next episode reset at global step: {(episodes_so_far + 1) * env.max_steps}')
        
        # Check if the slowdown aligns with episode boundaries
        next_reset = (episodes_so_far + 1) * env.max_steps
        slowdown_step = 212848 + 2100  # Where we expect the slowdown
        
        print(f'\n🎯 Slowdown analysis:')
        print(f'   📋 Expected slowdown at global step: {slowdown_step}')
        print(f'   📋 Next episode reset at global step: {next_reset}')
        print(f'   📋 Difference: {abs(slowdown_step - next_reset)} steps')
        
        if abs(slowdown_step - next_reset) < 100:
            print(f'   ⚠️ POTENTIAL MATCH! Slowdown aligns with episode reset!')
            return True
        else:
            print(f'   ✅ No alignment with episode reset')
            return False
        
    except Exception as e:
        print(f"❌ Episode length check failed: {e}")
        return False

def main():
    print("🔍 EPISODE RESET INVESTIGATION")
    print("=" * 60)
    
    # Check episode length settings first
    episode_alignment = check_episode_length_settings()
    
    if episode_alignment:
        print("\n🎯 POTENTIAL CAUSE FOUND: Episode resets!")
        print("💡 The slowdown might be caused by episode resets and LSTM state resets")
    
    # Run detailed episode tracking
    investigate_episode_resets()

if __name__ == "__main__":
    main()