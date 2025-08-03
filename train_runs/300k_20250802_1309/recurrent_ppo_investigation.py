#!/usr/bin/env python3
"""
RecurrentPPO Investigation - Check internal buffers and settings
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def investigate_recurrent_ppo_settings():
    """Check RecurrentPPO internal settings and buffer sizes"""
    print("🔍 RECURRENTPPO SETTINGS INVESTIGATION")
    print("=" * 50)
    
    try:
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
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
        
        # Load model
        print('🤖 Loading RecurrentPPO model...')
        model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip', env=env)
        
        print('✅ Model loaded, investigating settings...')
        
        # Check key RecurrentPPO parameters
        print(f'\n📊 RECURRENTPPO PARAMETERS:')
        print(f'   📋 n_steps: {model.n_steps}')  # This is the key one!
        print(f'   📋 batch_size: {model.batch_size}')
        print(f'   📋 n_epochs: {model.n_epochs}')
        print(f'   📋 learning_rate: {model.learning_rate}')
        print(f'   📋 clip_range: {model.clip_range}')
        print(f'   📋 ent_coef: {model.ent_coef}')
        print(f'   📋 vf_coef: {model.vf_coef}')
        print(f'   📋 max_grad_norm: {model.max_grad_norm}')
        
        # Check LSTM-specific parameters
        if hasattr(model.policy, 'lstm_hidden_size'):
            print(f'   📋 lstm_hidden_size: {model.policy.lstm_hidden_size}')
        if hasattr(model.policy, 'n_lstm_layers'):
            print(f'   📋 n_lstm_layers: {model.policy.n_lstm_layers}')
        
        # Check rollout buffer
        if hasattr(model, 'rollout_buffer'):
            buffer = model.rollout_buffer
            print(f'\n📊 ROLLOUT BUFFER:')
            print(f'   📋 buffer_size: {buffer.buffer_size}')
            print(f'   📋 n_envs: {buffer.n_envs}')
            print(f'   📋 obs_shape: {buffer.obs_shape}')
            print(f'   📋 action_dim: {buffer.action_dim}')
            
            # Check if buffer size aligns with our slowdown
            if buffer.buffer_size == 2048:
                print(f'   ⚠️ POTENTIAL MATCH! Buffer size is 2048, close to slowdown at ~2050!')
                return True
            elif abs(buffer.buffer_size - 2048) < 100:
                print(f'   ⚠️ CLOSE MATCH! Buffer size {buffer.buffer_size} is close to 2048')
                return True
            else:
                print(f'   ✅ Buffer size {buffer.buffer_size} doesn\'t match slowdown pattern')
        
        # Check if n_steps matches our pattern
        if model.n_steps == 2048:
            print(f'   ⚠️ POTENTIAL MATCH! n_steps is 2048, matches slowdown pattern!')
            return True
        elif abs(model.n_steps - 2048) < 100:
            print(f'   ⚠️ CLOSE MATCH! n_steps {model.n_steps} is close to 2048')
            return True
        else:
            print(f'   ✅ n_steps {model.n_steps} doesn\'t match slowdown pattern')
        
        return False
        
    except Exception as e:
        print(f"❌ RecurrentPPO investigation failed: {e}")
        traceback.print_exc()
        return False

def test_different_n_steps():
    """Test with different n_steps to see if it affects the slowdown"""
    print("\n🧪 TESTING DIFFERENT N_STEPS VALUES")
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
        
        # Load existing model to get its configuration
        existing_model = RecurrentPPO.load('train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip')
        
        print(f'📋 Original model n_steps: {existing_model.n_steps}')
        
        # Test with different n_steps value
        test_n_steps = 1024  # Half of 2048
        
        print(f'🧪 Testing with n_steps = {test_n_steps}...')
        
        # Create new model with different n_steps
        test_model = RecurrentPPO(
            'MlpLstmPolicy',
            env,
            n_steps=test_n_steps,
            verbose=0,
            device='cuda'
        )
        
        # Copy weights from existing model
        test_model.policy.load_state_dict(existing_model.policy.state_dict())
        test_model.num_timesteps = existing_model.num_timesteps
        
        print(f'✅ Test model created with n_steps = {test_n_steps}')
        
        # Quick test callback
        class NStepsTestCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f'🔬 Starting n_steps test from step {self.start_steps:,}')
                
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
        
        callback = NStepsTestCallback()
        
        start_time = time.time()
        
        test_model.learn(
            total_timesteps=2500,
            callback=callback,
            log_interval=None,
            progress_bar=False,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        avg_speed = 2500 / training_time
        
        print(f'\n✅ N_steps test completed in {training_time:.1f}s')
        print(f'📊 Average speed: {avg_speed:.1f} it/s')
        
        return avg_speed > 250
        
    except Exception as e:
        print(f"❌ N_steps test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 RECURRENTPPO INVESTIGATION")
    print("=" * 60)
    
    # Check RecurrentPPO settings
    potential_match = investigate_recurrent_ppo_settings()
    
    if potential_match:
        print("\n🎯 POTENTIAL CAUSE FOUND: RecurrentPPO buffer/n_steps alignment!")
        print("💡 The slowdown might be caused by rollout buffer collection or LSTM state management")
        
        # Test with different n_steps
        success = test_different_n_steps()
        
        if success:
            print("\n🎉 SUCCESS! Different n_steps resolved the issue!")
            print("💡 Solution: Adjust n_steps parameter to avoid the 2048 boundary")
        else:
            print("\n🤔 Different n_steps didn't solve it completely")
    else:
        print("\n🤔 No obvious RecurrentPPO parameter alignment found")

if __name__ == "__main__":
    main()