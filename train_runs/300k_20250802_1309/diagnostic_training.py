#!/usr/bin/env python3
"""
Diagnostic Training - Systematically test for 2K step slowdown causes
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

def main():
    try:
        print("🔍 DIAGNOSTIC TRAINING - FINDING 2K STEP SLOWDOWN CAUSE")
        print("=" * 70)
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Configuration for rapid testing
        config = {
            'checkpoint_path': 'train_runs/300k_20250802_1309/checkpoints/chunk5_final_212848steps.zip',
            'chunk_size': 4000,  # Larger chunk to see slowdown
            'save_dir': 'train_runs/300k_20250802_1309/diagnostics',
            'test_mode': 'cuda_sync'  # Change this to test different issues
        }
        
        # Create diagnostics directory
        os.makedirs(config['save_dir'], exist_ok=True)
        
        print(f"🧪 Test mode: {config['test_mode']}")
        print(f"📦 Checkpoint: {config['checkpoint_path']}")
        print(f"🎯 Chunk size: {config['chunk_size']:,} steps")
        
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
        
        # Diagnostic callback
        class DiagnosticCallback(BaseCallback):
            def __init__(self, chunk_size, test_mode):
                super().__init__()
                self.chunk_size = chunk_size
                self.test_mode = test_mode
                self.start_steps = 0
                self.step_times = []
                self.last_time = time.perf_counter()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                print(f"🔬 Starting diagnostic training from step {self.start_steps:,}")
                print(f"🧪 Test mode: {self.test_mode}")
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Apply diagnostic fixes based on test mode
                if self.test_mode == 'cuda_sync' and steps_in_chunk % 1000 == 0:
                    # Test 1: CUDA synchronization and cache clearing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        print(f"   🔧 CUDA sync + cache clear at step {steps_in_chunk}")
                
                elif self.test_mode == 'gc_manual' and steps_in_chunk % 500 == 0:
                    # Test 5: Manual garbage collection
                    gc.collect()
                    print(f"   🗑️ Manual GC at step {steps_in_chunk}")
                
                # Progress updates with timing info
                if steps_in_chunk % 200 == 0 and steps_in_chunk > 0:
                    recent_times = self.step_times[-200:] if len(self.step_times) >= 200 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    print(f"   📊 Step {steps_in_chunk:,}/{self.chunk_size:,} | "
                          f"Speed: {steps_per_sec:.1f} it/s | "
                          f"Avg step time: {avg_step_time*1000:.2f}ms")
                
                # Stop when chunk is completed
                if steps_in_chunk >= self.chunk_size:
                    print(f"🛑 Diagnostic chunk completed: {steps_in_chunk:,} steps")
                    
                    # Analyze timing patterns
                    self._analyze_timing_patterns()
                    
                    return False
                
                return True
            
            def _analyze_timing_patterns(self):
                print("\n📈 TIMING ANALYSIS:")
                print("=" * 50)
                
                # Split into segments
                segment_size = 500
                segments = []
                for i in range(0, len(self.step_times), segment_size):
                    segment = self.step_times[i:i+segment_size]
                    if segment:
                        avg_time = sum(segment) / len(segment)
                        segments.append(avg_time)
                
                for i, avg_time in enumerate(segments):
                    start_step = i * segment_size
                    end_step = min((i + 1) * segment_size, len(self.step_times))
                    steps_per_sec = 1.0 / avg_time if avg_time > 0 else 0
                    
                    status = "🟢" if steps_per_sec > 300 else "🟡" if steps_per_sec > 200 else "🔴"
                    print(f"   {status} Steps {start_step:4d}-{end_step:4d}: "
                          f"{steps_per_sec:6.1f} it/s ({avg_time*1000:5.2f}ms/step)")
        
        # Create callback
        callback = DiagnosticCallback(
            chunk_size=config['chunk_size'],
            test_mode=config['test_mode']
        )
        
        print(f'🔬 Starting diagnostic training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train diagnostic chunk
        model.learn(
            total_timesteps=config['chunk_size'] + initial_steps,
            callback=callback,
            log_interval=50,  # Less frequent logging
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        final_steps = model.num_timesteps
        steps_trained = final_steps - initial_steps
        
        print(f'\n✅ Diagnostic training completed in {training_time:.1f} seconds')
        print(f'📊 Steps trained: {steps_trained:,}')
        print(f'📊 Average speed: {steps_trained/training_time:.1f} it/s')
        
        print(f'\n🎯 TEST MODE: {config["test_mode"]}')
        print('💡 Try different test modes: cuda_sync, gc_manual, no_checkpoints')
        
    except Exception as e:
        print(f"❌ Diagnostic training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)