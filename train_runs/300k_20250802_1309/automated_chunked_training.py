#!/usr/bin/env python3
"""
Automated Chunked Training - Continue training in 2000-step chunks until 300K
Automatically detects current progress and continues from there
"""

import os
import sys
import time
import traceback
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def find_latest_checkpoint():
    """Find the latest checkpoint file and extract step count"""
    checkpoint_dir = 'train_runs/300k_20250802_1309/checkpoints'
    
    # Look for chunk*_final_*steps.zip files
    pattern = os.path.join(checkpoint_dir, 'chunk*_final_*steps.zip')
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return None, 0
    
    # Extract step counts and find the latest
    latest_file = None
    latest_steps = 0
    
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        # Extract steps from filename like "chunk6_final_214848steps.zip"
        try:
            steps_part = filename.split('_final_')[1].split('steps.zip')[0]
            steps = int(steps_part)
            if steps > latest_steps:
                latest_steps = steps
                latest_file = file_path
        except (IndexError, ValueError):
            continue
    
    return latest_file, latest_steps

def get_next_chunk_number():
    """Determine the next chunk number based on existing checkpoints"""
    checkpoint_dir = 'train_runs/300k_20250802_1309/checkpoints'
    pattern = os.path.join(checkpoint_dir, 'chunk*_final_*steps.zip')
    checkpoint_files = glob.glob(pattern)
    
    max_chunk = 0
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        try:
            chunk_part = filename.split('chunk')[1].split('_final_')[0]
            chunk_num = int(chunk_part)
            max_chunk = max(max_chunk, chunk_num)
        except (IndexError, ValueError):
            continue
    
    return max_chunk + 1

def train_chunk(chunk_number, current_steps, target_steps, checkpoint_path):
    """Train a single chunk"""
    try:
        print(f"\n🚀 CHUNK {chunk_number} TRAINING - STEPS {current_steps:,} → {target_steps:,}")
        print("=" * 70)
        print(f"📊 Target: {target_steps - current_steps:,} additional steps")
        print(f"🎯 Progress: {current_steps:,} / 300,000 ({current_steps/300000*100:.1f}%)")
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
        
        # Load model from previous checkpoint
        print(f'🔄 Loading model from {checkpoint_path}...')
        model = RecurrentPPO.load(checkpoint_path, env=env)
        
        print(f'✅ Model loaded from step {model.num_timesteps:,}')
        
        # Training callback
        class AutomatedChunkCallback(BaseCallback):
            def __init__(self, chunk_num, chunk_target):
                super().__init__()
                self.chunk_num = chunk_num
                self.start_time = time.time()
                self.start_steps = 0
                self.chunk_target = chunk_target
                self.step_times = []
                self.last_time = time.perf_counter()
                self.last_report_time = time.time()
                
            def _on_training_start(self):
                self.start_steps = self.model.num_timesteps
                self.last_time = time.perf_counter()
                self.last_report_time = time.time()
                print(f'🔬 Starting chunk {self.chunk_num} from step {self.start_steps:,}')
                print(f'🎯 Target: {self.start_steps + self.chunk_target:,} steps')
                
            def _on_step(self) -> bool:
                current_time = time.perf_counter()
                step_time = current_time - self.last_time
                self.step_times.append(step_time)
                self.last_time = current_time
                
                steps_in_chunk = self.model.num_timesteps - self.start_steps
                
                # Progress reporting every 30 seconds or every 500 steps
                time_since_report = time.time() - self.last_report_time
                if (time_since_report >= 30 or steps_in_chunk % 500 == 0) and steps_in_chunk > 0:
                    self.last_report_time = time.time()
                    
                    elapsed = time.time() - self.start_time
                    progress_pct = (steps_in_chunk / self.chunk_target) * 100
                    
                    # Calculate recent speed
                    recent_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
                    avg_step_time = sum(recent_times) / len(recent_times) if recent_times else 0
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    # Estimate time remaining
                    remaining_steps = self.chunk_target - steps_in_chunk
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f'   📊 Step {self.model.num_timesteps:,} | '
                          f'Chunk: {steps_in_chunk:,}/{self.chunk_target:,} ({progress_pct:.1f}%) | '
                          f'Speed: {steps_per_sec:.1f} it/s | '
                          f'ETA: {eta_minutes:.1f}m')
                
                # Stop when we reach the target
                if steps_in_chunk >= self.chunk_target:
                    elapsed = time.time() - self.start_time
                    print(f'\n🎉 Chunk {self.chunk_num} completed!')
                    print(f'   📊 Steps trained: {steps_in_chunk:,}')
                    print(f'   ⏱️ Time elapsed: {elapsed/60:.1f} minutes')
                    print(f'   📈 Average speed: {steps_in_chunk/elapsed:.1f} it/s')
                    return False
                
                return True
        
        callback = AutomatedChunkCallback(chunk_number, target_steps - current_steps)
        
        print(f'🚀 Starting chunk {chunk_number} training...')
        
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
        new_checkpoint_path = f'train_runs/300k_20250802_1309/checkpoints/chunk{chunk_number}_final_{model.num_timesteps}steps.zip'
        model.save(new_checkpoint_path)
        
        print(f'\n✅ CHUNK {chunk_number} COMPLETED SUCCESSFULLY!')
        print(f'   📊 Final steps: {model.num_timesteps:,}')
        print(f'   💾 Saved: {new_checkpoint_path}')
        print(f'   ⏱️ Training time: {training_time/60:.1f} minutes')
        
        return model.num_timesteps, new_checkpoint_path
        
    except Exception as e:
        print(f"❌ Chunk {chunk_number} training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return None, None

def main():
    try:
        print("🤖 AUTOMATED CHUNKED TRAINING TO 300K STEPS")
        print("=" * 60)
        print("🎯 Target: 300,000 steps")
        print("📦 Chunk size: 2,000 steps")
        print("💡 Will automatically continue until completion")
        print()
        
        # Find current progress
        latest_checkpoint, current_steps = find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("❌ No checkpoints found! Please ensure training has been started.")
            return 1
        
        print(f"📋 Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"📊 Current progress: {current_steps:,} / 300,000 ({current_steps/300000*100:.1f}%)")
        
        if current_steps >= 300000:
            print("🎉 Training already completed! 300K steps reached.")
            return 0
        
        # Calculate remaining work
        remaining_steps = 300000 - current_steps
        chunks_needed = (remaining_steps + 1999) // 2000  # Ceiling division
        
        print(f"📈 Remaining steps: {remaining_steps:,}")
        print(f"🔢 Chunks needed: {chunks_needed}")
        
        # Get starting chunk number
        next_chunk = get_next_chunk_number()
        print(f"🚀 Starting from chunk {next_chunk}")
        
        # Training loop
        current_checkpoint = latest_checkpoint
        
        for chunk_num in range(next_chunk, next_chunk + chunks_needed):
            # Calculate target for this chunk
            chunk_target_steps = min(current_steps + 2000, 300000)
            
            print(f"\n{'='*70}")
            print(f"🔄 PREPARING CHUNK {chunk_num}")
            print(f"📊 Current: {current_steps:,} → Target: {chunk_target_steps:,}")
            print(f"📈 Overall progress: {current_steps/300000*100:.1f}% → {chunk_target_steps/300000*100:.1f}%")
            
            # Train this chunk
            new_steps, new_checkpoint = train_chunk(
                chunk_num, 
                current_steps, 
                chunk_target_steps, 
                current_checkpoint
            )
            
            if new_steps is None:
                print(f"❌ Chunk {chunk_num} failed! Stopping automated training.")
                return 1
            
            # Update for next iteration
            current_steps = new_steps
            current_checkpoint = new_checkpoint
            
            # Check if we've reached the target
            if current_steps >= 300000:
                print(f"\n🎉🎉🎉 300K TRAINING COMPLETED! 🎉🎉🎉")
                print(f"📊 Final steps: {current_steps:,}")
                print(f"💾 Final checkpoint: {new_checkpoint}")
                break
            
            # Brief pause between chunks
            print(f"\n⏸️ Brief pause before next chunk...")
            time.sleep(2)
        
        print(f"\n✅ AUTOMATED TRAINING COMPLETED!")
        print(f"📊 Final progress: {current_steps:,} / 300,000")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print(f"📊 You can resume by running this script again")
        return 1
    except Exception as e:
        print(f"❌ Automated training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)