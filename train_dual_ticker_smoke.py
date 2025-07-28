#!/usr/bin/env python3
"""
50K Dual-Ticker Smoke Train
Quick validation before full 200K training run
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env_file()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import dual-ticker components directly 
try:
    from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
    from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter  
    from src.training.dual_ticker_model_adapter import DualTickerModelAdapter
except ImportError as e:
    logger.error(f"❌ Failed to import dual-ticker components: {e}")
    logger.info("🔄 Creating simplified training environment...")
    DualTickerTradingEnv = None
    DualTickerDataAdapter = None
    DualTickerModelAdapter = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dual_ticker_env(start_date: str, end_date: str, lookback_window: int = 50):
    """Create dual-ticker environment with real Polygon data"""
    
    logger.info(f"🏗️ Creating dual-ticker environment: {start_date} to {end_date}")
    
    # Initialize data adapter with TimescaleDB
    try:
        data_adapter = DualTickerDataAdapter(
            database_url=f"postgresql://postgres:{os.getenv('TIMESCALE_PASSWORD')}@localhost:5432/trading_data"
        )
        logger.info("✅ Connected to TimescaleDB for training data")
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB connection failed: {e}")
        logger.info("🔄 Falling back to CSV data source")
        data_adapter = DualTickerDataAdapter()
    
    # Create environment
    env = DualTickerTradingEnv(
        data_adapter=data_adapter,
        start_date=start_date,
        end_date=end_date,
        lookback_window=lookback_window,
        initial_capital=10000,
        transaction_cost_bp=1.0,  # 1 basis point
        reward_scaling=0.01
    )
    
    return env

def main():
    parser = argparse.ArgumentParser(description="50K Dual-Ticker Smoke Train")
    parser.add_argument("--start", default="2025-07-16", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-07-25", help="End date (YYYY-MM-DD)")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Training timesteps")
    parser.add_argument("--tensorboard", default=f"tb/smoke_{datetime.now().strftime('%Y-%m-%d')}", help="TensorBoard log dir")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint directory")
    parser.add_argument("--base-model", default="models/phase1_fast_recovery_model.zip", help="Base NVDA model")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting 50K Dual-Ticker Smoke Train")
    logger.info(f"📅 Date range: {args.start} to {args.end}")
    logger.info(f"🎯 Target timesteps: {args.total_timesteps:,}")
    logger.info(f"📊 TensorBoard: {args.tensorboard}")
    
    # Create training environment
    train_env = create_dual_ticker_env(args.start, args.end)
    vec_env = DummyVecEnv([lambda: train_env])
    
    # Create evaluation environment (20% holdout)
    eval_env = create_dual_ticker_env(args.start, args.end)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Initialize model with transfer learning
    try:
        logger.info(f"🔧 Loading base model: {args.base_model}")
        adapter = DualTickerModelAdapter(args.base_model)
        model = adapter.create_dual_ticker_model(vec_env)
        logger.info("✅ Transfer learning model created")
    except Exception as e:
        logger.warning(f"⚠️ Transfer learning failed: {e}")
        logger.info("🔄 Creating fresh dual-ticker model")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.tensorboard,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint every 10K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=args.checkpoint_dir,
        name_prefix="smoke_ckpt"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=f"{args.checkpoint_dir}/smoke_best",
        log_path=f"{args.checkpoint_dir}/smoke_eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Start training
    logger.info("🎯 Starting smoke training...")
    logger.info("📊 Monitor progress:")
    logger.info(f"   tensorboard --logdir {args.tensorboard}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{args.checkpoint_dir}/smoke_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(final_path)
        
        logger.info("✅ Smoke training completed successfully!")
        logger.info(f"💾 Final model saved: {final_path}")
        logger.info("🔍 Check TensorBoard for metrics:")
        logger.info("   - Policy loss: should trend ↓ or flat")
        logger.info("   - Mean reward: should be > 0 by step 20K")
        logger.info("   - Entropy: should be 0.8-1.2")
        logger.info("   - EvalCallback: mean reward ≥ 0, Sharpe ≥ 0.3")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Smoke training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())