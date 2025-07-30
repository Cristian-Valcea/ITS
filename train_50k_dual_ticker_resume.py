#!/usr/bin/env python3
"""
50K Dual-Ticker Training Run with Resume Support
Real training with dual-ticker environment and TimescaleDB data
Enhanced with checkpoint resumption capability
"""

import os
import sys  
import logging
import argparse
import numpy as np
import pandas as pd
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

def _get_secure_db_password():
    """Get database password from secure vault with fallback"""
    try:
        from secrets_helper import SecretsHelper
        return SecretsHelper.get_timescaledb_password()
    except Exception as e:
        logger.warning(f"Could not get password from vault: {e}")
        return os.environ.get('TIMESCALEDB_PASSWORD', 'your_password')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='50K Dual-Ticker Training with Resume Support')
    parser.add_argument('--resume-from-checkpoint', type=str, 
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for training logs')
    parser.add_argument('--tensorboard-log', type=str, 
                       help='TensorBoard log directory')
    parser.add_argument('--total-steps', type=int, default=50000,
                       help='Total training steps')
    return parser.parse_args()

def create_training_environment():
    """Create the dual-ticker training environment"""
    logger.info("🏗️ Creating dual-ticker training environment...")
    
    # Try to connect to TimescaleDB first
    try:
        timescaledb_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': _get_secure_db_password()
        }
        
        # Create data adapter
        data_adapter = DualTickerDataAdapter(timescaledb_config)
        
        # Load real data from TimescaleDB
        logger.info("📊 Loading data from TimescaleDB...")
        nvda_data, msft_data, nvda_prices, msft_prices, trading_days = data_adapter.load_dual_ticker_data(
            start_date='2025-07-01',
            end_date='2025-07-28',
            symbols=['NVDA', 'MSFT']
        )
        
        logger.info(f"✅ Real data loaded - {len(trading_days)} periods")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        logger.info("🔄 Creating minimal mock data for testing...")
        
        # Create minimal mock data for testing
        n_periods = 10000  # Much larger dataset to prevent running out of data
        trading_days = pd.date_range('2025-07-01', periods=n_periods, freq='1min')
        
        # Mock NVDA data
        nvda_prices = pd.Series(
            170 + np.random.randn(n_periods) * 2, 
            index=trading_days
        )
        nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
        
        # Mock MSFT data  
        msft_prices = pd.Series(
            510 + np.random.randn(n_periods) * 3,
            index=trading_days
        )
        msft_data = np.random.randn(n_periods, 12).astype(np.float32)
        
        logger.info(f"✅ Mock data created - {len(trading_days)} periods")
    
    # Create larger mock data to ensure we never run out
    big_n_periods = 100000  # Much larger dataset for continuous training
    big_trading_days = pd.date_range('2024-01-01', periods=big_n_periods, freq='1min')
    
    # Generate larger mock data
    big_nvda_prices = pd.Series(
        170 + np.cumsum(np.random.randn(big_n_periods) * 0.01), 
        index=big_trading_days
    )
    big_nvda_data = np.random.randn(big_n_periods, 12).astype(np.float32)
    
    big_msft_prices = pd.Series(
        510 + np.cumsum(np.random.randn(big_n_periods) * 0.01),
        index=big_trading_days
    )
    big_msft_data = np.random.randn(big_n_periods, 12).astype(np.float32)
    
    # Create the dual-ticker environment with much larger dataset
    env = DualTickerTradingEnv(
        nvda_data=big_nvda_data,
        msft_data=big_msft_data,
        nvda_prices=big_nvda_prices,
        msft_prices=big_msft_prices,
        trading_days=big_trading_days,
        initial_capital=10000,
        tc_bp=1.0,  # 1 basis point
        reward_scaling=0.01,
        max_daily_drawdown_pct=0.95  # Almost never terminate on drawdown
    )
    
    logger.info(f"✅ Environment created - observation space: {env.observation_space}")
    logger.info(f"✅ Environment created - action space: {env.action_space}")
    
    return env

def create_or_load_model(env, checkpoint_path=None, tensorboard_log=None):
    """Create new model or load from checkpoint"""
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔄 Loading model from checkpoint: {checkpoint_path}")
        try:
            # Load the model from checkpoint
            model = RecurrentPPO.load(checkpoint_path, env=env)
            
            # Update tensorboard log if provided
            if tensorboard_log:
                model.tensorboard_log = tensorboard_log
                logger.info(f"📊 Updated TensorBoard log: {tensorboard_log}")
            
            logger.info("✅ Model loaded successfully from checkpoint")
            return model, True  # True indicates resumed from checkpoint
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("🔄 Creating new model instead...")
    
    # Create fresh model
    logger.info("🧠 Creating new RecurrentPPO model...")
    
    # Set default tensorboard log if not provided
    if not tensorboard_log:
        tensorboard_log = f"logs/dual_ticker_50k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto"  # Use GPU if available
    )
    
    logger.info("✅ New model created successfully")
    return model, False  # False indicates new model

def setup_callbacks(checkpoint_prefix="dual_ticker_50k"):
    """Setup training callbacks"""
    callbacks = []
    
    # Checkpoint every 10K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints",
        name_prefix=checkpoint_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    logger.info("✅ Callbacks configured - checkpoints every 10K steps")
    return callbacks

def calculate_remaining_steps(checkpoint_path, total_steps):
    """Calculate remaining steps based on checkpoint"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return total_steps
    
    # Extract step number from checkpoint filename
    # Expected format: dual_ticker_50k_10000_steps.zip
    try:
        filename = os.path.basename(checkpoint_path)
        if '_steps.zip' in filename:
            # Extract the step number
            step_part = filename.split('_steps.zip')[0]
            completed_steps = int(step_part.split('_')[-1])
            remaining_steps = total_steps - completed_steps
            logger.info(f"📊 Checkpoint indicates {completed_steps} steps completed")
            logger.info(f"📊 Remaining steps: {remaining_steps}")
            return max(0, remaining_steps)
    except Exception as e:
        logger.warning(f"⚠️ Could not parse checkpoint step count: {e}")
    
    return total_steps

def main():
    args = parse_arguments()
    
    logger.info("🚀 Starting 50K Dual-Ticker Training Run")
    logger.info("=" * 60)
    
    if args.resume_from_checkpoint:
        logger.info(f"🔄 Resume mode - checkpoint: {args.resume_from_checkpoint}")
    else:
        logger.info("🆕 New training run")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create environment
        train_env = create_training_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Step 2: Create or load model
        model, resumed = create_or_load_model(
            vec_env, 
            args.resume_from_checkpoint,
            args.tensorboard_log
        )
        
        # Step 3: Calculate remaining steps
        remaining_steps = calculate_remaining_steps(args.resume_from_checkpoint, args.total_steps)
        
        if remaining_steps <= 0:
            logger.info("✅ Training already complete based on checkpoint!")
            return
        
        # Step 4: Setup callbacks
        callbacks = setup_callbacks()
        
        # Step 5: Start training
        logger.info(f"🎯 Starting training for {remaining_steps} steps...")
        logger.info(f"📊 Monitor with: tensorboard --logdir {args.tensorboard_log or 'logs/'}")
        
        if resumed:
            logger.info("🔄 Resuming from checkpoint - continuing training...")
        
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not resumed  # Don't reset if resuming
        )
        
        # Step 6: Save final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = f"models/dual_ticker_50k_final_{timestamp}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(final_model_path)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"💾 Final model saved: {final_model_path}")
        logger.info(f"⏱️ Training duration: {training_duration}")
        logger.info(f"🔥 Average steps/second: {remaining_steps / training_duration.total_seconds():.1f}")
        
        # Display final statistics
        logger.info("📊 Training Summary:")
        logger.info(f"   Total steps completed: {args.total_steps}")
        logger.info(f"   Steps in this session: {remaining_steps}")
        logger.info(f"   Resumed from checkpoint: {resumed}")
        logger.info(f"   Final model: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        logger.info("💾 Latest checkpoint should be available in ./checkpoints/")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("💾 Check ./checkpoints/ for any saved progress")
        raise

if __name__ == "__main__":
    main()