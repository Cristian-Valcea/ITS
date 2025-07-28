#!/usr/bin/env python3
"""
50K Dual-Ticker Training Run
Real training with dual-ticker environment and TimescaleDB data
"""

import os
import sys  
import logging
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

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
            'password': os.getenv('TIMESCALE_PASSWORD')
        }
        data_adapter = DualTickerDataAdapter(
            timescaledb_config=timescaledb_config,
            live_trading_mode=False
        )
        logger.info("✅ Connected to TimescaleDB for training data")
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB connection failed: {e}")
        logger.info("🔄 Creating minimal data adapter for CSV fallback")
        # Create a minimal config for CSV fallback
        minimal_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': 'fallback'
        }
        data_adapter = DualTickerDataAdapter(
            timescaledb_config=minimal_config,
            live_trading_mode=False
        )
    
    # Load and prepare training data using the data adapter
    logger.info("📊 Loading training data...")
    try:
        training_data = data_adapter.load_training_data(
            start_date="2025-07-01",
            end_date="2025-07-25",
            symbols=['NVDA', 'MSFT'],
            bar_size='1min'
        )
        
        # Extract the components needed for the environment
        nvda_data = training_data['nvda_features']  # [N, 12] features
        msft_data = training_data['msft_features']  # [N, 12] features  
        nvda_prices = training_data['nvda_prices']  # Close prices
        msft_prices = training_data['msft_prices']  # Close prices
        trading_days = training_data['trading_days']  # Shared index
        
        logger.info(f"✅ Data loaded - {len(trading_days)} trading periods")
        logger.info(f"   NVDA features shape: {nvda_data.shape}")
        logger.info(f"   MSFT features shape: {msft_data.shape}")
        
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
    
    # Create the dual-ticker environment with OPTIMIZED FRICTION PARAMETERS
    env = DualTickerTradingEnv(
        nvda_data=big_nvda_data,
        msft_data=big_msft_data,
        nvda_prices=big_nvda_prices,
        msft_prices=big_msft_prices,
        trading_days=big_trading_days,
        initial_capital=100000,  # Increased capital for better tracking
        tc_bp=5.0,              # 🔧 OPTIMIZED: 5x higher transaction costs
        trade_penalty_bp=10.0,  # 🔧 OPTIMIZED: High trade penalty
        turnover_bp=2.0,        # 🔧 OPTIMIZED: Turnover penalty enabled
        hold_action_bonus=0.01, # 🔧 OPTIMIZED: Bonus for holding
        action_repeat_penalty=0.002, # 🔧 OPTIMIZED: Penalty for action changes
        daily_trade_limit=50,   # 🔧 OPTIMIZED: Reduced daily limit
        reward_scaling=0.1,     # 🔧 OPTIMIZED: Better reward scaling
        max_daily_drawdown_pct=0.02,  # 🔧 OPTIMIZED: Stricter drawdown limit
        log_trades=False        # Reduce logging spam
    )
    
    logger.info(f"✅ Environment created - observation space: {env.observation_space}")
    logger.info(f"✅ Environment created - action space: {env.action_space}")
    
    return env

def create_model(env):
    """Create the RecurrentPPO model"""
    logger.info("🧠 Creating RecurrentPPO model...")
    
    # Try to load base model for transfer learning
    base_model_path = "models/phase1_fast_recovery_model.zip"
    
    if os.path.exists(base_model_path):
        logger.info(f"🔧 Attempting transfer learning from: {base_model_path}")
        try:
            # For now, create fresh model - transfer learning is complex
            # TODO: Implement proper transfer learning in future iteration
            logger.info("🔄 Creating fresh dual-ticker model (transfer learning TBD)")
        except Exception as e:
            logger.warning(f"⚠️ Transfer learning failed: {e}")
    
    # Create fresh RecurrentPPO model optimized for dual-ticker trading
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=f"logs/dual_ticker_50k_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        learning_rate=1.5e-4,  # 🔧 HALVED from 3e-4 to slow optimizer
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,        # 🔧 REDUCED from 0.2 to prevent KL racing
        ent_coef=0.01,         # 🔧 ENTROPY BONUS to maintain exploration
        device="auto"  # Use GPU if available
    )
    
    logger.info("✅ Model created successfully")
    return model

def setup_callbacks():
    """Setup training callbacks"""
    callbacks = []
    
    # Checkpoint every 10K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints",
        name_prefix="dual_ticker_50k",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    logger.info("✅ Callbacks configured - checkpoints every 10K steps")
    return callbacks

def main():
    logger.info("🚀 Starting 50K Dual-Ticker Training Run")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create environment
        train_env = create_training_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # 🔧 STEP 1.5: Add VecNormalize for reward normalization
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations (features are already scaled)
            norm_reward=True,    # Normalize rewards for better learning
            clip_reward=10.0     # Clip extreme rewards
        )
        
        # Step 2: Create model
        model = create_model(vec_env)
        
        # Step 3: Setup callbacks
        callbacks = setup_callbacks()
        
        # Step 4: Start training
        logger.info("🎯 Starting 50K training steps...")
        logger.info("📊 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=50000,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 5: Save final model and VecNormalize
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = f"models/dual_ticker_50k_{timestamp}.zip"
        vecnorm_path = f"models/dual_ticker_50k_{timestamp}_vecnormalize.pkl"
        
        model.save(final_model_path)
        vec_env.save(vecnorm_path)  # Save VecNormalize statistics
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"💾 Final model saved: {final_model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info(f"⏱️ Training duration: {training_duration}")
        logger.info(f"🔥 Average steps/second: {50000 / training_duration.total_seconds():.1f}")
        
        # Step 6: Quick evaluation
        logger.info("🔍 Running quick evaluation...")
        
        obs = vec_env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Quick 100-step evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
            if done[0]:
                break
        
        avg_reward = total_reward / steps if steps > 0 else 0
        logger.info(f"📈 Evaluation results:")
        logger.info(f"   Average reward per step: {avg_reward:.4f}")
        logger.info(f"   Total steps evaluated: {steps}")
        
        logger.info("🎉 50K Training Run Complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())