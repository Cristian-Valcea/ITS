#!/usr/bin/env python3
"""
🚨 EMERGENCY OPTIMIZED 50K TRAINING
Fixed version with ALL friction optimizations applied
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

def _get_secure_db_password():
    """Get database password from secure vault with fallback"""
    try:
        from secrets_helper import SecretsHelper
        return SecretsHelper.get_timescaledb_password()
    except Exception as e:
        print(f"Could not get password from vault: {e}")
        return os.getenv('TIMESCALEDB_PASSWORD', 'password')

def create_training_environment():
    """Create the dual-ticker training environment with OPTIMIZED PARAMETERS"""
    logger.info("🏗️ Creating OPTIMIZED dual-ticker training environment...")
    
    # Try to connect to TimescaleDB first
    try:
        timescaledb_config = {
            'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
            'port': int(os.getenv('TIMESCALEDB_PORT', 5432)),
            'database': os.getenv('TIMESCALEDB_DATABASE', 'trading_data'),
            'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
            'password': _get_secure_db_password()
        }
        
        logger.info("🔌 Attempting TimescaleDB connection...")
        data_adapter = DualTickerDataAdapter(**timescaledb_config)
        
        # Get training data
        nvda_data, msft_data, nvda_prices, msft_prices, trading_days = data_adapter.get_training_data(
            start_date='2024-01-01',
            end_date='2024-12-31',
            tickers=['NVDA', 'MSFT']
        )
        
        logger.info(f"✅ TimescaleDB data loaded: {len(trading_days)} periods")
        
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB failed ({e}), using LARGE mock dataset...")
        
        # Generate LARGE mock dataset for 50K training
        big_n_periods = 60000  # Much larger dataset
        big_trading_days = pd.date_range('2024-01-01', periods=big_n_periods, freq='1min')
        
        # Mock feature data with some realistic patterns
        big_nvda_data = np.random.randn(big_n_periods, 12).astype(np.float32)
        big_nvda_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
        
        # Mock price data with trend and volatility
        nvda_base_price = 170.0
        nvda_returns = np.random.normal(0.0001, 0.02, big_n_periods)  # Realistic returns
        nvda_prices = pd.Series(
            nvda_base_price * np.exp(np.cumsum(nvda_returns)),
            index=big_trading_days
        )
        
        msft_base_price = 510.0
        msft_returns = np.random.normal(0.0001, 0.015, big_n_periods)  # Realistic returns
        msft_prices = pd.Series(
            msft_base_price * np.exp(np.cumsum(msft_returns)),
            index=big_trading_days
        )
        big_msft_data = np.random.randn(big_n_periods, 12).astype(np.float32)
        big_msft_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
        
        # Use the big mock data
        nvda_data, msft_data = big_nvda_data, big_msft_data
        trading_days = big_trading_days
    
    # 🚨 CREATE ENVIRONMENT WITH ALL OPTIMIZATIONS APPLIED
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,     # 🔧 OPTIMIZED: Higher capital for tracking
        tc_bp=5.0,                  # 🔧 OPTIMIZED: 5x higher transaction costs (50 bp)
        trade_penalty_bp=10.0,      # 🔧 OPTIMIZED: High trade penalty (100 bp)
        turnover_bp=2.0,            # 🔧 OPTIMIZED: Turnover penalty enabled (20 bp)
        hold_action_bonus=0.01,     # 🔧 OPTIMIZED: Bonus for holding positions
        action_repeat_penalty=0.002, # 🔧 OPTIMIZED: Penalty for action changes
        daily_trade_limit=50,       # 🔧 OPTIMIZED: Reduced daily limit (was 100)
        reward_scaling=0.1,         # 🔧 OPTIMIZED: Better reward scaling
        training_drawdown_pct=0.07, # 🔧 NEW: 7% drawdown for training exploration
        evaluation_drawdown_pct=0.02, # 🔧 NEW: 2% drawdown for evaluation/production
        is_training=True,           # 🔧 NEW: Training mode enabled
        log_trades=False            # Reduce logging spam
    )
    
    logger.info(f"✅ OPTIMIZED Environment created - observation space: {env.observation_space}")
    logger.info(f"✅ OPTIMIZED Environment created - action space: {env.action_space}")
    logger.info("🔧 FRICTION PARAMETERS APPLIED:")
    logger.info(f"   💰 Transaction Cost: {5.0} bp")
    logger.info(f"   💰 Trade Penalty: {10.0} bp") 
    logger.info(f"   💰 Turnover Penalty: {2.0} bp")
    logger.info(f"   🎯 Daily Trade Limit: {50}")
    logger.info(f"   📈 Hold Bonus: {0.01}")
    logger.info(f"   🔄 Action Change Penalty: {0.002}")
    logger.info("🔧 ADAPTIVE DRAWDOWN LIMITS:")
    logger.info(f"   🎓 Training Drawdown: {7.0}% (exploration allowed)")
    logger.info(f"   🛡️ Evaluation Drawdown: {2.0}% (strict risk control)")
    
    return env

def create_model(env):
    """Create the RecurrentPPO model with optimized parameters"""
    logger.info("🧠 Creating OPTIMIZED RecurrentPPO model...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=0.00015,      # Slightly lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,             # Tighter clipping for stability
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/",
        verbose=1,
        seed=42,
        device="auto"
    )
    
    logger.info("✅ OPTIMIZED RecurrentPPO model created")
    return model

def setup_callbacks():
    """Setup training callbacks"""
    logger.info("📋 Setting up callbacks...")
    
    # Checkpoint callback - save every 10K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dual_ticker_optimized"
    )
    
    callbacks = [checkpoint_callback]
    logger.info("✅ Callbacks configured")
    return callbacks

def main():
    """Main training function with ALL OPTIMIZATIONS"""
    logger.info("🚨 EMERGENCY OPTIMIZED 50K DUAL-TICKER TRAINING STARTED")
    logger.info("🔧 ALL FRICTION OPTIMIZATIONS APPLIED")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create optimized environment
        train_env = create_training_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # 🔧 STEP 1.5: Add VecNormalize for reward normalization
        logger.info("🔧 Adding VecNormalize for reward stability...")
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations (features are already scaled)
            norm_reward=True,    # 🔧 OPTIMIZED: Normalize rewards for better learning
            clip_reward=10.0,    # 🔧 OPTIMIZED: Clip extreme rewards
            gamma=0.99
        )
        
        # Step 2: Create optimized model
        model = create_model(vec_env)
        
        # Step 3: Setup callbacks
        callbacks = setup_callbacks()
        
        # Step 4: Train with monitoring
        logger.info("🎯 Starting OPTIMIZED 50K training...")
        logger.info("📊 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=50000,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 5: Save final optimized model and VecNormalize
        model_path = "models/dual_ticker_optimized_50k_final.zip"
        vecnorm_path = "models/dual_ticker_optimized_50k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)  # Save VecNormalize statistics
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 OPTIMIZED TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("✅ ALL FRICTION OPTIMIZATIONS SUCCESSFULLY APPLIED")
        
    except Exception as e:
        logger.error(f"❌ OPTIMIZED Training failed: {e}")
        raise

if __name__ == "__main__":
    main()