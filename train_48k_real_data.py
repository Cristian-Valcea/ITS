#!/usr/bin/env python3
"""
🎯 48K REAL DATA TRAINING - Direct Execution
Train PPO model on 245K real market data records (2022-2024)
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Execute 48K training on real data"""
    
    logger.info("🚀 48K REAL DATA TRAINING")
    logger.info("=" * 50)
    
    # Training configuration
    total_timesteps = 48000
    run_name = f"real_data_48k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = f"train_runs/{run_name}"
    
    # Create directories
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(f"{save_path}/checkpoints").mkdir(exist_ok=True)
    
    logger.info(f"📊 Training Configuration:")
    logger.info(f"  Run: {run_name}")
    logger.info(f"  Steps: {total_timesteps:,}")
    logger.info(f"  Data: 245K real market records (2022-2024)")
    
    # Database configuration with vault password
    try:
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        logger.info("✅ Database credentials from vault")
    except Exception as e:
        logger.error(f"❌ Vault access failed: {e}")
        return False
    
    # Load real market data
    try:
        logger.info("📈 Loading real market data...")
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        market_data = adapter.load_training_data(
            start_date='2022-01-03',
            end_date='2024-12-31',
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        logger.info("✅ Real market data loaded successfully")
        
        # Extract the correct data structure
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        logger.info(f"  NVDA features: {nvda_features.shape}")
        logger.info(f"  NVDA prices: {nvda_prices.shape}")
        logger.info(f"  MSFT features: {msft_features.shape}")
        logger.info(f"  MSFT prices: {msft_prices.shape}")
        logger.info(f"  Trading days: {len(trading_days)}")
        
        # Combine features for dual-ticker environment (26-dim observation)
        # 12 NVDA features + 12 MSFT features + 2 position features = 26 total
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        
        # Add 2 position features (initialized to 0 - will be updated by environment)
        position_features = np.zeros((combined_features.shape[0], 2))
        combined_features = np.concatenate([combined_features, position_features], axis=1)
        
        # Create 4-column price data: [NVDA_close, NVDA_returns, MSFT_close, MSFT_returns]
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        
        combined_prices = np.column_stack([
            nvda_prices,     # Column 0: NVDA close prices
            nvda_returns,    # Column 1: NVDA returns  
            msft_prices,     # Column 2: MSFT close prices
            msft_returns     # Column 3: MSFT returns
        ])
        
        logger.info(f"  Combined features: {combined_features.shape} (should be (N, 26))")
        logger.info(f"  Combined prices: {combined_prices.shape} (should be (N, 4))")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create trading environment
    try:
        logger.info("🏗️ Creating trading environment...")
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=1000,
            transaction_cost_pct=0.001
        )
        
        # Wrap for stable-baselines3
        env = DummyVecEnv([lambda: env])
        logger.info("✅ Trading environment created")
        
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        return False
    
    # Create PPO model
    try:
        logger.info("🤖 Creating PPO model...")
        
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=f"tensorboard_logs/{run_name}",
            verbose=1
        )
        
        logger.info("✅ PPO model created")
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return False
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'{save_path}/checkpoints/',
        name_prefix='real_data_48k_model'
    )
    
    # Launch training
    try:
        logger.info("🎯 Starting 48K training on real market data...")
        logger.info("=" * 50)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            tb_log_name=run_name
        )
        
        # Save final model
        final_model_path = f'{save_path}/final_model.zip'
        model.save(final_model_path)
        
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"📁 Model saved: {final_model_path}")
        logger.info(f"📊 Tensorboard: tensorboard --logdir tensorboard_logs/{run_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 48K REAL DATA TRAINING: SUCCESS")
        sys.exit(0)
    else:
        print("❌ 48K REAL DATA TRAINING: FAILED")
        sys.exit(1)