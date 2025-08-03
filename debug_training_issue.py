#!/usr/bin/env python3
"""
Debug script to identify the 'low >= high' error in Cycle 1 training
"""

import logging
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
from src.gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_environment_creation():
    """Debug the exact point where the error occurs"""
    
    logger.info("🔍 Debugging environment creation...")
    
    try:
        # Create test data adapter (same as training script)
        logger.info("Step 1: Creating data adapter...")
        data_config = {
            'mock_data': True,
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'trading_user'
        }
        
        adapter = DualTickerDataAdapter(
            timescaledb_config=data_config,
            live_trading_mode=False
        )
        
        # Load training data
        logger.info("Step 2: Loading training data...")
        training_data = adapter.load_training_data(
            start_date='2024-01-01',
            end_date='2024-12-31',
            symbols=['NVDA', 'MSFT'],
            data_split='train'
        )
        
        # Create properly formatted data
        logger.info("Step 3: Formatting data...")
        nvda_features = training_data['nvda_features']
        msft_features = training_data['msft_features']
        
        n_timesteps = nvda_features.shape[0]
        logger.info(f"   Data timesteps: {n_timesteps}")
        
        combined_features = np.zeros((n_timesteps, 26), dtype=np.float32)
        combined_features[:, 0:12] = nvda_features
        combined_features[:, 12:24] = msft_features
        combined_features[:, 24] = 0.0
        combined_features[:, 25] = 0.0
        
        nvda_prices = training_data['nvda_prices'].values
        msft_prices = training_data['msft_prices'].values
        
        combined_prices = np.zeros((n_timesteps, 4), dtype=np.float32)
        combined_prices[:, 0] = nvda_prices
        combined_prices[:, 1] = nvda_prices
        combined_prices[:, 2] = msft_prices
        combined_prices[:, 3] = msft_prices
        
        logger.info(f"   Feature data shape: {combined_features.shape}")
        logger.info(f"   Price data shape: {combined_prices.shape}")
        
        # Try creating environment with short episode length
        episode_length = 250  # Much shorter than 306 to be safe
        lookback_window = 50
        
        logger.info(f"Step 4: Creating environment (episode_length={episode_length}, lookback={lookback_window})...")
        logger.info(f"   Available data: {n_timesteps} timesteps")
        logger.info(f"   Required minimum: {lookback_window + episode_length} timesteps")
        
        if n_timesteps < (lookback_window + episode_length):
            logger.error(f"❌ Insufficient data: need {lookback_window + episode_length}, have {n_timesteps}")
            return False
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=training_data['trading_days'],
            max_episode_steps=episode_length,
            lookback_window=lookback_window,
            enable_controller=True,
            enable_regime_detection=True,
            controller_target_hold_rate=0.75,
            bootstrap_days=20,  # Reduced for limited data
            verbose=True
        )
        
        logger.info("✅ Environment created successfully!")
        
        # Try resetting environment
        logger.info("Step 5: Testing environment reset...")
        obs, info = env.reset(seed=42)
        logger.info(f"✅ Environment reset successful! Observation shape: {obs.shape}")
        
        # Try a few steps
        logger.info("Step 6: Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"   Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")
            if done:
                break
        
        logger.info("✅ Environment test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔍 DEBUGGING TRAINING ISSUE")
    logger.info("=" * 50)
    
    success = debug_environment_creation()
    
    if success:
        logger.info("🎉 Debug completed successfully - environment works!")
    else:
        logger.error("❌ Debug failed - issue identified")
        sys.exit(1)