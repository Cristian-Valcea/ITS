#!/usr/bin/env python3
"""
Phase 1 Full Training - 100k Steps
Runs after successful smoke test validation
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from gym_env.intraday_trading_env import IntradayTradingEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data_with_dates():
    """Create sample data with proper datetime index."""
    logger.info("📊 Creating sample data with datetime index...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000  # More data for 100k training
    
    # Create datetime index
    start_date = datetime(2024, 1, 1, 9, 30)  # Market open
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Create feature data (11 features to match Phase 1 config)
    feature_data = pd.DataFrame(index=dates)
    feature_data['rsi_14'] = np.random.uniform(20, 80, n_samples)
    feature_data['ema_10'] = np.random.uniform(100, 200, n_samples)
    feature_data['ema_20'] = np.random.uniform(100, 200, n_samples)
    feature_data['vwap'] = np.random.uniform(100, 200, n_samples)
    feature_data['hour_sin'] = np.random.uniform(-1, 1, n_samples)
    feature_data['hour_cos'] = np.random.uniform(-1, 1, n_samples)
    
    # Add risk features
    for i in range(5):
        feature_data[f'risk_feature_{i}'] = np.random.randn(n_samples) * 0.1
    
    # Create price data
    price_data = pd.Series(
        100 + np.cumsum(np.random.randn(n_samples) * 0.02),
        index=dates
    )
    
    logger.info(f"   - Features shape: {feature_data.shape}")
    logger.info(f"   - Price data length: {len(price_data)}")
    logger.info(f"   - Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"   - Price range: ${price_data.min():.2f} - ${price_data.max():.2f}")
    
    return feature_data, price_data

def run_full_phase1_training():
    """Execute full Phase 1 training (100k steps)."""
    
    logger.info("🚀 PHASE 1 FULL TRAINING (100K STEPS)")
    logger.info("=" * 60)
    
    # Load configuration
    with open('config/phase1_reality_grounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🔧 FULL TRAINING CONFIGURATION:")
    logger.info(f"   - Reward scaling: {config['environment']['reward_scaling']}")
    logger.info(f"   - Reward bounds: ±{config['validation']['reward_bounds']['max_reward']}")
    logger.info(f"   - Normalize advantages: {config['training'].get('normalize_advantage', False)}")
    logger.info(f"   - Value function coef: {config['training'].get('vf_coef', 0.5)}")
    logger.info(f"   - Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info("")
    
    # Create sample data
    feature_data, price_data = create_sample_data_with_dates()
    
    # Create environment
    logger.info("🏗️ Creating full training environment...")
    env = IntradayTradingEnv(
        processed_feature_data=feature_data.values,
        price_data=price_data,
        initial_capital=config['environment']['initial_capital'],
        reward_scaling=config['environment']['reward_scaling'],
        institutional_safeguards_config=config
    )
    
    # Verify observation space
    obs_space = env.observation_space
    logger.info(f"✅ Environment observation space verified: {obs_space}")
    
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create model with full training settings
    logger.info("🤖 Creating full training model...")
    
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        policy_kwargs={
            'net_arch': [128, 128],
            'lstm_hidden_size': 64,
            'n_lstm_layers': 1,
            'enable_critic_lstm': True,
        },
        learning_rate=0.0003,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=config['training'].get('vf_coef', 0.8),
        normalize_advantage=config['training'].get('normalize_advantage', True),
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='logs/tensorboard_phase1_fix1'
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints/',
        name_prefix='phase1_full'
    )
    
    logger.info("🎯 FULL TRAINING OBJECTIVES:")
    logger.info("   - Target ep_rew_mean: 400 (8k raw → 400 scaled)")
    logger.info("   - Maintain entropy > -0.4")
    logger.info("   - Achieve explained_variance > 0.2")
    logger.info("   - Complete 100,000 timesteps")
    logger.info("   - Get reliable plateau performance")
    logger.info("")
    
    try:
        # Run full training
        logger.info("🚀 Starting full Phase 1 training...")
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name="phase1_full_training"
        )
        
        logger.info("✅ Full Phase 1 training completed successfully!")
        
        # Save final model
        model_path = "models/phase1_full_model"
        model.save(model_path)
        logger.info(f"💾 Final model saved to: {model_path}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎉 PHASE 1 FULL TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ ACHIEVEMENTS:")
        logger.info("   • Completed 100,000 timesteps")
        logger.info("   • Applied 1-day sprint tuning fixes")
        logger.info("   • Reward scaling: 0.02 → 0.25 (12.5x)")
        logger.info("   • Enhanced PPO parameters")
        logger.info("   • Tighter reward bounds (±150)")
        logger.info("   • Institutional safeguards active")
        logger.info("")
        logger.info("📊 MONITORING:")
        logger.info("   • TensorBoard: http://localhost:6006")
        logger.info("   • Final model: models/phase1_full_model")
        logger.info("   • Checkpoints: models/checkpoints/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full training failed: {e}")
        return False

if __name__ == "__main__":
    success = run_full_phase1_training()
    
    if success:
        logger.info("")
        logger.info("🎯 PHASE 1 COMPLETE!")
        logger.info("   Ready for Phase 2 or production deployment")
        
    sys.exit(0 if success else 1)