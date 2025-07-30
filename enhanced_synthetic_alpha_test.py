#!/usr/bin/env python3
"""
🔬 ENHANCED SYNTHETIC ALPHA TEST
Test 40bp persistent alpha in 4K steps to isolate fundamental issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

def create_persistent_alpha_data(n_periods: int = 1000, alpha_mag: float = 0.4):
    """Create data with PERSISTENT alpha signal"""
    
    np.random.seed(42)
    
    # Create price series with persistent alpha
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    prices = []
    alpha_signals = []
    
    for i in range(n_periods):
        # PERSISTENT alpha signal - always positive
        alpha_signal = alpha_mag  # 40bp persistent edge
        
        # Price follows alpha with some noise
        if i == 0:
            price = base_price
        else:
            # Alpha-driven price movement + small noise
            alpha_return = alpha_signal * 0.01  # Convert bp to return
            noise = np.random.normal(0, 0.001)  # Small noise
            price_change = alpha_return + noise
            price = prices[-1] * (1 + price_change)
        
        prices.append(price)
        alpha_signals.append(alpha_signal)
    
    # Create features: 12 random + persistent alpha
    random_features = np.random.randn(n_periods, 12).astype(np.float32) * 0.1  # Smaller random features
    alpha_feature = np.array(alpha_signals).reshape(-1, 1).astype(np.float32)
    features = np.hstack([random_features, alpha_feature])
    
    price_series = pd.Series(prices, index=trading_days)
    
    logger.info(f"🔬 PERSISTENT alpha data created:")
    logger.info(f"   Alpha magnitude: {alpha_mag} (persistent)")
    logger.info(f"   Expected return: {alpha_mag * 0.01:.1%} per step")
    logger.info(f"   If agent can't learn this, fundamental issue exists")
    
    return features, price_series

def test_persistent_alpha(alpha_mag: float = 0.4, steps: int = 4000):
    """Test if agent can learn PERSISTENT alpha"""
    
    logger.info(f"🔬 ENHANCED SYNTHETIC ALPHA TEST")
    logger.info(f"   Alpha magnitude: {alpha_mag} bp (persistent)")
    logger.info(f"   Training steps: {steps}")
    logger.info(f"   Diagnostic purpose: Isolate advantage/reward issues")
    
    # Create PERSISTENT alpha data
    features, prices = create_persistent_alpha_data(1000, alpha_mag)
    
    # Create environment with minimal friction
    env = IntradayTradingEnvV3(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=100000,
        max_daily_drawdown_pct=0.05,  # Looser DD limit for testing
        base_impact_bp=10.0,          # Minimal impact for testing
        verbose=False
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    # Create model with standard settings
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-4,      # Standard LR
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,              # Standard gamma
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # Standard exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,               # Show training progress
        seed=42,
        device="auto"
    )
    
    logger.info("🔥 Training with PERSISTENT alpha signal...")
    model.learn(total_timesteps=steps, progress_bar=True)
    
    # Test performance
    obs = vec_env.reset()
    actions = []
    rewards = []
    portfolio_values = []
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        actions.append(action[0])
        rewards.append(reward[0])
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
        
        if done[0]:
            break
    
    # Analysis
    action_counts = np.bincount(actions, minlength=3)
    hold_freq = action_counts[1] / len(actions)
    buy_freq = action_counts[2] / len(actions)
    total_reward = sum(rewards)
    
    final_portfolio = portfolio_values[-1] if portfolio_values else 100000
    total_return = (final_portfolio - 100000) / 100000
    
    logger.info(f"\n📊 PERSISTENT ALPHA TEST RESULTS:")
    logger.info(f"   HOLD frequency: {hold_freq:.1%}")
    logger.info(f"   BUY frequency: {buy_freq:.1%}")  
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Total reward: {total_reward:.1f}")
    
    # Diagnostic assessment
    if buy_freq > 0.5 and total_return > 0.02:
        logger.info("✅ Agent learned PERSISTENT alpha")
        logger.info("   Advantage/reward systems working correctly")
        logger.info("   Issue is likely in alpha signal strength or hyperparameters")
        success = True
    elif buy_freq > 0.1:
        logger.info("⚠️ Agent partially learned alpha")
        logger.info("   Some exploration but not optimal exploitation")
        logger.info("   May need hyperparameter tuning")
        success = False
    else:
        logger.info("❌ Agent failed to learn PERSISTENT alpha")
        logger.info("   Fundamental issue in advantage normalization or reward scaling")
        logger.info("   Problem deeper than hyperparameters")
        success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Enhanced Synthetic Alpha Test')
    parser.add_argument('--alpha_mag', type=float, default=0.4, help='Alpha magnitude in bp')
    parser.add_argument('--steps', type=int, default=4000, help='Training steps')
    args = parser.parse_args()
    
    success = test_persistent_alpha(args.alpha_mag, args.steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
