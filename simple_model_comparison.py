#!/usr/bin/env python3
"""
🎯 SIMPLE V3 MODEL COMPARISON
Quick comparison between original and tuned V3 models
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

from sb3_contrib import RecurrentPPO
from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
from src.gym_env.dual_ticker_trading_env_v3_tuned import DualTickerTradingEnvV3Tuned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_timesteps=1000):
    """Create synthetic test data"""
    # Generate 26-dimensional features
    features = np.random.randn(n_timesteps, 26).astype(np.float32)
    
    # Generate price data
    base_prices = np.array([500.0, 500.0, 300.0, 300.0])
    price_changes = np.random.randn(n_timesteps, 4) * 0.01
    prices = base_prices * (1 + price_changes.cumsum(axis=0))
    
    # Generate timestamps
    timestamps = pd.date_range('2025-01-01', periods=n_timesteps, freq='1min')
    
    return features, prices, timestamps

def test_single_model(model_path, env_type="original"):
    """Test a single model"""
    logger.info(f"🔍 Testing {env_type} model: {model_path}")
    
    # Load model
    model = RecurrentPPO.load(model_path)
    
    # Create test data
    features, prices, timestamps = create_test_data(500)
    
    # Create appropriate environment
    if env_type == "original":
        # Original V3 environment expects pandas Series for price_data
        price_series = pd.Series(prices[:, 1])  # NVDA close prices
        env = DualTickerTradingEnvV3(
            processed_feature_data=features,
            price_data=price_series,
            verbose=False
        )
    else:
        # Tuned environment
        env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=features,
            processed_price_data=prices,
            trading_days=timestamps.values,
            hold_bonus_weight=0.0005,
            ticket_cost_per_trade=0.20,
            verbose=False
        )
    
    # Run a single episode
    obs, _ = env.reset()
    total_reward = 0
    actions_taken = []
    step_count = 0
    
    try:
        for step in range(200):  # Short episode
            # Get model prediction
            action, _states = model.predict(obs, deterministic=True)
            actions_taken.append(int(action))
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
                
    except Exception as e:
        logger.error(f"Error during episode: {str(e)}")
        return None
    
    # Analyze actions
    action_counts = np.bincount(actions_taken, minlength=9)
    hold_action = 4  # Middle action is typically hold
    hold_percentage = action_counts[hold_action] / len(actions_taken) * 100
    
    results = {
        'model_type': env_type,
        'total_reward': total_reward,
        'steps': step_count,
        'actions_taken': len(actions_taken),
        'hold_percentage': hold_percentage,
        'action_distribution': action_counts.tolist(),
        'avg_reward_per_step': total_reward / step_count if step_count > 0 else 0
    }
    
    logger.info(f"✅ {env_type.title()} model results:")
    logger.info(f"   Steps: {step_count}")
    logger.info(f"   Total reward: {total_reward:.2f}")
    logger.info(f"   Hold percentage: {hold_percentage:.1f}%")
    logger.info(f"   Avg reward/step: {results['avg_reward_per_step']:.4f}")
    
    return results

def main():
    """Main comparison function"""
    logger.info("🎯 Starting Simple V3 Model Comparison")
    
    # Model paths
    original_model = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
    tuned_model = "train_runs/v3_tuned_warmstart_50k_20250802_233810/best_model.zip"
    
    # Check if models exist
    if not os.path.exists(original_model):
        logger.error(f"Original model not found: {original_model}")
        return
        
    if not os.path.exists(tuned_model):
        logger.error(f"Tuned model not found: {tuned_model}")
        return
    
    # Test both models
    logger.info("=" * 60)
    original_results = test_single_model(original_model, "original")
    
    logger.info("=" * 60)
    tuned_results = test_single_model(tuned_model, "tuned")
    
    if original_results and tuned_results:
        logger.info("=" * 60)
        logger.info("🎯 COMPARISON SUMMARY")
        logger.info("=" * 60)
        
        # Compare key metrics
        hold_change = tuned_results['hold_percentage'] - original_results['hold_percentage']
        reward_change = tuned_results['avg_reward_per_step'] - original_results['avg_reward_per_step']
        
        logger.info(f"📊 HOLD BEHAVIOR:")
        logger.info(f"   Original: {original_results['hold_percentage']:.1f}%")
        logger.info(f"   Tuned:    {tuned_results['hold_percentage']:.1f}%")
        logger.info(f"   Change:   {hold_change:+.1f}% {'✅ REDUCED' if hold_change < 0 else '❌ INCREASED'}")
        
        logger.info(f"📊 REWARD PERFORMANCE:")
        logger.info(f"   Original: {original_results['avg_reward_per_step']:.4f}")
        logger.info(f"   Tuned:    {tuned_results['avg_reward_per_step']:.4f}")
        logger.info(f"   Change:   {reward_change:+.4f} {'✅ IMPROVED' if reward_change > 0 else '❌ DEGRADED'}")
        
        # Trading activity analysis
        original_trades = sum(original_results['action_distribution']) - original_results['action_distribution'][4]
        tuned_trades = sum(tuned_results['action_distribution']) - tuned_results['action_distribution'][4]
        
        logger.info(f"📊 TRADING ACTIVITY:")
        logger.info(f"   Original trades: {original_trades}")
        logger.info(f"   Tuned trades:    {tuned_trades}")
        logger.info(f"   Change:          {tuned_trades - original_trades:+d} {'✅ MORE ACTIVE' if tuned_trades > original_trades else '❌ LESS ACTIVE'}")
        
        # Overall assessment
        logger.info("=" * 60)
        if hold_change < -5 and reward_change > -0.01:  # Reduced holding by >5% without major reward loss
            logger.info("🎉 TUNING SUCCESS: Reduced holding behavior while maintaining performance!")
            logger.info("✅ RECOMMENDATION: Deploy tuned model")
        elif hold_change < 0 and reward_change > -0.05:  # Some improvement
            logger.info("✅ TUNING PARTIAL SUCCESS: Some improvement in trading behavior")
            logger.info("🔄 RECOMMENDATION: Consider further tuning or deploy with monitoring")
        else:
            logger.info("❌ TUNING NEEDS IMPROVEMENT: Limited behavioral change or performance loss")
            logger.info("🔄 RECOMMENDATION: Iterate with stronger tuning weights")
        
    else:
        logger.error("❌ Comparison failed - could not test both models")

if __name__ == "__main__":
    main()