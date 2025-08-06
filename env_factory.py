#!/usr/bin/env python3
"""
🏭 ENVIRONMENT FACTORY
Clean factory for creating environments with custom reward systems
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gym_env.dual_ticker_custom_reward import DualTickerTradingEnvV3CustomReward

def make_env_with_refined_reward(
    processed_feature_data,
    processed_price_data, 
    trading_days,
    initial_capital=10000.0,
    max_daily_drawdown_pct=0.30,  # ← Increased from 0.20 to let episodes run longer
    **refined_reward_kwargs
):
    """
    Factory function to create environment with RefinedRewardSystem
    
    Args:
        Standard environment parameters plus refined_reward_kwargs
        
    Returns:
        DualTickerTradingEnvV3CustomReward instance
    """
    return DualTickerTradingEnvV3CustomReward(
        processed_feature_data=processed_feature_data,
        processed_price_data=processed_price_data,
        trading_days=trading_days,
        initial_capital=initial_capital,
        lookback_window=50,
        max_episode_steps=390,
        max_daily_drawdown_pct=max_daily_drawdown_pct,
        transaction_cost_pct=0.001,
        refined_reward_kwargs=refined_reward_kwargs
    )