#!/usr/bin/env python3
"""
🎯 DEBUG ALPHA EXTRACTION
Debug why the agent won't trade despite strong alpha signals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

def debug_alpha_extraction():
    """Debug reward calculations during alpha signal periods"""
    
    logger.info("🎯 DEBUG ALPHA EXTRACTION")
    
    # Create strong alpha data
    enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
        n_periods=1000, 
        seed=42, 
        alpha_strength=0.2  # Very strong alpha
    )
    
    logger.info(f"Alpha signals: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish")
    
    # Create environment
    env = IntradayTradingEnvV3(
        processed_feature_data=enhanced_features,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        transaction_cost_pct=0.0001,
        log_trades=True,  # Enable trade logging
        base_impact_bp=68.0,
        impact_exponent=0.5,
        verbose=True
    )
    
    # Reset environment
    obs = env.reset()
    
    logger.info(f"\n🔍 Analyzing first 10 steps with alpha signals:")
    logger.info(f"{'Step':>4} {'Alpha':>8} {'Action':>8} {'Reward':>10} {'Portfolio':>12} {'Components'}")
    logger.info("-" * 80)
    
    for step in range(10):
        # Extract alpha signal (last feature)
        alpha_signal = obs[-1] if len(obs) > 12 else 0.0
        
        # Try different actions based on alpha signal
        if alpha_signal > 0.05:      # Strong bullish signal
            action = 2  # BUY
        elif alpha_signal < -0.05:   # Strong bearish signal  
            action = 0  # SELL
        else:
            action = 1  # HOLD
        
        # Step environment
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, _, info = result
        
        # Get reward components if available
        components_str = ""
        if hasattr(env.reward_calculator, 'return_history') and env.reward_calculator.return_history:
            # Get last calculated components - this is tricky to access
            components_str = f"R={reward:.1f}"
        
        action_name = ['SELL', 'HOLD', 'BUY'][action]
        portfolio_value = info.get('portfolio_value', 100000)
        
        logger.info(f"{step:4d} {alpha_signal:8.3f} {action_name:>8} {reward:10.2f} ${portfolio_value:11,.0f} {components_str}")
        
        if done:
            logger.info("Episode ended early")
            break
    
    logger.info(f"\n🎯 ANALYSIS:")
    logger.info(f"   Strong alpha signals present: {abs(alpha_signal) > 0.05}")
    logger.info(f"   Agent reward for trading: {reward:.2f}")
    logger.info(f"   Final portfolio: ${portfolio_value:,.0f}")
    
    # Test manual reward calculation
    logger.info(f"\n🧪 MANUAL REWARD TEST:")
    
    # Simulate a $500 alpha trade manually
    reward_calc = env.reward_calculator
    
    test_reward, test_components = reward_calc.calculate_reward(
        prev_portfolio_value=100000.0,
        curr_portfolio_value=100500.0,  # $500 profit
        nvda_trade_value=42500.0,       # $42.5K trade (250 shares)
        msft_trade_value=0.0,
        nvda_position=250.0,
        msft_position=0.0,
        nvda_price=170.0,
        msft_price=510.0,
        step=1
    )
    
    logger.info(f"   Manual test - $500 alpha trade:")
    logger.info(f"   Risk-free NAV: ${test_components.risk_free_nav_change:.2f}")
    logger.info(f"   Impact cost: ${test_components.embedded_impact:.2f}")
    logger.info(f"   Net reward: ${test_reward:.2f}")
    logger.info(f"   Trade worthwhile: {'YES' if test_reward > 0 else 'NO'}")
    
    return test_reward > 0

if __name__ == "__main__":
    success = debug_alpha_extraction()
    print(f"\n🎯 DEBUG COMPLETE: Alpha extraction {'WORKING' if success else 'BLOCKED'}")