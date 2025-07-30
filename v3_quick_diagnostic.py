#!/usr/bin/env python3
"""
🩺 V3 QUICK DIAGNOSTIC
Fast check to understand why V3 is failing under realistic volatility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_volatile_data(n_periods=1000):
    """Create data with realistic volatility that might break V3"""
    np.random.seed(42)
    
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    
    # Realistic intraday volatility (same as gate test)
    returns = np.random.normal(0.0001, 0.015, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    logger.info(f"Price volatility: {returns.std()*np.sqrt(390)*100:.1f}% daily")
    
    return feature_data, price_series

def test_v3_under_volatility():
    """Test V3 environment under realistic volatility"""
    
    logger.info("🩺 V3 QUICK DIAGNOSTIC")
    
    # Create volatile data
    feature_data, price_series = create_volatile_data(1000)
    
    # Create V3 environment
    env = IntradayTradingEnvV3(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        verbose=True  # Enable logging to see what's happening
    )
    
    # Test a few random actions to see reward behavior
    obs, info = env.reset()
    logger.info(f"Initial: Portfolio ${info['portfolio_value']:,.0f}")
    
    portfolio_history = [info['portfolio_value']]
    reward_history = []
    
    # Test sequence of actions
    actions = [2, 1, 1, 0, 1, 2, 0, 1]  # BUY, HOLD, HOLD, SELL, HOLD, BUY, SELL, HOLD
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_history.append(info['portfolio_value'])
        reward_history.append(reward)
        
        action_names = ['SELL', 'HOLD', 'BUY']
        logger.info(f"Step {i+1}: {action_names[action]:4s} → Portfolio ${info['portfolio_value']:8,.0f} | "
                   f"Reward {reward:8.2f} | DD {info['current_drawdown']:.2%}")
        
        if terminated:
            logger.warning(f"Episode terminated at step {i+1}")
            break
    
    # Analysis
    final_portfolio = portfolio_history[-1]
    total_return = (final_portfolio - 100000) / 100000
    max_dd = max((max(portfolio_history[:i]) - portfolio_history[i]) / max(portfolio_history[:i]) 
                 for i in range(1, len(portfolio_history)))
    
    logger.info(f"\n📊 DIAGNOSTIC RESULTS:")
    logger.info(f"   Final portfolio: ${final_portfolio:,.0f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Max drawdown: {max_dd:.2%}")
    logger.info(f"   Cumulative reward: {sum(reward_history):.2f}")
    
    # Problem analysis
    large_moves = [(i, abs(portfolio_history[i+1] - portfolio_history[i])) 
                   for i in range(len(portfolio_history)-1) 
                   if abs(portfolio_history[i+1] - portfolio_history[i]) > 1000]
    
    if large_moves:
        logger.warning(f"🚨 LARGE PORTFOLIO MOVES DETECTED:")
        for step, move_size in large_moves[:5]:  # Show first 5
            logger.warning(f"   Step {step}: ${move_size:,.0f} change")
    
    if max_dd > 0.05:
        logger.warning(f"🚨 HIGH DRAWDOWN: {max_dd:.2%} > 5%")
        logger.warning("   This suggests V3 is not controlling risk properly under volatility")
    
    return max_dd < 0.02  # Pass if DD < 2%

def test_price_impact_scaling():
    """Test if price movements are overwhelming V3 impact costs"""
    
    logger.info("\n🔬 TESTING PRICE VS IMPACT SCALING")
    
    from src.gym_env.dual_reward_v3 import DualTickerRewardV3
    
    # Create V3 calculator
    reward_calc = DualTickerRewardV3(verbose=False)
    
    # Test different scenarios
    scenarios = [
        ("Small price move", 100000, 100100, 42500),    # $100 gain, $42.5K trade
        ("Large price move", 100000, 105000, 42500),    # $5K gain, $42.5K trade  
        ("Huge price move", 100000, 110000, 42500),     # $10K gain, $42.5K trade
    ]
    
    for name, prev_port, curr_port, trade_val in scenarios:
        reward, components = reward_calc.calculate_reward(
            prev_portfolio_value=prev_port,
            curr_portfolio_value=curr_port,
            nvda_trade_value=trade_val,
            msft_trade_value=0.0,
            nvda_position=trade_val/170,
            msft_position=0.0,
            nvda_price=170.0,
            msft_price=510.0,
            step=1
        )
        
        pnl = curr_port - prev_port
        impact = components.embedded_impact
        
        logger.info(f"{name:15s}: P&L ${pnl:5.0f} vs Impact ${impact:6.2f} → Net {reward:8.2f}")
        
        reward_calc.reset()
    
    logger.info("If P&L >> Impact consistently, V3 impact model is too weak for volatile markets")

if __name__ == "__main__":
    # Run diagnostics
    volatility_ok = test_v3_under_volatility()
    test_price_impact_scaling()
    
    if volatility_ok:
        print("✅ V3 handles volatility well")
    else:
        print("❌ V3 struggling with realistic volatility - impact model needs strengthening")