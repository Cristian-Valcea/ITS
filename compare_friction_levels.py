#!/usr/bin/env python3
"""
⚖️ FRICTION LEVEL COMPARISON
Compare high friction vs reduced friction environments
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_test_data():
    """Create consistent test data for both environments"""
    test_n_periods = 1000
    test_trading_days = pd.date_range('2024-01-01', periods=test_n_periods, freq='1min')
    
    # Create trending price data to test directional edge
    np.random.seed(42)  # Consistent seed
    
    # NVDA with upward trend
    nvda_base_price = 100.0
    nvda_trend = 0.0002  # 2 bp per minute upward trend
    nvda_noise = np.random.normal(0, 0.01, test_n_periods)
    nvda_returns = nvda_trend + nvda_noise
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=test_trading_days
    )
    
    # MSFT with sideways trend
    msft_base_price = 500.0
    msft_trend = 0.0001  # 1 bp per minute upward trend
    msft_noise = np.random.normal(0, 0.008, test_n_periods)
    msft_returns = msft_trend + msft_noise
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=test_trading_days
    )
    
    # Create simple feature data
    nvda_data = np.random.randn(test_n_periods, 12).astype(np.float32) * 0.1
    msft_data = np.random.randn(test_n_periods, 12).astype(np.float32) * 0.1
    
    # Add price momentum features
    nvda_data[:, 0] = np.diff(np.log(nvda_prices), prepend=0)  # Returns
    msft_data[:, 0] = np.diff(np.log(msft_prices), prepend=0)  # Returns
    
    return nvda_data, msft_data, nvda_prices, msft_prices, test_trading_days

def create_high_friction_env(nvda_data, msft_data, nvda_prices, msft_prices, trading_days):
    """Create environment with HIGH FRICTION (original parameters)"""
    return DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,
        tc_bp=5.0,                  # HIGH: 50 bp transaction costs
        trade_penalty_bp=10.0,      # HIGH: 100 bp trade penalty
        turnover_bp=2.0,            # Turnover penalty
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.20, # Allow 20% for testing
        evaluation_drawdown_pct=0.20,
        is_training=True,
        log_trades=False
    )

def create_low_friction_env(nvda_data, msft_data, nvda_prices, msft_prices, trading_days):
    """Create environment with REDUCED FRICTION (new parameters)"""
    return DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,
        tc_bp=1.0,                  # REDUCED: 10 bp transaction costs
        trade_penalty_bp=2.0,       # REDUCED: 20 bp trade penalty
        turnover_bp=2.0,            # KEPT: Turnover penalty
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.20, # Allow 20% for testing
        evaluation_drawdown_pct=0.20,
        is_training=True,
        log_trades=False
    )

def run_simple_strategy_test(env, strategy_name, actions):
    """Run a simple strategy and measure performance"""
    obs = env.reset()
    total_reward = 0
    trades = 0
    portfolio_values = [env.portfolio_value]
    
    for step, action in enumerate(actions):
        if step >= env.max_steps:
            break
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if 'total_trades' in info:
            trades = info['total_trades']
        
        portfolio_values.append(env.portfolio_value)
        
        if terminated or truncated:
            break
    
    final_portfolio = env.portfolio_value
    portfolio_return = (final_portfolio - 100000) / 100000
    
    return {
        'strategy': strategy_name,
        'final_portfolio': final_portfolio,
        'portfolio_return': portfolio_return,
        'total_reward': total_reward,
        'trades': trades,
        'steps': step + 1,
        'portfolio_values': portfolio_values
    }

def main():
    """Compare friction levels with different strategies"""
    logger.info("⚖️ FRICTION LEVEL COMPARISON STARTED")
    
    # Create consistent test data
    nvda_data, msft_data, nvda_prices, msft_prices, trading_days = create_test_data()
    
    logger.info(f"📊 Test data created: {len(trading_days)} periods")
    logger.info(f"📈 NVDA price range: ${nvda_prices.min():.2f} - ${nvda_prices.max():.2f}")
    logger.info(f"📈 MSFT price range: ${msft_prices.min():.2f} - ${msft_prices.max():.2f}")
    
    # Create environments
    high_friction_env = create_high_friction_env(nvda_data, msft_data, nvda_prices, msft_prices, trading_days)
    low_friction_env = create_low_friction_env(nvda_data, msft_data, nvda_prices, msft_prices, trading_days)
    
    logger.info("✅ Both environments created")
    
    # Define test strategies
    strategies = {
        "Buy and Hold NVDA": [7] * 500,  # BUY_NVDA_HOLD_MSFT repeatedly
        "Buy Both": [8] * 500,           # BUY_BOTH repeatedly
        "Momentum Trading": [8, 8, 4, 4, 8, 8, 4, 4] * 62,  # Buy on momentum, hold, repeat
        "Hold Only": [4] * 500,          # HOLD_BOTH only
    }
    
    logger.info("🎯 TESTING STRATEGIES WITH DIFFERENT FRICTION LEVELS")
    
    results = []
    
    for strategy_name, actions in strategies.items():
        logger.info(f"\n📊 Testing strategy: {strategy_name}")
        
        # Test with high friction
        high_result = run_simple_strategy_test(high_friction_env, f"{strategy_name} (High Friction)", actions)
        results.append(high_result)
        
        # Reset environment
        high_friction_env.reset()
        
        # Test with low friction
        low_result = run_simple_strategy_test(low_friction_env, f"{strategy_name} (Low Friction)", actions)
        results.append(low_result)
        
        # Reset environment
        low_friction_env.reset()
        
        # Compare results
        high_return = high_result['portfolio_return']
        low_return = low_result['portfolio_return']
        improvement = low_return - high_return
        
        logger.info(f"   🔴 High Friction: {high_return:+.2%} return, {high_result['trades']} trades")
        logger.info(f"   🟢 Low Friction:  {low_return:+.2%} return, {low_result['trades']} trades")
        logger.info(f"   📈 Improvement:   {improvement:+.2%} ({improvement*100000:.0f} bp)")
    
    # Summary analysis
    logger.info("\n📊 FRICTION COMPARISON SUMMARY:")
    
    high_friction_results = [r for r in results if "High Friction" in r['strategy']]
    low_friction_results = [r for r in results if "Low Friction" in r['strategy']]
    
    avg_high_return = np.mean([r['portfolio_return'] for r in high_friction_results])
    avg_low_return = np.mean([r['portfolio_return'] for r in low_friction_results])
    avg_improvement = avg_low_return - avg_high_return
    
    logger.info(f"📊 Average High Friction Return: {avg_high_return:+.2%}")
    logger.info(f"📊 Average Low Friction Return:  {avg_low_return:+.2%}")
    logger.info(f"📊 Average Improvement:          {avg_improvement:+.2%}")
    
    # Determine if friction reduction is beneficial
    if avg_improvement > 0.001:  # 10 bp improvement threshold
        logger.info("✅ CONCLUSION: Reduced friction is BENEFICIAL")
        logger.info("   📈 Directional edge is showing through")
        logger.info("   🎯 Lower friction allows profitable strategies to emerge")
        logger.info("   💡 Recommendation: Use reduced friction parameters")
    elif avg_improvement < -0.001:  # 10 bp degradation threshold
        logger.info("❌ CONCLUSION: Reduced friction is HARMFUL")
        logger.info("   📉 Lower friction leads to overtrading")
        logger.info("   🛡️ Higher friction provides better risk control")
        logger.info("   💡 Recommendation: Keep higher friction parameters")
    else:
        logger.info("⚖️ CONCLUSION: Friction levels have MINIMAL IMPACT")
        logger.info("   📊 Performance difference is negligible")
        logger.info("   🎯 Either friction level is acceptable")
        logger.info("   💡 Recommendation: Choose based on risk preference")
    
    logger.info("\n🎉 FRICTION COMPARISON COMPLETED!")

if __name__ == "__main__":
    main()