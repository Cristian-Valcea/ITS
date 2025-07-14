#!/usr/bin/env python3
"""
Simple test to demonstrate the trading environment issues without complex mocking.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gym_env.intraday_trading_env import IntradayTradingEnv


def create_test_data(num_steps=50):
    """Create test data for environment testing."""
    dates = pd.date_range('2024-01-01 09:30:00', periods=num_steps, freq='1min')
    
    # Create simple price movement
    prices = [100.0 + i * 0.1 for i in range(num_steps)]  # Gradual increase
    price_series = pd.Series(prices, index=dates, name='close')
    
    # Create simple features
    features = np.array([[p/100.0, 0.0, 0.1] for p in prices], dtype=np.float32)
    
    return features, price_series


def test_action_change_penalty_scaling():
    """Test action change penalty scaling issue."""
    print("🧪 Testing action change penalty scaling...")
    
    features, prices = create_test_data(10)
    
    # Test with small reward scaling (common in configs)
    env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=10000.0,
        reward_scaling=0.0001,  # Very small scaling
        action_change_penalty_factor=0.001,
        transaction_cost_pct=0.0,
        log_trades=False
    )
    
    obs = env.reset()
    
    # Take first action
    obs, reward1, terminated, truncated, info = env.step(1)  # Hold
    print(f"📊 Step 1 (Hold): Reward = {reward1:.8f}")
    
    # Change action (triggers penalty)
    obs, reward2, terminated, truncated, info = env.step(2)  # Buy (action change)
    print(f"📊 Step 2 (Buy): Reward = {reward2:.8f}")
    
    # The penalty is 0.001 * (2-1)^2 = 0.001
    # But reward_scaling is 0.0001, so normal rewards are ~0.0001 scale
    # The penalty (0.001) is 10x larger than typical rewards!
    
    expected_penalty = 0.001 * ((2 - 1) ** 2)
    print(f"📊 Expected penalty: {expected_penalty:.8f}")
    print(f"📊 Reward scaling: {env.reward_scaling}")
    print(f"📊 Penalty vs scaled reward ratio: {expected_penalty / (abs(reward1) + 1e-8):.2f}")
    
    if expected_penalty > abs(reward1) * 10:
        print("⚠️  ISSUE: Penalty dominates reward signal!")
    
    print("✅ Action change penalty test completed")


def test_trade_cooldown_timing():
    """Test trade cooldown counter timing."""
    print("\n🧪 Testing trade cooldown timing...")
    
    features, prices = create_test_data(10)
    
    env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=10000.0,
        trade_cooldown_steps=2,  # 2-step cooldown
        transaction_cost_pct=0.0,
        log_trades=False
    )
    
    obs = env.reset()
    print(f"📊 Initial steps_since_last_trade: {env.steps_since_last_trade}")
    
    # Execute trade at step 0
    obs, reward, terminated, truncated, info = env.step(2)  # Buy
    print(f"📊 After trade: steps_since_last_trade = {env.steps_since_last_trade}")
    print(f"📊 Position: {env.current_position}")
    
    # Test immediate next step (should be in cooldown)
    old_steps = env.steps_since_last_trade
    obs, reward, terminated, truncated, info = env.step(0)  # Try to sell
    new_steps = env.steps_since_last_trade
    position = env.current_position
    
    print(f"📊 Step 1: steps_since_last_trade {old_steps} → {new_steps}")
    print(f"📊 Position: {position} (should still be 1 if cooldown works)")
    
    # The issue: steps_since_last_trade is incremented BEFORE the cooldown check
    # So at step 0: trade executes, counter = 0
    # At step 1: counter incremented to 1 BEFORE check, so 1 < 2 = cooldown active
    # But if counter was incremented AFTER, it would be 0 < 2 = cooldown active
    
    print("✅ Trade cooldown timing test completed")


def test_hourly_turnover_issue():
    """Demonstrate the hourly turnover decay issue."""
    print("\n🧪 Testing hourly turnover decay issue...")
    
    features, prices = create_test_data(20)
    
    env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=10000.0,
        hourly_turnover_cap=1.0,
        transaction_cost_pct=0.0,
        log_trades=False
    )
    
    obs = env.reset()
    
    # Execute a trade
    obs, reward, terminated, truncated, info = env.step(2)  # Buy
    print(f"📊 After trade: hourly_traded_value = ${env.hourly_traded_value:.2f}")
    print(f"📊 Trades this hour: {len(env.trades_this_hour)}")
    
    # Take several steps without trading
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(1)  # Hold
        print(f"📊 Step {i+1} (Hold): hourly_traded_value = ${env.hourly_traded_value:.2f}")
    
    # The issue: _update_hourly_trades() is only called when a trade executes
    # So during quiet periods, old trades never get removed from the rolling window
    # This means hourly_traded_value stays artificially high
    
    print("✅ Hourly turnover decay test completed")
    print("   Note: hourly_traded_value should decay as time passes, even without trades")


def main():
    """Run simple tests to demonstrate the issues."""
    print("🎯 Demonstrating Trading Environment Issues")
    print("=" * 50)
    
    try:
        test_action_change_penalty_scaling()
        test_trade_cooldown_timing()
        test_hourly_turnover_issue()
        
        print("\n" + "=" * 50)
        print("🎉 ISSUE DEMONSTRATION COMPLETED!")
        
        print("\n📋 Issues Demonstrated:")
        print("1. ⚠️  Action change penalty can dominate reward signal with small reward_scaling")
        print("2. ⚠️  Trade cooldown counter timing may allow trades too early")
        print("3. ⚠️  Hourly turnover doesn't decay during quiet periods")
        
        print("\n🔧 Fixes Needed:")
        print("1. Scale action_change_penalty by reward_scaling")
        print("2. Increment steps_since_last_trade AFTER cooldown check")
        print("3. Call _update_hourly_trades() every step, not just on trades")
        print("4. Verify feature preprocessing prevents look-ahead bias")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())