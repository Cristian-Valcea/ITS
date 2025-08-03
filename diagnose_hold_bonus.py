#!/usr/bin/env python3
"""
🔍 DIAGNOSE HOLD BONUS APPLICATION
Check if the corrected controller is actually providing meaningful hold bonuses
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from stable_baselines3 import PPO
    from gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def diagnose_hold_bonus():
    """Diagnose if hold bonus is being applied correctly."""
    
    print("🔍 DIAGNOSING HOLD BONUS APPLICATION")
    print("=" * 50)
    
    # Load test data
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        print(f"✅ Loaded test data: {len(dual_data)} rows")
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return False
    
    # Create test environment
    feature_data = np.random.randn(1000, 26).astype(np.float32)
    price_data = np.random.randn(1000, 4).astype(np.float32) * 100 + 100
    trading_days = np.arange(1000)
    
    env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=feature_data,
        processed_price_data=price_data,
        trading_days=trading_days,
        initial_capital=100000,
        max_episode_steps=100,
        lookback_window=10,
        enable_controller=True,
        controller_target_hold_rate=0.67,
        hold_bonus_weight=0.010,  # FIXED parameter
        verbose=True
    )
    
    print("✅ Environment created with FIXED controller")
    
    # Track detailed controller behavior
    controller_calls = []
    hold_errors = []
    hold_bonuses = []
    base_bonuses = []
    enhanced_bonuses = []
    actions = []
    rewards = []
    
    # Monkey patch to track everything
    original_compute_bonus = env.controller.compute_bonus
    original_enhance_hold_bonus = env._enhance_hold_bonus_with_controller
    
    def tracked_compute_bonus(hold_error, regime_score):
        result = original_compute_bonus(hold_error, regime_score)
        controller_calls.append(len(controller_calls) + 1)
        hold_errors.append(hold_error)
        hold_bonuses.append(result)
        print(f"   🎛️ Controller: error={hold_error:.3f}, bonus={result:.6f}")
        return result
    
    def tracked_enhance_hold_bonus(base_bonus):
        result = original_enhance_hold_bonus(base_bonus)
        base_bonuses.append(base_bonus)
        enhanced_bonuses.append(result)
        print(f"   🔧 Enhancement: base={base_bonus:.6f} → enhanced={result:.6f}")
        return result
    
    env.controller.compute_bonus = tracked_compute_bonus
    env._enhance_hold_bonus_with_controller = tracked_enhance_hold_bonus
    
    # Run diagnostic simulation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("\n🚀 Running diagnostic simulation...")
    print("Testing different action patterns to see controller response:")
    
    # Test pattern: Force some hold actions to see controller response
    test_actions = [4, 4, 4, 4, 4,  # 5 hold actions
                   1, 2, 3, 0, 1,   # 5 trade actions
                   4, 4, 4, 4, 4]   # 5 more hold actions
    
    for step, action in enumerate(test_actions):
        print(f"\n--- Step {step+1}: Action {action} ({'HOLD' if action == 4 else 'TRADE'}) ---")
        
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        actions.append(action)
        rewards.append(reward)
        
        # Calculate current hold rate
        current_hold_rate = env._calculate_current_hold_rate()
        print(f"   📊 Current hold rate: {current_hold_rate:.1%}")
        print(f"   💰 Step reward: {reward:.2f}")
        
        if done:
            print(f"   ⚠️ Episode terminated at step {step+1}")
            break
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC ANALYSIS:")
    
    if controller_calls:
        print(f"   Controller calls: {len(controller_calls)}")
        print(f"   Hold error range: {min(hold_errors):.3f} to {max(hold_errors):.3f}")
        print(f"   Controller bonus range: {min(hold_bonuses):.6f} to {max(hold_bonuses):.6f}")
        
        # Check if controller is responding to hold rate changes
        if len(set(hold_bonuses)) > 1:
            print("   ✅ Controller bonus varies (responding to conditions)")
        else:
            print("   ❌ Controller bonus constant (not responding)")
    else:
        print("   ❌ No controller calls recorded")
    
    if base_bonuses and enhanced_bonuses:
        print(f"   Base bonus range: {min(base_bonuses):.6f} to {max(base_bonuses):.6f}")
        print(f"   Enhanced bonus range: {min(enhanced_bonuses):.6f} to {max(enhanced_bonuses):.6f}")
        
        # Check enhancement effectiveness
        avg_enhancement = np.mean([e/b for e, b in zip(enhanced_bonuses, base_bonuses) if b > 0])
        print(f"   Average enhancement factor: {avg_enhancement:.2f}x")
        
        if avg_enhancement > 1.1:
            print("   ✅ Controller providing meaningful enhancement")
        else:
            print("   ⚠️ Controller enhancement minimal")
    
    # Check action-reward correlation
    if actions and rewards:
        hold_rewards = [r for a, r in zip(actions, rewards) if a == 4]
        trade_rewards = [r for a, r in zip(actions, rewards) if a != 4]
        
        if hold_rewards and trade_rewards:
            avg_hold_reward = np.mean(hold_rewards)
            avg_trade_reward = np.mean(trade_rewards)
            
            print(f"   Average hold reward: {avg_hold_reward:.2f}")
            print(f"   Average trade reward: {avg_trade_reward:.2f}")
            
            if avg_hold_reward > avg_trade_reward:
                print("   ✅ Hold actions rewarded more than trades")
            else:
                print("   ❌ Trade actions still rewarded more than holds")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    
    if not controller_calls:
        print("   ❌ CRITICAL: Controller not being called")
        print("   🔧 Check: Controller integration in environment")
        return False
    
    if len(set(hold_bonuses)) == 1:
        print("   ❌ ISSUE: Controller bonus not varying")
        print("   🔧 Check: Hold error calculation and controller logic")
        return False
    
    if enhanced_bonuses and max(enhanced_bonuses) < 0.005:
        print("   ⚠️ ISSUE: Enhanced bonuses too small")
        print("   🔧 Recommendation: Increase base_hold_bonus by 1.5x")
        print(f"   📊 Current max enhanced bonus: {max(enhanced_bonuses):.6f}")
        print(f"   📊 Recommended min bonus: 0.005")
        return False
    
    print("   ✅ Controller appears to be working correctly")
    print("   🚀 Ready for micro-cycle training")
    return True

def test_bonus_scaling():
    """Test different base_hold_bonus values to find optimal scaling."""
    
    print("\n🔬 TESTING BONUS SCALING")
    print("=" * 30)
    
    test_values = [0.005, 0.010, 0.015, 0.020]
    
    for base_bonus in test_values:
        print(f"\n--- Testing base_hold_bonus = {base_bonus:.3f} ---")
        
        # Create test environment
        feature_data = np.random.randn(100, 26).astype(np.float32)
        price_data = np.random.randn(100, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(100)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=50,
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.67,
            hold_bonus_weight=base_bonus,
            verbose=False
        )
        
        # Test controller response
        bonuses = []
        
        # Simulate low hold rate scenario
        env.recent_actions = [1, 2, 3, 0, 1]  # All trades, 0% hold rate
        hold_error = env._calculate_hold_error()
        bonus = env.controller.compute_bonus(hold_error, 0.0)
        bonuses.append(bonus)
        
        print(f"   Hold error: {hold_error:.3f}")
        print(f"   Controller bonus: {bonus:.6f}")
        print(f"   Bonus/base ratio: {bonus/base_bonus:.2f}x")
        
        if bonus > 0.005:
            print(f"   ✅ Sufficient bonus magnitude")
        else:
            print(f"   ⚠️ Bonus too small")

def main():
    """Main diagnostic function."""
    
    # Run main diagnostic
    success = diagnose_hold_bonus()
    
    # Run bonus scaling test
    test_bonus_scaling()
    
    print("\n" + "=" * 50)
    print("📊 FINAL DIAGNOSTIC RESULT:")
    
    if success:
        print("✅ CONTROLLER WORKING - Ready for micro-cycle training")
    else:
        print("❌ CONTROLLER ISSUES - Needs adjustment before training")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)