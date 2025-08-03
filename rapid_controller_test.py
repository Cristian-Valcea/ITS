#!/usr/bin/env python3
"""
🔍 RAPID CONTROLLER TEST
Verify that the dual-lane controller is being called every step
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

def test_controller_integration():
    """Test if controller is being called during environment steps."""
    
    print("🔍 Testing Controller Integration...")
    
    # Load test data
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        print(f"✅ Loaded test data: {len(dual_data)} rows")
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return False
    
    # Create minimal dataset
    nvda_data = dual_data[dual_data['symbol'] == 'NVDA'].head(100).copy()
    msft_data = dual_data[dual_data['symbol'] == 'MSFT'].head(100).copy()
    
    # Create feature and price data
    feature_data = np.random.randn(100, 26).astype(np.float32)
    price_data = np.random.randn(100, 4).astype(np.float32) * 100 + 100  # Positive prices
    trading_days = np.arange(100)
    
    # Create environment with controller enabled
    env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=feature_data,
        processed_price_data=price_data,
        trading_days=trading_days,
        initial_capital=100000,
        max_episode_steps=50,
        lookback_window=10,
        enable_controller=True,
        controller_target_hold_rate=0.65,
        verbose=True  # Enable verbose logging
    )
    
    print("✅ Environment created with controller enabled")
    
    # Test controller calls
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    controller_calls = 0
    hold_bonuses = []
    
    # Monkey patch the controller to track calls
    original_compute_bonus = env.controller.compute_bonus
    
    def tracked_compute_bonus(*args, **kwargs):
        nonlocal controller_calls
        controller_calls += 1
        result = original_compute_bonus(*args, **kwargs)
        hold_bonuses.append(result)
        print(f"   🔧 Controller called #{controller_calls}: bonus={result:.6f}")
        return result
    
    env.controller.compute_bonus = tracked_compute_bonus
    
    # Run test steps
    print("\n🚀 Running 20 test steps...")
    for step in range(20):
        action = env.action_space.sample()  # Random action
        step_result = env.step(action)
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, Done={done}")
        
        if done:
            print("Episode terminated early")
            break
    
    # Analysis
    print(f"\n📊 CONTROLLER ANALYSIS:")
    print(f"   Total steps: {step+1}")
    print(f"   Controller calls: {controller_calls}")
    print(f"   Call rate: {controller_calls/(step+1)*100:.1f}%")
    
    if hold_bonuses:
        print(f"   Bonus range: {min(hold_bonuses):.6f} to {max(hold_bonuses):.6f}")
        print(f"   Non-zero bonuses: {sum(1 for b in hold_bonuses if abs(b) > 1e-6)}")
        print(f"   Average bonus: {np.mean(hold_bonuses):.6f}")
    
    # Check if controller is working
    if controller_calls == 0:
        print("❌ CONTROLLER NOT CALLED - Integration issue!")
        return False
    elif controller_calls < (step+1) * 0.8:
        print("⚠️ CONTROLLER CALLED INFREQUENTLY - Possible conditional issue")
        return False
    else:
        print("✅ CONTROLLER INTEGRATION WORKING")
        return True

def test_model_with_controller():
    """Test a trained model to see controller behavior."""
    
    print("\n🔍 Testing Trained Model with Controller...")
    
    # Find a model checkpoint
    model_path = "train_runs/stairways_8cycle_20250803_193928/cycle_04_hold_70%/model_checkpoint_cycle_04_hold_70%.zip"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded model: {Path(model_path).name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create test environment
    feature_data = np.random.randn(100, 26).astype(np.float32)
    price_data = np.random.randn(100, 4).astype(np.float32) * 100 + 100
    trading_days = np.arange(100)
    
    env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=feature_data,
        processed_price_data=price_data,
        trading_days=trading_days,
        initial_capital=100000,
        max_episode_steps=30,
        lookback_window=10,
        enable_controller=True,
        controller_target_hold_rate=0.70,  # Match cycle 4 target
        verbose=False
    )
    
    # Track controller activity
    controller_calls = 0
    hold_bonuses = []
    actions_taken = []
    
    original_compute_bonus = env.controller.compute_bonus
    
    def tracked_compute_bonus(*args, **kwargs):
        nonlocal controller_calls
        controller_calls += 1
        result = original_compute_bonus(*args, **kwargs)
        hold_bonuses.append(result)
        return result
    
    env.controller.compute_bonus = tracked_compute_bonus
    
    # Run model evaluation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("🚀 Running model evaluation...")
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        if done:
            break
    
    # Analysis
    print(f"\n📊 MODEL + CONTROLLER ANALYSIS:")
    print(f"   Steps completed: {step+1}")
    print(f"   Controller calls: {controller_calls}")
    unique_actions = len(set([int(a) if hasattr(a, '__iter__') else a for a in actions_taken]))
    print(f"   Actions taken: {unique_actions} unique actions")
    
    if hold_bonuses:
        print(f"   Bonus range: {min(hold_bonuses):.6f} to {max(hold_bonuses):.6f}")
        print(f"   Non-zero bonuses: {sum(1 for b in hold_bonuses if abs(b) > 1e-6)}")
    
    # Check for hold behavior
    hold_actions = sum(1 for a in actions_taken if a == 4)  # Assuming action 4 is hold
    hold_rate = hold_actions / len(actions_taken) if actions_taken else 0
    print(f"   Hold rate: {hold_rate:.1%}")
    
    return controller_calls > 0

def main():
    """Main test function."""
    
    print("🔍 RAPID CONTROLLER INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Basic controller integration
    test1_passed = test_controller_integration()
    
    # Test 2: Model with controller
    test2_passed = test_model_with_controller()
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    print(f"   Controller Integration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Model + Controller: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ CONTROLLER IS WORKING - Integration successful!")
        return True
    else:
        print("\n❌ CONTROLLER ISSUES DETECTED - Needs investigation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)