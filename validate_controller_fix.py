#!/usr/bin/env python3
"""
🔧 CONTROLLER FIX VALIDATION
Test that the sign error fix and parameter adjustments work correctly
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

def test_controller_fix():
    """Test that the controller fix produces correct behavior."""
    
    print("🔧 Testing Controller Fix...")
    
    # Load test data
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        print(f"✅ Loaded test data: {len(dual_data)} rows")
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return False
    
    # Create test environment with fixed controller
    feature_data = np.random.randn(200, 26).astype(np.float32)
    price_data = np.random.randn(200, 4).astype(np.float32) * 100 + 100
    trading_days = np.arange(200)
    
    env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=feature_data,
        processed_price_data=price_data,
        trading_days=trading_days,
        initial_capital=100000,
        max_episode_steps=100,
        lookback_window=10,
        enable_controller=True,
        controller_target_hold_rate=0.70,  # 70% target
        hold_bonus_weight=0.010,  # Fixed parameter
        verbose=True
    )
    
    print("✅ Environment created with fixed controller")
    
    # Track controller behavior
    controller_calls = 0
    hold_errors = []
    hold_bonuses = []
    hold_rates = []
    
    # Monkey patch to track controller calls
    original_compute_bonus = env.controller.compute_bonus
    original_calculate_hold_error = env._calculate_hold_error
    
    def tracked_compute_bonus(hold_error, regime_score):
        nonlocal controller_calls
        controller_calls += 1
        result = original_compute_bonus(hold_error, regime_score)
        hold_errors.append(hold_error)
        hold_bonuses.append(result)
        return result
    
    def tracked_calculate_hold_error():
        result = original_calculate_hold_error()
        current_hold_rate = env._calculate_current_hold_rate()
        hold_rates.append(current_hold_rate)
        return result
    
    env.controller.compute_bonus = tracked_compute_bonus
    env._calculate_hold_error = tracked_calculate_hold_error
    
    # Run test simulation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("\n🚀 Running 50-step validation...")
    
    for step in range(50):
        # Use a simple policy: hold more often to test controller response
        if step < 10:
            action = 4  # Hold action to build up hold rate
        else:
            action = env.action_space.sample()  # Random actions
        
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        if step % 10 == 0:
            current_hold_rate = hold_rates[-1] if hold_rates else 0.0
            current_error = hold_errors[-1] if hold_errors else 0.0
            current_bonus = hold_bonuses[-1] if hold_bonuses else 0.0
            print(f"   Step {step}: Hold Rate={current_hold_rate:.1%}, Error={current_error:.3f}, Bonus={current_bonus:.6f}")
        
        if done:
            print(f"Episode terminated at step {step}")
            break
    
    # Analysis
    print(f"\n📊 CONTROLLER FIX VALIDATION:")
    print(f"   Steps completed: {step+1}")
    print(f"   Controller calls: {controller_calls}")
    
    if hold_errors and hold_bonuses and hold_rates:
        print(f"   Hold rate range: {min(hold_rates):.1%} to {max(hold_rates):.1%}")
        print(f"   Hold error range: {min(hold_errors):.3f} to {max(hold_errors):.3f}")
        print(f"   Bonus range: {min(hold_bonuses):.6f} to {max(hold_bonuses):.6f}")
        
        # Check if fix is working
        avg_hold_rate = np.mean(hold_rates)
        avg_error = np.mean(hold_errors)
        avg_bonus = np.mean(hold_bonuses)
        
        print(f"   Average hold rate: {avg_hold_rate:.1%}")
        print(f"   Average error: {avg_error:.3f}")
        print(f"   Average bonus: {avg_bonus:.6f}")
        
        # Validation checks
        target_hold_rate = 0.70
        
        # Check 1: When hold rate < target, error should be positive
        low_hold_steps = [(hr, he) for hr, he in zip(hold_rates, hold_errors) if hr < target_hold_rate]
        if low_hold_steps:
            low_hold_errors = [he for hr, he in low_hold_steps]
            avg_low_error = np.mean(low_hold_errors)
            print(f"   When hold < target: avg error = {avg_low_error:.3f} (should be > 0)")
            
            if avg_low_error > 0:
                print("   ✅ SIGN FIX WORKING: Positive error when holding too little")
            else:
                print("   ❌ SIGN STILL WRONG: Negative error when holding too little")
                return False
        
        # Check 2: Bonus should vary (not constant)
        bonus_variance = np.var(hold_bonuses)
        print(f"   Bonus variance: {bonus_variance:.8f}")
        
        if bonus_variance > 1e-8:
            print("   ✅ CONTROLLER RESPONDING: Bonus varies with conditions")
        else:
            print("   ⚠️ CONTROLLER STATIC: Bonus not varying enough")
        
        # Check 3: Parameter range check
        max_bonus = max(hold_bonuses)
        expected_max = 2.0 * 0.010  # 2 * base_hold_bonus
        print(f"   Max bonus: {max_bonus:.6f} (expected max: {expected_max:.6f})")
        
        if max_bonus <= expected_max * 1.1:  # Allow 10% tolerance
            print("   ✅ PARAMETER FIX WORKING: Bonus within expected range")
        else:
            print("   ⚠️ PARAMETER ISSUE: Bonus exceeds expected range")
        
        return True
    else:
        print("   ❌ NO DATA COLLECTED")
        return False

def test_with_trained_model():
    """Test the fix with a trained model."""
    
    print("\n🤖 Testing Fix with Trained Model...")
    
    # Load model
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
    
    # Create environment
    feature_data = np.random.randn(200, 26).astype(np.float32)
    price_data = np.random.randn(200, 4).astype(np.float32) * 100 + 100
    trading_days = np.arange(200)
    
    env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=feature_data,
        processed_price_data=price_data,
        trading_days=trading_days,
        initial_capital=100000,
        max_episode_steps=50,
        lookback_window=10,
        enable_controller=True,
        controller_target_hold_rate=0.70,
        hold_bonus_weight=0.010,  # Fixed parameter
        verbose=False
    )
    
    # Track behavior
    hold_rates = []
    actions_taken = []
    
    # Run model evaluation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("🚀 Running model evaluation with fixed controller...")
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action) if hasattr(action, '__iter__') else action)
        
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        # Track hold rate
        current_hold_rate = env._calculate_current_hold_rate()
        hold_rates.append(current_hold_rate)
        
        if step % 10 == 0:
            print(f"   Step {step}: Hold Rate={current_hold_rate:.1%}, Action={action}")
        
        if done:
            break
    
    # Analysis
    if hold_rates:
        final_hold_rate = hold_rates[-1]
        avg_hold_rate = np.mean(hold_rates)
        
        print(f"\n📊 MODEL + FIX ANALYSIS:")
        print(f"   Steps completed: {step+1}")
        print(f"   Final hold rate: {final_hold_rate:.1%}")
        print(f"   Average hold rate: {avg_hold_rate:.1%}")
        print(f"   Target hold rate: 70.0%")
        
        # Check if model behavior improved
        if avg_hold_rate > 0.05:  # More than 5% holding
            print("   ✅ IMPROVEMENT: Model showing holding behavior")
            return True
        else:
            print("   ⚠️ LIMITED IMPROVEMENT: Still very low hold rate")
            return False
    else:
        print("   ❌ NO DATA COLLECTED")
        return False

def main():
    """Main validation function."""
    
    print("🔧 CONTROLLER FIX VALIDATION TEST")
    print("=" * 50)
    
    # Test 1: Controller fix validation
    test1_passed = test_controller_fix()
    
    # Test 2: Model with fix
    test2_passed = test_with_trained_model()
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS:")
    print(f"   Controller Fix: {'✅ WORKING' if test1_passed else '❌ ISSUES'}")
    print(f"   Model Behavior: {'✅ IMPROVED' if test2_passed else '⚠️ LIMITED'}")
    
    if test1_passed:
        print("\n✅ CONTROLLER FIX VALIDATED - Ready for training!")
        print("🚀 Recommended next step: Resume Cycle 5 or restart training")
        return True
    else:
        print("\n❌ CONTROLLER FIX NEEDS MORE WORK")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)