#!/usr/bin/env python3
"""
🚀 SIMPLE MICRO-CYCLE TEST
Test with increased base_hold_bonus (0.015) to provide stronger learning signal
"""

import os
import sys
import time
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

def run_simple_microcycle():
    """Run a simple micro-cycle test with increased hold bonus."""
    
    print("🚀 SIMPLE MICRO-CYCLE TEST")
    print("=" * 40)
    print("Purpose: Test learning with increased base_hold_bonus (0.015)")
    print("Expected: Hold rate should increase within 1000 steps")
    print("")
    
    # Load the fixed model
    model_path = "train_runs/stairways_8cycle_20250803_193928/cycle_05_hold_67%/model_checkpoint_cycle_05_hold_67%_FIXED.zip"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded model: {Path(model_path).name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create training environment with INCREASED hold bonus
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        
        feature_data = np.random.randn(3000, 26).astype(np.float32)
        price_data = np.random.randn(3000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(3000)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=200,  # Longer episodes
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.67,
            hold_bonus_weight=0.015,  # INCREASED from 0.010
            verbose=False
        )
        
        print("✅ Environment created with INCREASED hold bonus")
        print(f"   Base hold bonus: 0.015 (was 0.010)")
        print(f"   Target hold rate: 67%")
        print(f"   Max episode steps: 200")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Set up model
    model.set_env(env)
    
    # Quick pre-training evaluation
    print("\n📊 PRE-TRAINING EVALUATION:")
    hold_rate_before = evaluate_hold_rate(model, env)
    print(f"   Hold rate before training: {hold_rate_before:.1%}")
    
    # Run micro-cycle training
    print("\n🚀 Running micro-cycle training...")
    print("   Steps: 1,000")
    print("   Expected duration: ~2 minutes")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=1000,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        duration = time.time() - start_time
        print(f"✅ Training completed in {duration:.1f}s")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Post-training evaluation
    print("\n📊 POST-TRAINING EVALUATION:")
    hold_rate_after = evaluate_hold_rate(model, env)
    print(f"   Hold rate after training: {hold_rate_after:.1%}")
    
    # Analysis
    improvement = hold_rate_after - hold_rate_before
    print(f"   Improvement: {improvement:+.1%}")
    
    print("\n🎯 SUCCESS CRITERIA:")
    
    if hold_rate_after > 0.05:  # 5% threshold
        print(f"   ✅ BREAKTHROUGH: Hold rate > 5% ({hold_rate_after:.1%})")
        success = True
    else:
        print(f"   ❌ NO BREAKTHROUGH: Hold rate still ≤ 5% ({hold_rate_after:.1%})")
        success = False
    
    if improvement > 0.02:  # 2% improvement
        print(f"   ✅ LEARNING: Improvement > 2% ({improvement:+.1%})")
    else:
        print(f"   ⚠️ LIMITED LEARNING: Improvement ≤ 2% ({improvement:+.1%})")
    
    if hold_rate_after >= 0.08:  # Target range
        print(f"   ✅ TARGET RANGE: Hold rate ≥ 8% ({hold_rate_after:.1%})")
    else:
        print(f"   ⚠️ BELOW TARGET: Hold rate < 8% ({hold_rate_after:.1%})")
    
    # Save improved model
    if success:
        output_path = "train_runs/stairways_8cycle_20250803_193928/microcycle_improved.zip"
        model.save(output_path)
        print(f"💾 Improved model saved: {Path(output_path).name}")
    
    return success

def evaluate_hold_rate(model, env, episodes=5):
    """Evaluate current hold rate of the model."""
    
    total_actions = 0
    hold_actions = 0
    
    for episode in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_actions = 0
        episode_holds = 0
        
        for step in range(50):  # Limit steps per episode
            action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            if action == 4:  # Hold action
                episode_holds += 1
            episode_actions += 1
            
            if done:
                break
        
        total_actions += episode_actions
        hold_actions += episode_holds
    
    return hold_actions / total_actions if total_actions > 0 else 0.0

def main():
    """Main function."""
    
    success = run_simple_microcycle()
    
    print("\n" + "=" * 40)
    print("📊 MICRO-CYCLE TEST RESULT:")
    
    if success:
        print("✅ SUCCESS: Hold rate breakthrough achieved!")
        print("🚀 Ready to proceed with full Cycle 6")
        print("📈 Expected Cycle 6 performance: 20-35% hold rate")
    else:
        print("⚠️ PARTIAL: Some improvement but below breakthrough threshold")
        print("🔧 Consider: Further increase base_hold_bonus to 0.020")
        print("📊 Alternative: Proceed with current progress")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)