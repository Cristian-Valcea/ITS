#!/usr/bin/env python3
"""
🧪 QUICK CYCLE 7 EVALUATION
Comprehensive evaluation of Cycle 7 model to understand the 0% hold rate result
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

def evaluate_cycle7_comprehensive():
    """Comprehensive evaluation of Cycle 7 model."""
    
    print("🧪 COMPREHENSIVE CYCLE 7 EVALUATION")
    print("=" * 50)
    
    # Load Cycle 7 model
    cycle7_path = "train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%/model_checkpoint_cycle_07_hold_45%_PROGRESS.zip"
    
    if not Path(cycle7_path).exists():
        print(f"❌ Cycle 7 model not found: {cycle7_path}")
        return False
    
    try:
        model = PPO.load(cycle7_path)
        print(f"✅ Loaded Cycle 7 model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create evaluation environment with Cycle 7 parameters
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
            max_episode_steps=200,
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.45,  # Cycle 7 target
            hold_bonus_weight=0.015,
            verbose=False
        )
        
        print("✅ Evaluation environment created")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Run detailed evaluation
    print("\n🔍 DETAILED EVALUATION:")
    
    total_actions = 0
    hold_actions = 0
    action_counts = {}  # Count each action type
    episode_data = []
    
    num_episodes = 12
    
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_actions = []
        episode_rewards = []
        
        for step in range(100):  # Limit steps per episode
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert numpy array to int
            episode_actions.append(action)
            action_counts[action] = action_counts.get(action, 0) + 1
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            episode_rewards.append(reward)
            
            if action == 4:  # Hold action
                hold_actions += 1
            total_actions += 1
            
            if done:
                break
        
        episode_hold_rate = episode_actions.count(4) / len(episode_actions) if episode_actions else 0
        episode_avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        episode_data.append({
            'episode': episode + 1,
            'steps': len(episode_actions),
            'hold_rate': episode_hold_rate,
            'avg_reward': episode_avg_reward,
            'actions': episode_actions[:10]  # First 10 actions for analysis
        })
        
        print(f"   Episode {episode+1:2d}: {len(episode_actions):2d} steps, {episode_hold_rate:.1%} hold, avg_reward: {episode_avg_reward:.2e}")
    
    # Calculate overall metrics
    overall_hold_rate = hold_actions / total_actions if total_actions > 0 else 0.0
    avg_episode_length = np.mean([ep['steps'] for ep in episode_data])
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Overall hold rate: {overall_hold_rate:.1%}")
    print(f"   Average episode length: {avg_episode_length:.1f}")
    print(f"   Total actions: {total_actions}")
    
    print(f"\n📈 ACTION DISTRIBUTION:")
    action_names = {0: 'Buy A', 1: 'Sell A', 2: 'Buy B', 3: 'Sell B', 4: 'Hold'}
    for action, count in sorted(action_counts.items()):
        percentage = count / total_actions * 100 if total_actions > 0 else 0
        action_name = action_names.get(action, f'Unknown({action})')
        print(f"   {action_name}: {count:4d} ({percentage:5.1f}%)")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    
    # Check if model is stuck in one action
    max_action_pct = max(action_counts.values()) / total_actions * 100 if total_actions > 0 else 0
    if max_action_pct > 80:
        dominant_action = max(action_counts, key=action_counts.get)
        action_name = action_names.get(dominant_action, f'Unknown({dominant_action})')
        print(f"   ⚠️ MODEL STUCK: {max_action_pct:.1f}% of actions are '{action_name}'")
    
    # Check episode consistency
    hold_rates = [ep['hold_rate'] for ep in episode_data]
    hold_rate_std = np.std(hold_rates)
    if hold_rate_std < 0.05:
        print(f"   ⚠️ LOW VARIANCE: Hold rates very consistent ({hold_rate_std:.3f} std)")
    
    # Check for learning signs
    first_half_hold = np.mean(hold_rates[:6])
    second_half_hold = np.mean(hold_rates[6:])
    if second_half_hold > first_half_hold + 0.02:
        print(f"   ✅ LEARNING: Hold rate improved during evaluation")
    elif first_half_hold > second_half_hold + 0.02:
        print(f"   ⚠️ DEGRADING: Hold rate decreased during evaluation")
    else:
        print(f"   ⚠️ STABLE: No clear learning trend during evaluation")
    
    # Compare with Cycle 6
    print(f"\n📊 COMPARISON WITH CYCLE 6:")
    print(f"   Cycle 6 hold rate: 13.0%")
    print(f"   Cycle 7 hold rate: {overall_hold_rate:.1%}")
    
    if overall_hold_rate < 0.05:
        print(f"   ❌ REGRESSION: Significant drop from Cycle 6")
        print(f"   🔧 LIKELY CAUSE: Controller target too aggressive (45% vs 55%)")
        print(f"   💡 SOLUTION: Increase base_hold_bonus or adjust target")
    elif overall_hold_rate < 0.10:
        print(f"   ⚠️ PARTIAL REGRESSION: Some drop from Cycle 6")
        print(f"   🔧 LIKELY CAUSE: Parameter needs fine-tuning")
    else:
        print(f"   ✅ MAINTAINED: Similar or better than Cycle 6")
    
    # Recommendations
    print(f"\n🚀 RECOMMENDATIONS:")
    
    if overall_hold_rate < 0.05:
        print(f"   1. 🔧 INCREASE base_hold_bonus to 0.020 (from 0.015)")
        print(f"   2. 🎯 ADJUST controller target to 50% (from 45%)")
        print(f"   3. 🔄 RETRY Cycle 7 with adjusted parameters")
        recommendation = "ADJUST_PARAMETERS"
    elif overall_hold_rate < 0.10:
        print(f"   1. 🔧 SLIGHT increase base_hold_bonus to 0.018")
        print(f"   2. 🚀 PROCEED to Cycle 8 with monitoring")
        print(f"   3. 📊 EXPECT gradual improvement")
        recommendation = "PROCEED_WITH_CAUTION"
    else:
        print(f"   1. ✅ PROCEED to Cycle 8")
        print(f"   2. 🎯 TARGET 25-35% hold rate")
        print(f"   3. 📈 EXPECT continued improvement")
        recommendation = "PROCEED_CONFIDENTLY"
    
    return overall_hold_rate, recommendation

def main():
    """Main function."""
    
    hold_rate, recommendation = evaluate_cycle7_comprehensive()
    
    print("\n" + "=" * 50)
    print("🎯 CYCLE 7 EVALUATION SUMMARY:")
    
    if recommendation == "ADJUST_PARAMETERS":
        print("❌ CYCLE 7 NEEDS PARAMETER ADJUSTMENT")
        print("🔧 Controller target too aggressive - causing regression")
        print("📊 Increase base_hold_bonus and retry")
        success = False
    elif recommendation == "PROCEED_WITH_CAUTION":
        print("⚠️ CYCLE 7 PARTIAL SUCCESS")
        print("🚀 Can proceed but needs monitoring")
        print("📈 Expect gradual improvement in Cycle 8")
        success = True
    else:
        print("✅ CYCLE 7 SUCCESSFUL")
        print("🚀 Ready for Cycle 8")
        print("🎯 Strong foundation for final push")
        success = True
    
    print(f"\nFinal hold rate: {hold_rate:.1%}")
    print(f"Recommendation: {recommendation}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)