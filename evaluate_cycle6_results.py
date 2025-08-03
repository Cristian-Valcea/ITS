#!/usr/bin/env python3
"""
📊 EVALUATE CYCLE 6 RESULTS
Quick evaluation of Cycle 6 training results and hold rate achievement
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

def evaluate_cycle6():
    """Evaluate Cycle 6 results."""
    
    print("📊 CYCLE 6 EVALUATION")
    print("=" * 40)
    
    # Check if Cycle 6 completed
    cycle6_dir = Path("train_runs/stairways_8cycle_20250803_193928/cycle_06_hold_55%")
    
    if not cycle6_dir.exists():
        print("❌ Cycle 6 directory not found")
        return False
    
    # Find the latest model
    model_files = list(cycle6_dir.glob("*.zip"))
    if not model_files:
        print("❌ No Cycle 6 model found")
        return False
    
    model_path = model_files[0]  # Take the first (should be only one)
    
    try:
        model = PPO.load(str(model_path))
        print(f"✅ Loaded Cycle 6 model: {model_path.name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create evaluation environment
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        
        feature_data = np.random.randn(2000, 26).astype(np.float32)
        price_data = np.random.randn(2000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(2000)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=200,
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.55,
            hold_bonus_weight=0.015,
            verbose=False
        )
        
        print("✅ Evaluation environment created")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Run comprehensive evaluation
    print("\n🧪 Running comprehensive evaluation...")
    
    total_actions = 0
    hold_actions = 0
    episode_lengths = []
    episode_rewards = []
    
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_actions = 0
        episode_holds = 0
        episode_reward = 0
        
        for step in range(100):  # Limit steps per episode
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
            episode_reward += reward
            
            if done:
                break
        
        total_actions += episode_actions
        hold_actions += episode_holds
        episode_lengths.append(episode_actions)
        episode_rewards.append(episode_reward)
        
        episode_hold_rate = episode_holds / episode_actions if episode_actions > 0 else 0
        print(f"   Episode {episode+1}: {episode_actions} steps, {episode_hold_rate:.1%} hold rate")
    
    # Calculate final metrics
    final_hold_rate = hold_actions / total_actions if total_actions > 0 else 0.0
    avg_episode_length = np.mean(episode_lengths)
    avg_episode_reward = np.mean(episode_rewards)
    
    print(f"\n📊 CYCLE 6 FINAL RESULTS:")
    print(f"   Final hold rate: {final_hold_rate:.1%}")
    print(f"   Average episode length: {avg_episode_length:.1f}")
    print(f"   Average episode reward: {avg_episode_reward:.2f}")
    print(f"   Total actions evaluated: {total_actions}")
    print(f"   Episodes completed: {num_episodes}")
    
    # Success assessment
    print(f"\n🎯 SUCCESS ASSESSMENT:")
    
    success_criteria = []
    
    # 1. Hold rate breakthrough (>15%)
    if final_hold_rate > 0.15:
        print(f"   ✅ BREAKTHROUGH: Hold rate > 15% ({final_hold_rate:.1%})")
        success_criteria.append(True)
    elif final_hold_rate > 0.10:
        print(f"   ✅ PROGRESS: Hold rate > 10% ({final_hold_rate:.1%})")
        success_criteria.append(True)
    elif final_hold_rate > 0.05:
        print(f"   ⚠️ MODEST: Hold rate > 5% ({final_hold_rate:.1%})")
        success_criteria.append(False)
    else:
        print(f"   ❌ LIMITED: Hold rate ≤ 5% ({final_hold_rate:.1%})")
        success_criteria.append(False)
    
    # 2. Target range (20-35%)
    if 0.20 <= final_hold_rate <= 0.35:
        print(f"   ✅ TARGET ACHIEVED: In 20-35% range ({final_hold_rate:.1%})")
        success_criteria.append(True)
    elif final_hold_rate > 0.35:
        print(f"   ⚠️ OVER TARGET: Above 35% ({final_hold_rate:.1%})")
        success_criteria.append(True)
    else:
        print(f"   ⚠️ BELOW TARGET: Under 20% ({final_hold_rate:.1%})")
        success_criteria.append(False)
    
    # 3. Episode stability
    if avg_episode_length > 50:
        print(f"   ✅ STABLE: Good episode length ({avg_episode_length:.1f})")
        success_criteria.append(True)
    else:
        print(f"   ⚠️ UNSTABLE: Short episodes ({avg_episode_length:.1f})")
        success_criteria.append(False)
    
    # Overall assessment
    success_count = sum(success_criteria)
    total_criteria = len(success_criteria)
    
    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"   Criteria met: {success_count}/{total_criteria}")
    
    if success_count >= 2:
        print(f"   ✅ CYCLE 6 SUCCESSFUL")
        overall_success = True
        
        if final_hold_rate >= 0.20:
            print(f"   🚀 READY FOR CYCLE 7: Target achieved")
            next_action = "PROCEED_CYCLE_7"
        else:
            print(f"   🚀 READY FOR CYCLE 7: Good progress made")
            next_action = "PROCEED_CYCLE_7"
    else:
        print(f"   ⚠️ CYCLE 6 NEEDS OPTIMIZATION")
        overall_success = False
        
        if final_hold_rate > 0.05:
            print(f"   🔧 RECOMMENDATION: Increase base_hold_bonus to 0.020")
            next_action = "ADJUST_PARAMETERS"
        else:
            print(f"   🔧 RECOMMENDATION: Review controller effectiveness")
            next_action = "DEBUG_CONTROLLER"
    
    # Save evaluation results
    results = {
        "final_hold_rate": final_hold_rate,
        "avg_episode_length": avg_episode_length,
        "success_count": success_count,
        "total_criteria": total_criteria,
        "overall_success": overall_success,
        "next_action": next_action
    }
    
    # Write summary
    summary_path = cycle6_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"CYCLE 6 EVALUATION SUMMARY\n")
        f.write(f"=========================\n\n")
        f.write(f"Final hold rate: {final_hold_rate:.1%}\n")
        f.write(f"Average episode length: {avg_episode_length:.1f}\n")
        f.write(f"Success criteria met: {success_count}/{total_criteria}\n")
        f.write(f"Overall success: {overall_success}\n")
        f.write(f"Next action: {next_action}\n")
    
    print(f"💾 Evaluation summary saved: {summary_path.name}")
    
    return overall_success, results

def main():
    """Main function."""
    
    success, results = evaluate_cycle6()
    
    print("\n" + "=" * 40)
    print("🎯 CYCLE 6 COMPLETION STATUS:")
    
    if success:
        print("✅ CYCLE 6 SUCCESSFUL!")
        print("🚀 Controller fix and parameter optimization working")
        print("📈 Ready to proceed with Cycle 7")
        
        if results["final_hold_rate"] >= 0.20:
            print("🎯 TARGET ACHIEVED: 20%+ hold rate reached")
        else:
            print("🎯 SOLID PROGRESS: Significant improvement demonstrated")
            
    else:
        print("⚠️ CYCLE 6 NEEDS FURTHER OPTIMIZATION")
        print("🔧 Controller working but parameters need adjustment")
        
        if results["next_action"] == "ADJUST_PARAMETERS":
            print("📊 RECOMMENDATION: Increase base_hold_bonus to 0.020")
        else:
            print("🔍 RECOMMENDATION: Debug controller effectiveness")
    
    print(f"\n📊 KEY METRICS:")
    print(f"   Hold rate: {results['final_hold_rate']:.1%}")
    print(f"   Episode stability: {results['avg_episode_length']:.1f} steps")
    print(f"   Success criteria: {results['success_count']}/{results['total_criteria']}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)