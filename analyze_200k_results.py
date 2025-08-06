#!/usr/bin/env python3
"""
📊 ANALYZE 200K CURRICULUM RESULTS
Quick analysis of the completed 200K training run
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_results():
    """Analyze the 200K curriculum training results"""
    
    print("📊 ANALYZING 200K CURRICULUM TRAINING RESULTS")
    print("=" * 60)
    
    # Load monitor data
    monitor_path = Path("train_runs/stairways_v4_200k_20250804_121235/monitor.csv")
    
    if not monitor_path.exists():
        print("❌ Monitor file not found")
        return
    
    # Read the data
    df = pd.read_csv(monitor_path, skiprows=1)
    print(f"📈 Total episodes: {len(df)}")
    print(f"📊 Training duration: {df['t'].iloc[-1]:.1f} seconds ({df['t'].iloc[-1]/60:.1f} minutes)")
    
    # Phase analysis (approximate based on episode count)
    total_episodes = len(df)
    phase1_end = int(total_episodes * 0.25)  # Rough estimate for 50K/200K
    phase2_end = int(total_episodes * 0.75)  # Rough estimate for 150K/200K
    
    phase1_rewards = df['r'].iloc[:phase1_end]
    phase2_rewards = df['r'].iloc[phase1_end:phase2_end]
    phase3_rewards = df['r'].iloc[phase2_end:]
    
    print("\n📋 PHASE ANALYSIS:")
    print(f"Phase 1 (0-25%):   Mean reward: {phase1_rewards.mean():.3f}, Episodes: {len(phase1_rewards)}")
    print(f"Phase 2 (25-75%):  Mean reward: {phase2_rewards.mean():.3f}, Episodes: {len(phase2_rewards)}")
    print(f"Phase 3 (75-100%): Mean reward: {phase3_rewards.mean():.3f}, Episodes: {len(phase3_rewards)}")
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"Mean reward: {df['r'].mean():.3f}")
    print(f"Std reward: {df['r'].std():.3f}")
    print(f"Min reward: {df['r'].min():.3f}")
    print(f"Max reward: {df['r'].max():.3f}")
    print(f"Mean episode length: {df['l'].mean():.1f}")
    
    # Sharpe ratio
    sharpe = df['r'].mean() / (df['r'].std() + 1e-8)
    print(f"Sharpe ratio: {sharpe:.3f}")
    
    # Recent performance (last 100 episodes)
    recent_rewards = df['r'].tail(100)
    recent_sharpe = recent_rewards.mean() / (recent_rewards.std() + 1e-8)
    
    print("\n📈 RECENT PERFORMANCE (last 100 episodes):")
    print(f"Mean reward: {recent_rewards.mean():.3f}")
    print(f"Sharpe ratio: {recent_sharpe:.3f}")
    print(f"Mean episode length: {df['l'].tail(100).mean():.1f}")
    
    # Quality gate assessment
    print("\n🎯 QUALITY GATE ASSESSMENT:")
    
    # Mid-run gates (episodes around 50% mark)
    mid_point = len(df) // 2
    mid_rewards = df['r'].iloc[mid_point-50:mid_point+50]
    mid_sharpe = mid_rewards.mean() / (mid_rewards.std() + 1e-8)
    
    print(f"Mid-run (≈100K): Mean reward {mid_rewards.mean():.3f} (target: ≥0.75), Sharpe {mid_sharpe:.3f} (target: ≥0.25)")
    mid_reward_pass = mid_rewards.mean() >= 0.75
    mid_sharpe_pass = mid_sharpe >= 0.25
    print(f"  Mid-run gates: {'✅ PASS' if mid_reward_pass and mid_sharpe_pass else '❌ FAIL'}")
    
    # Final gates
    print(f"Final (200K): Mean reward {recent_rewards.mean():.3f} (target: ≥1.00), Sharpe {recent_sharpe:.3f} (target: ≥0.35)")
    final_reward_pass = recent_rewards.mean() >= 1.00
    final_sharpe_pass = recent_sharpe >= 0.35
    print(f"  Final gates: {'✅ PASS' if final_reward_pass and final_sharpe_pass else '❌ FAIL'}")
    
    # Trend analysis
    print("\n📈 TREND ANALYSIS:")
    
    # Split into quarters for trend
    q1 = df['r'].iloc[:len(df)//4].mean()
    q2 = df['r'].iloc[len(df)//4:len(df)//2].mean()
    q3 = df['r'].iloc[len(df)//2:3*len(df)//4].mean()
    q4 = df['r'].iloc[3*len(df)//4:].mean()
    
    print(f"Q1 (0-25%):   {q1:.3f}")
    print(f"Q2 (25-50%):  {q2:.3f}")
    print(f"Q3 (50-75%):  {q3:.3f}")
    print(f"Q4 (75-100%): {q4:.3f}")
    
    if q4 < q3 < q2:
        print("⚠️ DECLINING TREND: Curriculum tightening may have been too aggressive")
    elif q4 > q3 > q2:
        print("✅ IMPROVING TREND: Curriculum working well")
    else:
        print("📊 MIXED TREND: Some phases better than others")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    
    if recent_rewards.mean() < 0:
        print("❌ CRITICAL: Recent rewards are negative")
        print("   → Curriculum tightening was too aggressive")
        print("   → Consider using Phase 2 checkpoint (100K) as final model")
        print("   → Or retrain with gentler curriculum (DD: 20%→15%→10%)")
    elif recent_rewards.mean() < 0.5:
        print("⚠️ WARNING: Recent rewards are low but positive")
        print("   → Phase 2 model might be better than final")
        print("   → Consider validation on both checkpoints")
    else:
        print("✅ SUCCESS: Training completed successfully")
        print("   → Final model ready for paper trading")
    
    # Best checkpoint recommendation
    if q2 > q4:
        print(f"\n💡 BEST CHECKPOINT: Phase 2 (≈100K) with mean reward {q2:.3f}")
        print("   → Use checkpoint_100000.zip for deployment")
    else:
        print(f"\n💡 BEST CHECKPOINT: Final model with mean reward {q4:.3f}")
        print("   → Use stairways_v4_final_model.zip for deployment")

if __name__ == "__main__":
    analyze_training_results()