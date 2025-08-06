#!/usr/bin/env python3
"""
🔍 DRAWDOWN SWEEP LOG ANALYSIS
Extract key metrics from training logs since evaluation is failing
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def parse_training_log(log_path):
    """Parse training log to extract episode statistics"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract episode length and reward from rollout logs
    ep_len_matches = re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)
    ep_rew_matches = re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)
    
    if not ep_len_matches or not ep_rew_matches:
        return None
    
    # Pair up the matches (they should be in the same order)
    matches = list(zip(ep_len_matches, ep_rew_matches))
    
    if not matches:
        return None
    
    # Get the final (last) rollout statistics
    final_ep_len, final_ep_rew = matches[-1]
    
    # Extract total episodes completed from final log
    episodes_pattern = r'Total episodes:\s+(\d+)'
    episodes_match = re.search(episodes_pattern, content)
    total_episodes = int(episodes_match.group(1)) if episodes_match else 0
    
    # Extract final average reward
    final_reward_pattern = r'Final avg reward:\s+([-\d.]+)'
    final_reward_match = re.search(final_reward_pattern, content)
    final_avg_reward = float(final_reward_match.group(1)) if final_reward_match else float(final_ep_rew)
    
    # Count drawdown terminations
    drawdown_terminations = len(re.findall(r'Daily drawdown.*exceeded limit', content))
    
    return {
        'mean_episode_length': float(final_ep_len),
        'mean_episode_reward': float(final_ep_rew),
        'total_episodes': total_episodes,
        'final_avg_reward': final_avg_reward,
        'drawdown_terminations': drawdown_terminations
    }

def main():
    print("🔍 DRAWDOWN SWEEP LOG ANALYSIS")
    print("=" * 50)
    
    log_dir = Path("diagnostic_runs/drawdown_sweep/logs")
    drawdown_limits = [0.30, 0.40, 0.50, 0.75]
    
    results = {}
    
    for dd in drawdown_limits:
        log_path = log_dir / f"dd_{dd}_15k.log"
        
        print(f"📊 Analyzing DD={dd*100:.0f}% log...")
        
        result = parse_training_log(log_path)
        if result:
            results[dd] = result
            print(f"   Mean episode length: {result['mean_episode_length']:.1f} steps")
            print(f"   Mean episode reward: {result['mean_episode_reward']:.3f}")
            print(f"   Total episodes: {result['total_episodes']}")
            print(f"   Drawdown terminations: {result['drawdown_terminations']}")
        else:
            print(f"   ❌ Failed to parse log")
    
    if not results:
        print("❌ No results extracted")
        return
    
    # Create summary
    print(f"\n📊 DRAWDOWN SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'DD Limit':<10} {'Avg Length':<12} {'Avg Reward':<12} {'Episodes':<10} {'DD Terms':<10}")
    print("-" * 70)
    
    summary_data = []
    for dd in sorted(results.keys()):
        result = results[dd]
        print(f"{dd*100:.0f}%{'':<7} {result['mean_episode_length']:<12.1f} "
              f"{result['mean_episode_reward']:<12.3f} {result['total_episodes']:<10} "
              f"{result['drawdown_terminations']:<10}")
        
        summary_data.append({
            'drawdown_limit': dd,
            'mean_episode_length': result['mean_episode_length'],
            'mean_episode_reward': result['mean_episode_reward'],
            'total_episodes': result['total_episodes'],
            'drawdown_terminations': result['drawdown_terminations']
        })
    
    # Save results
    output_dir = Path("diagnostic_runs/drawdown_sweep/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'drawdown_sweep_log_analysis.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'log_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_analysis_plots(results, output_dir)
    
    # Generate insights
    generate_insights(results)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

def create_analysis_plots(results, output_dir):
    """Create analysis plots"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    drawdown_limits = sorted(results.keys())
    dd_labels = [f"{dd*100:.0f}%" for dd in drawdown_limits]
    
    # Episode length comparison
    episode_lengths = [results[dd]['mean_episode_length'] for dd in drawdown_limits]
    
    bars1 = axes[0, 0].bar(dd_labels, episode_lengths, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Mean Episode Length by Drawdown Limit')
    axes[0, 0].set_ylabel('Episode Length (steps)')
    axes[0, 0].axhline(y=80, color='green', linestyle='--', label='Target (80 steps)')
    axes[0, 0].axhline(y=390, color='red', linestyle='--', label='Maximum (390 steps)')
    axes[0, 0].legend()
    
    # Add value labels on bars
    for bar, length in zip(bars1, episode_lengths):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{length:.1f}', ha='center', va='bottom')
    
    # Episode reward comparison
    episode_rewards = [results[dd]['mean_episode_reward'] for dd in drawdown_limits]
    colors = ['red' if r < 0 else 'green' for r in episode_rewards]
    
    bars2 = axes[0, 1].bar(dd_labels, episode_rewards, alpha=0.7, color=colors)
    axes[0, 1].set_title('Mean Episode Reward by Drawdown Limit')
    axes[0, 1].set_ylabel('Mean Episode Reward')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars2, episode_rewards):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.1 if reward >= 0 else -0.2),
                       f'{reward:.2f}', ha='center', va='bottom' if reward >= 0 else 'top')
    
    # Total episodes comparison
    total_episodes = [results[dd]['total_episodes'] for dd in drawdown_limits]
    
    bars3 = axes[1, 0].bar(dd_labels, total_episodes, alpha=0.7, color='gold')
    axes[1, 0].set_title('Total Episodes Completed (15K steps)')
    axes[1, 0].set_ylabel('Episode Count')
    
    # Add value labels on bars
    for bar, episodes in zip(bars3, total_episodes):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{episodes}', ha='center', va='bottom')
    
    # Drawdown terminations
    dd_terminations = [results[dd]['drawdown_terminations'] for dd in drawdown_limits]
    
    bars4 = axes[1, 1].bar(dd_labels, dd_terminations, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('Drawdown Terminations Count')
    axes[1, 1].set_ylabel('Termination Count')
    
    # Add value labels on bars
    for bar, terms in zip(bars4, dd_terminations):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{terms}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown_sweep_log_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plots saved to: {output_dir / 'drawdown_sweep_log_analysis.png'}")

def generate_insights(results):
    """Generate insights from the results"""
    print(f"\n🧠 DRAWDOWN SWEEP INSIGHTS")
    print("=" * 50)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    drawdown_limits = sorted(results.keys())
    
    # Find best performing drawdown limit
    best_length_dd = max(drawdown_limits, key=lambda dd: results[dd]['mean_episode_length'])
    best_reward_dd = max(drawdown_limits, key=lambda dd: results[dd]['mean_episode_reward'])
    
    print(f"🎯 BEST EPISODE LENGTH: {best_length_dd*100:.0f}% drawdown limit")
    print(f"   Mean length: {results[best_length_dd]['mean_episode_length']:.1f} steps")
    print(f"   Mean reward: {results[best_length_dd]['mean_episode_reward']:.3f}")
    
    print(f"\n💰 BEST REWARD: {best_reward_dd*100:.0f}% drawdown limit")
    print(f"   Mean reward: {results[best_reward_dd]['mean_episode_reward']:.3f}")
    print(f"   Mean length: {results[best_reward_dd]['mean_episode_length']:.1f} steps")
    
    # Check if target episode length achieved
    target_achieved = False
    for dd in drawdown_limits:
        if results[dd]['mean_episode_length'] >= 80:
            print(f"\n✅ TARGET ACHIEVED: {dd*100:.0f}% drawdown allows ≥80 step episodes")
            target_achieved = True
            break
    
    if not target_achieved:
        print(f"\n⚠️  TARGET NOT ACHIEVED: No drawdown limit allows ≥80 step episodes")
        print(f"   Best achieved: {results[best_length_dd]['mean_episode_length']:.1f} steps")
        print(f"   Still {80 - results[best_length_dd]['mean_episode_length']:.1f} steps short of target")
    
    # Analyze improvement trend
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    for i in range(1, len(drawdown_limits)):
        prev_dd = drawdown_limits[i-1]
        curr_dd = drawdown_limits[i]
        
        length_improvement = results[curr_dd]['mean_episode_length'] - results[prev_dd]['mean_episode_length']
        reward_improvement = results[curr_dd]['mean_episode_reward'] - results[prev_dd]['mean_episode_reward']
        
        print(f"   {prev_dd*100:.0f}% → {curr_dd*100:.0f}%: "
              f"{length_improvement:+.1f} steps, {reward_improvement:+.3f} reward")
    
    # Check if drawdown is still the main issue
    print(f"\n🔍 DRAWDOWN TERMINATION ANALYSIS:")
    for dd in drawdown_limits:
        total_episodes = results[dd]['total_episodes']
        dd_terminations = results[dd]['drawdown_terminations']
        dd_percentage = (dd_terminations / total_episodes * 100) if total_episodes > 0 else 0
        
        print(f"   {dd*100:.0f}%: {dd_terminations}/{total_episodes} episodes ({dd_percentage:.1f}%) terminated by drawdown")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if target_achieved:
        print(f"   ✅ Use {best_length_dd*100:.0f}% drawdown limit for next experiments")
        print(f"   🔄 Proceed to Step 3: Optimizer mini-grid")
    else:
        # Check if we're making progress
        max_length = results[best_length_dd]['mean_episode_length']
        min_length = results[min(drawdown_limits, key=lambda dd: results[dd]['mean_episode_length'])]['mean_episode_length']
        improvement = max_length - min_length
        
        if improvement > 10:
            print(f"   📈 Good progress: {improvement:.1f} step improvement from relaxing drawdown")
            print(f"   🔄 Try even higher limits: 85%, 90%, 95%")
            print(f"   🎛️  Also proceed to optimizer tuning with {best_length_dd*100:.0f}% limit")
        else:
            print(f"   ⚠️  Limited improvement: Only {improvement:.1f} step gain")
            print(f"   🔍 Drawdown may not be the main issue")
            print(f"   🎛️  Focus on optimizer tuning (Step 3)")
            print(f"   📊 Investigate data quality (Step 4)")
    
    if all(results[dd]['mean_episode_reward'] < 0 for dd in drawdown_limits):
        print(f"   💸 All models still losing money - fundamental training issues remain")
        print(f"   🔧 Consider: lower learning rate, different exploration, data preprocessing")
    
    # Check if any drawdown limit shows significantly fewer terminations
    min_terminations = min(results[dd]['drawdown_terminations'] for dd in drawdown_limits)
    max_terminations = max(results[dd]['drawdown_terminations'] for dd in drawdown_limits)
    
    if max_terminations - min_terminations > 50:
        best_dd_term = min(drawdown_limits, key=lambda dd: results[dd]['drawdown_terminations'])
        print(f"   🎯 {best_dd_term*100:.0f}% limit shows significantly fewer drawdown terminations")

if __name__ == "__main__":
    main()