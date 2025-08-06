#!/usr/bin/env python3
"""
🔍 OPTIMIZER MINI-GRID ANALYSIS
Comprehensive analysis of the 6-job optimizer mini-grid experiment
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess

def parse_training_log(log_path, job_id, learning_rate, target_kl):
    """Parse training log to extract comprehensive metrics"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract episode statistics
    ep_len_matches = re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)
    ep_rew_matches = re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)
    
    if not ep_len_matches or not ep_rew_matches:
        return None
    
    # Get final values
    final_ep_len = float(ep_len_matches[-1])
    final_ep_rew = float(ep_rew_matches[-1])
    
    # Extract KL divergence data
    kl_matches = re.findall(r'approx_kl\s+\|\s+([\d.e-]+)', content)
    kl_values = [float(kl) for kl in kl_matches]
    
    # Calculate KL ceiling hits
    kl_ceiling_hits = sum(1 for kl in kl_values if kl >= target_kl * 0.9)  # Within 90% of ceiling
    kl_ceiling_rate = (kl_ceiling_hits / len(kl_values) * 100) if kl_values else 0
    
    # Extract total episodes
    episodes_pattern = r'Total episodes:\s+(\d+)'
    episodes_match = re.search(episodes_pattern, content)
    total_episodes = int(episodes_match.group(1)) if episodes_match else 0
    
    # Extract training completion status
    training_complete = 'Training complete' in content or 'Diagnostic training successful' in content
    
    # Extract drawdown terminations
    drawdown_terminations = len(re.findall(r'Daily drawdown.*exceeded limit', content))
    
    # Calculate episode length progression
    ep_len_progression = [float(x) for x in ep_len_matches[-5:]]  # Last 5 measurements
    ep_rew_progression = [float(x) for x in ep_rew_matches[-5:]]  # Last 5 measurements
    
    # Estimate max drawdown median (from termination messages)
    dd_pattern = r'Daily drawdown -([\d.]+)% exceeded limit'
    dd_matches = re.findall(dd_pattern, content)
    dd_values = [float(dd) for dd in dd_matches]
    max_dd_median = np.median(dd_values) if dd_values else 0.0
    
    return {
        'job_id': job_id,
        'learning_rate': learning_rate,
        'target_kl': target_kl,
        'final_episode_length': final_ep_len,
        'final_episode_reward': final_ep_rew,
        'total_episodes': total_episodes,
        'training_complete': training_complete,
        'kl_ceiling_hits': kl_ceiling_hits,
        'kl_ceiling_rate': kl_ceiling_rate,
        'kl_values': kl_values,
        'drawdown_terminations': drawdown_terminations,
        'max_dd_median': max_dd_median,
        'ep_len_progression': ep_len_progression,
        'ep_rew_progression': ep_rew_progression
    }

def evaluate_success_criteria(result):
    """Evaluate if a job meets the success criteria"""
    criteria = {
        'episode_length': result['final_episode_length'] >= 80,
        'mean_reward': result['final_episode_reward'] >= -1.0,
        'kl_ceiling_rate': result['kl_ceiling_rate'] <= 20.0,
        'max_dd_median': result['max_dd_median'] < 60.0
    }
    
    criteria['all_met'] = all(criteria.values())
    return criteria

def main():
    print("🔍 OPTIMIZER MINI-GRID ANALYSIS")
    print("=" * 50)
    
    # Job configuration
    jobs = {
        'G1': {'lr': 1e-4, 'kl': 0.015, 'focus': 'Baseline (checks if bigger DD alone fixes drift)'},
        'G2': {'lr': 1e-4, 'kl': 0.0075, 'focus': 'Same LR, tighter KL brake'},
        'G3': {'lr': 7e-5, 'kl': 0.015, 'focus': 'Moderate LR, default KL'},
        'G4': {'lr': 7e-5, 'kl': 0.0075, 'focus': 'Moderate LR + tight KL'},
        'G5': {'lr': 5e-5, 'kl': 0.015, 'focus': 'Conservative LR, default KL'},
        'G6': {'lr': 5e-5, 'kl': 0.0075, 'focus': 'Conservative LR + tight KL'}
    }
    
    results = {}
    
    # Parse each job's results
    for job_id, config in jobs.items():
        log_path = Path(f"train_runs/mini_grid_{job_id}/training.log")
        
        print(f"📊 Analyzing job {job_id} (LR={config['lr']}, KL={config['kl']})...")
        
        result = parse_training_log(log_path, job_id, config['lr'], config['kl'])
        if result:
            result['focus'] = config['focus']
            results[job_id] = result
            
            print(f"   Episode length: {result['final_episode_length']:.1f} steps")
            print(f"   Episode reward: {result['final_episode_reward']:.3f}")
            print(f"   KL ceiling rate: {result['kl_ceiling_rate']:.1f}%")
            print(f"   Training complete: {result['training_complete']}")
        else:
            print(f"   ❌ Failed to parse log")
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Create comprehensive analysis
    print(f"\n📊 MINI-GRID RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Job':<4} {'LR':<8} {'KL':<8} {'Ep Len':<8} {'Ep Rew':<8} {'KL Rate':<8} {'DD Med':<8} {'Episodes':<10}")
    print("-" * 100)
    
    summary_data = []
    for job_id in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
        if job_id in results:
            result = results[job_id]
            print(f"{job_id:<4} {result['learning_rate']:<8} {result['target_kl']:<8} "
                  f"{result['final_episode_length']:<8.1f} {result['final_episode_reward']:<8.2f} "
                  f"{result['kl_ceiling_rate']:<8.1f} {result['max_dd_median']:<8.1f} "
                  f"{result['total_episodes']:<10}")
            
            summary_data.append({
                'job_id': job_id,
                'learning_rate': result['learning_rate'],
                'target_kl': result['target_kl'],
                'final_episode_length': result['final_episode_length'],
                'final_episode_reward': result['final_episode_reward'],
                'kl_ceiling_rate': result['kl_ceiling_rate'],
                'max_dd_median': result['max_dd_median'],
                'total_episodes': result['total_episodes'],
                'training_complete': result['training_complete']
            })
        else:
            print(f"{job_id:<4} {'FAILED':<60}")
    
    # Evaluate success criteria
    print(f"\n🏆 SUCCESS CRITERIA EVALUATION")
    print("=" * 80)
    print(f"{'Job':<4} {'Ep≥80':<8} {'Rew≥-1':<8} {'KL≤20%':<8} {'DD<60':<8} {'WINNER':<8}")
    print("-" * 80)
    
    winners = []
    for job_id, result in results.items():
        criteria = evaluate_success_criteria(result)
        
        ep_check = "✅" if criteria['episode_length'] else "❌"
        rew_check = "✅" if criteria['mean_reward'] else "❌"
        kl_check = "✅" if criteria['kl_ceiling_rate'] else "❌"
        dd_check = "✅" if criteria['max_dd_median'] else "❌"
        winner = "🏆 YES" if criteria['all_met'] else "❌ NO"
        
        print(f"{job_id:<4} {ep_check:<8} {rew_check:<8} {kl_check:<8} {dd_check:<8} {winner:<8}")
        
        if criteria['all_met']:
            winners.append((job_id, result))
    
    # Save results
    output_dir = Path("train_runs/mini_grid_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'mini_grid_detailed_analysis.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'mini_grid_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualizations
    create_analysis_plots(results, output_dir)
    
    # Generate recommendations
    generate_recommendations(results, winners)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

def create_analysis_plots(results, output_dir):
    """Create comprehensive analysis plots"""
    if not results:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    job_ids = list(results.keys())
    job_labels = [f"{jid}\n(LR={results[jid]['learning_rate']}, KL={results[jid]['target_kl']})" 
                  for jid in job_ids]
    
    # Episode length comparison
    episode_lengths = [results[jid]['final_episode_length'] for jid in job_ids]
    colors = ['green' if length >= 80 else 'red' for length in episode_lengths]
    
    bars1 = axes[0, 0].bar(job_labels, episode_lengths, alpha=0.7, color=colors)
    axes[0, 0].set_title('Final Episode Length by Job')
    axes[0, 0].set_ylabel('Episode Length (steps)')
    axes[0, 0].axhline(y=80, color='green', linestyle='--', label='Target (≥80)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, length in zip(bars1, episode_lengths):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{length:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Episode reward comparison
    episode_rewards = [results[jid]['final_episode_reward'] for jid in job_ids]
    colors = ['green' if reward >= -1.0 else 'red' for reward in episode_rewards]
    
    bars2 = axes[0, 1].bar(job_labels, episode_rewards, alpha=0.7, color=colors)
    axes[0, 1].set_title('Final Episode Reward by Job')
    axes[0, 1].set_ylabel('Episode Reward')
    axes[0, 1].axhline(y=-1.0, color='green', linestyle='--', label='Target (≥-1.0)')
    axes[0, 1].axhline(y=0, color='blue', linestyle='--', label='Profitable (≥0)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, reward in zip(bars2, episode_rewards):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.1 if reward >= 0 else -0.2),
                       f'{reward:.2f}', ha='center', va='bottom' if reward >= 0 else 'top',
                       fontweight='bold')
    
    # KL ceiling rate comparison
    kl_rates = [results[jid]['kl_ceiling_rate'] for jid in job_ids]
    colors = ['green' if rate <= 20 else 'red' for rate in kl_rates]
    
    bars3 = axes[1, 0].bar(job_labels, kl_rates, alpha=0.7, color=colors)
    axes[1, 0].set_title('KL Ceiling Hit Rate by Job')
    axes[1, 0].set_ylabel('KL Ceiling Rate (%)')
    axes[1, 0].axhline(y=20, color='green', linestyle='--', label='Target (≤20%)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars3, kl_rates):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Max drawdown median comparison
    dd_medians = [results[jid]['max_dd_median'] for jid in job_ids]
    colors = ['green' if dd < 60 else 'red' for dd in dd_medians]
    
    bars4 = axes[1, 1].bar(job_labels, dd_medians, alpha=0.7, color=colors)
    axes[1, 1].set_title('Max Drawdown Median by Job')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].axhline(y=60, color='green', linestyle='--', label='Target (<60%)')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, dd in zip(bars4, dd_medians):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{dd:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Episode progression (if available)
    axes[2, 0].set_title('Episode Length Progression (Last 5 Measurements)')
    axes[2, 0].set_ylabel('Episode Length')
    axes[2, 0].set_xlabel('Measurement')
    
    for jid in job_ids:
        if 'ep_len_progression' in results[jid] and results[jid]['ep_len_progression']:
            progression = results[jid]['ep_len_progression']
            axes[2, 0].plot(range(len(progression)), progression, 
                           marker='o', label=f'{jid}', alpha=0.7)
    
    axes[2, 0].axhline(y=80, color='green', linestyle='--', alpha=0.5)
    axes[2, 0].legend()
    
    # Reward progression
    axes[2, 1].set_title('Episode Reward Progression (Last 5 Measurements)')
    axes[2, 1].set_ylabel('Episode Reward')
    axes[2, 1].set_xlabel('Measurement')
    
    for jid in job_ids:
        if 'ep_rew_progression' in results[jid] and results[jid]['ep_rew_progression']:
            progression = results[jid]['ep_rew_progression']
            axes[2, 1].plot(range(len(progression)), progression, 
                           marker='o', label=f'{jid}', alpha=0.7)
    
    axes[2, 1].axhline(y=-1.0, color='green', linestyle='--', alpha=0.5)
    axes[2, 1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mini_grid_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comprehensive plots saved to: {output_dir / 'mini_grid_comprehensive_analysis.png'}")

def generate_recommendations(results, winners):
    """Generate detailed recommendations based on results"""
    print(f"\n🧠 MINI-GRID INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    if winners:
        print(f"🏆 WINNERS FOUND: {len(winners)} job(s) meet all criteria!")
        
        for job_id, result in winners:
            print(f"\n✅ WINNER: {job_id}")
            print(f"   Learning Rate: {result['learning_rate']}")
            print(f"   Target KL: {result['target_kl']}")
            print(f"   Episode Length: {result['final_episode_length']:.1f} steps")
            print(f"   Episode Reward: {result['final_episode_reward']:.3f}")
            print(f"   KL Ceiling Rate: {result['kl_ceiling_rate']:.1f}%")
            print(f"   Focus: {result['focus']}")
        
        # Recommend best winner
        best_winner = max(winners, key=lambda x: x[1]['final_episode_reward'])
        print(f"\n🎯 RECOMMENDED WINNER: {best_winner[0]}")
        print(f"   Highest reward among winners: {best_winner[1]['final_episode_reward']:.3f}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Promote {best_winner[0]} to 50K confirmation run")
        print(f"   2. Use parameters: LR={best_winner[1]['learning_rate']}, KL={best_winner[1]['target_kl']}")
        print(f"   3. Implement curriculum learning: 75% → 50% drawdown after 20K steps")
        print(f"   4. Target: Maintain episode length while achieving profitable rewards")
        
    else:
        print(f"⚠️  NO CLEAR WINNERS: No job meets all 4 criteria")
        
        # Find best performers in each category
        best_length = max(results.items(), key=lambda x: x[1]['final_episode_length'])
        best_reward = max(results.items(), key=lambda x: x[1]['final_episode_reward'])
        best_kl = min(results.items(), key=lambda x: x[1]['kl_ceiling_rate'])
        
        print(f"\n📊 BEST PERFORMERS BY CATEGORY:")
        print(f"   Best Episode Length: {best_length[0]} ({best_length[1]['final_episode_length']:.1f} steps)")
        print(f"   Best Reward: {best_reward[0]} ({best_reward[1]['final_episode_reward']:.3f})")
        print(f"   Best KL Stability: {best_kl[0]} ({best_kl[1]['kl_ceiling_rate']:.1f}% ceiling rate)")
        
        # Analyze patterns
        print(f"\n🔍 PATTERN ANALYSIS:")
        
        # Learning rate analysis
        lr_groups = {}
        for job_id, result in results.items():
            lr = result['learning_rate']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(result)
        
        print(f"   Learning Rate Impact:")
        for lr in sorted(lr_groups.keys()):
            group = lr_groups[lr]
            avg_length = np.mean([r['final_episode_length'] for r in group])
            avg_reward = np.mean([r['final_episode_reward'] for r in group])
            print(f"     LR {lr}: {avg_length:.1f} steps avg, {avg_reward:.3f} reward avg")
        
        # KL target analysis
        kl_groups = {}
        for job_id, result in results.items():
            kl = result['target_kl']
            if kl not in kl_groups:
                kl_groups[kl] = []
            kl_groups[kl].append(result)
        
        print(f"   KL Target Impact:")
        for kl in sorted(kl_groups.keys()):
            group = kl_groups[kl]
            avg_length = np.mean([r['final_episode_length'] for r in group])
            avg_reward = np.mean([r['final_episode_reward'] for r in group])
            avg_kl_rate = np.mean([r['kl_ceiling_rate'] for r in group])
            print(f"     KL {kl}: {avg_length:.1f} steps, {avg_reward:.3f} reward, {avg_kl_rate:.1f}% ceiling")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Check if we're close to targets
        close_to_length = any(r['final_episode_length'] >= 70 for r in results.values())
        close_to_reward = any(r['final_episode_reward'] >= -2.0 for r in results.values())
        
        if close_to_length and close_to_reward:
            print(f"   📈 CLOSE TO SUCCESS: Consider fine-tuning best performers")
            print(f"   🔧 Try intermediate values around best performers")
            print(f"   🎛️  Consider extending training to 15K-20K steps")
        else:
            print(f"   ⚠️  FUNDAMENTAL ISSUES REMAIN:")
            if not close_to_length:
                print(f"     - Episode length still too short (best: {best_length[1]['final_episode_length']:.1f})")
                print(f"     - Consider 85%-90% drawdown limits")
            if not close_to_reward:
                print(f"     - Rewards still very negative (best: {best_reward[1]['final_episode_reward']:.3f})")
                print(f"     - Investigate data quality and reward system")
        
        # Suggest next experiment
        if best_length[1]['final_episode_length'] >= 60:
            print(f"\n🚀 SUGGESTED NEXT EXPERIMENT:")
            print(f"   Use {best_length[0]} as baseline for 50K run")
            print(f"   Parameters: LR={best_length[1]['learning_rate']}, KL={best_length[1]['target_kl']}")
            print(f"   Goal: See if longer training improves rewards while maintaining length")

if __name__ == "__main__":
    main()