#!/usr/bin/env python3
"""
🔍 PROGRESSIVE TRAINING ANALYSIS
Comprehensive analysis of the 3-phase progressive training plan
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

def parse_training_log(log_path, phase_name):
    """Parse training log to extract comprehensive metrics"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract episode statistics over time
    ep_len_matches = re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)
    ep_rew_matches = re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)
    
    if not ep_len_matches or not ep_rew_matches:
        return None
    
    # Convert to float arrays
    ep_lengths = [float(x) for x in ep_len_matches]
    ep_rewards = [float(x) for x in ep_rew_matches]
    
    # Extract KL divergence data
    kl_matches = re.findall(r'approx_kl\s+\|\s+([\d.e-]+)', content)
    kl_values = [float(kl) for kl in kl_matches]
    
    # Extract total episodes
    episodes_pattern = r'Total episodes:\s+(\d+)'
    episodes_match = re.search(episodes_pattern, content)
    total_episodes = int(episodes_match.group(1)) if episodes_match else 0
    
    # Extract training completion status
    training_complete = 'Training complete' in content or 'Diagnostic training successful' in content
    
    # Extract drawdown terminations
    drawdown_terminations = len(re.findall(r'Daily drawdown.*exceeded limit', content))
    
    return {
        'phase': phase_name,
        'ep_lengths': ep_lengths,
        'ep_rewards': ep_rewards,
        'kl_values': kl_values,
        'final_episode_length': ep_lengths[-1] if ep_lengths else 0,
        'final_episode_reward': ep_rewards[-1] if ep_rewards else 0,
        'total_episodes': total_episodes,
        'training_complete': training_complete,
        'drawdown_terminations': drawdown_terminations,
        'measurements': len(ep_lengths)
    }

def analyze_progression(phase_data):
    """Analyze progression across phases"""
    progression = {}
    
    for phase_name, data in phase_data.items():
        if data is None:
            continue
            
        ep_lengths = data['ep_lengths']
        ep_rewards = data['ep_rewards']
        
        # Calculate trends
        if len(ep_lengths) >= 2:
            length_trend = ep_lengths[-1] - ep_lengths[0]
            reward_trend = ep_rewards[-1] - ep_rewards[0]
        else:
            length_trend = 0
            reward_trend = 0
        
        # Calculate stability (coefficient of variation)
        length_stability = np.std(ep_lengths) / np.mean(ep_lengths) if ep_lengths else 0
        reward_stability = np.std(ep_rewards) / abs(np.mean(ep_rewards)) if ep_rewards and np.mean(ep_rewards) != 0 else 0
        
        progression[phase_name] = {
            'final_length': data['final_episode_length'],
            'final_reward': data['final_episode_reward'],
            'length_trend': length_trend,
            'reward_trend': reward_trend,
            'length_stability': length_stability,
            'reward_stability': reward_stability,
            'measurements': data['measurements']
        }
    
    return progression

def main():
    print("🔍 PROGRESSIVE TRAINING ANALYSIS")
    print("=" * 50)
    
    # Define phases and their log files
    phases = {
        'Phase 1 (50K Confirmation)': 'train_runs/confirm_G5/training.log',
        'Phase 2A (20K @ 75% DD)': 'train_runs/curriculum_G5/training_part_a.log',
        'Phase 2B (10K @ 60% DD)': 'train_runs/curriculum_G5/training_part_b.log',
        'Phase 3 (20K Profit Tune)': 'train_runs/profit_tune_G5/training.log'
    }
    
    phase_data = {}
    
    # Parse each phase
    for phase_name, log_file in phases.items():
        log_path = Path(log_file)
        
        print(f"📊 Analyzing {phase_name}...")
        
        data = parse_training_log(log_path, phase_name)
        if data:
            phase_data[phase_name] = data
            print(f"   Final episode length: {data['final_episode_length']:.1f} steps")
            print(f"   Final episode reward: {data['final_episode_reward']:.2f}")
            print(f"   Training complete: {data['training_complete']}")
            print(f"   Measurements: {data['measurements']}")
        else:
            print(f"   ❌ Failed to parse log or log not found")
    
    if not phase_data:
        print("❌ No phase data to analyze")
        return
    
    # Analyze progression
    progression = analyze_progression(phase_data)
    
    # Create comprehensive analysis
    print(f"\n📊 PROGRESSIVE TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Phase':<25} {'Final Length':<12} {'Final Reward':<12} {'Length Δ':<10} {'Reward Δ':<10}")
    print("-" * 80)
    
    summary_data = []
    for phase_name in phases.keys():
        if phase_name in progression:
            prog = progression[phase_name]
            print(f"{phase_name:<25} {prog['final_length']:<12.1f} {prog['final_reward']:<12.2f} "
                  f"{prog['length_trend']:<10.1f} {prog['reward_trend']:<10.2f}")
            
            summary_data.append({
                'phase': phase_name,
                'final_length': prog['final_length'],
                'final_reward': prog['final_reward'],
                'length_trend': prog['length_trend'],
                'reward_trend': prog['reward_trend'],
                'length_stability': prog['length_stability'],
                'reward_stability': prog['reward_stability']
            })
    
    # Save results
    output_dir = Path("train_runs/progressive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'progressive_training_summary.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'progressive_training_results.json', 'w') as f:
        json.dump(phase_data, f, indent=2, default=str)
    
    # Create visualizations
    create_progression_plots(phase_data, output_dir)
    
    # Generate insights
    generate_progressive_insights(phase_data, progression)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

def create_progression_plots(phase_data, output_dir):
    """Create comprehensive progression plots"""
    if not phase_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Combine all episode length data
    all_lengths = []
    all_rewards = []
    phase_boundaries = [0]
    phase_labels = []
    
    for phase_name, data in phase_data.items():
        if data is None:
            continue
        
        all_lengths.extend(data['ep_lengths'])
        all_rewards.extend(data['ep_rewards'])
        phase_boundaries.append(len(all_lengths))
        phase_labels.append(phase_name.split('(')[0].strip())
    
    # Episode length progression
    axes[0, 0].plot(range(len(all_lengths)), all_lengths, 'b-', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=80, color='green', linestyle='--', label='Target (80 steps)', linewidth=2)
    axes[0, 0].set_title('Episode Length Progression Across All Phases', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Episode Length (steps)')
    axes[0, 0].set_xlabel('Training Measurement')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add phase boundaries
    for i, boundary in enumerate(phase_boundaries[1:-1], 1):
        axes[0, 0].axvline(x=boundary, color='red', linestyle=':', alpha=0.7)
        if i < len(phase_labels):
            axes[0, 0].text(boundary, max(all_lengths) * 0.9, phase_labels[i], 
                           rotation=90, ha='right', va='top', fontsize=10)
    
    # Episode reward progression
    axes[0, 1].plot(range(len(all_rewards)), all_rewards, 'r-', alpha=0.7, linewidth=2)
    axes[0, 1].axhline(y=-1, color='green', linestyle='--', label='Target (-1.0)', linewidth=2)
    axes[0, 1].axhline(y=0, color='blue', linestyle='--', label='Profitable (0)', linewidth=2)
    axes[0, 1].set_title('Episode Reward Progression Across All Phases', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Episode Reward')
    axes[0, 1].set_xlabel('Training Measurement')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add phase boundaries
    for i, boundary in enumerate(phase_boundaries[1:-1], 1):
        axes[0, 1].axvline(x=boundary, color='red', linestyle=':', alpha=0.7)
    
    # Phase comparison - Final metrics
    phase_names = list(phase_data.keys())
    final_lengths = [phase_data[p]['final_episode_length'] for p in phase_names if phase_data[p]]
    final_rewards = [phase_data[p]['final_episode_reward'] for p in phase_names if phase_data[p]]
    
    # Shorten phase names for display
    short_names = [name.split('(')[0].strip() for name in phase_names if phase_data[name]]
    
    bars1 = axes[1, 0].bar(short_names, final_lengths, alpha=0.7, color='skyblue')
    axes[1, 0].axhline(y=80, color='green', linestyle='--', label='Target (80)', linewidth=2)
    axes[1, 0].set_title('Final Episode Length by Phase', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Episode Length (steps)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, length in zip(bars1, final_lengths):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{length:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Final rewards by phase
    colors = ['green' if r >= -1 else 'orange' if r >= -10 else 'red' for r in final_rewards]
    bars2 = axes[1, 1].bar(short_names, final_rewards, alpha=0.7, color=colors)
    axes[1, 1].axhline(y=-1, color='green', linestyle='--', label='Target (-1)', linewidth=2)
    axes[1, 1].axhline(y=0, color='blue', linestyle='--', label='Profitable (0)', linewidth=2)
    axes[1, 1].set_title('Final Episode Reward by Phase', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, reward in zip(bars2, final_rewards):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.5 if reward >= 0 else -1),
                       f'{reward:.1f}', ha='center', va='bottom' if reward >= 0 else 'top',
                       fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'progressive_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Progressive training plots saved to: {output_dir / 'progressive_training_analysis.png'}")

def generate_progressive_insights(phase_data, progression):
    """Generate insights from progressive training"""
    print(f"\n🧠 PROGRESSIVE TRAINING INSIGHTS")
    print("=" * 60)
    
    if not phase_data:
        print("❌ No data to analyze")
        return
    
    # Overall progression analysis
    first_phase = list(phase_data.keys())[0]
    last_phase = list(phase_data.keys())[-1]
    
    if first_phase in progression and last_phase in progression:
        start_length = progression[first_phase]['final_length']
        end_length = progression[last_phase]['final_length']
        start_reward = progression[first_phase]['final_reward']
        end_reward = progression[last_phase]['final_reward']
        
        length_improvement = end_length - start_length
        reward_improvement = end_reward - start_reward
        
        print(f"📈 OVERALL PROGRESSION:")
        print(f"   Episode Length: {start_length:.1f} → {end_length:.1f} ({length_improvement:+.1f} steps)")
        print(f"   Episode Reward: {start_reward:.2f} → {end_reward:.2f} ({reward_improvement:+.2f})")
    
    # Phase-by-phase analysis
    print(f"\n🔍 PHASE-BY-PHASE ANALYSIS:")
    
    for phase_name, prog in progression.items():
        print(f"\n{phase_name}:")
        print(f"   Final metrics: {prog['final_length']:.1f} steps, {prog['final_reward']:.2f} reward")
        print(f"   Trends: {prog['length_trend']:+.1f} steps, {prog['reward_trend']:+.2f} reward")
        print(f"   Stability: Length CV={prog['length_stability']:.3f}, Reward CV={prog['reward_stability']:.3f}")
    
    # Success criteria evaluation
    print(f"\n🏆 SUCCESS CRITERIA EVALUATION:")
    print("=" * 40)
    
    final_phase_data = phase_data[last_phase]
    final_length = final_phase_data['final_episode_length']
    final_reward = final_phase_data['final_episode_reward']
    
    length_success = final_length >= 80
    reward_success = final_reward >= -1.0
    
    print(f"✅ Episode Length: {final_length:.1f} ≥ 80" if length_success else f"❌ Episode Length: {final_length:.1f} < 80")
    print(f"✅ Profitability: {final_reward:.2f} ≥ -1.0" if reward_success else f"❌ Profitability: {final_reward:.2f} < -1.0")
    
    # Overall assessment
    print(f"\n🎯 PROGRESSIVE TRAINING ASSESSMENT:")
    print("=" * 40)
    
    if length_success and reward_success:
        print("🎉 COMPLETE SUCCESS!")
        print("✅ Both episode length and profitability targets achieved")
        print("🚀 Model ready for live trading evaluation")
    elif length_success:
        print("🎯 PARTIAL SUCCESS")
        print("✅ Episode length breakthrough locked in")
        print("⚠️  Profitability target not yet achieved")
        print("💡 Consider extended training or reward system refinement")
    elif reward_success:
        print("🎯 UNEXPECTED PATTERN")
        print("✅ Profitability achieved but episode length regressed")
        print("⚠️  May indicate overfitting or instability")
    else:
        print("⚠️  TARGETS NOT MET")
        print("❌ Both episode length and profitability need improvement")
        print("💡 Consider fundamental approach revision")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if length_success and reward_success:
        print("   🎯 Proceed to backtesting and paper trading")
        print("   📊 Validate performance on out-of-sample data")
        print("   🔄 Consider ensemble methods for robustness")
    elif length_success:
        print("   🔄 Extend profit fine-tuning phase")
        print("   🎛️  Try even lower learning rates (1e-5, 2e-6)")
        print("   📊 Investigate reward system calibration")
        print("   🔍 Analyze episode termination patterns")
    else:
        print("   🔄 Revisit fundamental approach")
        print("   📊 Investigate data quality issues")
        print("   🎛️  Consider different architectures or algorithms")
        print("   🔍 Analyze environment dynamics")
    
    # Training efficiency analysis
    total_measurements = sum(data['measurements'] for data in phase_data.values() if data)
    print(f"\n📊 TRAINING EFFICIENCY:")
    print(f"   Total measurements: {total_measurements}")
    print(f"   Phases completed: {len(phase_data)}")
    print(f"   Average measurements per phase: {total_measurements / len(phase_data):.1f}"