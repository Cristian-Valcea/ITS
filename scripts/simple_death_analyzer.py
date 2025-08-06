#!/usr/bin/env python3
"""
🔍 SIMPLE EPISODE DEATH ANALYZER
Use existing evaluation infrastructure to analyze termination patterns
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

def run_detailed_evaluation(checkpoint_path, config_path, episodes=50):
    """Run evaluation and capture detailed output"""
    print(f"🔍 Running detailed evaluation...")
    
    cmd = [
        "python", "evaluate_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--eval-episodes", str(episodes),
        "--verbose"  # If available
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/cristian/IntradayTrading/ITS")
        
        if result.returncode == 0:
            return result.stdout, result.stderr
        else:
            print(f"❌ Evaluation failed:")
            print(result.stderr)
            return None, result.stderr
            
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return None, str(e)

def parse_episode_patterns(output):
    """Parse evaluation output for episode termination patterns"""
    lines = output.split('\n')
    
    patterns = {
        'episode_lengths': [],
        'termination_reasons': [],
        'portfolio_values': [],
        'drawdowns': [],
        'rewards': []
    }
    
    # Look for episode-specific information
    current_episode = None
    
    for line in lines:
        line = line.strip()
        
        # Look for episode termination messages
        if 'Episode terminated' in line or 'terminated:' in line:
            # Try to extract termination reason
            if 'drawdown' in line.lower():
                patterns['termination_reasons'].append('drawdown')
            elif 'steps' in line.lower():
                patterns['termination_reasons'].append('max_steps')
            else:
                patterns['termination_reasons'].append('other')
        
        # Look for portfolio values
        if 'Final portfolio value:' in line:
            try:
                value_str = line.split('Final portfolio value:')[1].strip()
                value_str = value_str.replace('$', '').replace(',', '')
                patterns['portfolio_values'].append(float(value_str))
            except:
                pass
        
        # Look for episode lengths in action distribution
        if 'Total steps evaluated:' in line:
            try:
                steps_str = line.split('Total steps evaluated:')[1].strip()
                total_steps = int(steps_str)
                # This is total across all episodes, we'll need to infer individual lengths
            except:
                pass
    
    return patterns

def analyze_phase1b_results():
    """Analyze the existing Phase 1B detailed results"""
    print(f"📊 Analyzing Phase 1B detailed evaluation results...")
    
    results_dir = Path("diagnostic_runs/phase1b_reward_abc/results")
    
    variants = ['A', 'B', 'C']
    analysis = {}
    
    for variant in variants:
        csv_file = results_dir / f"eval_variant_{variant}_feb2024.csv"
        json_file = results_dir / f"eval_variant_{variant}_feb2024_detailed.json"
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            
            analysis[variant] = {
                'mean_reward': df['mean_reward'].iloc[0],
                'mean_episode_length': df['mean_episode_length'].iloc[0],
                'total_episodes': df['total_episodes'].iloc[0],
                'max_episode_length': df['max_episode_length'].iloc[0],
                'min_episode_length': df['min_episode_length'].iloc[0],
                'std_episode_length': df['std_episode_length'].iloc[0]
            }
            
            print(f"\n🎯 VARIANT {variant} ANALYSIS:")
            print(f"   Mean episode length: {analysis[variant]['mean_episode_length']:.1f} steps")
            print(f"   Episode length range: {analysis[variant]['min_episode_length']}-{analysis[variant]['max_episode_length']} steps")
            print(f"   Total episodes: {analysis[variant]['total_episodes']}")
            print(f"   Mean reward: {analysis[variant]['mean_reward']:.3f}")
        
        # Try to load detailed JSON if available
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    detailed_data = json.load(f)
                    
                if 'episodes' in detailed_data:
                    episodes = detailed_data['episodes']
                    episode_lengths = [ep.get('length', 0) for ep in episodes]
                    episode_rewards = [ep.get('total_reward', 0) for ep in episodes]
                    
                    print(f"   Detailed episode data available: {len(episodes)} episodes")
                    print(f"   Episode length distribution: {np.percentile(episode_lengths, [25, 50, 75])}")
                    
                    # Analyze very short episodes
                    short_episodes = [l for l in episode_lengths if l < 10]
                    if short_episodes:
                        print(f"   ⚠️  {len(short_episodes)} episodes < 10 steps ({len(short_episodes)/len(episodes)*100:.1f}%)")
                    
                    very_short = [l for l in episode_lengths if l < 5]
                    if very_short:
                        print(f"   🚨 {len(very_short)} episodes < 5 steps ({len(very_short)/len(episodes)*100:.1f}%)")
                        
            except Exception as e:
                print(f"   ❌ Error loading detailed JSON: {e}")
    
    return analysis

def create_death_analysis_plots(analysis):
    """Create visualization plots for episode death analysis"""
    print(f"📈 Creating death analysis plots...")
    
    variants = list(analysis.keys())
    if not variants:
        print("❌ No analysis data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode length comparison
    episode_lengths = [analysis[v]['mean_episode_length'] for v in variants]
    episode_stds = [analysis[v]['std_episode_length'] for v in variants]
    
    axes[0, 0].bar(variants, episode_lengths, yerr=episode_stds, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Mean Episode Length by Variant')
    axes[0, 0].set_ylabel('Episode Length (steps)')
    axes[0, 0].axhline(y=390, color='red', linestyle='--', label='Max possible (390)')
    axes[0, 0].axhline(y=100, color='orange', linestyle='--', label='Target minimum (100)')
    axes[0, 0].legend()
    
    # Episode length ranges
    min_lengths = [analysis[v]['min_episode_length'] for v in variants]
    max_lengths = [analysis[v]['max_episode_length'] for v in variants]
    
    x_pos = np.arange(len(variants))
    axes[0, 1].bar(x_pos - 0.2, min_lengths, 0.4, label='Min Length', alpha=0.7, color='lightcoral')
    axes[0, 1].bar(x_pos + 0.2, max_lengths, 0.4, label='Max Length', alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Episode Length Range by Variant')
    axes[0, 1].set_ylabel('Episode Length (steps)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(variants)
    axes[0, 1].legend()
    
    # Mean reward comparison
    rewards = [analysis[v]['mean_reward'] for v in variants]
    colors = ['red' if r < 0 else 'green' for r in rewards]
    
    axes[1, 0].bar(variants, rewards, alpha=0.7, color=colors)
    axes[1, 0].set_title('Mean Reward by Variant')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Episode count comparison
    episode_counts = [analysis[v]['total_episodes'] for v in variants]
    
    axes[1, 1].bar(variants, episode_counts, alpha=0.7, color='gold')
    axes[1, 1].set_title('Total Episodes Completed by Variant')
    axes[1, 1].set_ylabel('Episode Count')
    
    plt.tight_layout()
    
    output_dir = Path("diagnostic_runs/episode_death_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'episode_death_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to: {output_dir / 'episode_death_analysis.png'}")

def generate_death_hypothesis(analysis):
    """Generate hypotheses about why episodes die early"""
    print(f"\n🧠 EPISODE DEATH HYPOTHESIS GENERATION")
    print("=" * 50)
    
    if not analysis:
        print("❌ No analysis data available")
        return
    
    # Calculate overall statistics
    all_lengths = [analysis[v]['mean_episode_length'] for v in analysis.keys()]
    all_rewards = [analysis[v]['mean_reward'] for v in analysis.keys()]
    
    mean_length = np.mean(all_lengths)
    mean_reward = np.mean(all_rewards)
    
    print(f"📊 OVERALL PATTERNS:")
    print(f"   Average episode length across variants: {mean_length:.1f} steps")
    print(f"   Maximum possible episode length: 390 steps")
    print(f"   Episode completion rate: {mean_length/390*100:.1f}%")
    print(f"   Average reward across variants: {mean_reward:.3f}")
    
    print(f"\n🚨 CRITICAL FINDINGS:")
    
    if mean_length < 50:
        print(f"   🔥 SEVERE: Episodes dying extremely early (<50 steps)")
        print(f"   🔍 Likely causes:")
        print(f"     - Immediate drawdown limit triggers")
        print(f"     - Policy learned catastrophic actions")
        print(f"     - Environment termination conditions too strict")
    elif mean_length < 100:
        print(f"   ⚠️  MODERATE: Episodes dying early (<100 steps)")
        print(f"   🔍 Likely causes:")
        print(f"     - Gradual drawdown accumulation")
        print(f"     - Suboptimal but not catastrophic policy")
        print(f"     - Risk management triggering appropriately")
    
    if all(r < 0 for r in all_rewards):
        print(f"   💸 ALL VARIANTS LOSING MONEY")
        print(f"   🔍 This suggests fundamental training issues, not reward system bugs")
    
    # Variant-specific analysis
    print(f"\n🔍 VARIANT-SPECIFIC PATTERNS:")
    
    for variant in analysis.keys():
        data = analysis[variant]
        print(f"\n   Variant {variant}:")
        print(f"     Episode length: {data['mean_episode_length']:.1f} ± {data['std_episode_length']:.1f}")
        print(f"     Reward: {data['mean_reward']:.3f}")
        
        if data['mean_episode_length'] < 10:
            print(f"     🚨 CRITICAL: Immediate termination pattern")
        elif data['mean_episode_length'] < 30:
            print(f"     ⚠️  WARNING: Very early termination")
        
        if data['max_episode_length'] < 100:
            print(f"     🔍 Even best episodes die early (max: {data['max_episode_length']})")
    
    print(f"\n🎯 RECOMMENDED IMMEDIATE ACTIONS:")
    print(f"   1. 🔧 RELAX DRAWDOWN LIMITS: Try 40%, 50%, 75% instead of 30%")
    print(f"   2. 📊 STEP-BY-STEP LOGGING: Add detailed termination reason tracking")
    print(f"   3. 🎛️  OPTIMIZER TUNING: Lower learning rate, tighter KL constraints")
    print(f"   4. 📈 DATA QUALITY: Check for gaps causing artificial volatility spikes")
    print(f"   5. 🔄 BASELINE TEST: Try random policy to see if environment is fundamentally broken")

def main():
    print("🔍 SIMPLE EPISODE DEATH ANALYZER")
    print("=" * 50)
    
    # Analyze existing Phase 1B results
    analysis = analyze_phase1b_results()
    
    if analysis:
        # Create visualizations
        create_death_analysis_plots(analysis)
        
        # Generate hypotheses
        generate_death_hypothesis(analysis)
        
        # Save analysis results
        output_dir = Path("diagnostic_runs/episode_death_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'death_analysis_summary.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    else:
        print("❌ No analysis data available")

if __name__ == "__main__":
    main()