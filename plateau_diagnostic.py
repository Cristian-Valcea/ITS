#!/usr/bin/env python3
"""
Plateau diagnostic for RL trading training.

Problem: Training plateaus after 100 episodes due to early-stopping on raw reward
Solution: Analyze if Sharpe ratio was still improving when training stopped
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from rolling_sharpe_tracker import RollingSharpeTracker

def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load training episode data."""
    if not Path(csv_path).exists():
        print(f"❌ Training data not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} episodes from {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def analyze_plateau_vs_sharpe(episode_df: pd.DataFrame, 
                             window_size: int = 20,
                             plateau_threshold: float = 0.01) -> Dict:
    """
    Analyze if training plateaued on reward while Sharpe was still improving.
    
    Args:
        episode_df: DataFrame with episode data
        window_size: Rolling window for Sharpe calculation
        plateau_threshold: Threshold for detecting reward plateau
    
    Returns:
        Analysis results dictionary
    """
    if len(episode_df) < window_size * 2:
        return {'status': 'INSUFFICIENT_DATA', 'message': f'Need at least {window_size * 2} episodes'}
    
    # Simulate total rewards if not available (use returns as proxy)
    if 'total_reward' not in episode_df.columns:
        episode_df['total_reward'] = episode_df.get('total_return_pct', 0) * 0.1
    
    # Calculate rolling metrics
    rolling_reward_mean = episode_df['total_reward'].rolling(window=window_size).mean()
    rolling_reward_std = episode_df['total_reward'].rolling(window=window_size).std()
    rolling_return_mean = episode_df['total_return_pct'].rolling(window=window_size).mean()
    rolling_return_std = episode_df['total_return_pct'].rolling(window=window_size).std()
    
    # Calculate rolling Sharpe (simplified)
    risk_free_daily = 0.02 / 252 * 100  # 2% annual -> daily %
    rolling_sharpe = (rolling_return_mean - risk_free_daily) / rolling_return_std * np.sqrt(252)
    rolling_sharpe = rolling_sharpe.fillna(0)
    
    # Detect reward plateau in last 50 episodes
    last_50_episodes = min(50, len(episode_df) // 2)
    recent_rewards = episode_df['total_reward'].iloc[-last_50_episodes:]
    recent_sharpe = rolling_sharpe.iloc[-last_50_episodes:]
    
    # Check if rewards plateaued
    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
    reward_plateaued = abs(reward_trend) < plateau_threshold
    
    # Check if Sharpe was still improving
    sharpe_trend = np.polyfit(range(len(recent_sharpe)), recent_sharpe, 1)[0]
    sharpe_improving = sharpe_trend > 0.01  # Positive trend
    
    # Find best Sharpe and when it occurred
    best_sharpe_idx = rolling_sharpe.idxmax()
    best_sharpe_value = rolling_sharpe.max()
    episodes_since_best_sharpe = len(episode_df) - best_sharpe_idx - 1
    
    # Determine if early stopping was premature
    premature_stop = reward_plateaued and sharpe_improving
    
    result = {
        'status': 'PREMATURE_PLATEAU' if premature_stop else 'LEGITIMATE_PLATEAU',
        'total_episodes': len(episode_df),
        'reward_plateaued': reward_plateaued,
        'reward_trend': reward_trend,
        'sharpe_improving': sharpe_improving,
        'sharpe_trend': sharpe_trend,
        'best_sharpe': best_sharpe_value,
        'best_sharpe_episode': best_sharpe_idx,
        'episodes_since_best_sharpe': episodes_since_best_sharpe,
        'final_sharpe': rolling_sharpe.iloc[-1],
        'final_reward': episode_df['total_reward'].iloc[-1],
        'avg_return_last_20': rolling_return_mean.iloc[-1],
        'sharpe_data': rolling_sharpe.values,
        'reward_data': episode_df['total_reward'].values,
        'return_data': episode_df['total_return_pct'].values
    }
    
    return result

def plot_plateau_analysis(analysis: Dict, save_path: str = None):
    """Create diagnostic plots for plateau analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 PLATEAU DIAGNOSTIC: Reward vs Sharpe Analysis', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(analysis['reward_data']) + 1)
    sharpe_episodes = range(20, len(analysis['sharpe_data']) + 20)  # Sharpe starts after window
    
    # 1. Reward evolution
    ax1.plot(episodes, analysis['reward_data'], 'b-', alpha=0.7, label='Episode Reward')
    ax1.plot(episodes, pd.Series(analysis['reward_data']).rolling(20).mean(), 'b-', linewidth=2, label='20-ep Rolling Mean')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'Reward Evolution (Trend: {analysis["reward_trend"]:+.4f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Color code based on plateau status
    color = 'red' if analysis['reward_plateaued'] else 'green'
    ax1.text(0.05, 0.95, f'Plateaued: {analysis["reward_plateaued"]}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # 2. Rolling Sharpe ratio
    ax2.plot(sharpe_episodes, analysis['sharpe_data'], 'g-', linewidth=2, label='20-ep Rolling Sharpe')
    ax2.axhline(analysis['best_sharpe'], color='green', linestyle='--', alpha=0.7, 
                label=f'Best: {analysis["best_sharpe"]:.3f}')
    ax2.axvline(analysis['best_sharpe_episode'], color='orange', linestyle=':', alpha=0.7,
                label=f'Best at Ep {analysis["best_sharpe_episode"]}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rolling Sharpe Ratio')
    ax2.set_title(f'Sharpe Evolution (Trend: {analysis["sharpe_trend"]:+.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Color code based on improvement status
    color = 'green' if analysis['sharpe_improving'] else 'red'
    ax2.text(0.05, 0.95, f'Improving: {analysis["sharpe_improving"]}', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # 3. Return distribution
    ax3.hist(analysis['return_data'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(np.mean(analysis['return_data']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(analysis["return_data"]):.2f}%')
    ax3.set_xlabel('Episode Return %')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Return Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Plateau diagnosis summary
    ax4.axis('off')
    
    # Create diagnosis text
    status_color = 'red' if analysis['status'] == 'PREMATURE_PLATEAU' else 'green'
    diagnosis_text = f"""
🔍 PLATEAU DIAGNOSIS

Status: {analysis['status']}
Total Episodes: {analysis['total_episodes']}

📊 REWARD ANALYSIS:
• Plateaued: {analysis['reward_plateaued']}
• Trend: {analysis['reward_trend']:+.4f}
• Final Value: {analysis['final_reward']:.3f}

📈 SHARPE ANALYSIS:
• Still Improving: {analysis['sharpe_improving']}
• Trend: {analysis['sharpe_trend']:+.4f}
• Best Sharpe: {analysis['best_sharpe']:.3f}
• Episodes Since Best: {analysis['episodes_since_best_sharpe']}
• Final Sharpe: {analysis['final_sharpe']:.3f}

💡 RECOMMENDATION:
"""
    
    if analysis['status'] == 'PREMATURE_PLATEAU':
        diagnosis_text += """❌ PREMATURE STOP DETECTED!
• Rewards plateaued but Sharpe still improving
• Switch to Sharpe-based early stopping
• Could have continued training longer"""
    else:
        diagnosis_text += """✅ LEGITIMATE PLATEAU
• Both reward and Sharpe have stalled
• Early stopping was appropriate
• Consider hyperparameter tuning"""
    
    ax4.text(0.05, 0.95, diagnosis_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plateau analysis plot saved: {save_path}")
    
    plt.show()

def generate_sample_plateau_data():
    """Generate sample data showing plateau scenarios."""
    np.random.seed(42)
    
    # Scenario 1: Premature plateau (reward stalls, Sharpe improving)
    episodes = 150
    
    # Rewards plateau after episode 100
    rewards_early = np.cumsum(np.random.normal(0.1, 0.5, 100))  # Improving
    rewards_plateau = np.full(50, rewards_early[-1]) + np.random.normal(0, 0.1, 50)  # Plateau
    
    # But returns (and thus Sharpe) keep improving slightly
    returns_early = np.random.normal(0.5, 2.0, 100)
    returns_late = np.random.normal(0.8, 1.8, 50)  # Better risk-adjusted performance
    
    premature_data = pd.DataFrame({
        'episode': range(episodes),
        'total_reward': np.concatenate([rewards_early, rewards_plateau]),
        'total_return_pct': np.concatenate([returns_early, returns_late]),
        'net_pnl_after_fees': np.concatenate([returns_early * 100, returns_late * 100])
    })
    
    # Scenario 2: Legitimate plateau (both stall)
    rewards_legit = np.cumsum(np.random.normal(0.05, 0.3, episodes))
    returns_legit = np.random.normal(0.2, 2.0, episodes)
    
    legitimate_data = pd.DataFrame({
        'episode': range(episodes),
        'total_reward': rewards_legit,
        'total_return_pct': returns_legit,
        'net_pnl_after_fees': returns_legit * 100
    })
    
    return {'premature': premature_data, 'legitimate': legitimate_data}

def main():
    """Run plateau diagnostic."""
    print("🔍 PLATEAU DIAGNOSTIC: Early-Stopping Analysis")
    print("=" * 50)
    
    # Get CSV path from command line or use sample data
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        episode_df = load_training_data(csv_path)
        if episode_df is None:
            return 1
        
        print(f"\n📊 Analyzing training data from: {csv_path}")
        analysis = analyze_plateau_vs_sharpe(episode_df)
        
    else:
        print("\n💡 No data file provided - using sample data for demonstration")
        sample_data = generate_sample_plateau_data()
        
        print("\n❌ PREMATURE PLATEAU EXAMPLE:")
        analysis = analyze_plateau_vs_sharpe(sample_data['premature'])
        plot_plateau_analysis(analysis, 'premature_plateau_analysis.png')
        
        print(f"Status: {analysis['status']}")
        print(f"Reward plateaued: {analysis['reward_plateaued']}")
        print(f"Sharpe improving: {analysis['sharpe_improving']}")
        
        print("\n✅ LEGITIMATE PLATEAU EXAMPLE:")
        analysis = analyze_plateau_vs_sharpe(sample_data['legitimate'])
        plot_plateau_analysis(analysis, 'legitimate_plateau_analysis.png')
        
        print(f"Status: {analysis['status']}")
        print(f"Reward plateaued: {analysis['reward_plateaued']}")
        print(f"Sharpe improving: {analysis['sharpe_improving']}")
        
        return 0
    
    # Analyze real data
    plot_plateau_analysis(analysis, f'plateau_analysis_{analysis["status"].lower()}.png')
    
    print(f"\n🎯 ANALYSIS RESULTS:")
    print(f"Status: {analysis['status']}")
    print(f"Total episodes: {analysis['total_episodes']}")
    print(f"Reward plateaued: {analysis['reward_plateaued']} (trend: {analysis['reward_trend']:+.4f})")
    print(f"Sharpe improving: {analysis['sharpe_improving']} (trend: {analysis['sharpe_trend']:+.4f})")
    print(f"Best Sharpe: {analysis['best_sharpe']:.3f} at episode {analysis['best_sharpe_episode']}")
    print(f"Episodes since best: {analysis['episodes_since_best_sharpe']}")
    
    print(f"\n💡 RECOMMENDATION:")
    if analysis['status'] == 'PREMATURE_PLATEAU':
        print("❌ PREMATURE PLATEAU DETECTED!")
        print("   → Training stopped too early based on reward stagnation")
        print("   → Sharpe ratio was still improving")
        print("   → Switch to rolling Sharpe-based early stopping")
        print("   → Could resume training or retrain with better stopping criteria")
    else:
        print("✅ LEGITIMATE PLATEAU")
        print("   → Both reward and Sharpe have stalled")
        print("   → Early stopping was appropriate")
        print("   → Consider hyperparameter tuning or architecture changes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())