#!/usr/bin/env python3
"""
Fast diagnostic for Reward-P&L mismatch detection.

Symptom: Drawdown hits 2% stop repeatedly
Likely cause: Reward doesn't penalise risk; agent keeps averaging down
Fast diagnostic: Plot P&L vs. reward per episode; if correlation < 0.6 you have a reward-P&L mismatch
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def analyze_reward_pnl_correlation(episode_data: pd.DataFrame, min_correlation_threshold: float = 0.6):
    """
    Fast diagnostic for reward-P&L mismatch.
    
    Args:
        episode_data: DataFrame with columns ['episode', 'total_reward', 'net_pnl', 'drawdown_pct']
        min_correlation_threshold: Minimum acceptable correlation (default: 0.6)
    
    Returns:
        dict: Diagnostic results
    """
    if len(episode_data) < 10:
        return {
            'status': 'INSUFFICIENT_DATA',
            'message': f'Need at least 10 episodes, got {len(episode_data)}',
            'correlation': None,
            'p_value': None
        }
    
    # Calculate correlation between total reward and net P&L
    correlation, p_value = pearsonr(episode_data['total_reward'], episode_data['net_pnl'])
    
    # Count episodes hitting 2% drawdown
    drawdown_breaches = (episode_data['drawdown_pct'] <= -2.0).sum()
    breach_rate = drawdown_breaches / len(episode_data)
    
    # Diagnostic result
    status = 'HEALTHY' if correlation >= min_correlation_threshold else 'REWARD_PNL_MISMATCH'
    
    result = {
        'status': status,
        'correlation': correlation,
        'p_value': p_value,
        'threshold': min_correlation_threshold,
        'episodes_analyzed': len(episode_data),
        'drawdown_breaches': drawdown_breaches,
        'breach_rate_pct': breach_rate * 100,
        'avg_reward': episode_data['total_reward'].mean(),
        'avg_pnl': episode_data['net_pnl'].mean(),
        'reward_std': episode_data['total_reward'].std(),
        'pnl_std': episode_data['net_pnl'].std()
    }
    
    return result

def plot_reward_pnl_diagnostic(episode_data: pd.DataFrame, save_path: str = None):
    """
    Create diagnostic plots for reward-P&L analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 REWARD-P&L MISMATCH DIAGNOSTIC', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Reward vs P&L
    ax1.scatter(episode_data['total_reward'], episode_data['net_pnl'], alpha=0.6, color='blue')
    correlation, _ = pearsonr(episode_data['total_reward'], episode_data['net_pnl'])
    ax1.set_xlabel('Total Reward per Episode')
    ax1.set_ylabel('Net P&L per Episode ($)')
    ax1.set_title(f'Reward vs P&L Correlation: {correlation:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(episode_data['total_reward'], episode_data['net_pnl'], 1)
    p = np.poly1d(z)
    ax1.plot(episode_data['total_reward'], p(episode_data['total_reward']), "r--", alpha=0.8)
    
    # Color code based on correlation health
    color = 'green' if correlation >= 0.6 else 'red'
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # 2. Time series: Reward and P&L over episodes
    ax2_twin = ax2.twinx()
    ax2.plot(episode_data.index, episode_data['total_reward'], 'b-', label='Total Reward', alpha=0.7)
    ax2_twin.plot(episode_data.index, episode_data['net_pnl'], 'r-', label='Net P&L ($)', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward', color='blue')
    ax2_twin.set_ylabel('Net P&L ($)', color='red')
    ax2.set_title('Reward vs P&L Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown analysis
    drawdown_episodes = episode_data[episode_data['drawdown_pct'] <= -2.0]
    ax3.hist(episode_data['drawdown_pct'], bins=30, alpha=0.7, color='lightblue', label='All Episodes')
    if len(drawdown_episodes) > 0:
        ax3.hist(drawdown_episodes['drawdown_pct'], bins=30, alpha=0.8, color='red', label='2%+ Drawdown')
    ax3.axvline(-2.0, color='red', linestyle='--', label='2% Stop Level')
    ax3.set_xlabel('Drawdown %')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Drawdown Distribution ({len(drawdown_episodes)} breaches)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward distribution for drawdown episodes
    if len(drawdown_episodes) > 0:
        ax4.scatter(drawdown_episodes['total_reward'], drawdown_episodes['drawdown_pct'], 
                   color='red', alpha=0.7, label='Drawdown Episodes')
        ax4.scatter(episode_data[episode_data['drawdown_pct'] > -2.0]['total_reward'], 
                   episode_data[episode_data['drawdown_pct'] > -2.0]['drawdown_pct'], 
                   color='green', alpha=0.5, label='Normal Episodes')
    ax4.axhline(-2.0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Total Reward')
    ax4.set_ylabel('Drawdown %')
    ax4.set_title('Reward vs Drawdown Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Diagnostic plot saved: {save_path}")
    
    plt.show()

def generate_sample_data():
    """Generate sample data to demonstrate the diagnostic."""
    np.random.seed(42)
    n_episodes = 100
    
    # Simulate problematic scenario: rewards don't correlate with P&L
    episodes = np.arange(n_episodes)
    
    # Case 1: Healthy correlation
    healthy_pnl = np.random.normal(100, 500, n_episodes)
    healthy_reward = healthy_pnl * 0.01 + np.random.normal(0, 2, n_episodes)  # Strong correlation
    
    # Case 2: Problematic - reward mismatch
    problematic_pnl = np.random.normal(-50, 400, n_episodes)  # Losing money on average
    problematic_reward = np.random.normal(5, 3, n_episodes)   # But getting positive rewards!
    
    # Add some drawdown episodes
    drawdown_mask = np.random.choice([True, False], n_episodes, p=[0.15, 0.85])
    drawdown_pct = np.where(drawdown_mask, np.random.uniform(-5, -2, n_episodes), 
                           np.random.uniform(-1.5, 0, n_episodes))
    
    return {
        'healthy': pd.DataFrame({
            'episode': episodes,
            'total_reward': healthy_reward,
            'net_pnl': healthy_pnl,
            'drawdown_pct': drawdown_pct
        }),
        'problematic': pd.DataFrame({
            'episode': episodes,
            'total_reward': problematic_reward,
            'net_pnl': problematic_pnl,
            'drawdown_pct': drawdown_pct
        })
    }

def main():
    """Run the diagnostic with sample data."""
    print("🔍 REWARD-P&L MISMATCH DIAGNOSTIC")
    print("=" * 50)
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Test healthy case
    print("\n✅ HEALTHY CASE:")
    healthy_result = analyze_reward_pnl_correlation(sample_data['healthy'])
    print(f"Status: {healthy_result['status']}")
    print(f"Correlation: {healthy_result['correlation']:.3f}")
    print(f"Drawdown breaches: {healthy_result['drawdown_breaches']}/{healthy_result['episodes_analyzed']} ({healthy_result['breach_rate_pct']:.1f}%)")
    
    # Test problematic case
    print("\n❌ PROBLEMATIC CASE:")
    problematic_result = analyze_reward_pnl_correlation(sample_data['problematic'])
    print(f"Status: {problematic_result['status']}")
    print(f"Correlation: {problematic_result['correlation']:.3f}")
    print(f"Drawdown breaches: {problematic_result['drawdown_breaches']}/{problematic_result['episodes_analyzed']} ({problematic_result['breach_rate_pct']:.1f}%)")
    
    # Generate diagnostic plots
    print("\n📊 Generating diagnostic plots...")
    plot_reward_pnl_diagnostic(sample_data['healthy'], 'healthy_reward_pnl_diagnostic.png')
    plot_reward_pnl_diagnostic(sample_data['problematic'], 'problematic_reward_pnl_diagnostic.png')
    
    print("\n💡 INTERPRETATION:")
    print("- Correlation < 0.6: REWARD-P&L MISMATCH detected")
    print("- High drawdown breach rate: Agent not learning risk management")
    print("- Positive rewards with negative P&L: Reward function is broken")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())