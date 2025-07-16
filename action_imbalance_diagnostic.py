#!/usr/bin/env python3
"""
Action imbalance diagnostic for RL trading systems.

Problem: >800 SELL vs <40 BUY actions
Causes: Exploration skew & high trading fees
Solutions: Action-diversity bonus or higher per-trade fee in reward
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

def analyze_action_imbalance(episode_data: pd.DataFrame, 
                           action_column: str = 'action_histogram',
                           severity_threshold: float = 0.8) -> Dict:
    """
    Analyze action imbalance in trading episodes.
    
    Args:
        episode_data: DataFrame with episode summaries
        action_column: Column containing action histograms
        severity_threshold: Threshold for severe imbalance (0.8 = 80% skew)
    
    Returns:
        Analysis results dictionary
    """
    if action_column not in episode_data.columns:
        return {
            'status': 'NO_ACTION_DATA',
            'message': f'Column {action_column} not found in data'
        }
    
    # Parse action histograms
    total_actions = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
    episode_imbalances = []
    
    for idx, row in episode_data.iterrows():
        action_hist = row[action_column]
        
        # Parse action histogram (assuming format like "SELL: 45, HOLD: 12, BUY: 3")
        episode_actions = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        
        if isinstance(action_hist, str):
            try:
                # Parse string format
                for part in action_hist.split(','):
                    if ':' in part:
                        action, count = part.strip().split(':')
                        action = action.strip()
                        count = int(count.strip())
                        if action in episode_actions:
                            episode_actions[action] = count
                            total_actions[action] += count
            except:
                continue
        elif isinstance(action_hist, dict):
            # Direct dictionary format
            for action in ['SELL', 'HOLD', 'BUY']:
                count = action_hist.get(action, 0)
                episode_actions[action] = count
                total_actions[action] += count
        
        # Calculate episode imbalance
        total_trades = episode_actions['SELL'] + episode_actions['BUY']
        if total_trades > 0:
            sell_ratio = episode_actions['SELL'] / total_trades
            buy_ratio = episode_actions['BUY'] / total_trades
            imbalance_score = abs(sell_ratio - buy_ratio)
            episode_imbalances.append({
                'episode': idx,
                'sell_count': episode_actions['SELL'],
                'hold_count': episode_actions['HOLD'],
                'buy_count': episode_actions['BUY'],
                'sell_ratio': sell_ratio,
                'buy_ratio': buy_ratio,
                'imbalance_score': imbalance_score
            })
    
    if not episode_imbalances:
        return {
            'status': 'NO_VALID_DATA',
            'message': 'Could not parse action data'
        }
    
    # Overall analysis
    total_trades = total_actions['SELL'] + total_actions['BUY']
    if total_trades == 0:
        return {
            'status': 'NO_TRADES',
            'message': 'No trading actions found'
        }
    
    overall_sell_ratio = total_actions['SELL'] / total_trades
    overall_buy_ratio = total_actions['BUY'] / total_trades
    overall_imbalance = abs(overall_sell_ratio - overall_buy_ratio)
    
    # Determine severity
    if overall_imbalance >= severity_threshold:
        status = 'SEVERE_IMBALANCE'
    elif overall_imbalance >= 0.6:
        status = 'MODERATE_IMBALANCE'
    else:
        status = 'BALANCED'
    
    # Calculate statistics
    imbalance_df = pd.DataFrame(episode_imbalances)
    avg_episode_imbalance = imbalance_df['imbalance_score'].mean()
    severe_episodes = (imbalance_df['imbalance_score'] >= severity_threshold).sum()
    
    # Identify dominant action
    dominant_action = 'SELL' if overall_sell_ratio > overall_buy_ratio else 'BUY'
    dominance_ratio = max(overall_sell_ratio, overall_buy_ratio)
    
    result = {
        'status': status,
        'total_episodes': len(episode_imbalances),
        'total_actions': total_actions,
        'total_trades': total_trades,
        'sell_ratio': overall_sell_ratio,
        'buy_ratio': overall_buy_ratio,
        'hold_ratio': total_actions['HOLD'] / sum(total_actions.values()),
        'overall_imbalance': overall_imbalance,
        'avg_episode_imbalance': avg_episode_imbalance,
        'severe_episodes': severe_episodes,
        'severe_episode_rate': severe_episodes / len(episode_imbalances),
        'dominant_action': dominant_action,
        'dominance_ratio': dominance_ratio,
        'episode_data': imbalance_df
    }
    
    return result

def diagnose_imbalance_causes(analysis: Dict, episode_data: pd.DataFrame) -> Dict:
    """
    Diagnose likely causes of action imbalance.
    """
    causes = []
    recommendations = []
    
    # Check for exploration issues
    if analysis['dominance_ratio'] > 0.8:
        causes.append("EXPLORATION_SKEW")
        recommendations.append("Increase exploration rate or use action-diversity bonus")
    
    # Check fee impact
    if 'total_fees' in episode_data.columns and 'net_pnl_after_fees' in episode_data.columns:
        avg_fees = episode_data['total_fees'].mean()
        avg_pnl = episode_data['net_pnl_after_fees'].mean()
        fee_to_pnl_ratio = abs(avg_fees / max(abs(avg_pnl), 1.0))
        
        if fee_to_pnl_ratio > 0.5:  # Fees are >50% of P&L
            causes.append("HIGH_TRADING_FEES")
            recommendations.append("Reduce transaction costs or add per-trade fee penalty")
    
    # Check for reward bias
    if analysis['dominant_action'] == 'SELL':
        causes.append("SELL_BIAS")
        recommendations.append("Check if reward function favors short positions")
    elif analysis['dominant_action'] == 'BUY':
        causes.append("BUY_BIAS")
        recommendations.append("Check if reward function favors long positions")
    
    # Check hold ratio
    if analysis['hold_ratio'] > 0.7:
        causes.append("EXCESSIVE_HOLDING")
        recommendations.append("Add turnover bonus or reduce hold action rewards")
    elif analysis['hold_ratio'] < 0.1:
        causes.append("OVERTRADING")
        recommendations.append("Add trading cooldown or increase transaction costs")
    
    return {
        'likely_causes': causes,
        'recommendations': recommendations,
        'fee_analysis': {
            'avg_fees': episode_data.get('total_fees', pd.Series([0])).mean(),
            'avg_pnl': episode_data.get('net_pnl_after_fees', pd.Series([0])).mean(),
            'fee_impact': 'HIGH' if 'HIGH_TRADING_FEES' in causes else 'NORMAL'
        }
    }

def plot_action_imbalance_diagnostic(analysis: Dict, save_path: str = None):
    """Create diagnostic plots for action imbalance."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎯 ACTION IMBALANCE DIAGNOSTIC', fontsize=16, fontweight='bold')
    
    # 1. Overall action distribution
    actions = list(analysis['total_actions'].keys())
    counts = list(analysis['total_actions'].values())
    colors = ['red', 'gray', 'green']
    
    wedges, texts, autotexts = ax1.pie(counts, labels=actions, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Overall Action Distribution\n(Total: {sum(counts)} actions)')
    
    # Highlight imbalance
    if analysis['overall_imbalance'] >= 0.8:
        ax1.text(0, -1.3, f"⚠️ SEVERE IMBALANCE: {analysis['overall_imbalance']:.1%}", 
                ha='center', fontsize=12, color='red', weight='bold')
    
    # 2. Trading action ratio (SELL vs BUY only)
    trade_actions = ['SELL', 'BUY']
    trade_counts = [analysis['total_actions']['SELL'], analysis['total_actions']['BUY']]
    
    bars = ax2.bar(trade_actions, trade_counts, color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Action Count')
    ax2.set_title(f'Trading Actions: {analysis["sell_ratio"]:.1%} SELL vs {analysis["buy_ratio"]:.1%} BUY')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, trade_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(trade_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Episode-by-episode imbalance
    if 'episode_data' in analysis and not analysis['episode_data'].empty:
        episode_df = analysis['episode_data']
        ax3.plot(episode_df['episode'], episode_df['imbalance_score'], 'b-', alpha=0.7, label='Imbalance Score')
        ax3.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Severe Threshold')
        ax3.axhline(0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Imbalance Score')
        ax3.set_title('Episode Imbalance Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Diagnostic summary
    ax4.axis('off')
    
    # Status color
    status_colors = {
        'SEVERE_IMBALANCE': 'red',
        'MODERATE_IMBALANCE': 'orange',
        'BALANCED': 'green'
    }
    status_color = status_colors.get(analysis['status'], 'gray')
    
    summary_text = f"""
🎯 ACTION IMBALANCE ANALYSIS

Status: {analysis['status']}
Episodes Analyzed: {analysis['total_episodes']}

📊 ACTION BREAKDOWN:
• SELL: {analysis['total_actions']['SELL']} ({analysis['sell_ratio']:.1%})
• HOLD: {analysis['total_actions']['HOLD']} ({analysis['hold_ratio']:.1%})
• BUY: {analysis['total_actions']['BUY']} ({analysis['buy_ratio']:.1%})

⚖️ IMBALANCE METRICS:
• Overall Imbalance: {analysis['overall_imbalance']:.1%}
• Avg Episode Imbalance: {analysis['avg_episode_imbalance']:.1%}
• Severe Episodes: {analysis['severe_episodes']}/{analysis['total_episodes']} ({analysis['severe_episode_rate']:.1%})
• Dominant Action: {analysis['dominant_action']} ({analysis['dominance_ratio']:.1%})

💡 QUICK FIXES:
"""
    
    if analysis['status'] == 'SEVERE_IMBALANCE':
        summary_text += """❌ CRITICAL ACTION IMBALANCE!
• Add action-diversity bonus to reward
• Increase exploration rate
• Check fee structure
• Review reward function bias"""
    elif analysis['status'] == 'MODERATE_IMBALANCE':
        summary_text += """⚠️ MODERATE IMBALANCE
• Monitor action distribution
• Consider diversity incentives
• Check trading costs"""
    else:
        summary_text += """✅ BALANCED ACTIONS
• Action distribution is healthy
• Continue monitoring"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Action imbalance diagnostic plot saved: {save_path}")
    
    plt.show()

def generate_sample_imbalance_data():
    """Generate sample data showing action imbalance scenarios."""
    np.random.seed(42)
    
    # Scenario 1: Severe SELL bias (like the reported issue)
    severe_data = []
    for i in range(100):
        # Heavy SELL bias with some randomness
        sell_count = np.random.poisson(15) + 5  # 5-25 sells per episode
        buy_count = max(1, np.random.poisson(1))  # 1-3 buys per episode
        hold_count = np.random.poisson(8) + 2   # 2-15 holds per episode
        
        severe_data.append({
            'episode': i,
            'action_histogram': f"SELL: {sell_count}, HOLD: {hold_count}, BUY: {buy_count}",
            'total_trades': sell_count + buy_count,
            'net_pnl_after_fees': np.random.normal(-50, 200),  # Losing money
            'total_fees': (sell_count + buy_count) * 2.5  # High fees
        })
    
    # Scenario 2: Balanced actions
    balanced_data = []
    for i in range(100):
        sell_count = np.random.poisson(8) + 2
        buy_count = np.random.poisson(8) + 2   # Similar to sells
        hold_count = np.random.poisson(10) + 5
        
        balanced_data.append({
            'episode': i,
            'action_histogram': f"SELL: {sell_count}, HOLD: {hold_count}, BUY: {buy_count}",
            'total_trades': sell_count + buy_count,
            'net_pnl_after_fees': np.random.normal(25, 150),
            'total_fees': (sell_count + buy_count) * 2.5
        })
    
    return {
        'severe_imbalance': pd.DataFrame(severe_data),
        'balanced': pd.DataFrame(balanced_data)
    }

def main():
    """Run action imbalance diagnostic."""
    print("🎯 ACTION IMBALANCE DIAGNOSTIC")
    print("=" * 50)
    
    # Get CSV path from command line or use sample data
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not Path(csv_path).exists():
            print(f"❌ File not found: {csv_path}")
            return 1
        
        try:
            episode_data = pd.read_csv(csv_path)
            print(f"📊 Loaded {len(episode_data)} episodes from {csv_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return 1
    else:
        print("💡 No data file provided - using sample data for demonstration")
        sample_data = generate_sample_imbalance_data()
        
        print("\n❌ SEVERE IMBALANCE EXAMPLE (>800 SELL vs <40 BUY):")
        episode_data = sample_data['severe_imbalance']
    
    # Run analysis
    analysis = analyze_action_imbalance(episode_data)
    
    if analysis['status'] in ['NO_ACTION_DATA', 'NO_VALID_DATA', 'NO_TRADES']:
        print(f"❌ {analysis['status']}: {analysis['message']}")
        return 1
    
    # Print results
    print(f"\n🎯 ANALYSIS RESULTS:")
    print(f"Status: {analysis['status']}")
    print(f"Total episodes: {analysis['total_episodes']}")
    print(f"Total actions: {sum(analysis['total_actions'].values())}")
    print(f"Action breakdown:")
    for action, count in analysis['total_actions'].items():
        ratio = count / sum(analysis['total_actions'].values())
        print(f"  • {action}: {count} ({ratio:.1%})")
    
    print(f"\nTrading actions only:")
    print(f"  • SELL: {analysis['total_actions']['SELL']} ({analysis['sell_ratio']:.1%})")
    print(f"  • BUY: {analysis['total_actions']['BUY']} ({analysis['buy_ratio']:.1%})")
    print(f"  • Imbalance score: {analysis['overall_imbalance']:.1%}")
    
    # Diagnose causes
    diagnosis = diagnose_imbalance_causes(analysis, episode_data)
    
    print(f"\n🔍 LIKELY CAUSES:")
    for cause in diagnosis['likely_causes']:
        print(f"  • {cause}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in diagnosis['recommendations']:
        print(f"  • {rec}")
    
    # Generate diagnostic plot
    plot_action_imbalance_diagnostic(analysis, f"action_imbalance_{analysis['status'].lower()}.png")
    
    # Show sample fixes
    if analysis['status'] == 'SEVERE_IMBALANCE':
        print(f"\n🔧 SAMPLE REWARD FIXES:")
        print(f"""
# 1. Action diversity bonus
action_counts = np.bincount(recent_actions, minlength=3)
diversity_bonus = -np.sum(action_counts * np.log(action_counts + 1e-8))  # Entropy
reward += diversity_bonus * 0.1

# 2. Per-trade fee penalty
if action != 1:  # Not HOLD
    reward -= 5.0  # Higher per-trade penalty

# 3. Balance incentive
sell_buy_ratio = sell_count / max(buy_count, 1)
if sell_buy_ratio > 3.0 or sell_buy_ratio < 0.33:
    reward -= 2.0  # Penalty for extreme imbalance
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())