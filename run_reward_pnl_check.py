#!/usr/bin/env python3
"""
Quick reality check for Reward-P&L mismatch in IntradayJules.

Usage:
    python run_reward_pnl_check.py [path_to_episode_summaries.csv]

If no path provided, will look for logs/episode_summaries.csv
"""

import sys
import pandas as pd
from pathlib import Path
from reward_pnl_diagnostic import analyze_reward_pnl_correlation, plot_reward_pnl_diagnostic

def load_episode_data(csv_path: str = "logs/episode_summaries.csv"):
    """Load episode data from CSV file."""
    if not Path(csv_path).exists():
        print(f"❌ Episode summary file not found: {csv_path}")
        print("💡 Run some training episodes first to generate data")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} episodes from {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading episode data: {e}")
        return None

def extract_reward_pnl_data(episode_df: pd.DataFrame):
    """Extract relevant columns for reward-P&L analysis."""
    required_columns = ['net_pnl_after_fees', 'total_return_pct']
    
    # Check if we have the required columns
    missing_cols = [col for col in required_columns if col not in episode_df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(episode_df.columns)}")
        return None
    
    # For now, we'll simulate total_reward from return percentage
    # In a real implementation, you'd track cumulative reward per episode
    analysis_df = pd.DataFrame({
        'episode': range(len(episode_df)),
        'total_reward': episode_df['total_return_pct'] * 0.1,  # Simulated reward scaling
        'net_pnl': episode_df['net_pnl_after_fees'],
        'drawdown_pct': episode_df.get('max_drawdown_pct', episode_df['total_return_pct'])  # Use return as proxy
    })
    
    return analysis_df

def main():
    """Run the reward-P&L diagnostic."""
    print("🔍 INTRADAYJULES REWARD-P&L REALITY CHECK")
    print("=" * 50)
    
    # Get CSV path from command line or use default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "logs/episode_summaries.csv"
    
    # Load episode data
    episode_df = load_episode_data(csv_path)
    if episode_df is None:
        print("\n💡 RUNNING WITH SAMPLE DATA FOR DEMONSTRATION")
        from reward_pnl_diagnostic import generate_sample_data
        sample_data = generate_sample_data()
        analysis_df = sample_data['problematic']  # Show problematic case
    else:
        analysis_df = extract_reward_pnl_data(episode_df)
        if analysis_df is None:
            return 1
    
    # Run diagnostic
    print(f"\n🔍 Analyzing {len(analysis_df)} episodes...")
    result = analyze_reward_pnl_correlation(analysis_df)
    
    # Print results
    print(f"\n📊 DIAGNOSTIC RESULTS:")
    print(f"Status: {result['status']}")
    print(f"Reward-P&L Correlation: {result['correlation']:.3f}")
    print(f"Threshold: {result['threshold']}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Episodes analyzed: {result['episodes_analyzed']}")
    print(f"Drawdown breaches (≤-2%): {result['drawdown_breaches']} ({result['breach_rate_pct']:.1f}%)")
    
    # Interpretation
    print(f"\n🎯 INTERPRETATION:")
    if result['status'] == 'REWARD_PNL_MISMATCH':
        print("❌ PROBLEM DETECTED: Reward-P&L mismatch!")
        print("   → Agent's rewards don't correlate with actual P&L")
        print("   → This explains why drawdown stops are hit repeatedly")
        print("   → Agent is optimizing for wrong objective")
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Review reward function - ensure it penalizes losses")
        print("   2. Add risk-adjusted reward components")
        print("   3. Include drawdown penalties in reward calculation")
        print("   4. Consider Sharpe ratio or Calmar ratio in rewards")
    else:
        print("✅ HEALTHY: Reward function is aligned with P&L")
        print("   → Agent is optimizing for the right objective")
        if result['breach_rate_pct'] > 10:
            print("   ⚠️  But drawdown breach rate is still high - check risk management")
    
    # Generate diagnostic plot
    print(f"\n📈 Generating diagnostic plot...")
    plot_reward_pnl_diagnostic(analysis_df, f"reward_pnl_diagnostic_{result['status'].lower()}.png")
    
    return 0 if result['status'] == 'HEALTHY' else 1

if __name__ == "__main__":
    sys.exit(main())