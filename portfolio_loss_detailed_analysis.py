#!/usr/bin/env python3
"""
Detailed Portfolio Loss Analysis
Extract and analyze the exact portfolio decline pattern
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def extract_portfolio_data():
    """Extract portfolio data from training logs"""
    log_file = "logs/resume_50k_20250728_214805/training_errors.log"
    
    portfolio_data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Portfolio:" in line and "Completed" in line:
                # Extract using regex
                match = re.search(r'Completed (\d+) trades, Portfolio: \$(\d+\.\d+)', line)
                if match:
                    trade_count = int(match.group(1))
                    portfolio_value = float(match.group(2))
                    portfolio_data.append({
                        'trade_count': trade_count,
                        'portfolio_value': portfolio_value
                    })
    
    return pd.DataFrame(portfolio_data)

def analyze_loss_pattern(df):
    """Analyze the loss pattern in detail"""
    print("🚨 DETAILED PORTFOLIO LOSS ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_loss = initial_value - final_value
    loss_percentage = (total_loss / initial_value) * 100
    
    print(f"📊 Portfolio Performance:")
    print(f"   Initial Value: ${initial_value:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Loss: ${total_loss:,.2f}")
    print(f"   Loss Percentage: {loss_percentage:.2f}%")
    print(f"   Total Trades: {df['trade_count'].iloc[-1] - df['trade_count'].iloc[0]:,}")
    
    # Calculate loss per trade
    total_trades = df['trade_count'].iloc[-1] - df['trade_count'].iloc[0]
    loss_per_trade = total_loss / total_trades
    print(f"   Average Loss per Trade: ${loss_per_trade:.4f}")
    
    # Analyze trend consistency
    df['portfolio_change'] = df['portfolio_value'].diff()
    negative_changes = (df['portfolio_change'] < 0).sum()
    total_changes = len(df['portfolio_change'].dropna())
    
    print(f"\n📉 Loss Pattern Analysis:")
    print(f"   Negative Changes: {negative_changes}/{total_changes} ({negative_changes/total_changes*100:.1f}%)")
    print(f"   Average Change per 100 trades: ${df['portfolio_change'].mean():.4f}")
    
    # Check for any recovery periods
    df['rolling_change'] = df['portfolio_value'].rolling(10).mean().diff()
    recovery_periods = (df['rolling_change'] > 0).sum()
    print(f"   Recovery Periods (10-trade windows): {recovery_periods}")
    
    # Calculate maximum consecutive losses
    consecutive_losses = 0
    max_consecutive = 0
    for change in df['portfolio_change'].dropna():
        if change < 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0
    
    print(f"   Maximum Consecutive Losses: {max_consecutive}")
    
    return df

def identify_critical_issues(df):
    """Identify the most critical issues causing losses"""
    print(f"\n🔍 CRITICAL ISSUE IDENTIFICATION:")
    print("=" * 40)
    
    # Issue 1: Consistent decline rate
    loss_rate = (df['portfolio_value'].iloc[-1] - df['portfolio_value'].iloc[0]) / len(df)
    print(f"1. 📉 CONSISTENT DECLINE RATE:")
    print(f"   Loss per measurement: ${loss_rate:.4f}")
    print(f"   This suggests systematic over-trading or poor reward signal")
    
    # Issue 2: No recovery periods
    recovery_count = (df['portfolio_value'].diff() > 0).sum()
    print(f"\n2. 🚫 LACK OF RECOVERY:")
    print(f"   Positive movements: {recovery_count}/{len(df)-1}")
    print(f"   Recovery rate: {recovery_count/(len(df)-1)*100:.1f}%")
    print(f"   This indicates the agent never learns profitable strategies")
    
    # Issue 3: Transaction cost analysis
    total_trades = df['trade_count'].iloc[-1] - df['trade_count'].iloc[0]
    total_loss = df['portfolio_value'].iloc[0] - df['portfolio_value'].iloc[-1]
    
    # Assuming 1 basis point transaction cost (0.0001)
    estimated_transaction_costs = total_trades * 10000 * 0.0001  # Assuming $10k average position
    
    print(f"\n3. 💸 TRANSACTION COST IMPACT:")
    print(f"   Total trades: {total_trades:,}")
    print(f"   Estimated transaction costs: ${estimated_transaction_costs:.2f}")
    print(f"   Transaction costs as % of loss: {estimated_transaction_costs/total_loss*100:.1f}%")
    
    if estimated_transaction_costs > total_loss * 0.5:
        print(f"   ⚠️  HIGH IMPACT: Transaction costs are major factor")
    else:
        print(f"   ℹ️  MODERATE IMPACT: Other factors dominate")

def generate_recommendations():
    """Generate specific recommendations to fix the issues"""
    print(f"\n🎯 RECOMMENDED FIXES:")
    print("=" * 30)
    
    print("1. 🔧 IMMEDIATE FIXES:")
    print("   a) Reduce reward scaling from 0.01 to 0.001")
    print("   b) Increase transaction cost penalty")
    print("   c) Add hold action bias to reduce over-trading")
    print("   d) Implement minimum holding period")
    
    print("\n2. 🧪 REWARD SYSTEM FIXES:")
    print("   a) Add Sharpe ratio component to reward")
    print("   b) Penalize excessive trading frequency")
    print("   c) Reward portfolio stability")
    print("   d) Add drawdown penalties")
    
    print("\n3. 📊 DATA QUALITY FIXES:")
    print("   a) Use real market data instead of mock data")
    print("   b) Ensure proper technical indicators")
    print("   c) Add market regime detection")
    print("   d) Include volatility-adjusted rewards")
    
    print("\n4. 🎲 ACTION SPACE FIXES:")
    print("   a) Bias toward hold actions initially")
    print("   b) Implement position sizing limits")
    print("   c) Add portfolio rebalancing constraints")
    print("   d) Reduce action frequency")

def main():
    # Extract data
    df = extract_portfolio_data()
    
    if df.empty:
        print("❌ No portfolio data found")
        return
    
    # Analyze pattern
    df = analyze_loss_pattern(df)
    
    # Identify issues
    identify_critical_issues(df)
    
    # Generate recommendations
    generate_recommendations()
    
    # Save analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_file = f"portfolio_loss_analysis_{timestamp}.csv"
    df.to_csv(analysis_file, index=False)
    print(f"\n💾 Analysis data saved: {analysis_file}")
    
    print(f"\n🚨 CONCLUSION:")
    print("The agent is exhibiting classic over-trading behavior with")
    print("consistent losses and no learning of profitable strategies.")
    print("This requires immediate reward system and action space fixes.")

if __name__ == "__main__":
    main()