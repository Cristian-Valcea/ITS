#!/usr/bin/env python3
"""
🔍 EPISODE LENGTH ANALYSIS
Detailed analysis of episode termination patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_episode_patterns():
    """Analyze episode termination patterns"""
    
    # Load training data
    train_df = pd.read_csv('train_runs/v3_tuned_warmstart_50k_20250802_233810/train_monitor.csv', comment='#')
    eval_df = pd.read_csv('train_runs/v3_tuned_warmstart_50k_20250802_233810/eval_monitor.csv', comment='#')
    
    logger.info("🔍 EPISODE TERMINATION ANALYSIS")
    logger.info("=" * 50)
    
    # Training episodes analysis
    logger.info("📊 TRAINING EPISODES:")
    logger.info(f"   Total episodes: {len(train_df)}")
    logger.info(f"   Length stats: min={train_df['l'].min()}, max={train_df['l'].max()}, median={train_df['l'].median()}")
    
    # Categorize episode lengths
    very_short = (train_df['l'] <= 10).sum()
    short = ((train_df['l'] > 10) & (train_df['l'] <= 50)).sum()
    medium = ((train_df['l'] > 50) & (train_df['l'] <= 200)).sum()
    long = ((train_df['l'] > 200) & (train_df['l'] <= 500)).sum()
    very_long = (train_df['l'] > 500).sum()
    
    logger.info(f"   Very short (≤10 steps): {very_short} ({very_short/len(train_df)*100:.1f}%)")
    logger.info(f"   Short (11-50 steps): {short} ({short/len(train_df)*100:.1f}%)")
    logger.info(f"   Medium (51-200 steps): {medium} ({medium/len(train_df)*100:.1f}%)")
    logger.info(f"   Long (201-500 steps): {long} ({long/len(train_df)*100:.1f}%)")
    logger.info(f"   Very long (>500 steps): {very_long} ({very_long/len(train_df)*100:.1f}%)")
    
    # Evaluation episodes analysis
    logger.info("📊 EVALUATION EPISODES:")
    logger.info(f"   Total episodes: {len(eval_df)}")
    logger.info(f"   Length stats: min={eval_df['l'].min()}, max={eval_df['l'].max()}, median={eval_df['l'].median()}")
    
    # Categorize evaluation episode lengths
    eval_very_short = (eval_df['l'] <= 10).sum()
    eval_short = ((eval_df['l'] > 10) & (eval_df['l'] <= 50)).sum()
    eval_medium = ((eval_df['l'] > 50) & (eval_df['l'] <= 200)).sum()
    eval_long = ((eval_df['l'] > 200) & (eval_df['l'] <= 500)).sum()
    eval_very_long = (eval_df['l'] > 500).sum()
    
    logger.info(f"   Very short (≤10 steps): {eval_very_short} ({eval_very_short/len(eval_df)*100:.1f}%)")
    logger.info(f"   Short (11-50 steps): {eval_short} ({eval_short/len(eval_df)*100:.1f}%)")
    logger.info(f"   Medium (51-200 steps): {eval_medium} ({eval_medium/len(eval_df)*100:.1f}%)")
    logger.info(f"   Long (201-500 steps): {eval_long} ({eval_long/len(eval_df)*100:.1f}%)")
    logger.info(f"   Very long (>500 steps): {eval_very_long} ({eval_very_long/len(eval_df)*100:.1f}%)")
    
    # Reward analysis by episode length
    logger.info("💰 REWARD ANALYSIS BY EPISODE LENGTH:")
    
    # Training rewards by length category
    train_very_short_rewards = train_df[train_df['l'] <= 10]['r'].mean() if very_short > 0 else 0
    train_short_rewards = train_df[(train_df['l'] > 10) & (train_df['l'] <= 50)]['r'].mean() if short > 0 else 0
    train_medium_rewards = train_df[(train_df['l'] > 50) & (train_df['l'] <= 200)]['r'].mean() if medium > 0 else 0
    
    logger.info(f"   Very short episodes avg reward: {train_very_short_rewards:.2e}")
    logger.info(f"   Short episodes avg reward: {train_short_rewards:.2e}")
    logger.info(f"   Medium episodes avg reward: {train_medium_rewards:.2e}")
    
    # Time progression analysis
    logger.info("⏰ TIME PROGRESSION ANALYSIS:")
    
    # Look at first vs last 100 episodes
    first_100 = train_df.head(100)
    last_100 = train_df.tail(100)
    
    logger.info(f"   First 100 episodes avg length: {first_100['l'].mean():.1f}")
    logger.info(f"   Last 100 episodes avg length: {last_100['l'].mean():.1f}")
    logger.info(f"   First 100 episodes avg reward: {first_100['r'].mean():.2e}")
    logger.info(f"   Last 100 episodes avg reward: {last_100['r'].mean():.2e}")
    
    # Check if model is learning or degrading
    if last_100['l'].mean() > first_100['l'].mean():
        logger.info("   📈 TREND: Episode lengths INCREASING (model learning patience)")
    else:
        logger.info("   📉 TREND: Episode lengths DECREASING (model becoming more aggressive)")
    
    if last_100['r'].mean() > first_100['r'].mean():
        logger.info("   📈 TREND: Rewards IMPROVING")
    else:
        logger.info("   📉 TREND: Rewards DEGRADING")
    
    # Early termination hypothesis
    logger.info("🚨 EARLY TERMINATION ANALYSIS:")
    
    # Assume episodes <100 steps are early terminations (drawdown/risk limits)
    early_terminations = (train_df['l'] < 100).sum()
    early_termination_rate = early_terminations / len(train_df) * 100
    
    logger.info(f"   Episodes <100 steps: {early_terminations} ({early_termination_rate:.1f}%)")
    
    if early_termination_rate > 70:
        logger.info("   🚨 CRITICAL: >70% early terminations - likely hitting drawdown limits")
        logger.info("   💡 RECOMMENDATION: Reduce trading aggressiveness")
    elif early_termination_rate > 50:
        logger.info("   ⚠️  WARNING: >50% early terminations - excessive risk-taking")
        logger.info("   💡 RECOMMENDATION: Moderate parameter adjustment needed")
    else:
        logger.info("   ✅ ACCEPTABLE: <50% early terminations")
    
    return {
        'train_episodes': len(train_df),
        'eval_episodes': len(eval_df),
        'early_termination_rate': early_termination_rate,
        'avg_train_length': train_df['l'].mean(),
        'avg_eval_length': eval_df['l'].mean(),
        'trend_improving': last_100['l'].mean() > first_100['l'].mean()
    }

if __name__ == "__main__":
    import os
    os.chdir('/home/cristian/IntradayTrading/ITS')
    analyze_episode_patterns()