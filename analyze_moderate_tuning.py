#!/usr/bin/env python3
"""
🔍 MODERATE TUNING ANALYSIS
Quick analysis of the moderate tuning results
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_moderate_tuning():
    """Analyze moderate tuning results"""
    
    # Load moderate tuning data
    moderate_dir = "train_runs/v3_moderate_tuning_50k_20250803_000806"
    
    logger.info("🔍 MODERATE TUNING ANALYSIS")
    logger.info("=" * 50)
    logger.info(f"📁 Analyzing: {moderate_dir}")
    
    # Load training data
    train_df = pd.read_csv(f"{moderate_dir}/train_monitor.csv", comment='#')
    eval_df = pd.read_csv(f"{moderate_dir}/eval_monitor.csv", comment='#')
    
    logger.info("📊 MODERATE TUNING RESULTS:")
    logger.info(f"   Training episodes: {len(train_df)}")
    logger.info(f"   Evaluation episodes: {len(eval_df)}")
    
    # Episode length analysis
    train_avg_length = train_df['l'].mean()
    eval_avg_length = eval_df['l'].mean()
    
    logger.info(f"   Training avg episode length: {train_avg_length:.1f}")
    logger.info(f"   Evaluation avg episode length: {eval_avg_length:.1f}")
    
    # Early termination analysis
    train_early_term = (train_df['l'] < 100).sum() / len(train_df) * 100
    eval_early_term = (eval_df['l'] < 100).sum() / len(eval_df) * 100
    
    logger.info(f"   Training early termination rate: {train_early_term:.1f}%")
    logger.info(f"   Evaluation early termination rate: {eval_early_term:.1f}%")
    
    # Reward analysis
    train_avg_reward = train_df['r'].mean()
    eval_avg_reward = eval_df['r'].mean()
    
    logger.info(f"   Training avg reward: {train_avg_reward:.2e}")
    logger.info(f"   Evaluation avg reward: {eval_avg_reward:.2e}")
    
    # Compare with aggressive tuning
    logger.info("=" * 50)
    logger.info("🔄 COMPARISON WITH AGGRESSIVE TUNING:")
    
    # Load aggressive tuning data for comparison
    aggressive_dir = "train_runs/v3_tuned_warmstart_50k_20250802_233810"
    try:
        agg_train_df = pd.read_csv(f"{aggressive_dir}/train_monitor.csv", comment='#')
        agg_eval_df = pd.read_csv(f"{aggressive_dir}/eval_monitor.csv", comment='#')
        
        agg_train_length = agg_train_df['l'].mean()
        agg_eval_length = agg_eval_df['l'].mean()
        agg_train_early = (agg_train_df['l'] < 100).sum() / len(agg_train_df) * 100
        agg_eval_early = (agg_eval_df['l'] < 100).sum() / len(agg_eval_df) * 100
        
        logger.info("📊 AGGRESSIVE vs MODERATE COMPARISON:")
        logger.info(f"   Episode Length (Train): {agg_train_length:.1f} → {train_avg_length:.1f} ({train_avg_length/agg_train_length:.1f}x)")
        logger.info(f"   Episode Length (Eval): {agg_eval_length:.1f} → {eval_avg_length:.1f} ({eval_avg_length/agg_eval_length:.1f}x)")
        logger.info(f"   Early Termination (Train): {agg_train_early:.1f}% → {train_early_term:.1f}% ({train_early_term-agg_train_early:+.1f}%)")
        logger.info(f"   Early Termination (Eval): {agg_eval_early:.1f}% → {eval_early_term:.1f}% ({eval_early_term-agg_eval_early:+.1f}%)")
        
    except Exception as e:
        logger.warning(f"Could not load aggressive tuning data: {str(e)}")
    
    # Assessment
    logger.info("=" * 50)
    logger.info("🎯 MODERATE TUNING ASSESSMENT:")
    
    if eval_avg_length > 300 and eval_early_term < 40:
        logger.info("✅ EXCELLENT: Episodes are much longer and more stable!")
        logger.info("   - Episodes completing successfully")
        logger.info("   - Low early termination rate")
        logger.info("   - Likely better trading behavior")
        assessment = "EXCELLENT"
    elif eval_avg_length > 150 and eval_early_term < 60:
        logger.info("✅ GOOD: Significant improvement over aggressive tuning")
        logger.info("   - Episodes are longer")
        logger.info("   - Reduced early terminations")
        logger.info("   - More stable trading")
        assessment = "GOOD"
    elif eval_avg_length > 100:
        logger.info("🔄 MODERATE: Some improvement but still issues")
        logger.info("   - Episodes somewhat longer")
        logger.info("   - Still high early termination rate")
        logger.info("   - May need further adjustment")
        assessment = "MODERATE"
    else:
        logger.info("❌ POOR: Limited improvement")
        logger.info("   - Episodes still very short")
        logger.info("   - High early termination rate")
        logger.info("   - Consider different approach")
        assessment = "POOR"
    
    # Detailed episode distribution
    logger.info("📊 EPISODE LENGTH DISTRIBUTION:")
    
    # Categorize episodes
    very_short = (eval_df['l'] <= 10).sum()
    short = ((eval_df['l'] > 10) & (eval_df['l'] <= 50)).sum()
    medium = ((eval_df['l'] > 50) & (eval_df['l'] <= 200)).sum()
    long = ((eval_df['l'] > 200) & (eval_df['l'] <= 500)).sum()
    very_long = (eval_df['l'] > 500).sum()
    
    total_episodes = len(eval_df)
    
    logger.info(f"   Very short (≤10): {very_short} ({very_short/total_episodes*100:.1f}%)")
    logger.info(f"   Short (11-50): {short} ({short/total_episodes*100:.1f}%)")
    logger.info(f"   Medium (51-200): {medium} ({medium/total_episodes*100:.1f}%)")
    logger.info(f"   Long (201-500): {long} ({long/total_episodes*100:.1f}%)")
    logger.info(f"   Very long (>500): {very_long} ({very_long/total_episodes*100:.1f}%)")
    
    # Recommendation
    logger.info("=" * 50)
    logger.info("🎯 DEPLOYMENT RECOMMENDATION:")
    
    if assessment in ["EXCELLENT", "GOOD"]:
        logger.info("✅ PROCEED WITH DEPLOYMENT PREPARATION")
        logger.info("   - Moderate tuning shows significant improvement")
        logger.info("   - Episodes are more stable")
        logger.info("   - Ready for detailed backtesting")
        logger.info("")
        logger.info("🔍 NEXT STEPS:")
        logger.info("   1. Detailed behavioral analysis")
        logger.info("   2. Backtest on historical data")
        logger.info("   3. Paper trading validation")
        logger.info("   4. Gradual live deployment")
    elif assessment == "MODERATE":
        logger.info("🔄 CONSIDER FURTHER OPTIMIZATION")
        logger.info("   - Some improvement but not optimal")
        logger.info("   - May benefit from additional tuning")
        logger.info("   - Test current version while iterating")
    else:
        logger.info("❌ RETURN TO DRAWING BOARD")
        logger.info("   - Limited improvement achieved")
        logger.info("   - Consider different tuning approach")
        logger.info("   - May need to adjust other parameters")
    
    return {
        'assessment': assessment,
        'train_avg_length': train_avg_length,
        'eval_avg_length': eval_avg_length,
        'train_early_term': train_early_term,
        'eval_early_term': eval_early_term,
        'train_avg_reward': train_avg_reward,
        'eval_avg_reward': eval_avg_reward
    }

if __name__ == "__main__":
    import os
    os.chdir('/home/cristian/IntradayTrading/ITS')
    analyze_moderate_tuning()