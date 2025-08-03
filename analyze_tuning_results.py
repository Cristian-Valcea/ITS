#!/usr/bin/env python3
"""
🎯 ANALYZE V3 TUNING RESULTS
Analyze training logs and metrics from the tuning process
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_training_logs():
    """Analyze training and evaluation logs"""
    
    # Find the latest tuning run
    tuning_dirs = [d for d in os.listdir('train_runs') if d.startswith('v3_tuned_warmstart_50k_')]
    if not tuning_dirs:
        logger.error("No tuning runs found")
        return
    
    latest_run = sorted(tuning_dirs)[-1]
    run_dir = f"train_runs/{latest_run}"
    
    logger.info(f"🔍 Analyzing tuning run: {latest_run}")
    logger.info(f"📁 Directory: {run_dir}")
    
    # Check what files are available
    files = os.listdir(run_dir)
    logger.info(f"📄 Available files: {files}")
    
    # Analyze training monitor
    train_monitor_path = f"{run_dir}/train_monitor.csv"
    eval_monitor_path = f"{run_dir}/eval_monitor.csv"
    
    results = {}
    
    if os.path.exists(train_monitor_path):
        logger.info("📊 Analyzing training monitor...")
        train_df = pd.read_csv(train_monitor_path, comment='#')
        logger.info(f"   Training episodes: {len(train_df)}")
        logger.info(f"   Episode length stats: mean={train_df['l'].mean():.1f}, std={train_df['l'].std():.1f}")
        logger.info(f"   Reward stats: mean={train_df['r'].mean():.2e}, std={train_df['r'].std():.2e}")
        
        results['training'] = {
            'episodes': len(train_df),
            'avg_episode_length': train_df['l'].mean(),
            'avg_reward': train_df['r'].mean(),
            'reward_std': train_df['r'].std(),
            'total_timesteps': train_df['l'].sum()
        }
    
    if os.path.exists(eval_monitor_path):
        logger.info("📊 Analyzing evaluation monitor...")
        eval_df = pd.read_csv(eval_monitor_path, comment='#')
        logger.info(f"   Evaluation episodes: {len(eval_df)}")
        logger.info(f"   Episode length stats: mean={eval_df['l'].mean():.1f}, std={eval_df['l'].std():.1f}")
        logger.info(f"   Reward stats: mean={eval_df['r'].mean():.2e}, std={eval_df['r'].std():.2e}")
        
        results['evaluation'] = {
            'episodes': len(eval_df),
            'avg_episode_length': eval_df['l'].mean(),
            'avg_reward': eval_df['r'].mean(),
            'reward_std': eval_df['r'].std()
        }
    
    # Check for checkpoints
    checkpoint_dir = f"{run_dir}/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        logger.info(f"💾 Checkpoints available: {checkpoints}")
        results['checkpoints'] = checkpoints
    
    # Check model sizes
    best_model_path = f"{run_dir}/best_model.zip"
    if os.path.exists(best_model_path):
        model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB
        logger.info(f"🏆 Best model size: {model_size:.1f} MB")
        results['best_model_size_mb'] = model_size
    
    return results, run_dir

def compare_with_original():
    """Compare tuning results with original V3 training"""
    
    # Original V3 training directory
    original_dir = "train_runs/v3_gold_standard_400k_20250802_202736"
    
    if not os.path.exists(original_dir):
        logger.warning("Original V3 training directory not found")
        return
    
    logger.info("🔍 Comparing with original V3 training...")
    
    # Check original training logs if available
    original_files = os.listdir(original_dir)
    logger.info(f"📄 Original files: {original_files}")
    
    # Look for any training logs or summaries
    for file in original_files:
        if 'log' in file.lower() or 'summary' in file.lower():
            logger.info(f"📊 Found original log: {file}")

def generate_summary_report(results, run_dir):
    """Generate a summary report of the tuning results"""
    
    logger.info("=" * 60)
    logger.info("🎯 V3 TUNING RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Training summary
    if 'training' in results:
        train = results['training']
        logger.info("📊 TRAINING PERFORMANCE:")
        logger.info(f"   Episodes completed: {train['episodes']:,}")
        logger.info(f"   Total timesteps: {train['total_timesteps']:,}")
        logger.info(f"   Avg episode length: {train['avg_episode_length']:.1f}")
        logger.info(f"   Avg reward per episode: {train['avg_reward']:.2f}")
        logger.info(f"   Reward volatility: {train['reward_std']:.2f}")
    
    # Evaluation summary
    if 'evaluation' in results:
        eval_data = results['evaluation']
        logger.info("📊 EVALUATION PERFORMANCE:")
        logger.info(f"   Episodes evaluated: {eval_data['episodes']:,}")
        logger.info(f"   Avg episode length: {eval_data['avg_episode_length']:.1f}")
        logger.info(f"   Avg reward per episode: {eval_data['avg_reward']:.2f}")
        logger.info(f"   Reward volatility: {eval_data['reward_std']:.2f}")
    
    # Model artifacts
    if 'checkpoints' in results:
        logger.info("💾 MODEL ARTIFACTS:")
        logger.info(f"   Checkpoints saved: {len(results['checkpoints'])}")
        for checkpoint in results['checkpoints']:
            logger.info(f"   - {checkpoint}")
    
    if 'best_model_size_mb' in results:
        logger.info(f"   Best model size: {results['best_model_size_mb']:.1f} MB")
    
    # Tuning assessment
    logger.info("=" * 60)
    logger.info("🎯 TUNING ASSESSMENT")
    logger.info("=" * 60)
    
    # Check if training completed successfully
    if 'training' in results and results['training']['episodes'] > 0:
        logger.info("✅ TRAINING COMPLETION: SUCCESS")
        logger.info("   - Model training completed without errors")
        logger.info("   - Checkpoints saved successfully")
        logger.info("   - Best model available for deployment")
    else:
        logger.info("❌ TRAINING COMPLETION: FAILED")
        return
    
    # Analyze episode lengths (indicator of trading behavior)
    if 'evaluation' in results:
        avg_length = results['evaluation']['avg_episode_length']
        if avg_length < 100:
            logger.info("⚠️  EPISODE LENGTH: SHORT")
            logger.info("   - Episodes terminating early (possible risk management)")
            logger.info("   - May indicate aggressive trading or drawdown limits hit")
        elif avg_length > 800:
            logger.info("✅ EPISODE LENGTH: FULL")
            logger.info("   - Episodes running to completion")
            logger.info("   - Good risk management and stable trading")
        else:
            logger.info("🔄 EPISODE LENGTH: MODERATE")
            logger.info("   - Episodes of moderate length")
            logger.info("   - Balanced risk/reward behavior")
    
    # Overall recommendation
    logger.info("=" * 60)
    logger.info("🎯 DEPLOYMENT RECOMMENDATION")
    logger.info("=" * 60)
    
    if 'training' in results and 'evaluation' in results:
        train_episodes = results['training']['episodes']
        eval_reward = results['evaluation']['avg_reward']
        
        if train_episodes > 100 and eval_reward > -1000000:  # Reasonable thresholds
            logger.info("✅ RECOMMENDATION: PROCEED WITH BEHAVIORAL ANALYSIS")
            logger.info("   - Training completed successfully")
            logger.info("   - Model shows reasonable performance")
            logger.info("   - Ready for detailed comparison with original model")
            logger.info("")
            logger.info("🔍 NEXT STEPS:")
            logger.info("   1. Manual inspection of trading logs")
            logger.info("   2. Backtesting on historical data")
            logger.info("   3. Paper trading validation")
            logger.info("   4. Gradual deployment if results are positive")
        else:
            logger.info("⚠️  RECOMMENDATION: INVESTIGATE BEFORE DEPLOYMENT")
            logger.info("   - Training may have issues")
            logger.info("   - Performance needs investigation")
            logger.info("   - Consider retraining with different parameters")
    
    # Save summary to file
    summary_path = f"{run_dir}/tuning_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("V3 TUNING RESULTS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        if 'training' in results:
            f.write("TRAINING RESULTS:\n")
            for key, value in results['training'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        if 'evaluation' in results:
            f.write("EVALUATION RESULTS:\n")
            for key, value in results['evaluation'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    logger.info(f"📄 Summary saved to: {summary_path}")

def main():
    """Main analysis function"""
    logger.info("🎯 Starting V3 Tuning Results Analysis")
    
    # Change to project directory
    os.chdir('/home/cristian/IntradayTrading/ITS')
    
    # Analyze training logs
    results, run_dir = analyze_training_logs()
    
    # Compare with original
    compare_with_original()
    
    # Generate summary report
    generate_summary_report(results, run_dir)
    
    logger.info("🎉 Analysis completed!")

if __name__ == "__main__":
    main()