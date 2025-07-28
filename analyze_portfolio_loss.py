#!/usr/bin/env python3
"""
Portfolio Loss Analysis Tool
Investigate why the 50K training agent is consistently losing money
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_training_logs():
    """Analyze recent training logs for portfolio performance patterns"""
    logger.info("🔍 Analyzing training logs for portfolio loss patterns...")
    
    # Find the most recent training log
    log_dirs = list(Path("logs").glob("resume_50k_*"))
    if not log_dirs:
        log_dirs = list(Path("logs").glob("dual_ticker_50k_*"))
    
    if not log_dirs:
        logger.error("❌ No training logs found")
        return
    
    latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"📁 Analyzing log directory: {latest_log_dir}")
    
    # Look for training output log
    log_file = latest_log_dir / "training_output.log"
    if not log_file.exists():
        # Try alternative log locations
        log_files = list(latest_log_dir.glob("*.log"))
        if log_files:
            log_file = log_files[0]
        else:
            logger.error(f"❌ No log files found in {latest_log_dir}")
            return
    
    logger.info(f"📄 Reading log file: {log_file}")
    
    # Parse portfolio values from log
    portfolio_values = []
    trade_counts = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Portfolio:" in line and "Completed" in line:
                    # Extract trade count and portfolio value
                    # Format: "Completed 29700 trades, Portfolio: $8536.48"
                    parts = line.split("Completed ")[1].split(" trades, Portfolio: $")
                    if len(parts) == 2:
                        trade_count = int(parts[0])
                        portfolio_value = float(parts[1])
                        trade_counts.append(trade_count)
                        portfolio_values.append(portfolio_value)
        
        if not portfolio_values:
            logger.warning("⚠️ No portfolio values found in log")
            return
        
        # Analyze the data
        logger.info(f"📊 Portfolio Performance Analysis:")
        logger.info(f"   Initial Portfolio: ${portfolio_values[0]:.2f}")
        logger.info(f"   Final Portfolio: ${portfolio_values[-1]:.2f}")
        logger.info(f"   Total Loss: ${portfolio_values[0] - portfolio_values[-1]:.2f}")
        logger.info(f"   Loss Percentage: {((portfolio_values[0] - portfolio_values[-1]) / portfolio_values[0]) * 100:.2f}%")
        logger.info(f"   Trade Count Range: {trade_counts[0]} - {trade_counts[-1]}")
        
        # Calculate loss rate
        total_trades = trade_counts[-1] - trade_counts[0]
        total_loss = portfolio_values[0] - portfolio_values[-1]
        loss_per_trade = total_loss / total_trades if total_trades > 0 else 0
        
        logger.info(f"   Average Loss per Trade: ${loss_per_trade:.4f}")
        
        # Identify trend
        if len(portfolio_values) > 10:
            # Calculate moving average trend
            window = min(10, len(portfolio_values) // 2)
            early_avg = np.mean(portfolio_values[:window])
            late_avg = np.mean(portfolio_values[-window:])
            
            logger.info(f"   Early Average (first {window}): ${early_avg:.2f}")
            logger.info(f"   Late Average (last {window}): ${late_avg:.2f}")
            logger.info(f"   Trend: {'📉 Declining' if late_avg < early_avg else '📈 Improving'}")
        
        # Save analysis data
        analysis_data = pd.DataFrame({
            'trade_count': trade_counts,
            'portfolio_value': portfolio_values
        })
        
        analysis_file = f"portfolio_loss_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analysis_data.to_csv(analysis_file, index=False)
        logger.info(f"💾 Analysis data saved: {analysis_file}")
        
        return analysis_data
        
    except Exception as e:
        logger.error(f"❌ Error analyzing logs: {e}")
        return None

def analyze_reward_system():
    """Analyze the reward system configuration"""
    logger.info("🎯 Analyzing reward system configuration...")
    
    try:
        # Import the trading environment
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        
        # Check reward scaling and parameters
        logger.info("📋 Reward System Parameters:")
        
        # Create a sample environment to inspect parameters
        n_periods = 1000
        trading_days = pd.date_range('2025-07-01', periods=n_periods, freq='1min')
        
        sample_data = np.random.randn(n_periods, 12).astype(np.float32)
        sample_prices = pd.Series(100 + np.random.randn(n_periods), index=trading_days)
        
        env = DualTickerTradingEnv(
            nvda_data=sample_data,
            msft_data=sample_data,
            nvda_prices=sample_prices,
            msft_prices=sample_prices,
            trading_days=trading_days,
            initial_capital=10000,
            tc_bp=1.0,
            reward_scaling=0.01,
            max_daily_drawdown_pct=0.95
        )
        
        logger.info(f"   Initial Capital: ${env.initial_capital}")
        logger.info(f"   Transaction Cost: {env.tc_bp} basis points")
        logger.info(f"   Reward Scaling: {env.reward_scaling}")
        logger.info(f"   Max Drawdown: {env.max_daily_drawdown_pct * 100}%")
        
        # Test a few sample actions to see reward patterns
        logger.info("🧪 Testing sample actions...")
        
        obs = env.reset()
        for i in range(5):
            # Test different actions
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            logger.info(f"   Step {i+1}: Action={action}, Reward={reward:.6f}, Portfolio=${info.get('portfolio_value', 'N/A')}")
            
            if done:
                break
        
        return env
        
    except Exception as e:
        logger.error(f"❌ Error analyzing reward system: {e}")
        return None

def check_action_distribution():
    """Analyze action distribution from recent training"""
    logger.info("🎲 Analyzing action distribution patterns...")
    
    try:
        # Look for TensorBoard logs
        tb_dirs = list(Path("logs").glob("**/tensorboard"))
        if not tb_dirs:
            tb_dirs = list(Path("tensorboard_logs").glob("**/*"))
        
        if tb_dirs:
            latest_tb_dir = max(tb_dirs, key=lambda x: x.stat().st_mtime)
            logger.info(f"📊 TensorBoard logs found: {latest_tb_dir}")
            logger.info("   Use: tensorboard --logdir {} to view action distributions".format(latest_tb_dir))
        else:
            logger.warning("⚠️ No TensorBoard logs found for action analysis")
        
        # Check if we can load a recent checkpoint for action analysis
        checkpoints = list(Path("checkpoints").glob("dual_ticker_50k_*_steps.zip"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            logger.info(f"🔍 Latest checkpoint: {latest_checkpoint}")
            
            # We could load this and run a few episodes to see action patterns
            logger.info("   Checkpoint available for detailed action analysis")
        
    except Exception as e:
        logger.error(f"❌ Error checking action distribution: {e}")

def main():
    logger.info("🚨 Portfolio Loss Investigation")
    logger.info("=" * 50)
    
    # Step 1: Analyze training logs
    portfolio_data = analyze_training_logs()
    
    print("\n" + "="*50)
    
    # Step 2: Analyze reward system
    env = analyze_reward_system()
    
    print("\n" + "="*50)
    
    # Step 3: Check action distribution
    check_action_distribution()
    
    print("\n" + "="*50)
    
    logger.info("🎯 Investigation Summary:")
    logger.info("1. Portfolio shows consistent decline pattern")
    logger.info("2. Need to examine:")
    logger.info("   - Reward scaling (currently 0.01)")
    logger.info("   - Transaction costs (1 basis point)")
    logger.info("   - Action space balance")
    logger.info("   - Market data quality")
    
    logger.info("\n🔧 Recommended Next Steps:")
    logger.info("1. Load 30K checkpoint and run evaluation episodes")
    logger.info("2. Analyze action distribution and trading patterns")
    logger.info("3. Consider reward system adjustments")
    logger.info("4. Test with different reward scaling values")
    
    if portfolio_data is not None:
        logger.info(f"\n📊 Analysis data saved for further investigation")

if __name__ == "__main__":
    main()