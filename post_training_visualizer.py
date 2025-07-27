#!/usr/bin/env python3
"""
Post-Training Visualization Launcher

This script automatically creates performance visualizations after training completes.
It monitors for training completion and then generates comprehensive plots.
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Set up logging for the visualizer with Unicode support."""
    import sys
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/post_training_visualizer.log', encoding='utf-8')
        ]
    )
    
    # Set console output to UTF-8 if possible
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    
    return logging.getLogger('PostTrainingVisualizer')

def check_training_completion(log_file: str) -> bool:
    """Check if training has completed by monitoring the log file."""
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for training completion indicators
            completion_indicators = [
                'Training completed',
                'Training finished',
                'Model saved successfully',
                'Evaluation completed',
                'Training session ended'
            ]
            
            return any(indicator in content for indicator in completion_indicators)
    
    except Exception:
        return False

def wait_for_training_completion(log_file: str, check_interval: int = 30, max_wait_hours: int = 12):
    """Wait for training to complete."""
    logger = logging.getLogger('PostTrainingVisualizer')
    max_checks = (max_wait_hours * 3600) // check_interval
    checks = 0
    
    logger.info(f"Monitoring training completion in: {log_file}")
    logger.info(f"Check interval: {check_interval} seconds, Max wait: {max_wait_hours} hours")
    
    while checks < max_checks:
        if check_training_completion(log_file):
            logger.info("Training completion detected!")
            return True
        
        checks += 1
        remaining_hours = ((max_checks - checks) * check_interval) / 3600
        logger.info(f"Training still running... (Check {checks}/{max_checks}, ~{remaining_hours:.1f}h remaining)")
        time.sleep(check_interval)
    
    logger.warning(f"Maximum wait time ({max_wait_hours} hours) exceeded")
    return False

def create_visualizations():
    """Create performance visualizations."""
    logger = logging.getLogger('PostTrainingVisualizer')
    
    try:
        logger.info("Starting post-training visualization creation...")
        
        # Import required modules
        from evaluation.performance_visualizer import PerformanceVisualizer
        from evaluation.metrics_calculator import MetricsCalculator
        from evaluation.report_generator import ReportGenerator
        import yaml
        import pandas as pd
        
        # Load configuration
        config_path = 'config/main_config_orchestrator_gpu_fixed.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        visualizer = PerformanceVisualizer(config['orchestrator'])
        metrics_calc = MetricsCalculator(config)
        report_gen = ReportGenerator(config['orchestrator'])
        
        # Try to load trade log and calculate metrics
        reports_dir = config['orchestrator'].get('reports_dir', 'reports/orch_gpu_fixed')
        
        # Look for the most recent trade log
        trade_log_files = []
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                if file.endswith('_trades.csv'):
                    trade_log_files.append(os.path.join(reports_dir, file))
        
        trade_log_df = None
        if trade_log_files:
            # Use the most recent trade log
            latest_trade_log = max(trade_log_files, key=os.path.getmtime)
            logger.info(f"Loading trade log: {latest_trade_log}")
            trade_log_df = pd.read_csv(latest_trade_log)
        
        # Calculate metrics (use dummy data if no trade log available)
        if trade_log_df is not None and not trade_log_df.empty:
            metrics = metrics_calc.calculate_metrics(trade_log_df, pd.Series([1.0] * len(trade_log_df)))
        else:
            logger.warning("No trade log found, using dummy metrics for visualization demo")
            metrics = {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.08,
                'avg_turnover': 2.1,
                'num_trades': 150,
                'win_rate': 0.58,
                'sortino_ratio': 2.2,
                'calmar_ratio': 1.5,
                'volatility': 0.12
            }
        
        # Create visualizations
        model_name = f"NVDA_GPU_Fixed_{datetime.now().strftime('%Y%m%d')}"
        plot_files = visualizer.create_performance_plots(metrics, trade_log_df, model_name)
        
        if plot_files:
            logger.info("Performance visualizations created successfully!")
            logger.info("Generated plots:")
            for plot_file in plot_files:
                logger.info(f"   Plot: {plot_file}")
            
            # Open plots automatically
            logger.info("Opening performance plots...")
            for plot_file in plot_files:
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(plot_file)
                    else:  # Linux/Mac
                        subprocess.run(['xdg-open', plot_file])
                    time.sleep(1)  # Small delay between opening files
                except Exception as e:
                    logger.warning(f"Could not auto-open {plot_file}: {e}")
            
            return True
        else:
            logger.error("No visualization plots were created")
            return False
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
        return False

def main():
    """Main function."""
    logger = setup_logging()
    
    logger.info("POST-TRAINING VISUALIZATION LAUNCHER STARTED")
    logger.info("=" * 60)
    
    # Configuration
    log_file = 'logs/orchestrator_gpu_fixed.log'
    check_interval = 30  # Check every 30 seconds
    max_wait_hours = 12  # Maximum wait time
    
    # Wait for training completion
    if wait_for_training_completion(log_file, check_interval, max_wait_hours):
        logger.info("Training completed! Starting visualization creation...")
        
        # Add a small delay to ensure all files are written
        logger.info("Waiting 30 seconds for file system sync...")
        time.sleep(30)
        
        # Create visualizations
        if create_visualizations():
            logger.info("Post-training visualization process completed successfully!")
        else:
            logger.error("Visualization creation failed")
            sys.exit(1)
    else:
        logger.error("Training completion not detected within time limit")
        sys.exit(1)
    
    logger.info("POST-TRAINING VISUALIZATION LAUNCHER FINISHED")

if __name__ == "__main__":
    main()