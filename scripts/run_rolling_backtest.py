#!/usr/bin/env python3
"""
Rolling Window Walk-Forward Backtest Runner

This script runs a comprehensive 3-month rolling-window walk-forward backtest
to verify model robustness across different time periods and market conditions.

Usage:
    python scripts/run_rolling_backtest.py --model_path models/best_model.zip --config config/main_config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation.rolling_window_backtest import RollingWindowBacktest, create_rolling_backtest_config


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/rolling_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def validate_inputs(args) -> None:
    """Validate input arguments."""
    
    # Check model path
    if not Path(args.model_path).exists():
        logging.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Check config path
    if not Path(args.config_path).exists():
        logging.error(f"Config path does not exist: {args.config_path}")
        sys.exit(1)
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logging.error(f"Invalid date format (use YYYY-MM-DD): {e}")
        sys.exit(1)
    
    # Check date range
    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    if end_dt <= start_dt:
        logging.error("End date must be after start date")
        sys.exit(1)
    
    # Check minimum time range (need at least 4 months for 3-month training + 1-month eval)
    min_duration = timedelta(days=120)  # ~4 months
    if (end_dt - start_dt) < min_duration:
        logging.error(f"Date range too short. Need at least 4 months, got {(end_dt - start_dt).days} days")
        sys.exit(1)


def print_backtest_summary(results: dict) -> None:
    """Print a summary of backtest results."""
    
    print("\n" + "="*80)
    print("🎉 ROLLING WINDOW BACKTEST COMPLETE")
    print("="*80)
    
    # Basic stats
    num_windows = results.get('num_windows', 0)
    print(f"📊 Total Windows Processed: {num_windows}")
    
    if 'robustness_stats' in results and 'summary' in results['robustness_stats']:
        summary = results['robustness_stats']['summary']
        
        print(f"💰 Average Return: {summary.get('avg_return', 0):.2f}%")
        print(f"📈 Average Sharpe Ratio: {summary.get('avg_sharpe', 0):.2f}")
        print(f"📉 Worst Drawdown: {summary.get('max_drawdown_worst', 0):.2f}%")
        print(f"✅ Profitable Windows: {summary.get('profitable_windows', 0)}/{num_windows} ({summary.get('profitable_percentage', 0):.1f}%)")
        print(f"🎯 Consistency Rating: {summary.get('consistency_rating', 'UNKNOWN')}")
        
        # Robustness scores
        if 'robustness_scores' in results['robustness_stats']:
            scores = results['robustness_stats']['robustness_scores']
            print(f"🛡️  Overall Robustness Score: {scores.get('overall_robustness', 0):.3f}")
            print(f"   - Return Consistency: {scores.get('return_consistency', 0):.3f}")
            print(f"   - Sharpe Consistency: {scores.get('sharpe_consistency', 0):.3f}")
            print(f"   - Drawdown Control: {scores.get('drawdown_control', 0):.3f}")
            print(f"   - Win Rate Stability: {scores.get('win_rate_stability', 0):.3f}")
        
        # Executive summary
        if 'executive_summary' in results['robustness_stats']:
            exec_summary = results['robustness_stats']['executive_summary']
            if 'overall_assessment' in exec_summary:
                assessment = exec_summary['overall_assessment']
                recommendation = assessment.get('recommendation', 'UNKNOWN')
                
                print(f"\n🚀 DEPLOYMENT RECOMMENDATION: {recommendation}")
                
                # Recommendation explanations
                rec_explanations = {
                    'DEPLOY_FULL_CAPITAL': '✅ Model shows excellent robustness - ready for full deployment',
                    'DEPLOY_REDUCED_CAPITAL': '⚠️  Model shows good robustness - deploy with reduced position sizing',
                    'PAPER_TRADE_FIRST': '📝 Model shows fair robustness - paper trade before live deployment',
                    'REQUIRES_IMPROVEMENT': '❌ Model needs improvement before deployment'
                }
                
                if recommendation in rec_explanations:
                    print(f"   {rec_explanations[recommendation]}")
    
    # Report location
    if 'report_path' in results:
        print(f"\n📄 Detailed Report: {results['report_path']}")
    
    print("="*80)


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Run rolling window walk-forward backtest for model robustness validation"
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model file (e.g., models/best_model.zip)'
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        default='config/main_config_orchestrator_gpu_fixed.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--start_date',
        type=str,
        default='2023-01-01',
        help='Start date for backtest data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date',
        type=str,
        default='2024-01-01',
        help='End date for backtest data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Trading symbol for backtest'
    )
    
    parser.add_argument(
        '--training_window_months',
        type=int,
        default=3,
        help='Training window size in months'
    )
    
    parser.add_argument(
        '--evaluation_window_months',
        type=int,
        default=1,
        help='Evaluation window size in months'
    )
    
    parser.add_argument(
        '--step_size_months',
        type=int,
        default=1,
        help='Step size for walk-forward in months'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Rolling Window Walk-Forward Backtest")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Windows: {args.training_window_months}M train, {args.evaluation_window_months}M eval, {args.step_size_months}M step")
    
    # Validate inputs
    validate_inputs(args)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Create rolling backtest configuration
    rolling_config = create_rolling_backtest_config(config)
    
    # Override with command line arguments
    rolling_config['rolling_backtest'].update({
        'training_window_months': args.training_window_months,
        'evaluation_window_months': args.evaluation_window_months,
        'step_size_months': args.step_size_months
    })
    
    try:
        # Initialize rolling backtest system
        backtest_system = RollingWindowBacktest(rolling_config)
        
        # Run the backtest
        results = backtest_system.run_rolling_backtest(
            model_path=args.model_path,
            data_start_date=args.start_date,
            data_end_date=args.end_date,
            symbol=args.symbol
        )
        
        # Print summary
        print_backtest_summary(results)
        
        logger.info("✅ Rolling window backtest completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Rolling window backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()