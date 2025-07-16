#!/usr/bin/env python3
"""
Risk-Aware Training CLI

Easy-to-use command line interface for running training with different
risk control configurations.

Examples:
    # Basic training with default risk settings
    python scripts/run_risk_training.py --symbol AAPL --days 5
    
    # Training with custom volatility penalty
    python scripts/run_risk_training.py --symbol NVDA --days 10 --vol-penalty 0.5
    
    # Training with relaxed drawdown limit
    python scripts/run_risk_training.py --symbol TSLA --days 7 --dd-limit 0.04
    
    # Training with risk features disabled
    python scripts/run_risk_training.py --symbol MSFT --days 3 --no-risk-features
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RiskTrainingCLI")

def create_risk_config(args):
    """Create risk configuration from CLI arguments."""
    risk_config = {
        'vol_window': args.vol_window,
        'penalty_lambda': args.vol_penalty,
        'dd_limit': args.dd_limit,
        'eval_dd_limit': args.eval_dd_limit,
        'include_risk_features': not args.no_risk_features,
        'curriculum': {
            'enabled': False,
            'use_perf_trigger': False,
            'logic': 'and'
        }
    }
    
    logger.info(f"🛡️ Risk Configuration:")
    logger.info(f"   Volatility Window: {risk_config['vol_window']} steps")
    logger.info(f"   Penalty Lambda: {risk_config['penalty_lambda']}")
    logger.info(f"   Training DD Limit: {risk_config['dd_limit']:.1%}")
    logger.info(f"   Evaluation DD Limit: {risk_config['eval_dd_limit']:.1%}")
    logger.info(f"   Risk Features: {'Enabled' if risk_config['include_risk_features'] else 'Disabled'}")
    
    return risk_config

def update_config_file(risk_config, training_steps):
    """Update main config with risk settings."""
    config_path = PROJECT_ROOT / "config" / "main_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update risk section
    config['risk'] = risk_config
    
    # Update training steps if specified
    if training_steps:
        config['training']['total_timesteps'] = training_steps
        logger.info(f"🏃 Training steps set to: {training_steps:,}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("📝 Updated main_config.yaml with risk settings")

def run_training(symbol, start_date, end_date, interval):
    """Run training with risk controls."""
    logger.info(f"🚀 Starting risk-aware training for {symbol}...")
    
    try:
        from src.execution.orchestrator_agent import OrchestratorAgent
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config.yaml",
            model_params_path="config/model_params.yaml",
            risk_limits_path="config/risk_limits.yaml"
        )
        
        # Run training pipeline
        result = orchestrator.run_training_pipeline(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cached_data=True,
            run_evaluation_after_train=True
        )
        
        if result:
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"📁 Model saved at: {result}")
            return True
        else:
            logger.error("❌ Training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Risk-Aware Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic parameters
    parser.add_argument('--symbol', '-s', default='AAPL', help='Stock symbol to trade')
    parser.add_argument('--days', '-d', type=int, default=5, help='Number of days of data to use')
    parser.add_argument('--interval', '-i', default='1min', help='Data interval (1min, 5min, etc.)')
    
    # Risk control parameters
    parser.add_argument('--vol-window', type=int, default=60, 
                       help='Volatility calculation window size (default: 60)')
    parser.add_argument('--vol-penalty', type=float, default=0.25,
                       help='Volatility penalty weight lambda (default: 0.25)')
    parser.add_argument('--dd-limit', type=float, default=0.03,
                       help='Training drawdown limit (default: 0.03 = 3%%)')
    parser.add_argument('--eval-dd-limit', type=float, default=0.02,
                       help='Evaluation drawdown limit (default: 0.02 = 2%%)')
    parser.add_argument('--no-risk-features', action='store_true',
                       help='Disable risk features in observation space')
    
    # Training parameters
    parser.add_argument('--steps', type=int, help='Override training timesteps')
    parser.add_argument('--quick', action='store_true', help='Quick test with 1000 steps')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"📅 Training period: {start_date_str} to {end_date_str}")
    
    # Handle quick mode
    training_steps = args.steps
    if args.quick:
        training_steps = 1000
        logger.info("🏃 Quick mode: Using 1000 training steps")
    
    # Create and apply risk configuration
    risk_config = create_risk_config(args)
    update_config_file(risk_config, training_steps)
    
    # Run training
    success = run_training(args.symbol, start_date_str, end_date_str, args.interval)
    
    if success:
        logger.info("🎉 Risk-aware training completed successfully!")
        logger.info("🛡️ Check logs above for risk metrics and drawdown events")
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()