#!/usr/bin/env python3
"""
Example: Running Robustness Validation with Rolling Window Backtest

This example demonstrates how to run a comprehensive 3-month rolling-window 
walk-forward backtest to verify model robustness before deployment.

Usage:
    python examples/run_robustness_validation.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from execution.orchestrator_agent import OrchestratorAgent


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/robustness_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def main():
    """Main execution function."""
    
    print("🔄 IntradayJules - Model Robustness Validation")
    print("=" * 60)
    print("This example demonstrates running a 3-month rolling-window")
    print("walk-forward backtest to verify model robustness.")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration paths
    main_config_path = "config/main_config_orchestrator_gpu_fixed.yaml"
    model_params_path = "config/model_params.yaml"
    risk_limits_path = "config/risk_limits.yaml"
    
    # Model path (replace with your trained model)
    model_path = "models/orch_gpu_rainbow_qrdqn/best_model.zip"
    
    # Robustness validation parameters
    symbol = "SPY"
    data_start_date = "2023-01-01"
    data_end_date = "2024-01-01"
    
    try:
        logger.info("🚀 Initializing OrchestratorAgent...")
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(
            main_config_path=main_config_path,
            model_params_path=model_params_path,
            risk_limits_path=risk_limits_path,
            read_only=True  # Read-only mode for evaluation
        )
        
        logger.info("✅ OrchestratorAgent initialized successfully")
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("💡 Please train a model first or update the model_path variable")
            return
        
        logger.info(f"📊 Starting robustness validation for {symbol}")
        logger.info(f"📅 Data range: {data_start_date} to {data_end_date}")
        logger.info(f"🤖 Model: {model_path}")
        
        # Run rolling window backtest
        results = orchestrator.run_rolling_window_backtest(
            model_path=model_path,
            symbol=symbol,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            training_window_months=3,    # 3-month training windows
            evaluation_window_months=1,  # 1-month evaluation periods
            step_size_months=1           # Walk forward by 1 month
        )
        
        if results:
            print("\n" + "=" * 80)
            print("🎉 ROBUSTNESS VALIDATION COMPLETE")
            print("=" * 80)
            
            # Extract key results
            robustness_stats = results.get('robustness_stats', {})
            summary = robustness_stats.get('summary', {})
            
            print(f"📊 Windows Tested: {summary.get('total_windows', 0)}")
            print(f"💰 Average Return: {summary.get('avg_return', 0):.2f}%")
            print(f"📈 Average Sharpe: {summary.get('avg_sharpe', 0):.2f}")
            print(f"✅ Profitable Windows: {summary.get('profitable_percentage', 0):.1f}%")
            print(f"🎯 Consistency Rating: {summary.get('consistency_rating', 'UNKNOWN')}")
            
            # Deployment recommendation
            if 'executive_summary' in robustness_stats:
                exec_summary = robustness_stats['executive_summary']
                if 'overall_assessment' in exec_summary:
                    recommendation = exec_summary['overall_assessment'].get('recommendation', 'UNKNOWN')
                    
                    print(f"\n🚀 DEPLOYMENT RECOMMENDATION: {recommendation}")
                    
                    # Provide guidance based on recommendation
                    if recommendation == "DEPLOY_FULL_CAPITAL":
                        print("✅ Your model is ready for full deployment!")
                        print("   • Excellent robustness across market conditions")
                        print("   • Deploy with full position sizing")
                        
                    elif recommendation == "DEPLOY_REDUCED_CAPITAL":
                        print("⚠️  Your model shows good robustness")
                        print("   • Deploy with reduced position sizing (50-75%)")
                        print("   • Monitor performance closely")
                        
                    elif recommendation == "PAPER_TRADE_FIRST":
                        print("📝 Recommend paper trading first")
                        print("   • Run paper trading for 2-4 weeks")
                        print("   • Validate performance before going live")
                        
                    else:
                        print("❌ Model requires improvement")
                        print("   • Consider retraining with different parameters")
                        print("   • Gather more training data")
                        print("   • Review feature engineering")
            
            # Report location
            if 'report_path' in results:
                print(f"\n📄 Detailed report saved to: {results['report_path']}")
            
            print("=" * 80)
            
        else:
            logger.error("❌ Robustness validation failed")
            print("\n❌ Robustness validation failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"❌ Error during robustness validation: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Check logs for detailed error information.")


if __name__ == "__main__":
    main()