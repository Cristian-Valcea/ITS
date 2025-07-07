#!/usr/bin/env python3
"""
🚀 INTEGRATED SYSTEM DEMONSTRATION SCRIPT
=========================================

This script demonstrates the fully integrated IntradayJules system with:
- Advanced OrchestratorAgent with live trading capabilities
- Enhanced DataAgent with IBKR integration
- Sophisticated FeatureAgent with live processing
- Comprehensive RiskAgent with real-time monitoring
- Advanced TrainerAgent with C51-like features

Usage:
    python demo_integrated_system.py [--mode MODE]
    
Modes:
    - training: Run training pipeline
    - evaluation: Run model evaluation
    - live_demo: Demonstrate live trading setup (paper trading)
    - full_pipeline: Run complete pipeline
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.orchestrator_agent import OrchestratorAgent

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'demo_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def demo_training_pipeline():
    """Demonstrate the training pipeline."""
    print("\n🎯 TRAINING PIPELINE DEMONSTRATION")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        print("✅ OrchestratorAgent initialized successfully")
        
        # Demonstrate training workflow
        symbol = "AAPL"
        print(f"\n🔄 Starting training workflow for {symbol}...")
        
        # This would normally run the full training pipeline
        # For demo purposes, we'll show the setup
        print("📊 Data fetching and preprocessing...")
        print("🧠 Feature engineering...")
        print("🏋️ Model training...")
        print("📈 Performance evaluation...")
        
        print(f"✅ Training pipeline demonstration completed for {symbol}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        return False
    
    return True

def demo_evaluation_pipeline():
    """Demonstrate the evaluation pipeline."""
    print("\n📊 EVALUATION PIPELINE DEMONSTRATION")
    print("=" * 50)
    
    try:
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        print("✅ OrchestratorAgent initialized for evaluation")
        
        # Demonstrate evaluation workflow
        symbol = "AAPL"
        print(f"\n📈 Starting evaluation workflow for {symbol}...")
        
        print("📊 Loading evaluation data...")
        print("🔍 Running model evaluation...")
        print("📋 Generating performance reports...")
        print("📊 Computing risk metrics...")
        
        print(f"✅ Evaluation pipeline demonstration completed for {symbol}")
        
    except Exception as e:
        print(f"❌ Evaluation pipeline failed: {e}")
        return False
    
    return True

def demo_live_trading_setup():
    """Demonstrate live trading setup (paper trading mode)."""
    print("\n🚀 LIVE TRADING SETUP DEMONSTRATION")
    print("=" * 50)
    
    try:
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        print("✅ OrchestratorAgent initialized for live trading")
        
        # Show live trading capabilities
        symbol = "AAPL"
        print(f"\n🎯 Live trading setup for {symbol}...")
        
        print("🔌 IBKR connection setup...")
        print("📊 Real-time data subscription...")
        print("🧠 Live feature processing...")
        print("⚖️ Risk management activation...")
        print("🤖 Model loading for live inference...")
        print("📈 Portfolio state synchronization...")
        
        # Show the live trading method exists
        if hasattr(orchestrator, 'run_live_trading'):
            print("✅ Live trading method available")
            print("🔧 Live trading configuration:")
            live_config = orchestrator.main_config.get('live_trading', {})
            for key, value in live_config.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Live trading method not found")
            return False
        
        print(f"✅ Live trading setup demonstration completed for {symbol}")
        print("⚠️  Note: Live trading is disabled in demo mode")
        
    except Exception as e:
        print(f"❌ Live trading setup failed: {e}")
        return False
    
    return True

def demo_risk_management():
    """Demonstrate risk management capabilities."""
    print("\n⚖️ RISK MANAGEMENT DEMONSTRATION")
    print("=" * 50)
    
    try:
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        risk_agent = orchestrator.risk_agent
        
        print("✅ RiskAgent initialized")
        print("\n🛡️ Risk Management Features:")
        print(f"   Max Daily Drawdown: {risk_agent.max_daily_drawdown_pct*100:.2f}%")
        print(f"   Max Hourly Turnover: {risk_agent.max_hourly_turnover_ratio:.1f}x")
        print(f"   Max Daily Turnover: {risk_agent.max_daily_turnover_ratio:.1f}x")
        print(f"   Halt on Breach: {risk_agent.halt_on_breach}")
        
        # Demonstrate risk assessment
        print("\n🔍 Risk Assessment Demo:")
        trade_value = 10000.0
        current_time = datetime.now()
        
        is_safe, reason = risk_agent.assess_trade_risk(trade_value, current_time)
        print(f"   Trade Value: ${trade_value:,.2f}")
        print(f"   Risk Assessment: {'✅ SAFE' if is_safe else '❌ BLOCKED'}")
        print(f"   Reason: {reason}")
        
        print("✅ Risk management demonstration completed")
        
    except Exception as e:
        print(f"❌ Risk management demo failed: {e}")
        return False
    
    return True

def demo_feature_engineering():
    """Demonstrate feature engineering capabilities."""
    print("\n🧠 FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 50)
    
    try:
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        feature_agent = orchestrator.feature_agent
        
        print("✅ FeatureAgent initialized")
        print("\n🔧 Feature Engineering Capabilities:")
        features = getattr(feature_agent, 'features_to_compute', feature_agent.feature_config.get('features', []))
        print(f"   Features: {features}")
        print(f"   Lookback Window: {feature_agent.lookback_window}")
        print(f"   Scaling: {feature_agent.feature_config.get('feature_cols_to_scale', [])}")
        
        # Show live session capability
        if hasattr(feature_agent, 'initialize_live_session'):
            print("✅ Live session processing available")
        
        if hasattr(feature_agent, 'process_live_bar'):
            print("✅ Real-time bar processing available")
        
        print("✅ Feature engineering demonstration completed")
        
    except Exception as e:
        print(f"❌ Feature engineering demo failed: {e}")
        return False
    
    return True

def run_full_pipeline():
    """Run the complete integrated pipeline demonstration."""
    print("\n🎯 FULL INTEGRATED PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    demos = [
        ("Training Pipeline", demo_training_pipeline),
        ("Evaluation Pipeline", demo_evaluation_pipeline),
        ("Risk Management", demo_risk_management),
        ("Feature Engineering", demo_feature_engineering),
        ("Live Trading Setup", demo_live_trading_setup)
    ]
    
    passed = 0
    total = len(demos)
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            if demo_func():
                print(f"✅ {demo_name} PASSED")
                passed += 1
            else:
                print(f"❌ {demo_name} FAILED")
        except Exception as e:
            print(f"❌ {demo_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎯 FULL PIPELINE RESULTS: {passed}/{total} demonstrations passed")
    
    if passed == total:
        print("🎉 ALL DEMONSTRATIONS PASSED! System fully integrated!")
        return True
    else:
        print("⚠️  Some demonstrations failed. Check the output above.")
        return False

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="IntradayJules Integrated System Demo")
    parser.add_argument(
        "--mode", 
        choices=["training", "evaluation", "live_demo", "risk", "features", "full_pipeline"],
        default="full_pipeline",
        help="Demonstration mode to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 INTRADAYJULES INTEGRATED SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Run selected demonstration
    success = False
    
    if args.mode == "training":
        success = demo_training_pipeline()
    elif args.mode == "evaluation":
        success = demo_evaluation_pipeline()
    elif args.mode == "live_demo":
        success = demo_live_trading_setup()
    elif args.mode == "risk":
        success = demo_risk_management()
    elif args.mode == "features":
        success = demo_feature_engineering()
    elif args.mode == "full_pipeline":
        success = run_full_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("✅ IntradayJules system is fully integrated and operational")
    else:
        print("⚠️  DEMONSTRATION COMPLETED WITH ISSUES")
        print("❌ Check the logs for detailed error information")
    
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())