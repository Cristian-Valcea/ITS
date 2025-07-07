#!/usr/bin/env python3
"""
Comprehensive Integration Test Script
Tests all the newly integrated components and features.
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths (from tests directory, go up one level to project root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """Test that all critical imports work."""
    print("🔍 Testing imports...")
    
    try:
        from src.execution.orchestrator_agent import OrchestratorAgent
        print("✅ OrchestratorAgent import successful")
    except Exception as e:
        print(f"❌ OrchestratorAgent import failed: {e}")
        return False
    
    try:
        from src.agents.data_agent import DataAgent
        print("✅ DataAgent import successful")
    except Exception as e:
        print(f"❌ DataAgent import failed: {e}")
        return False
    
    try:
        from src.agents.feature_agent import FeatureAgent
        print("✅ FeatureAgent import successful")
    except Exception as e:
        print(f"❌ FeatureAgent import failed: {e}")
        return False
    
    try:
        from src.agents.risk_agent import RiskAgent
        print("✅ RiskAgent import successful")
    except Exception as e:
        print(f"❌ RiskAgent import failed: {e}")
        return False
    
    try:
        from src.training.trainer_agent import create_trainer_agent
        print("✅ TrainerAgent import successful")
    except Exception as e:
        print(f"❌ TrainerAgent import failed: {e}")
        return False
    
    return True

def test_config_files():
    """Test that all config files exist and are valid."""
    print("\n📋 Testing configuration files...")
    
    config_files = [
        "config/main_config_orchestrator_test.yaml",
        "config/model_params_orchestrator_test.yaml", 
        "config/risk_limits_orchestrator_test.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} exists")
            try:
                import yaml
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file} is valid YAML")
            except Exception as e:
                print(f"❌ {config_file} YAML parsing failed: {e}")
                return False
        else:
            print(f"❌ {config_file} missing")
            return False
    
    return True

def test_orchestrator_initialization():
    """Test OrchestratorAgent initialization."""
    print("\n🎯 Testing OrchestratorAgent initialization...")
    
    try:
        from src.execution.orchestrator_agent import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config_orchestrator_test.yaml",
            model_params_path="config/model_params_orchestrator_test.yaml",
            risk_limits_path="config/risk_limits_orchestrator_test.yaml"
        )
        
        print("✅ OrchestratorAgent initialized successfully")
        
        # Test that all agents are initialized
        if hasattr(orchestrator, 'data_agent') and orchestrator.data_agent:
            print("✅ DataAgent initialized")
        else:
            print("❌ DataAgent not initialized")
            return False
            
        if hasattr(orchestrator, 'feature_agent') and orchestrator.feature_agent:
            print("✅ FeatureAgent initialized")
        else:
            print("❌ FeatureAgent not initialized")
            return False
            
        if hasattr(orchestrator, 'risk_agent') and orchestrator.risk_agent:
            print("✅ RiskAgent initialized")
        else:
            print("❌ RiskAgent not initialized")
            return False
            
        if hasattr(orchestrator, 'trainer_agent') and orchestrator.trainer_agent:
            print("✅ TrainerAgent initialized")
        else:
            print("❌ TrainerAgent not initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OrchestratorAgent initialization failed: {e}")
        return False

def test_live_trading_config():
    """Test live trading configuration."""
    print("\n🚀 Testing live trading configuration...")
    
    try:
        import yaml
        with open("config/main_config_orchestrator_test.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        live_trading = config.get('live_trading', {})
        if live_trading:
            print("✅ Live trading configuration found")
            
            required_keys = ['enabled', 'symbol', 'data_interval', 'production_model_path']
            for key in required_keys:
                if key in live_trading:
                    print(f"✅ {key}: {live_trading[key]}")
                else:
                    print(f"❌ Missing required key: {key}")
                    return False
        else:
            print("❌ Live trading configuration not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Live trading config test failed: {e}")
        return False

def test_risk_management():
    """Test risk management configuration."""
    print("\n🛡️ Testing risk management...")
    
    try:
        import yaml
        with open("config/risk_limits_orchestrator_test.yaml", 'r') as f:
            risk_config = yaml.safe_load(f)
        
        required_risk_params = [
            'max_daily_drawdown_pct',
            'max_hourly_turnover_ratio', 
            'max_daily_turnover_ratio',
            'halt_on_breach'
        ]
        
        for param in required_risk_params:
            if param in risk_config:
                print(f"✅ {param}: {risk_config[param]}")
            else:
                print(f"❌ Missing risk parameter: {param}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering configuration."""
    print("\n🧠 Testing feature engineering...")
    
    try:
        import yaml
        with open("config/main_config_orchestrator_test.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        feature_eng = config.get('feature_engineering', {})
        if feature_eng:
            print("✅ Feature engineering configuration found")
            
            features = feature_eng.get('features', [])
            if features:
                print(f"✅ Features configured: {features}")
            else:
                print("❌ No features configured")
                return False
                
            if 'sin_cos_encode' in feature_eng:
                print(f"✅ Sin/cos encoding: {feature_eng['sin_cos_encode']}")
            
        else:
            print("❌ Feature engineering configuration not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 COMPREHENSIVE INTEGRATION TEST")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tests = [
        ("Import Tests", test_imports),
        ("Config File Tests", test_config_files),
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Live Trading Config", test_live_trading_config),
        ("Risk Management", test_risk_management),
        ("Feature Engineering", test_feature_engineering)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Integration successful!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)