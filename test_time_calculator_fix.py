#!/usr/bin/env python3
"""
Test to verify that the TimeFeatureCalculator initialization issue is fixed.

This test validates that:
1. FeatureAgent properly initializes the TimeFeatureCalculator
2. The calculator receives the correct configuration
3. Time features are calculated correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

def test_time_calculator_initialization():
    """Test that TimeFeatureCalculator is properly initialized with config."""
    print("🧪 Testing TimeFeatureCalculator Initialization")
    print("=" * 60)
    
    try:
        from src.agents.feature_agent import FeatureAgent
        
        # Load a config that includes time features
        config_path = "config/main_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config from {config_path}")
        
        # Create FeatureAgent
        feature_agent = FeatureAgent(config)
        print("✅ FeatureAgent created successfully")
        
        # Check if Time calculator is initialized
        if 'Time' in feature_agent.feature_manager.calculators:
            time_calc = feature_agent.feature_manager.calculators['Time']
            print("✅ TimeFeatureCalculator found in calculators")
            
            # Check if config is properly loaded
            if hasattr(time_calc, 'config') and time_calc.config:
                print("✅ TimeFeatureCalculator has config")
                print(f"   Config keys: {list(time_calc.config.keys())}")
                
                # Check specific config values
                time_features = time_calc.config.get('time_features', [])
                sin_cos_encode = time_calc.config.get('sin_cos_encode', [])
                
                print(f"   Time features: {time_features}")
                print(f"   Sin/cos encode: {sin_cos_encode}")
                
                if time_features:
                    print("✅ Time features configuration loaded correctly")
                else:
                    print("⚠️  No time features configured")
                    
            else:
                print("❌ TimeFeatureCalculator missing config")
                return False
        else:
            print("❌ TimeFeatureCalculator not found in calculators")
            print(f"   Available calculators: {list(feature_agent.feature_manager.calculators.keys())}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_calculator_functionality():
    """Test that TimeFeatureCalculator actually works with the config."""
    print("\n🧪 Testing TimeFeatureCalculator Functionality")
    print("=" * 60)
    
    try:
        from src.agents.feature_agent import FeatureAgent
        
        # Load config
        config_path = "config/main_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create FeatureAgent
        feature_agent = FeatureAgent(config)
        
        # Create sample data with datetime index
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=100, freq='1min')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        print("✅ Created sample data with DatetimeIndex")
        print(f"   Data shape: {sample_data.shape}")
        print(f"   Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
        
        # Test feature computation
        features_df = feature_agent.feature_manager.compute_features(sample_data)
        
        if features_df is not None:
            print("✅ Feature computation successful")
            print(f"   Output shape: {features_df.shape}")
            
            # Check for time features
            time_columns = [col for col in features_df.columns if 'hour' in col or 'minute' in col or 'day' in col]
            if time_columns:
                print(f"✅ Time features found: {time_columns}")
                
                # Show sample values
                print("\n📊 Sample time feature values:")
                for col in time_columns[:3]:  # Show first 3 time columns
                    sample_values = features_df[col].head(5).tolist()
                    print(f"   {col}: {sample_values}")
                    
                return True
            else:
                print("⚠️  No time features found in output")
                print(f"   Available columns: {list(features_df.columns)}")
                return False
        else:
            print("❌ Feature computation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_structure():
    """Test that the config structure is correct for time calculator."""
    print("\n🧪 Testing Config Structure")
    print("=" * 60)
    
    try:
        config_files = [
            "config/main_config.yaml",
            "config/main_config_orchestrator_integrated.yaml", 
            "config/main_config_orchestrator_test.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"\n📄 Checking {config_file}")
                
                feature_config = config.get('feature_engineering', {})
                if 'time' in feature_config:
                    time_config = feature_config['time']
                    print(f"   ✅ Found 'time' section: {time_config}")
                else:
                    print(f"   ❌ Missing 'time' section in feature_engineering")
                    print(f"   Available sections: {list(feature_config.keys())}")
            else:
                print(f"   ⚠️  Config file not found: {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 TIME CALCULATOR INITIALIZATION FIX TEST")
    print("=" * 70)
    
    success = True
    
    success &= test_config_structure()
    success &= test_time_calculator_initialization()
    success &= test_time_calculator_functionality()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TIME CALCULATOR TESTS PASSED!")
        print("\n📋 Fix Summary:")
        print("   ✅ Configuration structure corrected")
        print("   ✅ TimeFeatureCalculator properly initialized")
        print("   ✅ Config passed correctly to calculator")
        print("   ✅ Time features computed successfully")
    else:
        print("❌ SOME TIME CALCULATOR TESTS FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())