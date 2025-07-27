#!/usr/bin/env python3
"""
Training Setup Verification Script

This script verifies that all components are properly configured for training
with the new rolling window backtest functionality.
"""

import os
import sys
import yaml
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_directory_exists(dir_path, description):
    """Check if a directory exists and report status."""
    if Path(dir_path).exists():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"⚠️  {description}: {dir_path} - WILL BE CREATED")
        return True  # Directories can be created automatically

def verify_yaml_config(config_path, required_keys):
    """Verify YAML configuration contains required keys."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        missing_keys = []
        for key_path in required_keys:
            keys = key_path.split('.')
            current = config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    missing_keys.append(key_path)
                    break
        
        if missing_keys:
            print(f"⚠️  {config_path} - Missing keys: {missing_keys}")
            return False
        else:
            print(f"✅ {config_path} - All required keys present")
            return True
            
    except Exception as e:
        print(f"❌ {config_path} - Error reading: {e}")
        return False

def main():
    """Main verification function."""
    
    print("🔍 IntradayJules Training Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check core files
    print("\n📁 Core Files:")
    all_good &= check_file_exists("src/main.py", "Main training script")
    all_good &= check_file_exists("start_training_clean.bat", "Training launcher")
    all_good &= check_file_exists("monitor_live_logs.py", "Log monitor")
    all_good &= check_file_exists("post_training_visualizer.py", "Visualizer")
    
    # Check configuration files
    print("\n⚙️  Configuration Files:")
    all_good &= check_file_exists("config/main_config_orchestrator_gpu_fixed.yaml", "Main config")
    all_good &= check_file_exists("config/model_params.yaml", "Model parameters")
    all_good &= check_file_exists("config/risk_limits.yaml", "Risk limits")
    
    # Check virtual environment
    print("\n🐍 Python Environment:")
    all_good &= check_file_exists("venv/Scripts/python.exe", "Virtual environment")
    all_good &= check_file_exists("venv/Scripts/activate.bat", "Activation script")
    
    # Check directories
    print("\n📂 Required Directories:")
    check_directory_exists("logs", "Logs directory")
    check_directory_exists("data", "Data directory")
    check_directory_exists("models", "Models directory")
    check_directory_exists("reports", "Reports directory")
    
    # Verify main configuration
    print("\n🔧 Configuration Validation:")
    main_config_keys = [
        "evaluation.rolling_backtest.enabled",
        "evaluation.rolling_backtest.training_window_months",
        "evaluation.rolling_backtest.evaluation_window_months",
        "orchestrator.data_dir",
        "orchestrator.model_dir",
        "logging.log_file_path"
    ]
    
    config_valid = verify_yaml_config("config/main_config_orchestrator_gpu_fixed.yaml", main_config_keys)
    all_good &= config_valid
    
    # Check rolling window backtest configuration
    if config_valid:
        try:
            with open("config/main_config_orchestrator_gpu_fixed.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            rolling_config = config.get('evaluation', {}).get('rolling_backtest', {})
            
            print("\n🔄 Rolling Window Backtest Configuration:")
            print(f"   • Enabled: {rolling_config.get('enabled', False)}")
            print(f"   • Training Window: {rolling_config.get('training_window_months', 3)} months")
            print(f"   • Evaluation Window: {rolling_config.get('evaluation_window_months', 1)} months")
            print(f"   • Data Start: {rolling_config.get('data_start_date', 'Not set')}")
            print(f"   • Data End: {rolling_config.get('data_end_date', 'Not set')}")
            
            if rolling_config.get('enabled', False):
                print("   ✅ Rolling window backtest is ENABLED")
            else:
                print("   ⚠️  Rolling window backtest is DISABLED")
                
        except Exception as e:
            print(f"   ❌ Error reading rolling backtest config: {e}")
    
    # Check new implementation files
    print("\n🆕 New Rolling Window Backtest Files:")
    all_good &= check_file_exists("src/evaluation/rolling_window_backtest.py", "Rolling backtest implementation")
    all_good &= check_file_exists("tests/test_rolling_window_backtest.py", "Rolling backtest tests")
    all_good &= check_file_exists("examples/run_robustness_validation.py", "Example usage script")
    
    # Final assessment
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 SETUP VERIFICATION COMPLETE - ALL SYSTEMS GO!")
        print("✅ Your training environment is properly configured")
        print("✅ Rolling window backtest is ready")
        print("✅ All monitoring tools are available")
        print("\n🚀 You can now run: start_training_clean.bat")
    else:
        print("⚠️  SETUP ISSUES DETECTED")
        print("❌ Please resolve the issues above before training")
        print("💡 Check file paths and configuration settings")
    
    print("=" * 60)

if __name__ == "__main__":
    main()