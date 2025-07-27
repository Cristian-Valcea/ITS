#!/usr/bin/env python3
"""
Training Setup Validation Script

This script validates that all components are ready for training launch.
"""

import sys
import os
from pathlib import Path

def validate_training_setup():
    """Validate all training components."""
    
    print("🔍 TRAINING SETUP VALIDATION")
    print("=" * 50)
    
    # Add src to path
    sys.path.append('src')
    
    validation_results = []
    
    # 1. Test TensorBoard Integration
    print("\n📊 Testing TensorBoard Integration...")
    try:
        from training.core.tensorboard_exporter import TensorBoardExporter
        from training.core.tensorboard_monitoring import TensorBoardMonitoringCallback
        validation_results.append("✅ TensorBoard integration available")
        print("✅ TensorBoard exporter: OK")
        print("✅ TensorBoard monitoring callback: OK")
    except ImportError as e:
        validation_results.append(f"❌ TensorBoard integration error: {e}")
        print(f"❌ TensorBoard integration error: {e}")
    
    # 2. Test Turnover Penalty System
    print("\n🎯 Testing Turnover Penalty System...")
    try:
        from gym_env.components.turnover_penalty import TurnoverPenaltyCalculator
        
        # Test basic functionality
        calc = TurnoverPenaltyCalculator(
            portfolio_value_getter=50000.0,
            target_ratio=0.02,
            weight_factor=0.02,
            curve_sharpness=25.0,
            curve='sigmoid'
        )
        penalty = calc.compute_penalty(1000.0)  # Test penalty calculation
        
        validation_results.append("✅ Turnover penalty system functional")
        print("✅ Turnover penalty calculator: OK")
        print(f"✅ Test penalty calculation: {penalty:.4f}")
    except Exception as e:
        validation_results.append(f"❌ Turnover penalty system error: {e}")
        print(f"❌ Turnover penalty system error: {e}")
    
    # 3. Test Configuration Files
    print("\n⚙️ Testing Configuration Files...")
    config_files = [
        'config/turnover_penalty_orchestrator_gpu.yaml',
        'config/emergency_fix_orchestrator_gpu.yaml',
        'config/model_params.yaml',
        'config/risk_limits.yaml'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            validation_results.append(f"✅ {config_file} found")
            print(f"✅ {config_file}: OK")
        else:
            validation_results.append(f"❌ {config_file} missing")
            print(f"❌ {config_file}: MISSING")
    
    # 4. Test Directory Structure
    print("\n📁 Testing Directory Structure...")
    required_dirs = [
        'logs',
        'logs/tensorboard_turnover_penalty',
        'data',
        'models',
        'reports',
        'runs'
    ]
    
    for dir_path in required_dirs:
        dir_obj = Path(dir_path)
        if not dir_obj.exists():
            dir_obj.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
        validation_results.append(f"✅ Directory ready: {dir_path}")
    
    # 5. Test Main Training Script
    print("\n🚀 Testing Main Training Script...")
    if Path('src/main.py').exists():
        validation_results.append("✅ Main training script found")
        print("✅ src/main.py: OK")
    else:
        validation_results.append("❌ Main training script missing")
        print("❌ src/main.py: MISSING")
    
    # 6. Test Launch Scripts
    print("\n🎛️ Testing Launch Scripts...")
    launch_scripts = [
        'start_training_clean.bat',
        'start_training_turnover_penalty.bat',
        'launch_tensorboard.py'
    ]
    
    for script in launch_scripts:
        if Path(script).exists():
            validation_results.append(f"✅ {script} found")
            print(f"✅ {script}: OK")
        else:
            validation_results.append(f"❌ {script} missing")
            print(f"❌ {script}: MISSING")
    
    # 7. Test Python Environment
    print("\n🐍 Testing Python Environment...")
    try:
        import torch
        import tensorboard
        import numpy as np
        import pandas as pd
        import yaml
        
        validation_results.append("✅ Core Python packages available")
        print("✅ PyTorch: OK")
        print("✅ TensorBoard: OK")
        print("✅ NumPy/Pandas: OK")
        print("✅ YAML: OK")
    except ImportError as e:
        validation_results.append(f"❌ Python package error: {e}")
        print(f"❌ Python package error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    success_count = sum(1 for result in validation_results if result.startswith("✅"))
    total_count = len(validation_results)
    
    for result in validation_results:
        print(result)
    
    print(f"\n📊 Results: {success_count}/{total_count} checks passed")
    
    if success_count == total_count:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ System is ready for training launch")
        print("\n🚀 To start training:")
        print("   Option 1: .\\start_training_clean.bat")
        print("   Option 2: .\\start_training_turnover_penalty.bat")
        print("\n📊 TensorBoard will be available at:")
        print("   Primary: http://localhost:6006")
        print("   Enhanced: http://localhost:6007")
        return True
    else:
        print(f"\n⚠️ {total_count - success_count} VALIDATION(S) FAILED!")
        print("❌ Please fix the issues above before launching training")
        return False

if __name__ == "__main__":
    validate_training_setup()