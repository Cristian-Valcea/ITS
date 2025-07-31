#!/usr/bin/env python3
"""
🔍 Pre-flight Check for 200K Training Launch
Validates all requirements before starting 8-10 hour training
"""

import os
import sys
import torch
import pickle
from pathlib import Path

def check_gpu():
    """Check GPU availability and configuration"""
    print("🔍 GPU Check:")
    
    if not torch.cuda.is_available():
        print("  ❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"  ✅ GPU: {gpu_name}")
    print(f"  ✅ Memory: {gpu_memory:.1f} GB")
    
    # Check TF32 support (Ampere architecture)
    if "RTX 30" in gpu_name or "RTX 40" in gpu_name:
        print("  ✅ TF32 optimization available (Ampere/Ada)")
    else:
        print("  ⚠️  TF32 optimization may not be available")
    
    return True

def check_base_model():
    """Check base model and VecNormalize files"""
    print("\n🔍 Base Model Check:")
    
    base_model_path = "models/dual_ticker_enhanced_50k_final.zip"
    vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
    
    if not os.path.exists(base_model_path):
        print(f"  ❌ Base model not found: {base_model_path}")
        return False
    
    if not os.path.exists(vecnorm_path):
        print(f"  ❌ VecNormalize stats not found: {vecnorm_path}")
        return False
    
    # Check file sizes
    model_size = os.path.getsize(base_model_path) / 1e6
    vecnorm_size = os.path.getsize(vecnorm_path) / 1e3
    
    print(f"  ✅ Base model: {base_model_path} ({model_size:.1f} MB)")
    print(f"  ✅ VecNormalize: {vecnorm_path} ({vecnorm_size:.1f} KB)")
    
    # Try loading VecNormalize to validate
    try:
        with open(vecnorm_path, 'rb') as f:
            vecnorm_stats = pickle.load(f)
        print("  ✅ VecNormalize stats loaded successfully")
    except Exception as e:
        print(f"  ❌ VecNormalize loading failed: {e}")
        return False
    
    return True

def check_directories():
    """Check required directories exist"""
    print("\n🔍 Directory Check:")
    
    required_dirs = [
        "models/checkpoints",
        "logs",
        "runs",
        "reports"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ Created: {dir_path}")
        else:
            print(f"  ✅ Exists: {dir_path}")
    
    return all_good

def check_dependencies():
    """Check key dependencies"""
    print("\n🔍 Dependencies Check:")
    
    try:
        import stable_baselines3
        print(f"  ✅ stable-baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print("  ❌ stable-baselines3 not installed")
        return False
    
    try:
        import tensorboardX
        print(f"  ✅ tensorboardX available")
    except ImportError:
        print("  ❌ tensorboardX not installed")
        return False
    
    try:
        # Check if our custom modules can be imported
        sys.path.append("src")
        from gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        print("  ✅ DualTickerTradingEnv importable")
    except ImportError as e:
        print(f"  ❌ Custom modules import failed: {e}")
        return False
    
    return True

def check_disk_space():
    """Check available disk space"""
    print("\n🔍 Disk Space Check:")
    
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    
    print(f"  💾 Free space: {free_gb:.1f} GB")
    
    if free_gb < 5:
        print("  ⚠️  Low disk space - training may fail")
        return False
    elif free_gb < 10:
        print("  ⚠️  Moderate disk space - monitor during training")
    else:
        print("  ✅ Sufficient disk space")
    
    return True

def estimate_training_time():
    """Estimate training duration"""
    print("\n⏰ Training Time Estimate:")
    
    # RTX 3060 estimates
    steps_per_hour = 25000  # Conservative estimate
    total_steps = 200000
    estimated_hours = total_steps / steps_per_hour
    
    print(f"  📊 Total steps: {total_steps:,}")
    print(f"  🚀 Estimated rate: {steps_per_hour:,} steps/hour (RTX 3060)")
    print(f"  ⏰ Estimated duration: {estimated_hours:.1f} hours")
    print(f"  🎯 Completion: ~{estimated_hours:.0f}h from start")

def main():
    """Run all pre-flight checks"""
    print("🚀 200K Dual-Ticker Training - Pre-flight Check")
    print("=" * 50)
    
    checks = [
        ("GPU", check_gpu),
        ("Base Model", check_base_model),
        ("Directories", check_directories),
        ("Dependencies", check_dependencies),
        ("Disk Space", check_disk_space)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    estimate_training_time()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR 200K TRAINING!")
        print("\n🚀 Launch commands:")
        print("  • Background:  ./launch_200k_tmux.sh")
        print("  • Foreground:  python launch_200k_dual_ticker_training.py")
        print("\n💡 Recommended: Use tmux for 8-10 hour training")
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("\n🔧 Fix the issues above and run pre-flight check again")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)