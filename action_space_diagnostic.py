#!/usr/bin/env python3
"""
🔍 ACTION SPACE DIAGNOSTIC
Root-cause checklist for action space corruption
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from stable_baselines3 import PPO
    from gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
    import numpy as np
    import pandas as pd
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def check_action_space_integrity():
    """Check action space integrity across all components."""
    
    print("🔍 ACTION SPACE DIAGNOSTIC")
    print("=" * 50)
    
    # 1. Check environment action space
    print("1️⃣ ENVIRONMENT ACTION SPACE:")
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        feature_data = np.random.randn(1000, 26).astype(np.float32)
        price_data = np.random.randn(1000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(1000)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=200,
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.45,
            hold_bonus_weight=0.015,
            verbose=False
        )
        
        env_action_space = env.action_space.n
        print(f"   env.action_space.n = {env_action_space}")
        
        if env_action_space == 5:
            print("   ✅ CORRECT: Environment expects 5 actions")
        else:
            print(f"   ❌ ERROR: Environment expects {env_action_space} actions (should be 5)")
            
    except Exception as e:
        print(f"   ❌ Environment check failed: {e}")
        env_action_space = None
    
    # 2. Check Cycle 6 model (healthy checkpoint)
    print("\n2️⃣ CYCLE 6 MODEL (HEALTHY CHECKPOINT):")
    cycle6_path = "train_runs/stairways_8cycle_20250803_193928/cycle_06_hold_55%/model_checkpoint_cycle_06_hold_55%_PROGRESS.zip"
    
    if Path(cycle6_path).exists():
        try:
            model_c6 = PPO.load(cycle6_path, device="cpu")
            # Handle different policy architectures
            if hasattr(model_c6.policy, 'action_net'):
                if isinstance(model_c6.policy.action_net, list):
                    c6_action_features = model_c6.policy.action_net[-1].out_features
                else:
                    c6_action_features = model_c6.policy.action_net.out_features
            else:
                c6_action_features = "Unknown architecture"
            print(f"   policy.action_net[-1].out_features = {c6_action_features}")
            
            if c6_action_features == 5:
                print("   ✅ CORRECT: Cycle 6 model has 5 action logits")
                cycle6_healthy = True
            else:
                print(f"   ❌ ERROR: Cycle 6 model has {c6_action_features} action logits (should be 5)")
                cycle6_healthy = False
                
        except Exception as e:
            print(f"   ❌ Cycle 6 model check failed: {e}")
            cycle6_healthy = False
    else:
        print(f"   ❌ Cycle 6 model not found: {cycle6_path}")
        cycle6_healthy = False
    
    # 3. Check Cycle 7 model (corrupted)
    print("\n3️⃣ CYCLE 7 MODEL (CORRUPTED):")
    cycle7_path = "train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%/model_checkpoint_cycle_07_hold_45%_PROGRESS.zip"
    
    if Path(cycle7_path).exists():
        try:
            model_c7 = PPO.load(cycle7_path, device="cpu")
            # Handle different policy architectures
            if hasattr(model_c7.policy, 'action_net'):
                if isinstance(model_c7.policy.action_net, list):
                    c7_action_features = model_c7.policy.action_net[-1].out_features
                else:
                    c7_action_features = model_c7.policy.action_net.out_features
            else:
                c7_action_features = "Unknown architecture"
            print(f"   policy.action_net[-1].out_features = {c7_action_features}")
            
            if c7_action_features == 5:
                print("   ✅ UNEXPECTED: Cycle 7 model has correct 5 action logits")
                cycle7_corrupted = False
            else:
                print(f"   ❌ CONFIRMED: Cycle 7 model has {c7_action_features} action logits (corrupted)")
                cycle7_corrupted = True
                
        except Exception as e:
            print(f"   ❌ Cycle 7 model check failed: {e}")
            cycle7_corrupted = True
    else:
        print(f"   ⚠️ Cycle 7 model not found: {cycle7_path}")
        cycle7_corrupted = True
    
    # 4. Summary and recommendations
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 30)
    
    if env_action_space == 5:
        print("✅ Environment: HEALTHY (5 actions)")
    else:
        print("❌ Environment: CORRUPTED")
    
    if cycle6_healthy:
        print("✅ Cycle 6 Model: HEALTHY (5 logits)")
    else:
        print("❌ Cycle 6 Model: CORRUPTED")
    
    if not cycle7_corrupted:
        print("✅ Cycle 7 Model: HEALTHY (5 logits)")
    else:
        print("❌ Cycle 7 Model: CORRUPTED (>5 logits)")
    
    print("\n🚀 RECOVERY PLAN:")
    
    if cycle6_healthy and env_action_space == 5:
        print("✅ READY FOR RECOVERY:")
        print("   1. Use Cycle 6 model as base")
        print("   2. Increase base_hold_bonus to 0.020")
        print("   3. Add action space validation")
        print("   4. Retry Cycle 7 with fixed configuration")
        recovery_ready = True
    else:
        print("❌ RECOVERY BLOCKED:")
        if not cycle6_healthy:
            print("   - Cycle 6 model corrupted - need earlier checkpoint")
        if env_action_space != 5:
            print("   - Environment action space corrupted - need code fix")
        recovery_ready = False
    
    return recovery_ready, {
        'env_action_space': env_action_space,
        'cycle6_healthy': cycle6_healthy,
        'cycle7_corrupted': cycle7_corrupted
    }

def main():
    """Main diagnostic function."""
    
    recovery_ready, diagnostics = check_action_space_integrity()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC COMPLETE")
    
    if recovery_ready:
        print("✅ SYSTEM READY FOR CYCLE 7 RECOVERY")
        print("🚀 Proceed with parameter micro-tweak and retry")
    else:
        print("❌ SYSTEM NEEDS REPAIR BEFORE RECOVERY")
        print("🔧 Fix corrupted components first")
    
    return recovery_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)