#!/usr/bin/env python3
"""
Diagnostic Test - Find the exact failure point
"""

import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def test_model_loading():
    """Test if we can load the base model without training"""
    try:
        print("🔍 TEST 1: Loading base model...")
        from sb3_contrib import RecurrentPPO
        
        model_path = 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip'
        
        # Test loading without environment
        print("   Loading model without environment...")
        model = RecurrentPPO.load(model_path)
        print("   ✅ Model loaded successfully")
        
        # Test prediction
        print("   Testing prediction...")
        import numpy as np
        dummy_obs = np.zeros((1, 26), dtype=np.float32)
        action, state = model.predict(dummy_obs, deterministic=True)
        print(f"   ✅ Prediction successful: action={action}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test if we can create the environment without training"""
    try:
        print("🔍 TEST 2: Creating environment...")
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from secrets_helper import SecretsHelper
        
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        print("   Loading data adapter...")
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        print("   Loading training data...")
        training_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2024-03-01',  # MUCH smaller date range
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        print(f"   ✅ Data loaded: {len(training_data['trading_days'])} periods")
        
        print("   Creating environment...")
        env = DualTickerTradingEnv(
            nvda_data=training_data['nvda_features'],
            msft_data=training_data['msft_features'], 
            nvda_prices=training_data['nvda_prices'],
            msft_prices=training_data['msft_prices'],
            trading_days=training_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=100,  # Very short episodes
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        print("   ✅ Environment created successfully")
        
        # Test environment reset
        print("   Testing environment reset...")
        obs = env.reset()
        print(f"   ✅ Environment reset successful: obs shape={obs[0].shape if isinstance(obs, tuple) else obs.shape}")
        
        return True, env
        
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False, None

def test_model_env_compatibility():
    """Test if model and environment are compatible"""
    try:
        print("🔍 TEST 3: Testing model-environment compatibility...")
        
        # Load model
        from sb3_contrib import RecurrentPPO
        model_path = 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip'
        
        # Get environment from previous test
        success, env = test_environment_creation()
        if not success:
            return False
            
        print("   Loading model with environment...")
        model = RecurrentPPO.load(model_path, env=env)
        print("   ✅ Model loaded with environment")
        
        # Test single step
        print("   Testing single training step...")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        action, state = model.predict(obs, deterministic=True)
        print(f"   ✅ Prediction successful: action={action}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model-environment compatibility failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 DIAGNOSTIC TESTS FOR 300K TRAINING ISSUES")
    print("=" * 60)
    
    # Run diagnostic tests
    test1_passed = test_model_loading()
    print()
    
    test2_passed = test_environment_creation()[0]
    print()
    
    test3_passed = test_model_env_compatibility()
    print()
    
    # Summary
    print("📋 DIAGNOSTIC SUMMARY:")
    print(f"   Model Loading: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Environment Creation: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Model-Env Compatibility: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED - Issue is likely in training loop")
    else:
        print("\n🚨 TESTS FAILED - Found the root cause!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)