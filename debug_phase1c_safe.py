#!/usr/bin/env python3
"""
🔍 PHASE 1C DEBUG - SAFE UNICODE HANDLING
Isolates each step to identify JSON encoding issues
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def safe_json_dump(data, filepath):
    """Safely dump data to JSON with proper Unicode handling"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"JSON dump failed: {e}")
        return False

def test_step1_data_loading():
    """Test 1: Data loading with Unicode safety"""
    logger.info("🧪 TEST 1: Safe data loading...")
    
    try:
        # Database connection
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        # Load small sample first (3 days from 2022 - known to have data)
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        logger.info("📈 Loading 3-day sample from 2022...")
        
        market_data = adapter.load_training_data(
            start_date='2022-01-03',  # Known data range
            end_date='2022-01-05',    # 3 days
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        # Check for Unicode issues in data
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        trading_days = market_data['trading_days']
        
        logger.info(f"✅ Data loaded: {len(trading_days)} timesteps")
        logger.info(f"   NVDA features shape: {nvda_features.shape}")
        logger.info(f"   NVDA prices shape: {nvda_prices.shape}")
        
        # Test data types and ranges
        logger.info("🔍 Checking for NaN/Inf values...")
        nvda_nan_count = np.isnan(nvda_features).sum()
        nvda_inf_count = np.isinf(nvda_features).sum() 
        
        if nvda_nan_count > 0:
            logger.warning(f"⚠️ Found {nvda_nan_count} NaN values in NVDA features")
        if nvda_inf_count > 0:
            logger.warning(f"⚠️ Found {nvda_inf_count} Inf values in NVDA features")
            
        # Test safe serialization
        sample_data = {
            'timesteps': len(trading_days),
            'feature_shape': list(nvda_features.shape),
            'price_range': [float(nvda_prices.min()), float(nvda_prices.max())],
            'sample_trading_days': [str(d) for d in trading_days[:5]]  # Convert to string
        }
        
        # Test JSON serialization
        test_json_path = Path("debug_sample_data.json")
        if safe_json_dump(sample_data, test_json_path):
            logger.info("✅ JSON serialization test passed")
            test_json_path.unlink()  # Clean up
        else:
            logger.error("❌ JSON serialization test failed")
            return False
            
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        traceback.print_exc()
        return None

def test_step2_feature_processing(market_data):
    """Test 2: Feature processing with Unicode safety"""
    logger.info("🧪 TEST 2: Safe feature processing...")
    
    try:
        # Extract components
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']  
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        # Process features safely
        logger.info("🔄 Processing original features...")
        original_features = np.concatenate([nvda_features, msft_features], axis=1)
        position_features = np.zeros((original_features.shape[0], 2))
        original_features = np.concatenate([original_features, position_features], axis=1)
        
        # Create price data
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        original_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
        
        # Clean any problematic values
        original_features = np.nan_to_num(original_features, nan=0.0, posinf=1.0, neginf=-1.0)
        original_prices = np.nan_to_num(original_prices, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logger.info("✅ Feature processing completed safely")
        logger.info(f"   Final features shape: {original_features.shape}")
        logger.info(f"   Final prices shape: {original_prices.shape}")
        
        # Test parquet save (avoids JSON issues)
        test_df = pd.DataFrame(original_features[:100])  # Small sample
        test_path = Path("debug_features_test.parquet")
        test_df.to_parquet(test_path)
        logger.info("✅ Parquet save test passed")
        test_path.unlink()  # Clean up
        
        return {
            'features': original_features,
            'prices': original_prices,
            'trading_days': trading_days
        }
        
    except Exception as e:
        logger.error(f"❌ Feature processing test failed: {e}")
        traceback.print_exc()
        return None

def test_step3_environment_basic(processed_data):
    """Test 3: Basic environment creation without Monitor wrapper"""
    logger.info("🧪 TEST 3: Basic environment creation...")
    
    try:
        # Import environment
        from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
        
        # Simple config
        env_config = {
            'initial_capital': 10000.0,
            'lookback_window': 50,
            'max_episode_steps': 100,  # Small for testing
            'max_daily_drawdown_pct': 0.30,
            'transaction_cost_pct': 0.001
        }
        
        # Create environment WITHOUT Monitor wrapper
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=processed_data['features'],
            processed_price_data=processed_data['prices'],
            trading_days=processed_data['trading_days'],
            **env_config
        )
        
        # Test basic operations
        logger.info("🔄 Testing environment reset...")
        obs, info = env.reset(seed=42)
        logger.info(f"   Observation shape: {obs.shape}")
        logger.info(f"   Info keys: {list(info.keys()) if info else 'None'}")
        
        # Test a few steps
        logger.info("🔄 Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"   Step {i+1}: action={action}, reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        logger.info("✅ Basic environment test passed")
        return env
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Safe debugging main function"""
    logger.info("🔍 PHASE 1C SAFE DEBUG")
    logger.info("=" * 50)
    
    # Test 1: Data loading
    market_data = test_step1_data_loading()
    if market_data is None:
        logger.error("❌ Failed at data loading step")
        return False
    
    # Test 2: Feature processing
    processed_data = test_step2_feature_processing(market_data)
    if processed_data is None:
        logger.error("❌ Failed at feature processing step")
        return False
    
    # Test 3: Basic environment
    env = test_step3_environment_basic(processed_data)
    if env is None:
        logger.error("❌ Failed at environment creation step")
        return False
    
    logger.info("🎉 ALL SAFE TESTS PASSED!")
    logger.info("✅ Ready to proceed with full Phase 1C implementation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)