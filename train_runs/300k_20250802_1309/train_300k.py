#\!/usr/bin/env python3
"""
300K Model Training - Live Data Integration
Transfer learning from 201K model foundation
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.dual_ticker_model_adapter import DualTickerModelAdapter
from secrets_helper import SecretsHelper
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print('🚀 STARTING 300K MODEL TRAINING')
    print('=' * 40)
    
    # Configuration
    config = {
        'base_model': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
        'target_steps': 300000,
        'learning_rate': 7e-5,
        'checkpoint_every': 25000,
        'log_dir': 'logs/dual_300k_20250802_1309',
        'save_dir': 'models/300k_training'
    }
    
    # Database config for live data
    db_password = SecretsHelper.get_timescaledb_password()
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading_data',
        'user': 'postgres',
        'password': db_password
    }
    
    print('✅ Configuration loaded')
    print(f'📦 Base model: {config["base_model"]}')
    print(f'🎯 Target steps: {config["target_steps"]:,}')
    
    # Initialize training
    try:
        # Initialize adapter with base model path
        adapter = DualTickerModelAdapter(base_model_path=config['base_model'])
        print('✅ Model adapter initialized')
        
        # Load training data (using available historical data)
        print('📊 Loading training data...')
        # This will use our newly fetched historical data
        
        # Start training preparation
        print('🎯 Training preparation complete')
        print('⚠️  Ready for actual training launch')
        
    except Exception as e:
        logger.error(f'Training preparation error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print('✅ Training preparation complete')
    else:
        print('❌ Training preparation failed')
