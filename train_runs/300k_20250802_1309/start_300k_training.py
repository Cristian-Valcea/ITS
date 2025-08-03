#!/usr/bin/env python3
"""
300K Model Training - Full Implementation
Transfer learning from 201K model foundation with actual training loop
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.dual_ticker_model_adapter import DualTickerModelAdapter
from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from secrets_helper import SecretsHelper
import logging
import time
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress"""
    
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            logger.info(f"Step {self.n_calls}: Episode reward mean = {self.locals.get('ep_rew_mean', 'N/A')}")
        return True

def main():
    print('🚀 STARTING 300K MODEL TRAINING - FULL IMPLEMENTATION')
    print('=' * 50)
    
    # Configuration
    config = {
        'base_model': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
        'current_steps': 201000,  # Starting from 201K
        'target_steps': 300000,
        'training_steps': 99000,  # Additional steps needed (300K - 201K)
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
    print(f'🎯 Current steps: {config["current_steps"]:,}')
    print(f'🎯 Target steps: {config["target_steps"]:,}')
    print(f'🎯 Training steps: {config["training_steps"]:,}')
    
    # Initialize training
    try:
        print('🔧 Initializing model adapter...')
        adapter = DualTickerModelAdapter(base_model_path=config['base_model'])
        print('✅ Model adapter initialized')
        
        print('📊 Setting up dual-ticker environment...')
        # Create data adapter
        data_adapter = DualTickerDataAdapter(db_config)
        
        # Load training data from TimescaleDB
        print('📥 Loading training data from TimescaleDB...')
        training_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2025-08-01', 
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        print(f'✅ Loaded {len(training_data["trading_days"])} trading periods')
        print(f'📊 NVDA features shape: {training_data["nvda_features"].shape}')
        print(f'📊 MSFT features shape: {training_data["msft_features"].shape}')
        
        # Combine features for V3 environment (expects single processed_feature_data array)
        # V3 format: [N, 26] where 26 = 12 NVDA + 1 NVDA_pos + 12 MSFT + 1 MSFT_pos
        nvda_features = training_data["nvda_features"]  # [N, 12]
        msft_features = training_data["msft_features"]  # [N, 12]
        
        # Combine features: [NVDA_12_features, MSFT_12_features] = [N, 24]
        # Note: V3 environment will add position features internally
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        
        # Use NVDA prices as primary price data (V3 environment expects single price series)
        price_data = training_data["nvda_prices"]
        
        print(f'📐 Combined features shape: {combined_features.shape}')
        
        # Create V3 environment
        env = DualTickerTradingEnvV3(
            processed_feature_data=combined_features,
            price_data=price_data,
            initial_capital=100000.0,
            max_episode_steps=1000,
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=True,
            verbose=True
        )
        
        print('✅ Environment initialized')
        
        print('🤖 Creating new adapted model...')
        # Create new model adapted for dual-ticker trading
        model = adapter.prepare_dual_ticker_model(env)
        print('✅ Adapted model created successfully')
        
        print('📈 Starting training...')
        start_time = time.time()
        
        # Create callback for progress tracking
        callback = TrainingProgressCallback(log_freq=5000)
        
        # Train the model
        model.learn(
            total_timesteps=config['training_steps'],
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f'✅ Training completed in {training_time:.1f} seconds')
        
        # Save the final model
        os.makedirs(config['save_dir'], exist_ok=True)
        final_model_path = f"{config['save_dir']}/dual_ticker_300k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.save(final_model_path)
        print(f'💾 Model saved: {final_model_path}')
        
        print('🎉 300K TRAINING COMPLETE!')
        print('=' * 50)
        
    except Exception as e:
        logger.error(f'Training error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print('✅ 300K training completed successfully')
    else:
        print('❌ 300K training failed')