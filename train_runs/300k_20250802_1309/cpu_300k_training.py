#!/usr/bin/env python3
"""
CPU-Only 300K Training - Eliminate GPU instability issues
"""

import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime

# Force CPU-only training
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 STARTING CPU-ONLY 300K MODEL TRAINING")
        print("=" * 60)
        print("🖥️  FORCING CPU-ONLY MODE TO ELIMINATE GPU INSTABILITY")
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        import torch
        
        # Verify CPU-only mode
        print(f"🔧 PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"🔧 PyTorch device: {torch.device('cpu')}")
        
        # Configuration - smaller steps for CPU
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 25000,  # Start with smaller chunk for CPU
            'save_dir': 'train_runs/300k_20250802_1309/models'
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,} (CPU chunk)")
        
        # Setup environment
        print("📊 Setting up environment...")
        
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        # Load training data
        print('📥 Loading training data from TimescaleDB...')
        training_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2025-08-01', 
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        print(f'✅ Loaded {len(training_data["trading_days"])} trading periods')
        
        # Create environment with conservative settings for CPU
        env = DualTickerTradingEnv(
            nvda_data=training_data['nvda_features'],
            msft_data=training_data['msft_features'], 
            nvda_prices=training_data['nvda_prices'],
            msft_prices=training_data['msft_prices'],
            trading_days=training_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=500,  # Shorter episodes for CPU
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        print('✅ CPU Environment initialized')
        
        # Load base model for CPU
        print('🤖 Loading 201K base model for CPU...')
        model = RecurrentPPO.load(config['base_model_path'], env=env, device='cpu')
        print('✅ CPU model loaded successfully')
        
        # CPU-optimized callback
        class CPUCallback(BaseCallback):
            def __init__(self, save_freq=5000, save_path="train_runs/300k_20250802_1309/models"):
                super().__init__()
                self.save_freq = save_freq
                self.save_path = save_path
                os.makedirs(save_path, exist_ok=True)
                
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = f"{self.save_path}/cpu_checkpoint_{self.n_calls}_steps"
                    self.model.save(checkpoint_path)
                    print(f"💾 CPU checkpoint saved: {checkpoint_path}")
                
                if self.n_calls % 1000 == 0:
                    print(f"📈 CPU progress: {self.n_calls:,}/{config['training_steps']:,} steps")
                
                return True
        
        # Create callback
        callback = CPUCallback()
        
        print('📈 Starting CPU-only training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train on CPU only
        model.learn(
            total_timesteps=config['training_steps'],
            callback=callback,
            log_interval=20,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        print(f'✅ CPU training completed in {training_time:.1f} seconds')
        
        # Save final model
        os.makedirs(config['save_dir'], exist_ok=True)
        final_model_path = f"{config['save_dir']}/cpu_300k_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.save(final_model_path)
        print(f'💾 CPU model saved: {final_model_path}')
        
        print('🎉 CPU-ONLY 300K TRAINING COMPLETED!')
        print('💡 If stable, we can continue with more CPU chunks or try GPU again')
        
    except Exception as e:
        print(f"❌ CPU training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)