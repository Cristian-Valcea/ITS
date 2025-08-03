#!/usr/bin/env python3
"""
🎯 V3 MODERATE TUNING IMPLEMENTATION
Conservative parameter adjustments to maintain V3 philosophy while increasing trading activity
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.gym_env.dual_ticker_trading_env_v3_tuned import DualTickerTradingEnvV3Tuned
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V3ModerateTuner:
    """V3 Moderate Tuning Implementation"""
    
    def __init__(self, config_path: str = "config/v3_moderate_tuning.yml"):
        """Initialize moderate tuner"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.config['run_name']}_{timestamp}"
        self.output_dir = f"train_runs/{self.run_name}"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("✅ Configuration loaded from " + config_path)
        logger.info("🎯 V3 Moderate Tuner initialized")
        logger.info(f"   Run name: {self.run_name}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def load_training_data(self):
        """Load training data for tuning"""
        
        logger.info("📊 Loading training data...")
        
        # Try to load real data first
        try:
            data_adapter = DualTickerDataAdapter()
            
            # Load dual-ticker data
            feature_data, price_data, timestamps = data_adapter.load_dual_ticker_data(
                tickers=self.config['data']['tickers'],
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date'],
                split=self.config['data']['split']
            )
            
            logger.info(f"✅ Real data loaded: {len(feature_data)} timesteps")
            return feature_data, price_data, timestamps
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load real data: {str(e)}")
            logger.info("📊 Falling back to synthetic data for tuning...")
            
            # Generate synthetic data
            n_timesteps = self.config['data']['synthetic_timesteps']
            
            # Generate realistic features (26-dimensional)
            features = np.random.randn(n_timesteps, 26).astype(np.float32)
            
            # Generate correlated price movements
            base_prices = np.array([500.0, 500.0, 300.0, 300.0])  # NVDA OHLC, MSFT OHLC
            price_changes = np.random.randn(n_timesteps, 4) * 0.015  # 1.5% volatility
            prices = base_prices * (1 + price_changes.cumsum(axis=0))
            
            # Generate timestamps
            timestamps = pd.date_range('2025-01-01', periods=n_timesteps, freq='1min')
            
            logger.info(f"✅ Synthetic data generated: {n_timesteps:,} timesteps")
            logger.info(f"   Features shape: {features.shape}")
            logger.info(f"   Prices shape: {prices.shape}")
            
            return features, prices, timestamps.values
    
    def create_tuned_environment(self, features, prices, timestamps):
        """Create tuned environment with moderate parameters"""
        
        # Get tuning parameters
        tuning_params = self.config['reward_tuning']
        env_params = self.config['environment']
        
        logger.info("🎯 V3 Moderate Reward System initialized")
        logger.info(f"   Hold bonus: {tuning_params['hold_bonus_weight']} (moderate reduction from 0.01)")
        logger.info(f"   Ticket cost: ${tuning_params['ticket_cost_per_trade']} (unchanged from original)")
        logger.info(f"   Base version: v3_gold_standard_400k_20250802_202736")
        
        # Create environment
        env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=features,
            processed_price_data=prices,
            trading_days=timestamps,
            
            # Environment parameters
            initial_capital=env_params['initial_capital'],
            max_position_size=env_params['max_position_size'],
            transaction_cost_pct=env_params['transaction_cost_pct'],
            max_daily_drawdown_pct=env_params['max_daily_drawdown_pct'],
            
            # Moderate tuning weights
            hold_bonus_weight=tuning_params['hold_bonus_weight'],
            ticket_cost_per_trade=tuning_params['ticket_cost_per_trade'],
            
            # Other weights (unchanged)
            downside_penalty_weight=tuning_params['downside_penalty_weight'],
            kelly_bonus_weight=tuning_params['kelly_bonus_weight'],
            position_decay_weight=tuning_params['position_decay_weight'],
            turnover_penalty_weight=tuning_params['turnover_penalty_weight'],
            size_penalty_weight=tuning_params['size_penalty_weight'],
            action_change_penalty_weight=tuning_params['action_change_penalty_weight'],
            
            # Alpha parameters
            alpha_mode=env_params['alpha_mode'],
            alpha_strength=env_params['alpha_strength'],
            alpha_persistence=env_params['alpha_persistence'],
            alpha_on_probability=env_params['alpha_on_probability'],
            
            # Logging
            log_trades=env_params['log_trades'],
            verbose=env_params['verbose']
        )
        
        logger.info("✅ Moderate tuned environment created")
        logger.info(f"   Hold bonus weight: {tuning_params['hold_bonus_weight']}")
        logger.info(f"   Ticket cost: ${tuning_params['ticket_cost_per_trade']}")
        
        return env
    
    def run_moderate_tuning(self):
        """Run the moderate tuning process"""
        
        logger.info("🚀 Starting V3 moderate tuning...")
        
        # Load training data
        features, prices, timestamps = self.load_training_data()
        
        # Create environments
        train_env = self.create_tuned_environment(features, prices, timestamps)
        eval_env = self.create_tuned_environment(features, prices, timestamps)
        
        # Wrap environments
        train_env = Monitor(train_env, f"{self.output_dir}/train_monitor.csv")
        eval_env = Monitor(eval_env, f"{self.output_dir}/eval_monitor.csv")
        
        train_env = DummyVecEnv([lambda: train_env])
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Load base model
        base_model_path = self.config['base_model_path']
        logger.info(f"📥 Loading base model: {base_model_path}")
        
        model = RecurrentPPO.load(
            base_model_path,
            env=train_env,
            device='auto'
        )
        
        # Update model parameters if needed
        training_params = self.config['training']
        model.learning_rate = training_params['learning_rate']
        
        logger.info("✅ Base model loaded and configured")
        logger.info(f"   Starting from: {self.config['starting_timesteps']:,} steps")
        logger.info(f"   Additional training: {training_params['total_timesteps']:,} steps")
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.output_dir,
            log_path=f"{self.output_dir}/eval_logs",
            eval_freq=training_params['eval_freq'],
            n_eval_episodes=training_params['n_eval_episodes'],
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=training_params['save_freq'],
            save_path=f"{self.output_dir}/checkpoints",
            name_prefix="v3_moderate"
        )
        
        callbacks = [eval_callback, checkpoint_callback]
        
        # Run training
        logger.info("🎯 Starting moderate tuning training...")
        logger.info(f"   Target episode length: {self.config['expected_outcomes']['episode_length_target']}")
        logger.info(f"   Target trades/day: {self.config['expected_outcomes']['target_trades_per_day']}")
        
        model.learn(
            total_timesteps=training_params['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = f"{self.output_dir}/v3_moderate_final_{self.config['starting_timesteps'] + training_params['total_timesteps']}steps.zip"
        model.save(final_model_path)
        
        logger.info("✅ Moderate tuning completed successfully!")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Best model: {self.output_dir}/best_model.zip")
        logger.info(f"   Training logs: {self.output_dir}/")
        
        return self.output_dir

def main():
    """Main execution function"""
    
    try:
        # Initialize tuner
        tuner = V3ModerateTuner()
        
        # Run moderate tuning
        output_dir = tuner.run_moderate_tuning()
        
        logger.info("🎉 V3 moderate tuning completed successfully!")
        logger.info(f"   Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Moderate tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()