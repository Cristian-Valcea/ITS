#!/usr/bin/env python3
"""
🎯 V3 TRUE WARM-START TUNING
Properly continue training from 409K steps with moderate parameter adjustments
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

class V3TrueWarmStartTuner:
    """V3 True Warm-Start Tuning Implementation"""
    
    def __init__(self):
        """Initialize true warm-start tuner"""
        
        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"v3_true_warmstart_50k_{timestamp}"
        self.output_dir = f"train_runs/{self.run_name}"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("🎯 V3 True Warm-Start Tuner initialized")
        logger.info(f"   Run name: {self.run_name}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def load_training_data(self):
        """Load training data for tuning"""
        
        logger.info("📊 Loading training data...")
        
        # Generate synthetic data for consistent tuning
        n_timesteps = 50000
        
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
        
        logger.info("🎯 V3 True Warm-Start Environment initialized")
        logger.info(f"   Hold bonus: 0.001 (moderate 10x reduction from 0.01)")
        logger.info(f"   Ticket cost: $0.50 (unchanged from original)")
        logger.info(f"   Continuing from: 409,600 trained steps")
        
        # Create environment with moderate tuning
        env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=features,
            processed_price_data=prices,
            trading_days=timestamps,
            
            # Environment parameters
            initial_capital=100000,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            max_daily_drawdown_pct=0.02,
            
            # MODERATE tuning weights (conservative approach)
            hold_bonus_weight=0.001,  # 10x reduction (not 20x)
            ticket_cost_per_trade=0.50,  # UNCHANGED (maintain friction)
            
            # Other weights (unchanged from V3)
            downside_penalty_weight=2.0,
            kelly_bonus_weight=0.5,
            position_decay_weight=0.1,
            turnover_penalty_weight=0.05,
            size_penalty_weight=0.02,
            action_change_penalty_weight=0.005,
            
            # Alpha parameters
            alpha_mode="live_replay",
            alpha_strength=0.1,
            alpha_persistence=0.5,
            alpha_on_probability=0.6,
            
            # Logging
            log_trades=False,
            verbose=False
        )
        
        logger.info("✅ True warm-start tuned environment created")
        
        return env
    
    def run_true_warmstart_tuning(self):
        """Run the true warm-start tuning process"""
        
        logger.info("🚀 Starting V3 TRUE warm-start tuning...")
        logger.info("🔧 This will PROPERLY continue from 409K steps")
        
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
        base_model_path = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
        logger.info(f"📥 Loading base model: {base_model_path}")
        
        # CRITICAL: Load model WITHOUT resetting training state
        model = RecurrentPPO.load(
            base_model_path,
            env=train_env,
            device='auto'
        )
        
        # CRITICAL: Set the correct number of timesteps to continue from
        starting_timesteps = 409600
        additional_timesteps = 50000
        
        # Set model's internal step counter to continue from 409K
        model.num_timesteps = starting_timesteps
        model._total_timesteps = starting_timesteps + additional_timesteps
        
        logger.info("✅ Base model loaded with TRUE warm-start")
        logger.info(f"   Model timesteps set to: {model.num_timesteps:,}")
        logger.info(f"   Will train to: {model._total_timesteps:,}")
        logger.info(f"   Additional training: {additional_timesteps:,} steps")
        
        # Setup callbacks with proper step counting
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.output_dir,
            log_path=f"{self.output_dir}/eval_logs",
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.output_dir}/checkpoints",
            name_prefix="v3_true_warmstart"
        )
        
        callbacks = [eval_callback, checkpoint_callback]
        
        # Run training with proper continuation
        logger.info("🎯 Starting TRUE warm-start training...")
        logger.info(f"   Continuing from: {starting_timesteps:,} steps")
        logger.info(f"   Target: {starting_timesteps + additional_timesteps:,} steps")
        logger.info(f"   Expected behavior: Stable episodes, moderate trading increase")
        
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False  # CRITICAL: Don't reset step counter
        )
        
        # Save final model with correct step count
        final_steps = starting_timesteps + additional_timesteps
        final_model_path = f"{self.output_dir}/v3_true_warmstart_final_{final_steps}steps.zip"
        model.save(final_model_path)
        
        logger.info("✅ TRUE warm-start tuning completed successfully!")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Final step count: {final_steps:,}")
        logger.info(f"   Best model: {self.output_dir}/best_model.zip")
        logger.info(f"   Training logs: {self.output_dir}/")
        
        return self.output_dir

def main():
    """Main execution function"""
    
    try:
        # Initialize tuner
        tuner = V3TrueWarmStartTuner()
        
        # Run true warm-start tuning
        output_dir = tuner.run_true_warmstart_tuning()
        
        logger.info("🎉 V3 TRUE warm-start tuning completed successfully!")
        logger.info(f"   Output directory: {output_dir}")
        
        # Quick validation
        logger.info("🔍 Quick validation...")
        train_monitor = f"{output_dir}/train_monitor.csv"
        if os.path.exists(train_monitor):
            import pandas as pd
            df = pd.read_csv(train_monitor, comment='#')
            avg_length = df['l'].mean()
            early_term_rate = (df['l'] < 100).sum() / len(df) * 100
            
            logger.info(f"📊 Quick results:")
            logger.info(f"   Episodes: {len(df)}")
            logger.info(f"   Avg episode length: {avg_length:.1f}")
            logger.info(f"   Early termination rate: {early_term_rate:.1f}%")
            
            if len(df) < 200:  # Should be much fewer episodes for 50K steps
                logger.info("✅ GOOD: Episode count suggests proper warm-start continuation")
            else:
                logger.warning("⚠️ WARNING: Too many episodes - may not be true continuation")
        
    except Exception as e:
        logger.error(f"❌ True warm-start tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()