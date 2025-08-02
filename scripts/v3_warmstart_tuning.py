#!/usr/bin/env python3
"""
🎯 V3 WARM-START TUNING SCRIPT
Fine-tune hold-bonus and ticket-cost weights with warm-start from existing model

OBJECTIVE: Increase trading activity while preserving core performance
- Load v3_gold_standard_final_409600steps.zip
- Retrain 50K steps with tuned weights
- Preserve 409K steps of existing learning

TUNING CHANGES:
- Hold bonus: 0.01 → 0.0005 (20x reduction)
- Ticket cost: $0.50 → $0.20 (60% reduction)
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.gym_env.dual_ticker_trading_env_v3_tuned import DualTickerTradingEnvV3Tuned

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class V3WarmStartTuner:
    """
    V3 warm-start tuning system for trading behavior adjustment
    """
    
    def __init__(self, config_path: str):
        """
        Initialize warm-start tuner
        
        Args:
            config_path: Path to tuning configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.run_name = f"{self.config['output_config']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(self.config['output_config']['base_dir']) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 V3 Warm-Start Tuner initialized")
        logger.info(f"   Run name: {self.run_name}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def _load_config(self) -> dict:
        """Load tuning configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Configuration loaded from {self.config_path}")
        return config
    
    def _load_data(self) -> tuple:
        """
        Load and prepare training data
        
        Returns:
            Tuple of (features, prices, timestamps)
        """
        logger.info("📊 Loading training data...")
        
        try:
            # Try to load real data using the same approach as original V3 training
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
            
            # Create data adapter
            data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
            
            # Load training data
            data_config = self.config['data_config']
            training_data = data_adapter.load_training_data(
                start_date=data_config['start_date'],
                end_date=data_config['end_date'],
                symbols=data_config['symbols'],
                bar_size=data_config['bar_size'],
                data_split='train'
            )
            
            # Prepare data for V3 environment (same as original)
            nvda_features = training_data['nvda_features']
            msft_features = training_data['msft_features']
            nvda_prices = training_data['nvda_prices']
            msft_prices = training_data['msft_prices']
            
            # Combine features: [NVDA features, MSFT features, alpha signals]
            # V3 expects 26 features: 12 NVDA + 12 MSFT + 2 alpha signals
            n_timesteps = len(nvda_features)
            
            # Add placeholder alpha signals (will be generated dynamically in environment)
            nvda_alpha_placeholder = np.zeros((n_timesteps, 1))
            msft_alpha_placeholder = np.zeros((n_timesteps, 1))
            
            features = np.concatenate([
                nvda_features,      # 12 features
                nvda_alpha_placeholder,  # 1 alpha signal
                msft_features,      # 12 features  
                msft_alpha_placeholder   # 1 alpha signal
            ], axis=1)  # Total: 26 features
            
            # Combine prices: [NVDA open, close, MSFT open, close]
            # Extract OHLC from price data (assuming columns: open, high, low, close)
            if nvda_prices.shape[1] >= 4 and msft_prices.shape[1] >= 4:
                prices = np.column_stack([
                    nvda_prices[:, 0],  # NVDA open
                    nvda_prices[:, 3],  # NVDA close
                    msft_prices[:, 0],  # MSFT open
                    msft_prices[:, 3]   # MSFT close
                ])
            else:
                # Fallback: use close prices for both open and close
                prices = np.column_stack([
                    nvda_prices[:, -1],  # NVDA close as open
                    nvda_prices[:, -1],  # NVDA close
                    msft_prices[:, -1],  # MSFT close as open
                    msft_prices[:, -1]   # MSFT close
                ])
            
            timestamps = training_data['trading_days']
            
            logger.info(f"✅ Real data loaded: {len(features):,} timesteps")
            logger.info(f"   Features shape: {features.shape}")
            logger.info(f"   Prices shape: {prices.shape}")
            
            return features, prices, timestamps
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load real data: {str(e)}")
            logger.info("📊 Falling back to synthetic data for tuning...")
            
            # Fallback to synthetic data
            n_timesteps = 50000  # Enough for 50K step training
            
            # Generate realistic features (26 dimensions: 12 NVDA + 1 alpha + 12 MSFT + 1 alpha)
            features = np.random.randn(n_timesteps, 26).astype(np.float32)
            
            # Generate realistic prices (NVDA open, close, MSFT open, close)
            base_prices = np.array([500.0, 500.0, 300.0, 300.0])
            price_changes = np.random.randn(n_timesteps, 4) * 0.015  # 1.5% volatility
            prices = base_prices * (1 + price_changes.cumsum(axis=0))
            
            # Generate timestamps
            timestamps = pd.date_range('2025-01-01', periods=n_timesteps, freq='1min')
            
            logger.info(f"✅ Synthetic data generated: {n_timesteps:,} timesteps")
            logger.info(f"   Features shape: {features.shape}")
            logger.info(f"   Prices shape: {prices.shape}")
            
            return features, prices, timestamps.values
    
    def _create_environment(self, features: np.ndarray, prices: np.ndarray, timestamps: np.ndarray):
        """
        Create tuned V3 environment
        
        Args:
            features: Feature data
            prices: Price data  
            timestamps: Timestamp data
            
        Returns:
            Tuned environment instance
        """
        env_config = self.config['environment_config']
        
        env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=features,
            processed_price_data=prices,
            trading_days=timestamps,
            # Core parameters (unchanged)
            initial_capital=env_config['initial_capital'],
            lookback_window=env_config['lookback_window'],
            max_episode_steps=env_config['max_episode_steps'],
            max_daily_drawdown_pct=env_config['max_daily_drawdown_pct'],
            max_position_size=env_config['max_position_size'],
            transaction_cost_pct=env_config['transaction_cost_pct'],
            base_impact_bp=env_config['base_impact_bp'],
            impact_exponent=env_config['impact_exponent'],
            risk_free_rate_annual=env_config['risk_free_rate_annual'],
            # 🎯 TUNED WEIGHTS
            hold_bonus_weight=env_config['hold_bonus_weight'],
            ticket_cost_per_trade=env_config['ticket_cost_per_trade'],
            # Other weights (unchanged)
            downside_penalty_weight=env_config.get('downside_penalty_weight', 2.0),
            kelly_bonus_weight=env_config.get('kelly_bonus_weight', 0.5),
            position_decay_weight=env_config.get('position_decay_weight', 0.1),
            turnover_penalty_weight=env_config.get('turnover_penalty_weight', 0.05),
            size_penalty_weight=env_config.get('size_penalty_weight', 0.02),
            action_change_penalty_weight=env_config.get('action_change_penalty_weight', 0.005),
            # Alpha configuration
            alpha_mode=self.config['curriculum_phases'][0]['alpha_mode'],
            log_trades=env_config['log_trades'],
            verbose=env_config['verbose']
        )
        
        logger.info("✅ Tuned V3 environment created")
        logger.info(f"   Hold bonus weight: {env_config['hold_bonus_weight']}")
        logger.info(f"   Ticket cost: ${env_config['ticket_cost_per_trade']}")
        
        return env
    
    def _load_base_model(self) -> RecurrentPPO:
        """
        Load the base V3 model for warm-start
        
        Returns:
            Loaded RecurrentPPO model
        """
        base_model_path = self.config['training_config']['base_model_path']
        
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        
        logger.info(f"📦 Loading base model from {base_model_path}")
        
        # Load the model
        model = RecurrentPPO.load(base_model_path)
        
        logger.info("✅ Base model loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        
        return model
    
    def _setup_callbacks(self, eval_env):
        """
        Setup training callbacks
        
        Args:
            eval_env: Evaluation environment
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['checkpoint_config']['save_freq'],
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="v3_tuned"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir),
            log_path=str(self.output_dir / "eval_logs"),
            eval_freq=self.config['monitoring_config']['eval_freq'],
            n_eval_episodes=self.config['monitoring_config']['eval_episodes'],
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        logger.info("✅ Training callbacks configured")
        
        return callbacks
    
    def run_tuning(self):
        """
        Execute the warm-start tuning process
        """
        logger.info("🚀 Starting V3 warm-start tuning...")
        
        # Load data
        features, prices, timestamps = self._load_data()
        
        # Create environments
        train_env = self._create_environment(features, prices, timestamps)
        eval_env = self._create_environment(features, prices, timestamps)
        
        # Wrap environments
        train_env = Monitor(train_env, str(self.output_dir / "train_monitor.csv"))
        eval_env = Monitor(eval_env, str(self.output_dir / "eval_monitor.csv"))
        
        # Load base model
        model = self._load_base_model()
        
        # Update environment for the loaded model
        model.set_env(DummyVecEnv([lambda: train_env]))
        
        # Setup callbacks
        callbacks = self._setup_callbacks(eval_env)
        
        # Configure training parameters
        total_timesteps = self.config['training_config']['total_timesteps']
        
        logger.info(f"🎯 Starting warm-start training for {total_timesteps:,} steps")
        logger.info(f"   Base model: {self.config['training_config']['base_model_path']}")
        logger.info(f"   Tuned weights: hold_bonus={self.config['environment_config']['hold_bonus_weight']}, "
                   f"ticket_cost=${self.config['environment_config']['ticket_cost_per_trade']}")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=self.config['monitoring_config']['log_interval'],
            tb_log_name=self.run_name,
            reset_num_timesteps=False  # Continue from existing timesteps
        )
        
        # Save final model
        final_model_path = self.output_dir / f"v3_tuned_final_{total_timesteps}steps.zip"
        model.save(str(final_model_path))
        
        logger.info(f"✅ Tuning complete! Model saved to {final_model_path}")
        
        # Save tuning summary
        self._save_tuning_summary(train_env, final_model_path)
        
        return str(final_model_path)
    
    def _save_tuning_summary(self, env, model_path: Path):
        """
        Save tuning summary and analysis
        
        Args:
            env: Training environment
            model_path: Path to saved model
        """
        summary = {
            'tuning_info': {
                'run_name': self.run_name,
                'tuning_date': datetime.now().isoformat(),
                'base_model': self.config['training_config']['base_model_path'],
                'final_model': str(model_path),
                'total_timesteps': self.config['training_config']['total_timesteps']
            },
            'weight_changes': {
                'hold_bonus_weight': {
                    'original': 0.01,
                    'tuned': self.config['environment_config']['hold_bonus_weight'],
                    'reduction_factor': 0.01 / self.config['environment_config']['hold_bonus_weight']
                },
                'ticket_cost_per_trade': {
                    'original': 0.50,
                    'tuned': self.config['environment_config']['ticket_cost_per_trade'],
                    'reduction_factor': 0.50 / self.config['environment_config']['ticket_cost_per_trade']
                }
            },
            'environment_info': env.get_tuning_summary(),
            'objectives': self.config.get('tuning_objectives', {}),
            'config': self.config
        }
        
        summary_path = self.output_dir / "tuning_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"📋 Tuning summary saved to {summary_path}")

def main():
    """Main execution function"""
    
    # Configuration path
    config_path = "config/v3_tuned_warmstart.yml"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    try:
        # Initialize tuner
        tuner = V3WarmStartTuner(config_path)
        
        # Run tuning
        final_model_path = tuner.run_tuning()
        
        logger.info("🎉 V3 warm-start tuning completed successfully!")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Output directory: {tuner.output_dir}")
        
        # Print next steps
        print("\n🎯 NEXT STEPS:")
        print(f"1. Evaluate tuned model: {final_model_path}")
        print(f"2. Compare trading frequency with original V3 model")
        print(f"3. Analyze reward components in: {tuner.output_dir}")
        print(f"4. If satisfactory, deploy for paper trading")
        
    except Exception as e:
        logger.error(f"❌ Tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()