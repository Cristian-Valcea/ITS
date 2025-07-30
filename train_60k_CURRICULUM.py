#!/usr/bin/env python3
"""
📚 FRICTION CURRICULUM TRAINING - 60K STEPS
Progressive friction: 0.5bp → 1.2bp + domain randomization
Prevents over-fitting to unrealistically low trading costs
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import our components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from friction_curriculum_callback import FrictionCurriculumCallback, DomainRandomizedTradingEnv

class CurriculumTradingEnv(DualTickerTradingEnv):
    """Enhanced environment with friction curriculum support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store initial friction for curriculum
        self.base_tc_bp = self.tc_bp
        self.base_trade_penalty_bp = self.trade_penalty_bp
        logger.info(f"📚 CURRICULUM ENV: Base friction {self.base_tc_bp}bp/{self.base_trade_penalty_bp}bp")
    
    def set_friction(self, tc_bp, trade_penalty_bp):
        """Update friction parameters dynamically"""
        self.tc_bp = tc_bp
        self.trade_penalty_bp = trade_penalty_bp
        # Note: changes take effect on next episode reset
    
    def reset(self, **kwargs):
        """Reset with domain randomization per episode"""
        # Apply small episode-level randomization
        episode_tc_noise = np.random.uniform(-0.1, 0.1)  # ±0.1bp per episode
        episode_penalty_noise = np.random.uniform(-0.2, 0.2)  # ±0.2bp per episode
        
        self.current_episode_tc = max(0.1, self.tc_bp + episode_tc_noise)
        self.current_episode_penalty = max(0.1, self.trade_penalty_bp + episode_penalty_noise)
        
        return super().reset(**kwargs)
    
    def _calculate_transaction_cost(self, action, nvda_trade_value, msft_trade_value):
        """Use episode-specific friction for transaction costs"""
        if hasattr(self, 'current_episode_tc'):
            # Use curriculum + episode randomized costs
            total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
            transaction_cost = total_trade_value * (self.current_episode_tc / 10000)
            return transaction_cost
        else:
            # Fallback to original calculation
            return super()._calculate_transaction_cost(action, nvda_trade_value, msft_trade_value)

def create_curriculum_environment():
    """Create curriculum training environment"""
    logger.info("📚 Creating FRICTION CURRICULUM training environment...")
    
    # Generate training data (same seed for consistency)
    np.random.seed(42)
    n_periods = 60000  # Full dataset for 60K training
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Mock feature data
    nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    # Mock price data
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.02, n_periods)
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=trading_days
    )
    
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.015, n_periods)
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=trading_days
    )
    msft_data = np.random.randn(n_periods, 12).astype(np.float32)
    msft_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    # Create curriculum environment
    env = CurriculumTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,
        tc_bp=0.5,                      # Starting friction (will be adjusted by curriculum)
        trade_penalty_bp=0.7,           # Starting penalty (will be adjusted by curriculum)
        turnover_bp=2.0,                # Keep turnover penalty constant
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.15,     # Still allow 15% for training exploration
        evaluation_drawdown_pct=0.02,   # Strict 2% for evaluation
        is_training=True,
        log_trades=False
    )
    
    # Wrap with Monitor
    env = Monitor(env)
    
    logger.info("✅ CURRICULUM Environment created")
    logger.info(f"📊 Training periods: {n_periods:,}")
    logger.info(f"📊 Initial Capital: ${env.env.initial_capital:,.2f}")
    logger.info("📚 CURRICULUM SPECIFICATIONS:")
    logger.info("   🚀 Phase 1 (0-10K): Warm-up at 0.5bp/0.7bp")
    logger.info("   📈 Phase 2 (10K-40K): Anneal to 1.0bp/2.0bp")
    logger.info("   🛡️ Phase 3 (40K-60K): Overshoot to 1.2bp/2.5bp")
    logger.info("   🎲 Domain randomization: ±0.2bp TC, ±0.4bp penalty per step")
    logger.info("   🎲 Episode randomization: ±0.1bp TC, ±0.2bp penalty per episode")
    
    return env

def create_curriculum_model(env):
    """Create RecurrentPPO model for curriculum training"""
    logger.info("🧠 Creating CURRICULUM RecurrentPPO model...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=1e-4,         # Stable learning rate
        n_steps=2048,               # Standard rollout length
        batch_size=64,              # Memory efficient
        n_epochs=10,                # Standard epochs
        gamma=0.99,                 # Standard discount
        gae_lambda=0.95,            # Standard GAE
        clip_range=0.1,             # Conservative clipping
        ent_coef=0.01,              # Standard entropy
        vf_coef=0.5,                # Standard value function
        max_grad_norm=0.5,          # Gradient clipping
        tensorboard_log="logs/",
        verbose=1,
        seed=42,                    # Reproducible
        device="auto"
    )
    
    logger.info("✅ CURRICULUM RecurrentPPO model created")
    return model

def main():
    """Main curriculum training function"""
    logger.info("📚 FRICTION CURRICULUM TRAINING - 60K STEPS")
    logger.info("🎯 Progressive friction: 0.5bp → 1.2bp + domain randomization")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create curriculum training environment
        train_env = create_curriculum_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Step 2: Add VecNormalize
        logger.info("🔧 Adding VecNormalize for stability...")
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations
            norm_reward=True,    # Normalize rewards
            clip_reward=10.0,    # Clip extreme rewards
            gamma=0.99
        )
        
        # Step 3: Create curriculum model
        model = create_curriculum_model(vec_env)
        
        # Step 4: Setup callbacks
        curriculum_callback = FrictionCurriculumCallback(
            env=train_env,  # Pass the base environment for friction control
            total_steps=60000,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix="dual_ticker_curriculum"
        )
        
        # Step 5: Start curriculum training
        logger.info("🎯 Starting CURRICULUM 60K training...")
        logger.info("📚 PROGRESSIVE FRICTION CURRICULUM:")
        logger.info("   🚀 0-10K: Easy exploration (0.5bp/0.7bp)")
        logger.info("   📈 10K-40K: Linear annealing to production (1.0bp/2.0bp)")
        logger.info("   🛡️ 40K-60K: Overshoot for safety margin (1.2bp/2.5bp)")
        logger.info("   🎲 Continuous domain randomization")
        logger.info("💾 Model checkpoints every 10,000 steps")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        # Train with curriculum callback
        model.learn(
            total_timesteps=60000,
            callback=[curriculum_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 6: Save final model
        model_path = "models/dual_ticker_curriculum_60k_final.zip"
        vecnorm_path = "models/dual_ticker_curriculum_60k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 FRICTION CURRICULUM TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("📚 CURRICULUM SPECIFICATIONS SUCCESSFULLY APPLIED:")
        logger.info("   🚀 Progressive friction: 0.5bp → 1.2bp")
        logger.info("   🎲 Domain randomization: Per-step + per-episode")
        logger.info("   🛡️ Safety margin: 20% overshoot beyond production")
        logger.info("   🎯 Training steps: 60,000 (20% more than standard)")
        logger.info("✅ MODEL SHOULD NOW GENERALIZE TO REALISTIC TRADING COSTS")
        
        # Final portfolio check
        try:
            portfolio = train_env.env.portfolio_value
            peak = train_env.env.peak_portfolio_value
            drawdown = ((peak - portfolio) / peak) if peak > 0 else 0.0
            total_return = (portfolio - train_env.env.initial_capital) / train_env.env.initial_capital
            
            logger.info("📊 FINAL CURRICULUM PORTFOLIO:")
            logger.info(f"   💰 Final Portfolio: ${portfolio:,.2f}")
            logger.info(f"   🏔️ Peak Portfolio: ${peak:,.2f}")
            logger.info(f"   📉 Final Drawdown: {drawdown:.2%}")
            logger.info(f"   📈 Total Return: {total_return:+.2%}")
            
        except Exception as e:
            logger.warning(f"Could not access final portfolio values: {e}")
        
    except Exception as e:
        logger.error(f"❌ Curriculum training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()