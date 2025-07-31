#!/usr/bin/env python3
"""
🚀 200K Dual-Ticker Training Launch Script
Transfer learning from dual_ticker_enhanced_50k_final.zip with optimizations
"""

import os
import sys
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from tensorboardX import SummaryWriter

from gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from training.enhanced_callbacks import (
    TensorBoardCallback, 
    PnLTrackingCallback,
    ExecutiveDashboardCallback
)

# 🔥 RTX 3060 OPTIMIZATION
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/200k_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_base_model_and_vecnorm():
    """Load the 50K foundation model + VecNormalize stats"""
    logger = logging.getLogger(__name__)
    
    # Load the enhanced 50K model
    base_model_path = "models/dual_ticker_enhanced_50k_final.zip"
    vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnorm_path}")
    
    logger.info(f"Loading base model: {base_model_path}")
    logger.info(f"Loading VecNormalize stats: {vecnorm_path}")
    
    return base_model_path, vecnorm_path

def create_training_environment():
    """Create the dual-ticker training environment"""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating dual-ticker environment for 200K training")
    
    # Generate training data (consistent with 50K foundation)
    np.random.seed(42)  # Same seed as foundation model
    n_periods = 200000  # Extended dataset for 200K training
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # NVDA mock data
    nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)  # Price-like feature
    
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.02, n_periods)
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=trading_days
    )
    
    # MSFT mock data
    msft_data = np.random.randn(n_periods, 12).astype(np.float32)
    msft_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)  # Price-like feature
    
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.015, n_periods)
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=trading_days
    )
    
    logger.info(f"Generated {n_periods:,} periods of training data")
    logger.info(f"NVDA price range: ${nvda_prices.min():.2f} - ${nvda_prices.max():.2f}")
    logger.info(f"MSFT price range: ${msft_prices.min():.2f} - ${msft_prices.max():.2f}")
    
    def make_env():
        env = DualTickerTradingEnv(
            nvda_data=nvda_data,
            msft_data=msft_data,
            nvda_prices=nvda_prices,
            msft_prices=msft_prices,
            trading_days=trading_days,
            initial_capital=100000,
            tc_bp=1.0,                      # Enhanced transaction costs
            trade_penalty_bp=2.0,           # Penalty for overtrading
            turnover_bp=1.5,                # Turnover penalty
            hold_action_bonus=0.01,
            action_repeat_penalty=0.002,
            high_water_mark_reward=0.001,
            daily_trade_limit=50,
            reward_scaling=0.1,
            training_drawdown_pct=0.10,     # 10% training drawdown limit
            evaluation_drawdown_pct=0.02,   # 2% evaluation drawdown limit
            is_training=True
        )
        env = Monitor(env, filename=None)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    env_config = {
        'n_periods': n_periods,
        'symbols': ['NVDA', 'MSFT'],
        'initial_capital': 100000
    }
    
    return env, env_config

def setup_callbacks_and_monitoring():
    """Setup comprehensive monitoring and callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Checkpoint every 25K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="models/checkpoints/",
        name_prefix=f"dual_ticker_200k_{timestamp}"
    )
    
    # TensorBoard with P&L tracking
    tensorboard_callback = TensorBoardCallback(
        log_dir=f"runs/200k_training_{timestamp}",
        track_pnl=True,
        track_drawdown=True
    )
    
    # Executive dashboard metrics
    dashboard_callback = ExecutiveDashboardCallback(
        eval_freq=5000,
        log_path=f"reports/executive_dashboard_{timestamp}.json"
    )
    
    # P&L tracking for demo preparation
    pnl_callback = PnLTrackingCallback(
        target_pnl=1000,  # $1K target for demo
        max_drawdown=0.02  # 2% max drawdown
    )
    
    return [checkpoint_callback, tensorboard_callback, dashboard_callback, pnl_callback]

def main():
    """Launch 200K dual-ticker training with transfer learning"""
    logger = setup_logging()
    logger.info("🚀 Starting 200K Dual-Ticker Training Launch")
    
    try:
        # 1. Load base model and VecNormalize stats
        base_model_path, vecnorm_path = load_base_model_and_vecnorm()
        
        # 2. Create training environment
        env, env_config = create_training_environment()
        
        # 3. Load and preserve VecNormalize stats (CRITICAL!)
        logger.info("Loading VecNormalize stats - preserving reward scaling")
        with open(vecnorm_path, 'rb') as f:
            vecnorm_stats = pickle.load(f)
        
        # Apply VecNormalize with preserved stats
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True  # Enable training mode
        
        # 4. Load base model for transfer learning
        logger.info("Loading base model for transfer learning")
        model = RecurrentPPO.load(
            base_model_path,
            env=env,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 5. Enhanced training configuration for 200K (keep existing hyperparams)
        logger.info("Using existing model hyperparameters for transfer learning")
        
        logger.info(f"Training configuration:")
        logger.info(f"  Device: {model.device}")
        logger.info(f"  Learning rate: {model.learning_rate}")
        logger.info(f"  Batch size: {model.batch_size}")
        logger.info(f"  Steps per update: {model.n_steps}")
        logger.info(f"  Clip range: {model.clip_range}")
        logger.info(f"  Gamma: {model.gamma}")
        
        # 6. Setup callbacks and monitoring
        callbacks = setup_callbacks_and_monitoring()
        
        # 7. Launch training (200K steps ≈ 8-10 hours on RTX 3060)
        logger.info("🚀 LAUNCHING 200K TRAINING - Estimated 8-10 hours")
        logger.info("💡 TIP: Run in tmux for background execution")
        
        model.learn(
            total_timesteps=200000,
            callback=callbacks,
            tb_log_name="200k_dual_ticker_training",
            reset_num_timesteps=False  # Continue from 50K foundation
        )
        
        # 8. Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"models/dual_ticker_200k_final_{timestamp}.zip"
        final_vecnorm_path = f"models/dual_ticker_200k_vecnorm_{timestamp}.pkl"
        
        model.save(final_model_path)
        env.save(final_vecnorm_path)
        
        logger.info(f"✅ 200K Training Complete!")
        logger.info(f"📁 Final model: {final_model_path}")
        logger.info(f"📁 VecNormalize: {final_vecnorm_path}")
        
        # 9. Quick evaluation
        logger.info("🔍 Running quick evaluation...")
        mean_reward, std_reward = evaluate_model(model, env, n_eval_episodes=10)
        logger.info(f"📊 Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return final_model_path, final_vecnorm_path
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

def evaluate_model(model, env, n_eval_episodes=10):
    """Quick model evaluation"""
    from stable_baselines3.common.evaluation import evaluate_policy
    
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    
    return mean_reward, std_reward

if __name__ == "__main__":
    # Check for tmux recommendation
    if 'TMUX' not in os.environ:
        print("💡 RECOMMENDATION: Run in tmux for background execution")
        print("   tmux new-session -d -s training200k")
        print("   tmux send-keys -t training200k 'python launch_200k_dual_ticker_training.py' Enter")
        print()
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Start tmux session first.")
            sys.exit(0)
    
    main()