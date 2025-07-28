#!/usr/bin/env python3
"""
📊 SIMPLE 50K TRAINING WITH PORTFOLIO MONITORING
Streamlined version that just trains and shows portfolio values
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

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_simple_environment():
    """Create simple training environment"""
    logger.info("📊 Creating SIMPLE training environment...")
    
    # Generate mock dataset
    n_periods = 60000
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
    
    # Create environment with all optimizations
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,     # Starting capital
        tc_bp=1.0,                  # REDUCED friction
        trade_penalty_bp=2.0,       # REDUCED friction
        turnover_bp=2.0,            # Turnover penalty
        hold_action_bonus=0.01,     # Hold bonus
        action_repeat_penalty=0.002, # Action change penalty
        high_water_mark_reward=0.001, # High-water mark reward
        daily_trade_limit=50,       # Daily trade limit
        reward_scaling=0.1,         # Reward scaling
        training_drawdown_pct=0.07, # 7% training drawdown
        evaluation_drawdown_pct=0.02, # 2% evaluation drawdown
        is_training=True,           # Training mode
        log_trades=False
    )
    
    logger.info(f"✅ SIMPLE Environment created")
    logger.info(f"📊 Initial Capital: ${env.initial_capital:,.2f}")
    logger.info(f"📊 Current Portfolio: ${env.portfolio_value:,.2f}")
    logger.info("📈 ALL OPTIMIZATIONS APPLIED:")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (REDUCED)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (REDUCED)") 
    logger.info(f"   💰 Turnover Penalty: {2.0} bp (KEPT)")
    logger.info(f"   🏆 High-Water Mark Reward: {0.001}")
    logger.info(f"   🎓 Training Drawdown: {7.0}% (exploration)")
    
    return env

def main():
    """Main simple training function"""
    logger.info("📊 SIMPLE 50K PORTFOLIO MONITORING TRAINING STARTED")
    
    start_time = datetime.now()
    
    try:
        # Create environment
        train_env = create_simple_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        # Log initial portfolio
        logger.info("📊 INITIAL PORTFOLIO STATE:")
        logger.info(f"   💰 Starting Capital: ${train_env.initial_capital:,.2f}")
        logger.info(f"   📊 Current Portfolio: ${train_env.portfolio_value:,.2f}")
        logger.info(f"   🏔️ Peak Portfolio: ${train_env.peak_portfolio_value:,.2f}")
        
        # Create model
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=0.00015,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/",
            verbose=1,
            seed=42,
            device="auto"
        )
        
        # Simple checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix="dual_ticker_simple"
        )
        
        # Train directly - no chunks, no evaluation
        logger.info("🎯 Starting SIMPLE 50K training...")
        logger.info("💾 Model checkpoints every 10,000 steps")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=50000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Final portfolio check
        logger.info("📊 FINAL PORTFOLIO CHECK:")
        try:
            final_portfolio = train_env.portfolio_value
            final_peak = train_env.peak_portfolio_value
            final_drawdown = ((final_peak - final_portfolio) / final_peak) if final_peak > 0 else 0.0
            total_return = (final_portfolio - train_env.initial_capital) / train_env.initial_capital
            
            logger.info(f"   💰 Final Portfolio: ${final_portfolio:,.2f}")
            logger.info(f"   🏔️ Peak Portfolio: ${final_peak:,.2f}")
            logger.info(f"   📉 Final Drawdown: {final_drawdown:.2%}")
            logger.info(f"   📈 Total Return: {total_return:+.2%}")
            
        except Exception as e:
            logger.warning(f"Could not access final portfolio values: {e}")
        
        # Save model
        model_path = "models/dual_ticker_simple_50k_final.zip"
        vecnorm_path = "models/dual_ticker_simple_50k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 SIMPLE PORTFOLIO MONITORING TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("✅ SIMPLE TRAINING SYSTEM SUCCESSFULLY COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ SIMPLE Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()