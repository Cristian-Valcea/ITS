#!/usr/bin/env python3
"""
📊 QUICK PORTFOLIO MONITORING DEMO
Shows how portfolio values change during training
"""

import os
import sys  
import logging
import numpy as np
import pandas as pd
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

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_demo_environment():
    """Create a small demo environment"""
    logger.info("📊 Creating DEMO environment for portfolio monitoring...")
    
    # Small dataset for quick demo
    n_periods = 5000
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Mock feature data
    nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    # Mock price data with some trend
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0002, 0.02, n_periods)  # Slight upward trend
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
    
    return env

def demo_portfolio_monitoring():
    """Demonstrate portfolio monitoring during training"""
    logger.info("🚀 PORTFOLIO MONITORING DEMO STARTED")
    
    # Create environment
    env = create_demo_environment()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    # Log initial state
    logger.info("📊 INITIAL STATE:")
    logger.info(f"   💰 Starting Capital: ${env.initial_capital:,.2f}")
    logger.info(f"   📊 Current Portfolio: ${env.portfolio_value:,.2f}")
    logger.info(f"   🏔️ Peak Portfolio: ${env.peak_portfolio_value:,.2f}")
    
    # Create model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=0.0003,  # Higher LR for faster demo
        n_steps=512,           # Smaller steps for demo
        batch_size=32,         # Smaller batch for demo
        n_epochs=5,            # Fewer epochs for demo
        verbose=1,
        device="auto"
    )
    
    # Train in small chunks with monitoring
    total_steps = 5000
    chunk_size = 1000
    
    for i in range(0, total_steps, chunk_size):
        chunk_steps = min(chunk_size, total_steps - i)
        
        logger.info(f"🎯 Training steps {i} to {i + chunk_steps}")
        
        # Check portfolio before training
        try:
            portfolio = env.portfolio_value
            peak = env.peak_portfolio_value
            drawdown = ((peak - portfolio) / peak) if peak > 0 else 0.0
            total_return = (portfolio - env.initial_capital) / env.initial_capital
            
            logger.info(f"📊 BEFORE CHUNK {i//chunk_size + 1}:")
            logger.info(f"   💰 Portfolio: ${portfolio:,.2f}")
            logger.info(f"   🏔️ Peak: ${peak:,.2f}")
            logger.info(f"   📉 Drawdown: {drawdown:.2%}")
            logger.info(f"   📈 Return: {total_return:+.2%}")
            
        except Exception as e:
            logger.warning(f"Could not check portfolio: {e}")
        
        # Train this chunk
        model.learn(
            total_timesteps=chunk_steps,
            progress_bar=True,
            reset_num_timesteps=(i == 0)
        )
        
        # Check portfolio after training
        try:
            portfolio = env.portfolio_value
            peak = env.peak_portfolio_value
            drawdown = ((peak - portfolio) / peak) if peak > 0 else 0.0
            total_return = (portfolio - env.initial_capital) / env.initial_capital
            
            logger.info(f"📊 AFTER CHUNK {i//chunk_size + 1}:")
            logger.info(f"   💰 Portfolio: ${portfolio:,.2f}")
            logger.info(f"   🏔️ Peak: ${peak:,.2f}")
            logger.info(f"   📉 Drawdown: {drawdown:.2%}")
            logger.info(f"   📈 Return: {total_return:+.2%}")
            logger.info("   " + "="*50)
            
        except Exception as e:
            logger.warning(f"Could not check portfolio: {e}")
    
    # Final summary
    logger.info("🎉 PORTFOLIO MONITORING DEMO COMPLETED!")
    logger.info("✅ This demonstrates how to monitor portfolio values during training")
    logger.info("📈 You can see how the portfolio changes as the agent learns")

if __name__ == "__main__":
    demo_portfolio_monitoring()