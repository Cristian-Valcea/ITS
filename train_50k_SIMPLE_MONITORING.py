#!/usr/bin/env python3
"""
📊 SIMPLE 50K TRAINING WITH PORTFOLIO MONITORING
Simplified version that shows portfolio values without complex callbacks
"""

import os
import sys  
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env_file()

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
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

def create_training_environment():
    """Create training environment with all optimizations"""
    logger.info("📊 Creating SIMPLE MONITORING training environment...")
    
    # Generate LARGE mock dataset for 50K training
    big_n_periods = 60000  # Much larger dataset
    big_trading_days = pd.date_range('2024-01-01', periods=big_n_periods, freq='1min')
    
    # Mock feature data with some realistic patterns
    big_nvda_data = np.random.randn(big_n_periods, 12).astype(np.float32)
    big_nvda_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
    
    # Mock price data with trend and volatility
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.02, big_n_periods)  # Realistic returns
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=big_trading_days
    )
    
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.015, big_n_periods)  # Realistic returns
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=big_trading_days
    )
    big_msft_data = np.random.randn(big_n_periods, 12).astype(np.float32)
    big_msft_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
    
    # 📊 CREATE TRAINING ENVIRONMENT WITH ALL OPTIMIZATIONS
    env = DualTickerTradingEnv(
        nvda_data=big_nvda_data,
        msft_data=big_msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=big_trading_days,
        initial_capital=100000,     # Starting capital
        tc_bp=1.0,                  # 📈 REDUCED: Lower transaction costs (10 bp)
        trade_penalty_bp=2.0,       # 📈 REDUCED: Lower trade penalty (20 bp)
        turnover_bp=2.0,            # 🔧 KEPT: Turnover penalty (20 bp)
        hold_action_bonus=0.01,     # 🔧 OPTIMIZED: Bonus for holding positions
        action_repeat_penalty=0.002, # 🔧 OPTIMIZED: Penalty for action changes
        high_water_mark_reward=0.001, # 🏆 NEW: High-water mark reward system
        daily_trade_limit=50,       # 🔧 OPTIMIZED: Daily trade limit
        reward_scaling=0.1,         # 🔧 OPTIMIZED: Better reward scaling
        training_drawdown_pct=0.07, # 🎓 NEW: 7% drawdown for training exploration
        evaluation_drawdown_pct=0.02, # 🛡️ NEW: 2% drawdown for evaluation/production
        is_training=True,           # 🎓 TRAINING MODE: Allow exploration
        log_trades=False            # Reduce logging spam
    )
    
    logger.info(f"✅ SIMPLE MONITORING Environment created")
    logger.info(f"📊 Initial Capital: ${env.initial_capital:,.2f}")
    logger.info(f"📊 Current Portfolio: ${env.portfolio_value:,.2f}")
    logger.info(f"📊 Peak Portfolio: ${env.peak_portfolio_value:,.2f}")
    logger.info("📈 ALL OPTIMIZATIONS APPLIED:")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (REDUCED)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (REDUCED)") 
    logger.info(f"   💰 Turnover Penalty: {2.0} bp (KEPT)")
    logger.info(f"   🏆 High-Water Mark Reward: {0.001}")
    logger.info(f"   🎓 Training Drawdown: {7.0}% (exploration)")
    logger.info(f"   🛡️ Evaluation Drawdown: {2.0}% (strict)")
    
    return env

def create_model(env):
    """Create the RecurrentPPO model with optimized parameters"""
    logger.info("🧠 Creating SIMPLE MONITORING RecurrentPPO model...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=0.00015,      # Slightly lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,             # Tighter clipping for stability
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/",
        verbose=1,
        seed=42,
        device="auto"
    )
    
    logger.info("✅ SIMPLE MONITORING RecurrentPPO model created")
    return model

def manual_portfolio_check(env, step):
    """Manually check portfolio values"""
    try:
        portfolio_value = env.portfolio_value
        peak_value = env.peak_portfolio_value
        drawdown = ((peak_value - portfolio_value) / peak_value) if peak_value > 0 else 0.0
        
        logger.info(f"📊 STEP {step:,}: Portfolio=${portfolio_value:,.2f}, Peak=${peak_value:,.2f}, DD={drawdown:.2%}")
        return portfolio_value, peak_value, drawdown
    except Exception as e:
        logger.warning(f"Could not check portfolio at step {step}: {e}")
        return None, None, None

def main():
    """Main training function with simple portfolio monitoring"""
    logger.info("📊 SIMPLE 50K PORTFOLIO MONITORING TRAINING STARTED")
    logger.info("🎯 Goal: Monitor portfolio values with simple approach")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create training environment
        train_env = create_training_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Log initial portfolio state
        logger.info("📊 INITIAL PORTFOLIO STATE:")
        logger.info(f"   💰 Starting Capital: ${train_env.initial_capital:,.2f}")
        logger.info(f"   📊 Current Portfolio: ${train_env.portfolio_value:,.2f}")
        logger.info(f"   🏔️ Peak Portfolio: ${train_env.peak_portfolio_value:,.2f}")
        
        # Step 2: Add VecNormalize for reward normalization
        logger.info("🔧 Adding VecNormalize for reward stability...")
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations (features are already scaled)
            norm_reward=True,    # 🔧 OPTIMIZED: Normalize rewards for better learning
            clip_reward=10.0,    # 🔧 OPTIMIZED: Clip extreme rewards
            gamma=0.99
        )
        
        # Step 3: Create model
        model = create_model(vec_env)
        
        # Step 4: Setup simple checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix="dual_ticker_simple_monitored"
        )
        
        # Step 5: Train with manual monitoring
        logger.info("🎯 Starting SIMPLE 50K training...")
        logger.info("📊 Portfolio will be checked manually during training")
        logger.info("💾 Model checkpoints every 10,000 steps")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        # Train in chunks to allow manual monitoring
        steps_per_chunk = 10000
        total_steps = 50000
        
        for chunk in range(0, total_steps, steps_per_chunk):
            chunk_steps = min(steps_per_chunk, total_steps - chunk)
            
            logger.info(f"🎯 Training chunk: steps {chunk} to {chunk + chunk_steps}")
            
            # Check portfolio before chunk
            manual_portfolio_check(train_env, chunk)
            
            # Train this chunk
            model.learn(
                total_timesteps=chunk_steps,
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=(chunk == 0)  # Only reset on first chunk
            )
            
            # Check portfolio after chunk
            manual_portfolio_check(train_env, chunk + chunk_steps)
            
            logger.info(f"✅ Completed chunk: steps {chunk} to {chunk + chunk_steps}")
        
        # Step 6: Final portfolio check
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
        
        # Step 7: Save final model and VecNormalize
        model_path = "models/dual_ticker_simple_monitored_50k_final.zip"
        vecnorm_path = "models/dual_ticker_simple_monitored_50k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)  # Save VecNormalize statistics
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 SIMPLE PORTFOLIO MONITORING TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("✅ SIMPLE MONITORING SYSTEM SUCCESSFULLY COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ SIMPLE MONITORING Training failed: {e}")
        raise

if __name__ == "__main__":
    main()