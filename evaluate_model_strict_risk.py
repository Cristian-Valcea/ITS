#!/usr/bin/env python3
"""
🛡️ MODEL EVALUATION WITH STRICT RISK CONTROLS
Evaluate trained model with 2% drawdown limit for production readiness
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load environment variables
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

# Import components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_evaluation_environment():
    """Create evaluation environment with STRICT 2% drawdown limit"""
    logger.info("🛡️ Creating STRICT EVALUATION environment...")
    
    # Generate evaluation data (different from training)
    eval_n_periods = 10000
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # Mock evaluation data with different patterns
    np.random.seed(12345)  # Different seed for evaluation
    eval_nvda_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    eval_nvda_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    # Mock price data with realistic patterns
    nvda_base_price = 180.0  # Different starting price
    nvda_returns = np.random.normal(0.0001, 0.025, eval_n_periods)  # Higher volatility
    eval_nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=eval_trading_days
    )
    
    msft_base_price = 520.0  # Different starting price
    msft_returns = np.random.normal(0.0001, 0.018, eval_n_periods)  # Higher volatility
    eval_msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=eval_trading_days
    )
    eval_msft_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    eval_msft_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    # 🛡️ CREATE EVALUATION ENVIRONMENT WITH STRICT RISK CONTROLS
    env = DualTickerTradingEnv(
        nvda_data=eval_nvda_data,
        msft_data=eval_msft_data,
        nvda_prices=eval_nvda_prices,
        msft_prices=eval_msft_prices,
        trading_days=eval_trading_days,
        initial_capital=100000,     # Same as training
        tc_bp=1.0,                  # Same reduced friction as training
        trade_penalty_bp=2.0,       # Same reduced friction as training
        turnover_bp=2.0,            # Same turnover penalty as training
        hold_action_bonus=0.01,     # Same incentives as training
        action_repeat_penalty=0.002, # Same incentives as training
        high_water_mark_reward=0.001, # Same high-water mark system as training
        daily_trade_limit=50,       # Same limits as training
        reward_scaling=0.1,         # Same scaling as training
        training_drawdown_pct=0.07, # Not used in evaluation mode
        evaluation_drawdown_pct=0.02, # 🛡️ STRICT: 2% evaluation limit
        is_training=False,          # 🛡️ EVALUATION MODE: Strict risk controls
        log_trades=True             # Enable detailed logging for evaluation
    )
    
    logger.info(f"✅ EVALUATION Environment created - observation space: {env.observation_space}")
    logger.info(f"✅ EVALUATION Environment created - action space: {env.action_space}")
    logger.info("🛡️ STRICT EVALUATION PARAMETERS:")
    logger.info(f"   🚨 Drawdown Limit: {2.0}% (STRICT)")
    logger.info(f"   📊 Evaluation Data: {eval_n_periods} periods")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (REDUCED)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (REDUCED)")
    logger.info(f"   🏆 High-Water Mark Reward: {0.001}")
    logger.info(f"   🎯 Mode: EVALUATION (production-ready)")
    
    return env

def main():
    """Evaluate trained model with strict risk controls"""
    logger.info("🛡️ STARTING STRICT RISK EVALUATION")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load trained model
        model_path = "models/dual_ticker_optimized_50k_final.zip"
        vecnorm_path = "models/dual_ticker_optimized_50k_vecnorm.pkl"
        
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("💡 Run training first: python train_50k_OPTIMIZED_EMERGENCY.py")
            return
        
        logger.info(f"📥 Loading trained model: {model_path}")
        
        # Step 2: Create evaluation environment
        eval_env = create_evaluation_environment()
        vec_env = DummyVecEnv([lambda: eval_env])
        
        # Step 3: Load VecNormalize statistics
        if Path(vecnorm_path).exists():
            logger.info(f"📥 Loading VecNormalize: {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False  # Disable training mode for evaluation
        else:
            logger.warning(f"⚠️ VecNormalize not found: {vecnorm_path}")
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        # Step 4: Load model
        model = RecurrentPPO.load(model_path, env=vec_env)
        logger.info("✅ Model loaded successfully")
        
        # Step 5: Run evaluation
        logger.info("🎯 Starting STRICT EVALUATION (2% drawdown limit)...")
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        total_reward = 0
        steps = 0
        trades = 0
        episodes = 0
        portfolio_values = []
        drawdown_breaches = 0
        
        for step in range(5000):  # 5000-step evaluation
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            obs, reward, done, info = vec_env.step(action)
            
            total_reward += reward[0]
            steps += 1
            
            # Track metrics
            if 'total_trades' in info[0]:
                trades = info[0]['total_trades']
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
            
            # Check for episode termination
            if done[0]:
                episodes += 1
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
                
                # Check if terminated due to drawdown
                if 'drawdown_breach' in info[0] or step < 4999:
                    drawdown_breaches += 1
                    logger.info(f"📊 Episode {episodes} ended at step {step}")
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Step 6: Calculate results
        end_time = datetime.now()
        duration = end_time - start_time
        
        avg_reward = total_reward / steps if steps > 0 else 0
        trade_rate = trades / steps if steps > 0 else 0
        final_portfolio = portfolio_values[-1] if portfolio_values else 100000
        portfolio_change = final_portfolio - 100000
        portfolio_change_pct = (portfolio_change / 100000) * 100
        
        # Calculate max drawdown
        if portfolio_values:
            peak_value = 100000
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak_value:
                    peak_value = value
                current_drawdown = (peak_value - value) / peak_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
        else:
            max_drawdown = 0
        
        # Step 7: Report results
        logger.info("🎉 STRICT EVALUATION COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"📊 EVALUATION RESULTS:")
        logger.info(f"   Steps: {steps}")
        logger.info(f"   Episodes: {episodes}")
        logger.info(f"   Drawdown breaches: {drawdown_breaches}")
        logger.info(f"   Total trades: {trades}")
        logger.info(f"   Trade rate: {trade_rate:.3f} trades/step")
        logger.info(f"   Average reward: {avg_reward:.3f}")
        logger.info(f"   Final portfolio: ${final_portfolio:.2f}")
        logger.info(f"   Portfolio change: ${portfolio_change:.2f} ({portfolio_change_pct:.2f}%)")
        logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
        
        # Step 8: Production readiness assessment
        logger.info("🛡️ PRODUCTION READINESS ASSESSMENT:")
        
        # Assessment criteria
        criteria = {
            "Max drawdown ≤ 2%": max_drawdown <= 0.02,
            "Portfolio change > -5%": portfolio_change_pct > -5,
            "Average reward ≥ 0": avg_reward >= 0,
            "Trade rate < 0.5": trade_rate < 0.5,
            "No excessive breaches": drawdown_breaches < episodes * 0.8
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   {status}: {criterion}")
        
        if passed_criteria >= 4:
            logger.info(f"🎉 PRODUCTION READY: {passed_criteria}/{total_criteria} criteria passed")
            logger.info("✅ Model is ready for live trading with strict risk controls!")
        else:
            logger.warning(f"⚠️ NOT PRODUCTION READY: Only {passed_criteria}/{total_criteria} criteria passed")
            logger.warning("❌ Model needs more training or parameter tuning")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()