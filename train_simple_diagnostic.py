#!/usr/bin/env python3
"""
🔧 Simple PPO Diagnostic (No LSTM)
Quick validation of friction optimizations without LSTM complexity
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

# Import training components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_simple_environment():
    """Create simple environment with mock data"""
    logger.info("🏗️ Creating simple diagnostic environment...")
    
    # Generate mock data
    n_periods = 5000
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Mock feature data
    nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
    msft_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Mock price data with some trend and volatility
    nvda_prices = pd.Series(
        [170.0 + i * 0.01 + np.random.normal(0, 0.5) for i in range(n_periods)], 
        index=trading_days
    )
    msft_prices = pd.Series(
        [510.0 + i * 0.005 + np.random.normal(0, 1.0) for i in range(n_periods)], 
        index=trading_days
    )
    
    # Create environment with AGGRESSIVE friction parameters
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,
        tc_bp=5.0,              # VERY HIGH transaction costs
        trade_penalty_bp=10.0,  # VERY HIGH trade penalty  
        turnover_bp=2.0,        # HIGHER turnover penalty
        hold_action_bonus=0.01, # Bonus for holding
        action_repeat_penalty=0.002, # Penalty for action changes
        reward_scaling=0.1,
        log_trades=False  # Disable trade logging for cleaner output
    )
    
    return env

def main():
    """Simple diagnostic training"""
    logger.info("🔧 Starting Simple PPO Diagnostic")
    logger.info("💰 EXTREME Friction: tc_bp=5.0, trade_penalty_bp=10.0, turnover_bp=2.0")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create environment
        train_env = create_simple_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Step 2: Add VecNormalize
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0
        )
        
        # Step 3: Create simple PPO model (no LSTM)
        logger.info("🤖 Creating simple PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/",
            verbose=1,
            seed=42
        )
        
        # Step 4: Train for 3000 steps
        logger.info("🎯 Starting 3000 diagnostic steps...")
        model.learn(
            total_timesteps=3000,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 5: Quick evaluation
        logger.info("🔍 Running evaluation...")
        obs = vec_env.reset()
        total_reward = 0
        steps = 0
        trades = 0
        portfolio_values = []
        
        for _ in range(200):  # 200-step evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Track metrics
            if 'total_trades' in info[0]:
                trades = info[0]['total_trades']
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
            
            if done[0]:
                break
        
        # Calculate results
        end_time = datetime.now()
        duration = end_time - start_time
        
        trade_rate = trades / steps if steps > 0 else 0
        avg_reward = total_reward / steps if steps > 0 else 0
        final_portfolio = portfolio_values[-1] if portfolio_values else 100000
        portfolio_change = final_portfolio - 100000
        
        logger.info("✅ Diagnostic completed!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"📊 Results:")
        logger.info(f"   Steps: {steps}")
        logger.info(f"   Total trades: {trades}")
        logger.info(f"   Trade rate: {trade_rate:.3f} trades/step")
        logger.info(f"   Avg reward: {avg_reward:.3f}")
        logger.info(f"   Portfolio: ${final_portfolio:.2f} (change: ${portfolio_change:.2f})")
        
        # Check diagnostic gates
        logger.info("🚦 Diagnostic Gates:")
        
        # Gate 1: Trade frequency
        trades_per_1000 = trade_rate * 1000
        if trades_per_1000 < 100:
            logger.info(f"✅ PASS: Trade frequency {trades_per_1000:.1f}/1000 steps (target: <100)")
        else:
            logger.warning(f"❌ FAIL: Trade frequency {trades_per_1000:.1f}/1000 steps (target: <100)")
        
        # Gate 2: Average reward
        if avg_reward >= 0:
            logger.info(f"✅ PASS: Average reward {avg_reward:.3f} (target: ≥0)")
        else:
            logger.warning(f"❌ FAIL: Average reward {avg_reward:.3f} (target: ≥0)")
        
        # Gate 3: Portfolio stability
        portfolio_change_pct = (portfolio_change / 100000) * 100
        if portfolio_change_pct > -5:  # Less than 5% loss
            logger.info(f"✅ PASS: Portfolio change {portfolio_change_pct:.2f}% (target: >-5%)")
        else:
            logger.warning(f"❌ FAIL: Portfolio change {portfolio_change_pct:.2f}% (target: >-5%)")
        
        # Summary
        passes = sum([
            trades_per_1000 < 100,
            avg_reward >= 0,
            portfolio_change_pct > -5
        ])
        
        if passes >= 2:
            logger.info(f"🎉 DIAGNOSTIC PASSED: {passes}/3 gates passed")
            logger.info("✅ Friction optimizations are working!")
        else:
            logger.warning(f"⚠️ DIAGNOSTIC FAILED: Only {passes}/3 gates passed")
            logger.warning("❌ Need more aggressive friction parameters")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        raise

if __name__ == "__main__":
    main()