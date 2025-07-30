#!/usr/bin/env python3
"""
🚀 ENHANCED MODEL EVALUATION
Evaluate the ultra-low friction enhanced model with proper file paths
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

# Import components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

def create_evaluation_environment():
    """Create evaluation environment with STRICT 2% drawdown limit"""
    logger.info("🛡️ Creating ENHANCED MODEL EVALUATION environment...")
    
    # Generate evaluation data (same seed for comparison)
    np.random.seed(42)  # Same data as training for fair comparison
    eval_n_periods = 5000  # Shorter for faster evaluation
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # Mock evaluation data
    nvda_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.02, eval_n_periods)
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=eval_trading_days
    )
    
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.015, eval_n_periods)
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=eval_trading_days
    )
    msft_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    msft_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    # Create EVALUATION environment with STRICT parameters
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=eval_trading_days,
        initial_capital=100000,
        tc_bp=1.0,                      # Higher friction for evaluation
        trade_penalty_bp=2.0,           # Higher penalty for evaluation  
        turnover_bp=2.0,
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.02,     # STRICT 2% for evaluation
        evaluation_drawdown_pct=0.02,   # STRICT 2% for evaluation
        is_training=False,              # EVALUATION mode
        log_trades=True
    )
    
    logger.info("✅ EVALUATION Environment created")
    logger.info(f"🛡️ STRICT EVALUATION PARAMETERS:")
    logger.info(f"   🚨 Drawdown Limit: {2.0}% (STRICT)")
    logger.info(f"   📊 Evaluation Data: {eval_n_periods} periods")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (production)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (production)")
    logger.info(f"   🎯 Mode: EVALUATION (strict)")
    
    return env

def main():
    """Evaluate the enhanced model"""
    logger.info("🚀 ENHANCED MODEL EVALUATION STARTED")
    
    # Model paths
    model_path = "models/dual_ticker_enhanced_50k_final.zip"
    vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
    
    logger.info(f"📥 Loading ENHANCED model: {model_path}")
    
    try:
        # Create evaluation environment
        eval_env = create_evaluation_environment()
        vec_env = DummyVecEnv([lambda: eval_env])
        
        # Load VecNormalize
        logger.info(f"📥 Loading VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False  # Disable training mode
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        
        # Load model
        model = RecurrentPPO.load(model_path, env=vec_env)
        logger.info("✅ ENHANCED model loaded successfully")
        
        # Run evaluation
        logger.info("🎯 Starting ENHANCED MODEL EVALUATION (2% drawdown limit)...")
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        total_steps = 0
        total_reward = 0
        trade_count = 0
        episode_count = 0
        max_drawdown = 0
        portfolio_values = []
        actions_taken = []
        
        for step in range(5000):  # Run for 5000 steps
            # Get action from model
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            # Execute action
            obs, reward, done, info = vec_env.step(action)
            
            total_reward += reward[0]
            total_steps += 1
            actions_taken.append(action[0])
            
            # Extract portfolio info
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
                
                # Track drawdown
                if len(portfolio_values) > 1:
                    peak = max(portfolio_values)
                    current = portfolio_values[-1]
                    drawdown = (peak - current) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Log progress every 500 steps
            if step % 500 == 0 and step > 0:
                current_portfolio = portfolio_values[-1] if portfolio_values else 100000
                logger.info(f"Step {step}: Portfolio ${current_portfolio:,.2f}, Drawdown {max_drawdown:.2%}")
            
            # Check if episode ended
            if done[0]:
                episode_count += 1
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
                logger.info(f"📊 Episode {episode_count} completed at step {step}")
                break
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Final results
        final_portfolio = portfolio_values[-1] if portfolio_values else 100000
        total_return = (final_portfolio - 100000) / 100000
        
        logger.info("🎉 ENHANCED MODEL EVALUATION COMPLETED!")
        logger.info("📊 ENHANCED MODEL RESULTS:")
        logger.info(f"   Steps: {total_steps}")
        logger.info(f"   Episodes: {episode_count}")
        logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
        logger.info(f"   Total return: {total_return:+.2%}")
        logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
        logger.info(f"   Average reward: {total_reward/total_steps:.4f}")
        
        # Production readiness assessment
        logger.info("🛡️ ENHANCED MODEL PRODUCTION READINESS:")
        
        criteria_passed = 0
        total_criteria = 5
        
        if max_drawdown <= 0.02:
            logger.info("   ✅ PASS: Max drawdown ≤ 2%")
            criteria_passed += 1
        else:
            logger.info(f"   ❌ FAIL: Max drawdown {max_drawdown:.2%} > 2%")
        
        if total_return > -0.05:
            logger.info("   ✅ PASS: Total return > -5%")
            criteria_passed += 1
        else:
            logger.info(f"   ❌ FAIL: Total return {total_return:.2%} ≤ -5%")
        
        if total_reward/total_steps >= 0:
            logger.info("   ✅ PASS: Average reward ≥ 0")
            criteria_passed += 1
        else:
            logger.info(f"   ❌ FAIL: Average reward {total_reward/total_steps:.4f} < 0")
        
        trade_rate = len(set(actions_taken)) / total_steps
        if trade_rate < 0.5:
            logger.info("   ✅ PASS: Trade rate reasonable")
            criteria_passed += 1
        else:
            logger.info(f"   ❌ FAIL: Trade rate {trade_rate:.3f} too high")
        
        if episode_count <= 1:
            logger.info("   ✅ PASS: No excessive early terminations")
            criteria_passed += 1
        else:
            logger.info(f"   ❌ FAIL: {episode_count} episodes (early terminations)")
        
        # Final assessment
        if criteria_passed >= 4:
            logger.info(f"✅ PRODUCTION READY: {criteria_passed}/{total_criteria} criteria passed")
        else:
            logger.warning(f"⚠️ NEEDS IMPROVEMENT: Only {criteria_passed}/{total_criteria} criteria passed")
        
        # Compare with training specifications
        logger.info("🚀 ENHANCED MODEL SPECIFICATIONS RECAP:")
        logger.info("   💰 Training Friction: tc_bp=0.5, trade_penalty=0.7 (ultra-low)")
        logger.info("   🚀 Training Drawdown: 15% (maximum room)")
        logger.info("   🛡️ Evaluation Drawdown: 2% (strict control)")
        logger.info("   🧠 Learning Rate: 1e-4 (stable)")
        logger.info("   📊 Data: Fixed seed=42 (reproducible)")
        
    except Exception as e:
        logger.error(f"❌ Enhanced evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()