#!/usr/bin/env python3
"""
🔍 CURRICULUM MODEL STRICT EVALUATION
Test if friction curriculum solved the production friction problem
Target: ≥+1% return, ≤2% max DD = GREEN LIGHT for real data
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

def create_strict_evaluation_environment():
    """Create STRICT 2% DD evaluation environment - curriculum test"""
    logger.info("🔍 Creating STRICT EVALUATION for CURRICULUM MODEL")
    
    # Generate evaluation data (same seed for fair comparison)
    np.random.seed(42)  
    eval_n_periods = 25000  # 25K evaluation
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # Mock evaluation data (same generation as previous tests)
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
    
    # Create STRICT EVALUATION environment (production friction)
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=eval_trading_days,
        initial_capital=100000,
        tc_bp=1.0,                      # Production friction
        trade_penalty_bp=2.0,           # Production penalty  
        turnover_bp=2.0,
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.02,     # STRICT 2% limit
        evaluation_drawdown_pct=0.02,   # STRICT 2% limit
        is_training=False,              # EVALUATION mode
        log_trades=False
    )
    
    logger.info("✅ STRICT EVALUATION Environment created")
    logger.info(f"🛡️ CURRICULUM TEST PARAMETERS:")
    logger.info(f"   🚨 Drawdown Limit: {2.0}% (STRICT)")
    logger.info(f"   📊 Evaluation Steps: {eval_n_periods:,}")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (production)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (production)")
    logger.info(f"   🎯 GREEN LIGHT CRITERIA: ≥+1% return, ≤2% max DD")
    
    return env

def main():
    """Evaluate curriculum model with strict criteria"""
    logger.info("🔍 CURRICULUM MODEL STRICT EVALUATION")
    logger.info("🎯 Testing if friction curriculum solved production friction problem")
    start_time = datetime.now()
    
    # Model paths (curriculum model)
    model_path = "models/dual_ticker_curriculum_60k_final.zip"
    vecnorm_path = "models/dual_ticker_curriculum_60k_vecnorm.pkl"
    
    logger.info(f"📥 Loading CURRICULUM model: {model_path}")
    
    try:
        # Create strict evaluation environment
        eval_env = create_strict_evaluation_environment()
        vec_env = DummyVecEnv([lambda: eval_env])
        
        # Load VecNormalize
        logger.info(f"📥 Loading VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False  # Disable training mode
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        
        # Load model
        model = RecurrentPPO.load(model_path, env=vec_env)
        logger.info("✅ CURRICULUM model loaded successfully")
        
        # Run evaluation
        logger.info("🎯 Starting CURRICULUM MODEL EVALUATION...")
        logger.info("🔍 TESTING: Can curriculum model handle production friction?")
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        total_steps = 0
        total_reward = 0
        portfolio_values = []
        actions_taken = []
        
        # Tracking variables
        initial_capital = 100000
        peak_portfolio = initial_capital
        max_drawdown = 0
        
        logger.info("🚀 Running 25K curriculum evaluation...")
        
        for step in range(25000):
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
                portfolio_value = info[0]['portfolio_value']
                portfolio_values.append(portfolio_value)
                
                # Update peak and calculate drawdown
                if portfolio_value > peak_portfolio:
                    peak_portfolio = portfolio_value
                
                current_drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Log progress every 5K steps
            if step % 5000 == 0 and step > 0:
                current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
                current_return = (current_portfolio - initial_capital) / initial_capital
                logger.info(f"Step {step:5d}: Portfolio ${current_portfolio:8,.2f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
            
            # Check if episode ended
            if done[0]:
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
                logger.info(f"📊 Episode ended at step {step}")
                break
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Final calculations
        final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_portfolio - initial_capital) / initial_capital
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # CURRICULUM MODEL RESULTS
        logger.info("🏁 CURRICULUM EVALUATION COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info("📊 CURRICULUM MODEL RESULTS:")
        logger.info(f"   Steps completed: {total_steps:,}")
        logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
        logger.info(f"   Total return: {total_return:+.2%}")
        logger.info(f"   Peak portfolio: ${peak_portfolio:,.2f}")
        logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
        logger.info(f"   Average reward: {avg_reward:.4f}")
        
        # COMPARISON WITH PREVIOUS RESULTS
        logger.info("🔍 FRICTION SENSITIVITY COMPARISON:")
        logger.info("   Enhanced (0.5bp training): -2.04% return, 2.08% DD at 1.0bp eval")
        logger.info(f"   Curriculum (progressive): {total_return:+.2%} return, {max_drawdown:.2%} DD at 1.0bp eval")
        
        # GREEN LIGHT CRITERIA
        logger.info("🎯 GREEN LIGHT CRITERIA:")
        
        return_pass = total_return >= 0.01  # ≥+1% return
        drawdown_pass = max_drawdown <= 0.02  # ≤2% max DD
        stability_pass = total_steps >= 20000  # No early termination
        
        logger.info(f"   📈 Return ≥ +1.0%: {'✅ PASS' if return_pass else '❌ FAIL'} ({total_return:+.2%})")
        logger.info(f"   📉 Max DD ≤ 2.0%: {'✅ PASS' if drawdown_pass else '❌ FAIL'} ({max_drawdown:.2%})")
        logger.info(f"   🛡️ Stability: {'✅ PASS' if stability_pass else '❌ FAIL'} ({total_steps:,} steps)")
        
        # FINAL VERDICT
        all_pass = return_pass and drawdown_pass and stability_pass
        
        if all_pass:
            logger.info("🎉 ✅ GREEN LIGHT: CURRICULUM MODEL READY FOR REAL DATA")
            logger.info("🚀 FRICTION CURRICULUM SUCCESSFULLY SOLVED THE PROBLEM")
            logger.info("📋 Next: Download real data and prepare 200K continuation")
        else:
            logger.warning("⚠️ ❌ RED LIGHT: Curriculum model needs further work")
            logger.warning("🔧 Consider: Higher curriculum ceiling or more training")
        
        return all_pass, {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_portfolio': final_portfolio,
            'steps': total_steps
        }
        
    except Exception as e:
        logger.error(f"❌ Curriculum evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    success, results = main()
    if success:
        print("🎉 ✅ GREEN LIGHT - CURRICULUM MODEL READY FOR REAL DATA")
    else:
        print("⚠️ ❌ RED LIGHT - CURRICULUM MODEL NEEDS MORE WORK")