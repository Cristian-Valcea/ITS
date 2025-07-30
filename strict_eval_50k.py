#!/usr/bin/env python3
"""
🔍 STRICT 2% DD EVALUATION - 50K STEPS
Professional validation: Load enhanced model, 2% DD wrapper, 50K steps
Target: ≥+1% return, ≤2% max DD = ✅ PASS
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
    """Create STRICT 2% DD evaluation environment - professional validation"""
    logger.info("🔍 Creating STRICT EVALUATION environment - PROFESSIONAL VALIDATION")
    
    # Generate evaluation data (same seed for consistency)
    np.random.seed(42)  
    eval_n_periods = 50000  # Full 50K evaluation
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # Mock evaluation data (same generation as training for fair comparison)
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
    
    # Create STRICT EVALUATION environment
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
    logger.info(f"🛡️ STRICT PARAMETERS:")
    logger.info(f"   🚨 Drawdown Limit: {2.0}% (STRICT)")
    logger.info(f"   📊 Evaluation Steps: {eval_n_periods:,}")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (production)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (production)")
    logger.info(f"   🎯 Target: ≥+1% return, ≤2% max DD")
    
    return env

def main():
    """Run strict professional evaluation"""
    logger.info("🔍 STRICT 2% DD EVALUATION - PROFESSIONAL VALIDATION")
    start_time = datetime.now()
    
    # Model paths
    model_path = "models/dual_ticker_enhanced_50k_final.zip"
    vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
    
    logger.info(f"📥 Loading enhanced model: {model_path}")
    
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
        logger.info("✅ Enhanced model loaded successfully")
        
        # Run strict evaluation
        logger.info("🎯 Starting STRICT 50K EVALUATION...")
        logger.info("🔍 PROFESSIONAL VALIDATION: 2% DD limit, production friction")
        
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
        
        # Tracking variables
        initial_capital = 100000
        peak_portfolio = initial_capital
        
        logger.info("🚀 Running 50K steps with strict 2% drawdown limit...")
        
        for step in range(50000):
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
            
            # Log progress every 10K steps
            if step % 10000 == 0 and step > 0:
                current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
                current_return = (current_portfolio - initial_capital) / initial_capital
                logger.info(f"Step {step:5d}: Portfolio ${current_portfolio:8,.2f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
            
            # Check if episode ended
            if done[0]:
                episode_count += 1
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
                logger.info(f"📊 Episode {episode_count} ended at step {step}")
                if episode_count > 1:  # Allow one episode end, but multiple = problem
                    logger.warning("⚠️ Multiple episode endings - may indicate instability")
                    break
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Final calculations
        final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_portfolio - initial_capital) / initial_capital
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # PROFESSIONAL VALIDATION RESULTS
        logger.info("🏁 STRICT EVALUATION COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info("📊 PROFESSIONAL VALIDATION RESULTS:")
        logger.info(f"   Steps completed: {total_steps:,}")
        logger.info(f"   Episodes: {episode_count}")
        logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
        logger.info(f"   Total return: {total_return:+.2%}")
        logger.info(f"   Peak portfolio: ${peak_portfolio:,.2f}")
        logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
        logger.info(f"   Average reward: {avg_reward:.4f}")
        
        # PASS/FAIL CRITERIA
        logger.info("🎯 PROFESSIONAL VALIDATION CRITERIA:")
        
        return_pass = total_return >= 0.01  # ≥+1% return
        drawdown_pass = max_drawdown <= 0.02  # ≤2% max DD
        stability_pass = episode_count <= 1  # No premature terminations
        
        logger.info(f"   📈 Return ≥ +1.0%: {'✅ PASS' if return_pass else '❌ FAIL'} ({total_return:+.2%})")
        logger.info(f"   📉 Max DD ≤ 2.0%: {'✅ PASS' if drawdown_pass else '❌ FAIL'} ({max_drawdown:.2%})")
        logger.info(f"   🛡️ Stability: {'✅ PASS' if stability_pass else '❌ FAIL'} ({episode_count} episodes)")
        
        # FINAL VERDICT
        all_pass = return_pass and drawdown_pass and stability_pass
        
        if all_pass:
            logger.info("🏆 ✅ PROFESSIONAL VALIDATION: PASSED")
            logger.info("🚀 Model ready for stress testing and real data evaluation")
        else:
            logger.warning("⚠️ ❌ PROFESSIONAL VALIDATION: FAILED")
            logger.warning("🔧 Model needs adjustment before proceeding")
        
        return all_pass, {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_portfolio': final_portfolio,
            'episodes': episode_count,
            'steps': total_steps
        }
        
    except Exception as e:
        logger.error(f"❌ Strict evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    success, results = main()
    if success:
        print("✅ STRICT EVALUATION PASSED - READY FOR NEXT PHASE")
    else:
        print("❌ STRICT EVALUATION FAILED - NEEDS ADJUSTMENT")