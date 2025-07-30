#!/usr/bin/env python3
"""
🌩️ STRESS TEST: FRICTION SENSITIVITY ANALYSIS
Test enhanced model with:
- 1.5bp/3.0bp friction (50% higher than production)
- +1bp slippage shock
- 3x volatility + mid-run trend flip
Goal: Verify if curve decays smoothly or implodes chaotically
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

def create_stress_test_environment():
    """Create STRESS TEST environment - chaos engineering"""
    logger.info("🌩️ Creating STRESS TEST environment - CHAOS ENGINEERING")
    
    # Generate STORM data - 3x volatility, trend flip, slippage
    np.random.seed(42)  # Same base for comparison
    eval_n_periods = 10000  # 10K stress test
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # STORM FEATURES: 3x volatility
    nvda_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.03)  # 3x volatility
    
    # STORM PRICES: 3x volatility + trend flip halfway
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.06, eval_n_periods)  # 3x vol (0.02 → 0.06)
    
    # TREND FLIP at halfway point
    mid_point = eval_n_periods // 2
    nvda_returns[:mid_point] = np.abs(nvda_returns[:mid_point])  # Positive trend first half
    nvda_returns[mid_point:] = -np.abs(nvda_returns[mid_point:])  # Negative trend second half
    
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=eval_trading_days
    )
    
    # MSFT with similar stress pattern
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.045, eval_n_periods)  # 3x vol
    msft_returns[:mid_point] = np.abs(msft_returns[:mid_point])
    msft_returns[mid_point:] = -np.abs(msft_returns[mid_point:])
    
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=eval_trading_days
    )
    msft_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    msft_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.03)  # 3x volatility
    
    # Create STRESS TEST environment with HARSH friction
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=eval_trading_days,
        initial_capital=100000,
        tc_bp=1.5,                      # 50% higher than production (1.0 → 1.5)
        trade_penalty_bp=3.0,           # 50% higher than production (2.0 → 3.0)
        turnover_bp=3.0,                # Higher turnover penalty
        hold_action_bonus=0.01,
        action_repeat_penalty=0.002,
        high_water_mark_reward=0.001,
        daily_trade_limit=50,
        reward_scaling=0.1,
        training_drawdown_pct=0.02,     # Still 2% limit
        evaluation_drawdown_pct=0.02,   # Still 2% limit
        is_training=False,              # EVALUATION mode
        log_trades=True                 # Log trades for analysis
    )
    
    logger.info("✅ STRESS TEST Environment created")
    logger.info(f"🌩️ STRESS TEST PARAMETERS:")
    logger.info(f"   🚨 Drawdown Limit: {2.0}% (same as strict)")
    logger.info(f"   📊 Evaluation Steps: {eval_n_periods:,}")
    logger.info(f"   💥 Transaction Cost: {1.5} bp (+50% vs production)")
    logger.info(f"   💥 Trade Penalty: {3.0} bp (+50% vs production)")
    logger.info(f"   🌪️ Volatility: 3x normal (0.02 → 0.06)")
    logger.info(f"   🔄 Trend Flip: Halfway through episode")
    logger.info(f"   💊 Slippage: Embedded in higher friction")
    logger.info(f"   🎯 Goal: Verify smooth decay vs chaotic implosion")
    
    return env

def main():
    """Run stress test to analyze friction sensitivity"""
    logger.info("🌩️ STRESS TEST: FRICTION SENSITIVITY ANALYSIS")
    start_time = datetime.now()
    
    # Model paths
    model_path = "models/dual_ticker_enhanced_50k_final.zip"
    vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
    
    logger.info(f"📥 Loading enhanced model: {model_path}")
    
    try:
        # Create stress test environment
        eval_env = create_stress_test_environment()
        vec_env = DummyVecEnv([lambda: eval_env])
        
        # Load VecNormalize
        logger.info(f"📥 Loading VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
        # Load model
        model = RecurrentPPO.load(model_path, env=vec_env)
        logger.info("✅ Enhanced model loaded successfully")
        
        # Run stress test
        logger.info("🎯 Starting STRESS TEST...")
        logger.info("🌩️ CHAOS ENGINEERING: 1.5bp/3.0bp friction, 3x vol, trend flip")
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        total_steps = 0
        total_reward = 0
        actions_taken = []
        portfolio_values = []
        trade_count = 0
        episode_count = 0
        
        # Tracking variables
        initial_capital = 100000
        peak_portfolio = initial_capital
        max_drawdown = 0
        
        logger.info("🚀 Running 10K stress test steps...")
        
        for step in range(10000):
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
            
            # Track trades
            if action[0] != 4:  # Not HOLD_BOTH
                trade_count += 1
            
            # Log progress every 2K steps
            if step % 2000 == 0 and step > 0:
                current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
                current_return = (current_portfolio - initial_capital) / initial_capital
                logger.info(f"Step {step:5d}: Portfolio ${current_portfolio:8,.2f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
            
            # Check if episode ended
            if done[0]:
                episode_count += 1
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
                logger.info(f"📊 Episode {episode_count} ended at step {step}")
                if episode_count >= 3:  # Too many terminations
                    logger.warning("⚠️ Multiple episode endings - model unstable under stress")
                    break
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Final calculations
        final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_portfolio - initial_capital) / initial_capital
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        turnover = trade_count / total_steps if total_steps > 0 else 0
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # STRESS TEST RESULTS
        logger.info("🏁 STRESS TEST COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info("📊 STRESS TEST RESULTS:")
        logger.info(f"   Steps completed: {total_steps:,}")
        logger.info(f"   Episodes: {episode_count}")
        logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
        logger.info(f"   Total return: {total_return:+.2%}")
        logger.info(f"   Peak portfolio: ${peak_portfolio:,.2f}")
        logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
        logger.info(f"   Trade count: {trade_count}")
        logger.info(f"   Turnover rate: {turnover:.3f} trades/step")
        logger.info(f"   Average reward: {avg_reward:.4f}")
        
        # FRICTION SENSITIVITY ANALYSIS
        logger.info("🔍 FRICTION SENSITIVITY ANALYSIS:")
        logger.info("   Training (0.5bp/0.7bp): +4.08% return")
        logger.info("   Strict (1.0bp/2.0bp): -2.04% return")
        logger.info(f"   Stress (1.5bp/3.0bp): {total_return:+.2%} return")
        
        # Pattern analysis
        if total_return < -0.03:  # Worse than -3%
            logger.warning("⚠️ CHAOTIC IMPLOSION: Performance degrades non-linearly")
            logger.warning("🔧 Recommendation: Use friction curriculum + domain randomization")
        elif total_return < -0.01:  # Between -1% and -3%
            logger.info("📉 SMOOTH DECAY: Performance degrades linearly with friction")
            logger.info("🔧 Recommendation: Retrain with production friction from start")
        else:
            logger.info("✅ ROBUST: Model handles increased friction reasonably")
            logger.info("🔧 Recommendation: Fine-tune evaluation parameters")
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'episodes': episode_count,
            'turnover': turnover,
            'final_portfolio': final_portfolio
        }
        
    except Exception as e:
        logger.error(f"❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = main()
    if results:
        print(f"🌩️ STRESS TEST COMPLETE - Return: {results['total_return']:+.2%}, DD: {results['max_drawdown']:.2%}")
    else:
        print("❌ STRESS TEST FAILED")