#!/usr/bin/env python3
"""
🚨 QUICK GATE RE-TEST: V3 vs V2 Killer Conditions (Fast Version)
Test V3 against REALISTIC volatility that killed V2
Pass criteria: Return ≥+1%, Max DD ≤2%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

def create_realistic_test_data(n_periods=5000, seed=42):
    """Create realistic test data (same volatility that killed V2)"""
    np.random.seed(seed)
    
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    
    # REALISTIC intraday volatility (this is what killed V2)
    returns = np.random.normal(0.0001, 0.015, n_periods)  # 29% daily vol
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    
    # Feature data
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    logger.info(f"Realistic test data: {n_periods} periods")
    logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    logger.info(f"Daily volatility: {returns.std()*np.sqrt(390)*100:.1f}%")
    
    return feature_data, price_series

def quick_gate_test():
    """Quick gate re-test with strengthened V3"""
    
    logger.info("🚨 QUICK GATE RE-TEST: V3 vs V2 Killer Conditions")
    start_time = datetime.now()
    
    # Create realistic test data  
    feature_data, price_series = create_realistic_test_data(5000, seed=42)
    
    # Create V3 environment with STRENGTHENED impact model
    env = IntradayTradingEnvV3(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,        # 2% hard limit
        transaction_cost_pct=0.0001,        # 1bp additional cost  
        log_trades=False,
        # STRENGTHENED V3 parameters
        base_impact_bp=100.0,               # 5x stronger than before
        impact_exponent=0.5,
        verbose=False
    )
    
    # Train model
    logger.info("🤖 Training V3 model (5K steps)...")
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-5,      # Conservative LR
        n_steps=512,
        batch_size=64,
        n_epochs=3,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.001,          # Low exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
        device="auto"
    )
    
    model.learn(total_timesteps=5000, progress_bar=True)
    
    # Evaluate model  
    logger.info("🔍 Evaluating V3 under realistic conditions...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    portfolio_values = []
    rewards = []
    initial_capital = 100000
    
    for step in range(2000):  # 2K step evaluation
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        obs, reward, done, info = vec_env.step(action)
        
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value']) 
            rewards.append(reward[0])
        
        if step % 500 == 0 and step > 0 and portfolio_values:
            current_portfolio = portfolio_values[-1]
            current_return = (current_portfolio - initial_capital) / initial_capital
            peak = max(portfolio_values)
            dd = (peak - current_portfolio) / peak if peak > 0 else 0
            logger.info(f"  Step {step:4d}: Portfolio ${current_portfolio:8,.0f} ({current_return:+.2%}), DD {dd:.2%}")
        
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Calculate results
    if portfolio_values:
        final_portfolio = portfolio_values[-1]
        total_return = (final_portfolio - initial_capital) / initial_capital
        peak_portfolio = max(portfolio_values)
        max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio
        cumulative_reward = sum(rewards)
    else:
        total_return = 0
        max_drawdown = 0
        cumulative_reward = 0
        final_portfolio = initial_capital
    
    elapsed_time = datetime.now() - start_time
    
    # Gate criteria
    return_pass = total_return >= 0.01    # ≥+1%
    dd_pass = max_drawdown <= 0.02        # ≤2%
    gate_pass = return_pass and dd_pass
    
    logger.info(f"\n📊 QUICK GATE RE-TEST RESULTS:")
    logger.info(f"   Final portfolio: ${final_portfolio:,.0f}")
    logger.info(f"   Total return: {total_return:+.2%} {'✅' if return_pass else '❌'} (≥+1%)")
    logger.info(f"   Max drawdown: {max_drawdown:.2%} {'✅' if dd_pass else '❌'} (≤2%)")
    logger.info(f"   Cumulative reward: {cumulative_reward:,.0f}")
    logger.info(f"   Elapsed time: {elapsed_time}")
    
    if gate_pass:
        logger.info("🎉 ✅ GATE RE-TEST PASSED!")
        logger.info("   V3 successfully handles realistic volatility that killed V2")
        logger.info("   ✅ Ready for Phase D: 50K learning run")
        return True
    else:
        logger.info("⚠️ ❌ GATE RE-TEST FAILED")
        logger.info("   V3 still struggling with realistic market conditions")
        if not return_pass:
            logger.info(f"   🔧 Return too low: {total_return:+.2%} - Need stronger alpha signal")
        if not dd_pass:
            logger.info(f"   🔧 Drawdown too high: {max_drawdown:.2%} - Need stronger impact model")
        logger.info("   🛑 DO NOT proceed to expensive 50K training")
        return False

if __name__ == "__main__":
    success = quick_gate_test()
    
    if success:
        print("🚨 ✅ QUICK GATE PASS - V3 ready for 50K training!")
    else:
        print("⚠️ ❌ QUICK GATE FAIL - V3 needs more work")
    
    sys.exit(0 if success else 1)