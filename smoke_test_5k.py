#!/usr/bin/env python3
"""
🔥 SMOKE TEST - 5K STEPS SURGICAL PATCH
Quick verification that surgical changes prevent early drawdown
Target: DD should stay < 1% for first 2K steps
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
from stable_baselines3.common.monitor import Monitor

# Import our components
from src.gym_env.intraday_trading_env import IntradayTradingEnv
from src.gym_env.dual_reward_v2 import DualTickerRewardV2

class SmokeTestEnv(IntradayTradingEnv):
    """Environment with surgical reward patches for smoke test"""
    
    def __init__(self, *args, **kwargs):
        # Initialize reward system with ESCALATED SURGICAL CHANGES
        self.reward_calculator = DualTickerRewardV2(
            tc_bp=1.0,
            market_impact_bp=0.5,
            lambda_turnover=0.02,           # 20x stronger (escalated)
            target_turnover=1.0,            # Conservative target
            beta_volatility=0.0,            # Disabled
            hold_bonus_coef=0.002,          # New hold bonus
            sharpe_bonus=0.0,               # Disabled
            max_dd_penalty=0.0,             # Disabled
            verbose=False
        )
        
        super().__init__(*args, **kwargs)
        
        logger.info("🔥 SmokeTestEnv initialized with SURGICAL PATCHES")
    
    def step(self, action):
        """Enhanced step with surgical reward patches"""
        prev_portfolio = self.portfolio_value
        prev_position = getattr(self, '_prev_position_quantity', 0.0)
        
        obs, reward, done, truncated, info = super().step(action)
        
        curr_position = getattr(self, 'position_quantity', 0.0)
        trade_value = abs(curr_position - prev_position) * self.price_data.iloc[self.current_step-1]
        self._prev_position_quantity = curr_position
        
        v2_reward, reward_components = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio,
            curr_portfolio_value=self.portfolio_value,
            nvda_trade_value=trade_value,
            msft_trade_value=0.0,
            nvda_position=curr_position,
            msft_position=0.0,
            nvda_price=self.price_data.iloc[self.current_step-1],
            msft_price=510.0,
            step=self.current_step
        )
        
        reward = v2_reward
        info.update({
            'reward_breakdown': reward_components.to_dict(),
            'v2_reward': v2_reward
        })
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.reward_calculator.reset()
        self._prev_position_quantity = 0.0
        return super().reset(**kwargs)

def main():
    """5K smoke test with surgical patches"""
    logger.info("🔥 SMOKE TEST - 5K STEPS WITH SURGICAL PATCHES")
    logger.info("🎯 Target: DD < 1% for first 2K steps")
    
    start_time = datetime.now()
    
    # Generate test data
    np.random.seed(0)  # Same seed as before
    n_periods = 10000
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    base_price = 170.0
    returns = np.random.normal(0.0001, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Create environment
    env = SmokeTestEnv(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        transaction_cost_pct=0.0001,
        max_daily_drawdown_pct=0.15,  # Generous for training
        hourly_turnover_cap=10.0,
        reward_scaling=1.0,
        log_trades=False
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create and train model with tightened parameters
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=7e-5,    # Lower LR as suggested
        n_steps=1024,          # Tighter credit assignment
        batch_size=128,
        n_epochs=5,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.002,        # Much lower exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=0,
        device="auto"
    )
    
    logger.info("🚀 Training 5K steps with surgical patches...")
    
    model.learn(
        total_timesteps=5000,
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    training_time = datetime.now() - start_time
    logger.info(f"✅ Training completed in {training_time}")
    
    # Quick evaluation
    logger.info("🔍 Quick evaluation to check drawdown behavior...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    portfolio_values = []
    initial_capital = 100000
    peak_portfolio = initial_capital
    max_drawdown = 0
    
    for step in range(2000):  # Check first 2K steps
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        obs, reward, done, info = vec_env.step(action)
        
        if 'portfolio_value' in info[0]:
            portfolio_value = info[0]['portfolio_value']
            portfolio_values.append(portfolio_value)
            
            if portfolio_value > peak_portfolio:
                peak_portfolio = portfolio_value
            
            current_drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
            max_drawdown = max(max_drawdown, current_drawdown)
        
        if step % 500 == 0 and step > 0:
            current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
            current_return = (current_portfolio - initial_capital) / initial_capital
            logger.info(f"Step {step:4d}: Portfolio ${current_portfolio:8,.2f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
        
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
            break
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Final assessment
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    
    logger.info("🏁 SMOKE TEST RESULTS:")
    logger.info(f"   Steps evaluated: {len(portfolio_values)}")
    logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
    
    # Smoke test criteria
    early_dd_ok = max_drawdown < 0.01  # < 1% DD in first 2K steps
    no_crash = len(portfolio_values) >= 1500  # Didn't crash early
    
    logger.info("🔥 SMOKE TEST CRITERIA:")
    logger.info(f"   📉 DD < 1% in first 2K: {'✅ PASS' if early_dd_ok else '❌ FAIL'} ({max_drawdown:.2%})")
    logger.info(f"   🛡️ No early crash: {'✅ PASS' if no_crash else '❌ FAIL'} ({len(portfolio_values)} steps)")
    
    smoke_pass = early_dd_ok and no_crash
    
    if smoke_pass:
        logger.info("🎉 ✅ SMOKE PASS: Surgical patches working!")
        logger.info("📋 Ready for full 10K gate evaluation")
    else:
        logger.warning("⚠️ ❌ SMOKE FAIL: Need parameter adjustment")
        if max_drawdown >= 0.01:
            logger.warning(f"🔧 Suggested: Increase λ_turnover to 0.02 (DD={max_drawdown:.2%})")
    
    return smoke_pass

if __name__ == "__main__":
    success = main()
    if success:
        print("🔥 ✅ SMOKE PASS - Ready for gate evaluation")
    else:
        print("⚠️ ❌ SMOKE FAIL - Need parameter adjustment")