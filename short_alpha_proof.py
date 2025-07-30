#!/usr/bin/env python3
"""
🎯 SHORT ALPHA PROOF - Calibrated V3 + Alpha Signals
Quick validation that V3 safety net + alpha signals work together
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
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

def run_short_alpha_proof(
    training_steps: int = 10000,
    eval_steps: int = 1000,
    alpha_strength: float = 0.15
):
    """Run short alpha proof test with calibrated V3"""
    
    logger.info(f"🎯 SHORT ALPHA PROOF - Calibrated V3 + Alpha")
    logger.info(f"   Training: {training_steps} steps")
    logger.info(f"   Evaluation: {eval_steps} steps")
    logger.info(f"   Alpha strength: {alpha_strength}")
    
    start_time = datetime.now()
    
    # Create alpha-enhanced data
    enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
        n_periods=5000, 
        seed=42, 
        alpha_strength=alpha_strength
    )
    
    logger.info(f"   Enhanced features: {enhanced_features.shape}")
    logger.info(f"   Alpha signals: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish")
    
    # Create V3 environment with calibrated parameters
    env = IntradayTradingEnvV3(
        processed_feature_data=enhanced_features,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        transaction_cost_pct=0.0001,
        log_trades=False,
        # V3 CALIBRATED PARAMETERS
        base_impact_bp=68.0,                # Calibrated for alpha extraction
        impact_exponent=0.5,
        verbose=False
    )
    
    # Wrap environment 
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create model
    logger.info(f"🤖 Training model with calibrated V3 + alpha signals...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=5e-5,              # Slightly higher LR for faster learning
        n_steps=512,                     # Shorter episodes for quick test
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.02,                   # Higher exploration to find alpha
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
        device="auto"
    )
    
    # Train model
    model.learn(total_timesteps=training_steps, progress_bar=True)
    
    # Evaluate model
    logger.info(f"🔍 Evaluating trained model...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    # Tracking
    portfolio_values = []
    rewards = []
    actions_taken = []
    
    initial_capital = 100000
    
    for step in range(eval_steps):
        # Get action
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        # Step environment
        obs, reward, done, info = vec_env.step(action)
        
        # Track metrics
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
            rewards.append(reward[0])
            actions_taken.append(action[0])
        
        # Progress logging
        if step % 250 == 0 and step > 0 and portfolio_values:
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
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    peak_portfolio = max(portfolio_values) if portfolio_values else initial_capital
    max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio if portfolio_values else 0
    cumulative_reward = sum(rewards) if rewards else 0
    
    # Action analysis
    action_counts = np.bincount(actions_taken, minlength=3) if actions_taken else [0, 0, 0]
    action_dist = action_counts / len(actions_taken) if actions_taken else [0, 0, 0]
    
    elapsed_time = datetime.now() - start_time
    
    # Gate criteria
    safety_gate_pass = max_drawdown <= 0.02        # DD ≤ 2%
    signal_gate_pass = total_return >= 0.005       # Return ≥ +0.5%
    overall_pass = safety_gate_pass and signal_gate_pass
    
    # Results
    logger.info(f"\n📊 SHORT ALPHA PROOF RESULTS:")
    logger.info(f"   Training: {training_steps} steps")
    logger.info(f"   Evaluation: {eval_steps} steps")
    logger.info(f"   Alpha strength: {alpha_strength}")
    logger.info(f"   Elapsed time: {elapsed_time}")
    
    logger.info(f"\n💰 PERFORMANCE:")
    logger.info(f"   Final portfolio: ${final_portfolio:,.0f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Max drawdown: {max_drawdown:.2%}")
    logger.info(f"   Cumulative reward: {cumulative_reward:,.0f}")
    
    logger.info(f"\n🎯 ACTION DISTRIBUTION:")
    logger.info(f"   SELL: {action_dist[0]:.1%}")
    logger.info(f"   HOLD: {action_dist[1]:.1%}")  
    logger.info(f"   BUY:  {action_dist[2]:.1%}")
    logger.info(f"   Trading frequency: {100 - action_dist[1]*100:.1f}%")
    
    logger.info(f"\n🚨 GATE CRITERIA:")
    logger.info(f"   Safety Gate (DD ≤ 2%): {'✅ PASS' if safety_gate_pass else '❌ FAIL'} ({max_drawdown:.2%})")
    logger.info(f"   Signal Gate (Return ≥ +0.5%): {'✅ PASS' if signal_gate_pass else '❌ FAIL'} ({total_return:+.2%})")
    logger.info(f"   Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        logger.info("🎉 ✅ SHORT ALPHA PROOF SUCCESS!")
        logger.info("   Calibrated V3 + alpha signals working correctly")
        logger.info("   Ready for full alpha proof and dual-ticker port")
    else:
        logger.info("⚠️ ❌ SHORT ALPHA PROOF INCOMPLETE")
        if not safety_gate_pass:
            logger.info("   🛡️ Safety issue: Need to debug V3 safety controls")
        if not signal_gate_pass:
            logger.info("   📈 Signal issue: Need stronger alpha or more training")
    
    return overall_pass

if __name__ == "__main__":
    success = run_short_alpha_proof()
    print(f"\n🎯 SHORT ALPHA PROOF: {'SUCCESS' if success else 'FAILED'}")