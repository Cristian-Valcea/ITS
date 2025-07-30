#!/usr/bin/env python3
"""
🎯 FOCUSED ALPHA TEST - Force Strong Alpha Signal Learning
Create a simple scenario where alpha signals are always strong and consistent
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

def create_strong_alpha_data(n_periods: int = 2000):
    """Create data with strong, consistent alpha signals"""
    
    np.random.seed(42)
    
    # Create price series with predictable patterns
    base_price = 170.0
    prices = []
    alpha_signals = []
    
    for i in range(n_periods):
        # Create a predictable price pattern
        if i % 20 < 10:  # First half of cycle - price goes up
            price_change = 0.002 + np.random.normal(0, 0.005)  # +0.2% plus noise
            alpha_signal = 0.15  # Strong bullish signal
        else:  # Second half - price goes down
            price_change = -0.002 + np.random.normal(0, 0.005)  # -0.2% plus noise
            alpha_signal = -0.15  # Strong bearish signal
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + price_change)
        
        prices.append(price)
        alpha_signals.append(alpha_signal)
    
    # Create features: 12 random features + alpha signal
    random_features = np.random.randn(n_periods, 12).astype(np.float32)
    alpha_feature = np.array(alpha_signals).reshape(-1, 1).astype(np.float32)
    features = np.hstack([random_features, alpha_feature])
    
    # Create price series
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    price_series = pd.Series(prices, index=trading_days)
    
    logger.info(f"🎯 Strong alpha data created:")
    logger.info(f"   Periods: {n_periods}")
    logger.info(f"   Features: {features.shape}")
    logger.info(f"   Alpha pattern: 10 bullish → 10 bearish (repeating)")
    logger.info(f"   Price pattern: +0.2% → -0.2% (predictable)")
    
    return features, price_series

def run_focused_alpha_test():
    """Run focused test with strong, predictable alpha"""
    
    logger.info("🎯 FOCUSED ALPHA TEST - Strong & Predictable Signals")
    
    start_time = datetime.now()
    
    # Create strong, predictable alpha data
    features, price_series = create_strong_alpha_data(2000)
    
    # Create V3 environment with calibrated parameters
    env = IntradayTradingEnvV3(
        processed_feature_data=features,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        transaction_cost_pct=0.0001,
        log_trades=False,
        base_impact_bp=68.0,       # Calibrated impact
        impact_exponent=0.5,
        verbose=False
    )
    
    # Wrap environment
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create model with settings optimized for pattern learning
    logger.info("🤖 Training model to learn predictable alpha pattern...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=1e-4,              # Higher learning rate for pattern learning
        n_steps=512,
        batch_size=64,
        n_epochs=8,                      # More epochs for pattern learning
        gamma=0.99,                      # Shorter horizon for immediate rewards
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,                   # High exploration to learn pattern
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
        device="auto"
    )
    
    # Train model
    model.learn(total_timesteps=15000, progress_bar=True)
    
    # Evaluate model
    logger.info("🔍 Evaluating pattern learning...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    # Tracking
    portfolio_values = []
    rewards = []
    actions_taken = []
    alpha_signals = []
    correct_predictions = 0
    total_predictions = 0
    
    initial_capital = 100000
    
    for step in range(1000):
        # Get action
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        # Step environment
        result = vec_env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, _, info = result
        
        # Track metrics
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
            rewards.append(reward[0])
            actions_taken.append(action[0])
            
            # Extract alpha signal
            if len(obs[0]) > 12:
                alpha_signal = obs[0][-1]
                alpha_signals.append(alpha_signal)
                
                # Check if action matches alpha signal
                if (alpha_signal > 0.1 and action[0] == 2) or (alpha_signal < -0.1 and action[0] == 0):
                    correct_predictions += 1
                total_predictions += 1
        
        # Progress logging
        if step % 200 == 0 and step > 0 and portfolio_values:
            current_portfolio = portfolio_values[-1]
            current_return = (current_portfolio - initial_capital) / initial_capital
            peak = max(portfolio_values)
            dd = (peak - current_portfolio) / peak if peak > 0 else 0
            pattern_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            logger.info(f"  Step {step:4d}: Portfolio ${current_portfolio:8,.0f} ({current_return:+.2%}), DD {dd:.2%}, Pattern Accuracy {pattern_accuracy:.1%}")
        
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Calculate results
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    peak_portfolio = max(portfolio_values) if portfolio_values else initial_capital
    max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio if portfolio_values else 0
    pattern_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Action analysis
    action_counts = np.bincount(actions_taken, minlength=3) if actions_taken else [0, 0, 0]
    action_dist = action_counts / len(actions_taken) if actions_taken else [0, 0, 0]
    
    elapsed_time = datetime.now() - start_time
    
    # Gate criteria
    safety_gate_pass = max_drawdown <= 0.02
    signal_gate_pass = total_return >= 0.005  # +0.5%
    pattern_gate_pass = pattern_accuracy >= 0.6  # 60% pattern accuracy
    overall_pass = safety_gate_pass and signal_gate_pass and pattern_gate_pass
    
    # Results
    logger.info(f"\n📊 FOCUSED ALPHA TEST RESULTS:")
    logger.info(f"   Training: 15000 steps")
    logger.info(f"   Evaluation: 1000 steps")
    logger.info(f"   Elapsed time: {elapsed_time}")
    
    logger.info(f"\n💰 PERFORMANCE:")
    logger.info(f"   Final portfolio: ${final_portfolio:,.0f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Max drawdown: {max_drawdown:.2%}")
    
    logger.info(f"\n🎯 PATTERN LEARNING:")
    logger.info(f"   Pattern accuracy: {pattern_accuracy:.1%}")
    logger.info(f"   Correct predictions: {correct_predictions}/{total_predictions}")
    
    logger.info(f"\n🎯 ACTION DISTRIBUTION:")
    logger.info(f"   SELL: {action_dist[0]:.1%}")
    logger.info(f"   HOLD: {action_dist[1]:.1%}")
    logger.info(f"   BUY:  {action_dist[2]:.1%}")
    logger.info(f"   Trading frequency: {100 - action_dist[1]*100:.1f}%")
    
    logger.info(f"\n🚨 GATE CRITERIA:")
    logger.info(f"   Safety Gate (DD ≤ 2%): {'✅ PASS' if safety_gate_pass else '❌ FAIL'} ({max_drawdown:.2%})")
    logger.info(f"   Signal Gate (Return ≥ +0.5%): {'✅ PASS' if signal_gate_pass else '❌ FAIL'} ({total_return:+.2%})")
    logger.info(f"   Pattern Gate (Accuracy ≥ 60%): {'✅ PASS' if pattern_gate_pass else '❌ FAIL'} ({pattern_accuracy:.1%})")
    logger.info(f"   Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        logger.info("🎉 ✅ FOCUSED ALPHA TEST SUCCESS!")
        logger.info("   Agent successfully learned to extract predictable alpha")
        logger.info("   V3 calibration working correctly")
    else:
        logger.info("⚠️ ❌ FOCUSED ALPHA TEST INCOMPLETE")
        if not safety_gate_pass:
            logger.info("   🛡️ Safety violation")
        if not signal_gate_pass:
            logger.info("   📈 Insufficient alpha extraction")
        if not pattern_gate_pass:
            logger.info("   🎯 Pattern learning insufficient")
    
    return overall_pass

if __name__ == "__main__":
    success = run_focused_alpha_test()
    print(f"\n🎯 FOCUSED ALPHA TEST: {'SUCCESS' if success else 'FAILED'}")