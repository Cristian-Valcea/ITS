#!/usr/bin/env python3
"""
🎯 PHASE ALPHA PROOF: V3 Safety Net + Alpha Signals
Test if agent can trade profitably when genuine alpha exists
while V3 safety net prevents over-trading

Gate Criteria:
- Safety Gate: DD ≤ 2% (always enforced by V3)
- Signal Gate: Return ≥ +0.5% vs do-nothing benchmark (when alpha present)
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
# Direct import to avoid module issues
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

def run_alpha_proof_test(
    training_steps: int = 25000,
    eval_steps: int = 2000,
    alpha_strength: float = 0.1,
    seed: int = 42
):
    """
    Run single-ticker proof with V3 safety net + alpha signals
    
    Test progression:
    1. No Alpha: Should learn do-nothing (0% return, 0% DD)
    2. Weak Alpha: Should trade cautiously (+0.5% return, <2% DD) 
    3. Strong Alpha: Should trade actively (+1%+ return, <2% DD)
    """
    
    logger.info(f"🎯 PHASE ALPHA PROOF: V3 Safety Net + Alpha Signals")
    logger.info(f"   Training: {training_steps} steps")
    logger.info(f"   Evaluation: {eval_steps} steps")
    logger.info(f"   Alpha strength: {alpha_strength}")
    
    start_time = datetime.now()
    
    # Create alpha-enhanced data
    logger.info("📊 Generating alpha-enhanced training data...")
    enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
        n_periods=10000, 
        seed=seed, 
        alpha_strength=alpha_strength
    )
    
    logger.info(f"   Enhanced features: {enhanced_features.shape}")
    logger.info(f"   Alpha signals: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish")
    
    # Create V3 environment with alpha features
    logger.info("🛡️ Creating V3 environment with locked safety parameters...")
    env = IntradayTradingEnvV3(
        processed_feature_data=enhanced_features,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,        # V3 safety gate: 2% hard limit
        transaction_cost_pct=0.0001,        # 1bp additional friction
        log_trades=False,
        # V3 CALIBRATED SAFETY PARAMETERS
        base_impact_bp=68.0,                # Calibrated for alpha signal extraction
        impact_exponent=0.5,                # Locked sqrt scaling
        verbose=False
    )
    
    # Wrap environment for training
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create model
    logger.info(f"🤖 Training model with alpha signals ({training_steps} steps)...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-5,          # Conservative learning rate
        n_steps=1024,                # Longer episodes for signal learning
        batch_size=128,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,               # Moderate exploration to find alpha
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        device="auto"
    )
    
    # Train model
    model.learn(total_timesteps=training_steps, progress_bar=True)
    
    # Evaluate model
    logger.info(f"🔍 Evaluating trained model ({eval_steps} steps)...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    # Tracking
    portfolio_values = []
    rewards = []
    actions_taken = []
    alpha_signals = []  # Track alpha signal when available
    
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
            
            # Extract alpha signal from observation (last feature)
            if len(obs[0]) > 12:  # Has alpha feature
                alpha_signals.append(obs[0][-1])  # Last feature is alpha signal
        
        # Progress logging
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
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    peak_portfolio = max(portfolio_values) if portfolio_values else initial_capital
    max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio if portfolio_values else 0
    cumulative_reward = sum(rewards) if rewards else 0
    
    # Action analysis
    action_counts = np.bincount(actions_taken, minlength=3) if actions_taken else [0, 0, 0]
    action_dist = action_counts / len(actions_taken) if actions_taken else [0, 0, 0]
    
    # Alpha signal analysis
    if alpha_signals:
        alpha_signal_correlation = np.corrcoef(alpha_signals[:-1], actions_taken[1:])[0, 1] if len(alpha_signals) > 1 else 0
        strong_alpha_steps = sum(1 for signal in alpha_signals if abs(signal) > 0.05)
    else:
        alpha_signal_correlation = 0
        strong_alpha_steps = 0
    
    elapsed_time = datetime.now() - start_time
    
    # Gate criteria evaluation
    safety_gate_pass = max_drawdown <= 0.02        # DD ≤ 2%
    signal_gate_pass = total_return >= 0.005       # Return ≥ +0.5%
    overall_pass = safety_gate_pass and signal_gate_pass
    
    # Results
    logger.info(f"\n📊 ALPHA PROOF TEST RESULTS:")
    logger.info(f"   Training steps: {training_steps}")
    logger.info(f"   Evaluation steps: {eval_steps}")
    logger.info(f"   Alpha strength: {alpha_strength}")
    logger.info(f"   Elapsed time: {elapsed_time}")
    
    logger.info(f"\n💰 PERFORMANCE METRICS:")
    logger.info(f"   Final portfolio: ${final_portfolio:,.0f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Max drawdown: {max_drawdown:.2%}")
    logger.info(f"   Cumulative reward: {cumulative_reward:,.0f}")
    
    logger.info(f"\n🎯 ACTION ANALYSIS:")
    logger.info(f"   SELL: {action_dist[0]:.1%}")
    logger.info(f"   HOLD: {action_dist[1]:.1%}")
    logger.info(f"   BUY:  {action_dist[2]:.1%}")
    logger.info(f"   Trading frequency: {100 - action_dist[1]*100:.1f}%")
    
    logger.info(f"\n🎯 ALPHA SIGNAL ANALYSIS:")
    logger.info(f"   Signal-action correlation: {alpha_signal_correlation:.3f}")
    logger.info(f"   Strong alpha signals: {strong_alpha_steps} ({strong_alpha_steps/len(alpha_signals)*100:.1f}%)")
    
    logger.info(f"\n🚨 GATE CRITERIA:")
    logger.info(f"   Safety Gate (DD ≤ 2%): {'✅ PASS' if safety_gate_pass else '❌ FAIL'} ({max_drawdown:.2%})")
    logger.info(f"   Signal Gate (Return ≥ +0.5%): {'✅ PASS' if signal_gate_pass else '❌ FAIL'} ({total_return:+.2%})")
    logger.info(f"   Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        logger.info("🎉 ✅ ALPHA PROOF SUCCESS!")
        logger.info("   V3 safety net allows profitable trading when genuine alpha exists")
        logger.info("   Ready for dual-ticker port and curriculum learning")
    else:
        logger.info("⚠️ ❌ ALPHA PROOF INCOMPLETE")
        if not safety_gate_pass:
            logger.info("   🛡️ Safety issue: Drawdown too high - V3 safety net may need strengthening")
        if not signal_gate_pass:
            logger.info("   📈 Signal issue: Not extracting enough value from alpha - May need stronger signals or more training")
    
    return overall_pass, {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_portfolio': final_portfolio,
        'cumulative_reward': cumulative_reward,
        'action_distribution': action_dist,
        'alpha_correlation': alpha_signal_correlation,
        'safety_gate_pass': safety_gate_pass,
        'signal_gate_pass': signal_gate_pass
    }

def main():
    """Run alpha proof test with different alpha strengths"""
    
    logger.info("🎯 ALPHA PROOF TEST SUITE")
    
    # Test different alpha strengths
    test_configs = [
        {'alpha_strength': 0.05, 'name': 'Weak Alpha'},
        {'alpha_strength': 0.10, 'name': 'Medium Alpha'},
        {'alpha_strength': 0.15, 'name': 'Strong Alpha'},
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing {config['name']} (strength={config['alpha_strength']})")
        logger.info(f"{'='*60}")
        
        success, metrics = run_alpha_proof_test(
            training_steps=25000,
            eval_steps=2000,
            alpha_strength=config['alpha_strength'],
            seed=42
        )
        
        results.append({
            'name': config['name'],
            'strength': config['alpha_strength'],
            'success': success,
            'metrics': metrics
        })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 ALPHA PROOF SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"{result['name']:12s}: {status} | Return {result['metrics']['total_return']:+.2%} | DD {result['metrics']['max_drawdown']:.2%}")
    
    successful_tests = sum(1 for r in results if r['success'])
    
    if successful_tests >= 2:
        logger.info(f"\n🎉 ✅ ALPHA PROOF SUITE PASSED ({successful_tests}/3 tests)")
        logger.info("   V3 + Alpha system ready for next phase")
        return True
    else:
        logger.info(f"\n⚠️ ❌ ALPHA PROOF SUITE FAILED ({successful_tests}/3 tests)")
        logger.info("   Need to debug V3 + Alpha interaction")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)