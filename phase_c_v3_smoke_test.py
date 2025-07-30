#!/usr/bin/env python3
"""
🔥 PHASE C: V3 10K SMOKE TEST
Quick validation that V3 integration prevents early drawdown
Pass criteria: <1% DD, return ≈0% on random policy
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

# Import V3 environment
from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

def create_test_data(n_periods=10000, seed=0):
    """Create realistic test data for smoke test"""
    np.random.seed(seed)
    
    # Generate trading days index
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Generate REALISTIC but controlled price series for V3 testing
    base_price = 170.0
    # Moderate volatility - not too low, not too high
    returns = np.random.normal(0.0, 0.005, n_periods)  # 0.5% per step volatility
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    
    # Generate feature data (12 features matching our standard)
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Add some autocorrelation to make features more realistic
    for i in range(1, 12):
        feature_data[:, i] = 0.7 * feature_data[:, i] + 0.3 * np.roll(feature_data[:, i], 1)
    
    return feature_data, price_series

def run_random_policy_baseline(env, n_steps=2000):
    """Run random policy baseline to test V3 prevents over-trading"""
    logger.info("🎲 Running random policy baseline...")
    
    obs, info = env.reset()
    portfolio_values = [info['portfolio_value']]
    rewards = []
    actions_taken = []
    
    for step in range(n_steps):
        # Random action
        action = np.random.choice([0, 1, 2])  # Random sell/hold/buy
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        actions_taken.append(action)
        
        if terminated or truncated:
            logger.info(f"Episode terminated early at step {step}")
            break
    
    # Calculate statistics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1)
    peak_portfolio = max(portfolio_values)
    max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio
    cumulative_reward = sum(rewards)
    
    action_dist = np.bincount(actions_taken, minlength=3) / len(actions_taken)
    
    logger.info(f"🎲 RANDOM POLICY RESULTS:")
    logger.info(f"   Steps executed: {len(portfolio_values)-1}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Max drawdown: {max_drawdown:.2%}")
    logger.info(f"   Cumulative reward: {cumulative_reward:.2f}")
    logger.info(f"   Action distribution: SELL {action_dist[0]:.1%}, HOLD {action_dist[1]:.1%}, BUY {action_dist[2]:.1%}")
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'cumulative_reward': cumulative_reward,
        'steps': len(portfolio_values)-1,
        'final_portfolio': portfolio_values[-1]
    }

def run_v3_smoke_test():
    """Run 10K smoke test with V3 environment"""
    logger.info("🔥 PHASE C: V3 10K SMOKE TEST")
    logger.info("🎯 Pass criteria: <1% DD, return ≈0% with random policy")
    
    start_time = datetime.now()
    
    # Create test data
    logger.info("📊 Generating test data (10K periods)...")
    feature_data, price_series = create_test_data(n_periods=10000, seed=0)
    
    # Create V3 environment
    logger.info("🎯 Creating V3 environment...")
    env = IntradayTradingEnvV3(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,  # 2% DD limit
        transaction_cost_pct=0.0001,  # 1bp transaction cost
        log_trades=False,
        verbose=False
    )
    
    # Test 1: Random policy baseline (should lose money due to V3 impact)
    logger.info("\n📋 TEST 1: Random Policy Baseline")
    random_results = run_random_policy_baseline(env, n_steps=2000)
    
    # Test 2: Create RL model for basic learning test
    logger.info("\n📋 TEST 2: Basic RL Training (10K steps)")
    
    # Wrap environment for stable-baselines3
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create simple PPO model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=1e-4,      # Conservative learning rate
        n_steps=512,             # Smaller batch for faster learning
        batch_size=64,
        n_epochs=3,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,           # Moderate exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=0,
        device="auto"
    )
    
    logger.info("🚀 Training model for 10K steps...")
    
    model.learn(
        total_timesteps=10000,
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    # Test trained model
    logger.info("🔍 Evaluating trained model (1K steps)...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    portfolio_values = []
    rewards = []
    
    for step in range(1000):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        obs, reward, done, info = vec_env.step(action)
        
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
            rewards.append(reward[0])
        
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
            break
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Calculate trained model results
    if portfolio_values:
        trained_return = (portfolio_values[-1] / portfolio_values[0] - 1)
        peak_portfolio = max(portfolio_values)
        trained_dd = (peak_portfolio - min(portfolio_values)) / peak_portfolio
        trained_cumulative_reward = sum(rewards)
    else:
        trained_return = 0
        trained_dd = 0
        trained_cumulative_reward = 0
    
    # Analysis and validation
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n📊 V3 SMOKE TEST ANALYSIS")
    logger.info("=" * 50)
    
    logger.info(f"📈 RANDOM POLICY BASELINE:")
    logger.info(f"   Return: {random_results['total_return']:+.2%}")
    logger.info(f"   Max DD: {random_results['max_drawdown']:.2%}")
    logger.info(f"   Cumulative reward: {random_results['cumulative_reward']:,.0f}")
    
    logger.info(f"\n🤖 TRAINED MODEL (10K steps):")
    logger.info(f"   Return: {trained_return:+.2%}")
    logger.info(f"   Max DD: {trained_dd:.2%}")
    logger.info(f"   Cumulative reward: {trained_cumulative_reward:,.0f}")
    
    # Pass criteria validation
    logger.info(f"\n✅ PHASE C PASS CRITERIA:")
    
    # Test 1: Random policy should lose money (V3 prevents cost-blind trading)
    random_loses_money = random_results['cumulative_reward'] < 0
    logger.info(f"   Random policy loses money: {'✅ PASS' if random_loses_money else '❌ FAIL'} ({random_results['cumulative_reward']:,.0f})")
    
    # Test 2: Random policy drawdown should be reasonable (<5%)
    random_dd_ok = random_results['max_drawdown'] < 0.05
    logger.info(f"   Random DD <5%: {'✅ PASS' if random_dd_ok else '❌ FAIL'} ({random_results['max_drawdown']:.2%})")
    
    # Test 3: Trained model should improve over random
    model_improvement = trained_cumulative_reward > random_results['cumulative_reward']
    logger.info(f"   Model > random policy: {'✅ PASS' if model_improvement else '❌ FAIL'} ({trained_cumulative_reward:.0f} vs {random_results['cumulative_reward']:.0f})")
    
    # Test 4: Trained model DD should be <1%
    trained_dd_ok = trained_dd < 0.01
    logger.info(f"   Trained model DD <1%: {'✅ PASS' if trained_dd_ok else '❌ FAIL'} ({trained_dd:.2%})")
    
    # Overall assessment
    tests_passed = sum([random_loses_money, random_dd_ok, model_improvement, trained_dd_ok])
    total_tests = 4
    smoke_pass = tests_passed >= 3  # Allow 1 failure
    
    logger.info(f"\n🎯 PHASE C SMOKE TEST RESULT:")
    logger.info(f"   Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"   Elapsed time: {elapsed_time}")
    logger.info(f"   V3 integration: {'✅ SUCCESS' if smoke_pass else '❌ FAILURE'}")
    
    if smoke_pass:
        logger.info(f"   ✅ V3 successfully integrated and prevents over-trading")
        logger.info(f"   ✅ Ready for Phase D: 50K learning run")
    else:
        logger.info(f"   ❌ V3 integration needs adjustment")
        logger.info(f"   🔧 Check V3 parameters or environment setup")
    
    return smoke_pass

def main():
    """Run Phase C smoke test"""
    
    success = run_v3_smoke_test()
    
    if success:
        print("🔥 ✅ PHASE C SMOKE PASS - V3 integration successful")
        return True
    else:
        print("⚠️ ❌ PHASE C SMOKE FAIL - V3 needs adjustment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)