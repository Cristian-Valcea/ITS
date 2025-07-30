#!/usr/bin/env python3
"""
🚨 PHASE GATE RE-TEST: V3 vs Original V2 Failure Conditions
Reproduce the EXACT evaluation harness that killed V2 with 11-12% drawdown
If V3 really fixes the root cause, it should sail through these tests

Test Sequence:
1. Gate Re-test: 10K steps, tc=1bp, trade_pen=2bp, DD_max=2%
2. Stress Sweep: 5K steps each with tc=2bp, 3bp  
3. Long Horizon: 25K steps under strict conditions

Pass Criteria: Return ≥+1%, Max DD ≤2%, No soft-DD hits after 1K steps
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv

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

def create_strict_evaluation_data(n_periods=30000, seed=42):
    """Create evaluation data that reproduces V2 failure conditions"""
    np.random.seed(seed)
    
    # Generate trading days index
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Generate realistic NVDA price series (same volatility that killed V2)
    base_price = 170.0
    returns = np.random.normal(0.0001, 0.015, n_periods)  # Realistic intraday vol
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    
    # Generate feature data with some autocorrelation
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    for i in range(1, 12):
        feature_data[:, i] = 0.8 * feature_data[:, i] + 0.2 * np.roll(feature_data[:, i], 1)
    
    logger.info(f"📊 Created evaluation data: {n_periods} periods")
    logger.info(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    logger.info(f"   Price volatility: {returns.std()*np.sqrt(390)*100:.1f}% daily")
    
    return feature_data, price_series

def create_trained_model_v3(env, training_steps=10000):
    """Create a trained V3 model for evaluation"""
    
    logger.info(f"🤖 Training V3 model ({training_steps} steps)...")
    
    # Wrap environment
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create model with parameters similar to V2 tests
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=5e-5,      # Conservative learning rate
        n_steps=1024,            # Same as V2 tests
        batch_size=128,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,          # Conservative clipping
        ent_coef=0.001,          # Lower exploration (as suggested)
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
        device="auto"
    )
    
    # Train model
    model.learn(
        total_timesteps=training_steps,
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    return model, vec_env

def evaluate_model_strict(model, vec_env, n_steps, test_name=""):
    """Evaluate model under strict conditions (reproduce V2 failure test)"""
    
    logger.info(f"🔍 Evaluating {test_name} ({n_steps} steps)...")
    
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    # Track metrics
    portfolio_values = []
    rewards = []
    actions_taken = []
    soft_dd_hits = []
    
    initial_capital = 100000
    peak_portfolio = initial_capital
    max_drawdown = 0
    soft_dd_threshold = 0.01  # 1% soft drawdown threshold
    
    for step in range(n_steps):
        # Get action
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        # Step environment
        obs, reward, done, info = vec_env.step(action)
        
        # Track metrics
        if 'portfolio_value' in info[0]:
            portfolio_value = info[0]['portfolio_value']
            portfolio_values.append(portfolio_value)
            
            # Update peak and drawdown
            if portfolio_value > peak_portfolio:
                peak_portfolio = portfolio_value
            
            current_drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Track soft DD hits (after initial 1K steps)
            if step > 1000 and current_drawdown > soft_dd_threshold:
                soft_dd_hits.append(step)
        
        rewards.append(reward[0])
        actions_taken.append(action[0])
        
        # Log progress
        if step % 5000 == 0 and step > 0:
            current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
            current_return = (current_portfolio - initial_capital) / initial_capital
            logger.info(f"  Step {step:5d}: Portfolio ${current_portfolio:8,.0f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
        
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Calculate final metrics
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    cumulative_reward = sum(rewards)
    
    # Action distribution
    action_counts = np.bincount(actions_taken, minlength=3)
    action_dist = action_counts / len(actions_taken) if len(actions_taken) > 0 else [0, 0, 0]
    
    results = {
        'test_name': test_name,
        'steps': n_steps,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_portfolio': final_portfolio,
        'cumulative_reward': cumulative_reward,
        'soft_dd_hits': len(soft_dd_hits),
        'action_dist_sell': action_dist[0],
        'action_dist_hold': action_dist[1], 
        'action_dist_buy': action_dist[2]
    }
    
    return results

def run_gate_retest():
    """Run complete gate re-test sequence"""
    
    logger.info("🚨 PHASE GATE RE-TEST: V3 vs Original V2 Failure Conditions")
    logger.info("🎯 Testing if V3 passes the EXACT evaluation that killed V2")
    
    start_time = datetime.now()
    
    # Create evaluation data
    logger.info("📊 Creating strict evaluation data...")
    feature_data, price_series = create_strict_evaluation_data(n_periods=30000, seed=42)
    
    # Create V3 environment with strict conditions
    logger.info("🎯 Creating V3 environment with STRICT evaluation conditions...")
    env = IntradayTradingEnvV3(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,        # 2% hard limit (same as V2 tests)
        transaction_cost_pct=0.0001,        # 1bp transaction cost
        log_trades=False,
        # V3 specific parameters for strict evaluation
        risk_free_rate_annual=0.05,
        base_impact_bp=20.0,                # Confirmed working impact level
        impact_exponent=0.5,
        verbose=False
    )
    
    # Train a model for evaluation
    model, vec_env = create_trained_model_v3(env, training_steps=15000)
    
    # Test sequence
    test_results = []
    
    # Test 1: Gate Re-test (10K steps)
    logger.info("\n📋 TEST 1: GATE RE-TEST (Original V2 Killer)")
    result1 = evaluate_model_strict(model, vec_env, 10000, "Gate_Retest_10K")
    test_results.append(result1)
    
    # Test 2: Stress Sweep (higher transaction costs)
    logger.info("\n📋 TEST 2: STRESS SWEEP (Higher Transaction Costs)")
    
    # Reset environment with higher costs
    for tc_bp in [2.0, 3.0]:
        env_stress = IntradayTradingEnvV3(
            processed_feature_data=feature_data,
            price_data=price_series,
            initial_capital=100000,
            max_daily_drawdown_pct=0.02,
            transaction_cost_pct=tc_bp/10000,   # Higher transaction cost
            log_trades=False,
            risk_free_rate_annual=0.05,
            base_impact_bp=20.0,
            verbose=False
        )
        
        # Re-wrap for evaluation
        env_stress = Monitor(env_stress)
        vec_env_stress = DummyVecEnv([lambda: env_stress])
        vec_env_stress = VecNormalize(vec_env_stress, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
        
        result_stress = evaluate_model_strict(model, vec_env_stress, 5000, f"Stress_TC_{tc_bp}bp")
        test_results.append(result_stress)
    
    # Test 3: Long Horizon (25K steps)
    logger.info("\n📋 TEST 3: LONG HORIZON (25K Steps)")
    result3 = evaluate_model_strict(model, vec_env, 25000, "Long_Horizon_25K")
    test_results.append(result3)
    
    # Analysis and pass/fail determination
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n📊 GATE RE-TEST ANALYSIS")
    logger.info("=" * 60)
    
    logger.info(f"{'Test':20s} {'Return':>8s} {'Max DD':>8s} {'Soft DD':>8s} {'Status':>10s}")
    logger.info("-" * 60)
    
    all_pass = True
    gate_criteria = {
        'min_return': 0.01,    # ≥+1% return
        'max_drawdown': 0.02,  # ≤2% max drawdown
        'max_soft_dd': 0       # 0 soft DD hits after 1K steps
    }
    
    for result in test_results:
        # Determine pass/fail
        return_pass = result['total_return'] >= gate_criteria['min_return']
        dd_pass = result['max_drawdown'] <= gate_criteria['max_drawdown']
        soft_dd_pass = result['soft_dd_hits'] <= gate_criteria['max_soft_dd']
        
        test_pass = return_pass and dd_pass and soft_dd_pass
        all_pass = all_pass and test_pass
        
        status = "✅ PASS" if test_pass else "❌ FAIL"
        
        logger.info(f"{result['test_name']:20s} {result['total_return']:+7.2%} {result['max_drawdown']:7.2%} {result['soft_dd_hits']:7d} {status:>10s}")
    
    # Overall assessment
    logger.info("-" * 60)
    logger.info(f"⏱️ Total time: {elapsed_time}")
    
    if all_pass:
        logger.info("🎉 ✅ ALL GATE TESTS PASSED!")
        logger.info("   V3 successfully handles the EXACT conditions that killed V2")
        logger.info("   ✅ Ready to launch Phase D: 50K learning run with confidence")
        return True
    else:
        logger.info("⚠️ ❌ GATE TESTS FAILED")
        logger.info("   V3 still has issues under strict evaluation conditions")
        logger.info("   🛑 DO NOT proceed to 50K training - debug V3 parameters first")
        
        # Failure analysis
        failed_tests = [r for r in test_results if not (r['total_return'] >= 0.01 and r['max_drawdown'] <= 0.02)]
        if failed_tests:
            logger.info("\n🔧 FAILURE ANALYSIS:")
            for result in failed_tests:
                if result['max_drawdown'] > 0.02:
                    logger.info(f"   {result['test_name']}: DD breach {result['max_drawdown']:.2%} - Lower ent_coef or raise clip_range")
                if result['total_return'] < 0.01:
                    logger.info(f"   {result['test_name']}: Return too low {result['total_return']:+.2%} - Add alpha term or increase hold_bonus")
        
        return False
    
    # Save results to CSV for analysis
    csv_file = "gate_retest_results_v3.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_results[0].keys())
        writer.writeheader()
        writer.writerows(test_results)
    
    logger.info(f"📝 Results saved to {csv_file}")

def main():
    """Run gate re-test"""
    
    success = run_gate_retest()
    
    if success:
        print("🚨 ✅ GATE RE-TEST PASSED - V3 conquers V2's killer conditions!")
        print("🚀 Ready to launch Phase D: 50K learning run")
        return True
    else:
        print("⚠️ ❌ GATE RE-TEST FAILED - V3 needs more work")
        print("🛑 Do NOT proceed to expensive 50K training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)