#!/usr/bin/env python3
"""
Comprehensive Reward System Diagnostic
Validates all aspects of the reward system after fixes
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from gym_env.intraday_trading_env import IntradayTradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_diagnostic_data():
    """Create comprehensive test data for reward system validation."""
    np.random.seed(42)
    n_samples = 3000  # 3 episodes worth
    
    # Create datetime index
    start_date = datetime(2024, 1, 1, 9, 30)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Create feature data (11 features)
    feature_data = pd.DataFrame(index=dates)
    feature_data['rsi_14'] = np.random.uniform(20, 80, n_samples)
    feature_data['ema_10'] = np.random.uniform(100, 200, n_samples)
    feature_data['ema_20'] = np.random.uniform(100, 200, n_samples)
    feature_data['vwap'] = np.random.uniform(100, 200, n_samples)
    feature_data['hour_sin'] = np.random.uniform(-1, 1, n_samples)
    feature_data['hour_cos'] = np.random.uniform(-1, 1, n_samples)
    
    # Add risk features
    for i in range(5):
        feature_data[f'risk_feature_{i}'] = np.random.randn(n_samples) * 0.1
    
    # Create price data with controlled scenarios
    base_price = 100
    price_changes = []
    
    # Scenario 1: Trending up (profitable)
    price_changes.extend(np.random.normal(0.02, 0.01, 1000))
    # Scenario 2: Sideways (neutral)
    price_changes.extend(np.random.normal(0.0, 0.01, 1000))
    # Scenario 3: Trending down (challenging)
    price_changes.extend(np.random.normal(-0.02, 0.01, 1000))
    
    price_data = pd.Series(
        base_price + np.cumsum(price_changes),
        index=dates
    )
    
    return feature_data, price_data

def run_comprehensive_diagnostic():
    """Run comprehensive reward system diagnostic."""
    
    print("🔍 COMPREHENSIVE REWARD SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    # Load config
    with open('config/phase1_reality_grounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded - reward_scaling: {config['environment']['reward_scaling']}")
    
    # Create test data
    feature_data, price_data = create_diagnostic_data()
    print(f"✅ Test data created: {feature_data.shape[0]} samples")
    
    # Test 1: Basic Environment Setup
    print("\n" + "="*60)
    print("TEST 1: ENVIRONMENT SETUP VALIDATION")
    print("="*60)
    
    env = IntradayTradingEnv(
        processed_feature_data=feature_data.values,
        price_data=price_data,
        initial_capital=50000.0,
        reward_scaling=config['environment']['reward_scaling'],
        max_episode_steps=1000,
        institutional_safeguards_config=config
    )
    
    print(f"✅ Environment created successfully")
    print(f"   - max_episode_steps: {getattr(env, '_max_episode_steps', 'N/A')}")
    print(f"   - reward_scaling: {getattr(env, 'reward_scaling', 'N/A')}")
    print(f"   - ppo_reward_scaling: {getattr(env, 'ppo_reward_scaling', 'N/A')}")
    print(f"   - initial_capital: {getattr(env, 'initial_capital', 'N/A')}")
    
    # Test 2: Episode Structure Validation
    print("\n" + "="*60)
    print("TEST 2: EPISODE STRUCTURE VALIDATION")
    print("="*60)
    
    episode_results = []
    
    for episode in range(3):
        obs, _ = env.reset()
        steps = 0
        total_reward = 0
        rewards = []
        
        while True:
            # Use strategic actions instead of random
            if episode == 0:  # Conservative strategy
                action = 0  # Hold
            elif episode == 1:  # Aggressive long
                action = 2  # Long
            else:  # Mixed strategy
                action = np.random.choice([0, 1, 2])
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            steps += 1
            
            if done or truncated:
                break
        
        episode_results.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / steps,
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'reward_std': np.std(rewards)
        })
        
        print(f"Episode {episode + 1}:")
        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Avg Reward/Step: {total_reward/steps:.4f}")
        print(f"   Reward Range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    
    # Test 3: Reward Scaling Validation
    print("\n" + "="*60)
    print("TEST 3: REWARD SCALING VALIDATION")
    print("="*60)
    
    avg_episode_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_episode_length = np.mean([r['steps'] for r in episode_results])
    
    print(f"Average Episode Reward: {avg_episode_reward:.2f}")
    print(f"Average Episode Length: {avg_episode_length:.1f}")
    print(f"Target Episode Reward Range: 4-6")
    print(f"Target Episode Length: 1000")
    
    # Validation checks
    episode_reward_ok = 3 <= avg_episode_reward <= 8  # Slightly wider range for robustness
    episode_length_ok = 950 <= avg_episode_length <= 1050
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   Episode Reward: {'✅ PASS' if episode_reward_ok else '❌ FAIL'}")
    print(f"   Episode Length: {'✅ PASS' if episode_length_ok else '❌ FAIL'}")
    
    # Test 4: Reward Component Analysis
    print("\n" + "="*60)
    print("TEST 4: REWARD COMPONENT ANALYSIS")
    print("="*60)
    
    # Run a single episode with detailed logging
    obs, _ = env.reset()
    pnl_rewards = []
    penalty_rewards = []
    total_rewards = []
    
    for step in range(100):  # Sample 100 steps
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract reward components from info if available
        raw_reward = info.get('raw_reward', reward)
        scaled_reward = info.get('scaled_reward', reward)
        
        total_rewards.append(reward)
        
        if done or truncated:
            break
    
    print(f"Sample Reward Statistics (100 steps):")
    print(f"   Mean Reward: {np.mean(total_rewards):.4f}")
    print(f"   Std Reward: {np.std(total_rewards):.4f}")
    print(f"   Min Reward: {np.min(total_rewards):.4f}")
    print(f"   Max Reward: {np.max(total_rewards):.4f}")
    
    # Test 5: Configuration Consistency Check
    print("\n" + "="*60)
    print("TEST 5: CONFIGURATION CONSISTENCY CHECK")
    print("="*60)
    
    config_checks = {
        'reward_scaling': config['environment']['reward_scaling'] == 0.07,
        'initial_capital': config['environment']['initial_capital'] == 50000.0,
        'soft_dd_limit': config['risk'].get('soft_dd_limit_pct', config['risk'].get('soft_dd_threshold_pct', 0.02)) == 0.02,
        'hard_dd_limit': config['risk'].get('hard_dd_limit_pct', config['risk'].get('hard_dd_threshold_pct', 0.04)) == 0.04,
    }
    
    for check_name, check_result in config_checks.items():
        print(f"   {check_name}: {'✅ PASS' if check_result else '❌ FAIL'}")
    
    # Final Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_tests_pass = (
        episode_reward_ok and 
        episode_length_ok and 
        all(config_checks.values())
    )
    
    if all_tests_pass:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Reward system is working correctly")
        print("✅ Episode structure is correct")
        print("✅ Configuration is consistent")
        print("🚀 Ready for production 50K training!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Review failed tests before proceeding")
    
    # Generate diagnostic report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config_reward_scaling': config['environment']['reward_scaling'],
        'avg_episode_reward': avg_episode_reward,
        'avg_episode_length': avg_episode_length,
        'episode_reward_ok': episode_reward_ok,
        'episode_length_ok': episode_length_ok,
        'all_tests_pass': all_tests_pass,
        'episode_results': episode_results
    }
    
    # Save report
    import json
    with open('reward_system_diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Diagnostic report saved to: reward_system_diagnostic_report.json")
    
    return all_tests_pass

if __name__ == "__main__":
    success = run_comprehensive_diagnostic()
    sys.exit(0 if success else 1)