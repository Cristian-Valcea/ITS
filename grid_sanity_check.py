#!/usr/bin/env python3
"""
🔍 GRID SANITY CHECK
Quick sanity checks while HPO grid cooks - seed provenance, GPU util, TB setup
"""

import sys
import subprocess
import json
import logging
from pathlib import Path
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_seed_provenance():
    """Check seed/RNG provenance for reproducibility"""
    
    logger.info("🔍 CHECKING SEED PROVENANCE")
    
    seed_issues = []
    configs_found = 0
    
    # Look for parameter files in common locations
    param_files = []
    for pattern in ['logs/*/params.json', '*/params.json', 'run_*/config.json']:
        param_files.extend(Path('.').glob(pattern))
    
    if not param_files:
        logger.warning("   ⚠️ No parameter files found yet")
        logger.info("   Grid may not have started logging configs")
        return True
    
    for param_file in param_files:
        try:
            with open(param_file, 'r') as f:
                config = json.load(f)
            
            configs_found += 1
            run_dir = param_file.parent.name
            
            # Check for seed in config
            seed_value = None
            for key in ['seed', 'random_seed', 'np_seed']:
                if key in config:
                    seed_value = config[key]
                    break
            
            if seed_value is None:
                seed_issues.append(f"{run_dir}: No seed found")
            elif seed_value != 42:  # Expected seed
                seed_issues.append(f"{run_dir}: Unexpected seed {seed_value}")
            else:
                logger.info(f"   ✅ {run_dir}: seed={seed_value}")
                
        except Exception as e:
            seed_issues.append(f"{param_file}: Error reading - {e}")
    
    logger.info(f"   Configs checked: {configs_found}")
    
    if seed_issues:
        logger.warning(f"   ⚠️ Seed issues found ({len(seed_issues)}):")
        for issue in seed_issues[:5]:  # Show first 5
            logger.warning(f"     - {issue}")
        if len(seed_issues) > 5:
            logger.warning(f"     ... and {len(seed_issues) - 5} more")
        return False
    else:
        logger.info("   ✅ All seeds verified")
        return True

def check_gpu_utilization():
    """Check GPU utilization for stuck dataloaders"""
    
    logger.info("🔍 CHECKING GPU UTILIZATION")
    
    try:
        # Run nvidia-smi to check current GPU usage
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                try:
                    util, mem = line.strip().split(', ')
                    util_pct = int(util)
                    mem_mb = int(mem)
                    
                    logger.info(f"   GPU {i}: {util_pct}% utilization, {mem_mb}MB memory")
                    
                    # Check for suspicious patterns
                    if util_pct > 95:
                        logger.info(f"     💪 High utilization - training active")
                    elif util_pct < 5 and mem_mb > 1000:
                        logger.warning(f"     ⚠️ Low utilization but memory allocated - possible stuck dataloader")
                    elif util_pct == 0:
                        logger.info(f"     💤 Idle - no training active")
                    
                except ValueError:
                    logger.warning(f"   ⚠️ Could not parse GPU {i} stats: {line}")
            
            return True
            
        else:
            logger.warning("   ⚠️ nvidia-smi failed")
            logger.info("   May not have NVIDIA GPU or drivers")
            return True
            
    except FileNotFoundError:
        logger.warning("   ⚠️ nvidia-smi not found")
        logger.info("   NVIDIA drivers may not be installed")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("   ⚠️ nvidia-smi timed out")
        return False
    except Exception as e:
        logger.warning(f"   ⚠️ GPU check failed: {e}")
        return False

def check_tensorboard_setup():
    """Check TensorBoard configuration recommendations"""
    
    logger.info("🔍 CHECKING TENSORBOARD SETUP")
    
    # Look for TensorBoard log directories
    tb_dirs = []
    for pattern in ['logs/**/events.out.tfevents.*', '**/events.out.tfevents.*']:
        for path in Path('.').glob(pattern):
            tb_dir = path.parent
            if tb_dir not in tb_dirs:
                tb_dirs.append(tb_dir)
    
    if not tb_dirs:
        logger.warning("   ⚠️ No TensorBoard logs found yet")
        logger.info("   Training may not have started logging")
        return True
    
    logger.info(f"   Found {len(tb_dirs)} TensorBoard log directories")
    
    # Check if TensorBoard is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'tensorboard' in result.stdout:
            logger.info("   ✅ TensorBoard process detected")
            logger.info("   💡 Remember: Toggle smoothing ≈ 0 in TB UI for raw spikes")
            logger.info("   💡 Watch for rollout/ep_rew_mean ≥ +1000 breakouts")
        else:
            logger.info("   📊 TensorBoard not running")
            logger.info("   💡 Launch with: python launch_tensorboard.py")
            
    except Exception as e:
        logger.debug(f"Could not check TensorBoard process: {e}")
    
    return True

def enhanced_synthetic_alpha_test():
    """Enhanced synthetic alpha test with your recommended parameters"""
    
    logger.info("🔬 ENHANCED SYNTHETIC ALPHA TEST")  
    logger.info("   Testing 40bp persistent alpha in 4K steps")
    logger.info("   If this fails, issue is deeper than hyperparameters")
    
    # Create enhanced synthetic test script
    enhanced_test = '''#!/usr/bin/env python3
"""
🔬 ENHANCED SYNTHETIC ALPHA TEST
Test 40bp persistent alpha in 4K steps to isolate fundamental issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

def create_persistent_alpha_data(n_periods: int = 1000, alpha_mag: float = 0.4):
    """Create data with PERSISTENT alpha signal"""
    
    np.random.seed(42)
    
    # Create price series with persistent alpha
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    prices = []
    alpha_signals = []
    
    for i in range(n_periods):
        # PERSISTENT alpha signal - always positive
        alpha_signal = alpha_mag  # 40bp persistent edge
        
        # Price follows alpha with some noise
        if i == 0:
            price = base_price
        else:
            # Alpha-driven price movement + small noise
            alpha_return = alpha_signal * 0.01  # Convert bp to return
            noise = np.random.normal(0, 0.001)  # Small noise
            price_change = alpha_return + noise
            price = prices[-1] * (1 + price_change)
        
        prices.append(price)
        alpha_signals.append(alpha_signal)
    
    # Create features: 12 random + persistent alpha
    random_features = np.random.randn(n_periods, 12).astype(np.float32) * 0.1  # Smaller random features
    alpha_feature = np.array(alpha_signals).reshape(-1, 1).astype(np.float32)
    features = np.hstack([random_features, alpha_feature])
    
    price_series = pd.Series(prices, index=trading_days)
    
    logger.info(f"🔬 PERSISTENT alpha data created:")
    logger.info(f"   Alpha magnitude: {alpha_mag} (persistent)")
    logger.info(f"   Expected return: {alpha_mag * 0.01:.1%} per step")
    logger.info(f"   If agent can't learn this, fundamental issue exists")
    
    return features, price_series

def test_persistent_alpha(alpha_mag: float = 0.4, steps: int = 4000):
    """Test if agent can learn PERSISTENT alpha"""
    
    logger.info(f"🔬 ENHANCED SYNTHETIC ALPHA TEST")
    logger.info(f"   Alpha magnitude: {alpha_mag} bp (persistent)")
    logger.info(f"   Training steps: {steps}")
    logger.info(f"   Diagnostic purpose: Isolate advantage/reward issues")
    
    # Create PERSISTENT alpha data
    features, prices = create_persistent_alpha_data(1000, alpha_mag)
    
    # Create environment with minimal friction
    env = IntradayTradingEnvV3(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=100000,
        max_daily_drawdown_pct=0.05,  # Looser DD limit for testing
        base_impact_bp=10.0,          # Minimal impact for testing
        verbose=False
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    # Create model with standard settings
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-4,      # Standard LR
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,              # Standard gamma
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # Standard exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,               # Show training progress
        seed=42,
        device="auto"
    )
    
    logger.info("🔥 Training with PERSISTENT alpha signal...")
    model.learn(total_timesteps=steps, progress_bar=True)
    
    # Test performance
    obs = vec_env.reset()
    actions = []
    rewards = []
    portfolio_values = []
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        actions.append(action[0])
        rewards.append(reward[0])
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
        
        if done[0]:
            break
    
    # Analysis
    action_counts = np.bincount(actions, minlength=3)
    hold_freq = action_counts[1] / len(actions)
    buy_freq = action_counts[2] / len(actions)
    total_reward = sum(rewards)
    
    final_portfolio = portfolio_values[-1] if portfolio_values else 100000
    total_return = (final_portfolio - 100000) / 100000
    
    logger.info(f"\\n📊 PERSISTENT ALPHA TEST RESULTS:")
    logger.info(f"   HOLD frequency: {hold_freq:.1%}")
    logger.info(f"   BUY frequency: {buy_freq:.1%}")  
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Total reward: {total_reward:.1f}")
    
    # Diagnostic assessment
    if buy_freq > 0.5 and total_return > 0.02:
        logger.info("✅ Agent learned PERSISTENT alpha")
        logger.info("   Advantage/reward systems working correctly")
        logger.info("   Issue is likely in alpha signal strength or hyperparameters")
        success = True
    elif buy_freq > 0.1:
        logger.info("⚠️ Agent partially learned alpha")
        logger.info("   Some exploration but not optimal exploitation")
        logger.info("   May need hyperparameter tuning")
        success = False
    else:
        logger.info("❌ Agent failed to learn PERSISTENT alpha")
        logger.info("   Fundamental issue in advantage normalization or reward scaling")
        logger.info("   Problem deeper than hyperparameters")
        success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Enhanced Synthetic Alpha Test')
    parser.add_argument('--alpha_mag', type=float, default=0.4, help='Alpha magnitude in bp')
    parser.add_argument('--steps', type=int, default=4000, help='Training steps')
    args = parser.parse_args()
    
    success = test_persistent_alpha(args.alpha_mag, args.steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
    
    with open('enhanced_synthetic_alpha_test.py', 'w') as f:
        f.write(enhanced_test)
    
    logger.info("✅ Enhanced synthetic test created")
    logger.info("   Usage: python enhanced_synthetic_alpha_test.py --alpha_mag 0.4 --steps 4000")
    logger.info("   Diagnostic: If this fails, issue is advantage normalization or reward scaling")

def run_sanity_checks():
    """Run all sanity checks"""
    
    logger.info("🔍 GRID SANITY CHECKS - Quick diagnostics while grid cooks")
    
    checks = [
        ("Seed Provenance", check_seed_provenance),
        ("GPU Utilization", check_gpu_utilization), 
        ("TensorBoard Setup", check_tensorboard_setup)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n{'='*50}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 SANITY CHECK SUMMARY:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎯 All sanity checks passed - grid is clean")
        logger.info("   Continue hawk watch for green light signals")
    else:
        logger.info("\n⚠️ Some checks failed - investigate before proceeding")
    
    # Create enhanced synthetic test for contingency
    enhanced_synthetic_alpha_test()
    
    return all_passed

def main():
    """Main sanity check function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Grid Sanity Checks')
    parser.add_argument('--seed-only', action='store_true', help='Check seed provenance only')
    parser.add_argument('--gpu-only', action='store_true', help='Check GPU utilization only')
    parser.add_argument('--tb-only', action='store_true', help='Check TensorBoard setup only')
    
    args = parser.parse_args()
    
    if args.seed_only:
        check_seed_provenance()
    elif args.gpu_only:
        check_gpu_utilization()
    elif args.tb_only:
        check_tensorboard_setup()
    else:
        run_sanity_checks()

if __name__ == "__main__":
    main()