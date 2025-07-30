#!/usr/bin/env python3
"""
🎯 HPO PROGRESS MONITOR
Watch the HPO grid search progress and surface key metrics
"""

import sys
from pathlib import Path
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_hpo_status():
    """Check current HPO grid search status"""
    
    # Check for success file
    success_file = Path('winning_hpo_config.json')
    failure_file = Path('hpo_failure_analysis.json')
    
    if success_file.exists():
        with open(success_file, 'r') as f:
            config = json.load(f)
        
        logger.info("🎉 ✅ HPO SUCCESS FOUND!")
        logger.info(f"   Winning config: {config['config']}")
        logger.info(f"   Performance: {config['performance']['total_return']:+.2%} return, {config['performance']['max_drawdown']:.2%} DD")
        logger.info(f"   Training steps: {config['training_steps']}")
        logger.info(f"   Timestamp: {config['timestamp']}")
        
        return 'success', config
    
    elif failure_file.exists():
        with open(failure_file, 'r') as f:
            analysis = json.load(f)
        
        logger.info("⚠️ ❌ HPO GRID COMPLETED - NO SUCCESS")
        logger.info(f"   Configs tested: {analysis['configs_tested']}")
        logger.info(f"   Best return: {analysis['best_return']:+.2%}")
        logger.info(f"   All configs learned do-nothing or exceeded DD limit")
        
        return 'failure', analysis
    
    else:
        logger.info("🔄 HPO grid search still running...")
        return 'running', None

def monitor_hpo_loop():
    """Monitor HPO progress in a loop"""
    
    logger.info("🎯 HPO PROGRESS MONITOR")
    logger.info("   Watching for winning_hpo_config.json or hpo_failure_analysis.json")
    logger.info("   Will check every 30 seconds...")
    
    start_time = datetime.now()
    check_count = 0
    
    while True:
        check_count += 1
        elapsed = datetime.now() - start_time
        
        logger.info(f"\\n📊 Check #{check_count} (Elapsed: {elapsed})")
        
        status, data = check_hpo_status()
        
        if status == 'success':
            logger.info("\\n🚀 READY FOR NEXT STEP:")
            logger.info("   1. Export successful policy weights:")
            logger.info(f"      python export_policy.py --model_path <model_path> --output models/singleticker_gatepass.zip")
            logger.info("\\n   2. Launch dual-ticker warm-start:")
            logger.info("      python dual_ticker_warmstart.py")
            break
            
        elif status == 'failure':
            logger.info("\\n🔬 SUGGESTED NEXT STEPS:")
            logger.info("   1. Try synthetic alpha burst test:")
            logger.info("      python synthetic_alpha_test.py")
            logger.info("\\n   2. Expand HPO grid with more aggressive parameters")
            logger.info("\\n   3. Stay single-ticker until RL learning loop solved")
            break
            
        else:  # still running
            logger.info("   Grid search continues...")
            time.sleep(30)  # Check every 30 seconds

def create_synthetic_alpha_test():
    """Create obvious synthetic alpha test to verify agent can ever deviate from cash"""
    
    synthetic_test = '''#!/usr/bin/env python3
"""
🎯 SYNTHETIC ALPHA TEST
Inject obvious alpha bursts to verify agent can learn to trade at all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3

def create_obvious_alpha_data(n_periods: int = 2000):
    """Create data with OBVIOUS alpha bursts every N steps"""
    
    np.random.seed(42)
    
    # Create price series with obvious predictable patterns
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    prices = []
    alpha_signals = []
    
    for i in range(n_periods):
        # Create OBVIOUS sine-wave alpha with 0.3 magnitude every 50 steps
        if i % 50 < 25:
            # Strong predictable bull run
            price_change = 0.005 + np.random.normal(0, 0.002)  # +0.5% plus small noise
            alpha_signal = 0.3  # OBVIOUS bullish signal
        else:
            # Strong predictable bear run  
            price_change = -0.005 + np.random.normal(0, 0.002)  # -0.5% plus small noise
            alpha_signal = -0.3  # OBVIOUS bearish signal
        
        if i == 0:
            price = base_price
        else:  
            price = prices[-1] * (1 + price_change)
        
        prices.append(price)
        alpha_signals.append(alpha_signal)
    
    # Create features: 12 random + obvious alpha
    random_features = np.random.randn(n_periods, 12).astype(np.float32)
    alpha_feature = np.array(alpha_signals).reshape(-1, 1).astype(np.float32)
    features = np.hstack([random_features, alpha_feature])
    
    price_series = pd.Series(prices, index=trading_days)
    
    logger.info(f"🎯 OBVIOUS synthetic alpha data created:")
    logger.info(f"   Pattern: 25 bull steps (+0.5%) → 25 bear steps (-0.5%)")
    logger.info(f"   Alpha magnitude: ±0.3 (OBVIOUS)")
    logger.info(f"   If agent can't learn this, problem is in RL settings not alpha strength")
    
    return features, price_series

def test_obvious_alpha():
    """Test if agent can learn OBVIOUS alpha pattern"""
    
    logger.info("🧪 SYNTHETIC ALPHA TEST - OBVIOUS Pattern")
    
    # Create OBVIOUS alpha data
    features, prices = create_obvious_alpha_data(2000)
    
    # Create environment
    env = IntradayTradingEnvV3(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        base_impact_bp=50.0,  # Even lower impact for easier learning
        verbose=False
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
    
    # Create model with AGGRESSIVE exploration
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=1e-3,      # Much higher LR
        n_steps=1024,
        batch_size=64,
        n_epochs=10,             # More epochs
        gamma=0.99,              # Shorter horizon
        gae_lambda=0.95,
        clip_range=0.3,          # Larger clip range
        ent_coef=0.1,            # VERY high exploration
        vf_coef=0.5,
        max_grad_norm=1.0,       # Larger gradient norm
        verbose=1,
        seed=42,
        device="auto"
    )
    
    logger.info("🔥 Training with AGGRESSIVE settings on OBVIOUS alpha...")
    model.learn(total_timesteps=30000, progress_bar=True)
    
    # Test if it learned ANYTHING
    obs = vec_env.reset()
    actions = []
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, _, done, _ = vec_env.step(action)
        if done[0]:
            break
    
    # Analysis
    action_counts = np.bincount(actions, minlength=3)
    hold_freq = action_counts[1] / len(actions)
    
    logger.info(f"\\n📊 SYNTHETIC ALPHA TEST RESULTS:")
    logger.info(f"   HOLD frequency: {hold_freq:.1%}")
    logger.info(f"   Trading happened: {'YES' if hold_freq < 0.9 else 'NO'}")
    
    if hold_freq < 0.9:
        logger.info("✅ Agent learned to trade on OBVIOUS alpha")
        logger.info("   Problem was alpha strength, not RL settings")
    else:
        logger.info("❌ Agent still won't trade even with OBVIOUS alpha")
        logger.info("   Problem is in RL hyperparameters or search space")
    
    return hold_freq < 0.9

if __name__ == "__main__":
    success = test_obvious_alpha()
    sys.exit(0 if success else 1)
'''
    
    with open('synthetic_alpha_test.py', 'w') as f:
        f.write(synthetic_test)
    
    logger.info("✅ Created synthetic_alpha_test.py")

if __name__ == "__main__":
    # Create synthetic test script for contingency
    create_synthetic_alpha_test()
    
    # Start monitoring
    monitor_hpo_loop()