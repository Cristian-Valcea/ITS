#!/usr/bin/env python3
"""
🔧 PHASE 1 DEBUG - Test Relaxed Environment
Test Phase 1 warmup with relaxed termination conditions
"""

import sys
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_setup():
    """Test that environment runs with relaxed settings"""
    
    logger.info("🔧 TESTING RELAXED ENVIRONMENT SETUP")
    logger.info("=" * 50)
    
    try:
        # Load config
        with open("config/ppo_200k_stairways_v4.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Database setup
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        # Load data (minimal for testing)
        logger.info("📈 Loading sample data...")
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        market_data = adapter.load_training_data(
            start_date='2022-01-03',
            end_date='2022-01-31',  # Just January for quick test
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        # Prepare data
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        position_features = np.zeros((combined_features.shape[0], 2))
        combined_features = np.concatenate([combined_features, position_features], axis=1)
        
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        combined_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
        
        logger.info(f"📊 Data loaded: {len(trading_days)} timesteps")
        
        # Create environment with VERY RELAXED settings
        logger.info("🏗️ Creating very relaxed environment...")
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=390,          # RELAXED: Full trading day
            max_daily_drawdown_pct=0.15,    # VERY RELAXED: 15% vs 2%
            transaction_cost_pct=0.001
        )
        
        logger.info("✅ Environment created successfully!")
        
        # Test a few episodes
        logger.info("🧪 Testing episode lengths...")
        episode_lengths = []
        
        for episode in range(3):
            obs, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            while not done and steps < 400:
                # Random action for testing
                action = env.action_space.sample()
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_lengths.append(steps)
            logger.info(f"Episode {episode + 1}: {steps} steps, total reward: {total_reward:.2f}")
        
        avg_length = np.mean(episode_lengths)
        logger.info(f"📊 Average episode length: {avg_length:.1f} steps")
        
        if avg_length > 50:
            logger.info("✅ Environment test PASSED - episodes are long enough for learning")
            return True
        else:
            logger.warning("⚠️ Episodes still too short - may need further relaxation")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_debug_phase1():
    """Run Phase 1 training with debug monitoring"""
    
    logger.info("🚀 PHASE 1 DEBUG TRAINING")
    logger.info("=" * 50)
    
    # Test environment first
    if not test_environment_setup():
        logger.error("❌ Environment test failed - aborting training")
        return False
    
    logger.info("🎯 Starting Phase 1 debug training (10K steps only)...")
    
    try:
        # Import trainer
        from train_200k_stairways_v4 import StairwaysV4Trainer
        
        # Create trainer
        trainer = StairwaysV4Trainer()
        
        # Create environment
        trainer.env = trainer._create_environment()
        
        # Create model for Phase 1
        phase1_config = trainer.config['phases']['phase_1_warmup']
        trainer.model = trainer._create_model(phase1_config)
        
        logger.info("🎪 Training Phase 1 (10K steps for debugging)...")
        
        # Train for 10K steps only
        trainer.model.learn(
            total_timesteps=10000,
            tb_log_name=f"debug_phase1_{trainer.timestamp}"
        )
        
        # Quick validation
        logger.info("🧪 Running validation...")
        results = trainer._run_validation(trainer.model, "debug_phase1")
        
        logger.info("📊 DEBUG RESULTS:")
        logger.info(f"  Average Reward: {results['avg_reward']:.3f}")
        logger.info(f"  Hold Rate: {results['hold_rate']:.1%}")
        
        if results['hold_rate'] < 0.9:  # Less than 90% hold
            logger.info("✅ Phase 1 debug PASSED - model is exploring!")
            return True
        else:
            logger.warning("⚠️ Model still too conservative - may need more relaxation")
            return False
            
    except Exception as e:
        logger.error(f"❌ Debug training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    
    success = run_debug_phase1()
    
    if success:
        print("✅ PHASE 1 DEBUG: SUCCESS - Ready for full 200K training")
        print("🚀 Launch full training with: python train_200k_stairways_v4.py")
    else:
        print("❌ PHASE 1 DEBUG: FAILED - Need further environment tuning")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)