#!/usr/bin/env python3
"""
🚀 200K TRAINING WITH CLEAN REWARD SHIM
Uses the elegant shim approach to swap in RefinedRewardSystem while preserving
all V3Enhanced features. This should reproduce the 10K probe success at scale.

ARCHITECTURE:
- DualTickerTradingEnvV3CustomReward: Clean inheritance shim
- RefinedRewardSystem: Proven parameters from 10K probe
- No curriculum complexity: Just scale up what works
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from env_factory import make_env_with_refined_reward

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCheckpointCallback(BaseCallback):
    """Simple callback for checkpointing and basic quality gates"""
    
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.last_checkpoint = 0
        self.checkpoint_interval = 50000  # Every 50K steps
        
        # Simple quality gates
        self.early_gate_timestep = 50000
        self.final_gate_timestep = 200000
        
    def _on_step(self) -> bool:
        """Checkpoint and basic quality checks"""
        
        # Checkpoint saving
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_interval:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(str(checkpoint_path))
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            self.last_checkpoint = self.num_timesteps
            
            # Basic quality check at 50K
            if self.num_timesteps >= self.early_gate_timestep:
                self._check_early_quality()
        
        return True
    
    def _check_early_quality(self):
        """Basic quality check at 50K steps"""
        try:
            monitor_path = self.save_path / "monitor.csv"
            if monitor_path.exists():
                import pandas as pd
                df = pd.read_csv(monitor_path, skiprows=1)
                if len(df) > 50:
                    recent_rewards = df['r'].tail(50).values
                    mean_reward = np.mean(recent_rewards)
                    
                    logger.info(f"🎯 Early Quality Check (50K): Mean reward = {mean_reward:.3f}")
                    
                    if mean_reward >= 0.50:
                        logger.info("✅ Early gate PASSED - continuing to 200K")
                    elif mean_reward >= 0.0:
                        logger.info("⚠️ Early gate MARGINAL - monitoring closely")
                    else:
                        logger.warning("❌ Early gate FAILED - negative rewards detected")
        except Exception as e:
            logger.warning(f"Could not check early quality: {e}")

def main():
    """Main 200K training with clean shim approach"""
    
    logger.info("🚀 200K TRAINING WITH CLEAN REWARD SHIM")
    logger.info("=" * 70)
    
    # Training configuration
    total_timesteps = 200000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"stairways_v4_shim_200k_{timestamp}"
    save_path = Path(f"train_runs/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎪 Run: {run_name}")
    logger.info(f"📊 Steps: {total_timesteps:,}")
    logger.info(f"🎯 Architecture: V3Enhanced + RefinedRewardSystem shim")
    logger.info(f"🔧 Strategy: Scale up proven 10K probe setup + 30% drawdown cap")
    
    # Log expected gates
    logger.info("🎯 QUALITY GATES:")
    logger.info("   Early (50K):  ep_rew_mean ≥ 0.50")
    logger.info("   Final (200K): ep_rew_mean ≥ 1.00, Sharpe ≥ 0.35")
    
    try:
        # Load real market data (same as probe)
        logger.info("📈 Loading real market data...")
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data', 
            'user': 'postgres',
            'password': db_password
        }
        
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        market_data = adapter.load_training_data(
            start_date='2022-01-03',
            end_date='2024-12-31',
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        # Prepare data (same as probe)
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        # Combine features (26-dim observation)
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        position_features = np.zeros((combined_features.shape[0], 2))
        combined_features = np.concatenate([combined_features, position_features], axis=1)
        
        # Create 4-column price data
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        combined_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
        
        logger.info(f"📊 Data loaded: {len(trading_days)} timesteps")
        
        # Create environment using clean shim factory
        logger.info("🏭 Creating environment with clean reward shim...")
        logger.info("🎯 PROVEN PARAMETERS from 10K probe:")
        logger.info("   pnl_epsilon=750.0, holding_alpha=0.05, penalty_beta=0.10")
        logger.info("   holding_lookback_k=5, exploration_coef=0.05")
        
        # Use factory with proven parameters + increased drawdown cap
        env = make_env_with_refined_reward(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            max_daily_drawdown_pct=0.30,  # ← Increased from 0.20 to let episodes run longer
            # RefinedRewardSystem parameters (proven from probe)
            pnl_epsilon=750.0,
            holding_alpha=0.05,
            penalty_beta=0.10,
            exploration_coef=0.05,
            exploration_decay=0.9999,
            holding_lookback_k=5,
            holding_min_ret=0.0,
            verbose=True
        )
        
        # Wrap for monitoring
        env = Monitor(env, str(save_path / "monitor.csv"))
        env = DummyVecEnv([lambda: env])
        
        logger.info("✅ Clean shim environment created")
        logger.info("   V3Enhanced features: regime detection, controller, fills")
        logger.info("   Reward system: RefinedRewardSystem (proven parameters)")
        
        # Create PPO model with proven parameters
        logger.info("🤖 Creating PPO model with proven parameters...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,             # Proven from probe
            n_steps=512,                    # Proven
            batch_size=128,                 # Proven
            n_epochs=4,                     # Proven
            gamma=0.99,                     # Proven
            gae_lambda=0.95,                # Proven
            clip_range=0.2,                 # Proven
            ent_coef=0.05,                  # Proven (no decay)
            vf_coef=0.5,                    # Proven
            max_grad_norm=0.5,              # Proven
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=f"tensorboard_logs/{run_name}",
            verbose=1
        )
        
        # Create checkpoint callback
        checkpoint_callback = SimpleCheckpointCallback(save_path, verbose=1)
        
        logger.info("✅ PPO model with proven parameters created")
        
        # Launch 200K training
        logger.info("🚀 LAUNCHING 200K SHIM TRAINING")
        logger.info("💡 Expected: Reproduce 10K probe success (0.962 mean reward)")
        logger.info("⏱️ Expected duration: ~9-10 minutes @ 500 FPS")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=run_name
        )
        
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        # Save final model
        final_model_path = save_path / "stairways_v4_shim_final_model.zip"
        model.save(str(final_model_path))
        
        logger.info("🎉 200K SHIM TRAINING COMPLETE!")
        logger.info(f"📁 Final model saved: {final_model_path}")
        logger.info(f"⏱️ Duration: {duration_minutes:.1f} minutes")
        logger.info(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs/{run_name}")
        
        # Final validation
        logger.info("🧪 Running final validation...")
        
        # Quick validation episodes
        obs, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        reward_std = np.std(episode_rewards)
        final_sharpe = avg_reward / (reward_std + 1e-8)
        
        logger.info("📊 FINAL VALIDATION RESULTS:")
        logger.info(f"  Average Reward: {avg_reward:.3f}")
        logger.info(f"  Average Length: {avg_length:.1f} steps")
        logger.info(f"  Reward Std: {reward_std:.3f}")
        logger.info(f"  Final Sharpe: {final_sharpe:.3f}")
        
        # Success assessment
        success_criteria = {
            'reward_excellent': avg_reward >= 1.00,
            'reward_good': avg_reward >= 0.75,
            'reward_positive': avg_reward >= 0.25,
            'sharpe_excellent': final_sharpe >= 0.35,
            'sharpe_good': final_sharpe >= 0.25,
            'length_adequate': avg_length >= 30.0
        }
        
        logger.info("🎯 SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {criterion}: {status}")
        
        # Overall assessment
        if success_criteria['reward_excellent'] and success_criteria['sharpe_excellent']:
            logger.info("🎉 SHIM TRAINING: EXCELLENT SUCCESS")
            return "excellent"
        elif success_criteria['reward_good'] and success_criteria['sharpe_good']:
            logger.info("✅ SHIM TRAINING: SUCCESS")
            return "success"
        elif success_criteria['reward_positive']:
            logger.info("⚠️ SHIM TRAINING: PARTIAL SUCCESS")
            return "partial"
        else:
            logger.error("❌ SHIM TRAINING: FAILED")
            return "failed"
        
    except Exception as e:
        logger.error(f"❌ Shim training failed: {e}")
        import traceback
        traceback.print_exc()
        return "failed"

if __name__ == "__main__":
    result = main()
    
    if result == "excellent":
        print("🎉 200K SHIM TRAINING: EXCELLENT SUCCESS")
        print("🚀 Ready for paper trading deployment")
        sys.exit(0)
    elif result == "success":
        print("✅ 200K SHIM TRAINING: SUCCESS")
        print("🚀 Ready for paper trading validation")
        sys.exit(0)
    elif result == "partial":
        print("⚠️ 200K SHIM TRAINING: PARTIAL SUCCESS")
        print("🔧 Consider using 50K checkpoint or parameter tuning")
        sys.exit(1)
    else:
        print("❌ 200K SHIM TRAINING: FAILED")
        sys.exit(1)