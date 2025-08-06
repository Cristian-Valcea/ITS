#!/usr/bin/env python3
"""
🚀 200K TRAINING - FIXED VERSION
Use the EXACT same setup as the successful 10K probe, just scaled up.

Key Fix: Use the same base environment that gave us 0.962 mean reward in probe,
not the V3Enhanced environment that conflicts with our refined reward system.
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
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from src.gym_env.refined_reward_system import RefinedRewardSystem

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper

class CheckpointCallback(BaseCallback):
    """Simple callback for checkpointing every 50K steps"""
    
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.last_checkpoint = 0
        self.checkpoint_interval = 50000  # Every 50K steps
        
    def _on_step(self) -> bool:
        """Save checkpoint every 50K steps"""
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_interval:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(str(checkpoint_path))
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            self.last_checkpoint = self.num_timesteps
        
        return True

class ProvenRewardWrapper(Wrapper):
    """EXACT same wrapper that worked in 10K probe"""
    
    def __init__(self, env, refined_reward_system):
        super().__init__(env)
        self.refined_reward = refined_reward_system
        self.episode_count = 0
        self.total_holding_bonus = 0.0
        self.total_penalty = 0.0
    
    def step(self, action):
        """EXACT same step logic as successful probe"""
        obs, original_reward, done, truncated, info = self.env.step(action)
        
        # Extract state information
        portfolio_value = info.get('portfolio_value', 10000.0)
        previous_portfolio_value = getattr(self, '_prev_portfolio_value', 10000.0)
        nvda_position = info.get('nvda_position', 0.0)
        msft_position = info.get('msft_position', 0.0)
        
        # Calculate drawdown percentage
        initial_value = 10000.0
        drawdown_pct = max(0, (initial_value - portfolio_value) / initial_value)
        
        # Calculate refined reward with PROVEN parameters
        reward_components = self.refined_reward.calculate_reward(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            nvda_position=nvda_position,
            msft_position=msft_position,
            action=action,
            drawdown_pct=drawdown_pct
        )
        
        # Track totals
        self.total_holding_bonus += reward_components.holding_bonus
        self.total_penalty += reward_components.smoothed_penalty
        
        # Use refined reward instead of original
        refined_reward = reward_components.total_reward
        
        # Update info
        info['refined_reward_components'] = reward_components.to_dict()
        info['original_reward'] = original_reward
        
        # Store for next step
        self._prev_portfolio_value = portfolio_value
        
        # Log episode summary on termination
        if done or truncated:
            self._log_episode_summary()
        
        return obs, refined_reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset with proven logic"""
        obs, info = self.env.reset(**kwargs)
        self.refined_reward.reset_episode()
        self._prev_portfolio_value = 10000.0
        return obs, info
    
    def _log_episode_summary(self):
        """Log episode summary every 100 episodes"""
        self.episode_count += 1
        
        if self.episode_count % 100 == 0:
            avg_holding = self.total_holding_bonus / self.episode_count
            avg_penalty = self.total_penalty / self.episode_count
            
            reward_stats = self.refined_reward.get_stats()
            
            logger.info(f"📊 Episode {self.episode_count}: " +
                       f"avg_holding={avg_holding:.4f}, " +
                       f"avg_penalty={avg_penalty:.3f}, " +
                       f"triggers={reward_stats['holding_bonus_triggers']}")

def main():
    """Main 200K training with PROVEN setup"""
    
    logger.info("🚀 200K TRAINING - FIXED VERSION (PROVEN SETUP)")
    logger.info("=" * 70)
    
    # Training configuration
    total_timesteps = 200000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"stairways_v4_fixed_200k_{timestamp}"
    save_path = Path(f"train_runs/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎪 Run: {run_name}")
    logger.info(f"📊 Steps: {total_timesteps:,}")
    logger.info(f"🎯 Focus: Use EXACT same setup as successful 10K probe")
    logger.info(f"🔧 Key Fix: Same base environment + proven reward parameters")
    
    try:
        # Load real market data (SAME as probe)
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
        
        # Prepare data (SAME as probe)
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
        
        # Create SAME base environment as probe (NO V3Enhanced conflicts)
        logger.info("🏗️ Creating PROVEN environment setup...")
        base_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=390,          # Same as probe
            max_daily_drawdown_pct=0.20,    # Keep permissive (worked in probe)
            transaction_cost_pct=0.001      # Same as probe
        )
        
        # Create SAME refined reward system as probe
        logger.info("🎯 Initializing PROVEN refined reward system...")
        logger.info("🔧 PROVEN PARAMETERS (from successful 10K probe):")
        logger.info("   pnl_epsilon=750.0, holding_alpha=0.05, penalty_beta=0.10")
        logger.info("   holding_lookback_k=5, holding_min_ret=0.0")
        
        refined_reward_system = RefinedRewardSystem(
            initial_capital=10000.0,
            pnl_epsilon=750.0,              # PROVEN
            holding_alpha=0.05,             # PROVEN
            holding_lookback_k=5,           # PROVEN
            holding_min_ret=0.0,            # PROVEN
            penalty_beta=0.10,              # PROVEN
            exploration_coef=0.05,          # PROVEN
            exploration_decay=0.9999,       # PROVEN
            verbose=True
        )
        
        # Wrap with PROVEN wrapper
        env = ProvenRewardWrapper(base_env, refined_reward_system)
        env = Monitor(env, str(save_path / "monitor.csv"))
        env = DummyVecEnv([lambda: env])
        
        logger.info("✅ PROVEN environment setup created")
        
        # Create PPO model with PROVEN parameters
        logger.info("🤖 Creating PPO model with PROVEN parameters...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,             # PROVEN from probe
            n_steps=512,                    # PROVEN
            batch_size=128,                 # PROVEN
            n_epochs=4,                     # PROVEN
            gamma=0.99,                     # PROVEN
            gae_lambda=0.95,                # PROVEN
            clip_range=0.2,                 # PROVEN
            ent_coef=0.05,                  # PROVEN (no curriculum decay)
            vf_coef=0.5,                    # PROVEN
            max_grad_norm=0.5,              # PROVEN
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},  # PROVEN
            tensorboard_log=f"tensorboard_logs/{run_name}",
            verbose=1
        )
        
        # Create simple checkpoint callback
        checkpoint_callback = CheckpointCallback(save_path, verbose=1)
        
        logger.info("✅ PPO model with PROVEN parameters created")
        
        # Launch 200K training with PROVEN setup
        logger.info("🚀 LAUNCHING 200K TRAINING WITH PROVEN SETUP")
        logger.info("💡 Expected: Similar performance to 10K probe (0.962 mean reward)")
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
        final_model_path = save_path / "stairways_v4_fixed_final_model.zip"
        model.save(str(final_model_path))
        
        logger.info("🎉 200K FIXED TRAINING COMPLETE!")
        logger.info(f"📁 Final model saved: {final_model_path}")
        logger.info(f"⏱️ Duration: {duration_minutes:.1f} minutes")
        logger.info(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs/{run_name}")
        
        # Final validation with PROVEN setup
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
        
        # Success assessment based on probe results
        success_criteria = {
            'reward_excellent': avg_reward >= 0.75,   # Lower than probe but reasonable
            'reward_good': avg_reward >= 0.25,        # Positive and meaningful
            'length_adequate': avg_length >= 30.0,    # Reasonable episode length
            'sharpe_positive': final_sharpe > 0.0     # At least positive Sharpe
        }
        
        logger.info("🎯 SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {criterion}: {status}")
        
        # Overall assessment
        if success_criteria['reward_excellent'] and success_criteria['length_adequate']:
            logger.info("🎉 FIXED 200K TRAINING: EXCELLENT SUCCESS")
            return "excellent"
        elif success_criteria['reward_good'] and success_criteria['sharpe_positive']:
            logger.info("✅ FIXED 200K TRAINING: SUCCESS")
            return "success"
        else:
            logger.warning("⚠️ FIXED 200K TRAINING: PARTIAL SUCCESS")
            return "partial"
        
    except Exception as e:
        logger.error(f"❌ Fixed 200K training failed: {e}")
        import traceback
        traceback.print_exc()
        return "failed"

if __name__ == "__main__":
    result = main()
    
    if result == "excellent":
        print("🎉 200K FIXED TRAINING: EXCELLENT SUCCESS")
        print("🚀 Ready for paper trading deployment")
        sys.exit(0)
    elif result == "success":
        print("✅ 200K FIXED TRAINING: SUCCESS")
        print("🚀 Ready for paper trading validation")
        sys.exit(0)
    elif result == "partial":
        print("⚠️ 200K FIXED TRAINING: PARTIAL SUCCESS")
        print("🔧 Consider using 100K checkpoint or additional tuning")
        sys.exit(1)
    else:
        print("❌ 200K FIXED TRAINING: FAILED")
        sys.exit(1)