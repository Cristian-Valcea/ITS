#!/usr/bin/env python3
"""
🔧 50K WARMUP TRAINING WITH REFINED REWARDS
Test refined reward system with:
- Normalized P&L rewards (bounded [-1,1]) 
- Stepwise holding bonuses
- Smooth penalty curves
- Higher exploration (ent_coef=0.05)
"""

import sys
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
from src.gym_env.refined_reward_system import RefinedRewardSystem

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper

class RefinedRewardWrapper(Wrapper):
    """Wrapper to integrate refined reward system into environment"""
    
    def __init__(self, env, refined_reward_system):
        super().__init__(env)
        self.refined_reward = refined_reward_system
    
    def step(self, action):
        """Step with refined reward calculation"""
        obs, original_reward, done, truncated, info = self.env.step(action)
        
        # Extract state information for refined reward calculation
        portfolio_value = info.get('portfolio_value', 10000.0)
        previous_portfolio_value = getattr(self, '_prev_portfolio_value', 10000.0)
        nvda_position = info.get('nvda_position', 0.0)
        msft_position = info.get('msft_position', 0.0)
        
        # Calculate drawdown percentage
        initial_value = 10000.0
        drawdown_pct = max(0, (initial_value - portfolio_value) / initial_value)
        
        # Calculate refined reward
        reward_components = self.refined_reward.calculate_reward(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            nvda_position=nvda_position,
            msft_position=msft_position,
            action=action,
            drawdown_pct=drawdown_pct
        )
        
        # Use refined reward instead of original
        refined_reward = reward_components.total_reward
        
        # Update info with reward breakdown
        info['refined_reward_components'] = reward_components.to_dict()
        info['original_reward'] = original_reward
        
        # Store for next step
        self._prev_portfolio_value = portfolio_value
        
        return obs, refined_reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and refined reward system"""
        obs, info = self.env.reset(**kwargs)
        self.refined_reward.reset_episode()
        self._prev_portfolio_value = 10000.0
        return obs, info

def main():
    """Main warmup training execution"""
    
    logger.info("🔧 50K WARMUP TRAINING WITH REFINED REWARDS")
    logger.info("=" * 60)
    
    # Training configuration
    total_timesteps = 50000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"warmup_refined_50k_{timestamp}"
    save_path = Path(f"train_runs/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎪 Run: {run_name}")
    logger.info(f"📊 Steps: {total_timesteps:,}")
    logger.info(f"🎯 Focus: Test refined reward system on real data")
    
    try:
        # Load real market data
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
        
        # Prepare data for environment
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
        
        # Create base environment with VERY RELAXED settings for warmup
        logger.info("🏗️ Creating warmup environment...")
        base_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=390,          # Full trading day
            max_daily_drawdown_pct=0.20,    # 20% very permissive
            transaction_cost_pct=0.001
        )
        
        # Create refined reward system
        logger.info("🎯 Initializing refined reward system...")
        refined_reward_system = RefinedRewardSystem(
            initial_capital=10000.0,
            pnl_epsilon=1000.0,             # Reasonable normalization
            holding_alpha=0.01,             # α ≈ 0.01 as suggested
            penalty_beta=0.5,               # β ≈ 0.5 as suggested  
            exploration_coef=0.05,          # Higher exploration for warmup
            exploration_decay=0.9999
        )
        
        # Wrap environment with refined rewards
        env = RefinedRewardWrapper(base_env, refined_reward_system)
        env = Monitor(env, str(save_path / "monitor.csv"))
        env = DummyVecEnv([lambda: env])
        
        logger.info("✅ Environment with refined rewards created")
        
        # Create PPO model with WARMUP configuration
        logger.info("🤖 Creating PPO model for warmup...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=512,                    # Proper learning iterations
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,                  # HIGH exploration for warmup
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=f"tensorboard_logs/{run_name}",
            verbose=1
        )
        
        logger.info("✅ PPO model created with high exploration")
        
        # Launch warmup training
        logger.info("🚀 Starting 50K warmup training...")
        logger.info("💡 Expected: Better episode rewards, longer episodes")
        logger.info("=" * 60)
        
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name
        )
        
        # Save model
        model_path = save_path / "warmup_refined_model.zip"
        model.save(str(model_path))
        
        logger.info("🎉 WARMUP TRAINING COMPLETE!")
        logger.info(f"📁 Model saved: {model_path}")
        logger.info(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs/{run_name}")
        
        # Quick validation
        logger.info("🧪 Running quick validation...")
        obs, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(3):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(f"Validation Episode {episode + 1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        logger.info("📊 VALIDATION RESULTS:")
        logger.info(f"  Average Reward: {avg_reward:.3f}")
        logger.info(f"  Average Length: {avg_length:.1f} steps")
        
        if avg_reward > -100 and avg_length > 10:
            logger.info("✅ WARMUP SUCCESS: Ready for full 200K training")
            return True
        else:
            logger.warning("⚠️ Warmup needs more tuning")
            return False
        
    except Exception as e:
        logger.error(f"❌ Warmup training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 50K WARMUP WITH REFINED REWARDS: SUCCESS")
        print("🚀 Ready for full 200K Stairways V4 training")
    else:
        print("❌ 50K WARMUP WITH REFINED REWARDS: NEEDS TUNING")
    
    sys.exit(0 if success else 1)