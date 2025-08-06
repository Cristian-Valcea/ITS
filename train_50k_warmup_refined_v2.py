#!/usr/bin/env python3
"""
🔧 50K WARMUP TRAINING WITH REFINED REWARDS V2
Based on diagnostic analysis - targeted parameter fixes:
- penalty_beta: 0.5 → 0.15 (70% reduction)
- holding_alpha: 0.01 → 0.03 (3x boost)  
- pnl_epsilon: 1000.0 → 750.0 (increased sensitivity)

Expected improvements:
- ep_rew_mean: toward ≥ 0.10 (from -0.877)
- smoothed_penalty sum: ≈ -25 to -30 (from -90.642)
- holding_bonus sum: positive & growing (from 0.000)
- Episode length: ≥ 17 steps (from 14.4)
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
        # Track reward components for monitoring
        self.episode_components = {
            'normalized_pnl': [],
            'holding_bonus': [],
            'smoothed_penalty': [],
            'exploration_bonus': [],
            'directional_bonus': []
        }
    
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
        
        # Track components for episode summary
        for component in self.episode_components:
            self.episode_components[component].append(getattr(reward_components, component))
        
        # Use refined reward instead of original
        refined_reward = reward_components.total_reward
        
        # Update info with reward breakdown
        info['refined_reward_components'] = reward_components.to_dict()
        info['original_reward'] = original_reward
        
        # Store for next step
        self._prev_portfolio_value = portfolio_value
        
        # Log episode summary on termination
        if done or truncated:
            self._log_episode_summary()
            self._reset_episode_tracking()
        
        return obs, refined_reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and refined reward system"""
        obs, info = self.env.reset(**kwargs)
        self.refined_reward.reset_episode()
        self._prev_portfolio_value = 10000.0
        self._reset_episode_tracking()
        return obs, info
    
    def _log_episode_summary(self):
        """Log episode-level reward component summary"""
        if not any(self.episode_components.values()):
            return
        
        summary = {}
        for component, values in self.episode_components.items():
            if values:
                summary[f'{component}_sum'] = sum(values)
                summary[f'{component}_mean'] = np.mean(values)
        
        logger.info(f"📊 Episode Components: " + 
                   f"penalty_sum={summary.get('smoothed_penalty_sum', 0):.3f}, " +
                   f"holding_sum={summary.get('holding_bonus_sum', 0):.3f}, " +
                   f"exploration_sum={summary.get('exploration_bonus_sum', 0):.3f}")
    
    def _reset_episode_tracking(self):
        """Reset episode component tracking"""
        for component in self.episode_components:
            self.episode_components[component] = []

def main():
    """Main warmup training execution with refined parameters v2"""
    
    logger.info("🔧 50K WARMUP TRAINING WITH REFINED REWARDS V2")
    logger.info("=" * 60)
    
    # Training configuration
    total_timesteps = 50000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"warmup_refined_v2_50k_{timestamp}"
    save_path = Path(f"train_runs/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎪 Run: {run_name}")
    logger.info(f"📊 Steps: {total_timesteps:,}")
    logger.info(f"🎯 Focus: Diagnostic-driven parameter fixes")
    logger.info(f"🔧 Changes: penalty_beta=0.15, holding_alpha=0.03, pnl_epsilon=750")
    
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
        
        # Create base environment with same settings as v1
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
        
        # Create refined reward system V2 with diagnostic-driven fixes
        logger.info("🎯 Initializing refined reward system V2...")
        logger.info("🔧 PARAMETER CHANGES:")
        logger.info("   penalty_beta: 0.5 → 0.15 (70% reduction)")
        logger.info("   holding_alpha: 0.01 → 0.03 (3x boost)")
        logger.info("   pnl_epsilon: 1000.0 → 750.0 (increased sensitivity)")
        
        refined_reward_system = RefinedRewardSystem(
            initial_capital=10000.0,
            pnl_epsilon=750.0,              # ↑ sensitivity (was 1000.0)
            holding_alpha=0.03,             # ↑ holding incentive 3x (was 0.01)
            penalty_beta=0.15,              # ↓ penalty 70% (was 0.5)
            exploration_coef=0.05,          # Same as v1
            exploration_decay=0.9999        # Same as v1
        )
        
        # Wrap environment with refined rewards
        env = RefinedRewardWrapper(base_env, refined_reward_system)
        env = Monitor(env, str(save_path / "monitor.csv"))
        env = DummyVecEnv([lambda: env])
        
        logger.info("✅ Environment with refined rewards V2 created")
        
        # Create PPO model with same configuration as v1
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
        
        # Launch warmup training V2
        logger.info("🚀 Starting 50K warmup training V2...")
        logger.info("💡 Expected improvements:")
        logger.info("   ep_rew_mean: toward ≥ 0.10 (from -0.877)")
        logger.info("   smoothed_penalty sum: ≈ -25 to -30 (from -90.642)")
        logger.info("   holding_bonus sum: positive & growing (from 0.000)")
        logger.info("   Episode length: ≥ 17 steps (from 14.4)")
        logger.info("=" * 60)
        
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name
        )
        
        # Save model
        model_path = save_path / "warmup_refined_v2_model.zip"
        model.save(str(model_path))
        
        logger.info("🎉 WARMUP TRAINING V2 COMPLETE!")
        logger.info(f"📁 Model saved: {model_path}")
        logger.info(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs/{run_name}")
        
        # Quick validation with component tracking
        logger.info("🧪 Running quick validation with component tracking...")
        obs, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        component_sums = {
            'normalized_pnl': 0,
            'holding_bonus': 0,
            'smoothed_penalty': 0,
            'exploration_bonus': 0,
            'directional_bonus': 0
        }
        
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
                
                # Track components if available
                if hasattr(env.envs[0], 'refined_reward'):
                    # This is a simplified tracking - full tracking is in the wrapper
                    pass
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(f"Validation Episode {episode + 1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        logger.info("📊 VALIDATION RESULTS V2:")
        logger.info(f"  Average Reward: {avg_reward:.3f}")
        logger.info(f"  Average Length: {avg_length:.1f} steps")
        
        # Success criteria based on diagnostic targets
        success_criteria = {
            'reward_positive': avg_reward > 0.00,
            'reward_target': avg_reward >= 0.10,
            'length_improved': avg_length >= 15.0,
            'length_target': avg_length >= 17.0
        }
        
        logger.info("🎯 SUCCESS CRITERIA CHECK:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {criterion}: {status}")
        
        # Overall assessment
        critical_passes = success_criteria['reward_positive'] and success_criteria['length_improved']
        target_passes = success_criteria['reward_target'] and success_criteria['length_target']
        
        if target_passes:
            logger.info("🎉 WARMUP V2 EXCELLENT: Ready for full 200K training")
            return "excellent"
        elif critical_passes:
            logger.info("✅ WARMUP V2 SUCCESS: Significant improvement, ready for 200K")
            return "success"
        else:
            logger.warning("⚠️ WARMUP V2 PARTIAL: Some improvement, may need more tuning")
            return "partial"
        
    except Exception as e:
        logger.error(f"❌ Warmup training V2 failed: {e}")
        import traceback
        traceback.print_exc()
        return "failed"

if __name__ == "__main__":
    result = main()
    
    if result == "excellent":
        print("🎉 50K WARMUP V2 WITH REFINED REWARDS: EXCELLENT")
        print("🚀 Ready for full 200K Stairways V4 training")
        sys.exit(0)
    elif result == "success":
        print("✅ 50K WARMUP V2 WITH REFINED REWARDS: SUCCESS")
        print("🚀 Ready for full 200K training")
        sys.exit(0)
    elif result == "partial":
        print("⚠️ 50K WARMUP V2 WITH REFINED REWARDS: PARTIAL SUCCESS")
        print("🔧 Consider additional parameter tuning")
        sys.exit(1)
    else:
        print("❌ 50K WARMUP V2 WITH REFINED REWARDS: FAILED")
        sys.exit(1)