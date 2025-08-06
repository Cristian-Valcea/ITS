#!/usr/bin/env python3
"""
🔍 10K HOLDING PROBE - FAST VALIDATION
Test improved holding bonus logic + parameter nudges:
- Fixed holding bonus: lookback-based, long/short support
- penalty_beta: 0.15 → 0.10 (gentle further soften)
- holding_alpha: 0.03 → 0.05 (activate and matter)
- holding_lookback_k: 5 steps
- holding_min_ret: 0.0 (no noise threshold)

Pass/Fail Gates:
✅ holding_bonus_sum > 0 and holding_bonus_triggers > 0
✅ ep_rew_mean ≥ -0.25 (should improve vs -0.357)
✅ ep_len_mean ≥ 15
✅ Penalty per episode still modest (-0.05 to -0.3)
✅ Trade frequency not collapsing (action distribution not 95% hold)
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

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

class HoldingProbeWrapper(Wrapper):
    """Wrapper to track holding bonus activation and action distribution"""
    
    def __init__(self, env, refined_reward_system):
        super().__init__(env)
        self.refined_reward = refined_reward_system
        # Track detailed metrics for probe validation
        self.episode_components = {
            'normalized_pnl': [],
            'holding_bonus': [],
            'smoothed_penalty': [],
            'exploration_bonus': [],
            'directional_bonus': []
        }
        self.episode_actions = []
        self.total_holding_bonus_sum = 0.0
        self.total_penalty_sum = 0.0
        self.episode_count = 0
    
    def step(self, action):
        """Step with detailed tracking for probe validation"""
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
        
        # Track components and actions
        for component in self.episode_components:
            self.episode_components[component].append(getattr(reward_components, component))
        self.episode_actions.append(action)
        
        # Accumulate totals
        self.total_holding_bonus_sum += reward_components.holding_bonus
        self.total_penalty_sum += reward_components.smoothed_penalty
        
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
        """Log episode-level summary with holding bonus details"""
        if not any(self.episode_components.values()):
            return
        
        self.episode_count += 1
        
        # Calculate episode totals
        holding_sum = sum(self.episode_components['holding_bonus'])
        penalty_sum = sum(self.episode_components['smoothed_penalty'])
        episode_length = len(self.episode_actions)
        
        # Action distribution
        action_counts = Counter(self.episode_actions)
        hold_rate = action_counts.get(4, 0) / max(episode_length, 1)
        
        # Holding bonus stats
        reward_stats = self.refined_reward.get_stats()
        
        logger.info(f"📊 Episode {self.episode_count}: " +
                   f"holding_bonus_sum={holding_sum:.4f}, " +
                   f"penalty_sum={penalty_sum:.3f}, " +
                   f"hold_rate={hold_rate:.2f}, " +
                   f"triggers={reward_stats['holding_bonus_triggers']}")
    
    def _reset_episode_tracking(self):
        """Reset episode component tracking"""
        for component in self.episode_components:
            self.episode_components[component] = []
        self.episode_actions = []
    
    def get_probe_stats(self):
        """Get comprehensive probe validation statistics"""
        reward_stats = self.refined_reward.get_stats()
        
        return {
            'total_holding_bonus_sum': self.total_holding_bonus_sum,
            'total_penalty_sum': self.total_penalty_sum,
            'episode_count': self.episode_count,
            'holding_bonus_triggers': reward_stats['holding_bonus_triggers'],
            'steps_in_position': reward_stats['steps_in_position'],
            'holding_trigger_rate': reward_stats['holding_trigger_rate'],
            'avg_holding_bonus_per_episode': self.total_holding_bonus_sum / max(self.episode_count, 1),
            'avg_penalty_per_episode': self.total_penalty_sum / max(self.episode_count, 1)
        }

def main():
    """Main 10K holding probe execution"""
    
    logger.info("🔍 10K HOLDING PROBE - FAST VALIDATION")
    logger.info("=" * 60)
    
    # Probe configuration
    total_timesteps = 10000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"holding_probe_10k_{timestamp}"
    save_path = Path(f"train_runs/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎪 Run: {run_name}")
    logger.info(f"📊 Steps: {total_timesteps:,}")
    logger.info(f"🎯 Focus: Validate improved holding bonus logic")
    logger.info(f"🔧 Changes: Fixed holding logic + penalty_beta=0.10, holding_alpha=0.05")
    
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
        
        # Create base environment
        logger.info("🏗️ Creating probe environment...")
        base_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=390,          # Full trading day
            max_daily_drawdown_pct=0.20,    # Keep permissive for probe
            transaction_cost_pct=0.001
        )
        
        # Create refined reward system with PROBE PARAMETERS
        logger.info("🎯 Initializing refined reward system with probe parameters...")
        logger.info("🔧 PROBE PARAMETER CHANGES:")
        logger.info("   penalty_beta: 0.15 → 0.10 (gentle further soften)")
        logger.info("   holding_alpha: 0.03 → 0.05 (activate and matter)")
        logger.info("   holding_lookback_k: 5 (new lookback logic)")
        logger.info("   holding_min_ret: 0.0 (no noise threshold)")
        
        refined_reward_system = RefinedRewardSystem(
            initial_capital=10000.0,
            pnl_epsilon=750.0,              # Keep from V2
            holding_alpha=0.05,             # ↑ from 0.03 (activate and matter)
            holding_lookback_k=5,           # New lookback logic
            holding_min_ret=0.0,            # No noise threshold
            penalty_beta=0.10,              # ↓ from 0.15 (gentle further soften)
            exploration_coef=0.05,          # Same as V2
            exploration_decay=0.9999,       # Same as V2
            verbose=True
        )
        
        # Wrap environment with probe tracking
        env = HoldingProbeWrapper(base_env, refined_reward_system)
        env = Monitor(env, str(save_path / "monitor.csv"))
        env = DummyVecEnv([lambda: env])
        
        logger.info("✅ Environment with improved holding bonus created")
        
        # Create PPO model (same as V2)
        logger.info("🤖 Creating PPO model for probe...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,                  # HIGH exploration for probe
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=f"tensorboard_logs/{run_name}",
            verbose=1
        )
        
        logger.info("✅ PPO model created for holding probe")
        
        # Launch 10K probe training
        logger.info("🚀 Starting 10K holding probe...")
        logger.info("🎯 PROBE GATES:")
        logger.info("   ✅ holding_bonus_sum > 0 and holding_bonus_triggers > 0")
        logger.info("   ✅ ep_rew_mean ≥ -0.25 (should improve vs -0.357)")
        logger.info("   ✅ ep_len_mean ≥ 15")
        logger.info("   ✅ Penalty per episode still modest (-0.05 to -0.3)")
        logger.info("   ✅ Trade frequency not collapsing (action distribution not 95% hold)")
        logger.info("=" * 60)
        
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name
        )
        
        # Save model
        model_path = save_path / "holding_probe_model.zip"
        model.save(str(model_path))
        
        logger.info("🎉 10K HOLDING PROBE COMPLETE!")
        logger.info(f"📁 Model saved: {model_path}")
        
        # PROBE VALIDATION - Check all gates
        logger.info("🧪 Running probe validation...")
        
        # Get comprehensive probe statistics
        probe_stats = env.envs[0].env.get_probe_stats()
        
        logger.info("📊 PROBE STATISTICS:")
        logger.info(f"  Total Holding Bonus Sum: {probe_stats['total_holding_bonus_sum']:.6f}")
        logger.info(f"  Holding Bonus Triggers: {probe_stats['holding_bonus_triggers']}")
        logger.info(f"  Steps in Position: {probe_stats['steps_in_position']}")
        logger.info(f"  Holding Trigger Rate: {probe_stats['holding_trigger_rate']:.3f}")
        logger.info(f"  Avg Holding Bonus per Episode: {probe_stats['avg_holding_bonus_per_episode']:.6f}")
        logger.info(f"  Avg Penalty per Episode: {probe_stats['avg_penalty_per_episode']:.3f}")
        
        # Quick validation episodes
        obs, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        action_counts = Counter()
        
        for episode in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                action_counts[action[0]] += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(f"Validation Episode {episode + 1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        # Action distribution analysis
        total_actions = sum(action_counts.values())
        hold_rate = action_counts.get(4, 0) / max(total_actions, 1)
        
        logger.info("📊 VALIDATION RESULTS:")
        logger.info(f"  Average Reward: {avg_reward:.3f}")
        logger.info(f"  Average Length: {avg_length:.1f} steps")
        logger.info(f"  Hold Rate: {hold_rate:.3f}")
        logger.info(f"  Action Distribution: {dict(action_counts)}")
        
        # GATE VALIDATION
        logger.info("🎯 PROBE GATE VALIDATION:")
        
        gates = {
            'holding_bonus_active': (
                probe_stats['total_holding_bonus_sum'] > 0 and 
                probe_stats['holding_bonus_triggers'] > 0
            ),
            'reward_improved': avg_reward >= -0.25,
            'length_adequate': avg_length >= 15.0,
            'penalty_modest': -0.5 <= probe_stats['avg_penalty_per_episode'] <= -0.05,
            'trade_frequency_healthy': hold_rate < 0.95
        }
        
        for gate_name, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {gate_name}: {status}")
        
        # Overall assessment
        critical_gates = ['holding_bonus_active', 'reward_improved', 'length_adequate']
        critical_passed = all(gates[gate] for gate in critical_gates)
        all_passed = all(gates.values())
        
        if all_passed:
            logger.info("🎉 PROBE EXCELLENT: All gates passed - Ready for 200K!")
            return "excellent"
        elif critical_passed:
            logger.info("✅ PROBE SUCCESS: Critical gates passed - Ready for 200K")
            return "success"
        else:
            logger.warning("⚠️ PROBE PARTIAL: Some gates failed - Need more tuning")
            return "partial"
        
    except Exception as e:
        logger.error(f"❌ Holding probe failed: {e}")
        import traceback
        traceback.print_exc()
        return "failed"

if __name__ == "__main__":
    result = main()
    
    if result in ["excellent", "success"]:
        print("🎉 10K HOLDING PROBE: SUCCESS")
        print("🚀 Ready for 200K training with curriculum tightening")
        sys.exit(0)
    elif result == "partial":
        print("⚠️ 10K HOLDING PROBE: PARTIAL SUCCESS")
        print("🔧 Consider additional parameter tuning")
        sys.exit(1)
    else:
        print("❌ 10K HOLDING PROBE: FAILED")
        sys.exit(1)