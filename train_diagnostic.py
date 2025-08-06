#!/usr/bin/env python3
"""
🔬 DIAGNOSTIC TRAINING SCRIPT
Phase 1A: Freeze-Early Validity Test

Trains models to specific timestep checkpoints for diagnostic analysis.
"""

import sys
import yaml
import logging
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from src.gym_env.refined_reward_system import RefinedRewardSystem

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticCallback(BaseCallback):
    """Custom callback for diagnostic training with enhanced monitoring"""
    
    def __init__(self, target_steps: int, checkpoint_path: str, report_freq: int = 1000):
        super().__init__()
        self.target_steps = target_steps
        self.checkpoint_path = checkpoint_path
        self.report_freq = report_freq
        self.last_report_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Collect episode data
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        
        # Progress reporting
        if self.num_timesteps - self.last_report_step >= self.report_freq:
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            logger.info(f"📊 Step {self.num_timesteps}/{self.target_steps} | "
                       f"Recent avg reward: {avg_reward:.3f} | "
                       f"Episodes completed: {len(self.episode_rewards)}")
            
            self.last_report_step = self.num_timesteps
        
        # Check if we've reached target steps
        if self.num_timesteps >= self.target_steps:
            logger.info(f"🎯 Reached target steps {self.target_steps}, saving checkpoint...")
            self.model.save(self.checkpoint_path)
            
            # Log final statistics
            if self.episode_rewards:
                final_avg_reward = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else np.mean(self.episode_rewards)
                logger.info(f"✅ Training complete! Final avg reward: {final_avg_reward:.3f}")
                logger.info(f"📈 Total episodes: {len(self.episode_rewards)}")
                logger.info(f"💾 Checkpoint saved: {self.checkpoint_path}")
            
            return False  # Stop training
        
        return True

class DiagnosticTrainer:
    """Diagnostic trainer for Phase 1A freeze-early tests"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.secrets = SecretsHelper()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_environment(self) -> DummyVecEnv:
        """Create the training environment with RefinedRewardSystem"""
        logger.info("🏗️ Creating diagnostic training environment...")
        
        # Create TimescaleDB config using vault
        timescaledb_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': self.secrets.get_timescaledb_password()
        }
        
        # Create data adapter
        data_adapter = DualTickerDataAdapter(
            timescaledb_config=timescaledb_config,
            live_trading_mode=False
        )
        
        # Load training data
        market_data = data_adapter.load_training_data(
            start_date=self.config['environment']['start_date'],
            end_date=self.config['environment']['end_date'],
            symbols=self.config['environment']['symbols'],
            bar_size=self.config['environment']['bar_size'],
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
        
        combined_prices = np.column_stack([
            nvda_prices, nvda_returns, msft_prices, msft_returns
        ])
        
        logger.info(f"📊 Data loaded: {len(trading_days)} timesteps, {combined_features.shape[1]} features")
        
        # Create base environment
        base_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=self.config['environment']['initial_capital'],
            lookback_window=self.config['environment']['lookback_window'],
            max_episode_steps=self.config['environment']['max_episode_steps'],
            max_daily_drawdown_pct=self.config['environment']['max_drawdown_pct'],
            transaction_cost_pct=self.config['environment']['transaction_cost_pct']
        )
        
        # Configure reward system based on type
        reward_type = self.config['reward_system'].get('type', 'refined_wrapper')
        
        if reward_type == 'v3_enhanced':
            # Variant A: Use stock V3Enhanced reward (no wrapper)
            logger.info("🎯 Using stock V3Enhanced reward system (Variant A)")
            monitored_env = Monitor(base_env)
            
        elif reward_type == 'refined_wrapper':
            # Variant B: RefinedRewardSystem as wrapper (current shim approach)
            logger.info("🎯 Using RefinedRewardSystem as wrapper (Variant B)")
            reward_params = self.config['reward_system']['parameters']
            refined_reward_system = RefinedRewardSystem(
                initial_capital=self.config['environment']['initial_capital'],
                pnl_epsilon=reward_params['pnl_epsilon'],
                holding_alpha=reward_params['holding_alpha'],
                penalty_beta=reward_params['penalty_beta'],
                exploration_coef=reward_params['exploration_coef'],
                early_exit_tax=reward_params.get('early_exit_tax', 0.0),
                min_episode_length=reward_params.get('min_episode_length', 80),
                time_bonus=reward_params.get('time_bonus', 0.0),
                time_bonus_threshold=reward_params.get('time_bonus_threshold', 60),
                completion_bonus=reward_params.get('completion_bonus', 0.0),
                completion_threshold=reward_params.get('completion_threshold', 80),
                secondary_tax=reward_params.get('secondary_tax', 0.0),
                secondary_threshold=reward_params.get('secondary_threshold', 70),
                major_completion_bonus=reward_params.get('major_completion_bonus', 0.0),
                major_threshold=reward_params.get('major_threshold', 80)
            )
            
            # Create reward wrapper
            import gymnasium as gym
            class RewardWrapper(gym.Wrapper):
                def __init__(self, env, reward_system):
                    super().__init__(env)
                    self.reward_system = reward_system
                    self.episode_step_count = 0
                    
                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    self.reward_system.reset_episode()
                    self.episode_step_count = 0
                    return obs, info
                    
                def step(self, action):
                    obs, original_reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    self.episode_step_count += 1
                    
                    # Get refined reward
                    refined_reward = self.reward_system.calculate_reward(
                        portfolio_value=info.get('portfolio_value', 10000.0),
                        previous_portfolio_value=info.get('previous_portfolio_value', 10000.0),
                        nvda_position=info.get('positions', [0.0, 0.0])[0],
                        msft_position=info.get('positions', [0.0, 0.0])[1],
                        action=action if isinstance(action, int) else int(action),
                        drawdown_pct=info.get('drawdown_pct', 0.0)
                    )
                    
                    # Apply early-exit tax and completion bonuses if episode is ending
                    final_reward = refined_reward.total_reward if hasattr(refined_reward, 'total_reward') else refined_reward
                    if done:
                        early_exit_penalty, completion_bonus, major_completion_bonus = self.reward_system.apply_early_exit_tax(self.episode_step_count)
                        final_reward += early_exit_penalty + completion_bonus + major_completion_bonus
                        if early_exit_penalty < 0:
                            info['early_exit_penalty'] = early_exit_penalty
                        if completion_bonus > 0:
                            info['completion_bonus'] = completion_bonus
                        if major_completion_bonus > 0:
                            info['major_completion_bonus'] = major_completion_bonus
                    
                    # Update info with reward breakdown
                    if hasattr(refined_reward, 'to_dict'):
                        info['reward_breakdown'] = refined_reward.to_dict()
                        if done:
                            if 'early_exit_penalty' in info:
                                info['reward_breakdown']['early_exit_tax'] = early_exit_penalty
                            if 'completion_bonus' in info:
                                info['reward_breakdown']['completion_bonus'] = completion_bonus
                            if 'major_completion_bonus' in info:
                                info['reward_breakdown']['major_completion_bonus'] = major_completion_bonus
                    
                    return obs, final_reward, terminated, truncated, info
                    
                def __getattr__(self, name):
                    return getattr(self.env, name)
            
            wrapped_env = RewardWrapper(base_env, refined_reward_system)
            monitored_env = Monitor(wrapped_env)
            
        elif reward_type == 'refined_internal':
            # Variant C: RefinedRewardSystem integrated into environment
            logger.info("🎯 Using RefinedRewardSystem integrated into environment (Variant C)")
            # For now, this will be the same as wrapper but could be implemented differently
            # TODO: Implement true integration into environment reward calculation
            reward_params = self.config['reward_system']['parameters']
            refined_reward_system = RefinedRewardSystem(
                initial_capital=self.config['environment']['initial_capital'],
                pnl_epsilon=reward_params['pnl_epsilon'],
                holding_alpha=reward_params['holding_alpha'],
                penalty_beta=reward_params['penalty_beta'],
                exploration_coef=reward_params['exploration_coef'],
                early_exit_tax=reward_params.get('early_exit_tax', 0.0),
                min_episode_length=reward_params.get('min_episode_length', 80),
                time_bonus=reward_params.get('time_bonus', 0.0),
                time_bonus_threshold=reward_params.get('time_bonus_threshold', 60),
                completion_bonus=reward_params.get('completion_bonus', 0.0),
                completion_threshold=reward_params.get('completion_threshold', 80),
                secondary_tax=reward_params.get('secondary_tax', 0.0),
                secondary_threshold=reward_params.get('secondary_threshold', 70),
                major_completion_bonus=reward_params.get('major_completion_bonus', 0.0),
                major_threshold=reward_params.get('major_threshold', 80)
            )
            
            # Use same wrapper for now - true integration would require environment modification
            import gymnasium as gym
            class RewardWrapper(gym.Wrapper):
                def __init__(self, env, reward_system):
                    super().__init__(env)
                    self.reward_system = reward_system
                    
                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    self.reward_system.reset_episode()
                    return obs, info
                    
                def step(self, action):
                    obs, original_reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Get refined reward
                    refined_reward = self.reward_system.calculate_reward(
                        portfolio_value=info.get('portfolio_value', 10000.0),
                        previous_portfolio_value=info.get('previous_portfolio_value', 10000.0),
                        nvda_position=info.get('positions', [0.0, 0.0])[0],
                        msft_position=info.get('positions', [0.0, 0.0])[1],
                        action=action if isinstance(action, int) else int(action),
                        drawdown_pct=info.get('drawdown_pct', 0.0)
                    )
                    
                    # Update info with reward breakdown
                    if hasattr(refined_reward, 'to_dict'):
                        info['reward_breakdown'] = refined_reward.to_dict()
                        final_reward = refined_reward.total_reward
                    else:
                        final_reward = refined_reward
                    
                    return obs, final_reward, terminated, truncated, info
                    
                def __getattr__(self, name):
                    return getattr(self.env, name)
            
            wrapped_env = RewardWrapper(base_env, refined_reward_system)
            monitored_env = Monitor(wrapped_env)
            
        else:
            raise ValueError(f"Unknown reward system type: {reward_type}")
        
        logger.info(f"✅ Environment created with RefinedRewardSystem")
        logger.info(f"   Data period: {self.config['environment']['start_date']} to {self.config['environment']['end_date']}")
        logger.info(f"   Max drawdown: {self.config['environment']['max_drawdown_pct']*100:.1f}%")
        
        return DummyVecEnv([lambda: monitored_env])
    
    def train(self, total_timesteps: int, save_path: str, overrides: Dict[str, Any] = None, args=None) -> str:
        """Train model to specified timesteps and save checkpoint"""
        logger.info(f"🚀 Starting diagnostic training to {total_timesteps} steps...")
        logger.info(f"💾 Will save checkpoint to: {save_path}")
        
        # Apply overrides to config
        if overrides is None:
            overrides = {}
        
        # Apply reward system overrides
        reward_overrides = ['early_exit_tax', 'min_episode_length', 'time_bonus', 'time_bonus_threshold', 'completion_bonus', 'completion_threshold', 'secondary_tax', 'secondary_threshold', 'major_completion_bonus', 'major_threshold']
        for key in reward_overrides:
            if key in overrides:
                self.config['reward_system']['parameters'][key] = overrides[key]
                logger.info(f"🔧 Override applied: {key} = {overrides[key]}")
        
        # Apply environment overrides
        if 'max_position_ratio' in overrides:
            self.config['environment']['max_position_ratio'] = overrides['max_position_ratio']
            logger.info(f"🔧 Environment override applied: max_position_ratio = {overrides['max_position_ratio']}")
        
        # Create environment
        env = self._create_environment()
        
        # Get PPO parameters with overrides
        learning_rate = overrides.get('learning_rate', self.config['ppo']['learning_rate'])
        target_kl = overrides.get('target_kl', self.config['ppo'].get('target_kl', 0.01))
        ent_coef = overrides.get('entropy_coef', self.config['ppo']['ent_coef'])
        
        # Handle resume from checkpoint
        if hasattr(args, 'resume_from') and args.resume_from:
            logger.info(f"📂 Resuming from checkpoint: {args.resume_from}")
            model = PPO.load(args.resume_from, env=env)
            
            # Update parameters for curriculum
            if learning_rate != model.learning_rate:
                model.learning_rate = learning_rate
                logger.info(f"🔧 Updated learning rate: {learning_rate}")
            if target_kl != model.target_kl:
                model.target_kl = target_kl
                logger.info(f"🔧 Updated target KL: {target_kl}")
            if ent_coef != model.ent_coef:
                model.ent_coef = ent_coef
                logger.info(f"🔧 Updated entropy coef: {ent_coef}")
        else:
            # Create PPO model from scratch
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=self.config['ppo']['n_steps'],
                batch_size=self.config['ppo']['batch_size'],
                n_epochs=self.config['ppo']['n_epochs'],
                gamma=self.config['ppo']['gamma'],
                gae_lambda=self.config['ppo']['gae_lambda'],
                clip_range=self.config['ppo']['clip_range'],
                ent_coef=ent_coef,
                vf_coef=self.config['ppo']['vf_coef'],
                max_grad_norm=self.config['ppo']['max_grad_norm'],
                target_kl=target_kl,
                seed=self.config['training']['seed'],
                verbose=1
            )
        
        logger.info(f"🤖 PPO model created with:")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Target KL: {target_kl}")
        logger.info(f"   Entropy coef: {ent_coef}")
        logger.info(f"   N steps: {self.config['ppo']['n_steps']}")
        logger.info(f"   Seed: {self.config['training']['seed']}")
        
        if overrides:
            logger.info(f"🔧 Parameter overrides applied: {overrides}")
        
        # Create diagnostic callback
        callback = DiagnosticCallback(
            target_steps=total_timesteps,
            checkpoint_path=save_path,
            report_freq=1000
        )
        
        # Start training
        start_time = datetime.now()
        logger.info(f"⏰ Training started at {start_time}")
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"✅ Training completed in {duration}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        finally:
            env.close()

def main():
    parser = argparse.ArgumentParser(description='Diagnostic Training Script')
    parser.add_argument('--config', required=True, help='Path to training config file')
    parser.add_argument('--total_timesteps', type=int, required=True, help='Total timesteps to train')
    parser.add_argument('--save_path', required=True, help='Path to save checkpoint')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--target_kl', type=float, help='Override target KL divergence')
    parser.add_argument('--entropy_coef', type=float, help='Override entropy coefficient')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--lr_adapt_patience', type=int, default=5000, help='Steps to wait before adapting LR')
    parser.add_argument('--lr_adapt_factor', type=float, default=0.4, help='Factor to reduce LR by')
    parser.add_argument('--max_daily_drawdown_pct_schedule', type=str, help='JSON schedule for drawdown limits')
    parser.add_argument('--early_exit_tax', type=float, default=0.0, help='Early exit tax penalty')
    parser.add_argument('--early_exit_threshold', type=int, default=80, help='Episode length threshold for tax')
    parser.add_argument('--time_bonus', type=float, default=0.0, help='Per-step time bonus for staying in market')
    parser.add_argument('--time_bonus_threshold', type=int, default=60, help='Step threshold to start time bonus')
    parser.add_argument('--completion_bonus', type=float, default=0.0, help='Bonus for completing long episodes')
    parser.add_argument('--completion_threshold', type=int, default=80, help='Episode length threshold for completion bonus')
    parser.add_argument('--secondary_tax', type=float, default=0.0, help='Secondary tax for intermediate episode lengths')
    parser.add_argument('--secondary_threshold', type=int, default=70, help='Secondary tax threshold')
    parser.add_argument('--major_completion_bonus', type=float, default=0.0, help='Major bonus for very long episodes')
    parser.add_argument('--major_threshold', type=int, default=80, help='Major completion threshold')
    parser.add_argument('--max_position_ratio', type=float, default=1.0, help='Maximum position ratio to prevent YOLO sizing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.config).exists():
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if args.total_timesteps <= 0:
        logger.error(f"❌ Invalid timesteps: {args.total_timesteps}")
        sys.exit(1)
    
    # Create output directory
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        trainer = DiagnosticTrainer(args.config)
        
        # Override config parameters if provided
        overrides = {}
        if args.learning_rate is not None:
            overrides['learning_rate'] = args.learning_rate
        if args.target_kl is not None:
            overrides['target_kl'] = args.target_kl
        if args.entropy_coef is not None:
            overrides['entropy_coef'] = args.entropy_coef
        if args.early_exit_tax is not None:
            overrides['early_exit_tax'] = args.early_exit_tax
        if args.early_exit_threshold is not None:
            overrides['min_episode_length'] = args.early_exit_threshold
        if args.time_bonus is not None:
            overrides['time_bonus'] = args.time_bonus
        if args.time_bonus_threshold is not None:
            overrides['time_bonus_threshold'] = args.time_bonus_threshold
        if args.completion_bonus is not None:
            overrides['completion_bonus'] = args.completion_bonus
        if args.completion_threshold is not None:
            overrides['completion_threshold'] = args.completion_threshold
        if args.secondary_tax is not None:
            overrides['secondary_tax'] = args.secondary_tax
        if args.secondary_threshold is not None:
            overrides['secondary_threshold'] = args.secondary_threshold
        if args.major_completion_bonus is not None:
            overrides['major_completion_bonus'] = args.major_completion_bonus
        if args.major_threshold is not None:
            overrides['major_threshold'] = args.major_threshold
        if args.max_position_ratio is not None:
            overrides['max_position_ratio'] = args.max_position_ratio
        
        checkpoint_path = trainer.train(args.total_timesteps, args.save_path, overrides, args)
        
        logger.info(f"🎉 Diagnostic training successful!")
        logger.info(f"📁 Checkpoint saved: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"💥 Diagnostic training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()