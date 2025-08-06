#!/usr/bin/env python3
"""
🔬 DIAGNOSTIC CHECKPOINT EVALUATION SCRIPT
Phase 1A: Freeze-Early Validity Test

Evaluates saved checkpoints on held-out data with LR=0, entropy=0.
"""

import sys
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointEvaluator:
    """Evaluates saved checkpoints on held-out data"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.secrets = SecretsHelper()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_environment(self) -> DummyVecEnv:
        """Create the evaluation environment"""
        logger.info("🏗️ Creating evaluation environment...")
        
        # Create TimescaleDB config using vault
        timescaledb_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': self.secrets.get_timescaledb_password()
        }
        
        # Create data adapter for evaluation period
        data_adapter = DualTickerDataAdapter(
            timescaledb_config=timescaledb_config,
            live_trading_mode=False
        )
        
        # Load evaluation data
        market_data = data_adapter.load_training_data(
            start_date=self.config['environment']['start_date'],
            end_date=self.config['environment']['end_date'],
            symbols=self.config['environment']['symbols'],
            bar_size=self.config['environment']['bar_size'],
            data_split='validation'  # Use validation split for evaluation
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
        
        logger.info(f"📊 Evaluation data loaded: {len(trading_days)} timesteps, {combined_features.shape[1]} features")
        
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
        
        # Wrap with RefinedRewardSystem (same as training)
        reward_params = self.config['reward_system']['parameters']
        refined_reward_system = RefinedRewardSystem(
            initial_capital=self.config['environment']['initial_capital'],
            pnl_epsilon=reward_params['pnl_epsilon'],
            holding_alpha=reward_params['holding_alpha'],
            penalty_beta=reward_params['penalty_beta'],
            exploration_coef=reward_params['exploration_coef']
        )
        
        # Create reward wrapper
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
        
        logger.info(f"✅ Evaluation environment created")
        logger.info(f"   Eval period: {self.config['environment']['start_date']} to {self.config['environment']['end_date']}")
        
        return DummyVecEnv([lambda: monitored_env])
    
    def evaluate_checkpoint(self, checkpoint_path: str, n_eval_steps: int = 5000) -> Dict[str, Any]:
        """Evaluate a single checkpoint"""
        logger.info(f"🔍 Evaluating checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create environment
        env = self._create_environment()
        
        # Load model
        model = PPO.load(checkpoint_path, env=env)
        
        # Set to evaluation mode (no learning, no exploration)
        model.policy.set_training_mode(False)
        
        logger.info(f"🤖 Model loaded, starting evaluation...")
        logger.info(f"   Evaluation steps: {n_eval_steps}")
        logger.info(f"   Learning rate: 0.0 (frozen)")
        logger.info(f"   Entropy coef: 0.0 (no exploration)")
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        episode_data = []
        
        obs = env.reset()
        total_steps = 0
        episode_reward = 0.0
        episode_length = 0
        episode_actions = []
        episode_positions = []
        
        start_time = datetime.now()
        
        while total_steps < n_eval_steps:
            # Get action (deterministic, no exploration)
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            total_steps += 1
            
            # Store episode data
            if len(info) > 0 and 'positions' in info[0]:
                episode_actions.append(action[0] if hasattr(action[0], '__len__') else action[0])
                episode_positions.append(info[0]['positions'])
            
            # Progress reporting
            if total_steps % 1000 == 0:
                logger.info(f"📊 Evaluation progress: {total_steps}/{n_eval_steps} steps")
            
            # Episode ended
            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Store detailed episode data
                episode_data.append({
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'actions': episode_actions.copy(),
                    'positions': episode_positions.copy()
                })
                
                logger.info(f"📈 Episode completed: reward={episode_reward:.3f}, length={episode_length}")
                
                # Reset for next episode
                episode_reward = 0.0
                episode_length = 0
                episode_actions = []
                episode_positions = []
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Calculate statistics
        results = {
            'checkpoint_path': checkpoint_path,
            'evaluation_period': f"{self.config['environment']['start_date']} to {self.config['environment']['end_date']}",
            'total_steps': total_steps,
            'total_episodes': len(episode_rewards),
            'duration_seconds': duration.total_seconds(),
            
            # Reward statistics
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
            'min_reward': np.min(episode_rewards) if episode_rewards else 0.0,
            'max_reward': np.max(episode_rewards) if episode_rewards else 0.0,
            'median_reward': np.median(episode_rewards) if episode_rewards else 0.0,
            
            # Episode length statistics
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'std_episode_length': np.std(episode_lengths) if episode_lengths else 0.0,
            'min_episode_length': np.min(episode_lengths) if episode_lengths else 0.0,
            'max_episode_length': np.max(episode_lengths) if episode_lengths else 0.0,
            
            # Success criteria
            'passes_reward_threshold': np.mean(episode_rewards) >= 0.5 if episode_rewards else False,
            'above_floor_threshold': np.mean(episode_rewards) >= 0.3 if episode_rewards else False,
            'min_episode_length_met': np.mean(episode_lengths) >= 40 if episode_lengths else False,
            
            # Raw data
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_data': episode_data
        }
        
        env.close()
        
        logger.info(f"✅ Evaluation completed in {duration}")
        logger.info(f"📊 Results summary:")
        logger.info(f"   Mean reward: {results['mean_reward']:.3f}")
        logger.info(f"   Episodes: {results['total_episodes']}")
        logger.info(f"   Avg episode length: {results['mean_episode_length']:.1f}")
        logger.info(f"   Passes ≥0.5 threshold: {results['passes_reward_threshold']}")
        logger.info(f"   Above ≥0.3 floor: {results['above_floor_threshold']}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary results (without raw episode data)
        summary_results = {k: v for k, v in results.items() 
                          if k not in ['episode_rewards', 'episode_lengths', 'episode_data']}
        
        # Save as CSV for easy analysis
        df = pd.DataFrame([summary_results])
        df.to_csv(output_path, index=False)
        
        # Save detailed results as JSON if requested
        if self.config.get('output', {}).get('save_episode_data', False):
            import json
            detailed_path = output_path.replace('.csv', '_detailed.json')
            with open(detailed_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = results.copy()
                for key in ['episode_rewards', 'episode_lengths']:
                    if key in json_results:
                        json_results[key] = [float(x) for x in json_results[key]]
                
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"💾 Detailed results saved: {detailed_path}")
        
        logger.info(f"💾 Summary results saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Diagnostic Checkpoint Evaluation Script')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--config', required=True, help='Path to evaluation config file')
    parser.add_argument('--output', required=True, help='Path to save results CSV')
    parser.add_argument('--steps', type=int, default=5000, help='Number of evaluation steps')
    parser.add_argument('--return_mean_reward', action='store_true', help='Return mean reward for scripting')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Run evaluation
    try:
        evaluator = CheckpointEvaluator(args.config)
        results = evaluator.evaluate_checkpoint(args.checkpoint, args.steps)
        evaluator.save_results(results, args.output)
        
        logger.info(f"🎉 Checkpoint evaluation successful!")
        
        # Return mean reward for scripting
        if args.return_mean_reward:
            print(results['mean_reward'])
        
    except Exception as e:
        logger.error(f"💥 Checkpoint evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()