#!/usr/bin/env python3
"""
🔬 MICRO-OVERFIT ANALYSIS
Compare the 8K micro-overfit model vs 5K success model to understand degradation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
from datetime import datetime

# Add src to path
sys.path.append("src")

from stable_baselines3 import PPO
from src.gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.reward_system.refined_reward_system import RefinedRewardSystem
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def create_evaluation_environment(config_path):
    """Create the same environment used for training"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data adapter
    data_adapter = DualTickerDataAdapter(
        timescaledb_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'trading_user',
            'password': 'your_password'
        },
        live_trading_mode=False
    )
    
    # Load evaluation data (Feb 2024)
    market_data = data_adapter.load_training_data(
        start_date='2024-02-01',
        end_date='2024-02-29',
        symbols=['NVDA', 'MSFT'],
        bar_size='1min',
        data_split='eval'
    )
    
    # Prepare data
    nvda_features = market_data['nvda_features']
    nvda_prices = market_data['nvda_prices']
    msft_features = market_data['msft_features']
    msft_prices = market_data['msft_prices']
    trading_days = market_data['trading_days']
    
    # Combine features
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    position_features = np.zeros((combined_features.shape[0], 2))
    combined_features = np.concatenate([combined_features, position_features], axis=1)
    
    # Create price data
    nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
    msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
    combined_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
    
    # Create base environment
    base_env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=combined_features,
        processed_price_data=combined_prices,
        trading_days=trading_days,
        initial_capital=config['environment']['initial_capital'],
        lookback_window=config['environment']['lookback_window'],
        max_episode_steps=config['environment']['max_episode_steps'],
        max_daily_drawdown_pct=config['environment']['max_drawdown_pct'],
        transaction_cost_pct=config['environment']['transaction_cost_pct']
    )
    
    # Add RefinedRewardSystem wrapper
    reward_params = config['reward_system']['parameters']
    refined_reward_system = RefinedRewardSystem(
        initial_capital=config['environment']['initial_capital'],
        pnl_epsilon=reward_params['pnl_epsilon'],
        holding_alpha=reward_params['holding_alpha'],
        penalty_beta=reward_params['penalty_beta'],
        exploration_coef=reward_params['exploration_coef']
    )
    
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
            
            refined_reward = self.reward_system.calculate_reward(
                portfolio_value=info.get('portfolio_value', 10000.0),
                previous_portfolio_value=info.get('previous_portfolio_value', 10000.0),
                nvda_position=info.get('positions', [0.0, 0.0])[0],
                msft_position=info.get('positions', [0.0, 0.0])[1],
                action=action if isinstance(action, int) else int(action),
                drawdown_pct=info.get('drawdown_pct', 0.0)
            )
            
            if hasattr(refined_reward, 'to_dict'):
                info['reward_breakdown'] = refined_reward.to_dict()
                final_reward = refined_reward.total_reward
            else:
                final_reward = refined_reward
            
            return obs, final_reward, terminated, truncated, info
    
    wrapped_env = RewardWrapper(base_env, refined_reward_system)
    return wrapped_env

def detailed_model_evaluation(model, env, num_episodes=5, max_steps_per_episode=500):
    """Run detailed evaluation of a model"""
    print(f"🔍 Running detailed evaluation ({num_episodes} episodes, max {max_steps_per_episode} steps each)")
    
    results = {
        'episodes': [],
        'step_rewards': [],
        'step_actions': [],
        'step_portfolio_values': [],
        'step_positions': [],
        'reward_breakdowns': []
    }
    
    for episode in range(num_episodes):
        print(f"📊 Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_data = {
            'episode': episode,
            'rewards': [],
            'actions': [],
            'portfolio_values': [],
            'positions': [],
            'reward_breakdowns': [],
            'termination_reason': None
        }
        
        for step in range(max_steps_per_episode):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Record step data
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(int(action))
            episode_data['portfolio_values'].append(info.get('portfolio_value', 10000))
            episode_data['positions'].append(info.get('positions', [0.0, 0.0]).copy())
            
            if 'reward_breakdown' in info:
                episode_data['reward_breakdowns'].append(info['reward_breakdown'])
            
            if terminated or truncated:
                episode_data['termination_reason'] = 'terminated' if terminated else 'truncated'
                break
        
        episode_data['total_reward'] = episode_reward
        episode_data['length'] = episode_length
        episode_data['final_portfolio_value'] = episode_data['portfolio_values'][-1] if episode_data['portfolio_values'] else 10000
        
        results['episodes'].append(episode_data)
        
        print(f"   Reward: {episode_reward:.3f}, Length: {episode_length}, "
              f"Final Value: ${episode_data['final_portfolio_value']:.2f}, "
              f"Termination: {episode_data['termination_reason']}")
    
    return results

def compare_models():
    """Compare 5K success model vs 8K micro-overfit model"""
    print("🔬 MICRO-OVERFIT ANALYSIS")
    print("=" * 50)
    
    config_path = "config/reward_shim.yaml"
    
    # Load models
    success_model_path = "preserved_models/phase1a_5k_success_model.zip"
    overfit_model_path = "diagnostic_runs/micro_overfit_probe/micro_probe_final.zip"
    
    print(f"📊 Loading models...")
    print(f"   Success (5K): {success_model_path}")
    print(f"   Overfit (8K): {overfit_model_path}")
    
    try:
        success_model = PPO.load(success_model_path)
        overfit_model = PPO.load(overfit_model_path)
        print("✅ Both models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Create evaluation environment
    print(f"🏗️ Creating evaluation environment...")
    try:
        env = create_evaluation_environment(config_path)
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # Evaluate both models
    print(f"\n🎯 EVALUATING SUCCESS MODEL (5K steps)")
    print("-" * 40)
    success_results = detailed_model_evaluation(success_model, env, num_episodes=3, max_steps_per_episode=300)
    
    print(f"\n🎯 EVALUATING OVERFIT MODEL (8K steps)")
    print("-" * 40)
    overfit_results = detailed_model_evaluation(overfit_model, env, num_episodes=3, max_steps_per_episode=300)
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    # Success model stats
    success_rewards = [ep['total_reward'] for ep in success_results['episodes']]
    success_lengths = [ep['length'] for ep in success_results['episodes']]
    success_final_values = [ep['final_portfolio_value'] for ep in success_results['episodes']]
    
    # Overfit model stats
    overfit_rewards = [ep['total_reward'] for ep in overfit_results['episodes']]
    overfit_lengths = [ep['length'] for ep in overfit_results['episodes']]
    overfit_final_values = [ep['final_portfolio_value'] for ep in overfit_results['episodes']]
    
    print(f"SUCCESS MODEL (5K steps):")
    print(f"   Mean Reward: {np.mean(success_rewards):.3f} ± {np.std(success_rewards):.3f}")
    print(f"   Mean Length: {np.mean(success_lengths):.1f} ± {np.std(success_lengths):.1f}")
    print(f"   Mean Final Value: ${np.mean(success_final_values):.2f} ± ${np.std(success_final_values):.2f}")
    print(f"   Return: {(np.mean(success_final_values) - 10000) / 10000 * 100:.2f}%")
    
    print(f"\nOVERFIT MODEL (8K steps):")
    print(f"   Mean Reward: {np.mean(overfit_rewards):.3f} ± {np.std(overfit_rewards):.3f}")
    print(f"   Mean Length: {np.mean(overfit_lengths):.1f} ± {np.std(overfit_lengths):.1f}")
    print(f"   Mean Final Value: ${np.mean(overfit_final_values):.2f} ± ${np.std(overfit_final_values):.2f}")
    print(f"   Return: {(np.mean(overfit_final_values) - 10000) / 10000 * 100:.2f}%")
    
    print(f"\n🔍 DEGRADATION ANALYSIS:")
    reward_degradation = np.mean(overfit_rewards) - np.mean(success_rewards)
    length_degradation = np.mean(overfit_lengths) - np.mean(success_lengths)
    value_degradation = np.mean(overfit_final_values) - np.mean(success_final_values)
    
    print(f"   Reward Change: {reward_degradation:.3f} ({reward_degradation/np.mean(success_rewards)*100:.1f}%)")
    print(f"   Length Change: {length_degradation:.1f} ({length_degradation/np.mean(success_lengths)*100:.1f}%)")
    print(f"   Value Change: ${value_degradation:.2f} ({value_degradation/np.mean(success_final_values)*100:.1f}%)")
    
    # Analyze termination patterns
    print(f"\n🚨 TERMINATION ANALYSIS:")
    success_terminations = [ep['termination_reason'] for ep in success_results['episodes']]
    overfit_terminations = [ep['termination_reason'] for ep in overfit_results['episodes']]
    
    print(f"Success Model Terminations: {success_terminations}")
    print(f"Overfit Model Terminations: {overfit_terminations}")
    
    # Action pattern analysis
    print(f"\n🎯 ACTION PATTERN ANALYSIS:")
    
    # Flatten all actions for each model
    success_actions = []
    for ep in success_results['episodes']:
        success_actions.extend(ep['actions'])
    
    overfit_actions = []
    for ep in overfit_results['episodes']:
        overfit_actions.extend(ep['actions'])
    
    action_names = ['SELL_BOTH', 'SELL_NVDA_HOLD_MSFT', 'SELL_NVDA_BUY_MSFT', 
                   'HOLD_NVDA_SELL_MSFT', 'HOLD_BOTH', 'HOLD_NVDA_BUY_MSFT',
                   'BUY_NVDA_SELL_MSFT', 'BUY_NVDA_HOLD_MSFT', 'BUY_BOTH']
    
    print(f"Success Model Action Distribution:")
    for i, name in enumerate(action_names):
        count = success_actions.count(i)
        pct = count / len(success_actions) * 100 if success_actions else 0
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    print(f"\nOverfit Model Action Distribution:")
    for i, name in enumerate(action_names):
        count = overfit_actions.count(i)
        pct = count / len(overfit_actions) * 100 if overfit_actions else 0
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    # Save detailed results
    results_file = Path("diagnostic_runs/micro_overfit_probe/detailed_analysis.txt")
    with open(results_file, 'w') as f:
        f.write(f"MICRO-OVERFIT ANALYSIS RESULTS\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Success Model (5K): Mean Reward {np.mean(success_rewards):.3f}\n")
        f.write(f"Overfit Model (8K): Mean Reward {np.mean(overfit_rewards):.3f}\n")
        f.write(f"Degradation: {reward_degradation:.3f}\n")
    
    print(f"\n✅ Analysis complete! Results saved to: {results_file}")

if __name__ == "__main__":
    compare_models()