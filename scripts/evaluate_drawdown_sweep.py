#!/usr/bin/env python3
"""
🔍 DRAWDOWN SWEEP EVALUATION
Evaluate all drawdown sweep models and generate comprehensive analysis
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
from gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from gym_env.refined_reward_system import RefinedRewardSystem
import gymnasium as gym

def create_evaluation_environment(config_path, drawdown_limit):
    """Create evaluation environment with specified drawdown limit"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override drawdown limit
    config['environment']['max_drawdown_pct'] = drawdown_limit
    
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
    try:
        market_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2024-02-29',
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='eval'
        )
    except Exception as e:
        print(f"❌ Failed to load market data: {e}")
        return None
    
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
        max_daily_drawdown_pct=drawdown_limit,
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
            
            # Calculate refined reward
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

def evaluate_model(checkpoint_path, config_path, drawdown_limit, num_episodes=10):
    """Evaluate a single model"""
    print(f"🔍 Evaluating DD={drawdown_limit} model...")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Create environment
    env = create_evaluation_environment(config_path, drawdown_limit)
    if env is None:
        return None
    
    # Load model
    try:
        model = PPO.load(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Run evaluation episodes
    episode_results = []
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            done = terminated or truncated
        
        episode_results.append({
            'episode': episode,
            'length': episode_steps,
            'reward': episode_reward,
            'final_portfolio_value': info.get('portfolio_value', 10000.0),
            'final_drawdown': info.get('drawdown_pct', 0.0),
            'terminated': terminated,
            'truncated': truncated
        })
    
    # Calculate summary statistics
    episode_lengths = [ep['length'] for ep in episode_results]
    episode_rewards = [ep['reward'] for ep in episode_results]
    
    results = {
        'drawdown_limit': drawdown_limit,
        'num_episodes': num_episodes,
        'total_steps': total_steps,
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'min_episode_length': np.min(episode_lengths),
        'max_episode_length': np.max(episode_lengths),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'episodes': episode_results
    }
    
    print(f"   Mean episode length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    print(f"   Episode length range: {results['min_episode_length']}-{results['max_episode_length']}")
    print(f"   Mean reward: {results['mean_reward']:.3f}")
    
    return results

def main():
    print("🔍 DRAWDOWN SWEEP EVALUATION")
    print("=" * 50)
    
    base_config = "config/reward_shim.yaml"
    drawdown_limits = [0.30, 0.40, 0.50, 0.75]
    
    results = {}
    
    for dd in drawdown_limits:
        checkpoint_path = f"diagnostic_runs/drawdown_sweep/checkpoints/dd_{dd}_15k.zip"
        
        result = evaluate_model(checkpoint_path, base_config, dd, num_episodes=10)
        if result:
            results[dd] = result
    
    if not results:
        print("❌ No successful evaluations")
        return
    
    # Create summary
    print(f"\n📊 DRAWDOWN SWEEP SUMMARY")
    print("=" * 50)
    
    summary_data = []
    for dd, result in results.items():
        summary_data.append({
            'drawdown_limit': dd,
            'mean_episode_length': result['mean_episode_length'],
            'max_episode_length': result['max_episode_length'],
            'mean_reward': result['mean_reward'],
            'total_episodes': result['num_episodes']
        })
        
        print(f"DD {dd*100:.0f}%: {result['mean_episode_length']:.1f} steps avg, {result['max_episode_length']} max, {result['mean_reward']:.3f} reward")
    
    # Save results
    output_dir = Path("diagnostic_runs/drawdown_sweep/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'drawdown_sweep_analysis.csv', index=False)
    
    # Create visualization
    create_analysis_plots(results, output_dir)
    
    # Generate insights
    generate_insights(results)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

def create_analysis_plots(results, output_dir):
    """Create analysis plots"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    drawdown_limits = list(results.keys())
    
    # Episode length comparison
    mean_lengths = [results[dd]['mean_episode_length'] for dd in drawdown_limits]
    std_lengths = [results[dd]['std_episode_length'] for dd in drawdown_limits]
    
    axes[0, 0].bar([f"{dd*100:.0f}%" for dd in drawdown_limits], mean_lengths, 
                   yerr=std_lengths, alpha=0.7, color='skyblue', capsize=5)
    axes[0, 0].set_title('Mean Episode Length by Drawdown Limit')
    axes[0, 0].set_ylabel('Episode Length (steps)')
    axes[0, 0].axhline(y=80, color='green', linestyle='--', label='Target (80 steps)')
    axes[0, 0].axhline(y=390, color='red', linestyle='--', label='Maximum (390 steps)')
    axes[0, 0].legend()
    
    # Max episode length
    max_lengths = [results[dd]['max_episode_length'] for dd in drawdown_limits]
    
    axes[0, 1].bar([f"{dd*100:.0f}%" for dd in drawdown_limits], max_lengths, 
                   alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Maximum Episode Length by Drawdown Limit')
    axes[0, 1].set_ylabel('Max Episode Length (steps)')
    axes[0, 1].axhline(y=390, color='red', linestyle='--', label='Theoretical Max')
    axes[0, 1].legend()
    
    # Mean reward comparison
    mean_rewards = [results[dd]['mean_reward'] for dd in drawdown_limits]
    colors = ['red' if r < 0 else 'green' for r in mean_rewards]
    
    axes[1, 0].bar([f"{dd*100:.0f}%" for dd in drawdown_limits], mean_rewards, 
                   alpha=0.7, color=colors)
    axes[1, 0].set_title('Mean Reward by Drawdown Limit')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Episode length distribution
    for i, dd in enumerate(drawdown_limits):
        episode_lengths = [ep['length'] for ep in results[dd]['episodes']]
        axes[1, 1].hist(episode_lengths, alpha=0.5, label=f"{dd*100:.0f}%", bins=20)
    
    axes[1, 1].set_title('Episode Length Distribution')
    axes[1, 1].set_xlabel('Episode Length (steps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown_sweep_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_insights(results):
    """Generate insights from the results"""
    print(f"\n🧠 DRAWDOWN SWEEP INSIGHTS")
    print("=" * 50)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    drawdown_limits = sorted(results.keys())
    
    # Find best performing drawdown limit
    best_length_dd = max(drawdown_limits, key=lambda dd: results[dd]['mean_episode_length'])
    best_reward_dd = max(drawdown_limits, key=lambda dd: results[dd]['mean_reward'])
    
    print(f"🎯 BEST EPISODE LENGTH: {best_length_dd*100:.0f}% drawdown limit")
    print(f"   Mean length: {results[best_length_dd]['mean_episode_length']:.1f} steps")
    print(f"   Max length: {results[best_length_dd]['max_episode_length']} steps")
    
    print(f"\n💰 BEST REWARD: {best_reward_dd*100:.0f}% drawdown limit")
    print(f"   Mean reward: {results[best_reward_dd]['mean_reward']:.3f}")
    
    # Check if target episode length achieved
    target_achieved = False
    for dd in drawdown_limits:
        if results[dd]['mean_episode_length'] >= 80:
            print(f"\n✅ TARGET ACHIEVED: {dd*100:.0f}% drawdown allows ≥80 step episodes")
            target_achieved = True
            break
    
    if not target_achieved:
        print(f"\n⚠️  TARGET NOT ACHIEVED: No drawdown limit allows ≥80 step episodes")
        print(f"   Best achieved: {results[best_length_dd]['mean_episode_length']:.1f} steps")
    
    # Analyze improvement trend
    improvements = []
    for i in range(1, len(drawdown_limits)):
        prev_dd = drawdown_limits[i-1]
        curr_dd = drawdown_limits[i]
        
        length_improvement = results[curr_dd]['mean_episode_length'] - results[prev_dd]['mean_episode_length']
        reward_improvement = results[curr_dd]['mean_reward'] - results[prev_dd]['mean_reward']
        
        improvements.append({
            'from': prev_dd,
            'to': curr_dd,
            'length_improvement': length_improvement,
            'reward_improvement': reward_improvement
        })
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    for imp in improvements:
        print(f"   {imp['from']*100:.0f}% → {imp['to']*100:.0f}%: "
              f"{imp['length_improvement']:+.1f} steps, {imp['reward_improvement']:+.3f} reward")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if target_achieved:
        print(f"   ✅ Use {best_length_dd*100:.0f}% drawdown limit for next experiments")
        print(f"   🔄 Proceed to Step 3: Optimizer mini-grid")
    else:
        print(f"   ⚠️  Consider even higher drawdown limits (85%, 90%)")
        print(f"   🔍 Investigate data quality issues (Step 4)")
        print(f"   🎛️  Try different optimizer settings regardless")
    
    if all(results[dd]['mean_reward'] < 0 for dd in drawdown_limits):
        print(f"   💸 All models still losing money - fundamental training issues remain")

if __name__ == "__main__":
    main()