#!/usr/bin/env python3
"""
Model Evaluation - Comprehensive evaluation of the 300K trained model
Tests on validation data and provides detailed performance metrics
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def evaluate_model_performance(model, env, num_episodes=10, max_steps_per_episode=1000):
    """Evaluate model performance over multiple episodes"""
    
    print(f"🧪 Running evaluation over {num_episodes} episodes...")
    
    episode_results = []
    all_actions = []
    all_rewards = []
    all_portfolio_values = []
    all_trades = []
    
    for episode in range(num_episodes):
        print(f"   📊 Episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_trades = 0
        episode_actions = []
        episode_portfolio_values = []
        step_count = 0
        
        # Get initial portfolio value
        initial_portfolio = env.portfolio_value
        episode_portfolio_values.append(initial_portfolio)
        
        done = False
        while not done and step_count < max_steps_per_episode:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Convert action to integer if it's an array
            if isinstance(action, np.ndarray):
                action = int(action.item())  # Use .item() for 0-dimensional arrays
            else:
                action = int(action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record data
            episode_reward += reward
            episode_actions.append(action)
            episode_portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
            
            # Count trades
            if (abs(info.get('nvda_position', 0) - info.get('prev_nvda_position', 0)) > 0 or 
                abs(info.get('msft_position', 0) - info.get('prev_msft_position', 0)) > 0):
                episode_trades += 1
            
            all_rewards.append(reward)
            step_count += 1
        
        # Calculate episode metrics
        final_portfolio = episode_portfolio_values[-1]
        total_return = (final_portfolio - initial_portfolio) / initial_portfolio
        
        episode_result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'total_return': total_return,
            'final_portfolio': final_portfolio,
            'initial_portfolio': initial_portfolio,
            'trades': episode_trades,
            'steps': step_count,
            'actions': episode_actions,
            'portfolio_values': episode_portfolio_values
        }
        
        episode_results.append(episode_result)
        all_actions.extend(episode_actions)
        all_portfolio_values.extend(episode_portfolio_values)
        all_trades.append(episode_trades)
        
        print(f"      💰 Return: {total_return:.2%} | Trades: {episode_trades} | Steps: {step_count}")
    
    return episode_results, all_actions, all_rewards, all_portfolio_values

def analyze_trading_behavior(episode_results, all_actions):
    """Analyze the trading behavior patterns"""
    
    print("\n📊 TRADING BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Action distribution
    action_counts = np.bincount(all_actions, minlength=9)
    action_names = [
        'SELL_BOTH', 'SELL_NVDA_HOLD_MSFT', 'SELL_NVDA_BUY_MSFT',
        'HOLD_NVDA_SELL_MSFT', 'HOLD_BOTH', 'HOLD_NVDA_BUY_MSFT',
        'BUY_NVDA_SELL_MSFT', 'BUY_NVDA_HOLD_MSFT', 'BUY_BOTH'
    ]
    
    print("🎯 Action Distribution:")
    for i, (action_name, count) in enumerate(zip(action_names, action_counts)):
        percentage = (count / len(all_actions)) * 100 if len(all_actions) > 0 else 0
        print(f"   {i}: {action_name:<20} | {count:>6} ({percentage:>5.1f}%)")
    
    # Trading frequency analysis
    total_trades = sum(result['trades'] for result in episode_results)
    total_steps = sum(result['steps'] for result in episode_results)
    avg_trades_per_episode = total_trades / len(episode_results)
    trade_frequency = total_trades / total_steps if total_steps > 0 else 0
    
    print(f"\n📈 Trading Frequency:")
    print(f"   Total trades: {total_trades}")
    print(f"   Average trades per episode: {avg_trades_per_episode:.1f}")
    print(f"   Trade frequency: {trade_frequency:.3f} (trades per step)")
    
    return {
        'action_distribution': dict(zip(action_names, action_counts)),
        'total_trades': total_trades,
        'avg_trades_per_episode': avg_trades_per_episode,
        'trade_frequency': trade_frequency
    }

def calculate_performance_metrics(episode_results):
    """Calculate comprehensive performance metrics"""
    
    print("\n📊 PERFORMANCE METRICS")
    print("=" * 50)
    
    # Extract returns
    returns = [result['total_return'] for result in episode_results]
    rewards = [result['total_reward'] for result in episode_results]
    
    # Basic statistics
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Risk metrics
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    
    # Drawdown analysis
    max_drawdown = 0
    peak_portfolio = 0
    
    for result in episode_results:
        portfolio_values = result['portfolio_values']
        for value in portfolio_values:
            if value > peak_portfolio:
                peak_portfolio = value
            drawdown = (peak_portfolio - value) / peak_portfolio if peak_portfolio > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
    
    print(f"💰 Return Metrics:")
    print(f"   Average Return: {avg_return:.2%}")
    print(f"   Return Std Dev: {std_return:.2%}")
    print(f"   Min Return: {min_return:.2%}")
    print(f"   Max Return: {max_return:.2%}")
    print(f"   Win Rate: {win_rate:.1%}")
    
    print(f"\n🎯 Risk Metrics:")
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"   Max Drawdown: {max_drawdown:.2%}")
    
    print(f"\n🏆 Reward Metrics:")
    print(f"   Average Reward: {avg_reward:.4f}")
    print(f"   Reward Std Dev: {std_reward:.4f}")
    
    return {
        'avg_return': avg_return,
        'std_return': std_return,
        'min_return': min_return,
        'max_return': max_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }

def create_evaluation_plots(episode_results, save_dir):
    """Create visualization plots for the evaluation"""
    
    print(f"\n📈 Creating evaluation plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results - 300K Steps', fontsize=16, fontweight='bold')
    
    # 1. Episode Returns
    returns = [result['total_return'] for result in episode_results]
    episodes = [result['episode'] for result in episode_results]
    
    axes[0, 0].bar(episodes, [r * 100 for r in returns], alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Episode Returns (%)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Portfolio Value Evolution (first 3 episodes)
    for i, result in enumerate(episode_results[:3]):
        portfolio_values = result['portfolio_values']
        steps = range(len(portfolio_values))
        axes[0, 1].plot(steps, portfolio_values, label=f'Episode {i+1}', alpha=0.8)
    
    axes[0, 1].set_title('Portfolio Value Evolution (First 3 Episodes)')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Return Distribution
    axes[1, 0].hist(returns, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(x=np.mean(returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(returns):.2%}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative Returns
    cumulative_returns = np.cumsum(returns)
    axes[1, 1].plot(episodes, [r * 100 for r in cumulative_returns], 
                    marker='o', linewidth=2, markersize=6, color='green')
    axes[1, 1].set_title('Cumulative Returns (%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Return (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved plot: {plot_path}")
    
    plt.close()

def save_evaluation_report(episode_results, performance_metrics, trading_behavior, save_dir):
    """Save a comprehensive evaluation report"""
    
    report_path = os.path.join(save_dir, 'evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Report - 300K Steps\n\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: chunk49_final_300000steps.zip\n")
        f.write(f"**Episodes Evaluated**: {len(episode_results)}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write(f"- **Average Return**: {performance_metrics['avg_return']:.2%}\n")
        f.write(f"- **Win Rate**: {performance_metrics['win_rate']:.1%}\n")
        f.write(f"- **Sharpe Ratio**: {performance_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"- **Max Drawdown**: {performance_metrics['max_drawdown']:.2%}\n")
        f.write(f"- **Average Trades per Episode**: {trading_behavior['avg_trades_per_episode']:.1f}\n\n")
        
        f.write("## Detailed Metrics\n\n")
        f.write("### Return Metrics\n")
        f.write(f"- Average Return: {performance_metrics['avg_return']:.4f}\n")
        f.write(f"- Return Std Dev: {performance_metrics['std_return']:.4f}\n")
        f.write(f"- Min Return: {performance_metrics['min_return']:.4f}\n")
        f.write(f"- Max Return: {performance_metrics['max_return']:.4f}\n\n")
        
        f.write("### Trading Behavior\n")
        f.write(f"- Total Trades: {trading_behavior['total_trades']}\n")
        f.write(f"- Trade Frequency: {trading_behavior['trade_frequency']:.3f}\n\n")
        
        f.write("### Action Distribution\n")
        for action, count in trading_behavior['action_distribution'].items():
            percentage = (count / sum(trading_behavior['action_distribution'].values())) * 100
            f.write(f"- {action}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n## Episode Results\n\n")
        f.write("| Episode | Return | Reward | Trades | Steps |\n")
        f.write("|---------|--------|--------|-----------|-------|\n")
        
        for result in episode_results:
            f.write(f"| {result['episode']} | {result['total_return']:.2%} | "
                   f"{result['total_reward']:.4f} | {result['trades']} | {result['steps']} |\n")
    
    print(f"   💾 Saved report: {report_path}")

def main():
    try:
        print("🧪 MODEL EVALUATION - 300K STEPS")
        print("=" * 60)
        print("🎯 Evaluating the fully trained model on validation data")
        print()
        
        # Import after path setup
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        from sb3_contrib import RecurrentPPO
        from secrets_helper import SecretsHelper
        
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        print('📥 Loading validation data...')
        # Use validation split for evaluation
        validation_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2025-08-01', 
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='validation'  # Use validation data for evaluation
        )
        
        # Create environment with validation data
        env = DualTickerTradingEnv(
            nvda_data=validation_data['nvda_features'],
            msft_data=validation_data['msft_features'], 
            nvda_prices=validation_data['nvda_prices'],
            msft_prices=validation_data['msft_prices'],
            trading_days=validation_data['trading_days'],
            initial_capital=100000,
            max_episode_steps=1000,
            max_daily_drawdown_pct=0.02,
            max_position_size=500,
            transaction_cost_pct=0.0001,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            risk_free_rate_annual=0.05,
            log_trades=False,
            verbose=False
        )
        
        # Set to evaluation mode
        env.set_training_mode(False)
        
        print('✅ Validation environment created')
        print(f'   📊 Data length: {len(validation_data["trading_days"])} days')
        
        # Load the final trained model
        model_path = 'train_runs/300k_20250802_1309/checkpoints/chunk49_final_300000steps.zip'
        print(f'🤖 Loading trained model: {model_path}')
        
        model = RecurrentPPO.load(model_path, env=env)
        print(f'✅ Model loaded with {model.num_timesteps:,} training steps')
        
        # Create results directory
        results_dir = 'train_runs/300k_20250802_1309/evaluation_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Run evaluation
        print(f'\n🚀 Starting model evaluation...')
        start_time = time.time()
        
        episode_results, all_actions, all_rewards, all_portfolio_values = evaluate_model_performance(
            model, env, num_episodes=20, max_steps_per_episode=1000
        )
        
        evaluation_time = time.time() - start_time
        
        # Analyze results
        trading_behavior = analyze_trading_behavior(episode_results, all_actions)
        performance_metrics = calculate_performance_metrics(episode_results)
        
        # Create visualizations
        create_evaluation_plots(episode_results, results_dir)
        
        # Save comprehensive report
        save_evaluation_report(episode_results, performance_metrics, trading_behavior, results_dir)
        
        print(f"\n✅ EVALUATION COMPLETED!")
        print(f"   ⏱️ Evaluation time: {evaluation_time:.1f} seconds")
        print(f"   📊 Episodes evaluated: {len(episode_results)}")
        print(f"   💾 Results saved to: {results_dir}")
        
        # Summary
        print(f"\n🎯 QUICK SUMMARY:")
        print(f"   💰 Average Return: {performance_metrics['avg_return']:.2%}")
        print(f"   🏆 Win Rate: {performance_metrics['win_rate']:.1%}")
        print(f"   📈 Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        print(f"   🔄 Avg Trades/Episode: {trading_behavior['avg_trades_per_episode']:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)