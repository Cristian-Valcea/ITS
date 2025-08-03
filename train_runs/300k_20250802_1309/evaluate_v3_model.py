#!/usr/bin/env python3
"""
🧪 EVALUATE V3-TRAINED MODEL
Evaluate the model trained with V3 environment and compare with original models
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

def create_v3_validation_environment(data_adapter):
    """Create V3 environment for validation"""
    
    print('📥 Loading validation data for V3 evaluation...')
    
    # Load validation data
    validation_data = data_adapter.load_training_data(
        start_date='2024-02-01',
        end_date='2025-08-01',
        symbols=['NVDA', 'MSFT'],
        bar_size='1min',
        data_split='validation'
    )
    
    # Prepare data for V3 environment
    nvda_features = validation_data['nvda_features']
    msft_features = validation_data['msft_features']
    nvda_prices = validation_data['nvda_prices']
    
    # Combine features: [NVDA features, MSFT features]
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    price_series = nvda_prices
    
    print(f'✅ Validation data prepared:')
    print(f'   📊 Combined features: {combined_features.shape}')
    print(f'   📊 Price series: {len(price_series)}')
    
    return combined_features, price_series, validation_data

def setup_v3_validation_env(combined_features, price_series):
    """Setup V3 environment for evaluation"""
    
    from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
    
    env = DualTickerTradingEnvV3(
        processed_feature_data=combined_features,
        price_data=price_series,
        initial_capital=100000,
        lookback_window=50,
        max_episode_steps=1000,
        max_daily_drawdown_pct=0.02,
        max_position_size=500,
        transaction_cost_pct=0.0001,
        
        # V3 reward parameters (same as training)
        base_impact_bp=68.0,
        impact_exponent=0.5,
        risk_free_rate_annual=0.05,
        
        log_trades=False,
        verbose=False
    )
    
    print('✅ V3 validation environment created')
    return env

def evaluate_v3_model_performance(model, env, num_episodes=20, max_steps_per_episode=1000):
    """Evaluate V3 model performance"""
    
    print(f"🧪 Running V3 model evaluation over {num_episodes} episodes...")
    
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
                action = int(action.item())
            else:
                action = int(action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record data
            episode_reward += reward
            episode_actions.append(action)
            episode_portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
            
            # Count trades (simplified)
            if action != 4:  # Not HOLD_BOTH
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

def analyze_v3_trading_behavior(episode_results, all_actions):
    """Analyze V3 model trading behavior"""
    
    print("\n📊 V3 TRADING BEHAVIOR ANALYSIS")
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

def calculate_v3_performance_metrics(episode_results):
    """Calculate V3 model performance metrics"""
    
    print("\n📊 V3 PERFORMANCE METRICS")
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

def create_v3_comparison_report(v3_results, v3_behavior, v3_metrics, save_dir):
    """Create comprehensive comparison report"""
    
    report_path = os.path.join(save_dir, 'v3_evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# 🌟 V3 Model Evaluation Report\n\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: V3-trained (200K→300K with improved reward system)\n")
        f.write(f"**Episodes Evaluated**: {len(v3_results)}\n\n")
        
        f.write("## 🎯 V3 Environment Features\n\n")
        f.write("The V3 environment includes several improvements:\n")
        f.write("- **Risk-free baseline**: Prevents cost-blind trading\n")
        f.write("- **Embedded impact costs**: Kyle lambda model (68bp calibrated)\n")
        f.write("- **Hold bonus**: Incentivizes doing nothing when no alpha\n")
        f.write("- **Action change penalties**: Reduces overtrading\n")
        f.write("- **Ticket costs**: Fixed costs per trade ($25)\n")
        f.write("- **Downside penalties**: Risk management\n\n")
        
        f.write("## 📊 Performance Summary\n\n")
        f.write(f"- **Average Return**: {v3_metrics['avg_return']:.2%}\n")
        f.write(f"- **Win Rate**: {v3_metrics['win_rate']:.1%}\n")
        f.write(f"- **Sharpe Ratio**: {v3_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"- **Max Drawdown**: {v3_metrics['max_drawdown']:.2%}\n")
        f.write(f"- **Average Trades per Episode**: {v3_behavior['avg_trades_per_episode']:.1f}\n\n")
        
        f.write("## 🎯 Trading Behavior\n\n")
        f.write("### Action Distribution\n")
        for action, count in v3_behavior['action_distribution'].items():
            percentage = (count / sum(v3_behavior['action_distribution'].values())) * 100
            f.write(f"- **{action}**: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\n### Trading Frequency\n")
        f.write(f"- **Total Trades**: {v3_behavior['total_trades']}\n")
        f.write(f"- **Trade Frequency**: {v3_behavior['trade_frequency']:.3f}\n\n")
        
        f.write("## 📈 Episode Results\n\n")
        f.write("| Episode | Return | Reward | Trades | Steps |\n")
        f.write("|---------|--------|--------|-----------|-------|\n")
        
        for result in v3_results:
            f.write(f"| {result['episode']} | {result['total_return']:.2%} | "
                   f"{result['total_reward']:.4f} | {result['trades']} | {result['steps']} |\n")
    
    print(f"   💾 V3 report saved: {report_path}")

def main():
    try:
        print("🌟 V3 MODEL EVALUATION")
        print("=" * 60)
        print("🧪 Evaluating model trained with improved V3 environment")
        print()
        
        # Import after path setup
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
        
        # Create data adapter
        data_adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        # Create V3 validation environment
        combined_features, price_series, validation_data = create_v3_validation_environment(data_adapter)
        env = setup_v3_validation_env(combined_features, price_series)
        
        # Load V3-trained model
        v3_model_path = 'train_runs/v3_from_200k_20250802_183726/v3_model_final_100000steps.zip'
        
        if not os.path.exists(v3_model_path):
            print(f"❌ V3 model not found at: {v3_model_path}")
            return 1
        
        print(f'🤖 Loading V3-trained model: {v3_model_path}')
        model = RecurrentPPO.load(v3_model_path, env=env)
        print('✅ V3 model loaded')
        
        # Evaluate V3 model
        print(f'\n🚀 Starting V3 model evaluation...')
        start_time = time.time()
        
        episode_results, all_actions, all_rewards, all_portfolio_values = evaluate_v3_model_performance(
            model, env, num_episodes=20, max_steps_per_episode=1000
        )
        
        evaluation_time = time.time() - start_time
        
        # Analyze results
        trading_behavior = analyze_v3_trading_behavior(episode_results, all_actions)
        performance_metrics = calculate_v3_performance_metrics(episode_results)
        
        # Create results directory
        results_dir = 'train_runs/300k_20250802_1309/evaluation_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save comprehensive report
        create_v3_comparison_report(episode_results, trading_behavior, performance_metrics, results_dir)
        
        print(f"\n✅ V3 MODEL EVALUATION COMPLETED!")
        print(f"   ⏱️ Evaluation time: {evaluation_time:.1f} seconds")
        print(f"   📊 Episodes evaluated: {len(episode_results)}")
        print(f"   💾 Results saved to: {results_dir}")
        
        # Summary
        print(f"\n🌟 V3 MODEL SUMMARY:")
        print(f"   💰 Average Return: {performance_metrics['avg_return']:.2%}")
        print(f"   🏆 Win Rate: {performance_metrics['win_rate']:.1%}")
        print(f"   📈 Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        print(f"   🔄 Avg Trades/Episode: {trading_behavior['avg_trades_per_episode']:.1f}")
        
        # Quick comparison with previous results
        print(f"\n📊 QUICK COMPARISON:")
        print(f"   Original 200K: -0.05% return, 184 trades/episode")
        print(f"   Original 300K: -0.05% return, 126 trades/episode")
        print(f"   V3 300K:       {performance_metrics['avg_return']:.2%} return, {trading_behavior['avg_trades_per_episode']:.1f} trades/episode")
        
        return 0
        
    except Exception as e:
        print(f"❌ V3 evaluation failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)