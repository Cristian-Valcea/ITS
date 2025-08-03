#!/usr/bin/env python3
"""
🧪 WALK-FORWARD BACKTEST - INSTITUTIONAL VALIDATION
6-month held-out validation with Sharpe, max-DD, and trade frequency analysis
"""

import os
import sys
import time
import yaml
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def load_validation_environment(test_start: str, test_end: str):
    """Load validation environment with held-out data"""
    
    from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
    from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
    from secrets_helper import SecretsHelper
    
    print(f"📥 Loading validation data: {test_start} to {test_end}")
    
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
    
    # Load validation data
    validation_data = data_adapter.load_training_data(
        start_date=test_start,
        end_date=test_end,
        symbols=['NVDA', 'MSFT'],
        bar_size='1min',
        data_split='validation'
    )
    
    # Prepare data for V3 environment
    nvda_features = validation_data['nvda_features']
    msft_features = validation_data['msft_features']
    nvda_prices = validation_data['nvda_prices']
    
    # Combine features
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    price_series = nvda_prices
    
    # Create V3 validation environment
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
    
    print(f"✅ Validation environment created:")
    print(f"   📊 Data length: {len(validation_data['trading_days'])} days")
    print(f"   📊 Features: {combined_features.shape}")
    
    return env, validation_data

def run_walk_forward_episodes(model, env, num_episodes: int = 50):
    """Run walk-forward validation episodes"""
    
    print(f"🧪 Running {num_episodes} validation episodes...")
    
    episode_results = []
    all_returns = []
    all_trades = []
    all_portfolio_values = []
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"   📊 Episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        episode_return = 0
        episode_trades = 0
        episode_portfolio_values = []
        
        # Get initial portfolio value
        initial_portfolio = env.portfolio_value
        episode_portfolio_values.append(initial_portfolio)
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Convert action to integer
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record data
            episode_portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
            
            # Count trades (simplified)
            if action != 4:  # Not HOLD_BOTH
                episode_trades += 1
            
            step_count += 1
        
        # Calculate episode metrics
        final_portfolio = episode_portfolio_values[-1]
        episode_return = (final_portfolio - initial_portfolio) / initial_portfolio
        
        episode_results.append({
            'episode': episode + 1,
            'return': episode_return,
            'trades': episode_trades,
            'steps': step_count,
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'portfolio_values': episode_portfolio_values
        })
        
        all_returns.append(episode_return)
        all_trades.append(episode_trades)
        all_portfolio_values.extend(episode_portfolio_values)
    
    return episode_results, all_returns, all_trades

def calculate_performance_metrics(episode_results: List[Dict], all_returns: List[float]) -> Dict:
    """Calculate comprehensive performance metrics"""
    
    print("📊 Calculating performance metrics...")
    
    # Basic return statistics
    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    min_return = np.min(all_returns)
    max_return = np.max(all_returns)
    
    # Risk metrics
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    win_rate = sum(1 for r in all_returns if r > 0) / len(all_returns)
    
    # Drawdown analysis
    max_drawdown = 0
    peak_value = 0
    
    for result in episode_results:
        portfolio_values = result['portfolio_values']
        for value in portfolio_values:
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
    
    # Trading frequency
    avg_trades = np.mean([r['trades'] for r in episode_results])
    avg_steps = np.mean([r['steps'] for r in episode_results])
    trade_frequency = avg_trades / avg_steps if avg_steps > 0 else 0
    
    # Annualized metrics (assuming daily episodes)
    trading_days_per_year = 252
    annualized_return = avg_return * trading_days_per_year
    annualized_volatility = std_return * np.sqrt(trading_days_per_year)
    annualized_sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    metrics = {
        'avg_return': avg_return,
        'std_return': std_return,
        'min_return': min_return,
        'max_return': max_return,
        'sharpe_ratio': sharpe_ratio,
        'annualized_sharpe': annualized_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'avg_trades_per_episode': avg_trades,
        'trade_frequency': trade_frequency,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility
    }
    
    return metrics

def create_validation_plots(episode_results: List[Dict], metrics: Dict, save_dir: str):
    """Create validation performance plots"""
    
    print("📈 Creating validation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Walk-Forward Validation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode returns
    returns = [r['return'] for r in episode_results]
    axes[0, 0].plot(returns, 'b-', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Return distribution
    axes[0, 1].hist(returns, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
    axes[0, 1].set_title('Return Distribution')
    axes[0, 1].set_xlabel('Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Trades per episode
    trades = [r['trades'] for r in episode_results]
    axes[1, 0].plot(trades, 'orange', alpha=0.7)
    axes[1, 0].set_title('Trades per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative portfolio value (sample episode)
    if episode_results:
        sample_episode = episode_results[0]
        portfolio_values = sample_episode['portfolio_values']
        axes[1, 1].plot(portfolio_values, 'purple', alpha=0.8)
        axes[1, 1].set_title('Sample Episode Portfolio Value')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{save_dir}/validation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Plots saved: {plot_path}")

def save_validation_results(metrics: Dict, episode_results: List[Dict], save_dir: str):
    """Save validation results to files"""
    
    # Save metrics
    metrics_path = f'{save_dir}/validation_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    # Save episode results
    results_df = pd.DataFrame([
        {
            'episode': r['episode'],
            'return': r['return'],
            'trades': r['trades'],
            'steps': r['steps'],
            'final_portfolio': r['final_portfolio']
        }
        for r in episode_results
    ])
    
    results_path = f'{save_dir}/validation_episodes.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"   💾 Metrics saved: {metrics_path}")
    print(f"   💾 Episodes saved: {results_path}")

def create_validation_report(metrics: Dict, save_dir: str, test_start: str, test_end: str):
    """Create comprehensive validation report"""
    
    report_path = f'{save_dir}/VALIDATION_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# 🧪 Walk-Forward Validation Report\n\n")
        f.write(f"**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Test Period**: {test_start} to {test_end}\n")
        f.write(f"**Model**: V3 Gold Standard (400K steps)\n\n")
        
        f.write("## 📊 Performance Summary\n\n")
        f.write(f"- **Average Return**: {metrics['avg_return']:.2%}\n")
        f.write(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}\n")
        f.write(f"- **Annualized Sharpe**: {metrics['annualized_sharpe']:.3f}\n")
        f.write(f"- **Win Rate**: {metrics['win_rate']:.1%}\n")
        f.write(f"- **Max Drawdown**: {metrics['max_drawdown']:.2%}\n")
        f.write(f"- **Annualized Return**: {metrics['annualized_return']:.1%}\n")
        f.write(f"- **Annualized Volatility**: {metrics['annualized_volatility']:.1%}\n\n")
        
        f.write("## 🔄 Trading Behavior\n\n")
        f.write(f"- **Avg Trades/Episode**: {metrics['avg_trades_per_episode']:.1f}\n")
        f.write(f"- **Trade Frequency**: {metrics['trade_frequency']:.3f}\n\n")
        
        f.write("## 🎯 Risk Assessment\n\n")
        f.write(f"- **Return Volatility**: {metrics['std_return']:.2%}\n")
        f.write(f"- **Min Return**: {metrics['min_return']:.2%}\n")
        f.write(f"- **Max Return**: {metrics['max_return']:.2%}\n\n")
        
        f.write("## ✅ Validation Status\n\n")
        
        # Check thresholds
        sharpe_threshold = 0.0
        dd_threshold = 0.02
        
        sharpe_pass = metrics['sharpe_ratio'] >= sharpe_threshold
        dd_pass = metrics['max_drawdown'] <= dd_threshold
        
        f.write(f"- **Sharpe Ratio**: {'✅ PASS' if sharpe_pass else '❌ FAIL'} (>= {sharpe_threshold})\n")
        f.write(f"- **Max Drawdown**: {'✅ PASS' if dd_pass else '❌ FAIL'} (<= {dd_threshold:.1%})\n\n")
        
        if sharpe_pass and dd_pass:
            f.write("🎉 **VALIDATION PASSED** - Model ready for live paper trading!\n\n")
        else:
            f.write("⚠️ **VALIDATION FAILED** - Iterate reward coefficients before demo.\n\n")
        
        f.write("## 📁 Files Generated\n\n")
        f.write("- `validation_metrics.yaml` - Detailed metrics\n")
        f.write("- `validation_episodes.csv` - Episode-by-episode results\n")
        f.write("- `validation_plots.png` - Performance visualizations\n")
        f.write("- `VALIDATION_REPORT.md` - This report\n")
    
    print(f"📋 Validation report created: {report_path}")

def main():
    try:
        print("🧪 WALK-FORWARD VALIDATION - INSTITUTIONAL STANDARD")
        print("=" * 60)
        
        # Configuration
        test_start = "2025-02-01"  # 6-month hold-out
        test_end = "2025-07-31"
        model_path = None  # Will be determined from latest training run
        
        # Find latest V3 model
        train_runs_dir = Path("train_runs")
        v3_runs = [d for d in train_runs_dir.iterdir() if d.is_dir() and "v3_gold_standard" in d.name]
        
        if not v3_runs:
            print("❌ No V3 training runs found")
            return 1
        
        # Get latest run
        latest_run = max(v3_runs, key=lambda x: x.stat().st_mtime)
        
        # Find final model
        model_files = list(latest_run.glob("*final*.zip"))
        if not model_files:
            print(f"❌ No final model found in {latest_run}")
            return 1
        
        model_path = model_files[0]
        print(f"🤖 Using model: {model_path}")
        
        # Load validation environment
        env, validation_data = load_validation_environment(test_start, test_end)
        
        # Load model
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, env=env)
        print("✅ Model loaded for validation")
        
        # Run validation
        start_time = time.time()
        episode_results, all_returns, all_trades = run_walk_forward_episodes(model, env, num_episodes=50)
        validation_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_performance_metrics(episode_results, all_returns)
        
        # Create results directory
        results_dir = latest_run / "validation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Create plots
        create_validation_plots(episode_results, metrics, str(results_dir))
        
        # Save results
        save_validation_results(metrics, episode_results, str(results_dir))
        
        # Create report
        create_validation_report(metrics, str(results_dir), test_start, test_end)
        
        print(f"\n✅ VALIDATION COMPLETED!")
        print(f"   ⏱️ Time: {validation_time:.1f} seconds")
        print(f"   📊 Episodes: {len(episode_results)}")
        print(f"   💾 Results: {results_dir}")
        
        # Summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   💰 Average Return: {metrics['avg_return']:.2%}")
        print(f"   📈 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   🏆 Win Rate: {metrics['win_rate']:.1%}")
        print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   🔄 Avg Trades: {metrics['avg_trades_per_episode']:.1f}")
        
        # Check validation status
        sharpe_pass = metrics['sharpe_ratio'] >= 0.0
        dd_pass = metrics['max_drawdown'] <= 0.02
        
        if sharpe_pass and dd_pass:
            print(f"\n🎉 VALIDATION PASSED - Ready for live paper trading!")
            return 0
        else:
            print(f"\n⚠️ VALIDATION FAILED - Iterate reward coefficients")
            return 1
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)