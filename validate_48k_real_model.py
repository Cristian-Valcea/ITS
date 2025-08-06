#!/usr/bin/env python3
"""
🎯 48K REAL DATA MODEL VALIDATION
Evaluate the trained model's performance on real market data
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from stable_baselines3 import PPO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str):
    """Load the trained PPO model"""
    try:
        model = PPO.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def create_validation_environment():
    """Create environment with same real data for validation"""
    try:
        # Database configuration with vault password
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        # Load real market data
        logger.info("📈 Loading real market data for validation...")
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        
        market_data = adapter.load_training_data(
            start_date='2022-01-03',
            end_date='2024-12-31',
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='train'
        )
        
        # Extract the correct data structure
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        # Combine features for dual-ticker environment (26-dim observation)
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        
        # Add 2 position features (initialized to 0)
        position_features = np.zeros((combined_features.shape[0], 2))
        combined_features = np.concatenate([combined_features, position_features], axis=1)
        
        # Create 4-column price data
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        
        combined_prices = np.column_stack([
            nvda_prices,     # Column 0: NVDA close prices
            nvda_returns,    # Column 1: NVDA returns  
            msft_prices,     # Column 2: MSFT close prices
            msft_returns     # Column 3: MSFT returns
        ])
        
        # Create environment
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=10000.0,
            lookback_window=50,
            max_episode_steps=1000,
            transaction_cost_pct=0.001
        )
        
        logger.info(f"✅ Validation environment created with {len(trading_days)} timesteps")
        return env
        
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        return None

def run_validation_episodes(model, env, num_episodes=10):
    """Run validation episodes and collect performance metrics"""
    
    logger.info(f"🧪 Running {num_episodes} validation episodes...")
    
    episode_results = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_step = 0
        total_reward = 0
        actions_taken = []
        portfolio_values = []
        
        initial_value = 10000.0
        
        while not done and episode_step < 1000:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, _, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            actions_taken.append(action)
            portfolio_values.append(info['portfolio_value'])
            
            episode_step += 1
        
        # Calculate episode metrics
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate hold rate
        hold_actions = sum(1 for a in actions_taken if a == 4)  # Action 4 = Hold Both
        hold_rate = hold_actions / len(actions_taken) if actions_taken else 0
        
        # Calculate max drawdown
        peak_value = initial_value
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        episode_result = {
            'episode': episode + 1,
            'total_return': total_return,
            'final_value': final_value,
            'total_reward': total_reward,
            'steps': episode_step,
            'hold_rate': hold_rate,
            'max_drawdown': max_drawdown,
            'trades': len(actions_taken) - hold_actions
        }
        
        episode_results.append(episode_result)
        
        logger.info(f"Episode {episode + 1}: Return {total_return:.2%}, Hold Rate {hold_rate:.1%}, Drawdown {max_drawdown:.2%}")
    
    return episode_results

def analyze_results(episode_results):
    """Analyze validation results and generate report"""
    
    logger.info("📊 Analyzing validation results...")
    
    df = pd.DataFrame(episode_results)
    
    # Summary statistics
    summary = {
        'episodes': len(episode_results),
        'avg_return': df['total_return'].mean(),
        'std_return': df['total_return'].std(),
        'win_rate': (df['total_return'] > 0).mean(),
        'avg_hold_rate': df['hold_rate'].mean(),
        'std_hold_rate': df['hold_rate'].std(),
        'avg_max_drawdown': df['max_drawdown'].mean(),
        'worst_drawdown': df['max_drawdown'].max(),
        'avg_trades_per_episode': df['trades'].mean(),
        'sharpe_ratio': df['total_return'].mean() / df['total_return'].std() if df['total_return'].std() > 0 else 0
    }
    
    return summary, df

def main():
    """Main validation execution"""
    
    logger.info("🎯 48K REAL DATA MODEL VALIDATION")
    logger.info("=" * 60)
    
    # Load trained model
    model_path = "train_runs/real_data_48k_20250804_103603/final_model.zip"
    model = load_trained_model(model_path)
    
    if not model:
        logger.error("❌ Failed to load model - validation cannot proceed")
        return False
    
    # Create validation environment
    env = create_validation_environment()
    
    if not env:
        logger.error("❌ Failed to create environment - validation cannot proceed")  
        return False
    
    # Run validation episodes
    episode_results = run_validation_episodes(model, env, num_episodes=20)
    
    # Analyze results
    summary, results_df = analyze_results(episode_results)
    
    # Generate report
    logger.info("🎉 VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info("📊 PERFORMANCE SUMMARY:")
    logger.info(f"  Episodes: {summary['episodes']}")
    logger.info(f"  Average Return: {summary['avg_return']:.2%} ± {summary['std_return']:.2%}")
    logger.info(f"  Win Rate: {summary['win_rate']:.1%}")
    logger.info(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    logger.info(f"  Average Hold Rate: {summary['avg_hold_rate']:.1%} ± {summary['std_hold_rate']:.1%}")
    logger.info(f"  Average Max Drawdown: {summary['avg_max_drawdown']:.2%}")
    logger.info(f"  Worst Drawdown: {summary['worst_drawdown']:.2%}")
    logger.info(f"  Average Trades/Episode: {summary['avg_trades_per_episode']:.1f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"validation_results_48k_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"📁 Detailed results saved to: {results_path}")
    
    # Performance assessment
    logger.info("🎯 ASSESSMENT:")
    if summary['avg_return'] > 0.02:  # 2% average return
        logger.info("  ✅ Return Performance: EXCELLENT")
    elif summary['avg_return'] > 0:
        logger.info("  ✅ Return Performance: POSITIVE")
    else:
        logger.info("  ⚠️ Return Performance: NEEDS IMPROVEMENT")
    
    if 0.4 <= summary['avg_hold_rate'] <= 0.7:  # Target hold rate range
        logger.info("  ✅ Hold Rate: OPTIMAL")
    elif summary['avg_hold_rate'] > 0.8:
        logger.info("  ⚠️ Hold Rate: TOO HIGH (over-conservative)")
    else:
        logger.info("  ⚠️ Hold Rate: TOO LOW (over-trading)")
    
    if summary['avg_max_drawdown'] < 0.05:  # Less than 5% drawdown
        logger.info("  ✅ Risk Management: EXCELLENT")
    elif summary['avg_max_drawdown'] < 0.1:
        logger.info("  ✅ Risk Management: GOOD")
    else:
        logger.info("  ⚠️ Risk Management: NEEDS IMPROVEMENT")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 48K REAL DATA MODEL VALIDATION: SUCCESS")
        sys.exit(0)
    else:
        print("❌ 48K REAL DATA MODEL VALIDATION: FAILED")
        sys.exit(1)