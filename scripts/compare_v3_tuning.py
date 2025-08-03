#!/usr/bin/env python3
"""
🎯 V3 TUNING COMPARISON SCRIPT
Compare original V3 model with tuned version for trading behavior analysis

ANALYSIS:
- Trading frequency comparison
- Reward component breakdown
- Performance metrics
- Holding vs trading behavior
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sb3_contrib import RecurrentPPO
from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
from src.gym_env.dual_ticker_trading_env_v3_tuned import DualTickerTradingEnvV3Tuned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V3TuningComparator:
    """
    Compare original V3 model with tuned version
    """
    
    def __init__(self, original_model_path: str, tuned_model_path: str):
        """
        Initialize comparator
        
        Args:
            original_model_path: Path to original V3 model
            tuned_model_path: Path to tuned V3 model
        """
        self.original_model_path = original_model_path
        self.tuned_model_path = tuned_model_path
        
        # Load models
        self.original_model = RecurrentPPO.load(original_model_path)
        self.tuned_model = RecurrentPPO.load(tuned_model_path)
        
        logger.info("✅ Models loaded for comparison")
        logger.info(f"   Original: {original_model_path}")
        logger.info(f"   Tuned: {tuned_model_path}")
    
    def _create_test_data(self, n_timesteps: int = 1000):
        """
        Create test data for evaluation
        
        Args:
            n_timesteps: Number of timesteps to generate
            
        Returns:
            Tuple of (features, prices, timestamps)
        """
        # Generate realistic test data
        features = np.random.randn(n_timesteps, 26).astype(np.float32)
        
        # Generate correlated price movements
        base_prices = np.array([500.0, 500.0, 300.0, 300.0])
        price_changes = np.random.randn(n_timesteps, 4) * 0.015  # 1.5% volatility
        prices = base_prices * (1 + price_changes.cumsum(axis=0))
        
        timestamps = pd.date_range('2025-01-01', periods=n_timesteps, freq='1min')
        
        return features, prices, timestamps.values
    
    def _run_episode(self, model, env, max_steps: int = 1000):
        """
        Run a single episode and collect statistics
        
        Args:
            model: Model to evaluate
            env: Environment to run in
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with episode statistics
        """
        obs = env.reset()
        done = False
        step = 0
        
        # Statistics tracking
        actions = []
        rewards = []
        portfolio_values = []
        trades = []
        positions = []
        reward_components_list = []
        
        while not done and step < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Collect statistics
            actions.append(action)
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            trades.append(info['tuning_info']['traded_this_step'] if 'tuning_info' in info else 
                         (info.get('nvda_trade', 0) != 0 or info.get('msft_trade', 0) != 0))
            positions.append((info['nvda_position'], info['msft_position']))
            reward_components_list.append(info['reward_components'])
            
            step += 1
        
        # Calculate statistics
        total_trades = sum(trades)
        holding_actions = sum(1 for a in actions if a == 4)  # Action 4 = Hold, Hold
        holding_percentage = holding_actions / len(actions) * 100
        
        # Portfolio performance
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Reward component analysis
        reward_df = pd.DataFrame(reward_components_list)
        avg_reward_components = reward_df.mean().to_dict()
        
        return {
            'total_steps': step,
            'total_trades': total_trades,
            'trades_per_step': total_trades / step,
            'holding_percentage': holding_percentage,
            'total_return': total_return,
            'final_portfolio_value': final_value,
            'avg_reward_components': avg_reward_components,
            'actions': actions,
            'rewards': rewards,
            'portfolio_values': portfolio_values,
            'reward_components': reward_components_list
        }
    
    def compare_models(self, n_episodes: int = 5):
        """
        Compare both models across multiple episodes
        
        Args:
            n_episodes: Number of episodes to run for each model
            
        Returns:
            Comparison results dictionary
        """
        logger.info(f"🔍 Comparing models across {n_episodes} episodes...")
        
        # Generate test data
        features, prices, timestamps = self._create_test_data(2000)
        
        # Convert prices to pandas Series (V3 environment expects this)
        price_series = pd.Series(prices[:, 1])  # Use NVDA close prices
        
        # Create environments
        original_env = DualTickerTradingEnvV3(
            processed_feature_data=features,
            price_data=price_series,
            log_trades=False,
            verbose=False
        )
        
        tuned_env = DualTickerTradingEnvV3Tuned(
            processed_feature_data=features,
            processed_price_data=prices,  # Tuned environment expects array format
            trading_days=timestamps,
            # Tuned weights
            hold_bonus_weight=0.0005,
            ticket_cost_per_trade=0.20,
            log_trades=False,
            verbose=False
        )
        
        # Run episodes for both models
        original_results = []
        tuned_results = []
        
        for episode in range(n_episodes):
            logger.info(f"   Episode {episode + 1}/{n_episodes}")
            
            # Original model
            original_stats = self._run_episode(self.original_model, original_env)
            original_results.append(original_stats)
            
            # Tuned model
            tuned_stats = self._run_episode(self.tuned_model, tuned_env)
            tuned_results.append(tuned_stats)
        
        # Aggregate results
        comparison = self._aggregate_results(original_results, tuned_results)
        
        logger.info("✅ Model comparison complete")
        
        return comparison
    
    def _aggregate_results(self, original_results: list, tuned_results: list):
        """
        Aggregate results from multiple episodes
        
        Args:
            original_results: Results from original model
            tuned_results: Results from tuned model
            
        Returns:
            Aggregated comparison dictionary
        """
        def aggregate_stats(results):
            return {
                'avg_trades_per_step': np.mean([r['trades_per_step'] for r in results]),
                'avg_holding_percentage': np.mean([r['holding_percentage'] for r in results]),
                'avg_total_return': np.mean([r['total_return'] for r in results]),
                'std_total_return': np.std([r['total_return'] for r in results]),
                'avg_final_value': np.mean([r['final_portfolio_value'] for r in results]),
                'total_episodes': len(results)
            }
        
        original_agg = aggregate_stats(original_results)
        tuned_agg = aggregate_stats(tuned_results)
        
        # Calculate improvements
        improvements = {
            'trading_frequency_change': (tuned_agg['avg_trades_per_step'] - original_agg['avg_trades_per_step']) / original_agg['avg_trades_per_step'] * 100,
            'holding_percentage_change': tuned_agg['avg_holding_percentage'] - original_agg['avg_holding_percentage'],
            'return_change': tuned_agg['avg_total_return'] - original_agg['avg_total_return'],
            'return_volatility_change': tuned_agg['std_total_return'] - original_agg['std_total_return']
        }
        
        return {
            'original': original_agg,
            'tuned': tuned_agg,
            'improvements': improvements,
            'raw_results': {
                'original': original_results,
                'tuned': tuned_results
            }
        }
    
    def generate_report(self, comparison_results: dict, output_path: str = None):
        """
        Generate detailed comparison report
        
        Args:
            comparison_results: Results from compare_models()
            output_path: Optional path to save report
        """
        if output_path is None:
            output_path = f"v3_tuning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = []
        report.append("🎯 V3 TUNING COMPARISON REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Original Model: {self.original_model_path}")
        report.append(f"Tuned Model: {self.tuned_model_path}")
        report.append("")
        
        # Summary statistics
        orig = comparison_results['original']
        tuned = comparison_results['tuned']
        improvements = comparison_results['improvements']
        
        report.append("📊 PERFORMANCE COMPARISON")
        report.append("-" * 30)
        report.append(f"Trading Frequency:")
        report.append(f"  Original: {orig['avg_trades_per_step']:.4f} trades/step")
        report.append(f"  Tuned:    {tuned['avg_trades_per_step']:.4f} trades/step")
        report.append(f"  Change:   {improvements['trading_frequency_change']:+.1f}%")
        report.append("")
        
        report.append(f"Holding Behavior:")
        report.append(f"  Original: {orig['avg_holding_percentage']:.1f}% holding")
        report.append(f"  Tuned:    {tuned['avg_holding_percentage']:.1f}% holding")
        report.append(f"  Change:   {improvements['holding_percentage_change']:+.1f} percentage points")
        report.append("")
        
        report.append(f"Returns:")
        report.append(f"  Original: {orig['avg_total_return']:.3f} ± {orig['std_total_return']:.3f}")
        report.append(f"  Tuned:    {tuned['avg_total_return']:.3f} ± {tuned['std_total_return']:.3f}")
        report.append(f"  Change:   {improvements['return_change']:+.3f}")
        report.append("")
        
        # Analysis
        report.append("🎯 ANALYSIS")
        report.append("-" * 30)
        
        if improvements['trading_frequency_change'] > 10:
            report.append("✅ SUCCESS: Trading frequency significantly increased")
        elif improvements['trading_frequency_change'] > 0:
            report.append("⚠️  PARTIAL: Trading frequency increased modestly")
        else:
            report.append("❌ ISSUE: Trading frequency did not increase")
        
        if improvements['holding_percentage_change'] < -5:
            report.append("✅ SUCCESS: Holding behavior reduced as intended")
        elif improvements['holding_percentage_change'] < 0:
            report.append("⚠️  PARTIAL: Holding behavior reduced modestly")
        else:
            report.append("❌ ISSUE: Holding behavior not reduced")
        
        if improvements['return_change'] > -0.01:
            report.append("✅ SUCCESS: Returns maintained or improved")
        elif improvements['return_change'] > -0.05:
            report.append("⚠️  ACCEPTABLE: Small return degradation")
        else:
            report.append("❌ CONCERN: Significant return degradation")
        
        report.append("")
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if improvements['trading_frequency_change'] > 10 and improvements['return_change'] > -0.02:
            report.append("✅ DEPLOY: Tuning successful, ready for paper trading")
        elif improvements['trading_frequency_change'] > 0:
            report.append("🔧 ITERATE: Further tuning may be beneficial")
        else:
            report.append("🔄 REVERT: Consider reverting to original weights")
        
        # Save report
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info(f"📋 Report saved to {output_path}")
        
        return output_path

def main():
    """Main execution function"""
    
    # Model paths
    original_model_path = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
    tuned_model_path = "train_runs/v3_tuned_warmstart_50k_*/best_model.zip"  # Use best model from tuning
    
    # Check if models exist
    if not os.path.exists(original_model_path):
        logger.error(f"Original model not found: {original_model_path}")
        return
    
    # For tuned model, find the latest one if using wildcard
    if "*" in tuned_model_path:
        from glob import glob
        tuned_models = glob(tuned_model_path)
        if not tuned_models:
            logger.error(f"No tuned models found matching: {tuned_model_path}")
            return
        tuned_model_path = sorted(tuned_models)[-1]  # Use latest
    
    if not os.path.exists(tuned_model_path):
        logger.error(f"Tuned model not found: {tuned_model_path}")
        logger.info("Run the tuning script first: python scripts/v3_warmstart_tuning.py")
        return
    
    try:
        # Initialize comparator
        comparator = V3TuningComparator(original_model_path, tuned_model_path)
        
        # Run comparison
        results = comparator.compare_models(n_episodes=5)
        
        # Generate report
        report_path = comparator.generate_report(results)
        
        logger.info("🎉 Comparison completed successfully!")
        logger.info(f"   Report: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()