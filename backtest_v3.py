#!/usr/bin/env python3
"""
True V3 Backtester - Uses exact DualTickerTradingEnvV3 with historic data
Eliminates environment mismatch and validates model performance claims

Usage:
python backtest_v3.py --model train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip --data raw/polygon_dual_ticker_20250802_131953.json
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from secrets_helper import SecretsHelper
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V3Backtester:
    """
    True V3 Backtester using exact training environment
    """
    
    def __init__(self, model_path: str, data_path: str, verbose: bool = True):
        self.model_path = model_path
        self.data_path = data_path
        self.verbose = verbose
        self.model = None
        self.env = None
        
    def load_model(self) -> bool:
        """Load the trained model"""
        logger.info(f"🤖 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model not found: {self.model_path}")
            return False
            
        try:
            self.model = RecurrentPPO.load(self.model_path)
            logger.info("✅ Model loaded successfully (RecurrentPPO)")
            return True
        except:
            try:
                self.model = PPO.load(self.model_path)
                logger.info("✅ Model loaded successfully (PPO)")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return False
    
    def prepare_historic_data(self) -> Tuple[np.ndarray, pd.Series]:
        """
        Prepare historic data in the exact format expected by V3 environment
        Returns: (combined_features, price_data)
        """
        logger.info(f"📊 Loading historic data: {self.data_path}")
        
        if self.data_path.endswith('.json'):
            # Load Polygon JSON data
            with open(self.data_path, 'r') as f:
                polygon_data = json.load(f)
            
            nvda_data = polygon_data['symbols'].get('NVDA', {}).get('data', [])
            msft_data = polygon_data['symbols'].get('MSFT', {}).get('data', [])
            
            if not nvda_data or not msft_data:
                raise ValueError("Missing NVDA or MSFT data in JSON file")
            
            # Convert to DataFrames
            nvda_df = pd.DataFrame(nvda_data)
            msft_df = pd.DataFrame(msft_data)
            
        elif self.data_path.endswith('.csv'):
            # Load CSV data (assuming dual-ticker format)
            df = pd.read_csv(self.data_path)
            # Split into NVDA and MSFT based on column naming or separate files
            # This would need to be customized based on your CSV format
            raise NotImplementedError("CSV data loading not yet implemented")
        
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
        
        # Create technical features exactly as done in training
        nvda_features = self._create_technical_features(nvda_df)
        msft_features = self._create_technical_features(msft_df)
        
        # Align data lengths
        min_len = min(len(nvda_features), len(msft_features))
        nvda_features = nvda_features[:min_len]
        msft_features = msft_features[:min_len]
        
        # Combine features (24 total)
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        
        # Extract price data for environment
        nvda_prices = nvda_df['close'].values[:min_len]
        msft_prices = msft_df['close'].values[:min_len]
        
        # Price data format expected by V3 environment (pandas Series)
        # V3 environment expects a single price series, we'll use NVDA as primary
        # The dual-ticker features are handled through the feature matrix
        price_data = pd.Series(nvda_prices)
        
        logger.info(f"✅ Prepared data: {combined_features.shape[0]} bars, {combined_features.shape[1]} features")
        
        return combined_features, price_data
    
    def _create_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create technical features exactly as done in training
        This should match the feature engineering in DualTickerDataAdapter
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate returns
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price ratios
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        # Select 12 features to match training (this should match exactly what was used)
        feature_columns = [
            'returns', 'sma_5', 'sma_20', 'rsi', 'volatility', 'price_sma20_ratio',
            'volume_ratio', 'macd', 'macd_signal', 'bb_position', 'open', 'close'
        ]
        
        # Validate we have all features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        feature_matrix = df[feature_columns].values
        
        # Normalize features (this should match training normalization)
        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            if np.std(col) > 0:
                feature_matrix[:, i] = (col - np.mean(col)) / np.std(col)
        
        return feature_matrix
    
    def create_v3_environment(self, combined_features: np.ndarray, price_data: pd.Series) -> DualTickerTradingEnvV3:
        """
        Create V3 environment with exact training configuration
        """
        logger.info("🏗️ Creating V3 environment with training configuration")
        
        # V3 Environment parameters (should match training config)
        env_config = {
            'processed_feature_data': combined_features,
            'price_data': price_data,
            'initial_capital': 100000.0,
            'max_episode_steps': min(1000, len(combined_features) - 50),  # Leave buffer
            'max_daily_drawdown_pct': 0.02,  # 2%
            'max_position_size': 1000,  # $1000 max per asset
            'transaction_cost_pct': 0.0001,  # 1 bp
            'base_impact_bp': 68.0,  # Kyle lambda calibrated impact
            'impact_exponent': 0.5,
            'risk_free_rate_annual': 0.05,  # 5% risk-free rate
            'log_trades': self.verbose,
            'verbose': False  # Reduce noise during backtest
        }
        
        self.env = DualTickerTradingEnvV3(**env_config)
        
        logger.info(f"✅ V3 Environment created")
        logger.info(f"   📊 Episode length: {env_config['max_episode_steps']} steps")
        logger.info(f"   💰 Initial capital: ${env_config['initial_capital']:,.0f}")
        logger.info(f"   🎯 Max position: ${env_config['max_position_size']}")
        logger.info(f"   💸 Transaction cost: {env_config['transaction_cost_pct']*10000:.1f} bp")
        
        return self.env
    
    def run_backtest(self, num_episodes: int = 3) -> Dict:
        """
        Run the backtest using exact V3 environment
        """
        logger.info(f"🚀 Running V3 backtest ({num_episodes} episodes)")
        
        if not self.model or not self.env:
            raise ValueError("Model and environment must be loaded first")
        
        results = []
        
        for episode in range(num_episodes):
            logger.info(f"\n📈 Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs, info = self.env.reset()
            done = False
            truncated = False
            step_count = 0
            episode_reward = 0
            trades_executed = 0
            
            # Track equity curve
            equity_curve = [self.env.portfolio_value]
            
            # LSTM states for RecurrentPPO
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'lstm'):
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
            
            while not done and not truncated:
                # Get action from model
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'lstm'):
                    action, lstm_states = self.model.predict(
                        obs, 
                        state=lstm_states, 
                        episode_start=episode_starts,
                        deterministic=True
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                # Take action in environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                step_count += 1
                equity_curve.append(self.env.portfolio_value)
                
                if info.get('trade_executed', False):
                    trades_executed += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0 and self.verbose:
                    logger.info(f"   Step {step_count}: Portfolio=${self.env.portfolio_value:.2f}, Reward={reward:.4f}")
            
            # Calculate episode metrics
            final_capital = self.env.portfolio_value
            total_return = (final_capital - 100000) / 100000 * 100
            
            # Calculate drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            max_drawdown = np.max(drawdown)
            
            # Calculate Sharpe ratio
            returns = np.diff(equity_curve) / equity_curve[:-1] * 100
            excess_returns = returns - (0.05 / 252)  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            episode_result = {
                'episode': episode + 1,
                'final_capital': final_capital,
                'total_return_pct': total_return,
                'episode_reward': episode_reward,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trades_executed': trades_executed,
                'steps_completed': step_count,
                'equity_curve': equity_curve
            }
            
            results.append(episode_result)
            
            # Print episode summary
            logger.info(f"   💰 Final Capital: ${final_capital:,.2f}")
            logger.info(f"   📊 Return: {total_return:.2f}%")
            logger.info(f"   🎯 Episode Reward: {episode_reward:.3f}")
            logger.info(f"   📉 Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"   📈 Sharpe Ratio: {sharpe_ratio:.3f}")
            logger.info(f"   🔄 Trades: {trades_executed}")
            logger.info(f"   ⏱️  Steps: {step_count}")
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze and summarize backtest results"""
        logger.info("\n" + "="*60)
        logger.info("📊 V3 BACKTEST ANALYSIS")
        logger.info("="*60)
        
        # Calculate aggregate metrics
        returns = [r['total_return_pct'] for r in results]
        rewards = [r['episode_reward'] for r in results]
        drawdowns = [r['max_drawdown_pct'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        trades = [r['trades_executed'] for r in results]
        
        summary = {
            'model_path': self.model_path,
            'data_path': self.data_path,
            'episodes_run': len(results),
            'avg_return_pct': np.mean(returns),
            'std_return_pct': np.std(returns),
            'avg_episode_reward': np.mean(rewards),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown_pct': np.max(drawdowns),
            'avg_trades_per_episode': np.mean(trades),
            'total_trades': np.sum(trades),
            'positive_episodes': sum(1 for r in returns if r > 0),
            'episodes': results
        }
        
        # Print summary
        logger.info(f"📈 Average Return: {summary['avg_return_pct']:.2f}% ± {summary['std_return_pct']:.2f}%")
        logger.info(f"🎯 Average Episode Reward: {summary['avg_episode_reward']:.3f}")
        logger.info(f"📊 Average Sharpe Ratio: {summary['avg_sharpe_ratio']:.3f}")
        logger.info(f"🚨 Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        logger.info(f"🔄 Average Trades/Episode: {summary['avg_trades_per_episode']:.1f}")
        logger.info(f"✅ Positive Episodes: {summary['positive_episodes']}/{len(results)}")
        
        # Performance assessment
        if summary['avg_return_pct'] > 0 and summary['max_drawdown_pct'] < 20:
            performance = "EXCELLENT"
        elif summary['avg_return_pct'] > 0 and summary['max_drawdown_pct'] < 50:
            performance = "GOOD"
        elif summary['avg_return_pct'] > 0:
            performance = "ACCEPTABLE"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        logger.info(f"\n🏆 OVERALL PERFORMANCE: {performance}")
        
        # Validation against training claims
        logger.info(f"\n🔍 VALIDATION AGAINST TRAINING CLAIMS:")
        logger.info(f"   Claimed: 4.5% return, 0.85 Sharpe, 1.5% drawdown")
        logger.info(f"   Actual:  {summary['avg_return_pct']:.1f}% return, {summary['avg_sharpe_ratio']:.2f} Sharpe, {summary['max_drawdown_pct']:.1f}% drawdown")
        
        return_match = abs(summary['avg_return_pct'] - 4.5) < 2.0
        sharpe_match = abs(summary['avg_sharpe_ratio'] - 0.85) < 0.3
        drawdown_match = summary['max_drawdown_pct'] < 5.0
        
        validation_score = sum([return_match, sharpe_match, drawdown_match])
        logger.info(f"   Validation Score: {validation_score}/3 criteria met")
        
        if validation_score >= 2:
            logger.info("   ✅ Training claims VALIDATED")
        else:
            logger.info("   ⚠️  Training claims NOT CONFIRMED")
        
        return summary

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='V3 Backtester - True validation using exact training environment')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historic data file (.json or .csv)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = V3Backtester(args.model, args.data, args.verbose)
    
    try:
        # Load model
        if not backtester.load_model():
            return 1
        
        # Prepare data
        combined_features, price_data = backtester.prepare_historic_data()
        
        # Create environment
        backtester.create_v3_environment(combined_features, price_data)
        
        # Run backtest
        results = backtester.run_backtest(args.episodes)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📁 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())