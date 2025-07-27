"""
Rolling Window Walk-Forward Backtest System

Implements a 3-month rolling-window walk-forward backtest to verify model robustness
across different time periods and market conditions.

Key Features:
- 3-month training windows with 1-month evaluation periods
- Walk-forward progression through historical data
- Comprehensive performance metrics per window
- Robustness statistics across all windows
- Market regime analysis and adaptation testing
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

from .backtest_runner import BacktestRunner
from .metrics_calculator import MetricsCalculator
from .model_loader import ModelLoader
from .report_generator import ReportGenerator
try:
    from ..gym_env.intraday_trading_env import IntradayTradingEnv
except ImportError:
    from gym_env.intraday_trading_env import IntradayTradingEnv


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    
    # Performance metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    turnover_ratio: float
    win_rate_pct: float
    
    # Risk metrics
    volatility_pct: float
    ulcer_index: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Market regime indicators
    market_volatility: float
    market_trend: float  # 1 = up, 0 = sideways, -1 = down
    market_return_pct: float
    
    # Model-specific metrics
    avg_position_size: float
    position_holding_time_minutes: float
    trade_frequency_per_day: float
    
    # Execution quality
    avg_trade_pnl: float
    profit_factor: float  # Gross profit / Gross loss
    max_consecutive_losses: int


class RollingWindowBacktest:
    """
    Implements rolling-window walk-forward backtesting for model robustness validation.
    
    Process:
    1. Split historical data into overlapping 3-month training windows
    2. Use each window to train/validate model performance
    3. Evaluate on subsequent 1-month period
    4. Walk forward by 1 month and repeat
    5. Aggregate results for robustness analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the rolling window backtest system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Window configuration
        self.training_window_months = config.get('rolling_backtest', {}).get('training_window_months', 3)
        self.evaluation_window_months = config.get('rolling_backtest', {}).get('evaluation_window_months', 1)
        self.step_size_months = config.get('rolling_backtest', {}).get('step_size_months', 1)
        self.min_trading_days = config.get('rolling_backtest', {}).get('min_trading_days', 20)
        
        # Components
        self.backtest_runner = BacktestRunner(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.model_loader = ModelLoader(config)
        self.report_generator = ReportGenerator(config)
        
        # Results storage
        self.window_results: List[WindowResult] = []
        self.robustness_stats: Dict[str, Any] = {}
        
        # Data management
        self.data_dir = Path(config.get('orchestrator', {}).get('data_dir', 'data/raw'))
        self.results_dir = Path(config.get('orchestrator', {}).get('reports_dir', 'reports')) / 'rolling_backtest'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Rolling Window Backtest initialized")
        self.logger.info(f"Training window: {self.training_window_months} months")
        self.logger.info(f"Evaluation window: {self.evaluation_window_months} months")
        self.logger.info(f"Step size: {self.step_size_months} months")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
    
    def run_rolling_backtest(
        self, 
        model_path: str,
        data_start_date: str,
        data_end_date: str,
        symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """
        Run the complete rolling window walk-forward backtest.
        
        Args:
            model_path: Path to the trained model
            data_start_date: Start date for historical data (YYYY-MM-DD)
            data_end_date: End date for historical data (YYYY-MM-DD)
            symbol: Trading symbol
            
        Returns:
            Dictionary containing robustness analysis results
        """
        self.logger.info("🚀 Starting Rolling Window Walk-Forward Backtest")
        self.logger.info("=" * 60)
        
        # Generate time windows
        windows = self._generate_time_windows(data_start_date, data_end_date)
        self.logger.info(f"Generated {len(windows)} walk-forward windows")
        
        # Load model
        model = self.model_loader.load_model(model_path)
        if model is None:
            raise ValueError(f"Failed to load model from {model_path}")
        
        # Process each window
        for i, window in enumerate(windows):
            self.logger.info(f"\n📊 Processing Window {i+1}/{len(windows)}")
            self.logger.info(f"   Training: {window['train_start']} → {window['train_end']}")
            self.logger.info(f"   Evaluation: {window['eval_start']} → {window['eval_end']}")
            
            try:
                result = self._evaluate_window(model, window, symbol, i+1)
                if result:
                    self.window_results.append(result)
                    self._log_window_result(result)
                else:
                    self.logger.warning(f"Window {i+1} evaluation failed - skipping")
                    
            except Exception as e:
                self.logger.error(f"Error processing window {i+1}: {e}")
                continue
        
        # Calculate robustness statistics
        self.robustness_stats = self._calculate_robustness_stats()
        
        # Generate comprehensive report
        report_path = self._generate_robustness_report()
        
        self.logger.info("\n🎉 Rolling Window Backtest Complete!")
        self.logger.info(f"📊 Processed {len(self.window_results)} windows successfully")
        self.logger.info(f"📄 Report saved to: {report_path}")
        
        return {
            'window_results': [asdict(r) for r in self.window_results],
            'robustness_stats': self.robustness_stats,
            'report_path': str(report_path),
            'num_windows': len(self.window_results)
        }
    
    def _generate_time_windows(self, start_date: str, end_date: str) -> List[Dict[str, str]]:
        """Generate overlapping time windows for walk-forward analysis."""
        windows = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        current_start = start_dt
        
        while True:
            # Training window
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.training_window_months)
            
            # Evaluation window
            eval_start = train_end
            eval_end = eval_start + pd.DateOffset(months=self.evaluation_window_months)
            
            # Check if we have enough data
            if eval_end > end_dt:
                break
            
            windows.append({
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'eval_start': eval_start.strftime('%Y-%m-%d'),
                'eval_end': eval_end.strftime('%Y-%m-%d')
            })
            
            # Move forward by step size
            current_start += pd.DateOffset(months=self.step_size_months)
        
        return windows
    
    def _evaluate_window(
        self, 
        model: Any, 
        window: Dict[str, str], 
        symbol: str, 
        window_id: int
    ) -> Optional[WindowResult]:
        """Evaluate model performance on a single time window."""
        try:
            # Load evaluation data for this window
            eval_data = self._load_window_data(
                symbol, 
                window['eval_start'], 
                window['eval_end']
            )
            
            if eval_data is None or len(eval_data) < self.min_trading_days:
                self.logger.warning(f"Insufficient data for window {window_id}")
                return None
            
            # Create evaluation environment
            eval_env = self._create_evaluation_environment(eval_data)
            
            # Run backtest
            trade_log, portfolio_history = self.backtest_runner.run_backtest(
                model, eval_env, deterministic=True
            )
            
            if trade_log is None or portfolio_history is None:
                self.logger.warning(f"Backtest failed for window {window_id}")
                return None
            
            # Calculate performance metrics
            metrics = self._calculate_window_metrics(
                trade_log, portfolio_history, eval_data, eval_env
            )
            
            # Calculate market regime indicators
            market_metrics = self._calculate_market_regime(eval_data)
            
            # Create window result
            result = WindowResult(
                window_id=window_id,
                train_start=window['train_start'],
                train_end=window['train_end'],
                eval_start=window['eval_start'],
                eval_end=window['eval_end'],
                **metrics,
                **market_metrics
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating window {window_id}: {e}")
            return None
    
    def _load_window_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load market data for a specific time window."""
        try:
            # This would typically load from your data source
            # For now, we'll create a placeholder that interfaces with your existing data loading
            
            # You would replace this with your actual data loading logic
            data_file = self.data_dir / f"{symbol}_1min.parquet"
            
            if not data_file.exists():
                self.logger.error(f"Data file not found: {data_file}")
                return None
            
            # Load and filter data
            data = pd.read_parquet(data_file)
            
            # Ensure datetime index
            if 'datetime' in data.columns:
                data.set_index('datetime', inplace=True)
            
            # Filter by date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            filtered_data = data[mask].copy()
            
            # Filter trading hours (9:30 AM - 4:00 PM ET)
            filtered_data = filtered_data.between_time('09:30', '16:00')
            
            self.logger.debug(f"Loaded {len(filtered_data)} records for {start_date} to {end_date}")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} ({start_date} to {end_date}): {e}")
            return None
    
    def _create_evaluation_environment(self, data: pd.DataFrame) -> IntradayTradingEnv:
        """Create trading environment for evaluation."""
        # Create environment configuration for this window
        env_config = self.config.copy()
        
        # Update with evaluation-specific settings
        env_config['environment'] = env_config.get('environment', {})
        env_config['environment']['initial_capital'] = 50000.0  # Standard evaluation capital
        env_config['environment']['log_trades_in_env'] = True
        
        # Create environment with window data
        env = IntradayTradingEnv(
            price_data=data['Close'],
            volume_data=data.get('Volume'),
            market_feature_data=data,  # Full OHLCV data
            dates=data.index,
            **env_config['environment']
        )
        
        return env
    
    def _calculate_window_metrics(
        self, 
        trade_log: pd.DataFrame, 
        portfolio_history: pd.Series,
        market_data: pd.DataFrame,
        env: IntradayTradingEnv
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for a single window."""
        
        # Basic performance metrics
        initial_capital = env.initial_capital
        final_value = portfolio_history.iloc[-1]
        total_return_pct = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate returns series
        returns = portfolio_history.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 390)  # 390 minutes per trading day
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = portfolio_history.expanding().max()
        drawdown = (portfolio_history - peak) / peak
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        # Trading activity metrics
        num_trades = len(trade_log) if not trade_log.empty else 0
        
        # Turnover ratio
        if not trade_log.empty:
            total_volume = trade_log['trade_value'].abs().sum()
            avg_portfolio_value = portfolio_history.mean()
            turnover_ratio = total_volume / avg_portfolio_value if avg_portfolio_value > 0 else 0
        else:
            turnover_ratio = 0
        
        # Win rate
        if not trade_log.empty and 'pnl' in trade_log.columns:
            winning_trades = (trade_log['pnl'] > 0).sum()
            win_rate_pct = (winning_trades / len(trade_log)) * 100
        else:
            win_rate_pct = 0
        
        # Volatility
        volatility_pct = returns.std() * np.sqrt(252 * 390) * 100 if len(returns) > 1 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
        
        # Calmar ratio
        calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_deviation = negative_returns.std() * np.sqrt(252 * 390)
            sortino_ratio = (returns.mean() * 252 * 390) / downside_deviation
        else:
            sortino_ratio = 0
        
        # Position and trade analysis
        if not trade_log.empty:
            avg_position_size = trade_log['quantity'].abs().mean()
            
            # Calculate holding times (simplified)
            position_holding_time_minutes = 60.0  # Placeholder - would need position tracking
            
            # Trade frequency
            trading_days = len(portfolio_history) / 390  # Approximate trading days
            trade_frequency_per_day = num_trades / trading_days if trading_days > 0 else 0
            
            # Average trade PnL
            avg_trade_pnl = trade_log['pnl'].mean() if 'pnl' in trade_log.columns else 0
            
            # Profit factor
            if 'pnl' in trade_log.columns:
                gross_profit = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
                gross_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Max consecutive losses
                losses = (trade_log['pnl'] < 0).astype(int)
                max_consecutive_losses = self._max_consecutive_ones(losses.values)
            else:
                profit_factor = 0
                max_consecutive_losses = 0
        else:
            avg_position_size = 0
            position_holding_time_minutes = 0
            trade_frequency_per_day = 0
            avg_trade_pnl = 0
            profit_factor = 0
            max_consecutive_losses = 0
        
        return {
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': num_trades,
            'turnover_ratio': turnover_ratio,
            'win_rate_pct': win_rate_pct,
            'volatility_pct': volatility_pct,
            'ulcer_index': ulcer_index,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_position_size': avg_position_size,
            'position_holding_time_minutes': position_holding_time_minutes,
            'trade_frequency_per_day': trade_frequency_per_day,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _calculate_market_regime(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market regime indicators for the evaluation period."""
        
        # Market volatility (realized volatility)
        returns = market_data['Close'].pct_change().dropna()
        market_volatility = returns.std() * np.sqrt(252 * 390) * 100  # Annualized volatility %
        
        # Market trend (simple trend classification)
        start_price = market_data['Close'].iloc[0]
        end_price = market_data['Close'].iloc[-1]
        market_return_pct = (end_price - start_price) / start_price * 100
        
        # Trend classification
        if market_return_pct > 2:
            market_trend = 1  # Up trend
        elif market_return_pct < -2:
            market_trend = -1  # Down trend
        else:
            market_trend = 0  # Sideways
        
        return {
            'market_volatility': market_volatility,
            'market_trend': market_trend,
            'market_return_pct': market_return_pct
        }
    
    def _max_consecutive_ones(self, arr: np.ndarray) -> int:
        """Calculate maximum consecutive 1s in binary array."""
        if len(arr) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in arr:
            if val == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _calculate_robustness_stats(self) -> Dict[str, Any]:
        """Calculate robustness statistics across all windows."""
        if not self.window_results:
            return {}
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame([asdict(r) for r in self.window_results])
        
        # Performance consistency metrics
        performance_stats = {}
        key_metrics = [
            'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
            'win_rate_pct', 'volatility_pct', 'turnover_ratio'
        ]
        
        for metric in key_metrics:
            if metric in results_df.columns:
                values = results_df[metric]
                performance_stats[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'consistency_score': float(1 / (1 + values.std())) if values.std() > 0 else 1.0
                }
        
        # Market regime analysis
        regime_analysis = self._analyze_market_regimes(results_df)
        
        # Robustness scores
        robustness_scores = self._calculate_robustness_scores(results_df)
        
        # Time-based analysis
        time_analysis = self._analyze_time_patterns(results_df)
        
        return {
            'performance_stats': performance_stats,
            'regime_analysis': regime_analysis,
            'robustness_scores': robustness_scores,
            'time_analysis': time_analysis,
            'summary': {
                'total_windows': len(self.window_results),
                'profitable_windows': int((results_df['total_return_pct'] > 0).sum()),
                'profitable_percentage': float((results_df['total_return_pct'] > 0).mean() * 100),
                'avg_return': float(results_df['total_return_pct'].mean()),
                'avg_sharpe': float(results_df['sharpe_ratio'].mean()),
                'max_drawdown_worst': float(results_df['max_drawdown_pct'].max()),
                'consistency_rating': self._calculate_overall_consistency(results_df)
            }
        }
    
    def _analyze_market_regimes(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        regime_analysis = {}
        
        # Group by market trend
        for trend_val, trend_name in [(-1, 'down'), (0, 'sideways'), (1, 'up')]:
            regime_data = results_df[results_df['market_trend'] == trend_val]
            
            if len(regime_data) > 0:
                regime_analysis[f'{trend_name}_market'] = {
                    'count': len(regime_data),
                    'avg_return': float(regime_data['total_return_pct'].mean()),
                    'avg_sharpe': float(regime_data['sharpe_ratio'].mean()),
                    'avg_drawdown': float(regime_data['max_drawdown_pct'].mean()),
                    'win_rate': float((regime_data['total_return_pct'] > 0).mean() * 100)
                }
        
        # Volatility regime analysis
        vol_median = results_df['market_volatility'].median()
        
        for vol_regime, vol_name in [(lambda x: x <= vol_median, 'low_vol'), 
                                     (lambda x: x > vol_median, 'high_vol')]:
            regime_data = results_df[vol_regime(results_df['market_volatility'])]
            
            if len(regime_data) > 0:
                regime_analysis[f'{vol_name}_regime'] = {
                    'count': len(regime_data),
                    'avg_return': float(regime_data['total_return_pct'].mean()),
                    'avg_sharpe': float(regime_data['sharpe_ratio'].mean()),
                    'avg_drawdown': float(regime_data['max_drawdown_pct'].mean())
                }
        
        return regime_analysis
    
    def _calculate_robustness_scores(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various robustness scores."""
        
        # Return consistency score (lower std relative to mean is better)
        returns = results_df['total_return_pct']
        return_consistency = float(1 / (1 + (returns.std() / abs(returns.mean()))) if returns.mean() != 0 else 0)
        
        # Sharpe consistency score
        sharpe_values = results_df['sharpe_ratio']
        sharpe_consistency = float(1 / (1 + sharpe_values.std()) if sharpe_values.std() > 0 else 1.0)
        
        # Drawdown control score (lower max drawdown is better)
        max_dd = results_df['max_drawdown_pct'].max()
        drawdown_control = float(max(0, 1 - max_dd / 20))  # Normalize to 20% max drawdown
        
        # Win rate stability
        win_rates = results_df['win_rate_pct']
        win_rate_stability = float(1 / (1 + win_rates.std() / 100) if win_rates.std() > 0 else 1.0)
        
        # Overall robustness score (weighted average)
        overall_robustness = float(
            0.3 * return_consistency +
            0.3 * sharpe_consistency +
            0.2 * drawdown_control +
            0.2 * win_rate_stability
        )
        
        return {
            'return_consistency': return_consistency,
            'sharpe_consistency': sharpe_consistency,
            'drawdown_control': drawdown_control,
            'win_rate_stability': win_rate_stability,
            'overall_robustness': overall_robustness
        }
    
    def _analyze_time_patterns(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance patterns over time."""
        
        # Convert dates to datetime for analysis
        results_df['eval_start_dt'] = pd.to_datetime(results_df['eval_start'])
        
        # Monthly performance analysis
        results_df['eval_month'] = results_df['eval_start_dt'].dt.month
        monthly_performance = results_df.groupby('eval_month')['total_return_pct'].agg(['mean', 'count']).to_dict()
        
        # Quarterly analysis
        results_df['eval_quarter'] = results_df['eval_start_dt'].dt.quarter
        quarterly_performance = results_df.groupby('eval_quarter')['total_return_pct'].agg(['mean', 'count']).to_dict()
        
        # Performance trend over time
        time_index = pd.Series(range(len(results_df)), index=results_df.index)
        correlation_with_time = float(results_df['total_return_pct'].corr(time_index))
        
        return {
            'monthly_performance': monthly_performance,
            'quarterly_performance': quarterly_performance,
            'performance_trend_correlation': correlation_with_time,
            'performance_degradation': correlation_with_time < -0.3  # Flag if strong negative trend
        }
    
    def _calculate_overall_consistency(self, results_df: pd.DataFrame) -> str:
        """Calculate overall consistency rating."""
        
        # Calculate coefficient of variation for key metrics
        return_cv = results_df['total_return_pct'].std() / abs(results_df['total_return_pct'].mean()) if results_df['total_return_pct'].mean() != 0 else float('inf')
        sharpe_cv = results_df['sharpe_ratio'].std() / abs(results_df['sharpe_ratio'].mean()) if results_df['sharpe_ratio'].mean() != 0 else float('inf')
        
        # Profitable window percentage
        profit_pct = (results_df['total_return_pct'] > 0).mean()
        
        # Overall rating logic
        if return_cv < 0.5 and sharpe_cv < 0.5 and profit_pct > 0.7:
            return "EXCELLENT"
        elif return_cv < 1.0 and sharpe_cv < 1.0 and profit_pct > 0.6:
            return "GOOD"
        elif return_cv < 2.0 and sharpe_cv < 2.0 and profit_pct > 0.5:
            return "FAIR"
        else:
            return "POOR"
    
    def _log_window_result(self, result: WindowResult) -> None:
        """Log results for a single window."""
        self.logger.info(f"   📈 Return: {result.total_return_pct:+.2f}% | "
                        f"Sharpe: {result.sharpe_ratio:.2f} | "
                        f"DD: {result.max_drawdown_pct:.2f}% | "
                        f"Trades: {result.num_trades} | "
                        f"Win Rate: {result.win_rate_pct:.1f}%")
        self.logger.info(f"   🌍 Market: {result.market_return_pct:+.2f}% | "
                        f"Vol: {result.market_volatility:.1f}% | "
                        f"Regime: {['Down', 'Sideways', 'Up'][result.market_trend + 1]}")
    
    def _generate_robustness_report(self) -> Path:
        """Generate comprehensive robustness report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"rolling_backtest_report_{timestamp}.json"
        
        # Prepare comprehensive report data
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'training_window_months': self.training_window_months,
                'evaluation_window_months': self.evaluation_window_months,
                'step_size_months': self.step_size_months,
                'total_windows_processed': len(self.window_results)
            },
            'window_results': [asdict(r) for r in self.window_results],
            'robustness_analysis': self.robustness_stats,
            'executive_summary': self._generate_executive_summary()
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Also save CSV for easy analysis
        csv_path = self.results_dir / f"rolling_backtest_results_{timestamp}.csv"
        results_df = pd.DataFrame([asdict(r) for r in self.window_results])
        results_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"📊 Detailed results saved to: {csv_path}")
        
        return report_path
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of robustness analysis."""
        
        if not self.robustness_stats:
            return {}
        
        summary = self.robustness_stats.get('summary', {})
        robustness_scores = self.robustness_stats.get('robustness_scores', {})
        
        return {
            'overall_assessment': {
                'consistency_rating': summary.get('consistency_rating', 'UNKNOWN'),
                'robustness_score': robustness_scores.get('overall_robustness', 0.0),
                'profitable_windows_pct': summary.get('profitable_percentage', 0.0),
                'recommendation': self._generate_recommendation()
            },
            'key_findings': {
                'average_return': f"{summary.get('avg_return', 0.0):.2f}%",
                'average_sharpe': f"{summary.get('avg_sharpe', 0.0):.2f}",
                'worst_drawdown': f"{summary.get('max_drawdown_worst', 0.0):.2f}%",
                'total_windows_tested': summary.get('total_windows', 0)
            },
            'risk_assessment': {
                'drawdown_control': robustness_scores.get('drawdown_control', 0.0),
                'return_consistency': robustness_scores.get('return_consistency', 0.0),
                'market_adaptability': self._assess_market_adaptability()
            }
        }
    
    def _generate_recommendation(self) -> str:
        """Generate deployment recommendation based on results."""
        
        if not self.robustness_stats:
            return "INSUFFICIENT_DATA"
        
        summary = self.robustness_stats.get('summary', {})
        robustness_scores = self.robustness_stats.get('robustness_scores', {})
        
        consistency_rating = summary.get('consistency_rating', 'POOR')
        overall_robustness = robustness_scores.get('overall_robustness', 0.0)
        profitable_pct = summary.get('profitable_percentage', 0.0)
        
        if consistency_rating == 'EXCELLENT' and overall_robustness > 0.8 and profitable_pct > 80:
            return "DEPLOY_FULL_CAPITAL"
        elif consistency_rating in ['EXCELLENT', 'GOOD'] and overall_robustness > 0.6 and profitable_pct > 70:
            return "DEPLOY_REDUCED_CAPITAL"
        elif consistency_rating in ['GOOD', 'FAIR'] and overall_robustness > 0.4 and profitable_pct > 60:
            return "PAPER_TRADE_FIRST"
        else:
            return "REQUIRES_IMPROVEMENT"
    
    def _assess_market_adaptability(self) -> float:
        """Assess how well the model adapts to different market conditions."""
        
        regime_analysis = self.robustness_stats.get('regime_analysis', {})
        
        if not regime_analysis:
            return 0.0
        
        # Check performance across different regimes
        regime_scores = []
        
        for regime in ['up_market', 'down_market', 'sideways_market']:
            if regime in regime_analysis:
                regime_data = regime_analysis[regime]
                # Score based on positive returns and reasonable Sharpe
                score = min(1.0, max(0.0, (regime_data.get('avg_return', 0) + 5) / 10))  # Normalize around 0
                regime_scores.append(score)
        
        return float(np.mean(regime_scores)) if regime_scores else 0.0


# Utility functions for integration
def create_rolling_backtest_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration for rolling backtest."""
    
    rolling_config = base_config.copy()
    
    # Add rolling backtest specific configuration
    rolling_config['rolling_backtest'] = {
        'training_window_months': 3,
        'evaluation_window_months': 1,
        'step_size_months': 1,
        'min_trading_days': 20,
        'enable_regime_analysis': True,
        'save_detailed_results': True
    }
    
    return rolling_config


# Export public interface
__all__ = [
    'RollingWindowBacktest',
    'WindowResult',
    'create_rolling_backtest_config'
]