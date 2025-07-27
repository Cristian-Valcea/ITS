"""
MetricsCalculator module for performance metrics calculation.

This module handles all performance metric calculations including:
- Sharpe ratio
- Maximum drawdown
- Total return
- Turnover ratio
- Additional metrics (Sortino, Calmar, Win rate)
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def calculate_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a returns series.
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sharpe ratio
    """
    if returns_series.std() == 0 or returns_series.empty:
        return 0.0
    excess_returns = returns_series - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate the maximum drawdown from a portfolio values series.
    
    Args:
        portfolio_values: Series of portfolio values over time
        
    Returns:
        Maximum drawdown as a decimal (negative value)
    """
    if portfolio_values.empty:
        return 0.0
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()


class MetricsCalculator:
    """
    Handles calculation of various performance metrics for trading strategies.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the MetricsCalculator.
        
        Args:
            config: Configuration dictionary containing metric settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.eval_metrics_config = config.get(
            'eval_metrics',
            ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'num_trades', 'turnover_ratio_period']
        )
        
    def calculate_metrics(
        self, 
        trade_log_df: pd.DataFrame, 
        portfolio_history: pd.Series, 
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Calculate all configured performance metrics.
        
        Args:
            trade_log_df: DataFrame containing trade log
            portfolio_history: Series of portfolio values over time
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {}
        
        if portfolio_history is None or portfolio_history.empty:
            self.logger.warning("Portfolio history is empty, cannot calculate most metrics.")
            metrics['total_return_pct'] = 0.0
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(initial_capital, 2)
            metrics['num_trades'] = len(trade_log_df) if trade_log_df is not None else 0
            return metrics

        final_capital = portfolio_history.iloc[-1]
        
        # Calculate total return
        if 'total_return_pct' in self.eval_metrics_config:
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital else 0
            metrics['total_return_pct'] = round(total_return_pct, 4)
            metrics['initial_capital'] = round(initial_capital, 2)
            metrics['final_capital'] = round(final_capital, 2)
            self.logger.info(f"Calculated Total Return: {metrics['total_return_pct']:.2f}%")

        # Calculate daily returns for ratio-based metrics
        daily_returns = self._calculate_daily_returns(portfolio_history)
        
        # Calculate Sharpe ratio
        if 'sharpe_ratio' in self.eval_metrics_config:
            if not daily_returns.empty:
                risk_free_rate_annual = self.config.get('risk_free_rate_annual', 0.0)
                periods_per_year = self.config.get('periods_per_year_for_sharpe', 252)
                metrics['sharpe_ratio'] = round(
                    calculate_sharpe_ratio(daily_returns, risk_free_rate_annual, periods_per_year), 4
                )
            else:
                metrics['sharpe_ratio'] = 0.0
            self.logger.info(f"Calculated Sharpe Ratio: {metrics['sharpe_ratio']}")

        # Calculate maximum drawdown
        if 'max_drawdown_pct' in self.eval_metrics_config:
            max_dd = calculate_max_drawdown(portfolio_history) * 100
            metrics['max_drawdown_pct'] = round(max_dd, 4)
            self.logger.info(f"Calculated Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

        # Calculate turnover ratio
        if 'turnover_ratio_period' in self.eval_metrics_config:
            turnover_metrics = self._calculate_turnover_ratio(trade_log_df, initial_capital)
            metrics.update(turnover_metrics)
            self.logger.info(f"Calculated Turnover Ratio (Period): {metrics['turnover_ratio_period']}")

        # Number of trades
        metrics['num_trades'] = len(trade_log_df) if trade_log_df is not None else 0
        self.logger.info(f"Number of trades in backtest: {metrics['num_trades']}")

        # Additional metrics (placeholders for future implementation)
        self._calculate_additional_metrics(metrics, daily_returns, portfolio_history)

        return metrics
    
    def _calculate_daily_returns(self, portfolio_history: pd.Series) -> pd.Series:
        """
        Calculate daily returns from portfolio history.
        
        Args:
            portfolio_history: Series of portfolio values
            
        Returns:
            Series of daily returns
        """
        daily_returns = pd.Series(dtype=float)
        if isinstance(portfolio_history.index, pd.DatetimeIndex):
            daily_portfolio_values = portfolio_history.resample('D').last().ffill()
            daily_returns = daily_portfolio_values.pct_change().dropna()
        else:
            self.logger.warning("Portfolio history does not have a DatetimeIndex. Cannot calculate daily returns for Sharpe/Sortino.")
        return daily_returns
    
    def _calculate_turnover_ratio(self, trade_log_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
        """
        Calculate turnover ratio metrics.
        
        Args:
            trade_log_df: DataFrame containing trade log
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary containing turnover metrics
        """
        if trade_log_df is not None and not trade_log_df.empty and 'trade_value' in trade_log_df.columns:
            total_traded_value = trade_log_df['trade_value'].sum()
            avg_capital = initial_capital
            turnover_ratio = total_traded_value / avg_capital if avg_capital else 0
            return {
                'turnover_ratio_period': round(turnover_ratio, 4),
                'total_traded_value': round(total_traded_value, 2)
            }
        else:
            return {
                'turnover_ratio_period': 0.0,
                'total_traded_value': 0.0
            }
    
    def _calculate_additional_metrics(
        self, 
        metrics: Dict[str, Any], 
        daily_returns: pd.Series, 
        portfolio_history: pd.Series
    ) -> None:
        """
        Calculate additional metrics like Sortino ratio, Calmar ratio, win rate.
        
        Args:
            metrics: Dictionary to update with additional metrics
            daily_returns: Series of daily returns
            portfolio_history: Series of portfolio values
        """
        # Placeholders for additional metrics - can be implemented later
        for metric in ['sortino_ratio', 'calmar_ratio', 'win_rate_pct']:
            if metric in self.eval_metrics_config:
                if metric == 'sortino_ratio':
                    metrics[metric] = self._calculate_sortino_ratio(daily_returns)
                elif metric == 'calmar_ratio':
                    metrics[metric] = self._calculate_calmar_ratio(daily_returns, portfolio_history)
                elif metric == 'win_rate_pct':
                    metrics[metric] = self._calculate_win_rate(daily_returns)
                else:
                    metrics[metric] = "TODO"
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate Sortino ratio (placeholder for future implementation)."""
        # TODO: Implement Sortino ratio calculation
        return 0.0
    
    def _calculate_calmar_ratio(self, daily_returns: pd.Series, portfolio_history: pd.Series) -> float:
        """Calculate Calmar ratio (placeholder for future implementation)."""
        # TODO: Implement Calmar ratio calculation
        return 0.0
    
    def _calculate_win_rate(self, daily_returns: pd.Series) -> float:
        """Calculate win rate percentage (placeholder for future implementation)."""
        # TODO: Implement win rate calculation
        if daily_returns.empty:
            return 0.0
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        return round((winning_days / total_days) * 100, 2) if total_days > 0 else 0.0