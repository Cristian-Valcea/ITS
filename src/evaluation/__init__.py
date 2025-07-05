"""
Evaluation module for the Intraday Trading System.

This module provides specialized components for model evaluation:
- MetricsCalculator: Performance metrics calculation
- ModelLoader: Model loading and validation
- BacktestRunner: Backtest execution
- ReportGenerator: Report generation and file I/O
"""

from .metrics_calculator import MetricsCalculator, calculate_sharpe_ratio, calculate_max_drawdown
from .model_loader import ModelLoader
from .backtest_runner import BacktestRunner
from .report_generator import ReportGenerator

__all__ = [
    'MetricsCalculator',
    'ModelLoader', 
    'BacktestRunner',
    'ReportGenerator',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown'
]