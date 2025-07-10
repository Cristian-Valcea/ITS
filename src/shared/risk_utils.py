"""
Shared Risk Utilities

This module contains shared risk management utilities that can be used
across both execution and training components to avoid code duplication.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR) for a series of returns.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        method: Calculation method ('historical', 'parametric')
        
    Returns:
        VaR value (positive number representing potential loss)
    """
    if len(returns) == 0:
        return 0.0
        
    if method == 'historical':
        # Historical simulation method
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        return -var if var < 0 else 0.0
        
    elif method == 'parametric':
        # Parametric method assuming normal distribution
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_score = -np.percentile(np.random.standard_normal(10000), (1 - confidence_level) * 100)
        var = mean_return + z_score * std_return
        return -var if var < 0 else 0.0
        
    else:
        raise ValueError(f"Unsupported VaR method: {method}")


def calculate_max_drawdown(
    equity_curve: np.ndarray
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from an equity curve.
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
        
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_idx]
    
    # Find the peak before the maximum drawdown
    peak_idx = np.argmax(running_max[:max_dd_idx + 1])
    
    return -max_drawdown, peak_idx, max_dd_idx


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility from returns.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
        
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_position_concentration(
    positions: Dict[str, float],
    total_portfolio_value: float
) -> Dict[str, float]:
    """
    Calculate position concentration for each symbol.
    
    Args:
        positions: Dictionary of symbol -> position value
        total_portfolio_value: Total portfolio value
        
    Returns:
        Dictionary of symbol -> concentration ratio
    """
    if total_portfolio_value == 0:
        return {symbol: 0.0 for symbol in positions.keys()}
        
    return {
        symbol: abs(value) / abs(total_portfolio_value)
        for symbol, value in positions.items()
    }


def check_risk_limits(
    current_metrics: Dict[str, Any],
    risk_limits: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, List[str]]:
    """
    Check if current risk metrics exceed defined limits.
    
    Args:
        current_metrics: Current risk metrics
        risk_limits: Risk limit configuration
        logger: Optional logger instance
        
    Returns:
        Tuple of (all_limits_ok, list_of_violations)
    """
    logger = logger or logging.getLogger(__name__)
    violations = []
    
    # Check VaR limits
    if 'var_95' in current_metrics and 'max_var_95' in risk_limits:
        if current_metrics['var_95'] > risk_limits['max_var_95']:
            violations.append(f"95% VaR exceeded: {current_metrics['var_95']:.4f} > {risk_limits['max_var_95']:.4f}")
            
    # Check drawdown limits
    if 'max_drawdown' in current_metrics and 'max_drawdown_limit' in risk_limits:
        if current_metrics['max_drawdown'] > risk_limits['max_drawdown_limit']:
            violations.append(f"Max drawdown exceeded: {current_metrics['max_drawdown']:.4f} > {risk_limits['max_drawdown_limit']:.4f}")
            
    # Check position concentration limits
    if 'position_concentration' in current_metrics and 'max_position_concentration' in risk_limits:
        max_concentration = max(current_metrics['position_concentration'].values()) if current_metrics['position_concentration'] else 0
        if max_concentration > risk_limits['max_position_concentration']:
            violations.append(f"Position concentration exceeded: {max_concentration:.4f} > {risk_limits['max_position_concentration']:.4f}")
            
    # Check leverage limits
    if 'leverage' in current_metrics and 'max_leverage' in risk_limits:
        if current_metrics['leverage'] > risk_limits['max_leverage']:
            violations.append(f"Leverage exceeded: {current_metrics['leverage']:.4f} > {risk_limits['max_leverage']:.4f}")
            
    if violations:
        for violation in violations:
            logger.warning(f"Risk limit violation: {violation}")
            
    return len(violations) == 0, violations


def calculate_portfolio_beta(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """
    Calculate portfolio beta relative to a benchmark.
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Portfolio beta
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0
        
    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return 0.0
        
    return covariance / benchmark_variance


def calculate_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0
        
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
        
    return np.mean(excess_returns) / tracking_error


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate annualized return
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Calculate max drawdown
    equity_curve = np.cumprod(1 + returns)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
        
    return annualized_return / max_dd


def generate_risk_report(
    returns: np.ndarray,
    positions: Dict[str, float],
    total_portfolio_value: float,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Generate comprehensive risk report.
    
    Args:
        returns: Array of portfolio returns
        positions: Dictionary of current positions
        total_portfolio_value: Total portfolio value
        benchmark_returns: Optional benchmark returns for relative metrics
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary containing comprehensive risk metrics
    """
    if len(returns) == 0:
        return {}
        
    equity_curve = np.cumprod(1 + returns)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_return': float(equity_curve[-1] - 1),
        'annualized_return': float((equity_curve[-1] ** (periods_per_year / len(returns))) - 1),
        'volatility': float(calculate_volatility(returns, periods_per_year)),
        'sharpe_ratio': float(calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)),
        'calmar_ratio': float(calculate_calmar_ratio(returns, periods_per_year)),
        'var_95': float(calculate_var(returns, 0.95)),
        'var_99': float(calculate_var(returns, 0.99)),
        'max_drawdown': float(calculate_max_drawdown(equity_curve)[0]),
        'position_concentration': calculate_position_concentration(positions, total_portfolio_value),
        'num_positions': len([p for p in positions.values() if p != 0]),
        'total_portfolio_value': float(total_portfolio_value)
    }
    
    # Add benchmark-relative metrics if available
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        report.update({
            'beta': float(calculate_portfolio_beta(returns, benchmark_returns)),
            'information_ratio': float(calculate_information_ratio(returns, benchmark_returns))
        })
        
    return report