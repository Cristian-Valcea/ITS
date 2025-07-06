# src/risk/calculators/var_calculator.py
"""
Value at Risk (VaR) Calculator
Implements parametric, historical, and Monte Carlo VaR methods with microsecond-level latency.
"""

import numpy as np
from scipy.stats import norm, t
from typing import Dict, Any, List, Optional
import logging
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class VaRCalculator(BaseRiskCalculator):
    """
    Value at Risk calculator supporting multiple methodologies:
    - Parametric VaR (normal and t-distribution)
    - Historical VaR
    - Modified VaR (Cornish-Fisher expansion)
    
    Designed for microsecond-level latency with vectorized operations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize VaR calculator.
        
        Config options:
        - confidence_levels: List of confidence levels (default: [0.95, 0.99])
        - window_days: Lookback window in days (default: 250)
        - method: 'parametric', 'historical', or 'modified' (default: 'parametric')
        - distribution: 'normal' or 't' for parametric method (default: 'normal')
        - min_observations: Minimum observations required (default: 30)
        """
        super().__init__(config, logger)
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.VAR
    
    def _validate_config(self) -> None:
        """Validate VaR calculator configuration."""
        # Set defaults
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        self.window_days = self.config.get('window_days', 250)
        self.method = self.config.get('method', 'parametric')
        self.distribution = self.config.get('distribution', 'normal')
        self.min_observations = self.config.get('min_observations', 30)
        
        # Validate confidence levels
        if not isinstance(self.confidence_levels, list):
            self.confidence_levels = [self.confidence_levels]
        
        for level in self.confidence_levels:
            if not (0 < level < 1):
                raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        
        # Validate method
        if self.method not in ['parametric', 'historical', 'modified']:
            raise ValueError(f"Invalid method: {self.method}")
        
        # Validate distribution
        if self.distribution not in ['normal', 't']:
            raise ValueError(f"Invalid distribution: {self.distribution}")
        
        # Validate window
        if self.window_days <= 0:
            raise ValueError(f"Window days must be positive, got {self.window_days}")
        
        if self.min_observations <= 0:
            raise ValueError(f"Min observations must be positive, got {self.min_observations}")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return ['returns']  # Daily returns or P&L series
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate VaR using specified method.
        
        Args:
            data: Dictionary containing:
                - returns: numpy array of daily returns/P&L
                - portfolio_value: current portfolio value (optional, for scaling)
        
        Returns:
            RiskCalculationResult with VaR metrics
        """
        returns = np.array(data['returns'])
        portfolio_value = data.get('portfolio_value', 1.0)
        
        # Apply window
        if len(returns) > self.window_days:
            returns = returns[-self.window_days:]
        
        # Check minimum observations
        if len(returns) < self.min_observations:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message=f"Insufficient data: {len(returns)} < {self.min_observations}"
            )
        
        # Calculate VaR based on method
        if self.method == 'parametric':
            var_results = self._calculate_parametric_var(returns)
        elif self.method == 'historical':
            var_results = self._calculate_historical_var(returns)
        elif self.method == 'modified':
            var_results = self._calculate_modified_var(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Scale by portfolio value
        scaled_results = {}
        for key, value in var_results.items():
            if isinstance(value, (int, float, np.number)):
                scaled_results[key] = float(value * portfolio_value)
            else:
                scaled_results[key] = value
        
        # Add summary statistics
        scaled_results.update({
            'mean_return': float(np.mean(returns)),
            'volatility': float(np.std(returns, ddof=1)),
            'skewness': float(self._calculate_skewness(returns)),
            'kurtosis': float(self._calculate_kurtosis(returns)),
            'observations': len(returns),
            'portfolio_value': float(portfolio_value)
        })
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values=scaled_results,
            metadata={
                'method': self.method,
                'distribution': self.distribution,
                'window_days': self.window_days,
                'confidence_levels': self.confidence_levels,
                'vectorized': True
            }
        )
    
    def _calculate_parametric_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate parametric VaR using normal or t-distribution."""
        results = {}
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            
            if self.distribution == 'normal':
                # Normal distribution VaR
                z_score = norm.ppf(alpha)
                var_value = -(mean_return + z_score * std_return)
            else:
                # t-distribution VaR
                df = len(returns) - 1
                t_score = t.ppf(alpha, df)
                var_value = -(mean_return + t_score * std_return)
            
            # Ensure VaR is positive (loss)
            var_value = max(var_value, 0)
            
            confidence_pct = int(confidence_level * 100)
            results[f'var_{confidence_pct}'] = var_value
        
        return results
    
    def _calculate_historical_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate historical VaR using empirical distribution."""
        results = {}
        
        # Sort returns (losses are negative)
        sorted_returns = np.sort(returns)
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            
            # Find percentile (convert to loss)
            percentile_index = int(np.floor(alpha * len(sorted_returns)))
            percentile_index = max(0, min(percentile_index, len(sorted_returns) - 1))
            
            var_value = -sorted_returns[percentile_index]
            var_value = max(var_value, 0)  # Ensure positive
            
            confidence_pct = int(confidence_level * 100)
            results[f'var_{confidence_pct}'] = var_value
        
        return results
    
    def _calculate_modified_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate modified VaR using Cornish-Fisher expansion."""
        results = {}
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            z_score = norm.ppf(alpha)
            
            # Cornish-Fisher expansion
            cf_adjustment = (
                z_score + 
                (z_score**2 - 1) * skewness / 6 +
                (z_score**3 - 3*z_score) * (kurtosis - 3) / 24 -
                (2*z_score**3 - 5*z_score) * skewness**2 / 36
            )
            
            var_value = -(mean_return + cf_adjustment * std_return)
            var_value = max(var_value, 0)  # Ensure positive
            
            confidence_pct = int(confidence_level * 100)
            results[f'var_{confidence_pct}'] = var_value
        
        return results
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(returns)
        if n < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        skew = np.mean(((returns - mean_return) / std_return) ** 3)
        # Bias correction
        skew = skew * (n * (n - 1)) ** 0.5 / (n - 2)
        
        return skew
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate sample excess kurtosis."""
        n = len(returns)
        if n < 4:
            return 3.0  # Normal distribution kurtosis
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 3.0
        
        kurt = np.mean(((returns - mean_return) / std_return) ** 4)
        # Bias correction for excess kurtosis
        kurt = (n - 1) * ((n + 1) * kurt - 3 * (n - 1)) / ((n - 2) * (n - 3)) + 3
        
        return kurt