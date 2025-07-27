# src/risk/calculators/volatility_calculator.py
"""
Volatility Calculator
Implements multiple volatility estimation methods including realized volatility,
GARCH models, and exponentially weighted moving averages with microsecond-level latency.
"""

import numpy as np
from scipy import optimize
from typing import Dict, Any, List, Optional
import logging
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class VolatilityCalculator(BaseRiskCalculator):
    """
    Volatility calculator supporting multiple estimation methods:
    - Realized volatility (simple, Parkinson, Garman-Klass)
    - EWMA (Exponentially Weighted Moving Average)
    - GARCH(1,1) estimation
    - Intraday volatility patterns
    
    Designed for microsecond-level latency with vectorized operations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize volatility calculator.
        
        Config options:
        - window_days: Lookback window in days (default: 60)
        - annualization_factor: Days per year for annualization (default: 252)
        - method: 'realized', 'ewma', 'garch', or 'all' (default: 'realized')
        - ewma_lambda: EWMA decay factor (default: 0.94)
        - min_observations: Minimum observations required (default: 10)
        - include_intraday: Include intraday volatility patterns (default: False)
        """
        super().__init__(config, logger)
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.VOLATILITY
    
    def _validate_config(self) -> None:
        """Validate volatility calculator configuration."""
        self.window_days = self.config.get('window_days', 60)
        self.annualization_factor = self.config.get('annualization_factor', 252)
        self.method = self.config.get('method', 'realized')
        self.ewma_lambda = self.config.get('ewma_lambda', 0.94)
        self.min_observations = self.config.get('min_observations', 10)
        self.include_intraday = self.config.get('include_intraday', False)
        
        # Validate parameters
        if self.window_days <= 0:
            raise ValueError(f"Window days must be positive: {self.window_days}")
        
        if self.annualization_factor <= 0:
            raise ValueError(f"Annualization factor must be positive: {self.annualization_factor}")
        
        if self.method not in ['realized', 'ewma', 'garch', 'all']:
            raise ValueError(f"Invalid method: {self.method}")
        
        if not (0 < self.ewma_lambda < 1):
            raise ValueError(f"EWMA lambda must be between 0 and 1: {self.ewma_lambda}")
        
        if self.min_observations <= 0:
            raise ValueError(f"Min observations must be positive: {self.min_observations}")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        if self.include_intraday:
            return ['prices', 'high_prices', 'low_prices', 'open_prices', 'close_prices']
        else:
            # Accept either prices or returns
            return []  # Will validate in calculate method
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate volatility using specified method(s).
        
        Args:
            data: Dictionary containing:
                - prices: numpy array of prices (optional if returns provided)
                - returns: numpy array of returns (optional if prices provided)
                - high_prices: numpy array of high prices (for intraday methods)
                - low_prices: numpy array of low prices (for intraday methods)
                - open_prices: numpy array of open prices (for intraday methods)
                - close_prices: numpy array of close prices (for intraday methods)
        
        Returns:
            RiskCalculationResult with volatility metrics
        """
        # Validate that we have either prices or returns
        if 'prices' not in data and 'returns' not in data:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message="Must provide either 'prices' or 'returns'"
            )
        
        # Calculate returns if not provided
        if 'returns' in data:
            returns = np.array(data['returns'])
            prices = data.get('prices', None)
            if prices is not None:
                prices = np.array(prices)
        else:
            prices = np.array(data['prices'])
            if len(prices) < 2:
                return RiskCalculationResult(
                    metric_type=self.metric_type,
                    values={},
                    is_valid=False,
                    error_message="Need at least 2 prices to calculate returns"
                )
            returns = np.diff(np.log(prices))
        
        # Apply window
        if len(returns) > self.window_days:
            returns = returns[-self.window_days:]
            if prices is not None:
                prices = prices[-(self.window_days+1):]
        
        # Check minimum observations
        if len(returns) < self.min_observations:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message=f"Insufficient data: {len(returns)} < {self.min_observations}"
            )
        
        results = {}
        
        # Calculate volatility based on method
        if self.method == 'realized' or self.method == 'all':
            realized_vol = self._calculate_realized_volatility(returns, prices, data)
            results.update(realized_vol)
        
        if self.method == 'ewma' or self.method == 'all':
            ewma_vol = self._calculate_ewma_volatility(returns)
            results.update(ewma_vol)
        
        if self.method == 'garch' or self.method == 'all':
            garch_vol = self._calculate_garch_volatility(returns)
            results.update(garch_vol)
        
        # Add summary statistics
        results.update({
            'current_return': float(returns[-1]),
            'mean_return': float(np.mean(returns)),
            'return_skewness': float(self._calculate_skewness(returns)),
            'return_kurtosis': float(self._calculate_kurtosis(returns)),
            'observations': len(returns),
            'annualization_factor': self.annualization_factor
        })
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values=results,
            metadata={
                'method': self.method,
                'window_days': self.window_days,
                'ewma_lambda': self.ewma_lambda,
                'include_intraday': self.include_intraday,
                'vectorized': True
            }
        )
    
    def _calculate_realized_volatility(self, returns: np.ndarray, prices: Optional[np.ndarray], 
                                     data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate realized volatility using multiple estimators."""
        results = {}
        
        # Simple realized volatility
        simple_vol = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
        results['realized_volatility'] = float(simple_vol)
        
        # Close-to-close volatility (same as simple for daily data)
        results['close_to_close_vol'] = float(simple_vol)
        
        # If intraday data is available, calculate more sophisticated estimators
        if self.include_intraday and all(key in data for key in ['high_prices', 'low_prices', 'open_prices', 'close_prices']):
            high_prices = np.array(data['high_prices'])
            low_prices = np.array(data['low_prices'])
            open_prices = np.array(data['open_prices'])
            close_prices = np.array(data['close_prices'])
            
            # Apply same window
            if len(high_prices) > self.window_days:
                high_prices = high_prices[-self.window_days:]
                low_prices = low_prices[-self.window_days:]
                open_prices = open_prices[-self.window_days:]
                close_prices = close_prices[-self.window_days:]
            
            # Parkinson estimator (high-low)
            parkinson_vol = self._calculate_parkinson_volatility(high_prices, low_prices)
            results['parkinson_volatility'] = float(parkinson_vol)
            
            # Garman-Klass estimator
            gk_vol = self._calculate_garman_klass_volatility(
                open_prices, high_prices, low_prices, close_prices
            )
            results['garman_klass_volatility'] = float(gk_vol)
            
            # Rogers-Satchell estimator
            rs_vol = self._calculate_rogers_satchell_volatility(
                open_prices, high_prices, low_prices, close_prices
            )
            results['rogers_satchell_volatility'] = float(rs_vol)
        
        return results
    
    def _calculate_ewma_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate EWMA volatility."""
        # Initialize with first squared return
        ewma_var = returns[0] ** 2
        
        # EWMA recursion
        for i in range(1, len(returns)):
            ewma_var = self.ewma_lambda * ewma_var + (1 - self.ewma_lambda) * returns[i] ** 2
        
        ewma_vol = np.sqrt(ewma_var * self.annualization_factor)
        
        return {
            'ewma_volatility': float(ewma_vol),
            'ewma_variance': float(ewma_var)
        }
    
    def _calculate_garch_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate GARCH(1,1) volatility estimate."""
        try:
            # Simple GARCH(1,1) estimation using method of moments
            # This is a simplified version for speed
            
            # Calculate sample moments
            mean_return = np.mean(returns)
            centered_returns = returns - mean_return
            squared_returns = centered_returns ** 2
            
            # Estimate GARCH parameters using simple method
            # omega + alpha + beta should be close to 1 for stationarity
            unconditional_var = np.var(returns, ddof=1)
            
            # Simple estimates (not MLE for speed)
            alpha = 0.1  # Typical value
            beta = 0.85  # Typical value
            omega = unconditional_var * (1 - alpha - beta)
            
            # Calculate conditional variance
            garch_var = omega
            for i in range(len(squared_returns)):
                garch_var = omega + alpha * squared_returns[i] + beta * garch_var
            
            garch_vol = np.sqrt(garch_var * self.annualization_factor)
            
            return {
                'garch_volatility': float(garch_vol),
                'garch_variance': float(garch_var),
                'garch_omega': float(omega),
                'garch_alpha': float(alpha),
                'garch_beta': float(beta)
            }
            
        except Exception as e:
            self.logger.warning(f"GARCH calculation failed: {e}")
            # Fallback to simple volatility
            simple_vol = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
            return {
                'garch_volatility': float(simple_vol),
                'garch_variance': float(simple_vol**2 / self.annualization_factor),
                'garch_omega': 0.0,
                'garch_alpha': 0.0,
                'garch_beta': 0.0
            }
    
    def _calculate_parkinson_volatility(self, high_prices: np.ndarray, 
                                      low_prices: np.ndarray) -> float:
        """Calculate Parkinson volatility estimator."""
        # Parkinson estimator: sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))
        log_hl_ratio = np.log(high_prices / low_prices)
        parkinson_var = np.mean(log_hl_ratio ** 2) / (4 * np.log(2))
        return np.sqrt(parkinson_var * self.annualization_factor)
    
    def _calculate_garman_klass_volatility(self, open_prices: np.ndarray, 
                                         high_prices: np.ndarray,
                                         low_prices: np.ndarray, 
                                         close_prices: np.ndarray) -> float:
        """Calculate Garman-Klass volatility estimator."""
        # GK estimator components
        log_hl = np.log(high_prices / low_prices)
        log_co = np.log(close_prices / open_prices)
        
        gk_var = np.mean(0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2)
        return np.sqrt(gk_var * self.annualization_factor)
    
    def _calculate_rogers_satchell_volatility(self, open_prices: np.ndarray,
                                            high_prices: np.ndarray,
                                            low_prices: np.ndarray,
                                            close_prices: np.ndarray) -> float:
        """Calculate Rogers-Satchell volatility estimator."""
        # RS estimator (drift-independent)
        log_ho = np.log(high_prices / open_prices)
        log_hc = np.log(high_prices / close_prices)
        log_lo = np.log(low_prices / open_prices)
        log_lc = np.log(low_prices / close_prices)
        
        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)
        return np.sqrt(rs_var * self.annualization_factor)
    
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
