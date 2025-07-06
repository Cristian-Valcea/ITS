# src/risk/calculators/expected_shortfall_calculator.py
"""
Expected Shortfall (CVaR) Calculator - HIGH priority sensor

Calculates Expected Shortfall (Conditional Value at Risk) to measure
the average loss in worst-case scenarios beyond VaR.

Priority: HIGH  
Latency Target: 500-800µs
Action: BLOCK when tail risk becomes excessive
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class ExpectedShortfallCalculator(VectorizedCalculator):
    """
    Expected Shortfall Calculator - Tail risk measurement.
    
    Expected Shortfall (ES) measures the average loss in the worst-case scenarios,
    providing better tail risk assessment than VaR alone.
    
    Formula: ES_α = E[L | L > VaR_α] where L is loss
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.EXPECTED_SHORTFALL
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.lookback_period = self.config.get('lookback_period', 100)
        self.min_periods = self.config.get('min_periods', 20)
        
        if not 0.5 <= self.confidence_level <= 0.999:
            raise ValueError("confidence_level must be between 0.5 and 0.999")
        if self.lookback_period < 10:
            raise ValueError("lookback_period must be at least 10")
        if self.min_periods < 5:
            raise ValueError("min_periods must be at least 5")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['returns']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate Expected Shortfall with vectorized operations.
        
        Args:
            data: Must contain 'returns' array
            
        Returns:
            RiskCalculationResult with ES metrics
        """
        returns = self._ensure_array(data['returns'])
        
        if len(returns) < self.min_periods:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'expected_shortfall': 0.0, 'var': 0.0},
                metadata={'insufficient_data': True}
            )
        
        # Use recent data for calculation
        recent_returns = returns[-self.lookback_period:] if len(returns) > self.lookback_period else returns
        
        # Convert returns to losses (negative returns)
        losses = -recent_returns
        
        # Calculate VaR at specified confidence level (vectorized)
        var_quantile = np.quantile(losses, self.confidence_level)
        
        # Calculate Expected Shortfall (average of losses beyond VaR)
        tail_losses = losses[losses >= var_quantile]
        
        if len(tail_losses) == 0:
            # If no tail losses, ES equals VaR
            expected_shortfall = var_quantile
            tail_count = 0
        else:
            expected_shortfall = np.mean(tail_losses)
            tail_count = len(tail_losses)
        
        # Calculate additional risk metrics
        var_95 = np.quantile(losses, 0.95) if len(losses) > 0 else 0.0
        var_99 = np.quantile(losses, 0.99) if len(losses) > 0 else 0.0
        
        # Calculate tail ratio (ES/VaR)
        tail_ratio = expected_shortfall / max(var_quantile, 1e-8)
        
        # Calculate worst loss in tail
        worst_tail_loss = np.max(tail_losses) if len(tail_losses) > 0 else var_quantile
        
        # Calculate tail volatility
        tail_volatility = np.std(tail_losses) if len(tail_losses) > 1 else 0.0
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'expected_shortfall': float(expected_shortfall),
                'var': float(var_quantile),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'tail_ratio': float(tail_ratio),
                'worst_tail_loss': float(worst_tail_loss),
                'tail_volatility': float(tail_volatility),
                'tail_count': int(tail_count),
                'tail_percentage': float(tail_count / len(recent_returns) * 100)
            },
            metadata={
                'confidence_level': self.confidence_level,
                'lookback_period': self.lookback_period,
                'data_points': len(recent_returns),
                'vectorized': True
            }
        )
    
    def calculate_parametric_es(self, returns: np.ndarray) -> float:
        """
        Calculate parametric Expected Shortfall assuming normal distribution.
        Faster alternative for real-time calculations.
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate mean and std
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        if sigma <= 0:
            return 0.0
        
        # Calculate parametric ES for normal distribution
        from scipy import stats
        
        # Z-score for confidence level
        z_alpha = stats.norm.ppf(self.confidence_level)
        
        # Parametric ES formula for normal distribution
        # ES = μ - σ * φ(z_α) / (1 - α)
        phi_z = stats.norm.pdf(z_alpha)
        parametric_es = -(mu - sigma * phi_z / (1 - self.confidence_level))
        
        return max(0.0, parametric_es)