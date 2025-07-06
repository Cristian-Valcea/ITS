# src/risk/calculators/ulcer_index_calculator.py
"""
Ulcer Index Calculator - HIGH priority sensor

Measures the depth and duration of drawdowns to detect when losses
are becoming persistent and painful (like an ulcer).

Priority: HIGH
Latency Target: 100-150µs
Action: BLOCK when ulcer index exceeds threshold
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class UlcerIndexCalculator(VectorizedCalculator):
    """
    Ulcer Index Calculator - Measures drawdown pain.
    
    The Ulcer Index captures both the depth and duration of drawdowns,
    providing a more comprehensive measure of downside risk than simple drawdown.
    
    Formula: UI = sqrt(mean((drawdown_pct)^2)) over lookback period
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.ULCER_INDEX
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.lookback_period = self.config.get('lookback_period', 14)
        self.min_periods = self.config.get('min_periods', 5)
        
        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.min_periods < 2:
            raise ValueError("min_periods must be at least 2")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['portfolio_values']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate Ulcer Index with vectorized operations.
        
        Args:
            data: Must contain 'portfolio_values' array
            
        Returns:
            RiskCalculationResult with Ulcer Index metrics
        """
        portfolio_values = self._ensure_array(data['portfolio_values'])
        
        if len(portfolio_values) < self.min_periods:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'ulcer_index': 0.0},
                metadata={'insufficient_data': True}
            )
        
        # Use recent data for calculation (vectorized)
        recent_values = portfolio_values[-self.lookback_period:] if len(portfolio_values) > self.lookback_period else portfolio_values
        
        # Calculate running maximum (peak values) - vectorized
        running_max = np.maximum.accumulate(recent_values)
        
        # Calculate drawdown percentages - vectorized
        drawdown_pct = ((running_max - recent_values) / running_max) * 100
        
        # Calculate Ulcer Index - vectorized
        ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
        
        # Calculate additional metrics for context
        current_drawdown_pct = drawdown_pct[-1] if len(drawdown_pct) > 0 else 0.0
        max_drawdown_pct = np.max(drawdown_pct) if len(drawdown_pct) > 0 else 0.0
        avg_drawdown_pct = np.mean(drawdown_pct) if len(drawdown_pct) > 0 else 0.0
        
        # Calculate pain duration (consecutive periods in drawdown)
        pain_duration = self._calculate_pain_duration(drawdown_pct)
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'ulcer_index': float(ulcer_index),
                'current_drawdown_pct': float(current_drawdown_pct),
                'max_drawdown_pct': float(max_drawdown_pct),
                'avg_drawdown_pct': float(avg_drawdown_pct),
                'pain_duration': int(pain_duration),
                'pain_intensity': float(ulcer_index / max(max_drawdown_pct, 0.01))  # Normalized pain
            },
            metadata={
                'lookback_period': self.lookback_period,
                'data_points': len(recent_values),
                'vectorized': True
            }
        )
    
    def _calculate_pain_duration(self, drawdown_pct: np.ndarray) -> int:
        """Calculate consecutive periods in drawdown (pain duration)."""
        if len(drawdown_pct) == 0:
            return 0
        
        # Count consecutive periods with drawdown > 0.1%
        pain_threshold = 0.1
        consecutive_pain = 0
        max_consecutive_pain = 0
        
        for dd in reversed(drawdown_pct):  # Start from most recent
            if dd > pain_threshold:
                consecutive_pain += 1
            else:
                break
        
        return consecutive_pain