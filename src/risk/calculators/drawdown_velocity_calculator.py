# src/risk/calculators/drawdown_velocity_calculator.py
"""
Drawdown Velocity Calculator - CRITICAL priority sensor

Measures the speed at which drawdowns are developing to detect
accelerating loss scenarios that could spiral out of control.

Priority: CRITICAL (kill-switch capable)
Latency Target: <100µs
Action: KILL_SWITCH when drawdown accelerates dangerously
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class DrawdownVelocityCalculator(VectorizedCalculator):
    """
    Drawdown Velocity Calculator - Detects accelerating losses.
    
    Measures the rate of change in drawdown to identify when losses
    are accelerating beyond normal market volatility.
    
    Formula: velocity = d(drawdown)/dt over rolling window
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.DRAWDOWN_VELOCITY
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.velocity_window = self.config.get('velocity_window', 10)
        self.min_periods = self.config.get('min_periods', 5)
        
        if self.velocity_window < 2:
            raise ValueError("velocity_window must be at least 2")
        if self.min_periods < 2:
            raise ValueError("min_periods must be at least 2")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['portfolio_values', 'timestamps']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate drawdown velocity with vectorized operations.
        
        Args:
            data: Must contain 'portfolio_values' and 'timestamps' arrays
            
        Returns:
            RiskCalculationResult with velocity metrics
        """
        portfolio_values = self._ensure_array(data['portfolio_values'])
        timestamps = self._ensure_array(data['timestamps'])
        
        if len(portfolio_values) < self.min_periods:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'velocity': 0.0, 'acceleration': 0.0},
                metadata={'insufficient_data': True}
            )
        
        # Calculate running maximum (peak values)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown as percentage from peak
        drawdown = (running_max - portfolio_values) / running_max
        
        # Calculate time differences (vectorized)
        time_diffs = np.diff(timestamps)
        
        # Calculate drawdown velocity over rolling window
        if len(drawdown) >= self.velocity_window and len(time_diffs) >= self.velocity_window - 1:
            # Use recent window for velocity calculation
            recent_drawdown = drawdown[-self.velocity_window:]
            recent_time_diffs = time_diffs[-(self.velocity_window-1):]
            
            # Ensure arrays are compatible
            if len(recent_drawdown) - 1 != len(recent_time_diffs):
                # Adjust to make compatible
                min_len = min(len(recent_drawdown) - 1, len(recent_time_diffs))
                recent_drawdown = recent_drawdown[-(min_len+1):]
                recent_time_diffs = recent_time_diffs[-min_len:]
            
            # Calculate velocity as change in drawdown per unit time
            drawdown_changes = np.diff(recent_drawdown)
            
            if len(recent_time_diffs) > 0 and len(drawdown_changes) == len(recent_time_diffs) and np.sum(recent_time_diffs) > 0:
                # Simple average velocity to avoid complexity
                velocity_values = drawdown_changes / recent_time_diffs
                velocity = np.mean(velocity_values)
                
                # Calculate acceleration (simplified)
                if len(velocity_values) >= 2:
                    acceleration = np.mean(np.diff(velocity_values))
                else:
                    acceleration = 0.0
            else:
                velocity = 0.0
                acceleration = 0.0
        else:
            # Fallback for insufficient data
            if len(drawdown) >= 2 and len(timestamps) >= 2:
                velocity = (drawdown[-1] - drawdown[-2]) / max(timestamps[-1] - timestamps[-2], 1e-6)
                acceleration = 0.0
            else:
                velocity = 0.0
                acceleration = 0.0
        
        # Calculate additional metrics
        current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'velocity': float(velocity),
                'acceleration': float(acceleration),
                'current_drawdown': float(current_drawdown),
                'max_drawdown': float(max_drawdown),
                'velocity_percentile': self._calculate_velocity_percentile(velocity, drawdown)
            },
            metadata={
                'window_size': self.velocity_window,
                'data_points': len(portfolio_values),
                'vectorized': True
            }
        )
    
    def _calculate_velocity_percentile(self, current_velocity: float, drawdown_series: np.ndarray) -> float:
        """Calculate what percentile the current velocity represents historically."""
        if len(drawdown_series) < self.velocity_window * 2:
            return 50.0  # Default to median if insufficient history
        
        # Calculate historical velocities
        historical_velocities = []
        for i in range(self.velocity_window, len(drawdown_series)):
            window_dd = drawdown_series[i-self.velocity_window:i]
            if len(window_dd) >= 2:
                hist_velocity = (window_dd[-1] - window_dd[0]) / len(window_dd)
                historical_velocities.append(hist_velocity)
        
        if not historical_velocities:
            return 50.0
        
        # Calculate percentile
        historical_velocities = np.array(historical_velocities)
        percentile = (np.sum(historical_velocities <= current_velocity) / len(historical_velocities)) * 100
        
        return float(percentile)