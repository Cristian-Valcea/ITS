# src/risk/sensors/path_fragility.py
"""
Path-Fragility Sensors - "How quickly can P/L spiral?"

These sensors detect accelerating losses and path-dependent risks that can cause
rapid portfolio deterioration beyond simple drawdown metrics.

Sensors:
1. UlcerIndexSensor - Detects accelerating losses, not just deep ones
2. DrawdownVelocitySensor - Measures speed of drawdown development  
3. TimeToRecoverySensor - Projects bars needed to return to high-water-mark
4. DrawdownAdjustedLeverageSensor - Leverage multiplied by recent drawdown slope
"""

import numpy as np
from typing import Dict, Any, List
from scipy import stats

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class UlcerIndexSensor(BaseSensor):
    """
    Ulcer Index Sensor - Detects accelerating losses.
    
    The Ulcer Index measures the depth and duration of drawdowns, giving more weight
    to sustained losses than simple max drawdown. It's particularly sensitive to
    "death by a thousand cuts" scenarios.
    
    Formula: UI = sqrt(mean(drawdown_pct^2))
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def _get_data_requirements(self) -> List[str]:
        return ['portfolio_values', 'timestamp']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute Ulcer Index from portfolio values."""
        portfolio_values = np.array(data['portfolio_values'])
        
        if len(portfolio_values) < 2:
            return 0.0
        
        # Calculate running maximum (high-water mark)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown percentages
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Ulcer Index: RMS of drawdowns
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        
        return abs(ulcer_index)  # Return positive value
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and sample size."""
        base_confidence = super()._compute_confidence(value, data)
        
        portfolio_values = np.array(data['portfolio_values'])
        
        # Higher confidence with more data points
        sample_bonus = min(0.2, len(portfolio_values) / 100)
        
        # Lower confidence if values are too volatile (may be noisy)
        if len(portfolio_values) > 5:
            volatility = np.std(portfolio_values) / np.mean(portfolio_values)
            volatility_penalty = min(0.3, volatility)
        else:
            volatility_penalty = 0.0
        
        confidence = base_confidence + sample_bonus - volatility_penalty
        return max(0.0, min(1.0, confidence))


class DrawdownVelocitySensor(BaseSensor):
    """
    Drawdown Velocity Sensor - Measures speed of drawdown development.
    
    This sensor detects when losses are accelerating, which can indicate
    a strategy breakdown or adverse market conditions. It measures the
    rate of change in drawdown over recent periods.
    
    Formula: velocity = d(drawdown)/dt over recent window
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def _get_data_requirements(self) -> List[str]:
        return ['portfolio_values', 'timestamps']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute drawdown velocity."""
        portfolio_values = np.array(data['portfolio_values'])
        timestamps = np.array(data['timestamps'])
        
        if len(portfolio_values) < 3:
            return 0.0
        
        # Calculate running maximum (high-water mark)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown percentages
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Use recent window for velocity calculation
        window_size = min(10, len(drawdowns) // 2)
        if window_size < 3:
            return 0.0
        
        recent_drawdowns = drawdowns[-window_size:]
        recent_timestamps = timestamps[-window_size:]
        
        # Calculate time differences (convert to hours)
        time_diffs = np.diff(recent_timestamps) / 3600.0  # Assuming timestamps in seconds
        
        if np.sum(time_diffs) == 0:
            return 0.0
        
        # Calculate drawdown velocity using linear regression
        time_cumsum = np.cumsum(np.concatenate([[0], time_diffs]))
        
        if len(time_cumsum) != len(recent_drawdowns):
            return 0.0
        
        try:
            slope, _, _, _, _ = stats.linregress(time_cumsum, recent_drawdowns)
            return abs(slope)  # Return absolute velocity
        except:
            return 0.0
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and trend strength."""
        base_confidence = super()._compute_confidence(value, data)
        
        portfolio_values = np.array(data['portfolio_values'])
        
        # Higher confidence with more recent data points
        if len(portfolio_values) >= 10:
            trend_bonus = 0.2
        elif len(portfolio_values) >= 5:
            trend_bonus = 0.1
        else:
            trend_bonus = 0.0
        
        confidence = base_confidence + trend_bonus
        return max(0.0, min(1.0, confidence))


class TimeToRecoverySensor(BaseSensor):
    """
    Time-to-Recovery (TTR) Sensor - Projects bars needed to return to high-water-mark.
    
    This sensor estimates how long it would take to recover from the current drawdown
    based on recent performance patterns. Long recovery times indicate fragile strategies.
    
    Formula: TTR = current_drawdown / recent_average_return_rate
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def _get_data_requirements(self) -> List[str]:
        return ['portfolio_values', 'timestamps']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute time to recovery in hours."""
        portfolio_values = np.array(data['portfolio_values'])
        timestamps = np.array(data['timestamps'])
        
        if len(portfolio_values) < 5:
            return 0.0
        
        # Calculate current drawdown
        current_value = portfolio_values[-1]
        high_water_mark = np.max(portfolio_values)
        current_drawdown = (high_water_mark - current_value) / high_water_mark
        
        if current_drawdown <= 0:
            return 0.0  # No drawdown, no recovery needed
        
        # Calculate recent return rates
        window_size = min(20, len(portfolio_values) - 1)
        recent_values = portfolio_values[-window_size-1:]
        recent_timestamps = timestamps[-window_size-1:]
        
        # Calculate returns
        returns = np.diff(recent_values) / recent_values[:-1]
        time_diffs = np.diff(recent_timestamps) / 3600.0  # Convert to hours
        
        if len(time_diffs) == 0 or np.sum(time_diffs) == 0:
            return float('inf')
        
        # Calculate average return rate per hour
        total_return = (recent_values[-1] - recent_values[0]) / recent_values[0]
        total_time = np.sum(time_diffs)
        avg_return_rate = total_return / total_time
        
        if avg_return_rate <= 0:
            return float('inf')  # Negative returns = infinite recovery time
        
        # Time to recovery = drawdown / return_rate
        time_to_recovery = current_drawdown / avg_return_rate
        
        # Cap at reasonable maximum (e.g., 1000 hours)
        return min(time_to_recovery, 1000.0)
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on return consistency."""
        base_confidence = super()._compute_confidence(value, data)
        
        portfolio_values = np.array(data['portfolio_values'])
        
        if len(portfolio_values) < 10:
            return base_confidence * 0.5  # Low confidence with little data
        
        # Calculate return consistency
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return_volatility = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
        
        # Lower confidence for highly volatile returns
        if return_volatility > 2.0:
            volatility_penalty = 0.4
        elif return_volatility > 1.0:
            volatility_penalty = 0.2
        else:
            volatility_penalty = 0.0
        
        confidence = base_confidence - volatility_penalty
        return max(0.0, min(1.0, confidence))


class DrawdownAdjustedLeverageSensor(BaseSensor):
    """
    Drawdown-Adjusted Leverage Sensor - Leverage multiplied by recent drawdown slope.
    
    This sensor detects when high leverage is combined with deteriorating performance,
    creating a toxic combination that can lead to rapid capital destruction.
    
    Formula: DAL = leverage * abs(drawdown_slope) * volatility_multiplier
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def _get_data_requirements(self) -> List[str]:
        return ['portfolio_values', 'positions', 'leverage', 'timestamps']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute drawdown-adjusted leverage."""
        portfolio_values = np.array(data['portfolio_values'])
        leverage = float(data.get('leverage', 1.0))
        timestamps = np.array(data['timestamps'])
        
        if len(portfolio_values) < 5:
            return 0.0
        
        # Calculate running maximum (high-water mark)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown percentages
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Calculate recent drawdown slope
        window_size = min(10, len(drawdowns) // 2)
        if window_size < 3:
            return 0.0
        
        recent_drawdowns = drawdowns[-window_size:]
        recent_timestamps = timestamps[-window_size:]
        
        # Calculate time differences (convert to hours)
        time_diffs = np.diff(recent_timestamps) / 3600.0
        
        if np.sum(time_diffs) == 0:
            return 0.0
        
        # Calculate drawdown slope using linear regression
        time_cumsum = np.cumsum(np.concatenate([[0], time_diffs]))
        
        try:
            slope, _, _, _, _ = stats.linregress(time_cumsum, recent_drawdowns)
            drawdown_slope = abs(slope)
        except:
            drawdown_slope = 0.0
        
        # Calculate volatility multiplier
        if len(portfolio_values) >= 10:
            returns = np.diff(portfolio_values[-10:]) / portfolio_values[-11:-1]
            volatility = np.std(returns)
            volatility_multiplier = 1.0 + volatility * 10  # Scale volatility impact
        else:
            volatility_multiplier = 1.0
        
        # Drawdown-adjusted leverage
        dal = leverage * drawdown_slope * volatility_multiplier
        
        return dal
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and leverage accuracy."""
        base_confidence = super()._compute_confidence(value, data)
        
        # Check if leverage data is available and reasonable
        leverage = float(data.get('leverage', 1.0))
        if leverage <= 0 or leverage > 100:  # Unreasonable leverage values
            return base_confidence * 0.3
        
        # Check position data quality
        positions = data.get('positions', {})
        if not positions:
            return base_confidence * 0.7  # Lower confidence without position data
        
        return base_confidence
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with leverage context."""
        leverage = self.config.get('current_leverage', 'unknown')
        return (f"{self.sensor_name}: {value:.4f} (threshold: {threshold:.4f}) "
                f"[Leverage: {leverage}] → {action.value}")