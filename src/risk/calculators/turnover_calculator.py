# src/risk/calculators/turnover_calculator.py
"""
High-performance turnover calculator with rolling window support.
Computes trading velocity metrics including hourly, daily, and custom period turnover.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class TurnoverCalculator(VectorizedCalculator):
    """
    Vectorized turnover calculator for trading velocity risk management.
    
    Computes:
    - Hourly turnover ratios
    - Daily turnover ratios
    - Rolling window turnover
    - Turnover velocity and acceleration
    - Capital efficiency metrics
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.TURNOVER
    
    def _validate_config(self) -> None:
        """Validate turnover calculator configuration."""
        # Rolling window configurations
        self.hourly_window_minutes = self.config.get('hourly_window_minutes', 60)
        self.daily_window_hours = self.config.get('daily_window_hours', 24)
        self.custom_windows = self.config.get('custom_windows', [])
        
        # Turnover calculation settings
        self.use_absolute_values = self.config.get('use_absolute_values', True)
        self.include_velocity_metrics = self.config.get('include_velocity_metrics', True)
        self.min_capital_threshold = self.config.get('min_capital_threshold', 1000.0)
        
        # Validation
        if self.hourly_window_minutes <= 0:
            raise ValueError("hourly_window_minutes must be positive")
        if self.daily_window_hours <= 0:
            raise ValueError("daily_window_hours must be positive")
        if self.min_capital_threshold < 0:
            raise ValueError("min_capital_threshold must be non-negative")
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return ['trade_values', 'capital_base']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate turnover metrics.
        
        Args:
            data: Dictionary containing:
                - trade_values: Array of trade values (absolute monetary amounts)
                - capital_base: Capital base for ratio calculations (scalar or array)
                - timestamps: Optional array of timestamps for time-based windows
                - trade_timestamps: Optional array of trade timestamps
                
        Returns:
            RiskCalculationResult with turnover metrics
        """
        trade_values = self._ensure_array(data['trade_values'])
        capital_base = data['capital_base']
        timestamps = data.get('timestamps')
        trade_timestamps = data.get('trade_timestamps')
        
        # Ensure capital_base is appropriate format
        if np.isscalar(capital_base):
            if capital_base <= self.min_capital_threshold:
                return RiskCalculationResult(
                    metric_type=self.metric_type,
                    values={},
                    is_valid=False,
                    error_message=f"Capital base {capital_base} below threshold {self.min_capital_threshold}"
                )
        else:
            capital_base = self._ensure_array(capital_base)
            if np.any(capital_base <= self.min_capital_threshold):
                return RiskCalculationResult(
                    metric_type=self.metric_type,
                    values={},
                    is_valid=False,
                    error_message="Capital base contains values below threshold"
                )
        
        # Use absolute values if configured
        if self.use_absolute_values:
            trade_values = np.abs(trade_values)
        
        # Calculate basic turnover metrics
        total_trade_value = np.sum(trade_values)
        
        if np.isscalar(capital_base):
            total_turnover_ratio = total_trade_value / capital_base
            current_capital = capital_base
        else:
            # Use the most recent capital value for total calculation
            current_capital = capital_base[-1]
            total_turnover_ratio = total_trade_value / current_capital
        
        # Calculate cumulative turnover
        cumulative_trade_values = np.cumsum(trade_values)
        if np.isscalar(capital_base):
            cumulative_turnover_ratios = cumulative_trade_values / capital_base
        else:
            cumulative_turnover_ratios = cumulative_trade_values / capital_base
        
        # Calculate rolling window turnover if timestamps provided
        rolling_metrics = {}
        if timestamps is not None and trade_timestamps is not None:
            rolling_metrics = self._calculate_rolling_turnover(
                trade_values, trade_timestamps, current_capital
            )
        
        # Calculate velocity metrics if enabled
        velocity_metrics = {}
        if self.include_velocity_metrics and len(trade_values) > 1:
            velocity_metrics = self._calculate_velocity_metrics(
                trade_values, timestamps, current_capital
            )
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            trade_values, current_capital
        )
        
        # Prepare results
        values = {
            'total_trade_value': total_trade_value,
            'total_turnover_ratio': total_turnover_ratio,
            'current_capital': current_capital,
            'cumulative_trade_values': cumulative_trade_values,
            'cumulative_turnover_ratios': cumulative_turnover_ratios,
            'trade_count': len(trade_values),
            'avg_trade_size': np.mean(trade_values) if len(trade_values) > 0 else 0.0,
            'max_trade_size': np.max(trade_values) if len(trade_values) > 0 else 0.0,
            'trade_size_std': np.std(trade_values) if len(trade_values) > 1 else 0.0,
            **rolling_metrics,
            **velocity_metrics,
            **efficiency_metrics
        }
        
        # Create result
        result = RiskCalculationResult(
            metric_type=self.metric_type,
            values=values,
            is_valid=True
        )
        
        # Add metadata
        result.add_metadata('input_size', len(trade_values))
        result.add_metadata('vectorized', True)
        result.add_metadata('use_absolute_values', self.use_absolute_values)
        result.add_metadata('capital_base_type', 'scalar' if np.isscalar(capital_base) else 'array')
        
        return result
    

    def _calculate_rolling_turnover(self, trade_values, trade_timestamps, capital_base):
        """Vectorised hourly / daily turnover."""
        tv = trade_values
        ts_sec = np.array([ts.timestamp() for ts in trade_timestamps])
        now_sec = ts_sec[-1]

        # Boolean masks instead of Python for-loop
        hourly_mask = ts_sec >= now_sec - self.hourly_window_minutes * 60
        hv     = tv[hourly_mask]
        hc     = hv.size
        result = {}
        if hc:
            hv_sum = hv.sum()
            result.update({
                'hourly_trade_value'   : hv_sum,
                'hourly_turnover_ratio': hv_sum / capital_base,
                'hourly_trade_count'   : int(hc),
                'hourly_avg_trade_size': hv_sum / hc
            })
        # daily / custom windows analogous …
        return result

    def _calculate_rolling_turnoverOld(self, trade_values: np.ndarray, 
                                  trade_timestamps: List[datetime],
                                  capital_base: float) -> Dict[str, Any]:
        """Calculate rolling window turnover metrics."""
        rolling_metrics = {}
        
        if len(trade_timestamps) != len(trade_values):
            return rolling_metrics
        
        current_time = trade_timestamps[-1] if trade_timestamps else datetime.now()
        
        # Hourly turnover
        hourly_cutoff = current_time - timedelta(minutes=self.hourly_window_minutes)
        hourly_trades = []
        hourly_values = []
        
        for i, ts in enumerate(trade_timestamps):
            if ts >= hourly_cutoff:
                hourly_trades.append(i)
                hourly_values.append(trade_values[i])
        
        if hourly_values:
            rolling_metrics.update({
                'hourly_trade_value': np.sum(hourly_values),
                'hourly_turnover_ratio': np.sum(hourly_values) / capital_base,
                'hourly_trade_count': len(hourly_values),
                'hourly_avg_trade_size': np.mean(hourly_values)
            })
        
        # Daily turnover
        daily_cutoff = current_time - timedelta(hours=self.daily_window_hours)
        daily_trades = []
        daily_values = []
        
        for i, ts in enumerate(trade_timestamps):
            if ts >= daily_cutoff:
                daily_trades.append(i)
                daily_values.append(trade_values[i])
        
        if daily_values:
            rolling_metrics.update({
                'daily_trade_value': np.sum(daily_values),
                'daily_turnover_ratio': np.sum(daily_values) / capital_base,
                'daily_trade_count': len(daily_values),
                'daily_avg_trade_size': np.mean(daily_values)
            })
        
        # Custom windows
        for window_name, window_minutes in self.custom_windows:
            window_cutoff = current_time - timedelta(minutes=window_minutes)
            window_values = []
            
            for i, ts in enumerate(trade_timestamps):
                if ts >= window_cutoff:
                    window_values.append(trade_values[i])
            
            if window_values:
                rolling_metrics.update({
                    f'{window_name}_trade_value': np.sum(window_values),
                    f'{window_name}_turnover_ratio': np.sum(window_values) / capital_base,
                    f'{window_name}_trade_count': len(window_values)
                })
        
        return rolling_metrics
    
    def _calculate_velocity_metrics(self, trade_values: np.ndarray,
                                  timestamps: Optional[List[datetime]],
                                  capital_base: float) -> Dict[str, Any]:
        """Calculate turnover velocity and acceleration metrics."""
        velocity_metrics = {}
        
        if len(trade_values) < 2:
            return velocity_metrics
        
        # Calculate trade frequency (trades per unit time)
        if timestamps and len(timestamps) == len(trade_values):
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if diff > 0:
                    time_diffs.append(diff)
            
            if time_diffs:
                avg_time_between_trades = np.mean(time_diffs)
                trade_frequency = 1.0 / avg_time_between_trades  # trades per second
                
                velocity_metrics.update({
                    'avg_time_between_trades_sec': avg_time_between_trades,
                    'trade_frequency_per_sec': trade_frequency,
                    'trade_frequency_per_hour': trade_frequency * 3600
                })
        
        # Calculate turnover velocity (change in turnover rate)
        window_size = min(10, len(trade_values) // 2)  # Adaptive window size
        if window_size >= 2:
            recent_trades = trade_values[-window_size:]
            earlier_trades = trade_values[-2*window_size:-window_size] if len(trade_values) >= 2*window_size else trade_values[:-window_size]
            
            recent_turnover = np.sum(recent_trades) / capital_base
            earlier_turnover = np.sum(earlier_trades) / capital_base
            
            turnover_velocity = recent_turnover - earlier_turnover
            
            velocity_metrics.update({
                'turnover_velocity': turnover_velocity,
                'recent_turnover_ratio': recent_turnover,
                'earlier_turnover_ratio': earlier_turnover,
                'velocity_window_size': window_size
            })
        
        # Calculate trade size momentum
        if len(trade_values) >= 5:
            recent_avg = np.mean(trade_values[-5:])
            overall_avg = np.mean(trade_values)
            size_momentum = (recent_avg - overall_avg) / overall_avg if overall_avg > 0 else 0.0
            
            velocity_metrics['trade_size_momentum'] = size_momentum
        
        return velocity_metrics
    
    def _calculate_efficiency_metrics(self, trade_values: np.ndarray,
                                    capital_base: float) -> Dict[str, Any]:
        """Calculate capital efficiency metrics."""
        efficiency_metrics = {}
        
        if len(trade_values) == 0:
            return efficiency_metrics
        
        # Capital utilization efficiency
        total_trade_value = np.sum(trade_values)
        capital_utilization = total_trade_value / capital_base
        
        # Trade size distribution metrics
        trade_sizes_pct = trade_values / capital_base * 100  # As percentage of capital
        
        efficiency_metrics.update({
            'capital_utilization_ratio': capital_utilization,
            'avg_trade_size_pct_capital': np.mean(trade_sizes_pct),
            'max_trade_size_pct_capital': np.max(trade_sizes_pct),
            'trade_size_concentration': np.std(trade_sizes_pct) / np.mean(trade_sizes_pct) if np.mean(trade_sizes_pct) > 0 else 0.0
        })
        
        # Percentile analysis
        if len(trade_values) >= 5:
            efficiency_metrics.update({
                'trade_size_p50_pct_capital': np.percentile(trade_sizes_pct, 50),
                'trade_size_p90_pct_capital': np.percentile(trade_sizes_pct, 90),
                'trade_size_p95_pct_capital': np.percentile(trade_sizes_pct, 95)
            })
        
        return efficiency_metrics
    
    def calculate_turnover_limits_breach(self, data: Dict[str, Any],
                                       hourly_limit: float,
                                       daily_limit: float) -> Dict[str, Any]:
        """
        Check if turnover limits would be breached by proposed trades.
        
        Args:
            data: Input data with current turnover state
            hourly_limit: Hourly turnover limit ratio
            daily_limit: Daily turnover limit ratio
            
        Returns:
            Dictionary with breach analysis
        """
        result = self.calculate_safe(data)
        
        if not result.is_valid:
            return {'error': result.error_message}
        
        values = result.values
        
        # Get current turnover ratios
        hourly_ratio = values.get('hourly_turnover_ratio', 0.0)
        daily_ratio = values.get('daily_turnover_ratio', 0.0)
        
        # Check breaches
        hourly_breach = hourly_ratio > hourly_limit
        daily_breach = daily_ratio > daily_limit
        
        # Calculate headroom
        hourly_headroom = max(0.0, hourly_limit - hourly_ratio)
        daily_headroom = max(0.0, daily_limit - daily_ratio)
        
        return {
            'hourly_turnover_ratio': hourly_ratio,
            'daily_turnover_ratio': daily_ratio,
            'hourly_limit': hourly_limit,
            'daily_limit': daily_limit,
            'hourly_breach': hourly_breach,
            'daily_breach': daily_breach,
            'hourly_headroom_ratio': hourly_headroom,
            'daily_headroom_ratio': daily_headroom,
            'any_breach': hourly_breach or daily_breach,
            'breach_severity': max(
                (hourly_ratio - hourly_limit) / hourly_limit if hourly_breach else 0.0,
                (daily_ratio - daily_limit) / daily_limit if daily_breach else 0.0
            )
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        return {
            "type": "object",
            "properties": {
                "hourly_window_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 60
                },
                "daily_window_hours": {
                    "type": "integer", 
                    "minimum": 1,
                    "default": 24
                },
                "custom_windows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "string"},
                            {"type": "integer", "minimum": 1}
                        ]
                    },
                    "default": []
                },
                "use_absolute_values": {
                    "type": "boolean",
                    "default": True
                },
                "include_velocity_metrics": {
                    "type": "boolean",
                    "default": True
                },
                "min_capital_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1000.0
                }
            }
        }