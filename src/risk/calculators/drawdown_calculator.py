# src/risk/calculators/drawdown_calculator.py
"""
High-performance drawdown calculator with vectorized operations.
Computes various drawdown metrics including maximum drawdown, current drawdown,
and drawdown duration.
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class DrawdownCalculator(VectorizedCalculator):
    """
    Vectorized drawdown calculator for portfolio risk management.
    
    Computes:
    - Current drawdown (from peak)
    - Maximum drawdown over period
    - Drawdown duration
    - Underwater curve
    - Recovery time statistics
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.DRAWDOWN
    
    def _validate_config(self) -> None:
        """Validate drawdown calculator configuration."""
        required_keys = []  # No required config for basic drawdown
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Optional parameters with defaults
        self.lookback_periods = self.config.get('lookback_periods', [1, 5, 20, 60, 252])
        self.min_periods = self.config.get('min_periods', 1)
        self.annualization_factor = self.config.get('annualization_factor', 252)
    
    def get_required_inputs(self) -> List[str]:
        """Return required input data keys."""
        return ['portfolio_values']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate drawdown metrics.
        
        Args:
            data: Dictionary containing:
                - portfolio_values: Array of portfolio values over time
                - timestamps: Optional array of timestamps
                - start_of_day_value: Optional starting value for daily drawdown
                
        Returns:
            RiskCalculationResult with drawdown metrics
        """
        portfolio_values = self._ensure_array(data['portfolio_values'])
        timestamps = data.get('timestamps')
        start_of_day_value = data.get('start_of_day_value')
        
        if len(portfolio_values) < self.min_periods:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message=f"Insufficient data: {len(portfolio_values)} < {self.min_periods}"
            )
        
        # Calculate running maximum (peak values)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown as percentage from peak
        drawdown_pct = (portfolio_values - running_max) / running_max
        
        # Current drawdown (most recent value)
        current_drawdown = drawdown_pct[-1]
        
        # Maximum drawdown over the entire period
        max_drawdown = np.min(drawdown_pct)
        
        # Calculate daily drawdown if start_of_day_value provided
        daily_drawdown = None
        if start_of_day_value is not None:
            current_value = portfolio_values[-1]
            daily_drawdown = (current_value - start_of_day_value) / start_of_day_value
        
        # Calculate drawdown duration (periods underwater)
        underwater_mask = drawdown_pct < 0
        current_underwater_duration = self._calculate_current_underwater_duration(underwater_mask)
        max_underwater_duration = self._calculate_max_underwater_duration(underwater_mask)
        
        # Calculate drawdown statistics for different lookback periods
        lookback_stats = {}
        for periods in self.lookback_periods:
            if len(portfolio_values) >= periods:
                recent_values = portfolio_values[-periods:]
                recent_max = np.maximum.accumulate(recent_values)
                recent_drawdown = (recent_values - recent_max) / recent_max
                
                lookback_stats[f'max_drawdown_{periods}d'] = np.min(recent_drawdown)
                lookback_stats[f'current_drawdown_{periods}d'] = recent_drawdown[-1]
        
        # Calculate recovery statistics
        recovery_stats = self._calculate_recovery_stats(portfolio_values, running_max, drawdown_pct)
        
        # Prepare results
        values = {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'current_underwater_duration': current_underwater_duration,
            'max_underwater_duration': max_underwater_duration,
            'drawdown_series': drawdown_pct,
            'underwater_curve': underwater_mask.astype(float),
            'running_max': running_max,
            **lookback_stats,
            **recovery_stats
        }
        
        # Add daily drawdown if available
        if daily_drawdown is not None:
            values['daily_drawdown'] = daily_drawdown
        
        # Create result
        result = RiskCalculationResult(
            metric_type=self.metric_type,
            values=values,
            is_valid=True
        )
        
        # Add metadata
        result.add_metadata('input_size', len(portfolio_values))
        result.add_metadata('vectorized', True)
        result.add_metadata('lookback_periods', self.lookback_periods)
        
        return result
    
    def _calculate_current_underwater_duration(self, underwater_mask: np.ndarray) -> int:
        """Calculate current consecutive periods underwater."""
        if not underwater_mask[-1]:
            return 0
        
        # Count consecutive True values from the end
        duration = 0
        for i in range(len(underwater_mask) - 1, -1, -1):
            if underwater_mask[i]:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_max_underwater_duration(self, underwater_mask: np.ndarray) -> int:
        """Calculate maximum consecutive periods underwater."""
        if not np.any(underwater_mask):
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for is_underwater in underwater_mask:
            if is_underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_recovery_stats(self, portfolio_values: np.ndarray, 
                                running_max: np.ndarray, 
                                drawdown_pct: np.ndarray) -> Dict[str, float]:
        """Calculate recovery time statistics."""
        recovery_stats = {}
        
        # Find all drawdown periods and their recovery times
        underwater_mask = drawdown_pct < 0
        
        if not np.any(underwater_mask):
            recovery_stats.update({
                'avg_recovery_time': 0.0,
                'max_recovery_time': 0.0,
                'total_recovery_periods': 0
            })
            return recovery_stats
        
        # Find start and end of underwater periods
        underwater_starts = []
        underwater_ends = []
        
        in_drawdown = False
        for i, is_underwater in enumerate(underwater_mask):
            if is_underwater and not in_drawdown:
                underwater_starts.append(i)
                in_drawdown = True
            elif not is_underwater and in_drawdown:
                underwater_ends.append(i - 1)
                in_drawdown = False
        
        # Handle case where we're still in drawdown
        if in_drawdown:
            underwater_ends.append(len(underwater_mask) - 1)
        
        # Calculate recovery times for completed drawdowns
        recovery_times = []
        for start, end in zip(underwater_starts, underwater_ends):
            if end < len(portfolio_values) - 1:  # Only for completed recoveries
                # Find when portfolio value recovers to the peak before drawdown
                peak_value = running_max[start]
                
                # Look for recovery after the drawdown end
                for recovery_idx in range(end + 1, len(portfolio_values)):
                    if portfolio_values[recovery_idx] >= peak_value:
                        recovery_time = recovery_idx - start
                        recovery_times.append(recovery_time)
                        break
        
        if recovery_times:
            recovery_stats.update({
                'avg_recovery_time': np.mean(recovery_times),
                'max_recovery_time': np.max(recovery_times),
                'min_recovery_time': np.min(recovery_times),
                'total_recovery_periods': len(recovery_times)
            })
        else:
            recovery_stats.update({
                'avg_recovery_time': 0.0,
                'max_recovery_time': 0.0,
                'min_recovery_time': 0.0,
                'total_recovery_periods': 0
            })
        
        return recovery_stats
    
    def calculate_var_style_drawdown(self, data: Dict[str, Any], 
                                   confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate VaR-style drawdown metrics.
        
        Args:
            data: Input data with portfolio_values
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Dictionary with VaR-style drawdown metrics
        """
        portfolio_values = self._ensure_array(data['portfolio_values'])
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate rolling drawdowns over different windows
        windows = [5, 10, 20, 60]
        var_metrics = {}
        
        for window in windows:
            if len(returns) >= window:
                rolling_returns = np.array([
                    np.sum(returns[i:i+window]) 
                    for i in range(len(returns) - window + 1)
                ])
                
                # Calculate VaR and CVaR for this window
                var_value = np.percentile(rolling_returns, confidence_level * 100)
                cvar_value = np.mean(rolling_returns[rolling_returns <= var_value])
                
                var_metrics[f'var_{window}d'] = var_value
                var_metrics[f'cvar_{window}d'] = cvar_value
        
        return var_metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        return {
            "type": "object",
            "properties": {
                "lookback_periods": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "default": [1, 5, 20, 60, 252]
                },
                "min_periods": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "annualization_factor": {
                    "type": "number",
                    "minimum": 1,
                    "default": 252
                },
                "use_vectorization": {
                    "type": "boolean",
                    "default": True
                }
            }
        }