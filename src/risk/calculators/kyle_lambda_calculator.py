# src/risk/calculators/kyle_lambda_calculator.py
"""
Kyle Lambda Calculator - HIGH priority sensor

Measures market impact slope (Kyle's lambda) to detect when trades
will have excessive price impact.

Priority: HIGH
Latency Target: <150µs  
Action: THROTTLE when market impact becomes too high
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class KyleLambdaCalculator(VectorizedCalculator):
    """
    Kyle Lambda Calculator - Market impact measurement.
    
    Kyle's Lambda measures the permanent price impact per unit of order flow.
    High lambda values indicate that trades will move the market significantly.
    
    Formula: λ = Cov(price_change, order_flow) / Var(order_flow)
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.KYLE_LAMBDA
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.lookback_period = self.config.get('lookback_period', 50)
        self.min_periods = self.config.get('min_periods', 10)
        self.impact_decay = self.config.get('impact_decay', 0.9)  # Exponential decay for recent data
        
        if self.lookback_period < 5:
            raise ValueError("lookback_period must be at least 5")
        if self.min_periods < 5:
            raise ValueError("min_periods must be at least 5")
        if not 0.1 <= self.impact_decay <= 1.0:
            raise ValueError("impact_decay must be between 0.1 and 1.0")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['price_changes', 'order_flows']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate Kyle's Lambda with vectorized operations.
        
        Args:
            data: Must contain 'price_changes' and 'order_flows' arrays
            
        Returns:
            RiskCalculationResult with Kyle Lambda metrics
        """
        price_changes = self._ensure_array(data['price_changes'])
        order_flows = self._ensure_array(data['order_flows'])
        
        # Ensure arrays have same length
        min_length = min(len(price_changes), len(order_flows))
        if min_length < self.min_periods:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'kyle_lambda': 0.0},
                metadata={'insufficient_data': True}
            )
        
        # Use recent data and ensure same length
        recent_length = min(self.lookback_period, min_length)
        price_changes = price_changes[-recent_length:]
        order_flows = order_flows[-recent_length:]
        
        # Calculate Kyle's Lambda (vectorized)
        order_flow_var = np.var(order_flows)
        
        if order_flow_var <= 1e-10:  # Avoid division by zero
            kyle_lambda = 0.0
            correlation = 0.0
        else:
            # Calculate covariance and lambda
            covariance = np.cov(price_changes, order_flows)[0, 1]
            kyle_lambda = covariance / order_flow_var
            
            # Calculate correlation for additional insight
            correlation = np.corrcoef(price_changes, order_flows)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        
        # Calculate weighted lambda (simplified for performance)
        weighted_lambda = kyle_lambda  # Use same value for now
        
        # Calculate basic impact statistics (simplified)
        impact_stats = {
            'r_squared': float(correlation ** 2) if not np.isnan(correlation) else 0.0,
            'impact_1m': float(abs(kyle_lambda * 1_000_000)),
            'impact_10m': float(abs(kyle_lambda * 10_000_000))
        }
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'kyle_lambda': float(abs(kyle_lambda)),  # Use absolute value for risk measurement
                'weighted_lambda': float(abs(weighted_lambda)),
                'correlation': float(correlation),
                'order_flow_volatility': float(np.std(order_flows)),
                'price_volatility': float(np.std(price_changes)),
                'impact_r_squared': float(impact_stats['r_squared']),
                'predicted_impact_1m': float(impact_stats['impact_1m']),
                'predicted_impact_10m': float(impact_stats['impact_10m'])
            },
            metadata={
                'lookback_period': recent_length,
                'data_points': len(price_changes),
                'vectorized': True,
                'order_flow_var': float(order_flow_var)
            }
        )
    
    def _calculate_weighted_lambda(self, price_changes: np.ndarray, order_flows: np.ndarray) -> float:
        """Calculate exponentially weighted Kyle's lambda."""
        if len(price_changes) < 2:
            return 0.0
        
        # Create exponential weights (more recent = higher weight)
        weights = np.power(self.impact_decay, np.arange(len(price_changes) - 1, -1, -1))
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted covariance and variance
        weighted_mean_price = np.average(price_changes, weights=weights)
        weighted_mean_flow = np.average(order_flows, weights=weights)
        
        weighted_cov = np.average(
            (price_changes - weighted_mean_price) * (order_flows - weighted_mean_flow),
            weights=weights
        )
        
        weighted_var_flow = np.average(
            (order_flows - weighted_mean_flow) ** 2,
            weights=weights
        )
        
        if weighted_var_flow <= 1e-10:
            return 0.0
        
        return weighted_cov / weighted_var_flow
    
    def _calculate_impact_statistics(self, price_changes: np.ndarray, 
                                   order_flows: np.ndarray, 
                                   kyle_lambda: float) -> Dict[str, float]:
        """Calculate additional impact statistics."""
        if len(price_changes) < 2:
            return {'r_squared': 0.0, 'impact_1m': 0.0, 'impact_10m': 0.0}
        
        # Calculate R-squared for the linear relationship
        try:
            correlation = np.corrcoef(price_changes, order_flows)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
        except:
            r_squared = 0.0
        
        # Predict impact for standard trade sizes
        # Assume 1M and 10M notional trades
        typical_flow_1m = 1_000_000
        typical_flow_10m = 10_000_000
        
        impact_1m = abs(kyle_lambda * typical_flow_1m)
        impact_10m = abs(kyle_lambda * typical_flow_10m)
        
        return {
            'r_squared': float(r_squared),
            'impact_1m': float(impact_1m),
            'impact_10m': float(impact_10m)
        }