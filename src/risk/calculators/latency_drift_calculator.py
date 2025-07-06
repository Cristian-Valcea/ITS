# src/risk/calculators/latency_drift_calculator.py
"""
Latency Drift Calculator - MEDIUM priority sensor

Monitors P99.9 latency drift on order round-trip times to detect
system performance degradation.

Priority: MEDIUM
Latency Target: <200µs
Action: ALERT when latency degrades significantly
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class LatencyDriftCalculator(VectorizedCalculator):
    """
    Latency Drift Calculator - System performance monitoring.
    
    Monitors P99.9 latency drift on order round-trip times. Latency spikes
    can indicate system stress and lead to slippage in fast markets.
    
    Formula: Drift = (current_P99.9 - baseline_P99.9) / baseline_P99.9
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.LATENCY_DRIFT
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.baseline_window = self.config.get('baseline_window', 1000)
        self.current_window = self.config.get('current_window', 100)
        self.percentile = self.config.get('percentile', 99.9)
        self.min_baseline_samples = self.config.get('min_baseline_samples', 100)
        
        if self.baseline_window < 50:
            raise ValueError("baseline_window must be at least 50")
        if self.current_window < 10:
            raise ValueError("current_window must be at least 10")
        if not 90.0 <= self.percentile <= 99.99:
            raise ValueError("percentile must be between 90.0 and 99.99")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['order_latencies']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate latency drift with vectorized operations.
        
        Args:
            data: Must contain 'order_latencies' array (in microseconds)
            
        Returns:
            RiskCalculationResult with latency drift metrics
        """
        order_latencies = self._ensure_array(data['order_latencies'])
        
        if len(order_latencies) < self.min_baseline_samples:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'latency_drift': 0.0, 'current_p999': 0.0, 'baseline_p999': 0.0},
                metadata={'insufficient_data': True}
            )
        
        # Split data into baseline and current windows
        if len(order_latencies) > self.baseline_window + self.current_window:
            # Use older data for baseline, recent data for current
            baseline_latencies = order_latencies[-(self.baseline_window + self.current_window):-self.current_window]
            current_latencies = order_latencies[-self.current_window:]
        else:
            # Use first 80% for baseline, last 20% for current
            split_point = int(len(order_latencies) * 0.8)
            baseline_latencies = order_latencies[:split_point]
            current_latencies = order_latencies[split_point:]
        
        if len(baseline_latencies) < 10 or len(current_latencies) < 5:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'latency_drift': 0.0},
                metadata={'insufficient_split_data': True}
            )
        
        # Calculate percentiles (vectorized)
        baseline_p999 = np.percentile(baseline_latencies, self.percentile)
        current_p999 = np.percentile(current_latencies, self.percentile)
        
        # Calculate drift
        if baseline_p999 <= 0:
            latency_drift = 0.0
        else:
            latency_drift = (current_p999 - baseline_p999) / baseline_p999
        
        # Calculate additional metrics (simplified for performance)
        baseline_p95 = np.percentile(baseline_latencies, 95) if len(baseline_latencies) > 0 else 0.0
        baseline_p50 = np.percentile(baseline_latencies, 50) if len(baseline_latencies) > 0 else 0.0
        current_p95 = np.percentile(current_latencies, 95) if len(current_latencies) > 0 else 0.0
        current_p50 = np.percentile(current_latencies, 50) if len(current_latencies) > 0 else 0.0
        
        # Simple spike detection
        spike_threshold = baseline_p999 * 2.0
        spike_count = int(np.sum(current_latencies > spike_threshold)) if len(current_latencies) > 0 else 0
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'latency_drift': float(latency_drift),
                'current_p999': float(current_p999),
                'baseline_p999': float(baseline_p999),
                'current_p95': float(current_p95),
                'current_p50': float(current_p50),
                'baseline_p95': float(baseline_p95),
                'baseline_p50': float(baseline_p50),
                'drift_p95': float((current_p95 - baseline_p95) / max(baseline_p95, 1)),
                'drift_p50': float((current_p50 - baseline_p50) / max(baseline_p50, 1)),
                'spike_count': int(spike_count),
                'spike_severity': float(spike_count / max(len(current_latencies), 1))
            },
            metadata={
                'baseline_samples': len(baseline_latencies),
                'current_samples': len(current_latencies),
                'percentile': self.percentile,
                'vectorized': True
            }
        )
    
    def _calculate_latency_stats(self, latencies: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive latency statistics."""
        if len(latencies) == 0:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'mean': 0.0, 'std': 0.0}
        
        return {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies))
        }
    
    def _detect_latency_spikes(self, current_latencies: np.ndarray, baseline_p999: float) -> Dict[str, Any]:
        """Detect latency spikes in current window."""
        if len(current_latencies) == 0:
            return {'spike_count': 0, 'max_spike': 0.0}
        
        # Define spike threshold (e.g., 2x baseline P99.9)
        spike_threshold = baseline_p999 * 2.0
        
        # Count spikes
        spikes = current_latencies[current_latencies > spike_threshold]
        spike_count = len(spikes)
        
        # Calculate maximum spike severity
        if spike_count > 0:
            max_spike = np.max(spikes) / baseline_p999 if baseline_p999 > 0 else 0.0
        else:
            max_spike = 0.0
        
        return {
            'spike_count': spike_count,
            'max_spike': float(max_spike),
            'spike_rate': float(spike_count / len(current_latencies))
        }