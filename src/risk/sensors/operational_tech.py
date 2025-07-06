# src/risk/sensors/operational_tech.py
"""
Operational & Tech Risk Sensors - "What if the engine cracks?"

These sensors monitor system health and operational risks that could
cause trading system failures or degraded performance.

Sensors:
1. LatencyDriftSensor - P99.9 latency drift on order round-trip
2. FeedStalenessSensor - Age of market data feeds
3. ExceptionRateSensor - Real-time exception rate monitoring
"""

import time
import numpy as np
from typing import Dict, Any, List
from collections import deque

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class LatencyDriftSensor(BaseSensor):
    """
    Latency Drift Sensor.
    
    Monitors P99.9 latency drift on order round-trip times. Latency spikes
    can indicate system stress and lead to slippage in fast markets.
    
    Formula: Latency_Drift = (current_P99.9 - baseline_P99.9) / baseline_P99.9
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.OPERATIONAL_TECH
    
    def _get_data_requirements(self) -> List[str]:
        return ['order_latencies', 'timestamp']
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        super().__init__(sensor_id, sensor_name, config)
        
        # Baseline latency tracking
        self.baseline_window = int(config.get('baseline_window', 1000))
        self.baseline_latencies = deque(maxlen=self.baseline_window)
        self.baseline_p999 = None
        
        # Current latency tracking
        self.current_window = int(config.get('current_window', 100))
        self.current_latencies = deque(maxlen=self.current_window)
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute latency drift."""
        order_latencies = data.get('order_latencies', [])
        
        if not order_latencies:
            return 0.0
        
        # Add new latencies to tracking
        for latency in order_latencies:
            self.baseline_latencies.append(latency)
            self.current_latencies.append(latency)
        
        # Calculate baseline P99.9 if we have enough data
        if len(self.baseline_latencies) >= 100:
            self.baseline_p999 = np.percentile(list(self.baseline_latencies), 99.9)
        
        # Calculate current P99.9
        if len(self.current_latencies) < 20:
            return 0.0
        
        current_p999 = np.percentile(list(self.current_latencies), 99.9)
        
        # Calculate drift if we have baseline
        if self.baseline_p999 is None or self.baseline_p999 <= 0:
            return 0.0
        
        latency_drift = (current_p999 - self.baseline_p999) / self.baseline_p999
        
        return max(0.0, latency_drift)  # Only positive drift is concerning
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on sample sizes."""
        base_confidence = super()._compute_confidence(value, data)
        
        # Higher confidence with more baseline data
        baseline_ratio = len(self.baseline_latencies) / self.baseline_window
        baseline_bonus = min(0.3, baseline_ratio * 0.3)
        
        # Higher confidence with more current data
        current_ratio = len(self.current_latencies) / self.current_window
        current_bonus = min(0.2, current_ratio * 0.2)
        
        confidence = base_confidence + baseline_bonus + current_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with latency context."""
        if self.baseline_p999 is not None:
            current_p999 = np.percentile(list(self.current_latencies), 99.9) if len(self.current_latencies) >= 20 else 0
            return (f"{self.sensor_name}: {value:.1%} drift "
                    f"(P99.9: {current_p999:.1f}µs vs baseline {self.baseline_p999:.1f}µs) "
                    f"→ {action.value}")
        else:
            return super()._format_message(value, threshold, action)


class FeedStalenessSensor(BaseSensor):
    """
    Feed Staleness Sensor.
    
    Monitors the age of market data feeds. Stale feeds can lead to
    phantom fills and incorrect risk calculations.
    
    Formula: Staleness = max(now - last_tick_time) across all feeds
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.OPERATIONAL_TECH
    
    def _get_data_requirements(self) -> List[str]:
        return ['feed_timestamps', 'current_time']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute maximum feed staleness in seconds."""
        feed_timestamps = data.get('feed_timestamps', {})
        current_time = data.get('current_time', time.time())
        
        if not feed_timestamps:
            return 0.0
        
        max_staleness = 0.0
        
        for feed_name, last_update in feed_timestamps.items():
            if last_update is None:
                continue
            
            staleness = current_time - last_update
            max_staleness = max(max_staleness, staleness)
        
        return max_staleness
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on feed coverage."""
        base_confidence = super()._compute_confidence(value, data)
        
        feed_timestamps = data.get('feed_timestamps', {})
        
        # Higher confidence with more feeds monitored
        active_feeds = len([ts for ts in feed_timestamps.values() if ts is not None])
        total_feeds = len(feed_timestamps)
        
        if total_feeds > 0:
            coverage_ratio = active_feeds / total_feeds
            coverage_bonus = coverage_ratio * 0.2
        else:
            coverage_bonus = 0.0
        
        confidence = base_confidence + coverage_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with staleness context."""
        return (f"{self.sensor_name}: {value:.1f}s staleness "
                f"(threshold: {threshold:.1f}s) → {action.value}")


class ExceptionRateSensor(BaseSensor):
    """
    Exception Rate Sensor.
    
    Monitors real-time exception rate. High exception rates can indicate
    imminent system failure or hidden logic bugs.
    
    Formula: Exception_Rate = exceptions_per_minute over rolling window
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.OPERATIONAL_TECH
    
    def _get_data_requirements(self) -> List[str]:
        return ['exception_events', 'timestamp']
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        super().__init__(sensor_id, sensor_name, config)
        
        # Exception tracking
        self.window_minutes = float(config.get('window_minutes', 5.0))
        self.exception_history = deque()
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute exception rate per minute."""
        exception_events = data.get('exception_events', [])
        current_time = data.get('timestamp', time.time())
        
        # Add new exceptions to history
        for event in exception_events:
            event_time = event.get('timestamp', current_time)
            self.exception_history.append(event_time)
        
        # Remove old exceptions outside window
        window_start = current_time - (self.window_minutes * 60)
        while self.exception_history and self.exception_history[0] < window_start:
            self.exception_history.popleft()
        
        # Calculate exception rate per minute
        exception_count = len(self.exception_history)
        exception_rate = exception_count / self.window_minutes
        
        return exception_rate
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on monitoring duration."""
        base_confidence = super()._compute_confidence(value, data)
        
        # Higher confidence with longer monitoring history
        if len(self.exception_history) >= 10:
            history_bonus = 0.2
        elif len(self.exception_history) >= 5:
            history_bonus = 0.1
        else:
            history_bonus = 0.0
        
        confidence = base_confidence + history_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with exception context."""
        return (f"{self.sensor_name}: {value:.1f} exceptions/min "
                f"(threshold: {threshold:.1f}/min) → {action.value}")
    
    def get_recent_exceptions(self, limit: int = 10) -> List[float]:
        """Get recent exception timestamps."""
        return list(self.exception_history)[-limit:]