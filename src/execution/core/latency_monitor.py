"""
Latency Monitoring System for Critical Trading Paths

This module provides real-time latency monitoring specifically for
critical trading operations like KILL_SWITCH, trade execution, and
risk checks. It's designed to detect and alert on latency spikes.
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import logging


class LatencyCategory(Enum):
    """Categories of operations being monitored."""
    KILL_SWITCH = "kill_switch"
    TRADE_EXECUTION = "trade_execution"
    RISK_CHECK = "risk_check"
    AUDIT_LOGGING = "audit_logging"
    ORDER_ROUTING = "order_routing"
    PNL_UPDATE = "pnl_update"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    timestamp_ns: int
    category: LatencyCategory
    operation: str
    latency_us: float
    thread_id: int
    metadata: Dict = None


class LatencyStats:
    """Statistics for a specific operation category."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.measurements = deque(maxlen=max_samples)
        self.total_count = 0
        self.lock = threading.RLock()
    
    def add_measurement(self, latency_us: float, timestamp_ns: int = None):
        """Add a latency measurement."""
        with self.lock:
            if timestamp_ns is None:
                timestamp_ns = time.time_ns()
            
            self.measurements.append((timestamp_ns, latency_us))
            self.total_count += 1
    
    def get_percentiles(self, percentiles: List[float] = None) -> Dict[float, float]:
        """Get latency percentiles."""
        if percentiles is None:
            percentiles = [50.0, 90.0, 95.0, 99.0, 99.9, 99.97, 99.99]
        
        with self.lock:
            if not self.measurements:
                return {p: 0.0 for p in percentiles}
            
            latencies = [lat for _, lat in self.measurements]
            latencies.sort()
            
            result = {}
            for p in percentiles:
                if p == 100.0:
                    result[p] = latencies[-1]
                else:
                    index = int((p / 100.0) * len(latencies))
                    index = min(index, len(latencies) - 1)
                    result[p] = latencies[index]
            
            return result
    
    def get_recent_stats(self, window_seconds: float = 60.0) -> Dict:
        """Get statistics for recent measurements."""
        cutoff_ns = time.time_ns() - int(window_seconds * 1e9)
        
        with self.lock:
            recent = [(ts, lat) for ts, lat in self.measurements if ts >= cutoff_ns]
            
            if not recent:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0
                }
            
            latencies = [lat for _, lat in recent]
            
            return {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }


class LatencyMonitor:
    """
    Real-time latency monitoring system for critical trading operations.
    
    Designed to detect the 38 µs spikes in KILL_SWITCH operations and
    other critical latency issues.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize latency monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Statistics storage
        self.stats: Dict[LatencyCategory, LatencyStats] = {}
        for category in LatencyCategory:
            max_samples = self.config.get('max_samples_per_category', 10000)
            self.stats[category] = LatencyStats(max_samples)
        
        # Alert thresholds (microseconds)
        self.alert_thresholds = {
            LatencyCategory.KILL_SWITCH: self.config.get('kill_switch_threshold_us', 10.0),
            LatencyCategory.TRADE_EXECUTION: self.config.get('trade_execution_threshold_us', 50.0),
            LatencyCategory.RISK_CHECK: self.config.get('risk_check_threshold_us', 20.0),
            LatencyCategory.AUDIT_LOGGING: self.config.get('audit_logging_threshold_us', 5.0),
            LatencyCategory.ORDER_ROUTING: self.config.get('order_routing_threshold_us', 100.0),
            LatencyCategory.PNL_UPDATE: self.config.get('pnl_update_threshold_us', 30.0),
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("LatencyMonitor initialized")
    
    def measure_latency(self, category: LatencyCategory, operation: str = ""):
        """
        Context manager for measuring operation latency.
        
        Usage:
            with latency_monitor.measure_latency(LatencyCategory.KILL_SWITCH, "emergency_stop"):
                # Critical operation here
                emergency_stop_logic()
        """
        return LatencyMeasurementContext(self, category, operation)
    
    def record_latency(self, category: LatencyCategory, latency_us: float, 
                      operation: str = "", metadata: Dict = None):
        """
        Record a latency measurement directly.
        
        Args:
            category: Category of operation
            latency_us: Latency in microseconds
            operation: Operation name
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        timestamp_ns = time.time_ns()
        
        # Store measurement
        self.stats[category].add_measurement(latency_us, timestamp_ns)
        
        # Check for alerts
        threshold = self.alert_thresholds.get(category, float('inf'))
        if latency_us > threshold:
            self._trigger_alert(category, latency_us, operation, metadata)
    
    def _trigger_alert(self, category: LatencyCategory, latency_us: float, 
                      operation: str, metadata: Dict):
        """Trigger latency alert."""
        alert_data = {
            'category': category.value,
            'latency_us': latency_us,
            'threshold_us': self.alert_thresholds[category],
            'operation': operation,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Log alert
        self.logger.warning(
            f"LATENCY ALERT: {category.value} operation '{operation}' took {latency_us:.2f}µs "
            f"(threshold: {self.alert_thresholds[category]:.2f}µs)"
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback for latency alerts."""
        self.alert_callbacks.append(callback)
    
    def get_stats_summary(self) -> Dict:
        """Get comprehensive statistics summary."""
        summary = {}
        
        for category in LatencyCategory:
            stats = self.stats[category]
            percentiles = stats.get_percentiles()
            recent = stats.get_recent_stats()
            
            summary[category.value] = {
                'total_measurements': stats.total_count,
                'recent_stats': recent,
                'percentiles': percentiles,
                'threshold_us': self.alert_thresholds[category]
            }
        
        return summary
    
    def get_kill_switch_analysis(self) -> Dict:
        """Get detailed analysis of KILL_SWITCH latency."""
        kill_switch_stats = self.stats[LatencyCategory.KILL_SWITCH]
        percentiles = kill_switch_stats.get_percentiles([99.97, 99.99, 100.0])
        recent = kill_switch_stats.get_recent_stats()
        
        # Check if we're still seeing the 38µs spike
        spike_detected = percentiles[99.97] > 35.0  # Close to the original 38µs issue
        
        return {
            'total_kill_switches': kill_switch_stats.total_count,
            'p99_97_latency_us': percentiles[99.97],
            'p99_99_latency_us': percentiles[99.99],
            'max_latency_us': percentiles[100.0],
            'recent_mean_us': recent['mean'],
            'recent_max_us': recent['max'],
            'spike_detected': spike_detected,
            'spike_threshold_us': 35.0,
            'target_latency_us': 10.0,
            'performance_status': 'GOOD' if percentiles[99.97] < 10.0 else 'NEEDS_ATTENTION'
        }
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        while self.monitoring_active:
            try:
                # Periodic analysis
                self._periodic_analysis()
                time.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring worker error: {e}")
    
    def _periodic_analysis(self):
        """Perform periodic latency analysis."""
        kill_switch_analysis = self.get_kill_switch_analysis()
        
        # Log periodic status
        if kill_switch_analysis['total_kill_switches'] > 0:
            self.logger.info(
                f"KILL_SWITCH Latency Status: P99.97={kill_switch_analysis['p99_97_latency_us']:.2f}µs, "
                f"Status={kill_switch_analysis['performance_status']}"
            )
            
            # Alert if we're still seeing spikes
            if kill_switch_analysis['spike_detected']:
                self.logger.warning(
                    f"KILL_SWITCH latency spike detected: {kill_switch_analysis['p99_97_latency_us']:.2f}µs "
                    f"at 99.97 percentile (target: <10µs)"
                )
    
    def shutdown(self):
        """Shutdown monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("LatencyMonitor shutdown complete")


class LatencyMeasurementContext:
    """Context manager for measuring latency."""
    
    def __init__(self, monitor: LatencyMonitor, category: LatencyCategory, operation: str):
        self.monitor = monitor
        self.category = category
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time_ns()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.time_ns()
            latency_us = (end_time - self.start_time) / 1000.0  # Convert to microseconds
            
            metadata = {}
            if exc_type is not None:
                metadata['exception'] = str(exc_val)
            
            self.monitor.record_latency(self.category, latency_us, self.operation, metadata)


# Global latency monitor instance
_global_latency_monitor: Optional[LatencyMonitor] = None


def initialize_global_latency_monitor(config: Dict = None) -> LatencyMonitor:
    """Initialize global latency monitor."""
    global _global_latency_monitor
    _global_latency_monitor = LatencyMonitor(config)
    return _global_latency_monitor


def get_global_latency_monitor() -> Optional[LatencyMonitor]:
    """Get global latency monitor."""
    return _global_latency_monitor


def measure_kill_switch_latency(operation: str = ""):
    """Quick context manager for KILL_SWITCH latency measurement."""
    if _global_latency_monitor:
        return _global_latency_monitor.measure_latency(LatencyCategory.KILL_SWITCH, operation)
    else:
        return DummyContext()


def measure_trade_latency(operation: str = ""):
    """Quick context manager for trade execution latency measurement."""
    if _global_latency_monitor:
        return _global_latency_monitor.measure_latency(LatencyCategory.TRADE_EXECUTION, operation)
    else:
        return DummyContext()


def measure_audit_latency(operation: str = ""):
    """Quick context manager for audit logging latency measurement."""
    if _global_latency_monitor:
        return _global_latency_monitor.measure_latency(LatencyCategory.AUDIT_LOGGING, operation)
    else:
        return DummyContext()


class DummyContext:
    """Dummy context manager when monitoring is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass