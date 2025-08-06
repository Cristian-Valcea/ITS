"""
Stress Test Metrics Collection

Prometheus-compatible metrics collection for stress testing with
real-time monitoring and alerting capabilities.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - using mock metrics")


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: float
    decision_latency_p99: float
    decision_latency_p95: float
    decision_latency_mean: float
    decisions_total: int
    decision_errors: int
    recovery_time_last: Optional[float]
    max_drawdown: float
    position_delta: float
    error_rate: float


class MockMetric:
    """Mock metric for when Prometheus is not available."""
    def inc(self, amount=1): pass
    def observe(self, value): pass
    def set(self, value): pass
    def time(self): return MockTimer()


class MockTimer:
    """Mock timer context manager."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


class StressTestMetrics:
    """
    Comprehensive metrics collection for stress testing.
    
    Provides Prometheus-compatible metrics with fallback to in-memory
    collection when Prometheus is not available.
    """
    
    def __init__(self, enable_prometheus: bool = True, port: int = 8000):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.port = port
        self._lock = threading.Lock()
        
        # In-memory storage for fallback
        self._latencies = deque(maxlen=100000)  # Keep last 100k measurements
        self._decisions_count = 0
        self._errors_count = 0
        self._recovery_times = []
        self._max_drawdown = 0.0
        self._position_delta = 0.0
        
        # Initialize metrics
        self._init_metrics()
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            try:
                start_http_server(port)
                logging.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                logging.warning(f"Failed to start Prometheus server: {e}")
                self.enable_prometheus = False
    
    def _init_metrics(self):
        """Initialize Prometheus metrics or mock equivalents."""
        if self.enable_prometheus:
            # Decision latency histogram (nanoseconds)
            self.decision_latency = Histogram(
                'stress_test_decision_latency_ns',
                'Decision latency in nanoseconds',
                buckets=[1_000_000, 2_000_000, 5_000_000, 10_000_000, 15_000_000, 
                        20_000_000, 30_000_000, 50_000_000, 100_000_000]
            )
            
            # Decision counters
            self.decisions_total = Counter(
                'stress_test_decisions_total',
                'Total number of decisions made'
            )
            
            self.decision_errors = Counter(
                'stress_test_decision_errors_total',
                'Total number of decision errors'
            )
            
            # Recovery time histogram (seconds)
            self.recovery_time = Histogram(
                'stress_test_recovery_time_seconds',
                'Recovery time after failure in seconds',
                buckets=[1, 5, 10, 15, 20, 25, 30, 45, 60]
            )
            
            # Risk metrics
            self.max_drawdown = Gauge(
                'stress_test_max_drawdown_percent',
                'Maximum drawdown percentage during test'
            )
            
            self.position_delta = Gauge(
                'stress_test_position_delta_usd',
                'Position delta in USD for integrity validation'
            )
            
            # System health
            self.memory_usage = Gauge(
                'stress_test_memory_usage_mb',
                'Memory usage in MB during stress test'
            )
            
            self.cpu_usage = Gauge(
                'stress_test_cpu_usage_percent',
                'CPU usage percentage during stress test'
            )
            
        else:
            # Use mock metrics
            self.decision_latency = MockMetric()
            self.decisions_total = MockMetric()
            self.decision_errors = MockMetric()
            self.recovery_time = MockMetric()
            self.max_drawdown = MockMetric()
            self.position_delta = MockMetric()
            self.memory_usage = MockMetric()
            self.cpu_usage = MockMetric()
    
    def timing(self, metric_name: str, value_ns: int):
        """Record timing metric in nanoseconds."""
        with self._lock:
            self._latencies.append(value_ns)
            
        if metric_name == 'decision_ns':
            self.decision_latency.observe(value_ns)
    
    def counter(self, metric_name: str):
        """Get counter metric."""
        if metric_name == 'decisions_total':
            return CounterWrapper(self.decisions_total, self._increment_decisions)
        elif metric_name == 'decision_errors':
            return CounterWrapper(self.decision_errors, self._increment_errors)
        else:
            return MockMetric()
    
    def gauge(self, metric_name: str):
        """Get gauge metric."""
        if metric_name == 'max_drawdown_percent':
            return GaugeWrapper(self.max_drawdown, self._set_max_drawdown)
        elif metric_name == 'position_delta_usd':
            return GaugeWrapper(self.position_delta, self._set_position_delta)
        else:
            return MockMetric()
    
    def _increment_decisions(self):
        """Increment decisions counter."""
        with self._lock:
            self._decisions_count += 1
    
    def _increment_errors(self):
        """Increment errors counter."""
        with self._lock:
            self._errors_count += 1
    
    def _set_max_drawdown(self, value: float):
        """Set maximum drawdown."""
        with self._lock:
            self._max_drawdown = max(self._max_drawdown, value)
    
    def _set_position_delta(self, value: float):
        """Set position delta."""
        with self._lock:
            self._position_delta = value
    
    def record_recovery_time(self, recovery_time_s: float):
        """Record recovery time after failure."""
        with self._lock:
            self._recovery_times.append(recovery_time_s)
        self.recovery_time.observe(recovery_time_s)
    
    def get_snapshot(self) -> MetricSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            latencies_ms = [lat / 1_000_000 for lat in self._latencies]  # Convert to ms
            
            if latencies_ms:
                p99 = statistics.quantiles(latencies_ms, n=100)[98]  # 99th percentile
                p95 = statistics.quantiles(latencies_ms, n=20)[18]   # 95th percentile
                mean = statistics.mean(latencies_ms)
            else:
                p99 = p95 = mean = 0.0
            
            error_rate = (self._errors_count / max(self._decisions_count, 1)) * 100
            
            return MetricSnapshot(
                timestamp=time.time(),
                decision_latency_p99=p99,
                decision_latency_p95=p95,
                decision_latency_mean=mean,
                decisions_total=self._decisions_count,
                decision_errors=self._errors_count,
                recovery_time_last=self._recovery_times[-1] if self._recovery_times else None,
                max_drawdown=self._max_drawdown,
                position_delta=self._position_delta,
                error_rate=error_rate
            )
    
    def get_latency_percentiles(self, percentiles: List[int] = [50, 95, 99]) -> Dict[int, float]:
        """Get latency percentiles in milliseconds."""
        with self._lock:
            if not self._latencies:
                return {p: 0.0 for p in percentiles}
            
            latencies_ms = [lat / 1_000_000 for lat in self._latencies]
            result = {}
            
            for p in percentiles:
                if p == 50:
                    result[p] = statistics.median(latencies_ms)
                else:
                    # Use quantiles for other percentiles
                    n = 100 if p <= 100 else 1000
                    idx = int((p / 100) * n) - 1
                    result[p] = statistics.quantiles(latencies_ms, n=n)[idx]
            
            return result
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._decisions_count = 0
            self._errors_count = 0
            self._recovery_times.clear()
            self._max_drawdown = 0.0
            self._position_delta = 0.0
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary for reporting."""
        snapshot = self.get_snapshot()
        percentiles = self.get_latency_percentiles([50, 95, 99])
        
        return {
            'timestamp': snapshot.timestamp,
            'latency': {
                'p50_ms': percentiles[50],
                'p95_ms': percentiles[95],
                'p99_ms': percentiles[99],
                'mean_ms': snapshot.decision_latency_mean
            },
            'decisions': {
                'total': snapshot.decisions_total,
                'errors': snapshot.decision_errors,
                'error_rate_pct': snapshot.error_rate
            },
            'recovery': {
                'last_time_s': snapshot.recovery_time_last,
                'all_times_s': self._recovery_times.copy()
            },
            'risk': {
                'max_drawdown_pct': snapshot.max_drawdown,
                'position_delta_usd': snapshot.position_delta
            }
        }


class CounterWrapper:
    """Wrapper for counter metrics to handle both Prometheus and in-memory."""
    
    def __init__(self, prometheus_counter, callback):
        self.prometheus_counter = prometheus_counter
        self.callback = callback
    
    def inc(self, amount=1):
        self.prometheus_counter.inc(amount)
        for _ in range(amount):
            self.callback()


class GaugeWrapper:
    """Wrapper for gauge metrics to handle both Prometheus and in-memory."""
    
    def __init__(self, prometheus_gauge, callback):
        self.prometheus_gauge = prometheus_gauge
        self.callback = callback
    
    def set(self, value):
        self.prometheus_gauge.set(value)
        self.callback(value)


# Global metrics instance
_global_metrics: Optional[StressTestMetrics] = None


def get_metrics() -> StressTestMetrics:
    """Get the global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = StressTestMetrics()
    return _global_metrics


def init_metrics(enable_prometheus: bool = True, port: int = 8000) -> StressTestMetrics:
    """Initialize global metrics instance."""
    global _global_metrics
    _global_metrics = StressTestMetrics(enable_prometheus, port)
    return _global_metrics