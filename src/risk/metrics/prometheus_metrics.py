# src/risk/metrics/prometheus_metrics.py
"""
Prometheus Metrics for Risk Management System

Comprehensive metrics collection for:
- Risk sensor evaluations
- VaR/stress test results
- Enforcement actions
- Performance monitoring
- False positive tracking

All metrics are designed for Prometheus/Grafana monitoring.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import logging

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class CollectorRegistry:
        def __init__(self): pass
    
    def generate_latest(registry): return b""
    CONTENT_TYPE_LATEST = "text/plain"


class MetricType(Enum):
    """Types of metrics collected."""
    SENSOR_EVALUATION = "sensor_evaluation"
    RULE_EVALUATION = "rule_evaluation"
    VAR_CALCULATION = "var_calculation"
    STRESS_TEST = "stress_test"
    ENFORCEMENT_ACTION = "enforcement_action"
    PERFORMANCE = "performance"
    FALSE_POSITIVE = "false_positive"


@dataclass
class MetricEvent:
    """Metric event data."""
    metric_type: MetricType
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float


class RiskMetricsCollector:
    """
    Comprehensive metrics collector for risk management system.
    
    Collects and exposes metrics in Prometheus format for monitoring
    and alerting on risk system performance and behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics collector.
        
        Args:
            config: Configuration dictionary with:
                - enabled: Enable metrics collection (default: True)
                - registry: Custom Prometheus registry (optional)
                - namespace: Metric namespace prefix (default: 'risk')
                - subsystem: Metric subsystem (default: 'management')
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.namespace = config.get('namespace', 'risk')
        self.subsystem = config.get('subsystem', 'management')
        
        # Use custom registry if provided, otherwise create new one
        self.registry = config.get('registry', CollectorRegistry() if PROMETHEUS_AVAILABLE else None)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available, metrics will be mocked")
        
        if self.enabled:
            self._initialize_metrics()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.metrics_collected = 0
        self.start_time = time.time()
        
        self.logger.info(f"RiskMetricsCollector initialized (enabled={self.enabled})")
    
    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        
        # === SENSOR METRICS ===
        
        # Sensor evaluation counter
        self.sensor_evaluations_total = Counter(
            'sensor_evaluations_total',
            'Total number of sensor evaluations',
            ['sensor_id', 'sensor_name', 'triggered', 'action'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Sensor evaluation latency
        self.sensor_evaluation_duration_seconds = Histogram(
            'sensor_evaluation_duration_seconds',
            'Time spent evaluating sensors',
            ['sensor_id', 'sensor_name'],
            buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Sensor trigger rate
        self.sensor_trigger_rate = Gauge(
            'sensor_trigger_rate',
            'Rate of sensor triggers per minute',
            ['sensor_id', 'sensor_name'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Current sensor values
        self.sensor_current_value = Gauge(
            'sensor_current_value',
            'Current sensor reading value',
            ['sensor_id', 'sensor_name'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === RULE ENGINE METRICS ===
        
        # Rule evaluation counter
        self.rule_evaluations_total = Counter(
            'rule_evaluations_total',
            'Total number of rule evaluations',
            ['rule_id', 'rule_name', 'triggered', 'action', 'severity'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Rule evaluation latency
        self.rule_evaluation_duration_seconds = Histogram(
            'rule_evaluation_duration_seconds',
            'Time spent evaluating rules',
            ['rule_id', 'rule_name'],
            buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === VAR METRICS ===
        
        # VaR calculation counter
        self.var_calculations_total = Counter(
            'var_calculations_total',
            'Total number of VaR calculations',
            ['method', 'confidence_level'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Current VaR values
        self.var_current_value = Gauge(
            'var_current_value_dollars',
            'Current VaR value in dollars',
            ['confidence_level', 'method'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # VaR limit utilization
        self.var_limit_utilization = Gauge(
            'var_limit_utilization_ratio',
            'VaR utilization as ratio of limit',
            ['confidence_level'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # VaR breaches
        self.var_breaches_total = Counter(
            'var_breaches_total',
            'Total number of VaR limit breaches',
            ['confidence_level', 'severity'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === STRESS TEST METRICS ===
        
        # Stress test runs
        self.stress_tests_total = Counter(
            'stress_tests_total',
            'Total number of stress tests run',
            ['test_type', 'status'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Stress test duration
        self.stress_test_duration_seconds = Histogram(
            'stress_test_duration_seconds',
            'Time spent running stress tests',
            ['test_type'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Stress test worst case loss
        self.stress_test_worst_case_loss = Gauge(
            'stress_test_worst_case_loss_dollars',
            'Worst case loss from stress tests',
            ['test_type'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Stress test scenario failures
        self.stress_test_scenario_failures = Gauge(
            'stress_test_scenario_failures_count',
            'Number of failed stress test scenarios',
            ['test_type'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === ENFORCEMENT METRICS ===
        
        # Enforcement actions
        self.enforcement_actions_total = Counter(
            'enforcement_actions_total',
            'Total number of enforcement actions taken',
            ['action_type', 'reason', 'monitoring_mode'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Monitoring mode status
        self.monitoring_mode_enabled = Gauge(
            'monitoring_mode_enabled',
            'Whether monitoring mode is enabled (1) or enforcement (0)',
            ['component'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === FALSE POSITIVE METRICS ===
        
        # False positive rate
        self.false_positive_rate = Gauge(
            'false_positive_rate_per_week',
            'False positive rate per week',
            ['sensor_id', 'sensor_name'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # False positive total
        self.false_positives_total = Counter(
            'false_positives_total',
            'Total number of false positives identified',
            ['sensor_id', 'sensor_name', 'reason'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === PERFORMANCE METRICS ===
        
        # System latency
        self.system_latency_seconds = Summary(
            'system_latency_seconds',
            'Overall system latency',
            ['component', 'operation'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Memory usage
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # CPU usage
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Data quality
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # === SYSTEM INFO ===
        
        # System information
        self.system_info = Info(
            'system_info',
            'Risk management system information',
            registry=self.registry,
            namespace=self.namespace,
            subsystem=self.subsystem
        )
        
        # Set system info
        self.system_info.info({
            'version': '2.0.0',
            'prometheus_enabled': str(PROMETHEUS_AVAILABLE),
            'namespace': self.namespace,
            'subsystem': self.subsystem
        })
    
    def record_sensor_evaluation(self, sensor_id: str, sensor_name: str,
                               result: Any, duration_seconds: float) -> None:
        """Record sensor evaluation metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                triggered = str(getattr(result, 'triggered', False)).lower()
                action = getattr(result, 'action', None)
                action_str = action.value if hasattr(action, 'value') else 'none'
                
                # Increment evaluation counter
                self.sensor_evaluations_total.labels(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    triggered=triggered,
                    action=action_str
                ).inc()
                
                # Record duration
                self.sensor_evaluation_duration_seconds.labels(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name
                ).observe(duration_seconds)
                
                # Update current value
                sensor_value = getattr(result, 'value', 0)
                if isinstance(sensor_value, (int, float)):
                    self.sensor_current_value.labels(
                        sensor_id=sensor_id,
                        sensor_name=sensor_name
                    ).set(float(sensor_value))
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording sensor metrics: {e}")
    
    def record_rule_evaluation(self, rule_id: str, rule_name: str,
                             result: Any, duration_seconds: float) -> None:
        """Record rule evaluation metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                triggered = str(getattr(result, 'triggered', False)).lower()
                action = getattr(result, 'action', None)
                action_str = action.value if hasattr(action, 'value') else 'none'
                severity = getattr(result, 'severity', None)
                severity_str = severity.value if hasattr(severity, 'value') else 'none'
                
                # Increment evaluation counter
                self.rule_evaluations_total.labels(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    triggered=triggered,
                    action=action_str,
                    severity=severity_str
                ).inc()
                
                # Record duration
                self.rule_evaluation_duration_seconds.labels(
                    rule_id=rule_id,
                    rule_name=rule_name
                ).observe(duration_seconds)
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording rule metrics: {e}")
    
    def record_var_calculation(self, method: str, results: Dict[str, Any],
                             limits: Dict[str, float]) -> None:
        """Record VaR calculation metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                # Record calculation
                for confidence_level in ['95', '99', '999']:
                    var_key = f'var_{confidence_level}'
                    if var_key in results:
                        self.var_calculations_total.labels(
                            method=method,
                            confidence_level=confidence_level
                        ).inc()
                        
                        # Update current VaR value
                        var_value = results[var_key]
                        self.var_current_value.labels(
                            confidence_level=confidence_level,
                            method=method
                        ).set(float(var_value))
                        
                        # Calculate limit utilization
                        limit_key = f'var_{confidence_level}_limit'
                        if limit_key in limits and limits[limit_key] > 0:
                            utilization = var_value / limits[limit_key]
                            self.var_limit_utilization.labels(
                                confidence_level=confidence_level
                            ).set(utilization)
                            
                            # Record breach if over limit
                            if utilization > 1.0:
                                severity = 'critical' if utilization > 2.0 else 'high'
                                self.var_breaches_total.labels(
                                    confidence_level=confidence_level,
                                    severity=severity
                                ).inc()
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording VaR metrics: {e}")
    
    def record_stress_test(self, test_type: str, results: Dict[str, Any],
                         duration_seconds: float, status: str = 'completed') -> None:
        """Record stress test metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                # Record test completion
                self.stress_tests_total.labels(
                    test_type=test_type,
                    status=status
                ).inc()
                
                # Record duration
                self.stress_test_duration_seconds.labels(
                    test_type=test_type
                ).observe(duration_seconds)
                
                # Record worst case loss
                worst_case = results.get('stress_worst_case', 0)
                self.stress_test_worst_case_loss.labels(
                    test_type=test_type
                ).set(float(worst_case))
                
                # Record scenario failures
                failed_scenarios = results.get('failed_scenarios', 0)
                self.stress_test_scenario_failures.labels(
                    test_type=test_type
                ).set(float(failed_scenarios))
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording stress test metrics: {e}")
    
    def record_enforcement_action(self, action_type: str, reason: str,
                                monitoring_mode: bool) -> None:
        """Record enforcement action metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                self.enforcement_actions_total.labels(
                    action_type=action_type,
                    reason=reason,
                    monitoring_mode=str(monitoring_mode).lower()
                ).inc()
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording enforcement metrics: {e}")
    
    def record_false_positive(self, sensor_id: str, sensor_name: str,
                            reason: str, rate_per_week: float) -> None:
        """Record false positive metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                # Increment false positive counter
                self.false_positives_total.labels(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    reason=reason
                ).inc()
                
                # Update false positive rate
                self.false_positive_rate.labels(
                    sensor_id=sensor_id,
                    sensor_name=sensor_name
                ).set(rate_per_week)
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording false positive metrics: {e}")
    
    def update_monitoring_mode_status(self, component: str, enabled: bool) -> None:
        """Update monitoring mode status."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                self.monitoring_mode_enabled.labels(
                    component=component
                ).set(1.0 if enabled else 0.0)
                
            except Exception as e:
                self.logger.error(f"Error updating monitoring mode status: {e}")
    
    def record_performance_metric(self, component: str, operation: str,
                                duration_seconds: float) -> None:
        """Record performance metric."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                self.system_latency_seconds.labels(
                    component=component,
                    operation=operation
                ).observe(duration_seconds)
                
                self.metrics_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error recording performance metric: {e}")
    
    def update_data_quality(self, data_source: str, quality_score: float) -> None:
        """Update data quality score."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                self.data_quality_score.labels(
                    data_source=data_source
                ).set(quality_score)
                
            except Exception as e:
                self.logger.error(f"Error updating data quality: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return ""
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating metrics: {e}")
            return ""
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get metrics collector performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'enabled': self.enabled,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'metrics_collected': self.metrics_collected,
            'metrics_per_second': self.metrics_collected / max(uptime, 1),
            'uptime_seconds': uptime,
            'namespace': self.namespace,
            'subsystem': self.subsystem
        }


def create_metrics_collector(config: Dict[str, Any] = None) -> RiskMetricsCollector:
    """Factory function to create metrics collector."""
    default_config = {
        'enabled': True,
        'namespace': 'risk',
        'subsystem': 'management'
    }
    
    if config:
        final_config = {**default_config, **config}
    else:
        final_config = default_config
    
    return RiskMetricsCollector(final_config)


__all__ = [
    'RiskMetricsCollector',
    'MetricType',
    'MetricEvent',
    'create_metrics_collector',
    'PROMETHEUS_AVAILABLE'
]