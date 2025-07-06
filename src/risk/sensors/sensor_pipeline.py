# src/risk/sensors/sensor_pipeline.py
"""
Sensor Pipeline - Multi-lane processing for risk sensors.

Implements fast/slow lane architecture:
- Fast Lane: <100µs for critical sensors (kill-switch, pre-trade)
- Slow Lane: <100ms for analytics sensors (monitoring, reporting)

Design inspired by self-driving car sensor fusion systems.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

from .base_sensor import BaseSensor, SensorResult, SensorPriority, SensorAction, FailureMode
from .sensor_registry import SensorRegistry


@dataclass
class PipelineResult:
    """Result from sensor pipeline execution."""
    fast_lane_results: List[SensorResult]
    slow_lane_results: List[SensorResult]
    execution_time_us: float
    kill_switch_triggered: bool
    throttle_recommended: bool
    alerts_generated: List[str]
    metadata: Dict[str, Any]


class FastLane:
    """
    Fast Lane - Ultra-low latency sensor execution for critical decisions.
    
    Target: <100µs total execution time
    Sensors: CRITICAL and HIGH priority only
    Actions: KILL_SWITCH, THROTTLE
    """
    
    def __init__(self, sensor_registry: SensorRegistry):
        self.sensor_registry = sensor_registry
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time_ns = 0
        self._kill_switch_count = 0
        self._throttle_count = 0
        
        # Latency budget enforcement
        self.latency_budget_us = 100.0
        self.latency_violations = 0
        
        self.logger.info("FastLane initialized with 100µs latency budget")
    
    def execute(self, data: Dict[str, Any]) -> List[SensorResult]:
        """Execute critical sensors in fast lane."""
        start_time = time.time_ns()
        results = []
        
        try:
            # Execute CRITICAL sensors first (synchronously for minimum latency)
            critical_results = self.sensor_registry.evaluate_priority_group(
                SensorPriority.CRITICAL, data
            )
            results.extend(critical_results)
            
            # Check for immediate kill switch
            for result in critical_results:
                if result.action == SensorAction.KILL_SWITCH:
                    self._kill_switch_count += 1
                    # Return immediately on kill switch
                    self._update_stats(start_time)
                    return results
            
            # Execute HIGH priority sensors if no kill switch
            high_results = self.sensor_registry.evaluate_priority_group(
                SensorPriority.HIGH, data
            )
            results.extend(high_results)
            
            # Count throttle recommendations
            for result in high_results:
                if result.action == SensorAction.THROTTLE:
                    self._throttle_count += 1
            
            self._update_stats(start_time)
            return results
            
        except Exception as e:
            self.logger.error(f"FastLane execution failed: {e}")
            self._update_stats(start_time)
            return results
    
    def _update_stats(self, start_time: int):
        """Update performance statistics."""
        execution_time = time.time_ns() - start_time
        self._execution_count += 1
        self._total_execution_time_ns += execution_time
        
        # Check latency budget
        execution_time_us = execution_time / 1000.0
        if execution_time_us > self.latency_budget_us:
            self.latency_violations += 1
            self.logger.warning(
                f"FastLane latency violation: {execution_time_us:.2f}µs > {self.latency_budget_us}µs"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get fast lane performance statistics."""
        if self._execution_count == 0:
            return {
                'execution_count': 0,
                'avg_execution_time_us': 0.0,
                'latency_budget_us': self.latency_budget_us,
                'latency_violations': 0,
                'kill_switch_count': 0,
                'throttle_count': 0
            }
        
        avg_time_ns = self._total_execution_time_ns / self._execution_count
        
        return {
            'execution_count': self._execution_count,
            'avg_execution_time_us': avg_time_ns / 1000.0,
            'latency_budget_us': self.latency_budget_us,
            'latency_violations': self.latency_violations,
            'violation_rate': self.latency_violations / self._execution_count,
            'kill_switch_count': self._kill_switch_count,
            'throttle_count': self._throttle_count
        }


class SlowLane:
    """
    Slow Lane - Higher latency sensor execution for analytics and monitoring.
    
    Target: <100ms total execution time
    Sensors: MEDIUM, LOW, BATCH priority
    Actions: ALERT, MONITOR, analytics
    """
    
    def __init__(self, sensor_registry: SensorRegistry, max_workers: int = 4):
        self.sensor_registry = sensor_registry
        self.max_workers = max_workers
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="SlowLane"
        )
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time_ns = 0
        self._alert_count = 0
        
        # Latency budget
        self.latency_budget_ms = 100.0
        self.latency_violations = 0
        
        self.logger.info(f"SlowLane initialized with {max_workers} workers, 100ms budget")
    
    def execute(self, data: Dict[str, Any]) -> List[SensorResult]:
        """Execute analytics sensors in slow lane."""
        start_time = time.time_ns()
        results = []
        
        try:
            # Execute MEDIUM priority sensors in parallel
            medium_futures = []
            medium_sensors = self.sensor_registry.get_sensors_by_priority(SensorPriority.MEDIUM)
            
            for sensor in medium_sensors:
                future = self._executor.submit(
                    self.sensor_registry.evaluate_sensor, 
                    sensor.sensor_id, 
                    data
                )
                medium_futures.append(future)
            
            # Collect MEDIUM results with timeout
            for future in as_completed(medium_futures, timeout=0.05):  # 50ms timeout
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if result.action == SensorAction.ALERT:
                            self._alert_count += 1
                except Exception as e:
                    self.logger.debug(f"Medium priority sensor failed: {e}")
            
            # Execute LOW priority sensors if time permits
            elapsed_time_ms = (time.time_ns() - start_time) / 1_000_000
            if elapsed_time_ms < self.latency_budget_ms * 0.7:  # 70ms remaining
                low_results = self.sensor_registry.evaluate_priority_group(
                    SensorPriority.LOW, data
                )
                results.extend(low_results)
                
                for result in low_results:
                    if result.action == SensorAction.ALERT:
                        self._alert_count += 1
            
            self._update_stats(start_time)
            return results
            
        except Exception as e:
            self.logger.error(f"SlowLane execution failed: {e}")
            self._update_stats(start_time)
            return results
    
    def execute_batch(self, data: Dict[str, Any]) -> List[SensorResult]:
        """Execute batch analytics sensors (no time limit)."""
        batch_results = self.sensor_registry.evaluate_priority_group(
            SensorPriority.BATCH, data
        )
        return batch_results
    
    def _update_stats(self, start_time: int):
        """Update performance statistics."""
        execution_time = time.time_ns() - start_time
        self._execution_count += 1
        self._total_execution_time_ns += execution_time
        
        # Check latency budget
        execution_time_ms = execution_time / 1_000_000
        if execution_time_ms > self.latency_budget_ms:
            self.latency_violations += 1
            self.logger.warning(
                f"SlowLane latency violation: {execution_time_ms:.2f}ms > {self.latency_budget_ms}ms"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get slow lane performance statistics."""
        if self._execution_count == 0:
            return {
                'execution_count': 0,
                'avg_execution_time_ms': 0.0,
                'latency_budget_ms': self.latency_budget_ms,
                'latency_violations': 0,
                'alert_count': 0
            }
        
        avg_time_ns = self._total_execution_time_ns / self._execution_count
        
        return {
            'execution_count': self._execution_count,
            'avg_execution_time_ms': avg_time_ns / 1_000_000,
            'latency_budget_ms': self.latency_budget_ms,
            'latency_violations': self.latency_violations,
            'violation_rate': self.latency_violations / self._execution_count,
            'alert_count': self._alert_count,
            'max_workers': self.max_workers
        }
    
    def shutdown(self):
        """Shutdown the slow lane executor."""
        self._executor.shutdown(wait=True)


class SensorPipeline:
    """
    Main Sensor Pipeline - Orchestrates fast and slow lane execution.
    
    Architecture:
    1. Fast Lane: Critical sensors for immediate decisions
    2. Slow Lane: Analytics sensors for monitoring
    3. Result aggregation and action determination
    4. Performance monitoring and alerting
    """
    
    def __init__(self, sensor_registry: SensorRegistry, config: Dict[str, Any] = None):
        self.sensor_registry = sensor_registry
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize execution lanes
        self.fast_lane = FastLane(sensor_registry)
        self.slow_lane = SlowLane(
            sensor_registry, 
            max_workers=self.config.get('slow_lane_workers', 4)
        )
        
        # Pipeline state
        self._execution_count = 0
        self._kill_switch_history = deque(maxlen=100)
        self._alert_history = deque(maxlen=1000)
        
        # Action callbacks
        self._action_callbacks: Dict[SensorAction, List[Callable]] = defaultdict(list)
        
        self.logger.info("SensorPipeline initialized with fast/slow lane architecture")
    
    def register_action_callback(self, action: SensorAction, callback: Callable):
        """Register callback for specific sensor actions."""
        self._action_callbacks[action].append(callback)
        self.logger.info(f"Registered callback for {action.value} action")
    
    def execute(self, data: Dict[str, Any]) -> PipelineResult:
        """Execute complete sensor pipeline."""
        start_time = time.time_ns()
        
        # Execute fast lane first
        fast_results = self.fast_lane.execute(data)
        
        # Check for kill switch
        kill_switch_triggered = any(
            result.action == SensorAction.KILL_SWITCH 
            for result in fast_results
        )
        
        if kill_switch_triggered:
            # Record kill switch event
            self._kill_switch_history.append({
                'timestamp': time.time(),
                'sensors': [r.sensor_name for r in fast_results if r.action == SensorAction.KILL_SWITCH],
                'data_snapshot': self._create_data_snapshot(data)
            })
            
            # Execute kill switch callbacks
            self._execute_action_callbacks(SensorAction.KILL_SWITCH, fast_results, data)
            
            # Return immediately without slow lane execution
            execution_time_us = (time.time_ns() - start_time) / 1000.0
            
            return PipelineResult(
                fast_lane_results=fast_results,
                slow_lane_results=[],
                execution_time_us=execution_time_us,
                kill_switch_triggered=True,
                throttle_recommended=False,
                alerts_generated=[],
                metadata={'kill_switch_immediate': True}
            )
        
        # Execute slow lane if no kill switch
        slow_results = self.slow_lane.execute(data)
        
        # Aggregate results and determine actions
        all_results = fast_results + slow_results
        
        throttle_recommended = any(
            result.action == SensorAction.THROTTLE 
            for result in all_results
        )
        
        alerts_generated = [
            result.message for result in all_results 
            if result.action == SensorAction.ALERT
        ]
        
        # Execute action callbacks
        for action in [SensorAction.THROTTLE, SensorAction.ALERT, SensorAction.MONITOR]:
            relevant_results = [r for r in all_results if r.action == action]
            if relevant_results:
                self._execute_action_callbacks(action, relevant_results, data)
        
        # Store alerts in history
        for alert in alerts_generated:
            self._alert_history.append({
                'timestamp': time.time(),
                'message': alert
            })
        
        # Calculate total execution time
        execution_time_us = (time.time_ns() - start_time) / 1000.0
        
        self._execution_count += 1
        
        return PipelineResult(
            fast_lane_results=fast_results,
            slow_lane_results=slow_results,
            execution_time_us=execution_time_us,
            kill_switch_triggered=False,
            throttle_recommended=throttle_recommended,
            alerts_generated=alerts_generated,
            metadata={
                'total_sensors': len(all_results),
                'triggered_sensors': len([r for r in all_results if r.triggered])
            }
        )
    
    def _execute_action_callbacks(self, action: SensorAction, results: List[SensorResult], data: Dict[str, Any]):
        """Execute registered callbacks for an action."""
        callbacks = self._action_callbacks.get(action, [])
        
        for callback in callbacks:
            try:
                callback(action, results, data)
            except Exception as e:
                self.logger.error(f"Action callback failed for {action.value}: {e}")
    
    def _create_data_snapshot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a snapshot of key data for debugging."""
        snapshot = {}
        
        # Include key fields for analysis
        key_fields = ['portfolio_value', 'returns', 'positions', 'leverage', 'timestamp']
        
        for field in key_fields:
            if field in data:
                value = data[field]
                # Truncate large arrays
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    snapshot[field] = f"Array[{len(value)}]: {value[-5:]}"  # Last 5 values
                else:
                    snapshot[field] = value
        
        return snapshot
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance statistics."""
        fast_stats = self.fast_lane.get_performance_stats()
        slow_stats = self.slow_lane.get_performance_stats()
        registry_stats = self.sensor_registry.get_performance_stats()
        
        return {
            'pipeline': {
                'execution_count': self._execution_count,
                'kill_switch_events': len(self._kill_switch_history),
                'recent_alerts': len(self._alert_history)
            },
            'fast_lane': fast_stats,
            'slow_lane': slow_stats,
            'sensor_registry': registry_stats
        }
    
    def get_recent_kill_switches(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent kill switch events."""
        return list(self._kill_switch_history)[-limit:]
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alert messages."""
        return list(self._alert_history)[-limit:]
    
    def shutdown(self):
        """Shutdown the sensor pipeline."""
        self.logger.info("Shutting down sensor pipeline")
        self.slow_lane.shutdown()
        self.sensor_registry.shutdown()