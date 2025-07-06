# src/risk/sensors/sensor_registry.py
"""
Sensor Registry - Fast sensor lookup and execution management.

Provides O(1) sensor lookup, priority-based execution, and performance monitoring.
Designed for microsecond-level latency in critical paths.
"""

import time
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .base_sensor import BaseSensor, SensorResult, SensorPriority, FailureMode, SensorAction


class SensorRegistry:
    """
    High-performance sensor registry with priority-based execution.
    
    Features:
    - O(1) sensor lookup by ID or failure mode
    - Priority-based execution lanes
    - Circuit breaker for failing sensors
    - Performance monitoring and alerting
    - Hot-swappable sensor updates
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Sensor storage
        self._sensors: Dict[str, BaseSensor] = {}
        self._sensors_by_priority: Dict[SensorPriority, List[BaseSensor]] = defaultdict(list)
        self._sensors_by_failure_mode: Dict[FailureMode, List[BaseSensor]] = defaultdict(list)
        
        # Execution management
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SensorWorker")
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Performance tracking
        self._execution_stats = {
            'total_evaluations': 0,
            'total_evaluation_time_ns': 0,
            'priority_stats': defaultdict(lambda: {'count': 0, 'time_ns': 0}),
            'failure_mode_stats': defaultdict(lambda: {'count': 0, 'triggers': 0}),
            'latency_violations': 0
        }
        
        # Recent results for analysis
        self._recent_results: deque = deque(maxlen=1000)
        
        self.logger.info("SensorRegistry initialized")
    
    def register_sensor(self, sensor: BaseSensor) -> None:
        """Register a sensor in the registry."""
        sensor_id = sensor.sensor_id
        
        if sensor_id in self._sensors:
            self.logger.warning(f"Overwriting existing sensor: {sensor_id}")
        
        # Store sensor
        self._sensors[sensor_id] = sensor
        self._sensors_by_priority[sensor.priority].append(sensor)
        self._sensors_by_failure_mode[sensor.failure_mode].append(sensor)
        
        # Initialize circuit breaker
        self._circuit_breakers[sensor_id] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            sensor_name=sensor.sensor_name
        )
        
        self.logger.info(f"Registered sensor {sensor.sensor_name} (ID: {sensor_id})")
    
    def unregister_sensor(self, sensor_id: str) -> bool:
        """Unregister a sensor from the registry."""
        if sensor_id not in self._sensors:
            return False
        
        sensor = self._sensors.pop(sensor_id)
        
        # Remove from priority and failure mode indexes
        self._sensors_by_priority[sensor.priority].remove(sensor)
        self._sensors_by_failure_mode[sensor.failure_mode].remove(sensor)
        
        # Remove circuit breaker
        self._circuit_breakers.pop(sensor_id, None)
        
        self.logger.info(f"Unregistered sensor {sensor.sensor_name} (ID: {sensor_id})")
        return True
    
    def get_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        """Get sensor by ID."""
        return self._sensors.get(sensor_id)
    
    def get_sensors_by_priority(self, priority: SensorPriority) -> List[BaseSensor]:
        """Get all sensors with specified priority."""
        return self._sensors_by_priority[priority].copy()
    
    def get_sensors_by_failure_mode(self, failure_mode: FailureMode) -> List[BaseSensor]:
        """Get all sensors for specified failure mode."""
        return self._sensors_by_failure_mode[failure_mode].copy()
    
    def evaluate_sensor(self, sensor_id: str, data: Dict[str, Any]) -> Optional[SensorResult]:
        """Evaluate a single sensor with circuit breaker protection."""
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            self.logger.error(f"Sensor not found: {sensor_id}")
            return None
        
        circuit_breaker = self._circuit_breakers[sensor_id]
        if not circuit_breaker.can_execute():
            self.logger.debug(f"Circuit breaker open for sensor: {sensor_id}")
            return None
        
        start_time = time.time_ns()
        
        try:
            result = sensor.evaluate(data)
            
            # Update circuit breaker
            if result and not result.message.startswith("ERROR"):
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()
            
            # Update performance stats
            evaluation_time = time.time_ns() - start_time
            self._update_stats(sensor, result, evaluation_time)
            
            # Store recent result
            self._recent_results.append(result)
            
            return result
            
        except Exception as e:
            circuit_breaker.record_failure()
            self.logger.error(f"Sensor evaluation failed for {sensor_id}: {e}")
            return None
    
    def evaluate_priority_group(self, priority: SensorPriority, data: Dict[str, Any]) -> List[SensorResult]:
        """Evaluate all sensors in a priority group."""
        sensors = self.get_sensors_by_priority(priority)
        if not sensors:
            return []
        
        results = []
        
        # For critical sensors, evaluate synchronously for minimum latency
        if priority == SensorPriority.CRITICAL:
            for sensor in sensors:
                result = self.evaluate_sensor(sensor.sensor_id, data)
                if result:
                    results.append(result)
        else:
            # For non-critical sensors, can use parallel execution
            futures = []
            for sensor in sensors:
                future = self._executor.submit(self.evaluate_sensor, sensor.sensor_id, data)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=1.0)  # 1 second timeout
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Sensor evaluation timeout/error: {e}")
        
        return results
    
    def evaluate_failure_mode(self, failure_mode: FailureMode, data: Dict[str, Any]) -> List[SensorResult]:
        """Evaluate all sensors for a specific failure mode."""
        sensors = self.get_sensors_by_failure_mode(failure_mode)
        if not sensors:
            return []
        
        results = []
        for sensor in sensors:
            result = self.evaluate_sensor(sensor.sensor_id, data)
            if result:
                results.append(result)
        
        return results
    
    def evaluate_all_sensors(self, data: Dict[str, Any]) -> Dict[SensorPriority, List[SensorResult]]:
        """Evaluate all sensors grouped by priority."""
        results = {}
        
        # Evaluate in priority order
        for priority in SensorPriority:
            priority_results = self.evaluate_priority_group(priority, data)
            if priority_results:
                results[priority] = priority_results
        
        return results
    
    def get_triggered_sensors(self, data: Dict[str, Any]) -> List[SensorResult]:
        """Get only triggered sensors (for fast kill-switch path)."""
        triggered = []
        
        # Check critical sensors first
        critical_results = self.evaluate_priority_group(SensorPriority.CRITICAL, data)
        for result in critical_results:
            if result.triggered:
                triggered.append(result)
        
        # If critical sensors triggered, return immediately
        if triggered:
            return triggered
        
        # Check high priority sensors
        high_results = self.evaluate_priority_group(SensorPriority.HIGH, data)
        for result in high_results:
            if result.triggered:
                triggered.append(result)
        
        return triggered
    
    def _update_stats(self, sensor: BaseSensor, result: SensorResult, evaluation_time_ns: int):
        """Update performance statistics."""
        self._execution_stats['total_evaluations'] += 1
        self._execution_stats['total_evaluation_time_ns'] += evaluation_time_ns
        
        # Priority stats
        priority_stats = self._execution_stats['priority_stats'][sensor.priority]
        priority_stats['count'] += 1
        priority_stats['time_ns'] += evaluation_time_ns
        
        # Failure mode stats
        failure_stats = self._execution_stats['failure_mode_stats'][sensor.failure_mode]
        failure_stats['count'] += 1
        if result and result.triggered:
            failure_stats['triggers'] += 1
        
        # Latency violations
        evaluation_time_us = evaluation_time_ns / 1000.0
        if evaluation_time_us > sensor.latency_budget_us:
            self._execution_stats['latency_violations'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._execution_stats.copy()
        
        # Add sensor-level stats
        sensor_stats = {}
        for sensor_id, sensor in self._sensors.items():
            sensor_stats[sensor_id] = sensor.get_performance_stats()
        
        stats['sensor_stats'] = sensor_stats
        
        # Add circuit breaker stats
        circuit_stats = {}
        for sensor_id, cb in self._circuit_breakers.items():
            circuit_stats[sensor_id] = cb.get_stats()
        
        stats['circuit_breaker_stats'] = circuit_stats
        
        # Add registry-level metrics
        stats['registered_sensors'] = len(self._sensors)
        stats['sensors_by_priority'] = {
            priority.name: len(sensors) 
            for priority, sensors in self._sensors_by_priority.items()
        }
        stats['sensors_by_failure_mode'] = {
            mode.name: len(sensors) 
            for mode, sensors in self._sensors_by_failure_mode.items()
        }
        
        return stats
    
    def get_recent_triggers(self, limit: int = 10) -> List[SensorResult]:
        """Get recent triggered sensor results."""
        triggered = [r for r in self._recent_results if r.triggered]
        return list(triggered)[-limit:]
    
    def shutdown(self):
        """Shutdown the sensor registry."""
        self.logger.info("Shutting down sensor registry")
        self._executor.shutdown(wait=True)


class CircuitBreaker:
    """Circuit breaker for sensor fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, sensor_name: str = ""):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.sensor_name = sensor_name
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(f"CircuitBreaker.{sensor_name}")
    
    def can_execute(self) -> bool:
        """Check if sensor can be executed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker half-open for {self.sensor_name}")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful sensor execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info(f"Circuit breaker closed for {self.sensor_name}")
    
    def record_failure(self):
        """Record failed sensor execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened for {self.sensor_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }