# src/risk/sensors/base_sensor.py
"""
Base Sensor Framework - Core abstractions for risk sensors.

Treats risk as real-time sensors that detect failure modes, not just metrics.
Each sensor has a latency budget, data requirements, and actionability level.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np


class SensorPriority(Enum):
    """Sensor priority levels with latency budgets."""
    CRITICAL = 0    # Kill-switch sensors: <50µs (pre-trade)
    HIGH = 1        # Throttle sensors: <100µs (execution)
    MEDIUM = 2      # Alert sensors: <1ms (monitoring)
    LOW = 3         # Analytics sensors: <100ms (reporting)
    BATCH = 4       # Heavy analytics: seconds (offline)


class FailureMode(Enum):
    """Types of failure modes that sensors detect."""
    PATH_FRAGILITY = "path_fragility"
    TAIL_REGIME_SHIFT = "tail_regime_shift"
    LIQUIDITY_EXECUTION = "liquidity_execution"
    FUNDING_MARGIN = "funding_margin"
    COUNTERPARTY_SETTLEMENT = "counterparty_settlement"
    OPERATIONAL_TECH = "operational_tech"


class SensorAction(Enum):
    """Actions that can be triggered by sensors."""
    KILL_SWITCH = "kill_switch"      # Immediate halt
    THROTTLE = "throttle"            # Reduce position size
    HEDGE = "hedge"                  # Add protective positions
    ALERT = "alert"                  # Notify operators
    MONITOR = "monitor"              # Log for analysis
    NONE = "none"                    # No action needed


@dataclass
class SensorResult:
    """Result from a sensor evaluation."""
    sensor_id: str
    sensor_name: str
    failure_mode: FailureMode
    priority: SensorPriority
    
    # Sensor readings
    value: float
    threshold: float
    triggered: bool
    confidence: float  # 0.0 to 1.0
    
    # Action recommendation
    action: SensorAction
    severity: float  # 0.0 to 1.0
    message: str
    
    # Performance tracking
    evaluation_time_ns: int = field(default_factory=lambda: time.time_ns())
    data_age_ns: int = 0  # Age of input data
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_evaluation_time_us(self) -> float:
        """Get evaluation time in microseconds."""
        return (time.time_ns() - self.evaluation_time_ns) / 1000.0
    
    def is_actionable(self) -> bool:
        """Check if sensor result requires action."""
        return self.triggered and self.action != SensorAction.NONE
    
    def get_risk_score(self) -> float:
        """Get normalized risk score (0.0 to 1.0)."""
        if self.threshold == 0:
            return 0.0
        return min(1.0, abs(self.value / self.threshold) * self.confidence)


class BaseSensor(ABC):
    """
    Abstract base class for all risk sensors.
    
    Design Principles:
    1. Each sensor detects one specific failure mode
    2. Latency budget enforced based on priority
    3. Data requirements clearly specified
    4. Actionability level defined (kill-switch, throttle, alert)
    """
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        self.sensor_id = sensor_id
        self.sensor_name = sensor_name
        self.config = config
        self.logger = logging.getLogger(f"Sensor.{sensor_name}")
        
        # Sensor characteristics
        self.failure_mode = self._get_failure_mode()
        self.priority = SensorPriority(config.get('priority', SensorPriority.MEDIUM.value))
        self.latency_budget_us = self._get_latency_budget()
        
        # Thresholds and parameters
        self.threshold = float(config.get('threshold', 1.0))
        self.confidence_threshold = float(config.get('confidence_threshold', 0.7))
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        self._trigger_count = 0
        self._false_positive_count = 0
        
        # State management
        self._last_result: Optional[SensorResult] = None
        self._calibration_data: List[float] = []
        self._enabled = bool(config.get('enabled', True))
        
        self.logger.info(f"Sensor {sensor_name} initialized with {self.priority.name} priority")
    
    @abstractmethod
    def _get_failure_mode(self) -> FailureMode:
        """Return the failure mode this sensor detects."""
        pass
    
    @abstractmethod
    def _get_data_requirements(self) -> List[str]:
        """Return list of required data fields."""
        pass
    
    @abstractmethod
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute the raw sensor value from input data."""
        pass
    
    def _get_latency_budget(self) -> float:
        """Get latency budget in microseconds based on priority."""
        budgets = {
            SensorPriority.CRITICAL: 50.0,
            SensorPriority.HIGH: 100.0,
            SensorPriority.MEDIUM: 1000.0,
            SensorPriority.LOW: 100000.0,
            SensorPriority.BATCH: 1000000.0
        }
        return budgets.get(self.priority, 1000.0)
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate that required data is present and valid."""
        required_fields = self._get_data_requirements()
        
        for field in required_fields:
            if field not in data:
                self.logger.warning(f"Missing required field: {field}")
                return False
            
            value = data[field]
            if value is None:
                self.logger.warning(f"Null value for required field: {field}")
                return False
            
            # Check for stale data
            if isinstance(value, dict) and 'timestamp' in value:
                age_ns = time.time_ns() - value['timestamp']
                if age_ns > 60_000_000_000:  # 60 seconds
                    self.logger.warning(f"Stale data for field {field}: {age_ns/1e9:.1f}s old")
                    return False
        
        return True
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence in the sensor reading (0.0 to 1.0)."""
        # Base confidence on data quality and sensor calibration
        base_confidence = 0.8
        
        # Adjust for data age
        data_age_penalty = 0.0
        if 'timestamp' in data:
            age_ns = time.time_ns() - data['timestamp']
            if age_ns > 1_000_000_000:  # 1 second
                data_age_penalty = min(0.3, age_ns / 10_000_000_000)  # Max 30% penalty
        
        # Adjust for sensor calibration
        calibration_bonus = 0.0
        if len(self._calibration_data) > 10:
            # Higher confidence if sensor has been calibrated
            calibration_bonus = 0.1
        
        confidence = base_confidence - data_age_penalty + calibration_bonus
        return max(0.0, min(1.0, confidence))
    
    def _determine_action(self, value: float, threshold: float, confidence: float) -> SensorAction:
        """Determine what action should be taken based on sensor reading."""
        if not self._is_triggered(value, threshold, confidence):
            return SensorAction.NONE
        
        # Action based on priority and severity
        severity = self._compute_severity(value, threshold)
        
        if self.priority == SensorPriority.CRITICAL:
            return SensorAction.KILL_SWITCH
        elif self.priority == SensorPriority.HIGH:
            if severity > 0.8:
                return SensorAction.KILL_SWITCH
            else:
                return SensorAction.THROTTLE
        elif self.priority == SensorPriority.MEDIUM:
            if severity > 0.9:
                return SensorAction.THROTTLE
            else:
                return SensorAction.ALERT
        else:
            return SensorAction.MONITOR
    
    def _is_triggered(self, value: float, threshold: float, confidence: float) -> bool:
        """Determine if sensor is triggered."""
        if confidence < self.confidence_threshold:
            return False
        
        return abs(value) > abs(threshold)
    
    def _compute_severity(self, value: float, threshold: float) -> float:
        """Compute severity of the sensor trigger (0.0 to 1.0)."""
        if threshold == 0:
            return 1.0 if value != 0 else 0.0
        
        ratio = abs(value / threshold)
        # Sigmoid function to map ratio to severity
        return min(1.0, 1.0 / (1.0 + np.exp(-2 * (ratio - 1))))
    
    def _format_message(self, value: float, threshold: float, action: SensorAction) -> str:
        """Format human-readable message for sensor result."""
        return (f"{self.sensor_name}: {value:.4f} (threshold: {threshold:.4f}) "
                f"→ {action.value}")
    
    def evaluate(self, data: Dict[str, Any]) -> SensorResult:
        """
        Evaluate sensor with latency budget enforcement.
        
        Args:
            data: Input data dictionary
            
        Returns:
            SensorResult with sensor reading and recommended action
        """
        if not self._enabled:
            return self._create_disabled_result()
        
        start_time = time.time_ns()
        
        try:
            # Validate input data
            if not self._validate_data(data):
                return self._create_error_result("Invalid input data")
            
            # Compute sensor value
            value = self._compute_sensor_value(data)
            
            # Compute confidence
            confidence = self._compute_confidence(value, data)
            
            # Determine if triggered and what action to take
            triggered = self._is_triggered(value, self.threshold, confidence)
            action = self._determine_action(value, self.threshold, confidence)
            severity = self._compute_severity(value, self.threshold) if triggered else 0.0
            
            # Format message
            message = self._format_message(value, self.threshold, action)
            
            # Create result
            result = SensorResult(
                sensor_id=self.sensor_id,
                sensor_name=self.sensor_name,
                failure_mode=self.failure_mode,
                priority=self.priority,
                value=value,
                threshold=self.threshold,
                triggered=triggered,
                confidence=confidence,
                action=action,
                severity=severity,
                message=message,
                evaluation_time_ns=start_time,
                data_age_ns=data.get('timestamp', start_time) - start_time,
                metadata={
                    'config': self.config,
                    'evaluation_count': self._evaluation_count
                }
            )
            
            # Update performance metrics
            evaluation_time = time.time_ns() - start_time
            self._evaluation_count += 1
            self._total_evaluation_time_ns += evaluation_time
            
            if triggered:
                self._trigger_count += 1
            
            # Check latency budget
            evaluation_time_us = evaluation_time / 1000.0
            if evaluation_time_us > self.latency_budget_us:
                self.logger.warning(
                    f"Sensor {self.sensor_name} exceeded latency budget: "
                    f"{evaluation_time_us:.2f}µs > {self.latency_budget_us:.2f}µs"
                )
            
            # Update calibration data
            self._calibration_data.append(value)
            if len(self._calibration_data) > 1000:
                self._calibration_data = self._calibration_data[-500:]  # Keep recent data
            
            self._last_result = result
            return result
            
        except Exception as e:
            self.logger.error(f"Sensor evaluation failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_disabled_result(self) -> SensorResult:
        """Create result for disabled sensor."""
        return SensorResult(
            sensor_id=self.sensor_id,
            sensor_name=self.sensor_name,
            failure_mode=self.failure_mode,
            priority=self.priority,
            value=0.0,
            threshold=self.threshold,
            triggered=False,
            confidence=0.0,
            action=SensorAction.NONE,
            severity=0.0,
            message=f"{self.sensor_name}: DISABLED"
        )
    
    def _create_error_result(self, error_message: str) -> SensorResult:
        """Create result for sensor error."""
        return SensorResult(
            sensor_id=self.sensor_id,
            sensor_name=self.sensor_name,
            failure_mode=self.failure_mode,
            priority=self.priority,
            value=float('nan'),
            threshold=self.threshold,
            triggered=True,  # Error is a trigger condition
            confidence=0.0,
            action=SensorAction.ALERT,
            severity=0.5,
            message=f"{self.sensor_name}: ERROR - {error_message}"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this sensor."""
        if self._evaluation_count == 0:
            return {
                'evaluation_count': 0,
                'avg_evaluation_time_us': 0.0,
                'trigger_rate': 0.0,
                'latency_budget_us': self.latency_budget_us
            }
        
        avg_time_ns = self._total_evaluation_time_ns / self._evaluation_count
        trigger_rate = self._trigger_count / self._evaluation_count
        
        return {
            'evaluation_count': self._evaluation_count,
            'avg_evaluation_time_us': avg_time_ns / 1000.0,
            'trigger_count': self._trigger_count,
            'trigger_rate': trigger_rate,
            'latency_budget_us': self.latency_budget_us,
            'budget_violations': sum(1 for _ in range(self._evaluation_count) 
                                   if avg_time_ns / 1000.0 > self.latency_budget_us),
            'last_value': self._last_result.value if self._last_result else None,
            'calibration_samples': len(self._calibration_data)
        }
    
    def enable(self):
        """Enable the sensor."""
        self._enabled = True
        self.logger.info(f"Sensor {self.sensor_name} enabled")
    
    def disable(self):
        """Disable the sensor."""
        self._enabled = False
        self.logger.info(f"Sensor {self.sensor_name} disabled")
    
    def is_enabled(self) -> bool:
        """Check if sensor is enabled."""
        return self._enabled
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        self._trigger_count = 0
        self._false_positive_count = 0
        self._calibration_data = []
        self.logger.info(f"Sensor {self.sensor_name} stats reset")