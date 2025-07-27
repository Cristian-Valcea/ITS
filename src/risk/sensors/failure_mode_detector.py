# src/risk/sensors/failure_mode_detector.py
"""
Failure Mode Detector - Pattern recognition for complex failure modes.

Analyzes sensor patterns to detect compound failure modes that might not
be caught by individual sensors alone.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .base_sensor import SensorResult, FailureMode, SensorAction


class CompoundFailureMode(Enum):
    """Complex failure modes detected by pattern analysis."""
    DEATH_SPIRAL = "death_spiral"  # Accelerating losses + high leverage
    LIQUIDITY_CRUNCH = "liquidity_crunch"  # Low liquidity + large positions
    REGIME_BREAKDOWN = "regime_breakdown"  # Multiple regime indicators
    OPERATIONAL_FAILURE = "operational_failure"  # System degradation
    MARGIN_SQUEEZE = "margin_squeeze"  # Funding + margin stress
    COUNTERPARTY_CONTAGION = "counterparty_contagion"  # Correlated CP risk


@dataclass
class FailureModePattern:
    """Pattern definition for compound failure mode detection."""
    name: str
    required_sensors: List[str]
    trigger_conditions: Dict[str, Any]
    confidence_threshold: float
    action: SensorAction


class FailureModeDetector:
    """
    Detects compound failure modes by analyzing sensor patterns.
    
    Uses pattern matching and machine learning techniques to identify
    complex failure scenarios that individual sensors might miss.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize failure mode patterns
        self.patterns = self._initialize_patterns()
        
        # Detection history for pattern analysis
        self.detection_history = []
        self.max_history = int(self.config.get('max_history', 1000))
        
        self.logger.info(f"FailureModeDetector initialized with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self) -> Dict[CompoundFailureMode, FailureModePattern]:
        """Initialize compound failure mode patterns."""
        patterns = {}
        
        # Death Spiral Pattern
        patterns[CompoundFailureMode.DEATH_SPIRAL] = FailureModePattern(
            name="Death Spiral",
            required_sensors=["drawdown_velocity", "ulcer_index", "drawdown_adjusted_leverage"],
            trigger_conditions={
                "drawdown_velocity": {"min_value": 0.01, "triggered": True},
                "ulcer_index": {"min_value": 0.05, "triggered": True},
                "drawdown_adjusted_leverage": {"min_value": 3.0, "triggered": True}
            },
            confidence_threshold=0.8,
            action=SensorAction.KILL_SWITCH
        )
        
        # Liquidity Crunch Pattern
        patterns[CompoundFailureMode.LIQUIDITY_CRUNCH] = FailureModePattern(
            name="Liquidity Crunch",
            required_sensors=["adv_participation", "depth_shock", "kyle_lambda"],
            trigger_conditions={
                "adv_participation": {"min_value": 0.2, "triggered": True},
                "depth_shock": {"min_value": 0.05, "triggered": True},
                "kyle_lambda": {"min_value": 0.001, "triggered": True}
            },
            confidence_threshold=0.7,
            action=SensorAction.THROTTLE
        )
        
        # Regime Breakdown Pattern
        patterns[CompoundFailureMode.REGIME_BREAKDOWN] = FailureModePattern(
            name="Regime Breakdown",
            required_sensors=["regime_switch", "vol_of_vol", "expected_shortfall"],
            trigger_conditions={
                "regime_switch": {"min_value": 2.0, "triggered": True},
                "vol_of_vol": {"min_value": 0.5, "triggered": True},
                "expected_shortfall": {"min_value": 0.03, "triggered": True}
            },
            confidence_threshold=0.75,
            action=SensorAction.HEDGE
        )
        
        # Operational Failure Pattern
        patterns[CompoundFailureMode.OPERATIONAL_FAILURE] = FailureModePattern(
            name="Operational Failure",
            required_sensors=["latency_drift", "feed_staleness", "exception_rate"],
            trigger_conditions={
                "latency_drift": {"min_value": 0.5, "triggered": True},
                "feed_staleness": {"min_value": 5.0, "triggered": True},
                "exception_rate": {"min_value": 10.0, "triggered": True}
            },
            confidence_threshold=0.6,
            action=SensorAction.KILL_SWITCH
        )
        
        # Margin Squeeze Pattern
        patterns[CompoundFailureMode.MARGIN_SQUEEZE] = FailureModePattern(
            name="Margin Squeeze",
            required_sensors=["time_to_margin_exhaustion", "liquidity_at_risk", "haircut_sensitivity"],
            trigger_conditions={
                "time_to_margin_exhaustion": {"max_value": 5.0, "triggered": True},
                "liquidity_at_risk": {"min_value": 100000, "triggered": True},
                "haircut_sensitivity": {"min_value": 50000, "triggered": True}
            },
            confidence_threshold=0.8,
            action=SensorAction.THROTTLE
        )
        
        return patterns
    
    def detect_failure_modes(self, sensor_results: List[SensorResult]) -> List[Dict[str, Any]]:
        """
        Detect compound failure modes from sensor results.
        
        Args:
            sensor_results: List of sensor evaluation results
            
        Returns:
            List of detected compound failure modes
        """
        detected_modes = []
        
        # Create sensor lookup for pattern matching
        sensor_lookup = {result.sensor_id: result for result in sensor_results}
        
        # Check each pattern
        for failure_mode, pattern in self.patterns.items():
            detection = self._check_pattern(pattern, sensor_lookup)
            
            if detection:
                detected_modes.append({
                    'failure_mode': failure_mode,
                    'pattern_name': pattern.name,
                    'confidence': detection['confidence'],
                    'action': pattern.action,
                    'triggered_sensors': detection['triggered_sensors'],
                    'pattern_score': detection['pattern_score'],
                    'timestamp': detection['timestamp']
                })
                
                # Store in history
                self.detection_history.append(detection)
                if len(self.detection_history) > self.max_history:
                    self.detection_history = self.detection_history[-self.max_history:]
        
        return detected_modes
    
    def _check_pattern(self, pattern: FailureModePattern, 
                      sensor_lookup: Dict[str, SensorResult]) -> Optional[Dict[str, Any]]:
        """Check if a specific pattern is triggered."""
        import time
        
        triggered_sensors = []
        pattern_scores = []
        
        # Check each required sensor
        for sensor_id in pattern.required_sensors:
            if sensor_id not in sensor_lookup:
                # Missing required sensor
                return None
            
            result = sensor_lookup[sensor_id]
            conditions = pattern.trigger_conditions.get(sensor_id, {})
            
            # Check trigger condition
            if conditions.get('triggered', False) and not result.triggered:
                # Required trigger not met
                return None
            
            # Check value conditions
            min_value = conditions.get('min_value')
            max_value = conditions.get('max_value')
            
            if min_value is not None and result.value < min_value:
                return None
            
            if max_value is not None and result.value > max_value:
                return None
            
            # Sensor contributes to pattern
            triggered_sensors.append(sensor_id)
            pattern_scores.append(result.confidence * result.severity)
        
        # Calculate overall pattern confidence
        if not pattern_scores:
            return None
        
        pattern_score = np.mean(pattern_scores)
        
        # Apply pattern-specific confidence adjustments
        confidence = self._calculate_pattern_confidence(pattern, sensor_lookup, pattern_score)
        
        if confidence < pattern.confidence_threshold:
            return None
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'triggered_sensors': triggered_sensors,
            'pattern_score': pattern_score,
            'timestamp': time.time()
        }
    
    def _calculate_pattern_confidence(self, pattern: FailureModePattern,
                                    sensor_lookup: Dict[str, SensorResult],
                                    base_score: float) -> float:
        """Calculate confidence in pattern detection."""
        confidence = base_score
        
        # Bonus for multiple high-confidence sensors
        high_confidence_sensors = len([
            s for s_id in pattern.required_sensors 
            if s_id in sensor_lookup and sensor_lookup[s_id].confidence > 0.8
        ])
        
        confidence += high_confidence_sensors * 0.05
        
        # Bonus for recent pattern history
        recent_detections = [
            d for d in self.detection_history[-10:] 
            if d['pattern'].name == pattern.name
        ]
        
        if len(recent_detections) >= 2:
            confidence += 0.1  # Pattern persistence bonus
        
        # Penalty for conflicting patterns
        conflicting_patterns = self._check_pattern_conflicts(pattern, sensor_lookup)
        confidence -= conflicting_patterns * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _check_pattern_conflicts(self, pattern: FailureModePattern,
                               sensor_lookup: Dict[str, SensorResult]) -> int:
        """Check for conflicting pattern signals."""
        conflicts = 0
        
        # Simple conflict detection - can be enhanced
        if pattern.name == "Death Spiral":
            # Conflicts with positive momentum indicators
            if "positive_momentum" in sensor_lookup:
                conflicts += 1
        
        elif pattern.name == "Liquidity Crunch":
            # Conflicts with high liquidity indicators
            if "high_liquidity" in sensor_lookup:
                conflicts += 1
        
        return conflicts
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics on pattern detection history."""
        if not self.detection_history:
            return {'total_detections': 0}
        
        # Count detections by pattern
        pattern_counts = {}
        for detection in self.detection_history:
            pattern_name = detection['pattern'].name
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        # Calculate average confidence by pattern
        pattern_confidence = {}
        for detection in self.detection_history:
            pattern_name = detection['pattern'].name
            if pattern_name not in pattern_confidence:
                pattern_confidence[pattern_name] = []
            pattern_confidence[pattern_name].append(detection['confidence'])
        
        avg_confidence = {
            pattern: np.mean(confidences) 
            for pattern, confidences in pattern_confidence.items()
        }
        
        return {
            'total_detections': len(self.detection_history),
            'pattern_counts': pattern_counts,
            'average_confidence': avg_confidence,
            'recent_detections': len([d for d in self.detection_history[-100:]])
        }
    
    def add_custom_pattern(self, failure_mode: CompoundFailureMode, 
                          pattern: FailureModePattern):
        """Add a custom failure mode pattern."""
        self.patterns[failure_mode] = pattern
        self.logger.info(f"Added custom pattern: {pattern.name}")
    
    def remove_pattern(self, failure_mode: CompoundFailureMode):
        """Remove a failure mode pattern."""
        if failure_mode in self.patterns:
            pattern_name = self.patterns[failure_mode].name
            del self.patterns[failure_mode]
            self.logger.info(f"Removed pattern: {pattern_name}")
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failure mode detections."""
        return self.detection_history[-limit:]