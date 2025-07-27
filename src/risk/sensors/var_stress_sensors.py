# src/risk/sensors/var_stress_sensors.py
"""
VaR and Stress Test Risk Sensors

Sensors for Value at Risk and nightly stress test monitoring:
1. VaRBreachSensor - Monitors VaR limit breaches
2. StressTestFailureSensor - Monitors stress test failures
3. TailRiskSensor - Monitors tail risk exposure
4. ConcentrationRiskSensor - Monitors position concentration
5. LeverageRiskSensor - Monitors portfolio leverage

All sensors support monitoring mode with comprehensive audit trails.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .base_sensor import (
    BaseSensor, SensorResult, SensorAction, SensorPriority, 
    FailureMode
)


class VaRBreachSensor(BaseSensor):
    """
    VaR Breach Detection Sensor
    
    Monitors portfolio VaR against configured limits and triggers
    appropriate actions when breaches occur.
    
    Supports monitoring mode for false-positive analysis.
    """
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        super().__init__(sensor_id, sensor_name, config)
        
        # VaR limits configuration
        self.var_95_limit = config.get('var_95_limit', 100000)  # $100k default
        self.var_99_limit = config.get('var_99_limit', 200000)  # $200k default
        self.var_999_limit = config.get('var_999_limit', 500000)  # $500k default
        
        # Monitoring mode configuration
        self.monitoring_mode = config.get('monitoring_mode', True)
        self.enforcement_enabled = config.get('enforcement_enabled', False)
        
        # Breach tracking for false positive analysis
        self.breach_history = []
        self.false_positive_threshold = config.params.get('false_positive_threshold', 1)  # per week
    
    def get_sensor_id(self) -> str:
        return "var_breach_sensor"
    
    def get_sensor_name(self) -> str:
        return "VaR Breach Detection"
    
    def get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def get_priority(self) -> SensorPriority:
        return SensorPriority.HIGH
    
    def _get_data_requirements(self) -> List[str]:
        return ['var_95', 'var_99', 'var_999', 'portfolio_value', 'timestamp']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute VaR breach severity score."""
        var_95 = data.get('var_95', 0)
        var_99 = data.get('var_99', 0)
        var_999 = data.get('var_999', 0)
        
        # Calculate breach ratios
        breach_95 = max(0, var_95 / self.var_95_limit - 1.0)
        breach_99 = max(0, var_99 / self.var_99_limit - 1.0)
        breach_999 = max(0, var_999 / self.var_999_limit - 1.0)
        
        # Return worst breach ratio
        return max(breach_95, breach_99, breach_999)
    
    def _evaluate_trigger_condition(self, sensor_value: float, data: Dict[str, Any]) -> bool:
        """Check if any VaR limit is breached."""
        return sensor_value > 0.0
    
    def _determine_action(self, sensor_value: float, data: Dict[str, Any]) -> SensorAction:
        """Determine action based on VaR breach severity and mode."""
        if not self._evaluate_trigger_condition(sensor_value, data):
            return SensorAction.NONE
        
        # In monitoring mode, only log breaches
        if self.monitoring_mode and not self.enforcement_enabled:
            return SensorAction.MONITOR
        
        # Enforcement mode - take action based on severity
        if sensor_value >= 2.0:  # 200% of limit
            return SensorAction.HALT
        elif sensor_value >= 1.0:  # 100% of limit
            return SensorAction.THROTTLE
        elif sensor_value >= 0.5:  # 50% of limit
            return SensorAction.WARN
        else:
            return SensorAction.MONITOR
    
    def _calculate_confidence(self, sensor_value: float, data: Dict[str, Any]) -> float:
        """Calculate confidence in VaR breach detection."""
        # Higher confidence for larger breaches
        if sensor_value >= 2.0:
            return 0.95
        elif sensor_value >= 1.0:
            return 0.90
        elif sensor_value >= 0.5:
            return 0.85
        else:
            return 0.80
    
    def _generate_message(self, sensor_value: float, data: Dict[str, Any]) -> str:
        """Generate descriptive message for VaR breach."""
        var_95 = data.get('var_95', 0)
        var_99 = data.get('var_99', 0)
        var_999 = data.get('var_999', 0)
        
        breach_details = []
        if var_95 > self.var_95_limit:
            breach_details.append(f"95% VaR: ${var_95:,.0f} > ${self.var_95_limit:,.0f}")
        if var_99 > self.var_99_limit:
            breach_details.append(f"99% VaR: ${var_99:,.0f} > ${self.var_99_limit:,.0f}")
        if var_999 > self.var_999_limit:
            breach_details.append(f"99.9% VaR: ${var_999:,.0f} > ${self.var_999_limit:,.0f}")
        
        mode_str = "MONITORING" if self.monitoring_mode else "ENFORCING"
        return f"VaR breach detected ({mode_str}): {'; '.join(breach_details)}"
    
    def _track_breach_for_false_positive_analysis(self, result: SensorResult) -> None:
        """Track breach for false positive analysis."""
        if result.triggered:
            breach_record = {
                'timestamp': datetime.now(),
                'sensor_value': result.value,
                'action': result.action,
                'message': result.message
            }
            self.breach_history.append(breach_record)
            
            # Clean old records (keep last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            self.breach_history = [
                record for record in self.breach_history 
                if record['timestamp'] > cutoff
            ]
    
    def get_false_positive_rate(self) -> float:
        """Calculate false positive rate for the last week."""
        if not self.breach_history:
            return 0.0
        
        # For now, assume all breaches are false positives in monitoring mode
        # In production, this would be validated against actual market outcomes
        return len(self.breach_history) / 7.0  # Per day average


class StressTestFailureSensor(BaseSensor):
    """
    Stress Test Failure Detection Sensor
    
    Monitors nightly stress test results and triggers alerts
    when portfolio fails stress scenarios.
    """
    
    def __init__(self, sensor_id: str, sensor_name: str, config: Dict[str, Any]):
        super().__init__(sensor_id, sensor_name, config)
        
        # Stress test limits
        self.max_stress_loss = config.get('max_stress_loss', 1000000)  # $1M default
        self.max_scenario_failures = config.get('max_scenario_failures', 3)
        
        # Monitoring configuration
        self.monitoring_mode = config.get('monitoring_mode', True)
        self.enforcement_enabled = config.get('enforcement_enabled', False)
    
    def get_sensor_id(self) -> str:
        return "stress_test_failure_sensor"
    
    def get_sensor_name(self) -> str:
        return "Stress Test Failure Detection"
    
    def get_failure_mode(self) -> FailureMode:
        return FailureMode.TAIL_REGIME_SHIFT
    
    def get_priority(self) -> SensorPriority:
        return SensorPriority.MEDIUM  # Nightly batch process
    
    def _get_data_requirements(self) -> List[str]:
        return ['stress_worst_case', 'stress_scenarios_count', 'failed_scenarios', 'stress_var_99']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute stress test failure severity."""
        worst_case_loss = data.get('stress_worst_case', 0)
        failed_scenarios = data.get('failed_scenarios', 0)
        total_scenarios = data.get('stress_scenarios_count', 1)
        
        # Calculate failure metrics
        loss_ratio = worst_case_loss / self.max_stress_loss
        failure_ratio = failed_scenarios / max(total_scenarios, 1)
        
        # Combined severity score
        return max(loss_ratio, failure_ratio)
    
    def _evaluate_trigger_condition(self, sensor_value: float, data: Dict[str, Any]) -> bool:
        """Check if stress test failures exceed thresholds."""
        return sensor_value > 0.5  # 50% threshold
    
    def _determine_action(self, sensor_value: float, data: Dict[str, Any]) -> SensorAction:
        """Determine action based on stress test results."""
        if not self._evaluate_trigger_condition(sensor_value, data):
            return SensorAction.NONE
        
        # Monitoring mode
        if self.monitoring_mode and not self.enforcement_enabled:
            return SensorAction.MONITOR
        
        # Enforcement mode
        if sensor_value >= 2.0:
            return SensorAction.HALT
        elif sensor_value >= 1.0:
            return SensorAction.REDUCE_POSITION
        else:
            return SensorAction.WARN
    
    def _calculate_confidence(self, sensor_value: float, data: Dict[str, Any]) -> float:
        """Calculate confidence in stress test failure detection."""
        scenarios_count = data.get('stress_scenarios_count', 0)
        
        # Higher confidence with more scenarios
        base_confidence = min(0.95, 0.5 + (scenarios_count / 100))
        
        # Adjust for severity
        severity_adjustment = min(0.1, sensor_value * 0.05)
        
        return min(0.95, base_confidence + severity_adjustment)
    
    def _generate_message(self, sensor_value: float, data: Dict[str, Any]) -> str:
        """Generate message for stress test failure."""
        worst_case = data.get('stress_worst_case', 0)
        failed_scenarios = data.get('failed_scenarios', 0)
        total_scenarios = data.get('stress_scenarios_count', 0)
        
        mode_str = "MONITORING" if self.monitoring_mode else "ENFORCING"
        return (f"Stress test failure ({mode_str}): "
                f"Worst case loss ${worst_case:,.0f}, "
                f"{failed_scenarios}/{total_scenarios} scenarios failed")


class TailRiskSensor(BaseSensor):
    """
    Tail Risk Monitoring Sensor
    
    Monitors extreme tail risk exposure using Expected Shortfall
    and other tail risk metrics.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        
        self.es_99_limit = config.params.get('es_99_limit', 250000)  # $250k default
        self.tail_ratio_threshold = config.params.get('tail_ratio_threshold', 1.5)  # ES/VaR ratio
        
        self.monitoring_mode = config.params.get('monitoring_mode', True)
        self.enforcement_enabled = config.params.get('enforcement_enabled', False)
    
    def get_sensor_id(self) -> str:
        return "tail_risk_sensor"
    
    def get_sensor_name(self) -> str:
        return "Tail Risk Monitoring"
    
    def get_failure_mode(self) -> FailureMode:
        return FailureMode.TAIL_REGIME_SHIFT
    
    def get_priority(self) -> SensorPriority:
        return SensorPriority.HIGH
    
    def _get_data_requirements(self) -> List[str]:
        return ['expected_shortfall_99', 'var_99', 'tail_ratio']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute tail risk severity."""
        es_99 = data.get('expected_shortfall_99', 0)
        tail_ratio = data.get('tail_ratio', 1.0)
        
        # Calculate breach ratios
        es_breach = max(0, es_99 / self.es_99_limit - 1.0)
        ratio_breach = max(0, tail_ratio / self.tail_ratio_threshold - 1.0)
        
        return max(es_breach, ratio_breach)
    
    def _evaluate_trigger_condition(self, sensor_value: float, data: Dict[str, Any]) -> bool:
        return sensor_value > 0.0
    
    def _determine_action(self, sensor_value: float, data: Dict[str, Any]) -> SensorAction:
        if not self._evaluate_trigger_condition(sensor_value, data):
            return SensorAction.NONE
        
        if self.monitoring_mode and not self.enforcement_enabled:
            return SensorAction.MONITOR
        
        if sensor_value >= 1.0:
            return SensorAction.THROTTLE
        else:
            return SensorAction.WARN
    
    def _calculate_confidence(self, sensor_value: float, data: Dict[str, Any]) -> float:
        return 0.85
    
    def _generate_message(self, sensor_value: float, data: Dict[str, Any]) -> str:
        es_99 = data.get('expected_shortfall_99', 0)
        tail_ratio = data.get('tail_ratio', 1.0)
        
        mode_str = "MONITORING" if self.monitoring_mode else "ENFORCING"
        return (f"Tail risk alert ({mode_str}): "
                f"ES99 ${es_99:,.0f}, tail ratio {tail_ratio:.2f}")


class ConcentrationRiskSensor(BaseSensor):
    """
    Position Concentration Risk Sensor
    
    Monitors portfolio concentration risk using Herfindahl index
    and single-position limits.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        
        self.max_single_position_pct = config.params.get('max_single_position_pct', 0.20)  # 20%
        self.max_herfindahl_index = config.params.get('max_herfindahl_index', 0.25)
        
        self.monitoring_mode = config.params.get('monitoring_mode', True)
        self.enforcement_enabled = config.params.get('enforcement_enabled', False)
    
    def get_sensor_id(self) -> str:
        return "concentration_risk_sensor"
    
    def get_sensor_name(self) -> str:
        return "Concentration Risk Monitoring"
    
    def get_failure_mode(self) -> FailureMode:
        return FailureMode.PATH_FRAGILITY
    
    def get_priority(self) -> SensorPriority:
        return SensorPriority.MEDIUM
    
    def _get_data_requirements(self) -> List[str]:
        return ['position_weights', 'herfindahl_index', 'max_position_weight']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute concentration risk severity."""
        max_position = data.get('max_position_weight', 0)
        herfindahl = data.get('herfindahl_index', 0)
        
        position_breach = max(0, max_position / self.max_single_position_pct - 1.0)
        herfindahl_breach = max(0, herfindahl / self.max_herfindahl_index - 1.0)
        
        return max(position_breach, herfindahl_breach)
    
    def _evaluate_trigger_condition(self, sensor_value: float, data: Dict[str, Any]) -> bool:
        return sensor_value > 0.0
    
    def _determine_action(self, sensor_value: float, data: Dict[str, Any]) -> SensorAction:
        if not self._evaluate_trigger_condition(sensor_value, data):
            return SensorAction.NONE
        
        if self.monitoring_mode and not self.enforcement_enabled:
            return SensorAction.MONITOR
        
        if sensor_value >= 1.0:
            return SensorAction.REDUCE_POSITION
        else:
            return SensorAction.WARN
    
    def _calculate_confidence(self, sensor_value: float, data: Dict[str, Any]) -> float:
        return 0.90
    
    def _generate_message(self, sensor_value: float, data: Dict[str, Any]) -> str:
        max_position = data.get('max_position_weight', 0)
        herfindahl = data.get('herfindahl_index', 0)
        
        mode_str = "MONITORING" if self.monitoring_mode else "ENFORCING"
        return (f"Concentration risk ({mode_str}): "
                f"Max position {max_position:.1%}, Herfindahl {herfindahl:.3f}")


class LeverageRiskSensor(BaseSensor):
    """
    Portfolio Leverage Risk Sensor
    
    Monitors portfolio leverage and margin utilization.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        
        self.max_gross_leverage = config.params.get('max_gross_leverage', 3.0)  # 3x
        self.max_net_leverage = config.params.get('max_net_leverage', 2.0)  # 2x
        self.max_margin_utilization = config.params.get('max_margin_utilization', 0.80)  # 80%
        
        self.monitoring_mode = config.params.get('monitoring_mode', True)
        self.enforcement_enabled = config.params.get('enforcement_enabled', False)
    
    def get_sensor_id(self) -> str:
        return "leverage_risk_sensor"
    
    def get_sensor_name(self) -> str:
        return "Leverage Risk Monitoring"
    
    def get_failure_mode(self) -> FailureMode:
        return FailureMode.FUNDING_MARGIN
    
    def get_priority(self) -> SensorPriority:
        return SensorPriority.HIGH
    
    def _get_data_requirements(self) -> List[str]:
        return ['gross_leverage', 'net_leverage', 'margin_utilization']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute leverage risk severity."""
        gross_leverage = data.get('gross_leverage', 1.0)
        net_leverage = data.get('net_leverage', 1.0)
        margin_util = data.get('margin_utilization', 0.0)
        
        gross_breach = max(0, gross_leverage / self.max_gross_leverage - 1.0)
        net_breach = max(0, net_leverage / self.max_net_leverage - 1.0)
        margin_breach = max(0, margin_util / self.max_margin_utilization - 1.0)
        
        return max(gross_breach, net_breach, margin_breach)
    
    def _evaluate_trigger_condition(self, sensor_value: float, data: Dict[str, Any]) -> bool:
        return sensor_value > 0.0
    
    def _determine_action(self, sensor_value: float, data: Dict[str, Any]) -> SensorAction:
        if not self._evaluate_trigger_condition(sensor_value, data):
            return SensorAction.NONE
        
        if self.monitoring_mode and not self.enforcement_enabled:
            return SensorAction.MONITOR
        
        if sensor_value >= 1.0:
            return SensorAction.REDUCE_POSITION
        elif sensor_value >= 0.5:
            return SensorAction.THROTTLE
        else:
            return SensorAction.WARN
    
    def _calculate_confidence(self, sensor_value: float, data: Dict[str, Any]) -> float:
        return 0.95
    
    def _generate_message(self, sensor_value: float, data: Dict[str, Any]) -> str:
        gross_lev = data.get('gross_leverage', 1.0)
        net_lev = data.get('net_leverage', 1.0)
        margin_util = data.get('margin_utilization', 0.0)
        
        mode_str = "MONITORING" if self.monitoring_mode else "ENFORCING"
        return (f"Leverage risk ({mode_str}): "
                f"Gross {gross_lev:.1f}x, Net {net_lev:.1f}x, "
                f"Margin {margin_util:.1%}")


__all__ = [
    'VaRBreachSensor',
    'StressTestFailureSensor', 
    'TailRiskSensor',
    'ConcentrationRiskSensor',
    'LeverageRiskSensor'
]