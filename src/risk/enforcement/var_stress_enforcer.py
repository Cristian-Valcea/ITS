# src/risk/enforcement/var_stress_enforcer.py
"""
VaR and Stress Test Enforcement System

Enforces VaR limits and stress test requirements with:
- Monitoring mode for false positive analysis
- Automatic enforcement once false positives < 1 per week
- Comprehensive audit trail
- Prometheus metrics integration
- Configurable enforcement actions

Supports gradual rollout from monitoring to enforcement.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading

try:
    from ..audit.audit_logger import AuditLogger, AuditEventType
    from ..metrics.prometheus_metrics import RiskMetricsCollector
    from ..calculators.var_calculator import VaRCalculator
    from ..calculators.stress_test_calculator import StressTestCalculator
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from audit.audit_logger import AuditLogger, AuditEventType
    from metrics.prometheus_metrics import RiskMetricsCollector
    from calculators.var_calculator import VaRCalculator
    from calculators.stress_test_calculator import StressTestCalculator


class EnforcementMode(Enum):
    """Enforcement modes."""
    MONITORING = "monitoring"      # Log only, no enforcement
    GRADUAL = "gradual"           # Gradual enforcement based on false positive rate
    FULL = "full"                 # Full enforcement enabled


class EnforcementAction(Enum):
    """Enforcement actions."""
    NONE = "none"
    WARN = "warn"
    THROTTLE = "throttle"
    REDUCE_POSITION = "reduce_position"
    BLOCK_NEW_TRADES = "block_new_trades"
    HALT_TRADING = "halt_trading"
    LIQUIDATE = "liquidate"


@dataclass
class EnforcementResult:
    """Result of enforcement evaluation."""
    action: EnforcementAction
    reason: str
    severity: str
    monitoring_mode: bool
    enforcement_enabled: bool
    false_positive_rate: float
    metadata: Dict[str, Any]


class VaRStressEnforcer:
    """
    VaR and Stress Test Enforcement System
    
    Manages enforcement of VaR limits and stress test requirements with
    sophisticated monitoring mode and false positive analysis.
    """
    
    def __init__(self, config: Dict[str, Any],
                 audit_logger: Optional[AuditLogger] = None,
                 metrics_collector: Optional[RiskMetricsCollector] = None):
        """
        Initialize VaR/Stress enforcer.
        
        Args:
            config: Configuration dictionary
            audit_logger: Audit logger instance
            metrics_collector: Metrics collector instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # External dependencies
        self.audit_logger = audit_logger
        self.metrics_collector = metrics_collector
        
        # Enforcement configuration
        self.enforcement_mode = EnforcementMode(config.get('enforcement_mode', 'monitoring'))
        self.false_positive_threshold = config.get('false_positive_threshold_per_week', 1.0)
        self.monitoring_period_days = config.get('monitoring_period_days', 30)
        
        # VaR limits
        self.var_limits = config.get('var_limits', {
            'var_95_limit': 100000,   # $100k
            'var_99_limit': 200000,   # $200k
            'var_999_limit': 500000   # $500k
        })
        
        # Stress test limits
        self.stress_limits = config.get('stress_limits', {
            'max_stress_loss': 1000000,      # $1M
            'max_scenario_failures': 3,
            'max_tail_ratio': 1.5
        })
        
        # Initialize calculators
        self.var_calculator = VaRCalculator(
            config.get('var_calculator', {}),
            self.logger
        )
        
        self.stress_calculator = StressTestCalculator(
            config.get('stress_calculator', {}),
            self.logger
        )
        
        # Initialize sensor configurations (simplified)
        self.sensors = self._initialize_sensor_configs()
        
        # False positive tracking
        self.false_positive_history = {}  # sensor_id -> list of events
        self.enforcement_decisions = []   # History of enforcement decisions
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.evaluations_count = 0
        self.enforcement_actions_count = 0
        self.start_time = time.time()
        
        self.logger.info(f"VaRStressEnforcer initialized in {self.enforcement_mode.value} mode")
    
    def _initialize_sensor_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize sensor configurations."""
        sensor_configs = self.config.get('sensors', {})
        
        # VaR breach sensor
        var_config = sensor_configs.get('var_breach', {})
        var_config.update({
            'monitoring_mode': self.enforcement_mode != EnforcementMode.FULL,
            'enforcement_enabled': self.enforcement_mode == EnforcementMode.FULL,
            **self.var_limits
        })
        
        # Stress test sensor
        stress_config = sensor_configs.get('stress_test', {})
        stress_config.update({
            'monitoring_mode': self.enforcement_mode != EnforcementMode.FULL,
            'enforcement_enabled': self.enforcement_mode == EnforcementMode.FULL,
            **self.stress_limits
        })
        
        return {
            'var_breach': var_config,
            'stress_test': stress_config,
            'tail_risk': sensor_configs.get('tail_risk', {}),
            'concentration': sensor_configs.get('concentration', {}),
            'leverage': sensor_configs.get('leverage', {})
        }
    
    def evaluate_var_enforcement(self, portfolio_data: Dict[str, Any]) -> EnforcementResult:
        """
        Evaluate VaR enforcement requirements.
        
        Args:
            portfolio_data: Portfolio data for VaR calculation
        
        Returns:
            EnforcementResult with enforcement decision
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self.evaluations_count += 1
        
        try:
            # Calculate VaR
            var_result = self.var_calculator.calculate(portfolio_data)
            
            if not var_result.is_valid:
                return EnforcementResult(
                    action=EnforcementAction.NONE,
                    reason=f"VaR calculation failed: {var_result.error_message}",
                    severity="error",
                    monitoring_mode=True,
                    enforcement_enabled=False,
                    false_positive_rate=0.0,
                    metadata={'error': var_result.error_message}
                )
            
            # Check VaR breaches
            breach_info = self._check_var_breaches(var_result.values)
            
            # Determine enforcement action
            enforcement_result = self._determine_enforcement_action(
                'var_breach',
                breach_info,
                var_result.values
            )
            
            # Log to audit trail
            if self.audit_logger:
                self.audit_logger.log_var_calculation(
                    'parametric',
                    var_result.values,
                    portfolio_data
                )
                
                if enforcement_result.action != EnforcementAction.NONE:
                    self.audit_logger.log_enforcement_action(
                        enforcement_result.action.value,
                        enforcement_result.reason,
                        {
                            'monitoring_mode': enforcement_result.monitoring_mode,
                            'enforcement_enabled': enforcement_result.enforcement_enabled,
                            'var_values': var_result.values,
                            'var_limits': self.var_limits
                        }
                    )
            
            # Record metrics
            if self.metrics_collector:
                duration = time.perf_counter() - start_time
                self.metrics_collector.record_var_calculation(
                    'parametric',
                    var_result.values,
                    self.var_limits
                )
                self.metrics_collector.record_performance_metric(
                    'var_enforcer',
                    'evaluate_var',
                    duration
                )
                
                if enforcement_result.action != EnforcementAction.NONE:
                    self.metrics_collector.record_enforcement_action(
                        enforcement_result.action.value,
                        enforcement_result.reason,
                        enforcement_result.monitoring_mode
                    )
            
            return enforcement_result
            
        except Exception as e:
            self.logger.error(f"Error in VaR enforcement evaluation: {e}")
            return EnforcementResult(
                action=EnforcementAction.NONE,
                reason=f"Evaluation error: {str(e)}",
                severity="error",
                monitoring_mode=True,
                enforcement_enabled=False,
                false_positive_rate=0.0,
                metadata={'error': str(e)}
            )
    
    def evaluate_stress_test_enforcement(self, portfolio_data: Dict[str, Any]) -> EnforcementResult:
        """
        Evaluate stress test enforcement requirements.
        
        Args:
            portfolio_data: Portfolio data for stress testing
        
        Returns:
            EnforcementResult with enforcement decision
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self.evaluations_count += 1
        
        try:
            # Run stress tests
            stress_result = self.stress_calculator.calculate(portfolio_data)
            
            if not stress_result.is_valid:
                return EnforcementResult(
                    action=EnforcementAction.NONE,
                    reason=f"Stress test failed: {stress_result.error_message}",
                    severity="error",
                    monitoring_mode=True,
                    enforcement_enabled=False,
                    false_positive_rate=0.0,
                    metadata={'error': stress_result.error_message}
                )
            
            # Check stress test failures
            failure_info = self._check_stress_test_failures(stress_result.values)
            
            # Determine enforcement action
            enforcement_result = self._determine_enforcement_action(
                'stress_test',
                failure_info,
                stress_result.values
            )
            
            # Log to audit trail
            if self.audit_logger:
                self.audit_logger.log_stress_test(
                    'comprehensive',
                    stress_result.values,
                    stress_result.metadata.get('scenarios_run', [])
                )
                
                if enforcement_result.action != EnforcementAction.NONE:
                    self.audit_logger.log_enforcement_action(
                        enforcement_result.action.value,
                        enforcement_result.reason,
                        {
                            'monitoring_mode': enforcement_result.monitoring_mode,
                            'enforcement_enabled': enforcement_result.enforcement_enabled,
                            'stress_results': stress_result.values,
                            'stress_limits': self.stress_limits
                        }
                    )
            
            # Record metrics
            if self.metrics_collector:
                duration = time.perf_counter() - start_time
                self.metrics_collector.record_stress_test(
                    'comprehensive',
                    stress_result.values,
                    duration
                )
                self.metrics_collector.record_performance_metric(
                    'stress_enforcer',
                    'evaluate_stress',
                    duration
                )
                
                if enforcement_result.action != EnforcementAction.NONE:
                    self.metrics_collector.record_enforcement_action(
                        enforcement_result.action.value,
                        enforcement_result.reason,
                        enforcement_result.monitoring_mode
                    )
            
            return enforcement_result
            
        except Exception as e:
            self.logger.error(f"Error in stress test enforcement evaluation: {e}")
            return EnforcementResult(
                action=EnforcementAction.NONE,
                reason=f"Evaluation error: {str(e)}",
                severity="error",
                monitoring_mode=True,
                enforcement_enabled=False,
                false_positive_rate=0.0,
                metadata={'error': str(e)}
            )
    
    def _check_var_breaches(self, var_values: Dict[str, Any]) -> Dict[str, Any]:
        """Check for VaR limit breaches."""
        breaches = {}
        
        for confidence_level in ['95', '99', '999']:
            var_key = f'var_{confidence_level}'
            limit_key = f'var_{confidence_level}_limit'
            
            if var_key in var_values and limit_key in self.var_limits:
                var_value = var_values[var_key]
                limit = self.var_limits[limit_key]
                
                if var_value > limit:
                    breach_ratio = var_value / limit
                    breaches[confidence_level] = {
                        'value': var_value,
                        'limit': limit,
                        'breach_ratio': breach_ratio,
                        'severity': self._assess_var_breach_severity(breach_ratio)
                    }
        
        return breaches
    
    def _check_stress_test_failures(self, stress_values: Dict[str, Any]) -> Dict[str, Any]:
        """Check for stress test failures."""
        failures = {}
        
        # Check worst case loss
        worst_case = stress_values.get('stress_worst_case', 0)
        max_loss = self.stress_limits.get('max_stress_loss', float('inf'))
        
        if worst_case > max_loss:
            failures['worst_case'] = {
                'value': worst_case,
                'limit': max_loss,
                'breach_ratio': worst_case / max_loss,
                'severity': 'high'
            }
        
        # Check scenario failures
        failed_scenarios = stress_values.get('failed_scenarios', 0)
        max_failures = self.stress_limits.get('max_scenario_failures', float('inf'))
        
        if failed_scenarios > max_failures:
            failures['scenario_failures'] = {
                'value': failed_scenarios,
                'limit': max_failures,
                'breach_ratio': failed_scenarios / max_failures,
                'severity': 'medium'
            }
        
        return failures
    
    def _determine_enforcement_action(self, sensor_type: str, 
                                    breach_info: Dict[str, Any],
                                    metric_values: Dict[str, Any]) -> EnforcementResult:
        """Determine appropriate enforcement action."""
        
        # Get false positive rate for this sensor
        fp_rate = self._get_false_positive_rate(sensor_type)
        
        # Determine if enforcement should be enabled
        monitoring_mode = self.enforcement_mode == EnforcementMode.MONITORING
        enforcement_enabled = (
            self.enforcement_mode == EnforcementMode.FULL or
            (self.enforcement_mode == EnforcementMode.GRADUAL and fp_rate < self.false_positive_threshold)
        )
        
        # No breaches detected
        if not breach_info:
            return EnforcementResult(
                action=EnforcementAction.NONE,
                reason="No limit breaches detected",
                severity="info",
                monitoring_mode=monitoring_mode,
                enforcement_enabled=enforcement_enabled,
                false_positive_rate=fp_rate,
                metadata={'sensor_type': sensor_type}
            )
        
        # Determine action based on severity and mode
        max_severity = max(
            breach['severity'] for breach in breach_info.values()
            if isinstance(breach, dict) and 'severity' in breach
        )
        
        if monitoring_mode and not enforcement_enabled:
            action = EnforcementAction.WARN
            reason = f"MONITORING: {sensor_type} breach detected (FP rate: {fp_rate:.2f}/week)"
        else:
            # Enforcement mode - take action based on severity
            if max_severity == 'critical':
                action = EnforcementAction.HALT_TRADING
                reason = f"Critical {sensor_type} breach - trading halted"
            elif max_severity == 'high':
                action = EnforcementAction.REDUCE_POSITION
                reason = f"High {sensor_type} breach - reducing positions"
            elif max_severity == 'medium':
                action = EnforcementAction.THROTTLE
                reason = f"Medium {sensor_type} breach - throttling trades"
            else:
                action = EnforcementAction.WARN
                reason = f"Low {sensor_type} breach - warning issued"
        
        # Track enforcement decision
        decision_record = {
            'timestamp': datetime.now(),
            'sensor_type': sensor_type,
            'action': action,
            'reason': reason,
            'monitoring_mode': monitoring_mode,
            'enforcement_enabled': enforcement_enabled,
            'false_positive_rate': fp_rate,
            'breach_info': breach_info
        }
        
        with self._lock:
            self.enforcement_decisions.append(decision_record)
            if action != EnforcementAction.NONE:
                self.enforcement_actions_count += 1
        
        return EnforcementResult(
            action=action,
            reason=reason,
            severity=max_severity,
            monitoring_mode=monitoring_mode,
            enforcement_enabled=enforcement_enabled,
            false_positive_rate=fp_rate,
            metadata={
                'sensor_type': sensor_type,
                'breach_info': breach_info,
                'metric_values': metric_values
            }
        )
    
    def _assess_var_breach_severity(self, breach_ratio: float) -> str:
        """Assess VaR breach severity."""
        if breach_ratio >= 3.0:
            return 'critical'
        elif breach_ratio >= 2.0:
            return 'high'
        elif breach_ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _get_false_positive_rate(self, sensor_type: str) -> float:
        """Get false positive rate for sensor type."""
        if sensor_type not in self.false_positive_history:
            return 0.0
        
        # Calculate rate over last week
        cutoff = datetime.now() - timedelta(days=7)
        recent_fps = [
            fp for fp in self.false_positive_history[sensor_type]
            if fp['timestamp'] > cutoff
        ]
        
        return len(recent_fps) / 7.0  # Per day average
    
    def record_false_positive(self, sensor_type: str, event_id: str,
                            reason: str, analysis: Dict[str, Any]) -> None:
        """Record a false positive for analysis."""
        fp_record = {
            'timestamp': datetime.now(),
            'event_id': event_id,
            'reason': reason,
            'analysis': analysis
        }
        
        with self._lock:
            if sensor_type not in self.false_positive_history:
                self.false_positive_history[sensor_type] = []
            
            self.false_positive_history[sensor_type].append(fp_record)
            
            # Clean old records (keep last 30 days)
            cutoff = datetime.now() - timedelta(days=self.monitoring_period_days)
            self.false_positive_history[sensor_type] = [
                fp for fp in self.false_positive_history[sensor_type]
                if fp['timestamp'] > cutoff
            ]
        
        # Log false positive
        if self.audit_logger:
            self.audit_logger.log_false_positive(
                sensor_type,
                {'event_id': event_id},
                analysis
            )
        
        # Record metrics
        if self.metrics_collector:
            fp_rate = self._get_false_positive_rate(sensor_type)
            self.metrics_collector.record_false_positive(
                sensor_type,
                sensor_type,
                reason,
                fp_rate
            )
    
    def should_enable_enforcement(self, sensor_type: str) -> bool:
        """Check if enforcement should be enabled for sensor type."""
        if self.enforcement_mode == EnforcementMode.FULL:
            return True
        elif self.enforcement_mode == EnforcementMode.MONITORING:
            return False
        else:  # GRADUAL mode
            fp_rate = self._get_false_positive_rate(sensor_type)
            return fp_rate < self.false_positive_threshold
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current enforcement status."""
        status = {
            'enforcement_mode': self.enforcement_mode.value,
            'false_positive_threshold': self.false_positive_threshold,
            'monitoring_period_days': self.monitoring_period_days,
            'sensors': {}
        }
        
        for sensor_type in self.sensors.keys():
            fp_rate = self._get_false_positive_rate(sensor_type)
            enforcement_enabled = self.should_enable_enforcement(sensor_type)
            
            status['sensors'][sensor_type] = {
                'false_positive_rate_per_week': fp_rate,
                'enforcement_enabled': enforcement_enabled,
                'monitoring_mode': not enforcement_enabled,
                'ready_for_enforcement': fp_rate < self.false_positive_threshold
            }
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enforcer performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'enforcement_mode': self.enforcement_mode.value,
            'evaluations_count': self.evaluations_count,
            'enforcement_actions_count': self.enforcement_actions_count,
            'evaluations_per_second': self.evaluations_count / max(uptime, 1),
            'action_rate': self.enforcement_actions_count / max(self.evaluations_count, 1),
            'uptime_seconds': uptime,
            'false_positive_threshold': self.false_positive_threshold,
            'sensors_count': len(self.sensors)
        }


def create_var_stress_enforcer(config: Dict[str, Any] = None,
                              audit_logger: Optional[AuditLogger] = None,
                              metrics_collector: Optional[RiskMetricsCollector] = None) -> VaRStressEnforcer:
    """Factory function to create VaR/Stress enforcer."""
    default_config = {
        'enforcement_mode': 'monitoring',
        'false_positive_threshold_per_week': 1.0,
        'monitoring_period_days': 30,
        'var_limits': {
            'var_95_limit': 100000,
            'var_99_limit': 200000,
            'var_999_limit': 500000
        },
        'stress_limits': {
            'max_stress_loss': 1000000,
            'max_scenario_failures': 3,
            'max_tail_ratio': 1.5
        }
    }
    
    if config:
        final_config = {**default_config, **config}
    else:
        final_config = default_config
    
    return VaRStressEnforcer(final_config, audit_logger, metrics_collector)


__all__ = [
    'VaRStressEnforcer',
    'EnforcementMode',
    'EnforcementAction',
    'EnforcementResult',
    'create_var_stress_enforcer'
]