# src/risk/audit/audit_logger.py
"""
Risk Audit Trail System - JSON-L logging for compliance and analysis.

Provides comprehensive audit logging for:
- Risk sensor evaluations
- Rule engine decisions
- VaR/stress test results
- False positive tracking
- Enforcement actions

All logs are structured JSON-L format for easy parsing and analysis.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue, Empty
import gzip
import os


class AuditEventType(Enum):
    """Types of audit events."""
    SENSOR_EVALUATION = "sensor_evaluation"
    RULE_EVALUATION = "rule_evaluation"
    VAR_CALCULATION = "var_calculation"
    STRESS_TEST = "stress_test"
    ENFORCEMENT_ACTION = "enforcement_action"
    FALSE_POSITIVE = "false_positive"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class AuditEvent:
    """Structured audit event."""
    event_type: AuditEventType
    timestamp: str
    event_id: str
    source: str
    severity: str
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        event_dict = asdict(self)
        event_dict['event_type'] = self.event_type.value
        return json.dumps(event_dict, default=str, separators=(',', ':'))


class AuditLogger:
    """
    High-performance audit logger with JSON-L output.
    
    Features:
    - Asynchronous logging to avoid latency impact
    - Structured JSON-L format
    - Automatic log rotation and compression
    - Performance metrics tracking
    - False positive analysis support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audit logger.
        
        Args:
            config: Configuration dictionary with:
                - log_directory: Directory for audit logs
                - max_file_size_mb: Max file size before rotation
                - max_files: Max number of files to keep
                - compress_old_files: Whether to compress rotated files
                - async_logging: Enable async logging (default: True)
                - buffer_size: Async buffer size (default: 1000)
        """
        self.config = config
        self.log_directory = Path(config.get('log_directory', 'logs/audit'))
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.max_files = config.get('max_files', 30)
        self.compress_old_files = config.get('compress_old_files', True)
        self.async_logging = config.get('async_logging', True)
        self.buffer_size = config.get('buffer_size', 1000)
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_log_file = None
        self.current_file_size = 0
        
        # Async logging setup
        if self.async_logging:
            self.log_queue = Queue(maxsize=self.buffer_size)
            self.logging_thread = threading.Thread(target=self._async_log_worker, daemon=True)
            self.logging_thread.start()
        
        # Performance tracking
        self.events_logged = 0
        self.bytes_written = 0
        self.start_time = time.time()
        
        # Initialize current log file
        self._rotate_log_file()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"AuditLogger initialized: {self.log_directory}")
    
    def log_sensor_evaluation(self, sensor_id: str, sensor_name: str, 
                            result: Any, data: Dict[str, Any]) -> None:
        """Log sensor evaluation event."""
        event = AuditEvent(
            event_type=AuditEventType.SENSOR_EVALUATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"sensor_{sensor_id}_{int(time.time_ns())}",
            source=sensor_id,
            severity=self._map_sensor_severity(result),
            message=f"Sensor {sensor_name} evaluation",
            data={
                'sensor_id': sensor_id,
                'sensor_name': sensor_name,
                'triggered': getattr(result, 'triggered', False),
                'value': getattr(result, 'value', None),
                'threshold': getattr(result, 'threshold', None),
                'action': getattr(result, 'action', None).value if hasattr(getattr(result, 'action', None), 'value') else None,
                'confidence': getattr(result, 'confidence', None),
                'evaluation_time_us': getattr(result, 'evaluation_time_ns', 0) / 1000,
                'input_data_keys': list(data.keys()) if data else []
            },
            metadata={
                'failure_mode': getattr(result, 'failure_mode', None).value if hasattr(getattr(result, 'failure_mode', None), 'value') else None,
                'priority': getattr(result, 'priority', None).value if hasattr(getattr(result, 'priority', None), 'value') else None,
                'data_age_ns': getattr(result, 'data_age_ns', 0)
            }
        )
        
        self._log_event(event)
    
    def log_rule_evaluation(self, rule_id: str, rule_name: str,
                          result: Any, context: Dict[str, Any]) -> None:
        """Log rule evaluation event."""
        event = AuditEvent(
            event_type=AuditEventType.RULE_EVALUATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"rule_{rule_id}_{int(time.time_ns())}",
            source=rule_id,
            severity=self._map_rule_severity(result),
            message=f"Rule {rule_name} evaluation",
            data={
                'rule_id': rule_id,
                'rule_name': rule_name,
                'triggered': getattr(result, 'triggered', False),
                'action': getattr(result, 'action', None).value if hasattr(getattr(result, 'action', None), 'value') else None,
                'severity': getattr(result, 'severity', None).value if hasattr(getattr(result, 'severity', None), 'value') else None,
                'evaluation_time_us': getattr(result, 'evaluation_time_ns', 0) / 1000,
                'context_keys': list(context.keys()) if context else []
            },
            metadata=getattr(result, 'metadata', {})
        )
        
        self._log_event(event)
    
    def log_var_calculation(self, calculator_type: str, results: Dict[str, Any],
                          portfolio_data: Dict[str, Any]) -> None:
        """Log VaR calculation event."""
        event = AuditEvent(
            event_type=AuditEventType.VAR_CALCULATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"var_{calculator_type}_{int(time.time_ns())}",
            source=f"var_calculator_{calculator_type}",
            severity="info",
            message=f"VaR calculation completed: {calculator_type}",
            data={
                'calculator_type': calculator_type,
                'var_95': results.get('var_95'),
                'var_99': results.get('var_99'),
                'var_999': results.get('var_999'),
                'portfolio_value': results.get('portfolio_value'),
                'volatility': results.get('volatility'),
                'observations': results.get('observations'),
                'method': results.get('method')
            },
            metadata={
                'portfolio_size': len(portfolio_data.get('positions', {})),
                'data_quality': self._assess_data_quality(portfolio_data),
                'calculation_timestamp': results.get('calculation_timestamp')
            }
        )
        
        self._log_event(event)
    
    def log_stress_test(self, test_type: str, results: Dict[str, Any],
                       scenarios: List[str]) -> None:
        """Log stress test event."""
        event = AuditEvent(
            event_type=AuditEventType.STRESS_TEST,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"stress_{test_type}_{int(time.time_ns())}",
            source=f"stress_test_{test_type}",
            severity=self._assess_stress_test_severity(results),
            message=f"Stress test completed: {test_type}",
            data={
                'test_type': test_type,
                'scenarios_run': scenarios,
                'worst_case_loss': results.get('stress_worst_case'),
                'stress_var_99': results.get('stress_var_99'),
                'stress_var_999': results.get('stress_var_999'),
                'failed_scenarios': results.get('failed_scenarios', 0),
                'total_scenarios': results.get('stress_scenarios_count', 0),
                'worst_case_scenario': results.get('worst_case_scenario', {})
            },
            metadata={
                'monte_carlo_runs': results.get('monte_carlo_runs'),
                'historical_scenarios': results.get('historical_scenarios'),
                'test_duration_seconds': results.get('test_duration_seconds')
            }
        )
        
        self._log_event(event)
    
    def log_enforcement_action(self, action_type: str, reason: str,
                             details: Dict[str, Any]) -> None:
        """Log enforcement action event."""
        event = AuditEvent(
            event_type=AuditEventType.ENFORCEMENT_ACTION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"enforce_{action_type}_{int(time.time_ns())}",
            source="risk_enforcement",
            severity=self._map_enforcement_severity(action_type),
            message=f"Enforcement action taken: {action_type}",
            data={
                'action_type': action_type,
                'reason': reason,
                'monitoring_mode': details.get('monitoring_mode', False),
                'enforcement_enabled': details.get('enforcement_enabled', True),
                'affected_positions': details.get('affected_positions', []),
                'risk_metrics': details.get('risk_metrics', {})
            },
            metadata={
                'triggered_by': details.get('triggered_by'),
                'override_available': details.get('override_available', False),
                'escalation_required': details.get('escalation_required', False)
            }
        )
        
        self._log_event(event)
    
    def log_false_positive(self, sensor_id: str, original_event: Dict[str, Any],
                          analysis: Dict[str, Any]) -> None:
        """Log false positive analysis event."""
        event = AuditEvent(
            event_type=AuditEventType.FALSE_POSITIVE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"fp_{sensor_id}_{int(time.time_ns())}",
            source=sensor_id,
            severity="warning",
            message=f"False positive identified: {sensor_id}",
            data={
                'sensor_id': sensor_id,
                'original_event_id': original_event.get('event_id'),
                'original_timestamp': original_event.get('timestamp'),
                'false_positive_reason': analysis.get('reason'),
                'market_outcome': analysis.get('market_outcome'),
                'expected_vs_actual': analysis.get('expected_vs_actual'),
                'confidence_score': analysis.get('confidence_score')
            },
            metadata={
                'analysis_method': analysis.get('method'),
                'analyst': analysis.get('analyst'),
                'review_timestamp': analysis.get('review_timestamp')
            }
        )
        
        self._log_event(event)
    
    def log_performance_metric(self, metric_name: str, value: float,
                             context: Dict[str, Any]) -> None:
        """Log performance metric event."""
        event = AuditEvent(
            event_type=AuditEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=f"perf_{metric_name}_{int(time.time_ns())}",
            source="performance_monitor",
            severity="info",
            message=f"Performance metric: {metric_name}",
            data={
                'metric_name': metric_name,
                'value': value,
                'unit': context.get('unit'),
                'threshold': context.get('threshold'),
                'target': context.get('target')
            },
            metadata=context.get('metadata', {})
        )
        
        self._log_event(event)
    
    def _log_event(self, event: AuditEvent) -> None:
        """Log event to file."""
        if self.async_logging:
            try:
                self.log_queue.put_nowait(event)
            except:
                # Queue full, log synchronously as fallback
                self._write_event_to_file(event)
        else:
            self._write_event_to_file(event)
        
        self.events_logged += 1
    
    def _async_log_worker(self) -> None:
        """Async logging worker thread."""
        while True:
            try:
                event = self.log_queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    break
                self._write_event_to_file(event)
                self.log_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in async log worker: {e}")
    
    def _write_event_to_file(self, event: AuditEvent) -> None:
        """Write event to current log file."""
        try:
            json_line = event.to_json() + '\n'
            
            # Check if rotation needed
            if self.current_file_size + len(json_line) > self.max_file_size_mb * 1024 * 1024:
                self._rotate_log_file()
            
            # Write to file
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json_line)
                f.flush()
            
            self.current_file_size += len(json_line)
            self.bytes_written += len(json_line)
            
        except Exception as e:
            self.logger.error(f"Error writing audit event: {e}")
    
    def _rotate_log_file(self) -> None:
        """Rotate log file when size limit reached."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_log_file = self.log_directory / f"audit_{timestamp}.jsonl"
        self.current_file_size = 0
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """Clean up old log files."""
        try:
            # Get all audit log files
            log_files = list(self.log_directory.glob("audit_*.jsonl*"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only max_files
            for old_file in log_files[self.max_files:]:
                old_file.unlink()
            
            # Compress old files if enabled
            if self.compress_old_files:
                for log_file in log_files[1:]:  # Skip current file
                    if not log_file.name.endswith('.gz'):
                        self._compress_file(log_file)
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress log file."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            file_path.unlink()  # Remove original
            
        except Exception as e:
            self.logger.error(f"Error compressing file {file_path}: {e}")
    
    def _map_sensor_severity(self, result: Any) -> str:
        """Map sensor result to severity level."""
        if not getattr(result, 'triggered', False):
            return "info"
        
        action = getattr(result, 'action', None)
        if hasattr(action, 'value'):
            action_value = action.value
            if action_value in ['halt', 'liquidate']:
                return "critical"
            elif action_value in ['block', 'throttle']:
                return "high"
            elif action_value == 'warn':
                return "medium"
            else:
                return "low"
        
        return "info"
    
    def _map_rule_severity(self, result: Any) -> str:
        """Map rule result to severity level."""
        severity = getattr(result, 'severity', None)
        if hasattr(severity, 'value'):
            return severity.value
        return "info"
    
    def _map_enforcement_severity(self, action_type: str) -> str:
        """Map enforcement action to severity level."""
        if action_type in ['halt', 'liquidate']:
            return "critical"
        elif action_type in ['block', 'reduce_position']:
            return "high"
        elif action_type in ['throttle', 'warn']:
            return "medium"
        else:
            return "low"
    
    def _assess_stress_test_severity(self, results: Dict[str, Any]) -> str:
        """Assess stress test result severity."""
        failed_scenarios = results.get('failed_scenarios', 0)
        total_scenarios = results.get('stress_scenarios_count', 1)
        failure_rate = failed_scenarios / max(total_scenarios, 1)
        
        if failure_rate >= 0.5:
            return "critical"
        elif failure_rate >= 0.3:
            return "high"
        elif failure_rate >= 0.1:
            return "medium"
        else:
            return "info"
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """Assess data quality for calculations."""
        if not data:
            return "poor"
        
        positions = data.get('positions', {})
        if len(positions) == 0:
            return "poor"
        elif len(positions) < 5:
            return "fair"
        else:
            return "good"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get audit logger performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'events_logged': self.events_logged,
            'bytes_written': self.bytes_written,
            'events_per_second': self.events_logged / max(uptime, 1),
            'bytes_per_second': self.bytes_written / max(uptime, 1),
            'uptime_seconds': uptime,
            'current_file': str(self.current_log_file),
            'current_file_size_mb': self.current_file_size / (1024 * 1024),
            'async_logging_enabled': self.async_logging,
            'queue_size': self.log_queue.qsize() if self.async_logging else 0
        }
    
    def shutdown(self) -> None:
        """Shutdown audit logger gracefully."""
        if self.async_logging:
            # Signal shutdown and wait for queue to empty
            self.log_queue.put(None)
            self.logging_thread.join(timeout=5.0)
        
        self.logger.info("AuditLogger shutdown complete")


def create_audit_logger(config: Dict[str, Any] = None) -> AuditLogger:
    """Factory function to create audit logger with default config."""
    default_config = {
        'log_directory': 'logs/audit',
        'max_file_size_mb': 100,
        'max_files': 30,
        'compress_old_files': True,
        'async_logging': True,
        'buffer_size': 1000
    }
    
    if config:
        final_config = {**default_config, **config}
    else:
        final_config = default_config
    
    return AuditLogger(final_config)


__all__ = [
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'create_audit_logger'
]