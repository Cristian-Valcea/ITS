# src/risk/rules_engine.py
"""
Micro-kernel rules engine for risk policy evaluation.
Designed for ultra-low latency (50-100 µs) policy evaluation with hot-swappable rules.
"""

import time
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import logging
import numpy as np


class RuleAction(Enum):
    """Actions that can be taken when a rule is triggered."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    HALT = "halt"
    LIQUIDATE = "liquidate"
    REDUCE_POSITION = "reduce_position"
    HEDGE = "hedge"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleResult:
    """Result of a rule evaluation."""
    rule_id: str
    rule_name: str
    triggered: bool
    action: RuleAction
    severity: RuleSeverity
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ns: int = field(default_factory=lambda: time.time_ns())
    
    def get_evaluation_time_us(self) -> float:
        """Get evaluation time in microseconds."""
        return (time.time_ns() - self.evaluation_time_ns) / 1000.0


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating a complete policy."""
    policy_id: str
    policy_hash: str
    rule_results: List[RuleResult]
    overall_action: RuleAction
    triggered_rules: List[str]
    evaluation_time_ns: int = field(default_factory=lambda: time.time_ns())
    
    def get_evaluation_time_us(self) -> float:
        """Get total evaluation time in microseconds."""
        return (time.time_ns() - self.evaluation_time_ns) / 1000.0
    
    def has_critical_violations(self) -> bool:
        """Check if any critical rules were triggered."""
        return any(r.severity == RuleSeverity.CRITICAL and r.triggered for r in self.rule_results)
    
    def get_violations_by_severity(self) -> Dict[RuleSeverity, List[RuleResult]]:
        """Group violations by severity."""
        violations = {severity: [] for severity in RuleSeverity}
        for result in self.rule_results:
            if result.triggered:
                violations[result.severity].append(result)
        return violations


class BaseRule(ABC):
    """
    Abstract base class for risk rules.
    Designed for microsecond-level evaluation performance.
    """
    
    def __init__(self, rule_id: str, rule_name: str, config: Dict[str, Any]):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.config = config
        self.logger = logging.getLogger(f"Rule.{rule_name}")
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        self._trigger_count = 0
        
        # Rule configuration
        self.action = RuleAction(config.get('action', 'warn'))
        self.severity = RuleSeverity(config.get('severity', 'medium'))
        self.enabled = config.get('enabled', True)
        
        # Validation
        self._validate_config()
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> RuleResult:
        """
        Evaluate the rule against input data.
        
        Args:
            data: Input data for rule evaluation
            
        Returns:
            RuleResult with evaluation outcome
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate rule configuration."""
        pass
    
    def evaluate_safe(self, data: Dict[str, Any]) -> RuleResult:
        """Safe wrapper around evaluate with performance tracking."""
        if not self.enabled:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=False,
                action=RuleAction.ALLOW,
                severity=self.severity,
                message="Rule disabled"
            )
        
        start_time = time.time_ns()
        
        try:
            result = self.evaluate(data)
            
            # Update performance metrics
            evaluation_time = time.time_ns() - start_time
            self._evaluation_count += 1
            self._total_evaluation_time_ns += evaluation_time
            
            if result.triggered:
                self._trigger_count += 1
            
            # Add performance metadata
            result.metadata['evaluation_time_ns'] = evaluation_time
            result.metadata['evaluation_time_us'] = evaluation_time / 1000.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rule evaluation failed: {e}")
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=RuleAction.BLOCK,
                severity=RuleSeverity.CRITICAL,
                message=f"Rule evaluation error: {str(e)}"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this rule."""
        if self._evaluation_count == 0:
            return {
                'evaluation_count': 0,
                'trigger_count': 0,
                'avg_evaluation_time_us': 0.0,
                'trigger_rate': 0.0
            }
        
        avg_time_ns = self._total_evaluation_time_ns / self._evaluation_count
        
        return {
            'evaluation_count': self._evaluation_count,
            'trigger_count': self._trigger_count,
            'avg_evaluation_time_us': avg_time_ns / 1000.0,
            'trigger_rate': self._trigger_count / self._evaluation_count
        }


class ThresholdRule(BaseRule):
    """Simple threshold-based rule for numeric comparisons."""
    
    def _validate_config(self) -> None:
        required_keys = ['field', 'threshold', 'operator']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.field = self.config['field']
        self.threshold = float(self.config['threshold'])
        self.operator = self.config['operator']  # 'gt', 'lt', 'gte', 'lte', 'eq', 'ne'
        self.message_template = self.config.get('message', f"{self.field} {self.operator} {self.threshold}")
        
        if self.operator not in ['gt', 'lt', 'gte', 'lte', 'eq', 'ne']:
            raise ValueError(f"Invalid operator: {self.operator}")
    
    def evaluate(self, data: Dict[str, Any]) -> RuleResult:
        """Evaluate threshold rule."""
        if self.field not in data:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=RuleAction.BLOCK,
                severity=RuleSeverity.HIGH,
                message=f"Required field '{self.field}' not found in data"
            )
        
        value = data[self.field]
        
        # Handle numpy arrays by taking the last value
        if isinstance(value, np.ndarray):
            if value.size == 0:
                triggered = False
            else:
                value = float(value[-1])
        else:
            value = float(value)
        
        # Evaluate condition
        triggered = False
        if self.operator == 'gt':
            triggered = value > self.threshold
        elif self.operator == 'lt':
            triggered = value < self.threshold
        elif self.operator == 'gte':
            triggered = value >= self.threshold
        elif self.operator == 'lte':
            triggered = value <= self.threshold
        elif self.operator == 'eq':
            triggered = abs(value - self.threshold) < 1e-10
        elif self.operator == 'ne':
            triggered = abs(value - self.threshold) >= 1e-10
        
        message = self.message_template.format(
            field=self.field,
            value=value,
            threshold=self.threshold,
            operator=self.operator
        )
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=triggered,
            action=self.action if triggered else RuleAction.ALLOW,
            severity=self.severity,
            message=message,
            metadata={'field_value': value, 'threshold': self.threshold}
        )


class CompositeRule(BaseRule):
    """Rule that combines multiple sub-rules with logical operators."""
    
    def _validate_config(self) -> None:
        if 'sub_rules' not in self.config:
            raise ValueError("Missing required config key: sub_rules")
        if 'logic' not in self.config:
            raise ValueError("Missing required config key: logic")
        
        self.sub_rules: List[BaseRule] = []
        self.logic = self.config['logic']  # 'and', 'or', 'not'
        
        if self.logic not in ['and', 'or', 'not']:
            raise ValueError(f"Invalid logic operator: {self.logic}")
    
    def add_sub_rule(self, rule: BaseRule) -> None:
        """Add a sub-rule to this composite rule."""
        self.sub_rules.append(rule)
    
    def evaluate(self, data: Dict[str, Any]) -> RuleResult:
        """Evaluate composite rule."""
        if not self.sub_rules:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=False,
                action=RuleAction.ALLOW,
                severity=self.severity,
                message="No sub-rules defined"
            )
        
        # Evaluate all sub-rules
        sub_results = []
        for rule in self.sub_rules:
            result = rule.evaluate_safe(data)
            sub_results.append(result)
        
        # Apply logic
        triggered_results = [r for r in sub_results if r.triggered]
        
        if self.logic == 'and':
            triggered = len(triggered_results) == len(sub_results)
        elif self.logic == 'or':
            triggered = len(triggered_results) > 0
        elif self.logic == 'not':
            triggered = len(triggered_results) == 0
        
        # Determine overall action (most severe)
        if triggered:
            actions = [r.action for r in triggered_results]
            action_priority = {
                RuleAction.ALLOW: 0,
                RuleAction.WARN: 1,
                RuleAction.HEDGE: 2,
                RuleAction.REDUCE_POSITION: 3,
                RuleAction.BLOCK: 4,
                RuleAction.HALT: 5,
                RuleAction.LIQUIDATE: 6
            }
            overall_action = max(actions, key=lambda a: action_priority.get(a, 0))
        else:
            overall_action = RuleAction.ALLOW
        
        # Create message
        triggered_names = [r.rule_name for r in triggered_results]
        message = f"Composite rule ({self.logic}): {len(triggered_results)}/{len(sub_results)} triggered"
        if triggered_names:
            message += f" - {', '.join(triggered_names)}"
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=triggered,
            action=overall_action,
            severity=self.severity,
            message=message,
            metadata={
                'sub_rule_results': [
                    {
                        'rule_id': r.rule_id,
                        'triggered': r.triggered,
                        'action': r.action.value,
                        'message': r.message
                    }
                    for r in sub_results
                ],
                'logic': self.logic
            }
        )


class RiskPolicy:
    """
    A collection of rules that define a risk management policy.
    Supports hot-swapping and versioning.
    """
    
    def __init__(self, policy_id: str, policy_name: str, version: str = "1.0.0"):
        self.policy_id = policy_id
        self.policy_name = policy_name
        self.version = version
        self.rules: List[BaseRule] = []
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"Policy.{policy_name}")
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        
        # Policy hash for change detection
        self._policy_hash = self._calculate_hash()
    
    def add_rule(self, rule: BaseRule) -> None:
        """Add a rule to this policy."""
        self.rules.append(rule)
        self._policy_hash = self._calculate_hash()
        self.logger.info(f"Added rule {rule.rule_name} to policy {self.policy_name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from this policy."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        
        if len(self.rules) < original_count:
            self._policy_hash = self._calculate_hash()
            self.logger.info(f"Removed rule {rule_id} from policy {self.policy_name}")
            return True
        return False
    
    def evaluate(self, data: Dict[str, Any]) -> PolicyEvaluationResult:
        """Evaluate all rules in this policy."""
        start_time = time.time_ns()
        
        rule_results = []
        for rule in self.rules:
            result = rule.evaluate_safe(data)
            rule_results.append(result)
        
        # Determine overall action
        triggered_results = [r for r in rule_results if r.triggered]
        triggered_rule_ids = [r.rule_id for r in triggered_results]
        
        if not triggered_results:
            overall_action = RuleAction.ALLOW
        else:
            # Find the most severe action
            action_priority = {
                RuleAction.ALLOW: 0,
                RuleAction.WARN: 1,
                RuleAction.HEDGE: 2,
                RuleAction.REDUCE_POSITION: 3,
                RuleAction.BLOCK: 4,
                RuleAction.HALT: 5,
                RuleAction.LIQUIDATE: 6
            }
            actions = [r.action for r in triggered_results]
            overall_action = max(actions, key=lambda a: action_priority.get(a, 0))
        
        # Update performance metrics
        evaluation_time = time.time_ns() - start_time
        self._evaluation_count += 1
        self._total_evaluation_time_ns += evaluation_time
        
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            policy_hash=self._policy_hash,
            rule_results=rule_results,
            overall_action=overall_action,
            triggered_rules=triggered_rule_ids,
            evaluation_time_ns=start_time
        )
    
    def _calculate_hash(self) -> str:
        """Calculate hash of policy configuration for change detection."""
        policy_data = {
            'policy_id': self.policy_id,
            'version': self.version,
            'rules': [
                {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'config': rule.config
                }
                for rule in self.rules
            ]
        }
        
        policy_json = json.dumps(policy_data, sort_keys=True)
        return hashlib.sha256(policy_json.encode()).hexdigest()[:16]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this policy."""
        if self._evaluation_count == 0:
            return {
                'evaluation_count': 0,
                'avg_evaluation_time_us': 0.0,
                'rule_count': len(self.rules)
            }
        
        avg_time_ns = self._total_evaluation_time_ns / self._evaluation_count
        
        return {
            'evaluation_count': self._evaluation_count,
            'avg_evaluation_time_us': avg_time_ns / 1000.0,
            'rule_count': len(self.rules),
            'policy_hash': self._policy_hash
        }


class RulesEngine:
    """
    Micro-kernel rules engine for ultra-low latency policy evaluation.
    
    Features:
    - Hot-swappable policies
    - Microsecond-level evaluation
    - Comprehensive audit trail
    - Performance monitoring
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.policies: Dict[str, RiskPolicy] = {}
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        
        self.logger.info("RulesEngine initialized")
    
    def register_policy(self, policy: RiskPolicy) -> None:
        """Register a risk policy."""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Registered policy {policy.policy_name} (ID: {policy.policy_id})")
    
    def unregister_policy(self, policy_id: str) -> bool:
        """Unregister a risk policy."""
        if policy_id in self.policies:
            policy = self.policies.pop(policy_id)
            self.logger.info(f"Unregistered policy {policy.policy_name} (ID: {policy_id})")
            return True
        return False
    
    def evaluate_policy(self, policy_id: str, data: Dict[str, Any]) -> Optional[PolicyEvaluationResult]:
        """Evaluate a specific policy."""
        if policy_id not in self.policies:
            self.logger.error(f"Policy not found: {policy_id}")
            return None
        
        start_time = time.time_ns()
        
        policy = self.policies[policy_id]
        result = policy.evaluate(data)
        
        # Update engine performance metrics
        evaluation_time = time.time_ns() - start_time
        self._evaluation_count += 1
        self._total_evaluation_time_ns += evaluation_time
        
        return result
    
    def evaluate_all_policies(self, data: Dict[str, Any]) -> Dict[str, PolicyEvaluationResult]:
        """Evaluate all registered policies."""
        results = {}
        
        for policy_id, policy in self.policies.items():
            result = self.evaluate_policy(policy_id, data)
            if result:
                results[policy_id] = result
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the rules engine."""
        if self._evaluation_count == 0:
            return {
                'evaluation_count': 0,
                'avg_evaluation_time_us': 0.0,
                'policy_count': len(self.policies)
            }
        
        avg_time_ns = self._total_evaluation_time_ns / self._evaluation_count
        
        return {
            'evaluation_count': self._evaluation_count,
            'avg_evaluation_time_us': avg_time_ns / 1000.0,
            'policy_count': len(self.policies),
            'policies': {
                policy_id: policy.get_performance_stats()
                for policy_id, policy in self.policies.items()
            }
        }


class PolicyValidator:
    """Validator for risk policy configurations."""
    
    @staticmethod
    def validate_policy_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a policy configuration.
        
        Args:
            config: Policy configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ['policy_id', 'policy_name', 'rules']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        if 'rules' in config:
            rules = config['rules']
            if not isinstance(rules, list):
                errors.append("'rules' must be a list")
            else:
                for i, rule_config in enumerate(rules):
                    rule_errors = PolicyValidator._validate_rule_config(rule_config, i)
                    errors.extend(rule_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_rule_config(rule_config: Dict[str, Any], rule_index: int) -> List[str]:
        """Validate a single rule configuration."""
        errors = []
        prefix = f"Rule {rule_index}: "
        
        # Check required keys
        required_keys = ['rule_id', 'rule_name', 'rule_type']
        for key in required_keys:
            if key not in rule_config:
                errors.append(f"{prefix}Missing required key: {key}")
        
        # Validate rule type
        if 'rule_type' in rule_config:
            rule_type = rule_config['rule_type']
            valid_types = ['threshold', 'composite']
            if rule_type not in valid_types:
                errors.append(f"{prefix}Invalid rule_type: {rule_type}. Must be one of {valid_types}")
        
        # Validate action
        if 'action' in rule_config:
            action = rule_config['action']
            valid_actions = [a.value for a in RuleAction]
            if action not in valid_actions:
                errors.append(f"{prefix}Invalid action: {action}. Must be one of {valid_actions}")
        
        # Validate severity
        if 'severity' in rule_config:
            severity = rule_config['severity']
            valid_severities = [s.value for s in RuleSeverity]
            if severity not in valid_severities:
                errors.append(f"{prefix}Invalid severity: {severity}. Must be one of {valid_severities}")
        
        return errors