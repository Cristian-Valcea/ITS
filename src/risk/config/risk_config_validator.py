# src/risk/config/risk_config_validator.py
"""
Risk Configuration Validator with JSON Schema validation.

Prevents malformed YAML from wiping all risk limits during hot-reload.
Validates configuration structure, types, and business rules before swap-in.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError
import yaml


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = "critical"    # Blocks hot-reload completely
    ERROR = "error"         # Blocks hot-reload with warnings
    WARNING = "warning"     # Allows hot-reload with warnings
    INFO = "info"          # Informational only


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    info: List[Dict[str, Any]]
    validation_time_ms: float
    
    def has_blocking_errors(self) -> bool:
        """Check if validation has errors that block hot-reload."""
        return any(
            error.get('severity') in [ValidationSeverity.CRITICAL.value, ValidationSeverity.ERROR.value]
            for error in self.errors
        )
    
    def get_summary(self) -> str:
        """Get validation summary string."""
        if self.is_valid and not self.has_blocking_errors():
            return f"✅ Valid ({len(self.warnings)} warnings, {len(self.info)} info)"
        else:
            return f"❌ Invalid ({len(self.errors)} errors, {len(self.warnings)} warnings)"


class RiskConfigValidator:
    """
    Risk Configuration Validator with JSON Schema validation.
    
    Features:
    - JSON Schema validation for structure and types
    - Business rule validation for risk limits
    - Diff analysis between old and new configurations
    - Hot-reload safety checks
    - Performance optimized validation
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        self.logger = logging.getLogger("RiskConfigValidator")
        self.schema = self._load_schema(schema_path)
        self.validation_count = 0
        self.last_valid_config: Optional[Dict[str, Any]] = None
        
        self.logger.info("RiskConfigValidator initialized")
    
    def _load_schema(self, schema_path: Optional[str]) -> Dict[str, Any]:
        """Load JSON schema for risk configuration validation."""
        if schema_path:
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load schema from {schema_path}: {e}")
        
        # Default embedded schema
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Risk Configuration Schema",
            "type": "object",
            "required": ["policies", "active_policy"],
            "properties": {
                "policies": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z0-9_]+$": {
                            "type": "object",
                            "required": ["calculators", "rules"],
                            "properties": {
                                "calculators": {
                                    "type": "object",
                                    "properties": {
                                        "var": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {"type": "boolean"},
                                                "config": {
                                                    "type": "object",
                                                    "properties": {
                                                        "confidence_levels": {
                                                            "type": "array",
                                                            "items": {"type": "number", "minimum": 0, "maximum": 1}
                                                        },
                                                        "window_days": {"type": "integer", "minimum": 1},
                                                        "method": {"type": "string", "enum": ["parametric", "historical", "monte_carlo"]}
                                                    }
                                                }
                                            }
                                        },
                                        "stress_test": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {"type": "boolean"},
                                                "config": {
                                                    "type": "object",
                                                    "properties": {
                                                        "scenarios": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        },
                                                        "confidence_levels": {
                                                            "type": "array",
                                                            "items": {"type": "number", "minimum": 0, "maximum": 1}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "rules": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["rule_id", "rule_name", "threshold", "action"],
                                        "properties": {
                                            "rule_id": {"type": "string"},
                                            "rule_name": {"type": "string"},
                                            "rule_type": {"type": "string"},
                                            "field": {"type": "string"},
                                            "threshold": {"type": "number"},
                                            "operator": {"type": "string", "enum": ["gt", "lt", "gte", "lte", "eq", "ne"]},
                                            "action": {"type": "string", "enum": ["warn", "throttle", "halt", "reduce_position"]},
                                            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                                            "monitoring_mode": {"type": "boolean"},
                                            "enforcement_enabled": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "active_policy": {"type": "string"},
                "enforcement": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["monitoring", "gradual", "full"]},
                        "false_positive_threshold_per_week": {"type": "number", "minimum": 0},
                        "var_limits": {
                            "type": "object",
                            "properties": {
                                "var_95_limit": {"type": "number", "minimum": 0},
                                "var_99_limit": {"type": "number", "minimum": 0},
                                "var_999_limit": {"type": "number", "minimum": 0}
                            }
                        },
                        "stress_limits": {
                            "type": "object",
                            "properties": {
                                "max_stress_loss": {"type": "number", "minimum": 0},
                                "max_scenario_failures": {"type": "integer", "minimum": 0},
                                "max_tail_ratio": {"type": "number", "minimum": 0}
                            }
                        }
                    }
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any], 
                       previous_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate risk configuration with comprehensive checks.
        
        Args:
            config: New configuration to validate
            previous_config: Previous configuration for diff analysis
            
        Returns:
            ValidationResult with validation status and details
        """
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        info = []
        
        try:
            # 1. JSON Schema validation
            schema_errors = self._validate_schema(config)
            errors.extend(schema_errors)
            
            # 2. Business rule validation
            business_errors, business_warnings = self._validate_business_rules(config)
            errors.extend(business_errors)
            warnings.extend(business_warnings)
            
            # 3. Configuration diff analysis
            if previous_config:
                diff_warnings, diff_info = self._analyze_config_diff(previous_config, config)
                warnings.extend(diff_warnings)
                info.extend(diff_info)
            
            # 4. Hot-reload safety checks
            safety_errors, safety_warnings = self._validate_hot_reload_safety(config)
            errors.extend(safety_errors)
            warnings.extend(safety_warnings)
            
            validation_time_ms = (time.time() - start_time) * 1000
            self.validation_count += 1
            
            is_valid = len([e for e in errors if e.get('severity') in ['critical', 'error']]) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                info=info,
                validation_time_ms=validation_time_ms
            )
            
            if is_valid:
                self.last_valid_config = config.copy()
            
            self.logger.info(f"Config validation completed: {result.get_summary()}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{
                    'type': 'validation_exception',
                    'severity': ValidationSeverity.CRITICAL.value,
                    'message': f"Validation failed: {str(e)}",
                    'path': 'root'
                }],
                warnings=[],
                info=[],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_schema(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate configuration against JSON schema."""
        errors = []
        
        try:
            validate(instance=config, schema=self.schema)
        except ValidationError as e:
            errors.append({
                'type': 'schema_validation',
                'severity': ValidationSeverity.ERROR.value,
                'message': e.message,
                'path': '.'.join(str(p) for p in e.absolute_path),
                'schema_path': '.'.join(str(p) for p in e.schema_path)
            })
        except Exception as e:
            errors.append({
                'type': 'schema_error',
                'severity': ValidationSeverity.CRITICAL.value,
                'message': f"Schema validation failed: {str(e)}",
                'path': 'root'
            })
        
        return errors
    
    def _validate_business_rules(self, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate business rules and risk management logic."""
        errors = []
        warnings = []
        
        # Check active policy exists
        active_policy = config.get('active_policy')
        if active_policy and active_policy not in config.get('policies', {}):
            errors.append({
                'type': 'missing_active_policy',
                'severity': ValidationSeverity.ERROR.value,
                'message': f"Active policy '{active_policy}' not found in policies",
                'path': 'active_policy'
            })
        
        # Validate VaR limits ordering
        enforcement = config.get('enforcement', {})
        var_limits = enforcement.get('var_limits', {})
        if var_limits:
            var_95 = var_limits.get('var_95_limit', 0)
            var_99 = var_limits.get('var_99_limit', 0)
            var_999 = var_limits.get('var_999_limit', 0)
            
            if var_95 > var_99:
                warnings.append({
                    'type': 'var_limit_ordering',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': "VaR 95% limit should be less than 99% limit",
                    'path': 'enforcement.var_limits'
                })
            
            if var_99 > var_999:
                warnings.append({
                    'type': 'var_limit_ordering',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': "VaR 99% limit should be less than 99.9% limit",
                    'path': 'enforcement.var_limits'
                })
        
        # Validate rule thresholds are positive
        policies = config.get('policies', {})
        for policy_name, policy in policies.items():
            rules = policy.get('rules', [])
            for i, rule in enumerate(rules):
                threshold = rule.get('threshold')
                if threshold is not None and threshold < 0:
                    warnings.append({
                        'type': 'negative_threshold',
                        'severity': ValidationSeverity.WARNING.value,
                        'message': f"Rule '{rule.get('rule_id', 'unknown')}' has negative threshold",
                        'path': f'policies.{policy_name}.rules[{i}].threshold'
                    })
        
        # Check for duplicate rule IDs
        for policy_name, policy in policies.items():
            rules = policy.get('rules', [])
            rule_ids = [rule.get('rule_id') for rule in rules if rule.get('rule_id')]
            duplicates = set([x for x in rule_ids if rule_ids.count(x) > 1])
            
            for duplicate in duplicates:
                errors.append({
                    'type': 'duplicate_rule_id',
                    'severity': ValidationSeverity.ERROR.value,
                    'message': f"Duplicate rule ID '{duplicate}' in policy '{policy_name}'",
                    'path': f'policies.{policy_name}.rules'
                })
        
        return errors, warnings
    
    def _analyze_config_diff(self, old_config: Dict[str, Any], 
                           new_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze differences between old and new configurations."""
        warnings = []
        info = []
        
        # Check if active policy changed
        old_active = old_config.get('active_policy')
        new_active = new_config.get('active_policy')
        if old_active != new_active:
            info.append({
                'type': 'active_policy_change',
                'severity': ValidationSeverity.INFO.value,
                'message': f"Active policy changed from '{old_active}' to '{new_active}'",
                'path': 'active_policy'
            })
        
        # Check enforcement mode changes
        old_mode = old_config.get('enforcement', {}).get('mode')
        new_mode = new_config.get('enforcement', {}).get('mode')
        if old_mode != new_mode:
            if new_mode == 'full' and old_mode in ['monitoring', 'gradual']:
                warnings.append({
                    'type': 'enforcement_mode_escalation',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': f"Enforcement mode escalated from '{old_mode}' to '{new_mode}'",
                    'path': 'enforcement.mode'
                })
            else:
                info.append({
                    'type': 'enforcement_mode_change',
                    'severity': ValidationSeverity.INFO.value,
                    'message': f"Enforcement mode changed from '{old_mode}' to '{new_mode}'",
                    'path': 'enforcement.mode'
                })
        
        # Check VaR limit changes
        old_var_limits = old_config.get('enforcement', {}).get('var_limits', {})
        new_var_limits = new_config.get('enforcement', {}).get('var_limits', {})
        
        for limit_type in ['var_95_limit', 'var_99_limit', 'var_999_limit']:
            old_val = old_var_limits.get(limit_type)
            new_val = new_var_limits.get(limit_type)
            
            if old_val != new_val and old_val is not None and new_val is not None:
                change_pct = ((new_val - old_val) / old_val) * 100 if old_val != 0 else 0
                
                if abs(change_pct) > 50:  # >50% change
                    warnings.append({
                        'type': 'large_limit_change',
                        'severity': ValidationSeverity.WARNING.value,
                        'message': f"{limit_type} changed by {change_pct:.1f}% (${old_val:,.0f} → ${new_val:,.0f})",
                        'path': f'enforcement.var_limits.{limit_type}'
                    })
                else:
                    info.append({
                        'type': 'limit_change',
                        'severity': ValidationSeverity.INFO.value,
                        'message': f"{limit_type} changed by {change_pct:.1f}% (${old_val:,.0f} → ${new_val:,.0f})",
                        'path': f'enforcement.var_limits.{limit_type}'
                    })
        
        return warnings, info
    
    def _validate_hot_reload_safety(self, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate configuration is safe for hot-reload."""
        errors = []
        warnings = []
        
        # Ensure critical sections exist
        if not config.get('policies'):
            errors.append({
                'type': 'missing_policies',
                'severity': ValidationSeverity.CRITICAL.value,
                'message': "No policies defined - would wipe all risk limits",
                'path': 'policies'
            })
        
        active_policy = config.get('active_policy')
        if active_policy:
            policy = config.get('policies', {}).get(active_policy, {})
            
            # Check active policy has rules
            if not policy.get('rules'):
                errors.append({
                    'type': 'no_rules_in_active_policy',
                    'severity': ValidationSeverity.CRITICAL.value,
                    'message': f"Active policy '{active_policy}' has no rules - would disable all risk controls",
                    'path': f'policies.{active_policy}.rules'
                })
            
            # Check active policy has calculators
            if not policy.get('calculators'):
                warnings.append({
                    'type': 'no_calculators_in_active_policy',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': f"Active policy '{active_policy}' has no calculators configured",
                    'path': f'policies.{active_policy}.calculators'
                })
        
        # Check enforcement configuration exists
        if not config.get('enforcement'):
            warnings.append({
                'type': 'missing_enforcement_config',
                'severity': ValidationSeverity.WARNING.value,
                'message': "No enforcement configuration - enforcement will be disabled",
                'path': 'enforcement'
            })
        
        return errors, warnings
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'validation_count': self.validation_count,
            'has_last_valid_config': self.last_valid_config is not None,
            'schema_loaded': self.schema is not None
        }


def create_risk_config_validator(schema_path: Optional[str] = None) -> RiskConfigValidator:
    """Factory function to create RiskConfigValidator."""
    return RiskConfigValidator(schema_path)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test configuration validation
    validator = create_risk_config_validator()
    
    # Test valid configuration
    valid_config = {
        "policies": {
            "test_policy": {
                "calculators": {
                    "var": {
                        "enabled": True,
                        "config": {
                            "confidence_levels": [0.95, 0.99, 0.999],
                            "window_days": 250,
                            "method": "parametric"
                        }
                    }
                },
                "rules": [
                    {
                        "rule_id": "var_95_limit",
                        "rule_name": "VaR 95% Limit",
                        "threshold": 100000,
                        "action": "warn"
                    }
                ]
            }
        },
        "active_policy": "test_policy",
        "enforcement": {
            "mode": "monitoring",
            "var_limits": {
                "var_95_limit": 100000,
                "var_99_limit": 200000,
                "var_999_limit": 500000
            }
        }
    }
    
    result = validator.validate_config(valid_config)
    print(f"Valid config test: {result.get_summary()}")
    
    # Test invalid configuration
    invalid_config = {
        "policies": {},  # Empty policies - should trigger critical error
        "active_policy": "nonexistent_policy"
    }
    
    result = validator.validate_config(invalid_config)
    print(f"Invalid config test: {result.get_summary()}")
    
    # Test diff analysis
    modified_config = valid_config.copy()
    modified_config['enforcement']['mode'] = 'full'
    modified_config['enforcement']['var_limits']['var_95_limit'] = 150000
    
    result = validator.validate_config(modified_config, valid_config)
    print(f"Config diff test: {result.get_summary()}")
    
    print(f"Validation stats: {validator.get_validation_stats()}")