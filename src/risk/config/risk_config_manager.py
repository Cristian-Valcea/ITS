# src/risk/config/risk_config_manager.py
"""
Risk Configuration Manager with Hot-Reload Integration.

Integrates hot-reload system with risk management framework,
providing thread-safe configuration access and change notifications.
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import time

from .risk_config_validator import RiskConfigValidator, ValidationResult
from .risk_config_hot_reload import RiskConfigHotReloader, ReloadResult, ReloadStatus


class ConfigChangeType(Enum):
    """Types of configuration changes."""
    POLICY_CHANGE = "policy_change"
    ENFORCEMENT_CHANGE = "enforcement_change"
    LIMITS_CHANGE = "limits_change"
    CALCULATOR_CHANGE = "calculator_change"
    RULES_CHANGE = "rules_change"


@dataclass
class ConfigChangeEvent:
    """Configuration change event."""
    change_type: ConfigChangeType
    old_value: Any
    new_value: Any
    path: str
    timestamp: float
    
    def get_summary(self) -> str:
        """Get change event summary."""
        return f"{self.change_type.value}: {self.path} changed"


class RiskConfigManager:
    """
    Risk Configuration Manager with Hot-Reload Integration.
    
    Features:
    - Thread-safe configuration access
    - Hot-reload with validation
    - Configuration change detection and notifications
    - Fallback to safe defaults on failures
    - Performance monitoring
    - Integration with risk management components
    """
    
    def __init__(self, config_paths: List[str], 
                 validator: Optional[RiskConfigValidator] = None,
                 enable_hot_reload: bool = True):
        self.config_paths = config_paths
        self.validator = validator or RiskConfigValidator()
        self.enable_hot_reload = enable_hot_reload
        self.logger = logging.getLogger("RiskConfigManager")
        
        # Configuration state
        self.config_lock = threading.RLock()
        self.current_config: Optional[Dict[str, Any]] = None
        self.safe_defaults = self._get_safe_defaults()
        
        # Hot-reload system
        self.hot_reloader: Optional[RiskConfigHotReloader] = None
        if enable_hot_reload:
            self.hot_reloader = RiskConfigHotReloader(config_paths, validator)
            self.hot_reloader.add_reload_callback(self._on_config_reload)
        
        # Change notification system
        self.change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # Statistics
        self.config_access_count = 0
        self.last_access_time: Optional[float] = None
        self.initialization_time: Optional[float] = None
        
        self.logger.info("RiskConfigManager initialized")
    
    def initialize(self) -> bool:
        """Initialize configuration manager and load initial config."""
        start_time = time.time()
        
        try:
            if self.hot_reloader:
                # Load initial configuration
                result = self.hot_reloader.load_initial_config()
                
                if result.is_success():
                    with self.config_lock:
                        self.current_config = result.config
                    
                    # Start file watching
                    self.hot_reloader.start_watching()
                    
                    self.initialization_time = time.time() - start_time
                    self.logger.info(f"Configuration manager initialized successfully ({self.initialization_time*1000:.1f}ms)")
                    return True
                else:
                    self.logger.error(f"Failed to load initial configuration: {result.get_summary()}")
                    # Fall back to safe defaults
                    with self.config_lock:
                        self.current_config = self.safe_defaults
                    
                    self.logger.warning("Using safe default configuration")
                    return False
            else:
                # Hot-reload disabled, use safe defaults
                with self.config_lock:
                    self.current_config = self.safe_defaults
                
                self.initialization_time = time.time() - start_time
                self.logger.info("Configuration manager initialized with safe defaults (hot-reload disabled)")
                return True
                
        except Exception as e:
            self.logger.error(f"Configuration manager initialization failed: {e}")
            # Emergency fallback to safe defaults
            with self.config_lock:
                self.current_config = self.safe_defaults
            return False
    
    def shutdown(self) -> None:
        """Shutdown configuration manager."""
        if self.hot_reloader:
            self.hot_reloader.stop_watching()
        self.logger.info("Configuration manager shutdown complete")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration (thread-safe)."""
        with self.config_lock:
            self.config_access_count += 1
            self.last_access_time = time.time()
            
            if self.current_config is None:
                self.logger.warning("No configuration available, returning safe defaults")
                return self.safe_defaults.copy()
            
            return self.current_config.copy()
    
    def get_active_policy(self) -> Dict[str, Any]:
        """Get active policy configuration."""
        config = self.get_config()
        active_policy_name = config.get('active_policy')
        
        if not active_policy_name:
            self.logger.warning("No active policy specified, using safe defaults")
            return self.safe_defaults['policies']['safe_default']
        
        policies = config.get('policies', {})
        if active_policy_name not in policies:
            self.logger.error(f"Active policy '{active_policy_name}' not found, using safe defaults")
            return self.safe_defaults['policies']['safe_default']
        
        return policies[active_policy_name]
    
    def get_enforcement_config(self) -> Dict[str, Any]:
        """Get enforcement configuration."""
        config = self.get_config()
        return config.get('enforcement', self.safe_defaults['enforcement'])
    
    def get_var_limits(self) -> Dict[str, float]:
        """Get VaR limits configuration."""
        enforcement = self.get_enforcement_config()
        return enforcement.get('var_limits', self.safe_defaults['enforcement']['var_limits'])
    
    def get_stress_limits(self) -> Dict[str, Any]:
        """Get stress test limits configuration."""
        enforcement = self.get_enforcement_config()
        return enforcement.get('stress_limits', self.safe_defaults['enforcement']['stress_limits'])
    
    def get_calculator_config(self, calculator_name: str) -> Dict[str, Any]:
        """Get configuration for specific calculator."""
        policy = self.get_active_policy()
        calculators = policy.get('calculators', {})
        
        if calculator_name not in calculators:
            self.logger.warning(f"Calculator '{calculator_name}' not configured, using defaults")
            return {}
        
        return calculators[calculator_name]
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get risk rules from active policy."""
        policy = self.get_active_policy()
        return policy.get('rules', [])
    
    def is_hot_reload_enabled(self) -> bool:
        """Check if hot-reload is enabled."""
        return self.enable_hot_reload and self.hot_reloader is not None
    
    def force_reload(self) -> ReloadResult:
        """Force configuration reload."""
        if not self.hot_reloader:
            return ReloadResult(
                status=ReloadStatus.DISABLED,
                config=self.current_config,
                validation_result=None,
                error_message="Hot-reload disabled",
                reload_time_ms=0,
                file_path=""
            )
        
        # Reload from first available config file
        for config_path in self.config_paths:
            if Path(config_path).exists():
                return self.hot_reloader._perform_hot_reload(config_path)
        
        return ReloadResult(
            status=ReloadStatus.LOAD_FAILED,
            config=self.current_config,
            validation_result=None,
            error_message="No configuration files found",
            reload_time_ms=0,
            file_path=""
        )
    
    def rollback_config(self) -> ReloadResult:
        """Rollback to previous configuration."""
        if not self.hot_reloader:
            return ReloadResult(
                status=ReloadStatus.DISABLED,
                config=self.current_config,
                validation_result=None,
                error_message="Hot-reload disabled",
                reload_time_ms=0,
                file_path=""
            )
        
        return self.hot_reloader.rollback_config()
    
    def add_change_callback(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Add callback for configuration change events."""
        self.change_callbacks.append(callback)
        self.logger.info(f"Added config change callback: {callback.__name__}")
    
    def remove_change_callback(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Remove configuration change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            self.logger.info(f"Removed config change callback: {callback.__name__}")
    
    def _on_config_reload(self, result: ReloadResult) -> None:
        """Handle configuration reload events."""
        if result.is_success():
            # Detect and notify configuration changes
            old_config = self.current_config
            new_config = result.config
            
            with self.config_lock:
                self.current_config = new_config
            
            if old_config:
                changes = self._detect_config_changes(old_config, new_config)
                for change in changes:
                    self._notify_change_callbacks(change)
            
            self.logger.info(f"Configuration reloaded successfully: {result.get_summary()}")
            
            if result.validation_result and result.validation_result.warnings:
                for warning in result.validation_result.warnings:
                    self.logger.warning(f"Config warning: {warning.get('message', 'Unknown warning')}")
        
        else:
            self.logger.error(f"Configuration reload failed: {result.get_summary()}")
            
            # Keep current config on failure (don't wipe limits)
            if result.validation_result:
                for error in result.validation_result.errors:
                    self.logger.error(f"Config error: {error.get('message', 'Unknown error')}")
    
    def _detect_config_changes(self, old_config: Dict[str, Any], 
                              new_config: Dict[str, Any]) -> List[ConfigChangeEvent]:
        """Detect changes between old and new configurations."""
        changes = []
        current_time = time.time()
        
        # Check active policy change
        old_active = old_config.get('active_policy')
        new_active = new_config.get('active_policy')
        if old_active != new_active:
            changes.append(ConfigChangeEvent(
                change_type=ConfigChangeType.POLICY_CHANGE,
                old_value=old_active,
                new_value=new_active,
                path='active_policy',
                timestamp=current_time
            ))
        
        # Check enforcement mode change
        old_mode = old_config.get('enforcement', {}).get('mode')
        new_mode = new_config.get('enforcement', {}).get('mode')
        if old_mode != new_mode:
            changes.append(ConfigChangeEvent(
                change_type=ConfigChangeType.ENFORCEMENT_CHANGE,
                old_value=old_mode,
                new_value=new_mode,
                path='enforcement.mode',
                timestamp=current_time
            ))
        
        # Check VaR limits changes
        old_var_limits = old_config.get('enforcement', {}).get('var_limits', {})
        new_var_limits = new_config.get('enforcement', {}).get('var_limits', {})
        
        for limit_name in set(old_var_limits.keys()) | set(new_var_limits.keys()):
            old_val = old_var_limits.get(limit_name)
            new_val = new_var_limits.get(limit_name)
            
            if old_val != new_val:
                changes.append(ConfigChangeEvent(
                    change_type=ConfigChangeType.LIMITS_CHANGE,
                    old_value=old_val,
                    new_value=new_val,
                    path=f'enforcement.var_limits.{limit_name}',
                    timestamp=current_time
                ))
        
        return changes
    
    def _notify_change_callbacks(self, change: ConfigChangeEvent) -> None:
        """Notify all registered callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(change)
            except Exception as e:
                self.logger.error(f"Change callback {callback.__name__} failed: {e}")
    
    def _get_safe_defaults(self) -> Dict[str, Any]:
        """Get safe default configuration to prevent system lockup."""
        return {
            "policies": {
                "safe_default": {
                    "calculators": {
                        "var": {
                            "enabled": True,
                            "config": {
                                "confidence_levels": [0.95, 0.99, 0.999],
                                "window_days": 250,
                                "method": "parametric",
                                "distribution": "normal",
                                "min_observations": 30
                            }
                        },
                        "stress_test": {
                            "enabled": True,
                            "config": {
                                "scenarios": ["historical", "monte_carlo"],
                                "confidence_levels": [0.95, 0.99, 0.999],
                                "monte_carlo_runs": 10000
                            }
                        }
                    },
                    "rules": [
                        {
                            "rule_id": "emergency_var_limit",
                            "rule_name": "Emergency VaR Limit",
                            "rule_type": "threshold",
                            "field": "var_99",
                            "threshold": 50000,  # Conservative $50k limit
                            "operator": "gt",
                            "action": "halt",
                            "severity": "critical",
                            "monitoring_mode": False,
                            "enforcement_enabled": True
                        }
                    ]
                }
            },
            "active_policy": "safe_default",
            "enforcement": {
                "mode": "full",  # Full enforcement for safety
                "false_positive_threshold_per_week": 0.5,
                "var_limits": {
                    "var_95_limit": 25000,   # Conservative limits
                    "var_99_limit": 50000,
                    "var_999_limit": 100000
                },
                "stress_limits": {
                    "max_stress_loss": 100000,  # Conservative $100k limit
                    "max_scenario_failures": 1,
                    "max_tail_ratio": 1.2
                },
                "audit": {
                    "enabled": True,
                    "log_directory": "logs/risk_audit",
                    "async_logging": True
                },
                "metrics": {
                    "enabled": True,
                    "namespace": "risk",
                    "subsystem": "management"
                }
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        stats = {
            'config_access_count': self.config_access_count,
            'last_access_time': self.last_access_time,
            'initialization_time': self.initialization_time,
            'hot_reload_enabled': self.is_hot_reload_enabled(),
            'has_current_config': self.current_config is not None,
            'config_paths': self.config_paths
        }
        
        if self.hot_reloader:
            stats['hot_reload_stats'] = self.hot_reloader.get_stats()
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def create_risk_config_manager(config_paths: List[str], 
                              validator: Optional[RiskConfigValidator] = None,
                              enable_hot_reload: bool = True) -> RiskConfigManager:
    """Factory function to create RiskConfigManager."""
    return RiskConfigManager(config_paths, validator, enable_hot_reload)


# Example usage and testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration manager
    config_path = Path("test_risk_config.yaml")
    
    # Create test config file
    test_config = {
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
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    print(f"Created test config: {config_path}")
    
    # Test configuration manager
    def config_change_callback(change: ConfigChangeEvent):
        print(f"Config change: {change.get_summary()}")
    
    try:
        with create_risk_config_manager([str(config_path)]) as config_manager:
            config_manager.add_change_callback(config_change_callback)
            
            # Test configuration access
            print(f"Active policy: {config_manager.get_active_policy()}")
            print(f"VaR limits: {config_manager.get_var_limits()}")
            print(f"Enforcement config: {config_manager.get_enforcement_config()}")
            
            # Test configuration change
            print("\nModifying config file...")
            test_config['enforcement']['mode'] = 'full'
            test_config['enforcement']['var_limits']['var_95_limit'] = 150000
            
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            time.sleep(2)  # Wait for hot-reload
            
            print(f"Updated VaR limits: {config_manager.get_var_limits()}")
            
            # Show stats
            print(f"\nConfig manager stats: {json.dumps(config_manager.get_stats(), indent=2, default=str)}")
            
            print("\nPress Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()
            print(f"Cleaned up test config: {config_path}")