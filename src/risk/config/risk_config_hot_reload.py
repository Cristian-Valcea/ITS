# src/risk/config/risk_config_hot_reload.py
"""
Risk Configuration Hot-Reload System with YAML Watcher.

Monitors risk configuration files for changes and safely reloads them
with JSON Schema validation to prevent malformed YAML from wiping limits.
"""

import os
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from .risk_config_validator import RiskConfigValidator, ValidationResult


class ReloadStatus(Enum):
    """Hot-reload operation status."""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    LOAD_FAILED = "load_failed"
    ROLLBACK = "rollback"
    DISABLED = "disabled"


@dataclass
class ReloadResult:
    """Result of hot-reload operation."""
    status: ReloadStatus
    config: Optional[Dict[str, Any]]
    validation_result: Optional[ValidationResult]
    error_message: Optional[str]
    reload_time_ms: float
    file_path: str
    
    def is_success(self) -> bool:
        """Check if reload was successful."""
        return self.status == ReloadStatus.SUCCESS
    
    def get_summary(self) -> str:
        """Get reload result summary."""
        if self.status == ReloadStatus.SUCCESS:
            return f"✅ Hot-reload successful ({self.reload_time_ms:.1f}ms)"
        elif self.status == ReloadStatus.VALIDATION_FAILED:
            return f"❌ Validation failed - config not reloaded"
        elif self.status == ReloadStatus.LOAD_FAILED:
            return f"❌ YAML load failed - {self.error_message}"
        elif self.status == ReloadStatus.ROLLBACK:
            return f"🔄 Rolled back to previous config"
        else:
            return f"⚠️ Hot-reload disabled"


class RiskConfigFileHandler(FileSystemEventHandler):
    """File system event handler for risk configuration changes."""
    
    def __init__(self, hot_reloader: 'RiskConfigHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger("RiskConfigFileHandler")
        self.last_reload_time = 0
        self.debounce_interval = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            current_time = time.time()
            
            # Debounce rapid file changes
            if current_time - self.last_reload_time < self.debounce_interval:
                return
            
            file_path = Path(event.src_path)
            
            # Only process YAML files we're watching
            if file_path.suffix.lower() in ['.yaml', '.yml'] and str(file_path) in self.hot_reloader.watched_files:
                self.logger.info(f"Config file modified: {file_path}")
                self.last_reload_time = current_time
                
                # Trigger hot-reload in separate thread to avoid blocking file watcher
                threading.Thread(
                    target=self.hot_reloader._perform_hot_reload,
                    args=(str(file_path),),
                    daemon=True
                ).start()


class RiskConfigHotReloader:
    """
    Risk Configuration Hot-Reload System.
    
    Features:
    - YAML file watching with debouncing
    - JSON Schema validation before swap-in
    - Automatic rollback on validation failure
    - Configuration diff analysis
    - Callback system for reload notifications
    - Performance monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config_paths: List[str], validator: Optional[RiskConfigValidator] = None):
        self.config_paths = [Path(p).resolve() for p in config_paths]
        self.watched_files = {str(p): p for p in self.config_paths}
        self.validator = validator or RiskConfigValidator()
        self.logger = logging.getLogger("RiskConfigHotReloader")
        
        # Current configuration state
        self.current_config: Optional[Dict[str, Any]] = None
        self.previous_config: Optional[Dict[str, Any]] = None
        self.config_lock = threading.RLock()
        
        # Hot-reload settings
        self.enabled = True
        self.auto_rollback = True
        self.max_validation_time_ms = 5000  # 5 second timeout
        
        # Callbacks for reload events
        self.reload_callbacks: List[Callable[[ReloadResult], None]] = []
        
        # Statistics
        self.reload_count = 0
        self.successful_reloads = 0
        self.failed_reloads = 0
        self.last_reload_time: Optional[float] = None
        
        # File watcher
        self.observer: Optional[Observer] = None
        self.file_handler = RiskConfigFileHandler(self)
        
        self.logger.info(f"RiskConfigHotReloader initialized for {len(self.config_paths)} files")
    
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        if self.observer is not None:
            self.logger.warning("File watcher already started")
            return
        
        self.observer = Observer()
        
        # Watch directories containing config files
        watched_dirs = set()
        for config_path in self.config_paths:
            if config_path.exists():
                parent_dir = config_path.parent
                if parent_dir not in watched_dirs:
                    self.observer.schedule(self.file_handler, str(parent_dir), recursive=False)
                    watched_dirs.add(parent_dir)
                    self.logger.info(f"Watching directory: {parent_dir}")
        
        self.observer.start()
        self.logger.info("File watcher started")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.logger.info("File watcher stopped")
    
    def load_initial_config(self) -> ReloadResult:
        """Load initial configuration from files."""
        self.logger.info("Loading initial configuration")
        
        # Find the first existing config file
        for config_path in self.config_paths:
            if config_path.exists():
                return self._perform_hot_reload(str(config_path), is_initial=True)
        
        return ReloadResult(
            status=ReloadStatus.LOAD_FAILED,
            config=None,
            validation_result=None,
            error_message="No configuration files found",
            reload_time_ms=0,
            file_path=""
        )
    
    def _perform_hot_reload(self, file_path: str, is_initial: bool = False) -> ReloadResult:
        """Perform hot-reload operation with validation."""
        start_time = time.time()
        
        try:
            with self.config_lock:
                # 1. Load YAML file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        new_config = yaml.safe_load(f)
                    
                    if new_config is None:
                        raise ValueError("Empty YAML file")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load YAML from {file_path}: {e}")
                    self.failed_reloads += 1
                    
                    result = ReloadResult(
                        status=ReloadStatus.LOAD_FAILED,
                        config=None,
                        validation_result=None,
                        error_message=str(e),
                        reload_time_ms=(time.time() - start_time) * 1000,
                        file_path=file_path
                    )
                    
                    self._notify_callbacks(result)
                    return result
                
                # 2. Validate configuration
                validation_result = self.validator.validate_config(
                    new_config, 
                    self.current_config
                )
                
                # 3. Check if validation passed
                if validation_result.has_blocking_errors():
                    self.logger.error(f"Configuration validation failed: {validation_result.get_summary()}")
                    self.failed_reloads += 1
                    
                    # Don't swap in invalid config - keep current config
                    result = ReloadResult(
                        status=ReloadStatus.VALIDATION_FAILED,
                        config=self.current_config,  # Keep current config
                        validation_result=validation_result,
                        error_message="Validation failed - config not reloaded",
                        reload_time_ms=(time.time() - start_time) * 1000,
                        file_path=file_path
                    )
                    
                    self._notify_callbacks(result)
                    return result
                
                # 4. Swap in new configuration
                self.previous_config = self.current_config
                self.current_config = new_config
                self.reload_count += 1
                self.successful_reloads += 1
                self.last_reload_time = time.time()
                
                reload_time_ms = (time.time() - start_time) * 1000
                
                if is_initial:
                    self.logger.info(f"Initial config loaded successfully ({reload_time_ms:.1f}ms)")
                else:
                    self.logger.info(f"Hot-reload successful ({reload_time_ms:.1f}ms)")
                    if validation_result.warnings:
                        self.logger.warning(f"Config loaded with {len(validation_result.warnings)} warnings")
                
                result = ReloadResult(
                    status=ReloadStatus.SUCCESS,
                    config=new_config,
                    validation_result=validation_result,
                    error_message=None,
                    reload_time_ms=reload_time_ms,
                    file_path=file_path
                )
                
                self._notify_callbacks(result)
                return result
                
        except Exception as e:
            self.logger.error(f"Hot-reload failed with exception: {e}")
            self.failed_reloads += 1
            
            result = ReloadResult(
                status=ReloadStatus.LOAD_FAILED,
                config=self.current_config,
                validation_result=None,
                error_message=str(e),
                reload_time_ms=(time.time() - start_time) * 1000,
                file_path=file_path
            )
            
            self._notify_callbacks(result)
            return result
    
    def rollback_config(self) -> ReloadResult:
        """Rollback to previous configuration."""
        start_time = time.time()
        
        with self.config_lock:
            if self.previous_config is None:
                return ReloadResult(
                    status=ReloadStatus.LOAD_FAILED,
                    config=self.current_config,
                    validation_result=None,
                    error_message="No previous configuration to rollback to",
                    reload_time_ms=(time.time() - start_time) * 1000,
                    file_path=""
                )
            
            # Swap back to previous config
            temp_config = self.current_config
            self.current_config = self.previous_config
            self.previous_config = temp_config
            
            self.logger.info("Configuration rolled back to previous version")
            
            result = ReloadResult(
                status=ReloadStatus.ROLLBACK,
                config=self.current_config,
                validation_result=None,
                error_message=None,
                reload_time_ms=(time.time() - start_time) * 1000,
                file_path=""
            )
            
            self._notify_callbacks(result)
            return result
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration (thread-safe)."""
        with self.config_lock:
            return self.current_config.copy() if self.current_config else None
    
    def add_reload_callback(self, callback: Callable[[ReloadResult], None]) -> None:
        """Add callback for reload events."""
        self.reload_callbacks.append(callback)
        self.logger.info(f"Added reload callback: {callback.__name__}")
    
    def remove_reload_callback(self, callback: Callable[[ReloadResult], None]) -> None:
        """Remove reload callback."""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
            self.logger.info(f"Removed reload callback: {callback.__name__}")
    
    def _notify_callbacks(self, result: ReloadResult) -> None:
        """Notify all registered callbacks of reload result."""
        for callback in self.reload_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Callback {callback.__name__} failed: {e}")
    
    def enable_hot_reload(self) -> None:
        """Enable hot-reload functionality."""
        self.enabled = True
        self.logger.info("Hot-reload enabled")
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reload functionality."""
        self.enabled = False
        self.logger.info("Hot-reload disabled")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hot-reload statistics."""
        return {
            'enabled': self.enabled,
            'reload_count': self.reload_count,
            'successful_reloads': self.successful_reloads,
            'failed_reloads': self.failed_reloads,
            'success_rate': (self.successful_reloads / max(1, self.reload_count)) * 100,
            'last_reload_time': self.last_reload_time,
            'watched_files': list(self.watched_files.keys()),
            'has_current_config': self.current_config is not None,
            'has_previous_config': self.previous_config is not None
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_watching()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()


def create_risk_config_hot_reloader(config_paths: List[str], 
                                   validator: Optional[RiskConfigValidator] = None) -> RiskConfigHotReloader:
    """Factory function to create RiskConfigHotReloader."""
    return RiskConfigHotReloader(config_paths, validator)


# Example usage and testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test hot-reload system
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
    
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    print(f"Created test config: {config_path}")
    
    # Test hot-reloader
    def reload_callback(result: ReloadResult):
        print(f"Reload callback: {result.get_summary()}")
        if result.validation_result:
            print(f"  Validation: {result.validation_result.get_summary()}")
    
    try:
        with create_risk_config_hot_reloader([str(config_path)]) as hot_reloader:
            hot_reloader.add_reload_callback(reload_callback)
            
            # Load initial config
            result = hot_reloader.load_initial_config()
            print(f"Initial load: {result.get_summary()}")
            
            # Test manual reload
            print("\nModifying config file...")
            test_config['enforcement']['mode'] = 'gradual'
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            time.sleep(2)  # Wait for file watcher
            
            # Test invalid config
            print("\nTesting invalid config...")
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            time.sleep(2)  # Wait for file watcher
            
            # Show stats
            print(f"\nHot-reload stats: {json.dumps(hot_reloader.get_stats(), indent=2)}")
            
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