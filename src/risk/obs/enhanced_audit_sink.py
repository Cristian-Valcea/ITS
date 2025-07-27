import json
import datetime
import pathlib
import os
import logging
from typing import Union, Optional
from ..event_types import RiskEvent

class EnhancedJsonAuditSink:
    """
    Production-ready audit sink that writes each RiskEvent as a JSON line.
    
    Features:
    - Configurable via environment variables
    - Automatic directory creation with proper permissions
    - Fallback to stdout if file writing fails
    - Volume mount validation for containerized deployments
    - Graceful error handling to prevent audit failures from breaking trading
    - Safe for moderate event rates (< 50k events/s) with sync writes
    """
    
    def __init__(self, path: Optional[Union[str, pathlib.Path]] = None):
        """
        Initialize audit sink with robust path resolution.
        
        Args:
            path: Audit log path. If None, uses environment variable or default.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Resolve audit path with environment variable support
        if path is None:
            path = os.getenv('RISK_AUDIT_LOG_PATH', 'logs/risk_audit.jsonl')
        
        self._path = pathlib.Path(path)
        self._fallback_to_stdout = False
        
        # Initialize file handle with robust error handling
        self._init_file_handle()
        
        self.logger.info(f"EnhancedJsonAuditSink initialized: {self._path}")
        if self._fallback_to_stdout:
            self.logger.warning("Audit sink falling back to stdout due to file access issues")
    
    def _init_file_handle(self):
        """Initialize file handle with comprehensive error handling."""
        try:
            # Create directory with proper permissions
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            
            # Validate directory is writable
            if not os.access(self._path.parent, os.W_OK):
                raise PermissionError(f"Directory {self._path.parent} is not writable")
            
            # Check if running in container and validate volume mount
            self._validate_container_volume_mount()
            
            # Open file with line buffering for immediate writes
            self._fh = self._path.open("a", buffering=1, encoding="utf-8")
            
            # Test write to ensure file is accessible
            test_pos = self._fh.tell()
            self._fh.write("")
            self._fh.flush()
            
            self.logger.info(f"Audit log file initialized: {self._path.absolute()}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit log file {self._path}: {e}")
            self.logger.warning("Falling back to stdout for audit logging")
            self._fallback_to_stdout = True
            self._fh = None
    
    def _validate_container_volume_mount(self):
        """Validate volume mount in containerized environments."""
        # Check if running in container
        if os.path.exists('/.dockerenv') or os.getenv('KUBERNETES_SERVICE_HOST'):
            # Check if logs directory is on a mounted volume
            logs_dir = self._path.parent.absolute()
            
            # Get mount info
            try:
                import subprocess
                result = subprocess.run(['df', str(logs_dir)], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    mount_info = result.stdout.strip().split('\n')[-1]
                    if 'tmpfs' in mount_info or logs_dir.as_posix().startswith('/tmp'):
                        self.logger.warning(
                            f"⚠️  AUDIT LOG RISK: {logs_dir} appears to be on tmpfs or /tmp. "
                            f"Logs will be lost on container restart! "
                            f"Mount a persistent volume to {logs_dir}"
                        )
                    else:
                        self.logger.info(f"✅ Audit logs on persistent storage: {mount_info}")
                        
            except Exception as e:
                self.logger.debug(f"Could not validate volume mount: {e}")
    
    def write(self, event: RiskEvent) -> None:
        """
        Write event to audit log with graceful error handling.
        
        Args:
            event: RiskEvent to audit
        """
        try:
            # Create audit payload
            payload = {
                "ts": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "event_id": event.event_id,
                "event_type": event.event_type.name,
                "priority": event.priority.name,
                "source": event.source,
                "data": event.data,
                "metadata": event.metadata,
                "timestamp_ns": event.timestamp_ns
            }
            
            # Serialize to JSON
            json_line = json.dumps(payload, separators=(",", ":")) + "\n"
            
            # Write to file or stdout
            if self._fallback_to_stdout:
                print(f"AUDIT: {json_line.rstrip()}")
            else:
                self._fh.write(json_line)
                
        except Exception as e:
            # Never let audit failures break trading
            self.logger.error(f"Audit write failed for event {event.event_id}: {e}")
            try:
                # Emergency fallback to stdout
                print(f"AUDIT_EMERGENCY: {event.event_type.name} {event.event_id}")
            except:
                pass  # Absolute last resort - don't break trading
    
    def flush(self) -> None:
        """Flush audit log to ensure writes are persisted."""
        try:
            if not self._fallback_to_stdout and self._fh:
                self._fh.flush()
                os.fsync(self._fh.fileno())
        except Exception as e:
            self.logger.error(f"Audit flush failed: {e}")
    
    def close(self) -> None:
        """Close audit log file handle."""
        try:
            if not self._fallback_to_stdout and self._fh:
                self._fh.close()
        except Exception as e:
            self.logger.error(f"Audit close failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backward compatibility alias
JsonAuditSink = EnhancedJsonAuditSink