"""
DuckDB Connection Manager

Handles DuckDB connections with proper concurrency control to prevent
write-write conflicts during training and evaluation phases.
"""

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any
import duckdb
import logging


class DuckDBConnectionManager:
    """
    Manages DuckDB connections with proper locking to prevent write-write conflicts.
    
    Features:
    - Thread-safe connection pooling
    - Read-only vs read-write connection modes
    - Automatic retry on write conflicts
    - Connection cleanup on phase transitions
    """
    
    def __init__(self):
        self._connections: Dict[str, duckdb.DuckDBPyConnection] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._connection_modes: Dict[str, str] = {}  # 'r' or 'rw'
        self._global_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def get_connection(self, db_path: str, mode: str = 'r', max_retries: int = 3):
        """
        Get a DuckDB connection with proper locking.
        
        Args:
            db_path: Path to DuckDB file
            mode: 'r' for read-only, 'rw' for read-write
            max_retries: Number of retries on write conflicts
        """
        db_path = str(Path(db_path).resolve())
        
        with self._global_lock:
            if db_path not in self._locks:
                self._locks[db_path] = threading.RLock()
        
        # Acquire database-specific lock
        with self._locks[db_path]:
            conn = None
            try:
                conn = self._get_or_create_connection(db_path, mode, max_retries)
                yield conn
            finally:
                # Don't close connection here - keep it in pool
                pass
    
    def _get_or_create_connection(self, db_path: str, mode: str, max_retries: int) -> duckdb.DuckDBPyConnection:
        """Get existing connection or create new one with retry logic."""
        
        # Check if we need to close existing connection due to mode change
        if db_path in self._connections and self._connection_modes.get(db_path) != mode:
            self.logger.info(f"Closing DuckDB connection for mode change: {mode}")
            self._close_connection(db_path)
        
        # Create new connection if needed
        if db_path not in self._connections:
            for attempt in range(max_retries):
                try:
                    if mode == 'r':
                        # Read-only connection
                        conn = duckdb.connect(db_path, read_only=True)
                    else:
                        # Read-write connection
                        conn = duckdb.connect(db_path)
                    
                    self._connections[db_path] = conn
                    self._connection_modes[db_path] = mode
                    self.logger.debug(f"Created DuckDB connection: {db_path} (mode: {mode})")
                    break
                    
                except Exception as e:
                    if "write-write conflict" in str(e) and attempt < max_retries - 1:
                        wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"DuckDB write conflict, retrying in {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise
        
        return self._connections[db_path]
    
    def _close_connection(self, db_path: str):
        """Close and remove connection from pool."""
        if db_path in self._connections:
            try:
                self._connections[db_path].close()
            except Exception as e:
                self.logger.warning(f"Error closing DuckDB connection: {e}")
            finally:
                del self._connections[db_path]
                if db_path in self._connection_modes:
                    del self._connection_modes[db_path]
    
    def close_all_connections(self):
        """Close all connections - call this between training phases."""
        with self._global_lock:
            for db_path in list(self._connections.keys()):
                self._close_connection(db_path)
            self.logger.info("Closed all DuckDB connections")
    
    def close_write_connections(self):
        """Close only write connections - useful before starting read-only evaluation."""
        with self._global_lock:
            write_paths = [
                path for path, mode in self._connection_modes.items() 
                if mode == 'rw'
            ]
            for db_path in write_paths:
                self._close_connection(db_path)
            if write_paths:
                self.logger.info(f"Closed {len(write_paths)} write connections")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about current connections."""
        with self._global_lock:
            return {
                'active_connections': len(self._connections),
                'connections': {
                    path: {
                        'mode': self._connection_modes.get(path, 'unknown'),
                        'active': path in self._connections
                    }
                    for path in self._connection_modes.keys()
                }
            }


# Global connection manager instance
_connection_manager = DuckDBConnectionManager()


def get_duckdb_connection(db_path: str, mode: str = 'r', max_retries: int = 3):
    """
    Get a managed DuckDB connection.
    
    Args:
        db_path: Path to DuckDB file
        mode: 'r' for read-only, 'rw' for read-write
        max_retries: Number of retries on conflicts
    
    Returns:
        Context manager yielding DuckDB connection
    """
    return _connection_manager.get_connection(db_path, mode, max_retries)


def close_all_duckdb_connections():
    """Close all DuckDB connections - call between training phases."""
    _connection_manager.close_all_connections()


def close_write_duckdb_connections():
    """Close write connections before read-only operations."""
    _connection_manager.close_write_connections()


def get_duckdb_connection_info():
    """Get information about active DuckDB connections."""
    return _connection_manager.get_connection_info()


class DuckDBPhaseManager:
    """
    Context manager for DuckDB operations during different pipeline phases.
    Ensures proper connection cleanup between training and evaluation.
    """
    
    def __init__(self, phase_name: str, cleanup_writes: bool = False):
        self.phase_name = phase_name
        self.cleanup_writes = cleanup_writes
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.logger.info(f"🚀 Starting {self.phase_name} phase...")
        if self.cleanup_writes:
            self.logger.info(f"🔒 Closing write connections before {self.phase_name}...")
            close_write_duckdb_connections()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"✅ {self.phase_name} phase completed successfully")
        else:
            self.logger.error(f"❌ {self.phase_name} phase failed: {exc_val}")
        
        # Always cleanup connections after phase completion
        self.logger.info(f"🧹 Cleaning up DuckDB connections after {self.phase_name}...")
        close_all_duckdb_connections()


def training_phase():
    """Context manager for training phase with proper DuckDB cleanup."""
    return DuckDBPhaseManager("TRAINING", cleanup_writes=False)


def evaluation_phase():
    """Context manager for evaluation phase with write connection cleanup."""
    return DuckDBPhaseManager("EVALUATION", cleanup_writes=True)