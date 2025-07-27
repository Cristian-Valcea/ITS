"""
Database utility module for DuckDB connections with proper concurrency control.

This module provides a clean interface for DuckDB connections with:
- File-based locking to prevent write conflicts
- Read-only mode support
- Cache control options
- Proper connection lifecycle management
"""

import os
import duckdb
from contextlib import contextmanager
from filelock import FileLock
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default database path - can be overridden via environment variable
DEFAULT_DB_PATH = os.path.expanduser("~/.feature_cache/manifest.duckdb")
DB_PATH = os.getenv("DUCKDB_FILE", DEFAULT_DB_PATH)

# Ensure directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def get_conn(read_only: bool = False, disable_cache: bool = False, timeout: int = 30):
    """
    Get a DuckDB connection with proper concurrency control.
    
    Args:
        read_only: If True, open in read-only mode
        disable_cache: If True, disable DuckDB object cache
        timeout: Timeout in seconds for acquiring file lock
        
    Yields:
        DuckDB connection object
    """
    config = {'access_mode': 'READ_ONLY'} if read_only else {}
    
    if read_only:
        logger.debug(f"Opening DuckDB in READ_ONLY mode: {DB_PATH}")
    else:
        logger.debug(f"Opening DuckDB in READ_WRITE mode: {DB_PATH}")
    
    # Use file lock for write operations to prevent conflicts
    if not read_only:
        lock_path = f"{DB_PATH}.lock"
        with FileLock(lock_path, timeout=timeout):
            conn = duckdb.connect(DB_PATH, config=config)
            if disable_cache:
                conn.execute("SET enable_object_cache=false;")
            try:
                yield conn
            finally:
                conn.close()
    else:
        # Read-only operations don't need file locks
        conn = duckdb.connect(DB_PATH, config=config)
        if disable_cache:
            conn.execute("SET enable_object_cache=false;")
        try:
            yield conn
        finally:
            conn.close()


@contextmanager 
def get_write_conn(disable_cache: bool = False, timeout: int = 30):
    """
    Get a write connection with explicit transaction control.
    
    Args:
        disable_cache: If True, disable DuckDB object cache
        timeout: Timeout in seconds for acquiring file lock
        
    Yields:
        DuckDB connection object with transaction started
    """
    with get_conn(read_only=False, disable_cache=disable_cache, timeout=timeout) as conn:
        conn.execute("BEGIN TRANSACTION;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def health_check() -> bool:
    """
    Perform a health check on the DuckDB database.
    
    Returns:
        True if database is accessible, False otherwise
    """
    try:
        with get_conn(read_only=True, disable_cache=True) as conn:
            # Simple query to verify database accessibility
            result = conn.execute("PRAGMA database_list;").fetchall()
            logger.info(f"DuckDB health check passed: {len(result)} database(s) accessible")
            return True
    except Exception as e:
        logger.error(f"DuckDB health check failed: {e}")
        return False


def get_db_info() -> dict:
    """
    Get database information for monitoring.
    
    Returns:
        Dictionary with database statistics
    """
    try:
        with get_conn(read_only=True, disable_cache=True) as conn:
            # Get database size
            size_result = conn.execute("PRAGMA database_size;").fetchone()
            db_size = size_result[0] if size_result else 0
            
            # Get table count
            tables_result = conn.execute("SHOW TABLES;").fetchall()
            table_count = len(tables_result)
            
            return {
                "db_path": DB_PATH,
                "db_size_bytes": db_size,
                "table_count": table_count,
                "accessible": True
            }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {
            "db_path": DB_PATH,
            "accessible": False,
            "error": str(e)
        }