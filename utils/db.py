#!/usr/bin/env python3
"""
Drop-in DuckDB connection management utilities.

Fixes the Windows file lock issue by properly closing DuckDB connections.
"""

import gc
import duckdb
import logging
from typing import List, Optional

def close_write_duckdb_connections() -> List[str]:
    """
    Scan all live objects for DuckDB connections and close them.
    
    Since DuckDB 1.3.1 doesn't reliably expose read_only status,
    we close ALL connections to ensure Windows file locks are released.
    
    Returns:
        List of connection info that were closed (for logging)
    """
    logger = logging.getLogger(__name__)
    released = []
    errors = 0
    
    for obj in gc.get_objects():
        if isinstance(obj, duckdb.DuckDBPyConnection):
            try:
                # Get connection info before closing
                conn_info = f"DuckDB connection {id(obj)}"
                
                # Close the connection (releases Windows file lock)
                obj.close()
                released.append(conn_info)
                
            except Exception as e:
                logger.debug(f"Failed to close DuckDB connection: {e}")
                errors += 1
    
    if released:
        logger.info(f"Closed {len(released)} DuckDB connections to release file locks ({errors} errors)")
    
    return released

def close_duckdb_connections_safe() -> int:
    """
    Safe version that returns count instead of connection info.
    
    Returns:
        Number of connections closed
    """
    closed_connections = close_write_duckdb_connections()
    return len(closed_connections)

def ensure_duckdb_cleanup():
    """
    Ensure all DuckDB connections are cleaned up.
    
    Call this before switching between training and evaluation phases
    to prevent Windows file lock conflicts.
    """
    import time
    
    logger = logging.getLogger(__name__)
    
    # Close connections
    closed_count = close_duckdb_connections_safe()
    
    if closed_count > 0:
        # Give Windows a moment to release file locks
        time.sleep(0.1)
        logger.info(f"DuckDB cleanup complete: {closed_count} connections closed")
    
    # Force garbage collection to ensure cleanup
    gc.collect()

# Backward compatibility aliases
close_all_duckdb_write_connections = close_write_duckdb_connections
cleanup_duckdb_connections = ensure_duckdb_cleanup