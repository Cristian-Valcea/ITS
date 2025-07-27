"""
PostgreSQL advisory lock utilities for FeatureStore manifest operations.

Advisory locks provide lightweight, fast locking without the overhead of row-level
locks. They are automatically released when the transaction ends, making them
perfect for eliminating manifest INSERT contention from parallel training workers.
"""

import hashlib
import logging
from contextlib import contextmanager
from typing import Generator

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

logger = logging.getLogger(__name__)


def symbol_to_lock_key(symbol: str) -> int:
    """
    Convert symbol to a consistent 64-bit lock key.
    
    Uses SHA-256 hash of symbol to generate a deterministic lock key
    that distributes evenly across the lock space.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "GOOGL")
        
    Returns:
        64-bit integer lock key
    """
    # Generate SHA-256 hash of symbol
    hash_bytes = hashlib.sha256(symbol.encode('utf-8')).digest()
    
    # Take first 8 bytes and convert to signed 64-bit integer
    # PostgreSQL advisory locks use bigint (signed 64-bit)
    lock_key = int.from_bytes(hash_bytes[:8], byteorder='big', signed=True)
    
    return lock_key


@contextmanager
def advisory_lock(conn: psycopg2.extensions.connection, lock_key: int) -> Generator[None, None, None]:
    """
    Acquire PostgreSQL advisory lock for the duration of the current transaction.
    
    This is a blocking lock - if another process holds the same lock key,
    this will wait until the lock is released. The lock is automatically
    released when the transaction commits or rolls back.
    
    Args:
        conn: PostgreSQL connection (must be in a transaction)
        lock_key: 64-bit integer lock key
        
    Yields:
        None (lock is held for the duration of the context)
        
    Example:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                lock_key = symbol_to_lock_key("AAPL")
                with advisory_lock(conn, lock_key):
                    # Critical section - only one process can execute this
                    # for the same symbol at a time
                    cur.execute("INSERT INTO manifest ...")
                cur.execute("COMMIT")
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for advisory locks. "
            "Install with: pip install psycopg2-binary"
        )
    
    try:
        with conn.cursor() as cur:
            # Acquire transaction-scoped advisory lock
            # This will block until the lock is available
            logger.debug(f"Acquiring advisory lock {lock_key}")
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            
            logger.debug(f"Advisory lock {lock_key} acquired")
            yield
            
            # Lock is automatically released when transaction ends
            logger.debug(f"Advisory lock {lock_key} will be released on transaction end")
            
    except Exception as e:
        logger.error(f"Error with advisory lock {lock_key}: {e}")
        raise


@contextmanager  
def try_advisory_lock(conn: psycopg2.extensions.connection, lock_key: int, 
                     timeout_ms: int = 5000) -> Generator[bool, None, None]:
    """
    Try to acquire PostgreSQL advisory lock with timeout.
    
    This is a non-blocking version that returns immediately if the lock
    cannot be acquired within the timeout period.
    
    Args:
        conn: PostgreSQL connection (must be in a transaction)
        lock_key: 64-bit integer lock key  
        timeout_ms: Timeout in milliseconds (default: 5000ms = 5s)
        
    Yields:
        bool: True if lock was acquired, False if timeout occurred
        
    Example:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                lock_key = symbol_to_lock_key("AAPL")
                with try_advisory_lock(conn, lock_key, timeout_ms=1000) as acquired:
                    if acquired:
                        # Got the lock - proceed with critical section
                        cur.execute("INSERT INTO manifest ...")
                    else:
                        # Timeout - handle gracefully
                        logger.warning(f"Could not acquire lock for {lock_key}")
                cur.execute("COMMIT")
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for advisory locks. "
            "Install with: pip install psycopg2-binary"
        )
    
    acquired = False
    
    try:
        with conn.cursor() as cur:
            # Set lock timeout
            cur.execute("SET lock_timeout = %s", (timeout_ms,))
            
            try:
                # Try to acquire transaction-scoped advisory lock
                logger.debug(f"Trying to acquire advisory lock {lock_key} (timeout: {timeout_ms}ms)")
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
                acquired = True
                logger.debug(f"Advisory lock {lock_key} acquired")
                
            except psycopg2.OperationalError as e:
                if "lock_timeout" in str(e).lower():
                    logger.debug(f"Advisory lock {lock_key} timeout after {timeout_ms}ms")
                    acquired = False
                else:
                    raise
            
            yield acquired
            
            if acquired:
                logger.debug(f"Advisory lock {lock_key} will be released on transaction end")
            
    except Exception as e:
        logger.error(f"Error with try_advisory_lock {lock_key}: {e}")
        raise
    
    finally:
        # Reset lock timeout to default
        try:
            with conn.cursor() as cur:
                cur.execute("SET lock_timeout = DEFAULT")
        except:
            pass  # Ignore cleanup errors


def get_lock_stats(conn: psycopg2.extensions.connection) -> dict:
    """
    Get statistics about currently held advisory locks.
    
    Args:
        conn: PostgreSQL connection
        
    Returns:
        Dictionary with lock statistics
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for lock statistics. "
            "Install with: pip install psycopg2-binary"
        )
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    locktype,
                    objid,
                    mode,
                    granted,
                    pid,
                    query_start,
                    state
                FROM pg_locks l
                LEFT JOIN pg_stat_activity a ON l.pid = a.pid
                WHERE locktype = 'advisory'
                ORDER BY objid
            """)
            
            locks = cur.fetchall()
            
            return {
                'total_advisory_locks': len(locks),
                'granted_locks': len([l for l in locks if l['granted']]),
                'waiting_locks': len([l for l in locks if not l['granted']]),
                'locks': locks
            }
            
    except Exception as e:
        logger.error(f"Error getting lock statistics: {e}")
        return {'error': str(e)}