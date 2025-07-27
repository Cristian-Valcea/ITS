"""
PostgreSQL connection pool for FeatureStore manifest operations.

This module provides a dedicated PostgreSQL connection pool for manifest table
operations with advisory lock support to eliminate row-lock contention when
multiple training workers access the same symbol/date combinations.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator
import threading

try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    SimpleConnectionPool = None
    RealDictCursor = None

logger = logging.getLogger(__name__)

# Prometheus metrics for connection pool monitoring
try:
    from prometheus_client import Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Use try-except to handle duplicate metric registration
    try:
        PG_POOL_SIZE = Gauge(
            "pg_manifest_pool_connections_total",
            "Total connections in PostgreSQL manifest pool"
        )
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already exists, create a dummy one that does nothing
            class DummyGauge:
                def set(self, value): pass
                def inc(self, amount=1): pass
                def dec(self, amount=1): pass
            PG_POOL_SIZE = DummyGauge()
        else:
            raise
    
    try:
        PG_POOL_ACTIVE = Gauge(
            "pg_manifest_pool_connections_active", 
            "Active connections in PostgreSQL manifest pool"
        )
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already exists, create a dummy one that does nothing
            class DummyGauge:
                def set(self, value): pass
                def inc(self, amount=1): pass
                def dec(self, amount=1): pass
            PG_POOL_ACTIVE = DummyGauge()
        else:
            raise
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    class MockGauge:
        def set(self, *args, **kwargs): pass
    
    PG_POOL_SIZE = MockGauge()
    PG_POOL_ACTIVE = MockGauge()

# Global connection pool instance
_pool: Optional[SimpleConnectionPool] = None
_pool_lock = threading.Lock()


def _get_dsn() -> str:
    """Get PostgreSQL DSN from environment variables."""
    # Try explicit manifest DSN first
    dsn = os.getenv("PG_MANIFEST_DSN")
    if dsn:
        return dsn
    
    # Fallback to individual components
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    database = os.getenv("PG_DATABASE", "featurestore_manifest")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    
    return f"host={host} port={port} dbname={database} user={user} password={password}"


def _initialize_pool() -> SimpleConnectionPool:
    """Initialize the PostgreSQL connection pool."""
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for PostgreSQL manifest operations. "
            "Install with: pip install psycopg2-binary"
        )
    
    dsn = _get_dsn()
    logger.info(f"Initializing PostgreSQL pool for manifest operations")
    
    try:
        pool = SimpleConnectionPool(
            minconn=2,
            maxconn=16,  # Support up to 16 concurrent workers
            dsn=dsn,
            cursor_factory=RealDictCursor
        )
        
        # Test the connection
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()
                logger.info(f"Connected to PostgreSQL: {version['version']}")
        finally:
            pool.putconn(conn)
        
        # Initialize pool metrics
        PG_POOL_SIZE.set(pool.maxconn)
        PG_POOL_ACTIVE.set(0)
        
        return pool
        
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL pool: {e}")
        raise


def get_pool() -> SimpleConnectionPool:
    """Get the global connection pool, initializing if necessary."""
    global _pool
    
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # Double-check locking
                _pool = _initialize_pool()
    
    return _pool


@contextmanager
def pg_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager for PostgreSQL connections from the pool.
    
    Yields:
        PostgreSQL connection with automatic return to pool
        
    Example:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM manifest WHERE symbol = %s", (symbol,))
                rows = cur.fetchall()
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for PostgreSQL manifest operations. "
            "Install with: pip install psycopg2-binary"
        )
    
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.getconn()
        conn.autocommit = False  # Explicit transaction control
        
        # Update active connections metric
        active_count = len(getattr(pool, '_used', []))
        PG_POOL_ACTIVE.set(active_count)
        
        yield conn
        
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass  # Ignore rollback errors
        logger.error(f"PostgreSQL connection error: {e}")
        raise
        
    finally:
        if conn:
            try:
                pool.putconn(conn)
                # Update active connections metric
                active_count = len(getattr(pool, '_used', []))
                PG_POOL_ACTIVE.set(active_count)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")


def close_pool():
    """Close the connection pool and all connections."""
    global _pool
    
    if _pool:
        with _pool_lock:
            if _pool:
                try:
                    _pool.closeall()
                    logger.info("PostgreSQL connection pool closed")
                except Exception as e:
                    logger.error(f"Error closing PostgreSQL pool: {e}")
                finally:
                    _pool = None


def is_available() -> bool:
    """Check if PostgreSQL connection is available."""
    if not PSYCOPG2_AVAILABLE:
        return False
    
    try:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False


def get_pool_stats() -> Optional[dict]:
    """Get connection pool statistics."""
    global _pool
    
    if not _pool or not PSYCOPG2_AVAILABLE:
        return None
    
    try:
        total_connections = _pool.maxconn
        active_connections = len(getattr(_pool, '_used', []))
        
        return {
            'total_connections': total_connections,
            'active_connections': active_connections,
            'available_connections': total_connections - active_connections
        }
    except Exception as e:
        logger.error(f"Error getting pool stats: {e}")
        return None


# Cleanup on module exit
import atexit
atexit.register(close_pool)