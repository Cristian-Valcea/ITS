"""
PostgreSQL manifest table schema and migration utilities.

This module handles the creation and migration of the manifest table
from DuckDB to PostgreSQL for improved concurrency with advisory locks.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

from .db_pool import pg_conn

logger = logging.getLogger(__name__)


def create_manifest_table(conn: psycopg2.extensions.connection) -> None:
    """
    Create the manifest table in PostgreSQL with optimized schema.
    
    The table includes a shard_hash column for efficient advisory locking
    and proper indexes for high-concurrency access patterns.
    
    Args:
        conn: PostgreSQL connection
    """
    with conn.cursor() as cur:
        # Create the manifest table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                key TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_ts BIGINT NOT NULL,
                end_ts BIGINT NOT NULL,
                rows INTEGER NOT NULL,
                file_size_bytes INTEGER,
                created_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_accessed_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                -- Computed shard hash for advisory locking
                shard_hash BIGINT GENERATED ALWAYS AS (
                    ('x' || substring(encode(sha256(symbol::bytea), 'hex'), 1, 15))::bit(60)::bigint
                ) STORED
            )
        """)
        
        # Create indexes for efficient queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_manifest_symbol ON manifest(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_manifest_created_ts ON manifest(created_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_manifest_last_accessed ON manifest(last_accessed_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_manifest_shard_hash ON manifest(shard_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_manifest_symbol_ts ON manifest(symbol, start_ts, end_ts)")
        
        # Create partial index for recent entries (hot data)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_manifest_recent 
            ON manifest(symbol, last_accessed_ts) 
            WHERE last_accessed_ts > CURRENT_TIMESTAMP - INTERVAL '7 days'
        """)
        
        conn.commit()
        logger.info("Manifest table and indexes created successfully")


def migrate_from_duckdb(duckdb_path: Path, batch_size: int = 1000) -> int:
    """
    Migrate manifest data from DuckDB to PostgreSQL.
    
    Args:
        duckdb_path: Path to the DuckDB manifest file
        batch_size: Number of records to migrate per batch
        
    Returns:
        Number of records migrated
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for manifest migration. "
            "Install with: pip install psycopg2-binary"
        )
    
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is required for migration")
    
    if not duckdb_path.exists():
        logger.info(f"DuckDB manifest file not found: {duckdb_path}")
        return 0
    
    logger.info(f"Starting migration from DuckDB: {duckdb_path}")
    
    # Connect to DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    
    try:
        # Get total count
        total_count = duck_conn.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]
        if total_count == 0:
            logger.info("No records to migrate")
            return 0
        
        logger.info(f"Migrating {total_count} records in batches of {batch_size}")
        
        migrated_count = 0
        
        with pg_conn() as pg_connection:
            # Ensure manifest table exists
            create_manifest_table(pg_connection)
            
            # Migrate in batches
            offset = 0
            while offset < total_count:
                # Fetch batch from DuckDB
                rows = duck_conn.execute("""
                    SELECT key, path, symbol, start_ts, end_ts, rows, 
                           file_size_bytes, created_ts, last_accessed_ts, access_count
                    FROM manifest 
                    ORDER BY created_ts
                    LIMIT ? OFFSET ?
                """, [batch_size, offset]).fetchall()
                
                if not rows:
                    break
                
                # Insert batch into PostgreSQL
                with pg_connection.cursor() as cur:
                    cur.execute("BEGIN")
                    try:
                        cur.executemany("""
                            INSERT INTO manifest 
                            (key, path, symbol, start_ts, end_ts, rows, 
                             file_size_bytes, created_ts, last_accessed_ts, access_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (key) DO UPDATE SET
                                path = EXCLUDED.path,
                                symbol = EXCLUDED.symbol,
                                start_ts = EXCLUDED.start_ts,
                                end_ts = EXCLUDED.end_ts,
                                rows = EXCLUDED.rows,
                                file_size_bytes = EXCLUDED.file_size_bytes,
                                last_accessed_ts = EXCLUDED.last_accessed_ts,
                                access_count = EXCLUDED.access_count
                        """, rows)
                        
                        cur.execute("COMMIT")
                        migrated_count += len(rows)
                        
                        logger.info(f"Migrated {migrated_count}/{total_count} records "
                                  f"({migrated_count/total_count*100:.1f}%)")
                        
                    except Exception as e:
                        cur.execute("ROLLBACK")
                        logger.error(f"Error migrating batch at offset {offset}: {e}")
                        raise
                
                offset += batch_size
        
        logger.info(f"Migration completed: {migrated_count} records migrated")
        return migrated_count
        
    finally:
        duck_conn.close()


def get_manifest_stats() -> Dict[str, Any]:
    """
    Get statistics about the manifest table.
    
    Returns:
        Dictionary with manifest statistics
    """
    if not PSYCOPG2_AVAILABLE:
        return {"error": "psycopg2 not available"}
    
    try:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                # Basic counts
                cur.execute("SELECT COUNT(*) as total_entries FROM manifest")
                total_entries = cur.fetchone()['total_entries']
                
                # Symbol distribution
                cur.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM manifest 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                top_symbols = cur.fetchall()
                
                # Recent activity
                cur.execute("""
                    SELECT COUNT(*) as recent_entries
                    FROM manifest 
                    WHERE last_accessed_ts > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)
                recent_entries = cur.fetchone()['recent_entries']
                
                # Storage stats
                cur.execute("""
                    SELECT 
                        SUM(file_size_bytes) as total_bytes,
                        AVG(file_size_bytes) as avg_bytes,
                        SUM(rows) as total_rows,
                        AVG(rows) as avg_rows
                    FROM manifest
                """)
                storage_stats = cur.fetchone()
                
                return {
                    'total_entries': total_entries,
                    'recent_entries': recent_entries,
                    'top_symbols': [dict(row) for row in top_symbols],
                    'storage': {
                        'total_bytes': storage_stats['total_bytes'] or 0,
                        'avg_bytes': float(storage_stats['avg_bytes'] or 0),
                        'total_rows': storage_stats['total_rows'] or 0,
                        'avg_rows': float(storage_stats['avg_rows'] or 0)
                    }
                }
                
    except Exception as e:
        logger.error(f"Error getting manifest stats: {e}")
        return {"error": str(e)}


def cleanup_old_entries(days: int = 30) -> int:
    """
    Clean up old manifest entries that haven't been accessed recently.
    
    Args:
        days: Number of days of inactivity before cleanup
        
    Returns:
        Number of entries cleaned up
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for cleanup operations. "
            "Install with: pip install psycopg2-binary"
        )
    
    try:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                # Find old entries
                cur.execute("""
                    SELECT key, path FROM manifest 
                    WHERE last_accessed_ts < CURRENT_TIMESTAMP - INTERVAL '%s days'
                """, (days,))
                
                old_entries = cur.fetchall()
                
                if not old_entries:
                    logger.info("No old entries to clean up")
                    return 0
                
                logger.info(f"Cleaning up {len(old_entries)} entries older than {days} days")
                
                # Delete old entries and their files
                deleted_count = 0
                for entry in old_entries:
                    try:
                        # Delete file if it exists
                        file_path = Path(entry['path'])
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Delete manifest entry
                        cur.execute("DELETE FROM manifest WHERE key = %s", (entry['key'],))
                        deleted_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error cleaning up entry {entry['key']}: {e}")
                
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old entries")
                return deleted_count
                
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


def initialize_manifest_db() -> None:
    """
    Initialize the PostgreSQL manifest database.
    
    This function should be called once during application startup
    to ensure the manifest table exists with proper schema.
    """
    if not PSYCOPG2_AVAILABLE:
        logger.warning("PostgreSQL not available - falling back to DuckDB manifest")
        return
    
    try:
        with pg_conn() as conn:
            create_manifest_table(conn)
            logger.info("Manifest database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize manifest database: {e}")
        raise