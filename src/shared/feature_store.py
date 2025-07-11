# src/shared/feature_store.py
import os
import json
import hashlib
import logging
import duckdb
import pandas as pd
import zstandard as zstd
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from io import BytesIO

# PostgreSQL imports for manifest operations with advisory locks
try:
    from .db_pool import pg_conn, is_available as pg_available
    from .db_locker import advisory_lock, symbol_to_lock_key
    from .manifest_schema import initialize_manifest_db, migrate_from_duckdb
    PG_MANIFEST_AVAILABLE = True
except ImportError:
    PG_MANIFEST_AVAILABLE = False
    pg_conn = None
    advisory_lock = None
    symbol_to_lock_key = None
    initialize_manifest_db = None
    migrate_from_duckdb = None

# Prometheus metrics for FeatureStore hit ratio SLO monitoring and advisory lock performance
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): 
            self._value = type('MockValue', (), {'get': lambda: 0})()
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): 
            self._value = type('MockValue', (), {'get': lambda: 0})()
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass

# Global FeatureStore metrics (shared across all instances)
# Handle duplicate registration gracefully
def _create_metrics():
    """Create metrics with duplicate registration handling."""
    try:
        hits = Counter(
            "featurestore_hits_total",
            "Features served from cache",
            ["symbol"]
        )
        misses = Counter(
            "featurestore_misses_total", 
            "Features recomputed",
            ["symbol"]
        )
        hit_ratio = Gauge(
            "featurestore_hit_ratio",
            "Rolling hit ratio (cache / total)"
        )
        
        # New advisory lock performance metrics
        manifest_insert_latency = Histogram(
            "manifest_insert_latency_ms",
            "Manifest insert operation latency in milliseconds",
            ["backend", "symbol"],
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        )
        
        manifest_read_latency = Histogram(
            "manifest_read_latency_ms", 
            "Manifest read operation latency in milliseconds",
            ["backend", "symbol"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        advisory_lock_wait_time = Histogram(
            "advisory_lock_wait_time_ms",
            "Time spent waiting for advisory locks in milliseconds", 
            ["symbol"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]
        )
        
        concurrent_workers = Gauge(
            "featurestore_concurrent_workers",
            "Number of concurrent workers accessing FeatureStore"
        )
        
        pg_connection_pool_size = Gauge(
            "pg_manifest_pool_connections_total",
            "Total connections in PostgreSQL pool"
        )
        
        pg_connection_pool_active = Gauge(
            "pg_manifest_pool_connections_active", 
            "Active connections in PostgreSQL pool"
        )
        
        return (hits, misses, hit_ratio, manifest_insert_latency, manifest_read_latency, 
                advisory_lock_wait_time, concurrent_workers, pg_connection_pool_size, 
                pg_connection_pool_active)
                
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Return mock objects that work but don't register
            class MockMetric:
                def __init__(self, *args, **kwargs): pass
                def inc(self, *args, **kwargs): pass
                def set(self, *args, **kwargs): pass
                def observe(self, *args, **kwargs): pass
                def labels(self, *args, **kwargs): return self
                def time(self): return self
                def __enter__(self): return self
                def __exit__(self, *args): pass
            
            return tuple(MockMetric() for _ in range(9))
        else:
            raise

(FEATURE_HITS, FEATURE_MISS, FEATURE_HIT_RT, MANIFEST_INSERT_LATENCY, 
 MANIFEST_READ_LATENCY, ADVISORY_LOCK_WAIT_TIME, CONCURRENT_WORKERS,
 PG_POOL_SIZE, PG_POOL_ACTIVE) = _create_metrics()


def _update_hit_ratio():
    """Update the global hit ratio gauge based on current counters."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        # Get current counter values
        hits = sum(metric.get() for metric in FEATURE_HITS._metrics.values()) if hasattr(FEATURE_HITS, '_metrics') else 0
        misses = sum(metric.get() for metric in FEATURE_MISS._metrics.values()) if hasattr(FEATURE_MISS, '_metrics') else 0
        
        total = hits + misses
        if total > 0:
            hit_ratio = hits / total
            FEATURE_HIT_RT.set(hit_ratio)
    except Exception:
        # Silently ignore metrics errors to avoid disrupting core functionality
        pass


class FeatureStore:
    """
    High-performance feature cache using DuckDB + Parquet + Zstandard compression.
    
    Uses composite SHA-256 hash key strategy:
    - symbol
    - start_ts (8-byte little-endian)  
    - end_ts (8-byte little-endian)
    - feature_cfg_sha256 (json-serialized config, sorted keys)
    - raw_data_sha256 (checksum of raw data)
    
    This ensures reproducible lineage while maximizing cache hit rates.
    """
    
    def __init__(self, root: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize FeatureStore.
        
        Args:
            root: Root directory for feature cache. Defaults to ~/.feature_cache
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Setup cache directory
        self.base = Path(root or os.getenv("FEATURE_STORE_PATH", "~/.feature_cache")).expanduser()
        self.base.mkdir(parents=True, exist_ok=True)
        
        # Determine manifest backend (PostgreSQL preferred for concurrency)
        self.use_pg_manifest = PG_MANIFEST_AVAILABLE and pg_available()
        
        if self.use_pg_manifest:
            # Initialize PostgreSQL manifest for high-concurrency operations
            try:
                initialize_manifest_db()
                self.logger.info("Using PostgreSQL manifest with advisory locks for high concurrency")
                
                # Migrate existing DuckDB data if present
                duckdb_path = self.base / "manifest.duckdb"
                if duckdb_path.exists():
                    migrated = migrate_from_duckdb(duckdb_path)
                    if migrated > 0:
                        self.logger.info(f"Migrated {migrated} entries from DuckDB to PostgreSQL")
                        # Keep DuckDB as backup
                        duckdb_path.rename(self.base / "manifest.duckdb.backup")
                
            except Exception as e:
                self.logger.warning(f"PostgreSQL manifest initialization failed: {e}")
                self.logger.info("Falling back to DuckDB manifest")
                self.use_pg_manifest = False
        
        if not self.use_pg_manifest:
            # Fallback to DuckDB manifest database
            self.db_path = self.base / "manifest.duckdb"
            self.db = duckdb.connect(str(self.db_path))
            self._initialize_manifest_table()
            self.logger.info(f"Using DuckDB manifest database: {self.db_path}")
        
        self.logger.info(f"FeatureStore initialized at {self.base}")
        self.logger.info(f"Manifest backend: {'PostgreSQL' if self.use_pg_manifest else 'DuckDB'}")
    
    def _initialize_manifest_table(self):
        """Initialize the manifest table for tracking cached features."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS manifest(
                key TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_ts BIGINT NOT NULL,
                end_ts BIGINT NOT NULL,
                rows INTEGER NOT NULL,
                file_size_bytes INTEGER,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)
        
        # Add access_count column if it doesn't exist (for existing databases)
        try:
            self.db.execute("ALTER TABLE manifest ADD COLUMN access_count INTEGER DEFAULT 1")
        except Exception:
            # Column already exists or other error - ignore
            pass
        
        # Create indexes for efficient queries
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON manifest(symbol)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_created_ts ON manifest(created_ts)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON manifest(last_accessed_ts)")
    
    def _sha256(self, *parts: bytes) -> str:
        """Compute SHA-256 hash of multiple byte parts."""
        h = hashlib.sha256()
        for part in parts:
            h.update(part)
        return h.hexdigest()
    
    def _compute_raw_data_hash(self, raw_df: pd.DataFrame) -> bytes:
        """
        Compute hash of raw data content with optimization for large tick data.
        For large datasets (>100MB), only hash the footer to avoid memory issues.
        """
        try:
            # Convert to parquet bytes for size estimation
            parquet_bytes = raw_df.to_parquet(index=True)
            data_size_mb = len(parquet_bytes) / 1024 / 1024
            
            # Use footer-only hashing for large tick data (>100MB)
            if data_size_mb > 100:
                self.logger.debug(f"Using footer hashing for large dataset ({data_size_mb:.1f} MB)")
                
                # Hash only the footer (last 1000 rows) + metadata for efficiency
                footer_rows = min(1000, len(raw_df))
                footer_df = raw_df.tail(footer_rows)
                
                # Include dataset metadata in hash for uniqueness
                import json
                metadata = {
                    'total_rows': len(raw_df),
                    'columns': raw_df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
                    'index_min': str(raw_df.index.min()),
                    'index_max': str(raw_df.index.max()),
                    'footer_rows': footer_rows
                }
                
                # Combine footer data + metadata for hash
                footer_parquet = footer_df.to_parquet(index=True)
                metadata_json = json.dumps(metadata, sort_keys=True)
                
                hash_input = footer_parquet + metadata_json.encode()
                return hashlib.sha256(hash_input).digest()
            
            else:
                # Use full hashing for smaller datasets
                return hashlib.sha256(parquet_bytes).digest()
                
        except Exception as e:
            self.logger.warning(f"Failed to hash via parquet, using string representation: {e}")
            # Fallback to string representation hash
            data_str = f"{raw_df.index.min()}_{raw_df.index.max()}_{len(raw_df)}_{raw_df.columns.tolist()}"
            return hashlib.sha256(data_str.encode()).digest()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> bytes:
        """Compute hash of feature configuration."""
        config_json = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).digest()
    
    def _generate_cache_key(self, symbol: str, start_ts: int, end_ts: int, 
                           config: Dict[str, Any], raw_df: pd.DataFrame) -> str:
        """
        Generate composite cache key using SHA-256 hash strategy.
        
        Args:
            symbol: Stock symbol
            start_ts: Start timestamp (Unix epoch)
            end_ts: End timestamp (Unix epoch)  
            config: Feature configuration dictionary
            raw_df: Raw data DataFrame
            
        Returns:
            SHA-256 hash string
        """
        cfg_hash = self._compute_config_hash(config)
        raw_hash = self._compute_raw_data_hash(raw_df)
        
        key = self._sha256(
            symbol.encode('utf-8'),
            start_ts.to_bytes(8, 'little'),
            end_ts.to_bytes(8, 'little'),
            cfg_hash,
            raw_hash
        )
        
        self.logger.debug(f"Generated cache key {key[:16]}... for {symbol} {start_ts}-{end_ts}")
        return key
    
    def _get_timestamps_from_dataframe(self, df: pd.DataFrame) -> tuple[int, int]:
        """Extract start and end timestamps from DataFrame index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for timestamp extraction")
        
        start_ts = int(df.index.min().timestamp())
        end_ts = int(df.index.max().timestamp())
        return start_ts, end_ts
    
    def get_or_compute(self, symbol: str, raw_df: pd.DataFrame, config: Dict[str, Any],
                      compute_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """
        Get cached features or compute and cache them.
        
        Args:
            symbol: Stock symbol
            raw_df: Raw OHLCV data with DatetimeIndex
            config: Feature configuration dictionary
            compute_func: Function to compute features if not cached
            
        Returns:
            DataFrame with computed features
        """
        start_ts, end_ts = self._get_timestamps_from_dataframe(raw_df)
        cache_key = self._generate_cache_key(symbol, start_ts, end_ts, config, raw_df)
        
        # Check if features are already cached
        cached_path = self._get_cached_features(cache_key)
        if cached_path:
            # Cache HIT - record metrics
            self.logger.info(f"Cache HIT for {symbol} {cache_key[:16]}...")
            FEATURE_HITS.labels(symbol=symbol).inc()
            _update_hit_ratio()
            return self._load_cached_features(cached_path, cache_key)
        
        # Cache MISS - record metrics and compute features
        self.logger.info(f"Cache MISS for {symbol} {cache_key[:16]}... - computing features")
        FEATURE_MISS.labels(symbol=symbol).inc()
        _update_hit_ratio()
        
        features_df = compute_func(raw_df, config)
        
        if features_df is not None and not features_df.empty:
            self._cache_features(cache_key, features_df, symbol, start_ts, end_ts)
        
        return features_df
    
    def _get_cached_features(self, cache_key: str) -> Optional[str]:
        """Check if features are cached and return file path."""
        if self.use_pg_manifest:
            return self._get_cached_features_pg(cache_key)
        else:
            return self._get_cached_features_duckdb(cache_key)
    
    def _get_cached_features_pg(self, cache_key: str) -> Optional[str]:
        """Check if features are cached using PostgreSQL manifest."""
        symbol = cache_key.split('_')[0] if '_' in cache_key else 'unknown'
        
        try:
            with MANIFEST_READ_LATENCY.labels(backend='postgresql', symbol=symbol).time():
                with pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT path FROM manifest WHERE key = %s", 
                            (cache_key,)
                        )
                        result = cur.fetchone()
                        
                        if result:
                            path = result['path']
                            if Path(path).exists():
                                # Update last accessed timestamp
                                cur.execute(
                                    "UPDATE manifest SET last_accessed_ts = CURRENT_TIMESTAMP WHERE key = %s",
                                    (cache_key,)
                                )
                                conn.commit()
                                return path
                            else:
                                # File missing - clean up manifest entry
                                self.logger.warning(f"Cached file missing: {path}")
                                cur.execute("DELETE FROM manifest WHERE key = %s", (cache_key,))
                                conn.commit()
                        
                        return None
                    
        except Exception as e:
            self.logger.error(f"Error checking PostgreSQL cache for key {cache_key[:16]}...: {e}")
            return None
    
    def _get_cached_features_duckdb(self, cache_key: str) -> Optional[str]:
        """Check if features are cached using DuckDB manifest."""
        symbol = cache_key.split('_')[0] if '_' in cache_key else 'unknown'
        
        try:
            with MANIFEST_READ_LATENCY.labels(backend='duckdb', symbol=symbol).time():
                # Use a separate connection for thread safety
                with duckdb.connect(str(self.db_path)) as conn:
                    result = conn.execute(
                        "SELECT path FROM manifest WHERE key = ?", 
                        [cache_key]
                    ).fetchone()
                    
                    if result:
                        path = result[0]
                        if Path(path).exists():
                            # Update last accessed timestamp
                            conn.execute(
                                "UPDATE manifest SET last_accessed_ts = CURRENT_TIMESTAMP WHERE key = ?",
                                [cache_key]
                            )
                            return path
                        else:
                            # File missing - clean up manifest entry
                            self.logger.warning(f"Cached file missing: {path}")
                            conn.execute("DELETE FROM manifest WHERE key = ?", [cache_key])
                    
                    return None
            
        except Exception as e:
            self.logger.error(f"Error checking DuckDB cache for key {cache_key[:16]}...: {e}")
            return None
    
    def _load_cached_features(self, file_path: str, cache_key: str) -> pd.DataFrame:
        """Load cached features from compressed parquet file."""
        try:
            if file_path.endswith('.zst'):
                # Load compressed parquet
                decompressor = zstd.ZstdDecompressor()
                with open(file_path, 'rb') as raw_fh:
                    with decompressor.stream_reader(raw_fh) as fh:
                        parquet_bytes = fh.read()
                        df = pd.read_parquet(BytesIO(parquet_bytes))
            else:
                # Load regular parquet
                df = pd.read_parquet(file_path)
            
            self.logger.debug(f"Loaded {len(df)} cached features from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading cached features from {file_path}: {e}")
            # Clean up bad cache entry
            self.db.execute("DELETE FROM manifest WHERE key = ?", [cache_key])
            raise
    
    def _cache_features(self, cache_key: str, features_df: pd.DataFrame, 
                       symbol: str, start_ts: int, end_ts: int):
        """Cache computed features to compressed parquet file with exclusive locking."""
        if self.use_pg_manifest:
            self._cache_features_pg(cache_key, features_df, symbol, start_ts, end_ts)
        else:
            self._cache_features_duckdb(cache_key, features_df, symbol, start_ts, end_ts)
    
    def _cache_features_pg(self, cache_key: str, features_df: pd.DataFrame, 
                          symbol: str, start_ts: int, end_ts: int):
        """Cache features using PostgreSQL manifest with advisory locks."""
        cache_file = None
        try:
            # Generate file path
            cache_file = self.base / f"{cache_key}.parquet.zst"
            
            # Save with zstandard compression (level 3 for good speed/compression balance)
            compressor = zstd.ZstdCompressor(level=3)
            with open(cache_file, 'wb') as raw_fh:
                with compressor.stream_writer(raw_fh) as fh:
                    parquet_bytes = features_df.to_parquet(index=True)
                    fh.write(parquet_bytes)
            
            file_size = cache_file.stat().st_size
            
            # Use PostgreSQL with advisory lock for high-concurrency INSERT
            lock_key = symbol_to_lock_key(symbol)
            
            with pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    try:
                        # Measure advisory lock wait time
                        with ADVISORY_LOCK_WAIT_TIME.labels(symbol=symbol).time():
                            # Acquire advisory lock for this symbol to prevent row-lock contention
                            with advisory_lock(conn, lock_key):
                                # Measure manifest insert latency
                                with MANIFEST_INSERT_LATENCY.labels(backend='postgresql', symbol=symbol).time():
                                    cur.execute("""
                                        INSERT INTO manifest 
                                        (key, path, symbol, start_ts, end_ts, rows, file_size_bytes) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                        ON CONFLICT (key) DO UPDATE SET
                                            path = EXCLUDED.path,
                                            symbol = EXCLUDED.symbol,
                                            start_ts = EXCLUDED.start_ts,
                                            end_ts = EXCLUDED.end_ts,
                                            rows = EXCLUDED.rows,
                                            file_size_bytes = EXCLUDED.file_size_bytes,
                                            last_accessed_ts = CURRENT_TIMESTAMP,
                                            access_count = manifest.access_count + 1
                                    """, (
                                        cache_key, str(cache_file), symbol, start_ts, end_ts, 
                                        len(features_df), file_size
                                    ))
                        
                        cur.execute("COMMIT")
                        
                        self.logger.info(f"Cached {len(features_df)} features for {symbol} "
                                       f"({file_size:,} bytes compressed) [PostgreSQL+AdvisoryLock]")
                        
                    except Exception as e:
                        cur.execute("ROLLBACK")
                        raise e
                
        except Exception as e:
            self.logger.error(f"Error caching features (PostgreSQL) for key {cache_key[:16]}...: {e}")
            # Clean up partial file
            if cache_file and cache_file.exists():
                cache_file.unlink()
            raise
    
    def _cache_features_duckdb(self, cache_key: str, features_df: pd.DataFrame, 
                              symbol: str, start_ts: int, end_ts: int):
        """Cache features using DuckDB manifest (fallback)."""
        cache_file = None
        try:
            # Generate file path
            cache_file = self.base / f"{cache_key}.parquet.zst"
            
            # Save with zstandard compression (level 3 for good speed/compression balance)
            compressor = zstd.ZstdCompressor(level=3)
            with open(cache_file, 'wb') as raw_fh:
                with compressor.stream_writer(raw_fh) as fh:
                    parquet_bytes = features_df.to_parquet(index=True)
                    fh.write(parquet_bytes)
            
            file_size = cache_file.stat().st_size
            
            # Use separate connection for thread safety
            with MANIFEST_INSERT_LATENCY.labels(backend='duckdb', symbol=symbol).time():
                with duckdb.connect(str(self.db_path)) as conn:
                    conn.execute("BEGIN TRANSACTION")
                    try:
                        conn.execute("""
                            INSERT INTO manifest 
                            (key, path, symbol, start_ts, end_ts, rows, file_size_bytes) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (key) DO UPDATE SET
                                path = EXCLUDED.path,
                                symbol = EXCLUDED.symbol,
                                start_ts = EXCLUDED.start_ts,
                                end_ts = EXCLUDED.end_ts,
                                rows = EXCLUDED.rows,
                                file_size_bytes = EXCLUDED.file_size_bytes,
                                last_accessed_ts = now(),
                                access_count = access_count + 1
                        """, [
                            cache_key, str(cache_file), symbol, start_ts, end_ts, 
                            len(features_df), file_size
                        ])
                        conn.execute("COMMIT")
                        
                        self.logger.info(f"Cached {len(features_df)} features for {symbol} "
                                       f"({file_size:,} bytes compressed) [DuckDB]")
                        
                    except Exception as e:
                        try:
                            conn.execute("ROLLBACK")
                        except:
                            pass  # Ignore rollback errors if no transaction is active
                        raise e
                
        except Exception as e:
            self.logger.error(f"Error caching features (DuckDB) for key {cache_key[:16]}...: {e}")
            # Clean up partial file
            if cache_file and cache_file.exists():
                cache_file.unlink()
            raise
    
    def warm_cache(self, symbol: str, raw_df: pd.DataFrame, config: Dict[str, Any],
                   compute_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]):
        """
        Warm cache by pre-computing features (used by DataAgent for offline preload).
        
        Args:
            symbol: Stock symbol
            raw_df: Raw data DataFrame
            config: Feature configuration
            compute_func: Function to compute features
        """
        self.logger.info(f"Warming cache for {symbol} with {len(raw_df)} rows")
        self.get_or_compute(symbol, raw_df, config, compute_func)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self.db.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    SUM(rows) as total_cached_rows,
                    SUM(file_size_bytes) as total_size_bytes,
                    MIN(created_ts) as oldest_entry,
                    MAX(created_ts) as newest_entry
                FROM manifest
            """).fetchone()
            
            if stats:
                return {
                    'total_entries': stats[0],
                    'unique_symbols': stats[1], 
                    'total_cached_rows': stats[2] or 0,
                    'total_size_mb': round((stats[3] or 0) / 1024 / 1024, 2),
                    'oldest_entry': stats[4],
                    'newest_entry': stats[5],
                    'cache_directory': str(self.base)
                }
            else:
                return {'total_entries': 0}
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_entries(self, days_old: int = 30):
        """
        Clean up cache entries older than specified days.
        
        Args:
            days_old: Remove entries older than this many days
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get files to delete
            old_entries = self.db.execute("""
                SELECT key, path FROM manifest 
                WHERE last_accessed_ts < ?
            """, [cutoff_date]).fetchall()
            
            deleted_count = 0
            freed_bytes = 0
            
            for key, path in old_entries:
                try:
                    file_path = Path(path)
                    if file_path.exists():
                        freed_bytes += file_path.stat().st_size
                        file_path.unlink()
                    
                    self.db.execute("DELETE FROM manifest WHERE key = ?", [key])
                    deleted_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error deleting cache entry {key}: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} old cache entries, "
                           f"freed {freed_bytes / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            symbol: If provided, only clear entries for this symbol
        """
        try:
            if symbol:
                entries = self.db.execute(
                    "SELECT path FROM manifest WHERE symbol = ?", [symbol]
                ).fetchall()
                self.db.execute("DELETE FROM manifest WHERE symbol = ?", [symbol])
                self.logger.info(f"Cleared cache for symbol {symbol}")
            else:
                entries = self.db.execute("SELECT path FROM manifest").fetchall()
                self.db.execute("DELETE FROM manifest")
                self.logger.info("Cleared entire cache")
            
            # Delete physical files
            deleted_count = 0
            for (path,) in entries:
                try:
                    Path(path).unlink(missing_ok=True)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Error deleting file {path}: {e}")
            
            self.logger.info(f"Deleted {deleted_count} cache files")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'db') and self.db:
            try:
                self.db.close()
                self.logger.info("FeatureStore database connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for global feature store instance
_global_feature_store = None

def get_feature_store(root: Optional[str] = None, logger: Optional[logging.Logger] = None) -> FeatureStore:
    """Get global FeatureStore instance."""
    global _global_feature_store
    if _global_feature_store is None:
        _global_feature_store = FeatureStore(root=root, logger=logger)
    return _global_feature_store