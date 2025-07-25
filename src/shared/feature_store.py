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
from .duckdb_manager import get_duckdb_connection, close_write_duckdb_connections, close_all_duckdb_connections

# PostgreSQL imports for manifest operations with advisory locks
try:
    from .db_pool import pg_conn, is_available as pg_available
    from .db_locker import advisory_lock, symbol_to_lock_key
    from .manifest_schema import initialize_manifest_db, migrate_from_duckdb
    import psycopg2.extras
    PG_MANIFEST_AVAILABLE = True
except ImportError:
    PG_MANIFEST_AVAILABLE = False
    pg_conn = None
    advisory_lock = None
    symbol_to_lock_key = None
    initialize_manifest_db = None
    migrate_from_duckdb = None
    psycopg2 = None

# Prometheus metrics for FeatureStore hit ratio SLO monitoring and advisory lock performance
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): 
            self._counter = 0
        def inc(self, *args, **kwargs): 
            self._counter += 1
        def labels(self, *args, **kwargs): return self
        def collect(self):
            return [type('MockSample', (), {'value': self._counter})()]
    
    class Gauge:
        def __init__(self, *args, **kwargs): 
            self._value = 0
        def set(self, value, *args, **kwargs): 
            self._value = value
        def labels(self, *args, **kwargs): return self
        def collect(self):
            return [type('MockSample', (), {'value': self._value})()]
    
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
    """
    Create metrics with duplicate registration handling.
    
    Note: In multi-process environments (e.g., SB3 workers), each process will
    create its own metrics registry. For production deployment, consider:
    1. Using multiprocessing.set_start_method("spawn") to avoid registry conflicts
    2. Or disable prometheus_client.REGISTRY in child processes  
    3. Or use a single metrics collection process with inter-process communication
    """
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
        
        pg_connection_pool_size = Gauge(
            "pg_manifest_pool_connections_total",
            "Total connections in PostgreSQL pool"
        )
        
        pg_connection_pool_active = Gauge(
            "pg_manifest_pool_connections_active", 
            "Active connections in PostgreSQL pool"
        )
        
        return (hits, misses, hit_ratio, manifest_insert_latency, manifest_read_latency, 
                advisory_lock_wait_time, pg_connection_pool_size, 
                pg_connection_pool_active)
                
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Return mock objects that work but don't register
            class MockMetric:
                def __init__(self, *args, **kwargs): 
                    self._counter = 0
                    self._value = 0
                def inc(self, *args, **kwargs): 
                    self._counter += 1
                def set(self, value, *args, **kwargs): 
                    self._value = value
                def observe(self, *args, **kwargs): pass
                def labels(self, *args, **kwargs): return self
                def time(self): return self
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def collect(self):
                    return [type('MockSample', (), {'value': max(self._counter, self._value)})()]
            
            return tuple(MockMetric() for _ in range(8))
        else:
            raise

(FEATURE_HITS, FEATURE_MISS, FEATURE_HIT_RT, MANIFEST_INSERT_LATENCY, 
 MANIFEST_READ_LATENCY, ADVISORY_LOCK_WAIT_TIME, 
 PG_POOL_SIZE, PG_POOL_ACTIVE) = _create_metrics()


def _get_metric_total(metric):
    """
    Get total value from Prometheus metric, handling both real and mock implementations.
    
    Note: For metrics with multiple time-series (labels), this sums ALL samples across
    all label combinations. This is exactly what we want for hit/miss totals where we
    need the grand total across all symbols.
    """
    try:
        collected = metric.collect()
        if not collected:
            return 0
        
        # Real Prometheus returns MetricFamily with samples
        if hasattr(collected[0], 'samples'):
            return sum(sample.value for sample in collected[0].samples)
        # Mock implementation returns objects with direct value attribute
        else:
            return sum(sample.value for sample in collected)
    except Exception:
        return 0


def _update_hit_ratio():
    """Update the global hit ratio gauge based on current counters."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        # Get current counter values using proper metric collection
        hits = _get_metric_total(FEATURE_HITS)
        misses = _get_metric_total(FEATURE_MISS)
        
        total = hits + misses
        if total > 0:
            hit_ratio = hits / total
            FEATURE_HIT_RT.set(hit_ratio)
    except Exception:
        # Silently ignore metrics errors to avoid disrupting core functionality
        pass


def _update_pool_metrics():
    """
    Update PostgreSQL connection pool metrics.
    
    Requires db_pool.py to implement get_pool_stats() function that returns:
    {
        'total_connections': int,
        'active_connections': int
    }
    
    If get_pool_stats() is not available, pool metrics will be silently skipped
    without affecting core FeatureStore functionality.
    """
    if not PROMETHEUS_AVAILABLE or not PG_MANIFEST_AVAILABLE:
        return
    
    try:
        # Update pool size metrics if available
        try:
            from .db_pool import get_pool_stats
            stats = get_pool_stats()
            if stats:
                PG_POOL_SIZE.set(stats.get('total_connections', 0))
                PG_POOL_ACTIVE.set(stats.get('active_connections', 0))
        except ImportError:
            # get_pool_stats not available - skip metrics update
            # This is expected if db_pool.py doesn't implement get_pool_stats()
            pass
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
    
    def __init__(self, root: Optional[str] = None, logger: Optional[logging.Logger] = None, read_only: bool = False):
        """
        Initialize FeatureStore.
        
        Args:
            root: Root directory for feature cache. Defaults to ~/.feature_cache
            logger: Logger instance
            read_only: If True, skip manifest table initialization (for monitoring/API use)
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.read_only = read_only
        
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
            # Don't keep persistent connection - use connection manager
            if not self.read_only:
                self._initialize_manifest_table()
            self.logger.info(f"Using DuckDB manifest database: {self.db_path}")
        
        self.logger.info(f"FeatureStore initialized at {self.base}")
        self.logger.info(f"Manifest backend: {'PostgreSQL' if self.use_pg_manifest else 'DuckDB'}")
    
    def _initialize_manifest_table(self):
        """Initialize the manifest table for tracking cached features."""
        from ..utils.db import get_write_conn
        with get_write_conn() as conn:
            conn.execute("""
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
                conn.execute("ALTER TABLE manifest ADD COLUMN access_count INTEGER DEFAULT 1")
            except Exception:
                # Column already exists or other error - ignore
                pass
            
            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON manifest(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_ts ON manifest(created_ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON manifest(last_accessed_ts)")
    
    def _sha256(self, *parts: bytes) -> str:
        """Compute SHA-256 hash of multiple byte parts."""
        h = hashlib.sha256()
        for part in parts:
            h.update(part)
        return h.hexdigest()
    
    def _compute_raw_data_hash(self, raw_df: pd.DataFrame) -> bytes:
        """
        Compute hash of raw data content with memory-efficient approach for large datasets.
        Uses memory usage estimation instead of full parquet conversion to avoid memory bloat.
        """
        try:
            # Use memory usage estimation instead of full parquet conversion
            memory_usage_mb = raw_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Use footer-only hashing for large tick data (>100MB memory usage)
            if memory_usage_mb > 100:
                self.logger.debug(f"Using footer hashing for large dataset ({memory_usage_mb:.1f} MB memory)")
                
                # Hash only the footer (last 1000 rows) + metadata for efficiency
                footer_rows = min(1000, len(raw_df))
                footer_df = raw_df.tail(footer_rows)
                
                # Include dataset metadata in hash for uniqueness
                metadata = {
                    'total_rows': len(raw_df),
                    'columns': raw_df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
                    'index_min': str(raw_df.index.min()),
                    'index_max': str(raw_df.index.max()),
                    'footer_rows': footer_rows,
                    'memory_usage_mb': memory_usage_mb
                }
                
                # Use footer values bytes + metadata for hash (avoid parquet conversion)
                footer_bytes = footer_df.values.tobytes()
                metadata_json = json.dumps(metadata, sort_keys=True)
                
                hash_input = footer_bytes + metadata_json.encode()
                return hashlib.sha256(hash_input).digest()
            
            else:
                # Use values bytes for smaller datasets (avoid parquet conversion)
                return hashlib.sha256(raw_df.values.tobytes()).digest()
                
        except Exception as e:
            self.logger.warning(f"Failed to hash via values bytes, using string representation: {e}")
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
        cached_path = self._get_cached_features(cache_key, symbol)
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
        
        if features_df is not None and not features_df.empty and not self.read_only:
            self._cache_features(cache_key, features_df, symbol, start_ts, end_ts)
        
        return features_df
    
    def _get_cached_features(self, cache_key: str, symbol: str) -> Optional[str]:
        """Check if features are cached and return file path."""
        if self.use_pg_manifest:
            return self._get_cached_features_pg(cache_key, symbol)
        else:
            return self._get_cached_features_duckdb(cache_key, symbol)
    
    def _get_cached_features_pg(self, cache_key: str, symbol: str) -> Optional[str]:
        """Check if features are cached using PostgreSQL manifest."""
        
        try:
            with MANIFEST_READ_LATENCY.labels(backend='postgresql', symbol=symbol).time():
                # Update pool metrics
                _update_pool_metrics()
                with pg_conn() as conn:
                    # Use DictCursor to get dictionary results instead of tuples
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(
                            "SELECT path FROM manifest WHERE key = %s", 
                            (cache_key,)
                        )
                        result = cur.fetchone()
                        
                        if result:
                            path = result['path']
                            if Path(path).exists():
                                # Update last accessed timestamp - let context manager handle transaction
                                cur.execute(
                                    "UPDATE manifest SET last_accessed_ts = CURRENT_TIMESTAMP WHERE key = %s",
                                    (cache_key,)
                                )
                                # No manual commit - context manager handles it
                                return path
                            else:
                                # File missing - clean up manifest entry
                                self.logger.warning(f"Cached file missing: {path}")
                                cur.execute("DELETE FROM manifest WHERE key = %s", (cache_key,))
                                # No manual commit - context manager handles it
                        
                        return None
                    
        except Exception as e:
            self.logger.error(f"Error checking PostgreSQL cache for key {cache_key[:16]}...: {e}")
            return None
    
    def _get_cached_features_duckdb(self, cache_key: str, symbol: str) -> Optional[str]:
        """Check if features are cached using DuckDB manifest."""
        
        try:
            from ..utils.db import get_conn
            with MANIFEST_READ_LATENCY.labels(backend='duckdb', symbol=symbol).time():
                # Use read-only connection for cache lookups
                with get_conn(read_only=True) as conn:
                    result = conn.execute(
                        "SELECT path FROM manifest WHERE key = ?", 
                        [cache_key]
                    ).fetchone()
                    
                    if result:
                        path = result[0]
                        if Path(path).exists():
                            # Update last accessed timestamp (skip in read-only mode)
                            if not self.read_only:
                                try:
                                    from ..utils.db import get_write_conn
                                    with get_write_conn() as write_conn:
                                        write_conn.execute(
                                            "UPDATE manifest SET last_accessed_ts = CURRENT_TIMESTAMP WHERE key = ?",
                                            [cache_key]
                                        )
                                except Exception as update_error:
                                    # Don't fail cache hit if timestamp update fails
                                    self.logger.debug(f"Failed to update access timestamp: {update_error}")
                            return path
                        else:
                            # File missing - clean up manifest entry (skip in read-only mode)
                            self.logger.warning(f"Cached file missing: {path}")
                            if not self.read_only:
                                try:
                                    from ..utils.db import get_write_conn
                                    with get_write_conn() as write_conn:
                                        write_conn.execute("DELETE FROM manifest WHERE key = ?", [cache_key])
                                except Exception as cleanup_error:
                                    self.logger.warning(f"Failed to cleanup missing cache entry: {cleanup_error}")
                    
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
            # Clean up bad cache entry from both PostgreSQL and DuckDB
            try:
                if self.use_pg_manifest:
                    with pg_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM manifest WHERE key = %s", (cache_key,))
                else:
                    with get_duckdb_connection(self.db_path, mode='rw') as conn:
                        conn.execute("DELETE FROM manifest WHERE key = ?", [cache_key])
            except Exception as cleanup_error:
                self.logger.warning(f"Error cleaning up bad cache entry: {cleanup_error}")
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
            
            # Convert to parquet once, then compress
            parquet_bytes = features_df.to_parquet(index=True)
            
            # Save with zstandard compression (level 3 for good speed/compression balance)
            compressor = zstd.ZstdCompressor(level=3)
            with open(cache_file, 'wb') as raw_fh:
                with compressor.stream_writer(raw_fh) as fh:
                    fh.write(parquet_bytes)
            
            file_size = cache_file.stat().st_size
            
            # Use PostgreSQL with advisory lock for high-concurrency INSERT
            # Include start_ts and end_ts in lock key for better granularity
            lock_key = symbol_to_lock_key(f"{symbol}_{start_ts}_{end_ts}")
            
            # Update pool metrics
            _update_pool_metrics()
            with pg_conn() as conn:
                # Let context manager handle transaction - remove manual BEGIN/COMMIT
                with conn.cursor() as cur:
                    # Measure advisory lock wait time
                    with ADVISORY_LOCK_WAIT_TIME.labels(symbol=symbol).time():
                        # Acquire advisory lock for this symbol+timerange to prevent row-lock contention
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
                    
                    self.logger.info(f"Cached {len(features_df)} features for {symbol} "
                                   f"({file_size:,} bytes compressed) [PostgreSQL+AdvisoryLock]")
                
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
            
            # Convert to parquet once, then compress
            parquet_bytes = features_df.to_parquet(index=True)
            
            # Save with zstandard compression (level 3 for good speed/compression balance)
            compressor = zstd.ZstdCompressor(level=3)
            with open(cache_file, 'wb') as raw_fh:
                with compressor.stream_writer(raw_fh) as fh:
                    fh.write(parquet_bytes)
            
            file_size = cache_file.stat().st_size
            
            # Use write connection with file locking
            from ..utils.db import get_write_conn
            with MANIFEST_INSERT_LATENCY.labels(backend='duckdb', symbol=symbol).time():
                with get_write_conn() as conn:
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
                            access_count = manifest.access_count + 1
                    """, [
                        cache_key, str(cache_file), symbol, start_ts, end_ts, 
                        len(features_df), file_size
                    ])
                    
                    self.logger.info(f"Cached {len(features_df)} features for {symbol} "
                                   f"({file_size:,} bytes compressed) [DuckDB]")
                
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
            if self.use_pg_manifest:
                return self._get_cache_stats_pg()
            else:
                return self._get_cache_stats_duckdb()
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def _get_cache_stats_pg(self) -> Dict[str, Any]:
        """Get cache statistics from PostgreSQL manifest."""
        with pg_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        SUM(rows) as total_cached_rows,
                        SUM(file_size_bytes) as total_size_bytes,
                        MIN(created_ts) as oldest_entry,
                        MAX(created_ts) as newest_entry
                    FROM manifest
                """)
                stats = cur.fetchone()
                
                if stats:
                    return {
                        'total_entries': stats['total_entries'],
                        'unique_symbols': stats['unique_symbols'], 
                        'total_cached_rows': stats['total_cached_rows'] or 0,
                        'total_size_mb': round((stats['total_size_bytes'] or 0) / 1024 / 1024, 2),
                        'oldest_entry': stats['oldest_entry'],
                        'newest_entry': stats['newest_entry'],
                        'cache_directory': str(self.base),
                        'backend': 'postgresql'
                    }
                else:
                    return {'total_entries': 0, 'backend': 'postgresql'}
    
    def _get_cache_stats_duckdb(self) -> Dict[str, Any]:
        """Get cache statistics from DuckDB manifest."""
        from ..utils.db import get_conn
        with get_conn(read_only=True) as conn:
            stats = conn.execute("""
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
                'cache_directory': str(self.base),
                'backend': 'duckdb'
            }
        else:
            return {'total_entries': 0, 'backend': 'duckdb'}
    
    def cleanup_old_entries(self, days_old: int = 30):
        """
        Clean up cache entries older than specified days.
        
        Args:
            days_old: Remove entries older than this many days
        """
        try:
            if self.use_pg_manifest:
                self._cleanup_old_entries_pg(days_old)
            else:
                self._cleanup_old_entries_duckdb(days_old)
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    def _cleanup_old_entries_pg(self, days_old: int):
        """Clean up old entries from PostgreSQL manifest."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with pg_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get files to delete
                cur.execute("""
                    SELECT key, path FROM manifest 
                    WHERE last_accessed_ts < %s
                """, (cutoff_date,))
                old_entries = cur.fetchall()
                
                if not old_entries:
                    self.logger.info("No old cache entries to clean up [PostgreSQL]")
                    return
                
                deleted_count = 0
                freed_bytes = 0
                keys_to_delete = []
                
                # Delete physical files and collect keys for bulk delete
                for entry in old_entries:
                    try:
                        file_path = Path(entry['path'])
                        if file_path.exists():
                            freed_bytes += file_path.stat().st_size
                            file_path.unlink()
                        
                        keys_to_delete.append(entry['key'])
                        deleted_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error deleting cache entry {entry['key']}: {e}")
                
                # Bulk delete from manifest for better performance
                # Note: For very large lists (10k+ entries), consider batching to avoid
                # PostgreSQL parameter limits, but this is acceptable for typical use cases
                if keys_to_delete:
                    cur.execute("""
                        DELETE FROM manifest 
                        WHERE key = ANY(%s)
                    """, (keys_to_delete,))
                
                self.logger.info(f"Cleaned up {deleted_count} old cache entries, "
                               f"freed {freed_bytes / 1024 / 1024:.1f} MB [PostgreSQL]")
    
    def _cleanup_old_entries_duckdb(self, days_old: int):
        """Clean up old entries from DuckDB manifest."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Get files to delete
        with get_duckdb_connection(self.db_path, mode='r') as conn:
            old_entries = conn.execute("""
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
                
                with get_duckdb_connection(self.db_path, mode='rw') as write_conn:
                    write_conn.execute("DELETE FROM manifest WHERE key = ?", [key])
                deleted_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error deleting cache entry {key}: {e}")
        
        self.logger.info(f"Cleaned up {deleted_count} old cache entries, "
                       f"freed {freed_bytes / 1024 / 1024:.1f} MB [DuckDB]")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            symbol: If provided, only clear entries for this symbol
        """
        try:
            if self.use_pg_manifest:
                self._clear_cache_pg(symbol)
            else:
                self._clear_cache_duckdb(symbol)
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def _clear_cache_pg(self, symbol: Optional[str] = None):
        """Clear cache entries from PostgreSQL manifest."""
        with pg_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if symbol:
                    cur.execute("SELECT path FROM manifest WHERE symbol = %s", (symbol,))
                    entries = cur.fetchall()
                    cur.execute("DELETE FROM manifest WHERE symbol = %s", (symbol,))
                    self.logger.info(f"Cleared cache for symbol {symbol} [PostgreSQL]")
                else:
                    cur.execute("SELECT path FROM manifest")
                    entries = cur.fetchall()
                    cur.execute("DELETE FROM manifest")
                    self.logger.info("Cleared entire cache [PostgreSQL]")
                
                # Delete physical files
                deleted_count = 0
                for entry in entries:
                    try:
                        Path(entry['path']).unlink(missing_ok=True)
                        deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error deleting file {entry['path']}: {e}")
                
                self.logger.info(f"Deleted {deleted_count} cache files [PostgreSQL]")
    
    def _clear_cache_duckdb(self, symbol: Optional[str] = None):
        """Clear cache entries from DuckDB manifest."""
        with get_duckdb_connection(self.db_path, mode='rw') as conn:
            if symbol:
                entries = conn.execute(
                    "SELECT path FROM manifest WHERE symbol = ?", [symbol]
                ).fetchall()
                conn.execute("DELETE FROM manifest WHERE symbol = ?", [symbol])
                self.logger.info(f"Cleared cache for symbol {symbol} [DuckDB]")
            else:
                entries = conn.execute("SELECT path FROM manifest").fetchall()
                conn.execute("DELETE FROM manifest")
                self.logger.info("Cleared entire cache [DuckDB]")
        
        # Delete physical files
        deleted_count = 0
        for (path,) in entries:
            try:
                Path(path).unlink(missing_ok=True)
                deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Error deleting file {path}: {e}")
        
        self.logger.info(f"Deleted {deleted_count} cache files [DuckDB]")
    
    def close(self):
        """Close database connections and cleanup resources."""
        try:
            if not self.use_pg_manifest:
                # Close any DuckDB connections for this database
                close_all_duckdb_connections()
                self.logger.info("FeatureStore DuckDB connections closed")
        except Exception as e:
            self.logger.warning(f"Error closing FeatureStore connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for global feature store instance
_global_feature_store = None

def get_feature_store(root: Optional[str] = None, logger: Optional[logging.Logger] = None, read_only: bool = False) -> FeatureStore:
    """Get global FeatureStore instance."""
    global _global_feature_store
    
    # If read_only mode is requested but existing instance is not read_only, create new instance
    if _global_feature_store is not None and read_only and not _global_feature_store.read_only:
        _global_feature_store = None
    
    if _global_feature_store is None:
        _global_feature_store = FeatureStore(root=root, logger=logger, read_only=read_only)
    return _global_feature_store


def reset_feature_store():
    """Reset global FeatureStore instance."""
    global _global_feature_store
    _global_feature_store = None