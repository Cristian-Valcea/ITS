"""
Evaluation-specific caching to prevent DuckDB re-locking during evaluation.

This module provides in-memory caching of feature DataFrames during evaluation
to avoid the "thundering herd" of DuckDB connections that can cause file locking issues.
"""

import pandas as pd
import logging
from typing import Dict, Optional, Any
from threading import Lock
import hashlib


class EvaluationFeatureCache:
    """
    In-memory cache for feature DataFrames during evaluation.
    
    This prevents multiple DuckDB connections during evaluation episodes,
    which can cause Windows file locking issues.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._lock = Lock()
        self._enabled = False
        
    def enable(self):
        """Enable caching for evaluation mode."""
        with self._lock:
            self._enabled = True
            self.logger.info("🔒 Evaluation feature cache ENABLED - preventing DuckDB re-connections")
    
    def disable(self):
        """Disable caching and clear cache."""
        with self._lock:
            self._enabled = False
            self._cache.clear()
            self.logger.info("🔓 Evaluation feature cache DISABLED and cleared")
    
    def _generate_cache_key(self, symbol: str, start_date: str, end_date: str, 
                           interval: str, feature_config: Dict[str, Any]) -> str:
        """Generate a cache key for the feature request."""
        # Create a deterministic key based on parameters
        key_data = f"{symbol}_{start_date}_{end_date}_{interval}_{str(sorted(feature_config.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get(self, symbol: str, start_date: str, end_date: str, 
            interval: str, feature_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get cached features if available."""
        if not self._enabled:
            return None
            
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, feature_config)
        
        with self._lock:
            if cache_key in self._cache:
                self.logger.debug(f"📋 Cache HIT for {symbol} features ({cache_key})")
                # Return deep copy of tuple contents to prevent modification
                cached_result = self._cache[cache_key]
                if isinstance(cached_result, tuple) and len(cached_result) == 3:
                    df, sequences, prices = cached_result
                    return (
                        df.copy() if df is not None else None,
                        sequences.copy() if sequences is not None else None,
                        prices.copy() if prices is not None else None
                    )
                return cached_result
            else:
                self.logger.debug(f"📋 Cache MISS for {symbol} features ({cache_key})")
                return None
    
    def put(self, symbol: str, start_date: str, end_date: str, 
            interval: str, feature_config: Dict[str, Any], features_df: pd.DataFrame):
        """Cache features for future use."""
        if not self._enabled:
            return
            
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, feature_config)
        
        with self._lock:
            # Store a deep copy to prevent external modification
            if isinstance(features_df, tuple) and len(features_df) == 3:
                df, sequences, prices = features_df
                cached_tuple = (
                    df.copy() if df is not None else None,
                    sequences.copy() if sequences is not None else None,
                    prices.copy() if prices is not None else None
                )
                self._cache[cache_key] = cached_tuple
                
                # Calculate memory usage
                memory_mb = 0
                if df is not None:
                    memory_mb += df.memory_usage(deep=True).sum() / 1024 / 1024
                if sequences is not None:
                    memory_mb += sequences.nbytes / 1024 / 1024
                if prices is not None:
                    memory_mb += prices.memory_usage(deep=True) / 1024 / 1024
                    
                self.logger.info(f"📋 Cached {symbol} features ({cache_key}) - {memory_mb:.1f} MB")
            else:
                # Fallback for non-tuple data
                self._cache[cache_key] = features_df
                self.logger.info(f"📋 Cached {symbol} features ({cache_key})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_memory = 0
            for cached_item in self._cache.values():
                if isinstance(cached_item, tuple) and len(cached_item) == 3:
                    df, sequences, prices = cached_item
                    if df is not None:
                        total_memory += df.memory_usage(deep=True).sum()
                    if sequences is not None:
                        total_memory += sequences.nbytes
                    if prices is not None:
                        total_memory += prices.memory_usage(deep=True)
                else:
                    # Fallback for non-tuple items
                    if hasattr(cached_item, 'memory_usage'):
                        total_memory += cached_item.memory_usage(deep=True).sum()
            
            return {
                "enabled": self._enabled,
                "entries": len(self._cache),
                "total_memory_mb": total_memory / 1024 / 1024,
                "cache_keys": list(self._cache.keys())
            }


# Global instance for evaluation caching
_evaluation_cache = EvaluationFeatureCache()

def get_evaluation_cache() -> EvaluationFeatureCache:
    """Get the global evaluation cache instance."""
    return _evaluation_cache