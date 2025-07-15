#!/usr/bin/env python3
"""
Verify that the FeatureStore DuckDB concurrency fix works correctly.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.feature_store import FeatureStore
from src.shared.duckdb_manager import close_write_duckdb_connections, get_duckdb_connection_info

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_features(rows: int = 100) -> pd.DataFrame:
    """Create sample feature data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=rows, freq='1min'),
        'close': 100 + np.random.randn(rows).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, rows),
        'rsi': np.random.uniform(20, 80, rows),
        'ema_20': 100 + np.random.randn(rows).cumsum() * 0.05,
        'position': np.random.choice([-1, 0, 1], rows)
    })


def test_feature_store_concurrency():
    """Test FeatureStore operations with the concurrency fix."""
    logger.info("🧪 Testing FeatureStore DuckDB concurrency fix...")
    
    # Create temporary feature store
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FeatureStore(root=temp_dir)
        
        # Test 1: Cache features (simulating training phase)
        logger.info("📝 Phase 1: Caching features (training simulation)...")
        features_df = create_sample_features(100)
        
        def dummy_compute_func(df, config):
            return df  # Just return the same data
        
        config = {'symbol': 'NVDA', 'interval': '1min'}
        
        # Cache the features
        cached_features = store.get_or_compute(
            symbol='NVDA',
            raw_df=features_df,
            config=config,
            compute_func=dummy_compute_func
        )
        
        logger.info(f"✅ Cached {len(cached_features)} feature rows")
        
        # Test 2: Close write connections (simulating training-evaluation transition)
        logger.info("🔒 Phase 2: Closing write connections...")
        close_write_duckdb_connections()
        
        # Test 3: Read from cache (simulating evaluation phase)
        logger.info("📖 Phase 3: Reading from cache (evaluation simulation)...")
        
        # Try to get the same features (should hit cache)
        cached_features_2 = store.get_or_compute(
            symbol='NVDA',
            raw_df=features_df,
            config=config,
            compute_func=dummy_compute_func
        )
        
        logger.info(f"✅ Retrieved {len(cached_features_2)} feature rows from cache")
        
        # Verify data integrity
        if cached_features.equals(cached_features_2):
            logger.info("✅ Cache hit successful - data integrity verified")
        else:
            logger.error("❌ Cache data mismatch!")
            return False
        
        # Test 4: Check cache stats
        logger.info("📊 Phase 4: Checking cache statistics...")
        stats = store.get_cache_stats()
        logger.info(f"Cache stats: {stats}")
        
        # Test 5: Connection info
        conn_info = get_duckdb_connection_info()
        logger.info(f"Connection info: {conn_info}")
        
        # Test 6: Explicit cleanup
        logger.info("🧹 Phase 6: Final cleanup...")
        store.close()
        from src.shared.duckdb_manager import close_all_duckdb_connections
        close_all_duckdb_connections()
        
        logger.info("🎉 FeatureStore concurrency fix verification PASSED!")
        return True


def main():
    """Run the verification test."""
    try:
        success = test_feature_store_concurrency()
        if success:
            logger.info("✅ ALL VERIFICATIONS PASSED!")
            return 0
        else:
            logger.error("❌ VERIFICATION FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())