# src/shared/__init__.py
"""Shared utilities and constants for the IntradayJules trading system."""

from . import constants

# Optional imports that may require additional dependencies
try:
    from .feature_store import FeatureStore, get_feature_store
    FEATURE_STORE_AVAILABLE = True
    __all__ = ["constants", "FeatureStore", "get_feature_store"]
except ImportError:
    # Create mock implementations for missing dependencies
    FEATURE_STORE_AVAILABLE = False
    
    class MockFeatureStore:
        """Mock FeatureStore for when duckdb is not available."""
        def __init__(self, *args, **kwargs):
            import warnings
            warnings.warn("FeatureStore not available - duckdb dependency missing", UserWarning)
        
        def __getattr__(self, name):
            raise NotImplementedError(f"FeatureStore method '{name}' not available - duckdb dependency missing")
    
    def get_feature_store(*args, **kwargs):
        """Mock get_feature_store for when duckdb is not available."""
        return MockFeatureStore(*args, **kwargs)
    
    # Export mock implementations
    FeatureStore = MockFeatureStore
    __all__ = ["constants", "FeatureStore", "get_feature_store"]
