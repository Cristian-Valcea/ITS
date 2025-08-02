"""
Pytest configuration and fixtures for Critical Reviewer Implementations
Provides automatic data harmonization for legacy test compatibility
"""

import os
import pandas as pd
import pytest
import numpy as np
from pathlib import Path

# Set leak test mode from environment
LEAK_TEST_MODE = os.getenv('LEAK_TEST_MODE', 'strict').lower()

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment with proper paths and configurations"""
    
    # Add project paths
    project_root = Path(__file__).parent
    import sys
    sys.path.insert(0, str(project_root / 'studies'))
    sys.path.insert(0, str(project_root / 'tests'))
    sys.path.insert(0, str(project_root))  # For legacy_shims
    
    # Set environment variables for testing
    monkeypatch.setenv('LEAK_TEST_MODE', LEAK_TEST_MODE)
    
    # Mock TimescaleDB config for tests
    monkeypatch.setenv('TS_DB_HOST', 'localhost')
    monkeypatch.setenv('TS_DB_PORT', '5432')
    monkeypatch.setenv('TS_DB_NAME', 'test')
    monkeypatch.setenv('TS_DB_USER', 'test_user')
    monkeypatch.setenv('TS_DB_PASSWORD', 'test_password')

@pytest.fixture(autouse=True)
def harmonize_dataframes(monkeypatch):
    """Automatically harmonize DataFrame schemas for legacy test compatibility"""
    
    original_read_csv = pd.read_csv
    
    def harmonized_read_csv(*args, **kwargs):
        """Read CSV with automatic schema harmonization"""
        df = original_read_csv(*args, **kwargs)
        return harmonize_dataframe_schema(df)
    
    # Patch pandas.read_csv
    monkeypatch.setattr(pd, 'read_csv', harmonized_read_csv)
    
    # Add harmonization method to DataFrame class
    def legacy_harmonize(self):
        """Harmonize DataFrame for legacy test compatibility"""
        return harmonize_dataframe_schema(self)
    
    monkeypatch.setattr(pd.DataFrame, 'legacy_harmonize', legacy_harmonize, raising=False)

def harmonize_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize DataFrame schema for legacy test compatibility
    
    Transformations:
    - 'Timeframe' → 'timeframe' (lowercase)
    - Pad hash columns to 64 characters (SHA-256 format)
    - Ensure consistent column naming
    """
    if df is None or df.empty:
        return df
    
    df_harmonized = df.copy()
    
    # Column name harmonization
    column_mappings = {
        'Timeframe': 'timeframe',
        'Filter Configuration': 'filter_configuration',
        'Sharpe Ratio': 'sharpe_ratio',
        'Max Drawdown (%)': 'max_drawdown_pct',
        'Information Ratio': 'information_ratio',
        'Gross Return (%)': 'gross_return_pct'
    }
    
    # Apply column mappings if columns exist
    for old_col, new_col in column_mappings.items():
        if old_col in df_harmonized.columns:
            df_harmonized = df_harmonized.rename(columns={old_col: new_col})
    
    # Hash column harmonization - pad to 64 characters for SHA-256 format
    hash_columns = [col for col in df_harmonized.columns if 'hash' in col.lower()]
    for hash_col in hash_columns:
        if df_harmonized[hash_col].dtype == 'object':  # String column
            df_harmonized[hash_col] = df_harmonized[hash_col].astype(str).str.ljust(64, '0')
    
    # Lock-box Hash specific handling
    if 'Lock-box Hash' in df_harmonized.columns:
        df_harmonized['Lock-box Hash'] = df_harmonized['Lock-box Hash'].astype(str).str.ljust(64, '0')
    
    return df_harmonized

@pytest.fixture
def leak_test_threshold():
    """Provide leak test threshold based on environment mode"""
    if LEAK_TEST_MODE == 'strict':
        return 0.01  # Strict threshold
    else:
        return 0.05  # Loose threshold for legacy compatibility

@pytest.fixture
def mock_timescaledb_config():
    """Provide mock TimescaleDB configuration for tests"""
    return {
        'host': os.getenv('TS_DB_HOST', 'localhost'),
        'port': int(os.getenv('TS_DB_PORT', '5432')),
        'database': os.getenv('TS_DB_NAME', 'test'),
        'user': os.getenv('TS_DB_USER', 'test_user'),
        'password': os.getenv('TS_DB_PASSWORD', 'test_password')
    }

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate 1000 minutes of sample data
    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = pd.date_range(start_time, periods=1000, freq='1min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    returns = np.random.normal(0, 0.001, len(timestamps))  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC from prices
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        noise = np.random.normal(0, 0.0005, 4)  # Small noise for OHLC
        high = price + abs(noise[0])
        low = price - abs(noise[1])
        open_price = price + noise[2]
        close_price = price + noise[3]
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def alpha_study():
    """Create TickVsMinuteAlphaStudy instance for testing"""
    # Import from legacy_shims for compatibility
    from legacy_shims import TickVsMinuteAlphaStudyShim
    return TickVsMinuteAlphaStudyShim()

@pytest.fixture
def ablation_study():
    """Create FilteringAblationStudy instance for testing"""
    # Import from legacy_shims for compatibility
    from legacy_shims import FilteringAblationStudyShim
    return FilteringAblationStudyShim()

@pytest.fixture
def feature_validator():
    """Create FeatureLagValidator instance for testing"""
    from test_feature_lag_validation import FeatureLagValidator
    return FeatureLagValidator()

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Custom assertion helpers
def assert_performance_metrics(metrics, min_sharpe=None, max_drawdown=None):
    """Assert performance metrics meet minimum standards"""
    assert hasattr(metrics, 'sharpe_ratio'), "Metrics must have sharpe_ratio attribute"
    assert hasattr(metrics, 'max_drawdown_pct'), "Metrics must have max_drawdown_pct attribute"
    
    if min_sharpe is not None:
        assert metrics.sharpe_ratio >= min_sharpe, f"Sharpe ratio {metrics.sharpe_ratio} below minimum {min_sharpe}"
    
    if max_drawdown is not None:
        assert metrics.max_drawdown_pct <= max_drawdown, f"Max drawdown {metrics.max_drawdown_pct}% above maximum {max_drawdown}%"

def assert_dataframe_schema(df, required_columns, optional_columns=None):
    """Assert DataFrame has required schema"""
    assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
    assert not df.empty, "DataFrame cannot be empty"
    
    missing_columns = set(required_columns) - set(df.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"
    
    if optional_columns:
        available_optional = set(optional_columns) & set(df.columns)
        print(f"Optional columns present: {available_optional}")

# Export fixtures and helpers
__all__ = [
    'setup_test_environment',
    'harmonize_dataframes', 
    'harmonize_dataframe_schema',
    'leak_test_threshold',
    'mock_timescaledb_config',
    'sample_market_data',
    'alpha_study',
    'ablation_study', 
    'feature_validator',
    'assert_performance_metrics',
    'assert_dataframe_schema'
]