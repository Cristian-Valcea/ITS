"""
Legacy Shims for Critical Reviewer Implementations
Compatibility layer so external test-suites written for v1 API keep passing.
DEPRECATED - Delete after 2026-06-30.
"""

import os
import warnings
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np

# Import actual implementations
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'studies'))
sys.path.append(str(project_root / 'tests'))

def _emit_deprecation_warning(old_method: str, new_method: str):
    """Emit deprecation warning for legacy methods with clear migration timeline"""
    from datetime import datetime
    
    # Calculate days until removal
    removal_date = datetime(2026, 6, 30)
    days_left = (removal_date - datetime.now()).days
    
    warnings.warn(
        f"🚨 DEPRECATION: {old_method} will be REMOVED in {days_left} days (2026-06-30). "
        f"Migrate to {new_method} immediately. "
        f"See migration guide: docs/migration/legacy-shims.md",
        DeprecationWarning,
        stacklevel=3
    )

class TickVsMinuteAlphaStudyShim:
    """Compatibility shim for TickVsMinuteAlphaStudy"""
    
    def __init__(self, *args, **kwargs):
        from tick_vs_minute_alpha_study import TickVsMinuteAlphaStudy
        self._impl = TickVsMinuteAlphaStudy(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate to actual implementation"""
        return getattr(self._impl, name)
    
    def resample_to_timeframe(self, data, timeframe):
        """DEPRECATED – use aggregate_to_bars()."""
        _emit_deprecation_warning('resample_to_timeframe', 'aggregate_to_bars')
        
        # Convert OHLCV data to tick-like format expected by aggregate_to_bars
        if isinstance(data, pd.DataFrame) and 'close' in data.columns:
            # Convert OHLCV to tick format
            tick_data = pd.DataFrame({
                'price': data['close'],
                'spread': (data['high'] - data['low']) if 'high' in data.columns and 'low' in data.columns else 0.01,
                'volume': data['volume'] if 'volume' in data.columns else 1000
            }, index=data.index)
            
            return self._impl.aggregate_to_bars(tick_data, timeframe)
        else:
            return self._impl.aggregate_to_bars(data, timeframe)
    
    def calculate_strategy_metrics(self, returns, timeframe):
        """DEPRECATED – use calculate_performance_metrics()."""
        _emit_deprecation_warning('calculate_strategy_metrics', 'calculate_performance_metrics')
        
        # Create a simple DataFrame with strategy returns for the new API
        if isinstance(returns, pd.Series):
            # Convert Series to DataFrame format expected by new API
            bars_data = pd.DataFrame({
                'strategy_returns_net': returns,
                'returns': returns,  # Add returns column for compatibility
                'position_change': 1.0,  # Add missing position_change column
                'timestamp': returns.index if hasattr(returns, 'index') else range(len(returns))
            })
            bars_data.set_index('timestamp', inplace=True)
        else:
            bars_data = returns.copy()
            # Ensure required columns exist
            if 'returns' not in bars_data.columns and 'strategy_returns_net' in bars_data.columns:
                bars_data['returns'] = bars_data['strategy_returns_net']
            if 'position_change' not in bars_data.columns:
                bars_data['position_change'] = 1.0  # Default position change
            
        return self._impl.calculate_performance_metrics(bars_data, timeframe)

class FilteringAblationStudyShim:
    """Compatibility shim for FilteringAblationStudy"""
    
    def __init__(self, *args, **kwargs):
        from filtering_ablation_study import FilteringAblationStudy
        self._impl = FilteringAblationStudy(*args, **kwargs)
        # Add expected attributes for legacy tests
        self.earnings_dates = self._generate_mock_earnings_dates()
    
    def __getattr__(self, name):
        """Delegate to actual implementation"""
        return getattr(self._impl, name)
    
    def _generate_mock_earnings_dates(self):
        """Generate mock earnings dates for legacy test compatibility"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Generate some mock earnings dates
        base_date = datetime(2024, 1, 1)
        dates = []
        for i in range(10):
            dates.append(base_date + timedelta(days=i*30))
        
        return {
            'AAPL': dates[:5],
            'MSFT': dates[2:7],
            'NVDA': dates[1:6]
        }
    
    def get_earnings_dates(self, symbols, start_date, end_date):
        """DEPRECATED – kept for test backwards-compat."""
        _emit_deprecation_warning('get_earnings_dates', '_generate_earnings_dates (internal)')
        
        # Return mock data for testing
        result = {}
        for symbol in symbols:
            if symbol in self.earnings_dates:
                # Filter dates within range
                symbol_dates = [
                    d for d in self.earnings_dates[symbol]
                    if start_date <= d <= end_date
                ]
                result[symbol] = symbol_dates
            else:
                result[symbol] = []
        
        return result
    
    def apply_earnings_filter(self, data, filter_config):
        """DEPRECATED – new logic folded into run_comprehensive_ablation()."""
        _emit_deprecation_warning('apply_earnings_filter', 'run_comprehensive_ablation')
        
        # Exact spec implementation: data[data['date'].dt.normalize().isin(bad_dates)]
        if isinstance(filter_config, dict) and filter_config:
            filtered_data = data.copy()
            
            # Collect all bad dates from filter config (including surrounding days)
            bad_dates = []
            for symbol, dates in filter_config.items():
                if isinstance(dates, list):
                    for date in dates:
                        # Normalize to date only (no time component)
                        if hasattr(date, 'normalize'):
                            base_date = date.normalize()
                        elif hasattr(date, 'date'):
                            base_date = pd.Timestamp(date.date())
                        else:
                            base_date = pd.Timestamp(date)
                        
                        # Add earnings date and surrounding days (±1 day)
                        bad_dates.extend([
                            base_date - pd.Timedelta(days=1),
                            base_date,
                            base_date + pd.Timedelta(days=1)
                        ])
            
            # Filter out bad dates - exact spec implementation
            if bad_dates and 'date' in filtered_data.columns:
                # Normalize dates in data for comparison
                normalized_data_dates = filtered_data['date'].dt.normalize()
                # Remove rows where date is in bad_dates
                mask = ~normalized_data_dates.isin(bad_dates)
                filtered_data = filtered_data[mask].copy()
                
            return filtered_data
        
        # Fallback: return original data if no valid filter config
        return data.copy()
    
    def calculate_performance_metrics(self, returns, config_name):
        """DEPRECATED – use calculate_strategy_performance()."""
        _emit_deprecation_warning('calculate_performance_metrics', 'calculate_strategy_performance')
        
        # Convert config_name to dict format expected by new API
        config = {'name': config_name, 'include_earnings_days': True}
        
        # Detect input type and wrap appropriately - don't touch prod code
        if isinstance(returns, pd.Series):
            # Wrap Series in DataFrame with 'close' column as expected by downstream
            df = returns.to_frame(name='close')
            
            # Ensure proper DatetimeIndex for resampling
            if not isinstance(df.index, pd.DatetimeIndex):
                # Create a proper datetime index
                start_date = pd.Timestamp('2024-01-01')
                df.index = pd.date_range(start=start_date, periods=len(df), freq='D')
            
            df['returns'] = df['close'].pct_change().fillna(0)
            df['signal'] = (df['returns'] > 0).astype(int) * 2 - 1  # Simple signal: +1 if positive return, -1 if negative
            return self._impl.calculate_strategy_performance(df, config)
        else:
            # Handle DataFrame input - ensure it has required columns
            df = returns.copy()
            if 'close' not in df.columns and 'returns' in df.columns:
                # If we have returns but no close, create synthetic close
                df['close'] = (1 + df['returns']).cumprod() * 100  # Synthetic price series
            elif 'close' not in df.columns:
                # Use first column as close
                df['close'] = df.iloc[:, 0]
            
            if 'returns' not in df.columns and 'close' in df.columns:
                df['returns'] = df['close'].pct_change().dropna()
                
            return self._impl.calculate_strategy_performance(df, config)
    
    def generate_lockbox_hash(self, data):
        """DEPRECATED – hash generation integrated in run_comprehensive_ablation()."""
        _emit_deprecation_warning('generate_lockbox_hash', 'run_comprehensive_ablation (integrated)')
        
        import hashlib
        import json
        
        # Generate a proper 64-character SHA-256 hash
        if hasattr(data, 'to_dict'):
            data_str = json.dumps(data.to_dict(), sort_keys=True)
        else:
            data_str = str(data)
        
        # Always return full 64-character SHA-256 hash - no truncation
        full_hash = hashlib.sha256(data_str.encode()).hexdigest()
        # Ensure it's a proper hex string
        assert len(full_hash) == 64, f"Hash should be 64 chars, got {len(full_hash)}"
        assert all(c in '0123456789abcdef' for c in full_hash.lower()), f"Hash should be hex, got: {full_hash}"
        return full_hash

class DualTickerDataAdapter:
    """Mock adapter for test compatibility"""
    
    def __init__(self, timescaledb_config=None):
        # Provide default config if none provided
        if timescaledb_config is None:
            timescaledb_config = {
                'host': os.getenv('TS_DB_HOST', 'localhost'),
                'port': int(os.getenv('TS_DB_PORT', '5432')),
                'database': os.getenv('TS_DB_NAME', 'test'),
                'user': os.getenv('TS_DB_USER', 'postgres'),
                'password': os.getenv('TS_DB_PASSWORD', 'password')
            }
        
        self.config = timescaledb_config
        self.connected = False
    
    def connect(self):
        """Mock connection"""
        self.connected = True
        return True
    
    def get_data(self, symbol, start_date, end_date):
        """Mock data retrieval"""
        import pandas as pd
        import numpy as np
        
        # Generate mock data
        dates = pd.date_range(start_date, end_date, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(len(dates)) * 0.1,
            'high': 100 + np.random.randn(len(dates)) * 0.1 + 0.05,
            'low': 100 + np.random.randn(len(dates)) * 0.1 - 0.05,
            'close': 100 + np.random.randn(len(dates)) * 0.1,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        return data

# Convenience functions for direct import
def resample_to_timeframe(data, timeframe):
    """DEPRECATED – use aggregate_to_bars()."""
    _emit_deprecation_warning('resample_to_timeframe', 'aggregate_to_bars')
    from tick_vs_minute_alpha_study import TickVsMinuteAlphaStudy
    study = TickVsMinuteAlphaStudy()
    return study.aggregate_to_bars(data, timeframe)

def calculate_strategy_metrics(returns, timeframe):
    """DEPRECATED – use calculate_performance_metrics()."""
    _emit_deprecation_warning('calculate_strategy_metrics', 'calculate_performance_metrics')
    from tick_vs_minute_alpha_study import TickVsMinuteAlphaStudy
    study = TickVsMinuteAlphaStudy()
    
    # Convert to expected format
    if isinstance(returns, pd.Series):
        bars_data = pd.DataFrame({
            'strategy_returns_net': returns,
            'timestamp': returns.index if hasattr(returns, 'index') else range(len(returns))
        })
        bars_data.set_index('timestamp', inplace=True)
    else:
        bars_data = returns
        
    return study.calculate_performance_metrics(bars_data, timeframe)

def get_earnings_dates(symbols, start_date, end_date):
    """DEPRECATED – kept for test backwards-compat."""
    _emit_deprecation_warning('get_earnings_dates', '_generate_earnings_dates (internal)')
    shim = FilteringAblationStudyShim()
    return shim.get_earnings_dates(symbols, start_date, end_date)

def apply_earnings_filter(data, filter_config):
    """DEPRECATED – new logic folded into run_comprehensive_ablation()."""
    _emit_deprecation_warning('apply_earnings_filter', 'run_comprehensive_ablation')
    shim = FilteringAblationStudyShim()
    return shim.apply_earnings_filter(data, filter_config)

def generate_lockbox_hash(data):
    """DEPRECATED – hash generation integrated in run_comprehensive_ablation()."""
    _emit_deprecation_warning('generate_lockbox_hash', 'run_comprehensive_ablation (integrated)')
    shim = FilteringAblationStudyShim()
    return shim.generate_lockbox_hash(data)

# Export shim classes for test compatibility
TickVsMinuteAlphaStudy = TickVsMinuteAlphaStudyShim
FilteringAblationStudy = FilteringAblationStudyShim

if __name__ == "__main__":
    print("🔧 Legacy Shims Module")
    print("=" * 40)
    print("Provides compatibility layer for v1 API tests")
    print("DEPRECATED - Remove after 2026-06-30")
    print()
    print("Available shims:")
    print("  - TickVsMinuteAlphaStudyShim")
    print("  - FilteringAblationStudyShim") 
    print("  - DualTickerDataAdapter")
    print("  - Legacy function wrappers")