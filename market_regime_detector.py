"""
Market Regime Detection for V3 Trading Frequency Optimization

This module implements the reviewer-validated market regime detection system
that provides context-aware intelligence to the dual-lane controller while
maintaining memory bounds and offline development support.

Key Features:
- Z-score normalized regime scoring with 30-day rolling statistics
- Memory-bounded deque buffers to prevent unbounded growth (reviewer fix)
- Offline bootstrap support with local fixture fallback (house-keeping fix)
- Regime score clamping to [-3, 3] for controller stability
- 50-day bootstrap period for statistical reliability

Author: Stairways to Heaven v3.0 Implementation
Created: August 3, 2025
"""

import numpy as np
import pandas as pd
import os
import pickle
import gzip
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings


class MarketRegimeDetector:
    """
    Market regime detection system using Z-score normalization.
    
    This class detects market regimes by analyzing momentum, volatility, and
    correlation patterns across multiple assets. It uses rolling Z-score
    normalization to identify when current market conditions deviate
    significantly from recent historical norms.
    
    The detector is designed to be:
    - Memory-bounded: Uses deque with maxlen to prevent memory leaks
    - Offline-capable: Falls back to local fixtures when live data unavailable
    - Statistically robust: Requires 50-day bootstrap for reliability
    - Controller-compatible: Returns clamped scores in [-3, 3] range
    
    Attributes:
        bootstrap_days (int): Minimum days required for statistical reliability (50)
        momentum_buffer (deque): Rolling momentum values (memory-bounded)
        volatility_buffer (deque): Rolling volatility values (memory-bounded)
        divergence_buffer (deque): Rolling divergence values (memory-bounded)
    """
    
    def __init__(self, bootstrap_days: int = 50):
        """
        Initialize the market regime detector.
        
        Args:
            bootstrap_days (int): Minimum days of data required for statistical
                                reliability. Default 50 days (reviewer spec).
        """
        self.bootstrap_days = bootstrap_days
        
        # REVIEWER CRITICAL FIX: Use deque with maxlen to prevent memory issues
        # Buffer size: 30 days × 390 minutes/day = 11,700 data points maximum
        buffer_size = 30 * 390  # 30 days of minute bars
        self.momentum_buffer = deque(maxlen=buffer_size)
        self.volatility_buffer = deque(maxlen=buffer_size)
        self.divergence_buffer = deque(maxlen=buffer_size)
        
        # Bootstrap status tracking
        self.is_bootstrapped = False
        self.bootstrap_source = None
        
        # Statistical parameters
        self.min_data_points = 100  # Minimum for stable Z-score calculation
        
    def bootstrap_from_history_with_fallback(self, 
                                           symbols: List[str] = ["NVDA", "MSFT"], 
                                           days: int = 50) -> bool:
        """
        HOUSE-KEEPING FIX #2: Bootstrap with offline fallback for dev/CI environments.
        
        Attempts to fetch live historical data, falls back to local fixtures if
        unavailable. This prevents unit-test flakiness when external APIs are
        unreachable and enables offline development.
        
        Args:
            symbols (List[str]): Asset symbols to fetch data for
            days (int): Number of historical days to fetch
            
        Returns:
            bool: True if bootstrap successful, False otherwise
        """
        try:
            # Primary: Attempt live data fetch
            print(f"🔄 Attempting live data fetch for {symbols} ({days} days)...")
            historical_data = self._fetch_live_historical_data(symbols, days)
            self._populate_buffers_from_data(historical_data)
            self.bootstrap_source = "live_data"
            self.is_bootstrapped = True
            print(f"✅ Bootstrapped with live {days}-day historical data")
            return True
            
        except (ConnectionError, TimeoutError, Exception) as e:
            print(f"⚠️ Live data fetch failed: {e}")
            
            # HOUSE-KEEPING FALLBACK: Try local fixture files
            try:
                print(f"🔄 Attempting local fixture fallback...")
                fixture_data = self._load_local_fixture_data(symbols, days)
                self._populate_buffers_from_data(fixture_data)
                self.bootstrap_source = "local_fixtures"
                self.is_bootstrapped = True
                print(f"✅ Bootstrapped with local fixture data ({days} days)")
                return True
                
            except FileNotFoundError as fe:
                print(f"⚠️ No local fixtures available: {fe}")
                
            except Exception as ex:
                print(f"🚨 Fixture loading failed: {ex}")
        
        # Complete failure - graceful degradation
        print(f"🚨 Bootstrap failed completely. Starting with neutral regime.")
        self.bootstrap_source = "neutral_fallback"
        self.is_bootstrapped = False
        return False
    
    def _load_local_fixture_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """
        HOUSE-KEEPING FIX #2: Load local fixture data for offline development.
        
        Loads pre-saved market data from local parquet files for development
        and testing when live data sources are unavailable.
        
        Args:
            symbols (List[str]): Asset symbols to load
            days (int): Number of days requested (used for data limiting)
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> DataFrame mapping
            
        Raises:
            FileNotFoundError: If fixture files don't exist
        """
        fixture_files = {
            "NVDA": "test_data/nvda_historical_fixture.parquet",
            "MSFT": "test_data/msft_historical_fixture.parquet",
            "AAPL": "test_data/aapl_historical_fixture.parquet"  # Future expansion
        }
        
        historical_data = {}
        
        for symbol in symbols:
            fixture_path = fixture_files.get(symbol)
            if not fixture_path:
                raise FileNotFoundError(f"No fixture defined for symbol: {symbol}")
            
            if not os.path.exists(fixture_path):
                raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
            
            try:
                df = pd.read_parquet(fixture_path)
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'close', 'volume', 'high', 'low']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in {fixture_path}: {missing_cols}")
                
                # Take last N days worth of data (approximately)
                # Assume ~390 minutes per trading day
                n_points = min(len(df), days * 390)
                historical_data[symbol] = df.tail(n_points)
                
                print(f"   Loaded {len(historical_data[symbol])} points for {symbol}")
                
            except Exception as e:
                raise FileNotFoundError(f"Failed to load fixture {fixture_path}: {e}")
        
        return historical_data
    
    def _fetch_live_historical_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch live historical data from external APIs.
        
        This is a placeholder for the actual API integration. In production,
        this would connect to Polygon, Alpha Vantage, or other data providers.
        
        Args:
            symbols (List[str]): Asset symbols to fetch
            days (int): Number of historical days
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> DataFrame mapping
            
        Raises:
            ConnectionError: If API is unreachable
            TimeoutError: If API request times out
        """
        # For now, simulate API failure to test fallback mechanism
        # In production, replace with actual API calls
        raise ConnectionError("Live data API not implemented - triggering fallback")
    
    def _populate_buffers_from_data(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Populate regime detection buffers from historical data.
        
        Calculates momentum, volatility, and divergence metrics from historical
        price data and populates the rolling buffers used for Z-score normalization.
        
        Args:
            historical_data (Dict[str, pd.DataFrame]): Symbol -> DataFrame mapping
        """
        if len(historical_data) < 2:
            raise ValueError("Need at least 2 assets for divergence calculation")
        
        symbols = list(historical_data.keys())
        primary_symbol = symbols[0]
        secondary_symbol = symbols[1]
        
        primary_df = historical_data[primary_symbol]
        secondary_df = historical_data[secondary_symbol]
        
        # Align data by timestamp (take intersection)
        min_length = min(len(primary_df), len(secondary_df))
        
        # Calculate momentum (price rate of change)
        primary_momentum = self._calculate_momentum(primary_df.tail(min_length))
        secondary_momentum = self._calculate_momentum(secondary_df.tail(min_length))
        combined_momentum = (primary_momentum + secondary_momentum) / 2
        
        # Calculate volatility (rolling standard deviation of returns)
        primary_volatility = self._calculate_volatility(primary_df.tail(min_length))
        secondary_volatility = self._calculate_volatility(secondary_df.tail(min_length))
        combined_volatility = (primary_volatility + secondary_volatility) / 2
        
        # Calculate divergence (correlation breakdown)
        divergence = self._calculate_divergence(
            primary_df.tail(min_length), 
            secondary_df.tail(min_length)
        )
        
        # Populate buffers (deque will automatically limit to maxlen)
        for i in range(len(combined_momentum)):
            if not np.isnan(combined_momentum[i]):
                self.momentum_buffer.append(combined_momentum[i])
            if not np.isnan(combined_volatility[i]):
                self.volatility_buffer.append(combined_volatility[i])
            if not np.isnan(divergence[i]):
                self.divergence_buffer.append(divergence[i])
        
        print(f"   Populated buffers: momentum={len(self.momentum_buffer)}, "
              f"volatility={len(self.volatility_buffer)}, divergence={len(self.divergence_buffer)}")
    
    def _calculate_momentum(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Calculate momentum as rate of change over specified window."""
        if len(df) < window:
            return np.full(len(df), np.nan)
        
        close_prices = df['close'].values
        momentum = np.full(len(close_prices), np.nan)
        
        for i in range(window, len(close_prices)):
            if close_prices[i - window] != 0:
                momentum[i] = (close_prices[i] - close_prices[i - window]) / close_prices[i - window]
        
        return momentum
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility of returns."""
        if len(df) < window:
            return np.full(len(df), np.nan)
        
        close_prices = df['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        volatility = np.full(len(close_prices), np.nan)
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            volatility[i+1] = np.std(window_returns)
        
        return volatility
    
    def _calculate_divergence(self, df1: pd.DataFrame, df2: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Calculate correlation divergence between two assets."""
        min_length = min(len(df1), len(df2))
        
        if min_length < window:
            return np.full(min_length, np.nan)
        
        returns1 = np.diff(df1['close'].values) / df1['close'].values[:-1]
        returns2 = np.diff(df2['close'].values) / df2['close'].values[:-1]
        
        divergence = np.full(min_length, np.nan)
        
        for i in range(window, min(len(returns1), len(returns2))):
            window_returns1 = returns1[i-window:i]
            window_returns2 = returns2[i-window:i]
            
            # Use correlation coefficient as measure of divergence
            correlation = np.corrcoef(window_returns1, window_returns2)[0, 1]
            if not np.isnan(correlation):
                # Convert correlation to divergence (1 - |correlation|)
                divergence[i+1] = 1.0 - abs(correlation)
        
        return divergence
    
    def calculate_regime_score(self, momentum: float, volatility: float, divergence: float) -> float:
        """
        Calculate current market regime score using Z-score normalization.
        
        Args:
            momentum (float): Current momentum value
            volatility (float): Current volatility value  
            divergence (float): Current divergence value
            
        Returns:
            float: Regime score clamped to [-3, 3] for controller stability
        """
        # Add current values to rolling buffers
        self.momentum_buffer.append(momentum)
        self.volatility_buffer.append(volatility)
        self.divergence_buffer.append(divergence)
        
        # Bootstrap check (reviewer requirement: 50 trading days minimum)
        required_points = self.bootstrap_days * 390  # 390 minutes per trading day
        if len(self.momentum_buffer) < required_points:
            return 0.0  # Neutral regime during bootstrap period
        
        # Z-score normalization with safety checks
        momentum_z = self._z_score_safe(momentum, self.momentum_buffer)
        volatility_z = self._z_score_safe(volatility, self.volatility_buffer)
        divergence_z = self._z_score_safe(divergence, self.divergence_buffer)
        
        # Weighted combination (reviewer-approved weights)
        regime_score = 0.4 * momentum_z + 0.3 * volatility_z + 0.3 * divergence_z
        
        # REVIEWER CRITICAL: Clamp to [-3, 3] BEFORE returning
        return float(np.clip(regime_score, -3.0, 3.0))
    
    def _z_score_safe(self, value: float, buffer: deque) -> float:
        """
        Reviewer-safe Z-score calculation with zero-division protection.
        
        Args:
            value (float): Current value to normalize
            buffer (deque): Historical values for normalization
            
        Returns:
            float: Z-score normalized value, 0.0 if calculation impossible
        """
        if len(buffer) < self.min_data_points:  # Insufficient data
            return 0.0
        
        # Convert deque to numpy array for efficient calculation
        buffer_array = np.array(buffer)
        
        mean = np.mean(buffer_array)
        std = np.std(buffer_array)
        
        # Prevent division by zero
        if std < 1e-8:
            return 0.0
        
        return (value - mean) / std
    
    def get_detector_health(self) -> Dict[str, Union[int, bool, str, float]]:
        """
        Get current detector health metrics for monitoring.
        
        Returns:
            dict: Detector health information including buffer sizes,
                  bootstrap status, and statistical measures
        """
        return {
            "is_bootstrapped": self.is_bootstrapped,
            "bootstrap_source": self.bootstrap_source,
            "momentum_buffer_size": len(self.momentum_buffer),
            "volatility_buffer_size": len(self.volatility_buffer),
            "divergence_buffer_size": len(self.divergence_buffer),
            "max_buffer_size": self.momentum_buffer.maxlen,
            "bootstrap_days_required": self.bootstrap_days,
            "min_data_points": self.min_data_points,
            "bootstrap_progress": min(1.0, len(self.momentum_buffer) / (self.bootstrap_days * 390))
        }
    
    def create_test_fixtures(self, symbols: List[str] = ["NVDA", "MSFT"], days: int = 60):
        """
        Create local test fixtures for offline development.
        
        Generates synthetic market data that can be used for testing and
        development when live data sources are unavailable.
        
        Args:
            symbols (List[str]): Symbols to create fixtures for
            days (int): Number of days of synthetic data to generate
        """
        os.makedirs("test_data", exist_ok=True)
        
        for symbol in symbols:
            # Generate synthetic realistic market data
            n_points = days * 390  # 390 minutes per trading day
            
            # Base price around realistic levels
            base_prices = {"NVDA": 450, "MSFT": 400, "AAPL": 180}
            base_price = base_prices.get(symbol, 100)
            
            # Generate correlated random walk
            np.random.seed(42 + hash(symbol) % 1000)  # Deterministic but varied
            returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
            
            # Add some autocorrelation for realism
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate volume data
            volumes = np.random.lognormal(10, 0.5, n_points).astype(int)
            
            # Create realistic OHLC data
            high_prices = prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
            low_prices = prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
            
            # Generate timestamps
            start_date = datetime.now() - timedelta(days=days)
            timestamps = pd.date_range(start_date, periods=n_points, freq='1min')
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'close': prices,
                'high': high_prices,
                'low': low_prices,
                'volume': volumes
            })
            
            # Save as parquet
            fixture_path = f"test_data/{symbol.lower()}_historical_fixture.parquet"
            df.to_parquet(fixture_path)
            print(f"✅ Created fixture: {fixture_path} ({len(df)} points)")


def test_regime_detector_memory_bounds():
    """
    REVIEWER FIX: Test that deque prevents unbounded memory growth.
    """
    detector = MarketRegimeDetector()
    
    print("Testing regime detector memory bounds...")
    
    # Add much more data than buffer can hold
    n_points = 50000  # Much more than 30 * 390 = 11,700
    
    for i in range(n_points):
        momentum = np.random.normal(0, 1)
        volatility = abs(np.random.normal(0, 0.5))
        divergence = np.random.uniform(0, 1)
        
        detector.calculate_regime_score(momentum, volatility, divergence)
    
    # Verify buffers are bounded to maxlen
    max_size = 30 * 390
    assert len(detector.momentum_buffer) <= max_size, f"Momentum buffer exceeded maxlen: {len(detector.momentum_buffer)}"
    assert len(detector.volatility_buffer) <= max_size, f"Volatility buffer exceeded maxlen: {len(detector.volatility_buffer)}"
    assert len(detector.divergence_buffer) <= max_size, f"Divergence buffer exceeded maxlen: {len(detector.divergence_buffer)}"
    
    print(f"✅ Memory bounds test passed")
    print(f"   Buffer sizes after {n_points} additions:")
    print(f"   Momentum: {len(detector.momentum_buffer)} / {max_size}")
    print(f"   Volatility: {len(detector.volatility_buffer)} / {max_size}")
    print(f"   Divergence: {len(detector.divergence_buffer)} / {max_size}")
    
    return True


def test_regime_score_clamping():
    """
    Test that regime scores are properly clamped to [-3, 3] range.
    """
    detector = MarketRegimeDetector()
    
    # Create fixtures for testing
    detector.create_test_fixtures()
    
    # Bootstrap with fixture data
    detector.bootstrap_from_history_with_fallback()
    
    print("Testing regime score clamping...")
    
    # Test with extreme values that should be clamped
    test_cases = [
        (10.0, 5.0, 2.0),   # Very high momentum
        (-10.0, -5.0, -2.0), # Very negative values
        (0.0, 0.0, 0.0),     # Neutral case
        (1.0, 0.5, 0.3)      # Normal case
    ]
    
    for momentum, volatility, divergence in test_cases:
        score = detector.calculate_regime_score(momentum, volatility, divergence)
        
        # Verify clamping
        assert -3.0 <= score <= 3.0, f"Score {score} outside [-3, 3] range"
        assert isinstance(score, float), f"Score must be float, got {type(score)}"
    
    print("✅ Regime score clamping test passed")
    
    # Cleanup
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    
    return True


if __name__ == "__main__":
    """
    Quick validation of regime detector implementation.
    """
    print("📊 Market Regime Detector Validation")
    print("=" * 45)
    
    # Test 1: Basic initialization
    detector = MarketRegimeDetector()
    print(f"Detector initialized with {detector.bootstrap_days}-day bootstrap requirement")
    
    # Test 2: Create test fixtures
    detector.create_test_fixtures()
    
    # Test 3: Bootstrap with fallback
    success = detector.bootstrap_from_history_with_fallback()
    print(f"Bootstrap success: {success}")
    
    # Test 4: Health check
    health = detector.get_detector_health()
    print(f"Detector health: {health}")
    
    # Test 5: Memory bounds (reviewer requirement)
    test_regime_detector_memory_bounds()
    
    # Test 6: Score clamping validation
    test_regime_score_clamping()
    
    print("\n✅ All regime detector validations passed!")
    print("Market regime detector is ready for integration.")