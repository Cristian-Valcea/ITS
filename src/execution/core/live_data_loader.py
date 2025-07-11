"""
Live Data Loader Core Module

Contains live data loading and caching logic extracted from OrchestratorAgent.
This module handles:
- Real-time data fetching
- Data warmup calculations
- Cache management
- Data validation and preprocessing

This is an internal module - use src.execution.OrchestratorAgent for public API.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import market impact calculation for live trading
from ...shared.market_impact import calc_market_impact_features_fast, calc_market_impact_features


class LiveDataLoader:
    """
    Core live data loading and caching system.
    
    Handles real-time data fetching, caching, and preprocessing
    for the trading system.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the live data loader.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = Path(config.get('cache_dir', 'cache_ibkr'))
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    def calculate_duration_for_warmup(self) -> timedelta:
        """
        Calculate the duration needed for model warmup based on feature requirements.
        
        Returns:
            timedelta: Duration needed for warmup
        """
        try:
            # Get feature configuration
            feature_config = self.config.get('feature_engineering', {})
            
            # Calculate required lookback based on features
            max_lookback = 0
            
            # RSI lookback
            if 'RSI' in feature_config.get('features', []):
                rsi_window = feature_config.get('rsi', {}).get('window', 14)
                max_lookback = max(max_lookback, rsi_window)
            
            # EMA lookback
            if 'EMA' in feature_config.get('features', []):
                ema_windows = feature_config.get('ema', {}).get('windows', [10, 20])
                max_ema_window = max(ema_windows) if ema_windows else 20
                max_lookback = max(max_lookback, max_ema_window)
            
            # General lookback window
            lookback_window = feature_config.get('lookback_window', 3)
            max_lookback = max(max_lookback, lookback_window)
            
            # Add buffer for safety (50% more)
            buffer_multiplier = 1.5
            total_lookback = int(max_lookback * buffer_multiplier)
            
            # Convert to duration (assuming 1-minute bars)
            duration = timedelta(minutes=total_lookback)
            
            self.logger.info(f"Calculated warmup duration: {total_lookback} minutes ({duration})")
            return duration
            
        except Exception as e:
            self.logger.error(f"Error calculating warmup duration: {e}")
            # Return default duration
            return timedelta(hours=1)
        
    def load_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = "5 mins",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            bar_size: Bar size for data
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with historical data or None if failed
        """
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{bar_size.replace(' ', '_')}"
        
        # Check cache first
        if use_cache and cache_key in self.data_cache:
            self.logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key].copy()
            
        # TODO: Extract actual data loading logic
        return None
        
    def load_warmup_data(
        self,
        symbol: str,
        data_agent,
        interval: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """
        Load warmup data for a symbol.
        
        Args:
            symbol: Trading symbol
            data_agent: Data agent for fetching data
            interval: Data interval
            
        Returns:
            DataFrame with warmup data or None if failed
        """
        try:
            # Calculate the duration needed
            warmup_duration = self.calculate_duration_for_warmup()
            
            # Load initial data for warmup
            warmup_end = datetime.now()
            warmup_start = warmup_end - warmup_duration
            
            self.logger.info(f"Loading warmup data from {warmup_start} to {warmup_end}")
            warmup_data = data_agent.fetch_data(
                symbol=symbol,
                start_date=warmup_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=warmup_end.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval,
                use_cache=True
            )
            
            if warmup_data is None or warmup_data.empty:
                self.logger.error("Failed to load warmup data")
                return None
                
            self.logger.info(f"Loaded {len(warmup_data)} bars for warmup")
            return warmup_data
            
        except Exception as e:
            self.logger.error(f"Error loading warmup data: {e}")
            return None
        
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate loaded data for completeness and quality.
        
        Args:
            data: DataFrame to validate
            symbol: Symbol the data is for
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning(f"No data available for {symbol}")
            return False
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for {symbol}: {missing_columns}")
            return False
            
        # Check for NaN values
        if data[required_columns].isnull().any().any():
            self.logger.warning(f"Data contains NaN values for {symbol}")
            return False
            
        return True
        
    def cache_data(self, key: str, data: pd.DataFrame) -> None:
        """Cache data in memory."""
        self.data_cache[key] = data.copy()
        
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear data cache, optionally for a specific symbol."""
        if symbol:
            keys_to_remove = [key for key in self.data_cache.keys() if symbol in key]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for {symbol}")
        else:
            self.data_cache.clear()
            self.logger.info("Cleared all cached data")
    
    def process_live_market_data(
        self,
        book_snapshot: Dict[str, Any],
        symbol: str,
        include_heavy_features: bool = False
    ) -> Dict[str, float]:
        """
        Process live market data with fast market impact features.
        
        This method is optimized for low-latency live trading and only
        computes the critical features (spread_bps, queue_imbalance) by default.
        
        Args:
            book_snapshot: Order book snapshot with bid/ask data
            symbol: Trading symbol
            include_heavy_features: Whether to include computationally heavy features
            
        Returns:
            Dictionary with market impact features
        """
        try:
            # Extract level-1 data
            bid_px1 = float(book_snapshot.get('bid_px1', 0))
            bid_sz1 = float(book_snapshot.get('bid_sz1', 0))
            ask_px1 = float(book_snapshot.get('ask_px1', 0))
            ask_sz1 = float(book_snapshot.get('ask_sz1', 0))
            
            # Validate data
            if bid_px1 <= 0 or ask_px1 <= 0 or bid_sz1 <= 0 or ask_sz1 <= 0:
                self.logger.warning(f"Invalid order book data for {symbol}")
                return self._get_default_impact_features()
            
            # Calculate mid price
            mid = (bid_px1 + ask_px1) / 2
            
            # Fast calculation of critical features (<5 μs)
            spread_bps, queue_imbalance = calc_market_impact_features_fast(
                bid_px1, bid_sz1, ask_px1, ask_sz1, mid
            )
            
            # Base features for live trading
            features = {
                'spread_bps': spread_bps,
                'queue_imbalance': queue_imbalance,
                'mid': mid
            }
            
            # Add heavy features if requested (for monitoring/logging)
            if include_heavy_features:
                try:
                    # Convert to pandas Series for full calculation
                    book_series = pd.Series(book_snapshot)
                    
                    # Get previous mid for Kyle's lambda (if available)
                    last_mid = getattr(self, '_last_mid', None)
                    
                    # Full feature calculation
                    full_features = calc_market_impact_features(
                        book=book_series,
                        mid=mid,
                        last_mid=last_mid,
                        signed_vol=None,  # Would need trade data
                        notional=10_000
                    )
                    
                    features.update({
                        'impact_10k': full_features['impact_10k'],
                        'kyle_lambda': full_features['kyle_lambda']
                    })
                    
                    # Store mid for next iteration
                    self._last_mid = mid
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating heavy features: {e}")
                    features.update({
                        'impact_10k': 0.0,
                        'kyle_lambda': np.nan
                    })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing live market data for {symbol}: {e}")
            return self._get_default_impact_features()
    
    def _get_default_impact_features(self) -> Dict[str, float]:
        """Return default market impact features."""
        return {
            'spread_bps': 0.0,
            'queue_imbalance': 0.0,
            'impact_10k': 0.0,
            'kyle_lambda': np.nan,
            'mid': 0.0
        }
        if symbol is None:
            self.data_cache.clear()
        else:
            keys_to_remove = [key for key in self.data_cache.keys() if symbol in key]
            for key in keys_to_remove:
                del self.data_cache[key]
                
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'cached_datasets': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys()),
            'memory_usage_mb': sum(
                df.memory_usage(deep=True).sum() for df in self.data_cache.values()
            ) / (1024 * 1024)
        }