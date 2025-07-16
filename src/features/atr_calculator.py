# src/features/atr_calculator.py
"""
Average True Range (ATR) Calculator

Calculates ATR and related volatility measures for microstructural analysis.
ATR provides a measure of volatility that accounts for gaps and is particularly
useful for intraday trading systems.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_calculator import BaseFeatureCalculator


class ATRCalculator(BaseFeatureCalculator):
    """
    Calculator for Average True Range (ATR) and related volatility features.
    
    ATR is a technical analysis indicator that measures market volatility by
    decomposing the entire range of an asset price for that period.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize ATR calculator.
        
        Args:
            config: Configuration containing ATR parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # ATR configuration
        self.atr_config = config.get('atr', {})
        self.windows = self.atr_config.get('windows', [14, 21])  # Multiple ATR periods
        self.smoothing_method = self.atr_config.get('smoothing_method', 'ema')  # 'ema' or 'sma'
        
        # Ensure windows is a list
        if isinstance(self.windows, int):
            self.windows = [self.windows]
        
        # Calculate max lookback for initialization
        self._max_lookback = max(self.windows) * 2  # Extra buffer for smoothing
        
        self.logger.info(f"ATR Calculator initialized with windows: {self.windows}, "
                        f"smoothing: {self.smoothing_method}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR and related volatility features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR features added
        """
        if len(data) < max(self.windows):
            self.logger.warning(f"Insufficient data for ATR calculation. "
                              f"Need {max(self.windows)}, got {len(data)}")
            return self._add_nan_features(data)
        
        result = data.copy()
        
        try:
            # Calculate True Range components
            result = self._calculate_true_range(result)
            
            # Calculate ATR for each window
            for window in self.windows:
                atr_col = f'atr_{window}'
                
                if self.smoothing_method == 'ema':
                    # Exponential moving average (Wilder's smoothing)
                    alpha = 2.0 / (window + 1)
                    result[atr_col] = result['true_range'].ewm(alpha=alpha, adjust=False).mean()
                else:
                    # Simple moving average
                    result[atr_col] = result['true_range'].rolling(window=window).mean()
                
                # Calculate normalized ATR (ATR / Close price)
                result[f'atr_{window}_normalized'] = result[atr_col] / result['close']
                
                # Calculate ATR percentile (rolling percentile of ATR)
                result[f'atr_{window}_percentile'] = (
                    result[atr_col].rolling(window=window*2)
                    .rank(pct=True)
                )
            
            # Calculate additional volatility features
            result = self._calculate_volatility_features(result)
            
            # Drop intermediate columns
            result = result.drop(['true_range'], axis=1, errors='ignore')
            
            self.logger.debug(f"ATR calculation completed for {len(result)} rows")
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            result = self._add_nan_features(data)
        
        return result
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate True Range components."""
        result = data.copy()
        
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = result['close'].shift(1)
        
        tr1 = result['high'] - result['low']  # Current high - current low
        tr2 = np.abs(result['high'] - prev_close)  # Current high - previous close
        tr3 = np.abs(result['low'] - prev_close)   # Current low - previous close
        
        result['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        return result
    
    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional volatility-based features."""
        result = data.copy()
        
        # Use the first ATR window for additional features
        primary_window = self.windows[0]
        primary_atr = f'atr_{primary_window}'
        
        if primary_atr in result.columns:
            # ATR-based volatility regime detection
            atr_ma = result[primary_atr].rolling(window=primary_window*2).mean()
            result['atr_regime'] = np.where(
                result[primary_atr] > atr_ma * 1.2, 1,  # High volatility
                np.where(result[primary_atr] < atr_ma * 0.8, -1, 0)  # Low volatility
            )
            
            # Volatility acceleration (change in ATR)
            result['atr_acceleration'] = result[primary_atr].pct_change()
            
            # ATR efficiency ratio (price change / ATR)
            price_change = np.abs(result['close'].pct_change())
            result['atr_efficiency'] = price_change / (result[f'{primary_atr}_normalized'] + 1e-8)
        
        return result
    
    def _add_nan_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add NaN features when calculation fails."""
        result = data.copy()
        
        for feature_name in self.get_feature_names():
            result[feature_name] = np.nan
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this calculator."""
        features = []
        
        for window in self.windows:
            features.extend([
                f'atr_{window}',
                f'atr_{window}_normalized',
                f'atr_{window}_percentile'
            ])
        
        # Additional volatility features (based on primary window)
        features.extend([
            'atr_regime',
            'atr_acceleration', 
            'atr_efficiency'
        ])
        
        return features
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period required."""
        return self._max_lookback