# src/features/vwap_ratio_calculator.py
"""
Intraday VWAP Ratio Calculator

Calculates various VWAP-based ratios and microstructural features that provide
insights into price efficiency and market microstructure dynamics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_calculator import BaseFeatureCalculator


class VWAPRatioCalculator(BaseFeatureCalculator):
    """
    Calculator for intraday VWAP ratios and related microstructural features.
    
    Provides various VWAP-based metrics that help assess price efficiency,
    market impact, and microstructural dynamics.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize VWAP ratio calculator.
        
        Args:
            config: Configuration containing VWAP ratio parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # VWAP ratio configuration
        self.vwap_ratio_config = config.get('vwap_ratio', {})
        self.windows = self.vwap_ratio_config.get('windows', [20, 60, 120])  # Multiple VWAP windows
        self.deviation_bands = self.vwap_ratio_config.get('deviation_bands', [1.0, 2.0])  # Standard deviation bands
        
        # Ensure windows is a list
        if isinstance(self.windows, int):
            self.windows = [self.windows]
        
        # Calculate max lookback
        self._max_lookback = max(self.windows) + 50  # Extra buffer for calculations
        
        self.logger.info(f"VWAP Ratio Calculator initialized with windows: {self.windows}, "
                        f"deviation bands: {self.deviation_bands}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP ratios and related microstructural features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP ratio features added
        """
        if len(data) < max(self.windows):
            self.logger.warning(f"Insufficient data for VWAP ratio calculation. "
                              f"Need {max(self.windows)}, got {len(data)}")
            return self._add_nan_features(data)
        
        result = data.copy()
        
        try:
            # Calculate rolling VWAP for different windows
            for window in self.windows:
                result = self._calculate_rolling_vwap(result, window)
            
            # Calculate VWAP ratios and deviations
            result = self._calculate_vwap_ratios(result)
            
            # Calculate VWAP-based microstructural features
            result = self._calculate_microstructural_features(result)
            
            self.logger.debug(f"VWAP ratio calculation completed for {len(result)} rows")
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP ratios: {e}")
            result = self._add_nan_features(data)
        
        return result
    
    def _calculate_rolling_vwap(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling VWAP for given window."""
        result = data.copy()
        
        # Typical price (HLC/3)
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        
        # Rolling VWAP calculation
        price_volume = typical_price * result['volume']
        rolling_pv_sum = price_volume.rolling(window=window).sum()
        rolling_volume_sum = result['volume'].rolling(window=window).sum()
        
        vwap_col = f'vwap_{window}'
        result[vwap_col] = rolling_pv_sum / (rolling_volume_sum + 1e-8)  # Avoid division by zero
        
        # VWAP standard deviation (for bands)
        price_diff_sq = (typical_price - result[vwap_col]) ** 2
        vwap_variance = (price_diff_sq * result['volume']).rolling(window=window).sum() / rolling_volume_sum
        result[f'vwap_{window}_std'] = np.sqrt(vwap_variance)
        
        return result
    
    def _calculate_vwap_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP ratios and deviations."""
        result = data.copy()
        
        for window in self.windows:
            vwap_col = f'vwap_{window}'
            vwap_std_col = f'vwap_{window}_std'
            
            if vwap_col in result.columns:
                # Price to VWAP ratio
                result[f'price_vwap_ratio_{window}'] = result['close'] / result[vwap_col]
                
                # VWAP deviation (normalized by standard deviation)
                result[f'vwap_deviation_{window}'] = (
                    (result['close'] - result[vwap_col]) / (result[vwap_std_col] + 1e-8)
                )
                
                # VWAP position (percentile within recent range)
                result[f'vwap_position_{window}'] = (
                    result['close'].rolling(window=window)
                    .apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
                )
                
                # VWAP momentum (rate of change)
                result[f'vwap_momentum_{window}'] = result[vwap_col].pct_change(periods=5)
                
                # Calculate VWAP bands
                for band in self.deviation_bands:
                    upper_band = result[vwap_col] + (band * result[vwap_std_col])
                    lower_band = result[vwap_col] - (band * result[vwap_std_col])
                    
                    # Band position (-1 to 1, where 0 is VWAP)
                    result[f'vwap_band_position_{window}_{band}'] = np.where(
                        result['close'] > result[vwap_col],
                        (result['close'] - result[vwap_col]) / (upper_band - result[vwap_col] + 1e-8),
                        (result['close'] - result[vwap_col]) / (result[vwap_col] - lower_band + 1e-8)
                    )
                    
                    # Band breach indicator
                    result[f'vwap_band_breach_{window}_{band}'] = np.where(
                        result['close'] > upper_band, 1,
                        np.where(result['close'] < lower_band, -1, 0)
                    )
        
        return result
    
    def _calculate_microstructural_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced microstructural features based on VWAP."""
        result = data.copy()
        
        # Use primary window for microstructural features
        primary_window = self.windows[0]
        primary_vwap = f'vwap_{primary_window}'
        
        if primary_vwap in result.columns:
            # VWAP efficiency ratio (how efficiently price moves toward VWAP)
            price_change = result['close'].diff()
            vwap_change = result[primary_vwap].diff()
            result['vwap_efficiency'] = np.where(
                np.abs(vwap_change) > 1e-8,
                price_change / vwap_change,
                0
            )
            
            # VWAP reversion strength (tendency to revert to VWAP)
            vwap_distance = result['close'] - result[primary_vwap]
            vwap_distance_change = vwap_distance.diff()
            result['vwap_reversion_strength'] = -vwap_distance_change / (np.abs(vwap_distance) + 1e-8)
            
            # Volume-weighted price pressure
            volume_ma = result['volume'].rolling(window=primary_window).mean()
            volume_ratio = result['volume'] / (volume_ma + 1e-8)
            price_pressure = (result['close'] - result[primary_vwap]) * volume_ratio
            result['vwap_volume_pressure'] = price_pressure.rolling(window=10).mean()
            
            # VWAP cross signals
            result['vwap_cross_signal'] = np.where(
                (result['close'] > result[primary_vwap]) & 
                (result['close'].shift(1) <= result[primary_vwap].shift(1)), 1,  # Bullish cross
                np.where(
                    (result['close'] < result[primary_vwap]) & 
                    (result['close'].shift(1) >= result[primary_vwap].shift(1)), -1,  # Bearish cross
                    0
                )
            )
            
            # Multi-timeframe VWAP alignment
            if len(self.windows) >= 2:
                short_vwap = f'vwap_{self.windows[0]}'
                long_vwap = f'vwap_{self.windows[1]}'
                
                result['vwap_alignment'] = np.where(
                    (result[short_vwap] > result[long_vwap]) & 
                    (result['close'] > result[short_vwap]), 1,  # Bullish alignment
                    np.where(
                        (result[short_vwap] < result[long_vwap]) & 
                        (result['close'] < result[short_vwap]), -1,  # Bearish alignment
                        0
                    )
                )
        
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
            # Basic VWAP features
            features.extend([
                f'vwap_{window}',
                f'vwap_{window}_std',
                f'price_vwap_ratio_{window}',
                f'vwap_deviation_{window}',
                f'vwap_position_{window}',
                f'vwap_momentum_{window}'
            ])
            
            # VWAP band features
            for band in self.deviation_bands:
                features.extend([
                    f'vwap_band_position_{window}_{band}',
                    f'vwap_band_breach_{window}_{band}'
                ])
        
        # Microstructural features
        features.extend([
            'vwap_efficiency',
            'vwap_reversion_strength',
            'vwap_volume_pressure',
            'vwap_cross_signal'
        ])
        
        # Multi-timeframe features (if multiple windows)
        if len(self.windows) >= 2:
            features.append('vwap_alignment')
        
        return features
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period required."""
        return self._max_lookback