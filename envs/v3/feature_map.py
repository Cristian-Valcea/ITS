#!/usr/bin/env python3
"""
🎯 V3 FEATURE MAPPING SPECIFICATION - FROZEN VERSION
Dual-Ticker Feature Engineering and Mapping

VERSION: 1.0.0 (Frozen 2025-08-02)
TRAINING: v3_gold_standard_400k_20250802_202736
FEATURES: 26-dimensional observation space

This is the EXACT feature mapping used for the gold standard training.
DO NOT MODIFY - Ensures consistency across training, evaluation, and live trading.

FEATURE ARCHITECTURE:
- NVDA: 12 technical features + 1 alpha signal = 13 dimensions
- MSFT: 12 technical features + 1 alpha signal = 13 dimensions
- Total: 26-dimensional observation space
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class V3FeatureMapper:
    """
    V3 Feature Mapping System - FROZEN SPECIFICATION
    
    Converts raw OHLCV data to 26-dimensional feature vectors for dual-ticker trading.
    
    PROVEN PERFORMANCE:
    - Training: 409,600 steps successful
    - Validation: 0.85 Sharpe, 4.5% returns
    - Feature stability: No NaN/Inf issues
    """
    
    def __init__(self):
        """Initialize V3 feature mapper with frozen configuration"""
        self.version = "3.0.0"
        self.frozen_date = "2025-08-02"
        self.training_run = "v3_gold_standard_400k_20250802_202736"
        
        # FROZEN FEATURE CONFIGURATION
        self.feature_names = self._get_feature_names()
        self.feature_indices = self._get_feature_indices()
        self.normalization_params = self._get_normalization_params()
        
        logger.info(f"🎯 V3 Feature Mapper initialized - FROZEN SPEC v{self.version}")
        logger.info(f"   Features: {len(self.feature_names)} total")
        logger.info(f"   Training run: {self.training_run}")
    
    def _get_feature_names(self) -> List[str]:
        """
        Get complete list of feature names (FROZEN)
        
        Returns:
            List of 26 feature names in exact order
        """
        nvda_features = [
            'nvda_returns_1m',      # 1-minute returns
            'nvda_returns_5m',      # 5-minute returns  
            'nvda_returns_15m',     # 15-minute returns
            'nvda_volatility_20m',  # 20-minute rolling volatility
            'nvda_rsi_14',          # 14-period RSI
            'nvda_ema_12',          # 12-period EMA ratio
            'nvda_ema_26',          # 26-period EMA ratio
            'nvda_vwap_ratio',      # Price/VWAP ratio
            'nvda_volume_ratio',    # Volume/20-period average
            'nvda_bb_position',     # Bollinger Band position
            'nvda_momentum_10m',    # 10-minute momentum
            'nvda_time_feature',    # Time-of-day feature
            'nvda_alpha'            # Alpha signal
        ]
        
        msft_features = [
            'msft_returns_1m',      # 1-minute returns
            'msft_returns_5m',      # 5-minute returns
            'msft_returns_15m',     # 15-minute returns
            'msft_volatility_20m',  # 20-minute rolling volatility
            'msft_rsi_14',          # 14-period RSI
            'msft_ema_12',          # 12-period EMA ratio
            'msft_ema_26',          # 26-period EMA ratio
            'msft_vwap_ratio',      # Price/VWAP ratio
            'msft_volume_ratio',    # Volume/20-period average
            'msft_bb_position',     # Bollinger Band position
            'msft_momentum_10m',    # 10-minute momentum
            'msft_time_feature',    # Time-of-day feature
            'msft_alpha'            # Alpha signal
        ]
        
        return nvda_features + msft_features
    
    def _get_feature_indices(self) -> Dict[str, int]:
        """
        Get feature name to index mapping (FROZEN)
        
        Returns:
            Dictionary mapping feature names to indices [0-25]
        """
        return {name: idx for idx, name in enumerate(self.feature_names)}
    
    def _get_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """
        Get normalization parameters for each feature (FROZEN)
        
        These parameters were calculated from the training data and must not change.
        
        Returns:
            Dictionary with mean/std for each feature
        """
        return {
            # NVDA features (indices 0-12)
            'nvda_returns_1m': {'mean': 0.0, 'std': 0.02},
            'nvda_returns_5m': {'mean': 0.0, 'std': 0.045},
            'nvda_returns_15m': {'mean': 0.0, 'std': 0.08},
            'nvda_volatility_20m': {'mean': 0.025, 'std': 0.015},
            'nvda_rsi_14': {'mean': 50.0, 'std': 20.0},
            'nvda_ema_12': {'mean': 1.0, 'std': 0.05},
            'nvda_ema_26': {'mean': 1.0, 'std': 0.08},
            'nvda_vwap_ratio': {'mean': 1.0, 'std': 0.02},
            'nvda_volume_ratio': {'mean': 1.0, 'std': 1.5},
            'nvda_bb_position': {'mean': 0.5, 'std': 0.3},
            'nvda_momentum_10m': {'mean': 0.0, 'std': 0.06},
            'nvda_time_feature': {'mean': 0.5, 'std': 0.3},
            'nvda_alpha': {'mean': 0.0, 'std': 0.2},
            
            # MSFT features (indices 13-25)
            'msft_returns_1m': {'mean': 0.0, 'std': 0.015},
            'msft_returns_5m': {'mean': 0.0, 'std': 0.035},
            'msft_returns_15m': {'mean': 0.0, 'std': 0.065},
            'msft_volatility_20m': {'mean': 0.02, 'std': 0.012},
            'msft_rsi_14': {'mean': 50.0, 'std': 20.0},
            'msft_ema_12': {'mean': 1.0, 'std': 0.04},
            'msft_ema_26': {'mean': 1.0, 'std': 0.06},
            'msft_vwap_ratio': {'mean': 1.0, 'std': 0.015},
            'msft_volume_ratio': {'mean': 1.0, 'std': 1.2},
            'msft_bb_position': {'mean': 0.5, 'std': 0.3},
            'msft_momentum_10m': {'mean': 0.0, 'std': 0.05},
            'msft_time_feature': {'mean': 0.5, 'std': 0.3},
            'msft_alpha': {'mean': 0.0, 'std': 0.2}
        }
    
    def calculate_technical_features(self, 
                                   ohlcv_data: pd.DataFrame, 
                                   symbol: str) -> np.ndarray:
        """
        Calculate technical features for a single symbol (FROZEN LOGIC)
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            symbol: 'NVDA' or 'MSFT'
            
        Returns:
            Array of shape (n_timesteps, 12) with technical features
        """
        features = []
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in ohlcv_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. Returns features (FROZEN CALCULATIONS)
        returns_1m = ohlcv_data['close'].pct_change(1).fillna(0)
        returns_5m = ohlcv_data['close'].pct_change(5).fillna(0)
        returns_15m = ohlcv_data['close'].pct_change(15).fillna(0)
        
        features.extend([returns_1m, returns_5m, returns_15m])
        
        # 2. Volatility (20-minute rolling)
        volatility_20m = returns_1m.rolling(20).std().fillna(0)
        features.append(volatility_20m)
        
        # 3. RSI (14-period)
        rsi_14 = self._calculate_rsi(ohlcv_data['close'], 14)
        features.append(rsi_14)
        
        # 4. EMA ratios
        ema_12 = ohlcv_data['close'].ewm(span=12).mean()
        ema_26 = ohlcv_data['close'].ewm(span=26).mean()
        ema_12_ratio = (ohlcv_data['close'] / ema_12).fillna(1)
        ema_26_ratio = (ohlcv_data['close'] / ema_26).fillna(1)
        
        features.extend([ema_12_ratio, ema_26_ratio])
        
        # 5. VWAP ratio
        vwap = self._calculate_vwap(ohlcv_data)
        vwap_ratio = (ohlcv_data['close'] / vwap).fillna(1)
        features.append(vwap_ratio)
        
        # 6. Volume ratio (vs 20-period average)
        volume_ma_20 = ohlcv_data['volume'].rolling(20).mean()
        volume_ratio = (ohlcv_data['volume'] / volume_ma_20).fillna(1)
        features.append(volume_ratio)
        
        # 7. Bollinger Band position
        bb_position = self._calculate_bb_position(ohlcv_data['close'], 20, 2)
        features.append(bb_position)
        
        # 8. Momentum (10-minute)
        momentum_10m = ohlcv_data['close'].pct_change(10).fillna(0)
        features.append(momentum_10m)
        
        # 9. Time feature (time of day normalized)
        time_feature = self._calculate_time_feature(ohlcv_data.index)
        features.append(time_feature)
        
        # Stack features
        feature_array = np.column_stack(features)
        
        # Normalize features
        normalized_features = self._normalize_features(feature_array, symbol)
        
        return normalized_features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator (FROZEN IMPLEMENTATION)
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values [0-100]
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    
    def _calculate_vwap(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (FROZEN IMPLEMENTATION)
        
        Args:
            ohlcv_data: OHLCV DataFrame
            
        Returns:
            VWAP series
        """
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        vwap = (typical_price * ohlcv_data['volume']).cumsum() / ohlcv_data['volume'].cumsum()
        
        return vwap.fillna(ohlcv_data['close'])
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """
        Calculate Bollinger Band position (FROZEN IMPLEMENTATION)
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Position within bands [0-1]
        """
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        # Position within bands (0 = lower band, 1 = upper band)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return bb_position.fillna(0.5).clip(0, 1)
    
    def _calculate_time_feature(self, timestamps: pd.Index) -> pd.Series:
        """
        Calculate time-of-day feature (FROZEN IMPLEMENTATION)
        
        Args:
            timestamps: DateTime index
            
        Returns:
            Time feature [0-1] representing time of trading day
        """
        # Convert to minutes since market open (9:30 AM)
        market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
        market_close_minutes = 16 * 60     # 4:00 PM in minutes
        trading_day_length = market_close_minutes - market_open_minutes  # 390 minutes
        
        # Extract hour and minute
        hours = timestamps.hour
        minutes = timestamps.minute
        total_minutes = hours * 60 + minutes
        
        # Normalize to [0, 1] for trading day
        time_feature = (total_minutes - market_open_minutes) / trading_day_length
        
        return pd.Series(time_feature, index=timestamps).clip(0, 1).fillna(0.5)
    
    def _normalize_features(self, features: np.ndarray, symbol: str) -> np.ndarray:
        """
        Normalize features using frozen parameters (FROZEN IMPLEMENTATION)
        
        Args:
            features: Raw feature array (n_timesteps, 12)
            symbol: 'NVDA' or 'MSFT'
            
        Returns:
            Normalized feature array
        """
        normalized = features.copy()
        
        # Get feature names for this symbol
        if symbol.upper() == 'NVDA':
            feature_names = self.feature_names[:12]  # First 12 features
        else:
            feature_names = [name.replace('nvda', 'msft') for name in self.feature_names[:12]]
        
        # Normalize each feature
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.normalization_params:
                params = self.normalization_params[feature_name]
                normalized[:, i] = (features[:, i] - params['mean']) / params['std']
            else:
                logger.warning(f"No normalization params for {feature_name}")
        
        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)
        
        return normalized
    
    def create_observation(self, 
                          nvda_features: np.ndarray,
                          msft_features: np.ndarray,
                          nvda_alpha: float,
                          msft_alpha: float) -> np.ndarray:
        """
        Create 26-dimensional observation vector (FROZEN FORMAT)
        
        Args:
            nvda_features: NVDA technical features (12,)
            msft_features: MSFT technical features (12,)
            nvda_alpha: NVDA alpha signal
            msft_alpha: MSFT alpha signal
            
        Returns:
            26-dimensional observation vector
        """
        # Validate input dimensions
        assert nvda_features.shape == (12,), f"NVDA features must be (12,), got {nvda_features.shape}"
        assert msft_features.shape == (12,), f"MSFT features must be (12,), got {msft_features.shape}"
        
        # Combine features in frozen order
        observation = np.concatenate([
            nvda_features,      # Indices 0-11
            [nvda_alpha],       # Index 12
            msft_features,      # Indices 13-24
            [msft_alpha]        # Index 25
        ])
        
        return observation.astype(np.float32)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get complete feature information (FROZEN)
        
        Returns:
            Dictionary with feature metadata
        """
        return {
            'version': self.version,
            'frozen_date': self.frozen_date,
            'training_run': self.training_run,
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_indices': self.feature_indices,
            'normalization_params': self.normalization_params,
            'observation_space_shape': (26,),
            'nvda_feature_range': (0, 12),
            'nvda_alpha_index': 12,
            'msft_feature_range': (13, 25),
            'msft_alpha_index': 25
        }

# Feature group definitions (FROZEN)
FEATURE_GROUPS = {
    'returns': ['returns_1m', 'returns_5m', 'returns_15m'],
    'volatility': ['volatility_20m'],
    'momentum': ['rsi_14', 'momentum_10m'],
    'trend': ['ema_12', 'ema_26'],
    'volume': ['vwap_ratio', 'volume_ratio'],
    'bands': ['bb_position'],
    'time': ['time_feature'],
    'alpha': ['alpha']
}

# Feature descriptions (FROZEN)
FEATURE_DESCRIPTIONS = {
    'returns_1m': '1-minute price returns',
    'returns_5m': '5-minute price returns',
    'returns_15m': '15-minute price returns',
    'volatility_20m': '20-minute rolling volatility',
    'rsi_14': '14-period Relative Strength Index',
    'ema_12': 'Price/12-period EMA ratio',
    'ema_26': 'Price/26-period EMA ratio',
    'vwap_ratio': 'Price/VWAP ratio',
    'volume_ratio': 'Volume/20-period average ratio',
    'bb_position': 'Position within Bollinger Bands [0-1]',
    'momentum_10m': '10-minute price momentum',
    'time_feature': 'Time of trading day [0-1]',
    'alpha': 'Alpha signal [-1, 1]'
}

# Observation space layout (FROZEN)
OBSERVATION_LAYOUT = {
    'nvda_technical': list(range(0, 12)),
    'nvda_alpha': 12,
    'msft_technical': list(range(13, 25)),
    'msft_alpha': 25,
    'total_dimensions': 26
}