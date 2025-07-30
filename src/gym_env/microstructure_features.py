#!/usr/bin/env python3
"""
📊 MICROSTRUCTURAL FEATURE EXTRACTION V2
1-min micro-structural stats (spread, book-imbalance, VWAP dev)
5-min realized vol, NVDA↔MSFT correlation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class MicroFeatures:
    """Container for microstructural features"""
    # 1-minute microstructure
    spread: float                    # Bid-ask spread proxy
    book_imbalance: float           # Order book imbalance proxy  
    vwap_deviation: float           # Price deviation from VWAP
    trade_intensity: float          # Trading activity measure
    price_momentum: float           # Short-term momentum
    
    # 5-minute features  
    realized_vol: float             # 5-min realized volatility
    vol_of_vol: float              # Volatility of volatility
    skewness: float                # Return skewness
    kurtosis: float                # Return kurtosis
    
    # Cross-asset features
    correlation: float              # NVDA-MSFT correlation
    spread_ratio: float            # Relative spread between assets
    momentum_divergence: float     # Momentum difference
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.spread,
            self.book_imbalance,
            self.vwap_deviation,
            self.trade_intensity,
            self.price_momentum,
            self.realized_vol,
            self.vol_of_vol,
            self.skewness,
            self.kurtosis,
            self.correlation,
            self.spread_ratio,
            self.momentum_divergence
        ], dtype=np.float32)

class MicrostructuralFeatureExtractor:
    """
    Advanced feature extraction for dual-ticker trading
    Generates market microstructure features and cross-asset signals
    """
    
    def __init__(
        self,
        lookback_1min: int = 60,        # 1-hour lookback for microstructure
        lookback_5min: int = 288,       # 24-hour lookback for volatility (288 5-min bars)
        vol_scaling: float = 100.0,     # Volatility scaling factor
        corr_window: int = 50,          # Correlation calculation window
        eps: float = 1e-8               # Numerical stability
    ):
        self.lookback_1min = lookback_1min
        self.lookback_5min = lookback_5min
        self.vol_scaling = vol_scaling
        self.corr_window = corr_window
        self.eps = eps
        
        # State tracking
        self.nvda_history = []
        self.msft_history = []
        self.nvda_volume_history = []
        self.msft_volume_history = []
        
        logger.info(f"📊 MicrostructuralFeatureExtractor initialized:")
        logger.info(f"   🕐 1-min lookback: {lookback_1min}")
        logger.info(f"   🕐 5-min lookback: {lookback_5min}")
        logger.info(f"   📈 Correlation window: {corr_window}")
    
    def extract_features(
        self,
        nvda_ohlcv: Dict[str, float],
        msft_ohlcv: Dict[str, float],
        step: int
    ) -> Tuple[MicroFeatures, MicroFeatures]:
        """
        Extract microstructural features for both assets
        
        Args:
            nvda_ohlcv: NVDA OHLCV data point
            msft_ohlcv: MSFT OHLCV data point
            step: Current time step
            
        Returns:
            (nvda_features, msft_features): Feature objects for both assets
        """
        
        # Update price history
        self._update_history(nvda_ohlcv, msft_ohlcv)
        
        # Extract individual asset features
        nvda_features = self._extract_single_asset_features(
            self.nvda_history, self.nvda_volume_history, 'NVDA'
        )
        
        msft_features = self._extract_single_asset_features(
            self.msft_history, self.msft_volume_history, 'MSFT'
        )
        
        # Add cross-asset features
        self._add_cross_asset_features(nvda_features, msft_features)
        
        return nvda_features, msft_features
    
    def _extract_single_asset_features(
        self, price_history: List[Dict], volume_history: List[float], symbol: str
    ) -> MicroFeatures:
        """Extract features for a single asset"""
        
        if len(price_history) < 2:
            return self._get_default_features()
        
        # Recent data
        recent_prices = price_history[-self.lookback_1min:]
        recent_volumes = volume_history[-self.lookback_1min:]
        
        # 1-minute microstructure features
        spread = self._calculate_spread_proxy(recent_prices)
        book_imbalance = self._calculate_book_imbalance_proxy(recent_prices, recent_volumes)
        vwap_deviation = self._calculate_vwap_deviation(recent_prices, recent_volumes)
        trade_intensity = self._calculate_trade_intensity(recent_volumes)
        price_momentum = self._calculate_price_momentum(recent_prices)
        
        # 5-minute features (using longer history)
        if len(price_history) >= 10:
            longer_prices = price_history[-min(self.lookback_5min, len(price_history)):]
            realized_vol = self._calculate_realized_volatility(longer_prices)
            vol_of_vol = self._calculate_volatility_of_volatility(longer_prices)
            skewness = self._calculate_return_skewness(longer_prices)
            kurtosis = self._calculate_return_kurtosis(longer_prices)
        else:
            realized_vol = vol_of_vol = skewness = kurtosis = 0.0
        
        return MicroFeatures(
            spread=spread,
            book_imbalance=book_imbalance,
            vwap_deviation=vwap_deviation, 
            trade_intensity=trade_intensity,
            price_momentum=price_momentum,
            realized_vol=realized_vol,
            vol_of_vol=vol_of_vol,
            skewness=skewness,
            kurtosis=kurtosis,
            correlation=0.0,  # Will be set by cross-asset function
            spread_ratio=0.0,  # Will be set by cross-asset function
            momentum_divergence=0.0  # Will be set by cross-asset function
        )
    
    def _calculate_spread_proxy(self, price_history: List[Dict]) -> float:
        """Calculate bid-ask spread proxy from OHLC data"""
        if len(price_history) < 5:
            return 0.0
        
        # Use high-low range as spread proxy
        recent_spreads = []
        for p in price_history[-5:]:
            if p['high'] > p['low'] > 0:
                spread = (p['high'] - p['low']) / p['close']
                recent_spreads.append(spread)
        
        return np.mean(recent_spreads) if recent_spreads else 0.0
    
    def _calculate_book_imbalance_proxy(
        self, price_history: List[Dict], volume_history: List[float]
    ) -> float:
        """Calculate order book imbalance proxy"""
        if len(price_history) < 3:
            return 0.0
        
        # Use price movement vs volume as imbalance proxy
        recent_moves = []
        for i in range(1, min(6, len(price_history))):
            curr = price_history[-i]
            prev = price_history[-i-1]
            
            if prev['close'] > 0 and len(volume_history) >= i:
                price_change = (curr['close'] - prev['close']) / prev['close']
                volume = volume_history[-i] if volume_history[-i] > 0 else 1.0
                
                # Imbalance = price change / volume (normalized)
                normalized_vol = volume / np.mean(volume_history[-10:]) if len(volume_history) >= 10 else 1.0
                imbalance = price_change / (normalized_vol + self.eps)
                recent_moves.append(imbalance)
        
        return np.mean(recent_moves) if recent_moves else 0.0
    
    def _calculate_vwap_deviation(
        self, price_history: List[Dict], volume_history: List[float]
    ) -> float:
        """Calculate deviation from VWAP"""
        if len(price_history) < 10 or len(volume_history) < 10:
            return 0.0
        
        # Calculate VWAP over recent period
        recent_prices = price_history[-10:]
        recent_volumes = volume_history[-10:]
        
        total_value = sum(p['close'] * v for p, v in zip(recent_prices, recent_volumes))
        total_volume = sum(recent_volumes)
        
        if total_volume <= 0:
            return 0.0
        
        vwap = total_value / total_volume
        current_price = price_history[-1]['close']
        
        return (current_price - vwap) / vwap if vwap > 0 else 0.0
    
    def _calculate_trade_intensity(self, volume_history: List[float]) -> float:
        """Calculate trading intensity measure"""
        if len(volume_history) < 10:
            return 0.0
        
        recent_vol = np.mean(volume_history[-5:])
        baseline_vol = np.mean(volume_history[-20:]) if len(volume_history) >= 20 else np.mean(volume_history)
        
        return (recent_vol / baseline_vol - 1.0) if baseline_vol > 0 else 0.0
    
    def _calculate_price_momentum(self, price_history: List[Dict]) -> float:
        """Calculate short-term price momentum"""
        if len(price_history) < 5:
            return 0.0
        
        # Simple momentum: (current - MA5) / MA5
        recent_closes = [p['close'] for p in price_history[-5:]]
        current_price = recent_closes[-1]
        avg_price = np.mean(recent_closes[:-1])
        
        return (current_price - avg_price) / avg_price if avg_price > 0 else 0.0
    
    def _calculate_realized_volatility(self, price_history: List[Dict]) -> float:
        """Calculate realized volatility over 5-minute intervals"""
        if len(price_history) < 10:
            return 0.0
        
        # Calculate log returns
        closes = [p['close'] for p in price_history]
        log_returns = np.diff(np.log(closes + self.eps))
        
        # Realized volatility (annualized)
        realized_vol = np.std(log_returns) * np.sqrt(252 * 390)
        return realized_vol * self.vol_scaling
    
    def _calculate_volatility_of_volatility(self, price_history: List[Dict]) -> float:
        """Calculate volatility of volatility"""
        if len(price_history) < 20:
            return 0.0
        
        # Calculate rolling 5-period volatilities
        closes = [p['close'] for p in price_history]
        volatilities = []
        
        for i in range(5, len(closes)):
            period_closes = closes[i-5:i]
            returns = np.diff(np.log(period_closes + self.eps))
            vol = np.std(returns) if len(returns) > 1 else 0.0
            volatilities.append(vol)
        
        return np.std(volatilities) * self.vol_scaling if len(volatilities) > 1 else 0.0
    
    def _calculate_return_skewness(self, price_history: List[Dict]) -> float:
        """Calculate return skewness"""
        if len(price_history) < 10:
            return 0.0
        
        closes = [p['close'] for p in price_history]
        returns = np.diff(np.log(closes + self.eps))
        
        return float(stats.skew(returns)) if len(returns) > 3 else 0.0
    
    def _calculate_return_kurtosis(self, price_history: List[Dict]) -> float:
        """Calculate return kurtosis"""
        if len(price_history) < 10:
            return 0.0
        
        closes = [p['close'] for p in price_history]
        returns = np.diff(np.log(closes + self.eps))
        
        return float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0
    
    def _add_cross_asset_features(
        self, nvda_features: MicroFeatures, msft_features: MicroFeatures
    ):
        """Add cross-asset correlation and relative features"""
        
        if len(self.nvda_history) < self.corr_window or len(self.msft_history) < self.corr_window:
            correlation = 0.0
            spread_ratio = 0.0
            momentum_divergence = 0.0
        else:
            # Calculate correlation
            nvda_closes = [p['close'] for p in self.nvda_history[-self.corr_window:]]
            msft_closes = [p['close'] for p in self.msft_history[-self.corr_window:]]
            
            nvda_returns = np.diff(np.log(nvda_closes + self.eps))
            msft_returns = np.diff(np.log(msft_closes + self.eps))
            
            correlation = np.corrcoef(nvda_returns, msft_returns)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else float(correlation)
            
            # Spread ratio
            spread_ratio = (nvda_features.spread / (msft_features.spread + self.eps) - 1.0)
            
            # Momentum divergence
            momentum_divergence = nvda_features.price_momentum - msft_features.price_momentum
        
        # Update both feature objects
        nvda_features.correlation = correlation
        nvda_features.spread_ratio = spread_ratio
        nvda_features.momentum_divergence = momentum_divergence
        
        msft_features.correlation = correlation
        msft_features.spread_ratio = -spread_ratio  # Inverse for MSFT
        msft_features.momentum_divergence = -momentum_divergence  # Inverse for MSFT
    
    def _update_history(self, nvda_ohlcv: Dict[str, float], msft_ohlcv: Dict[str, float]):
        """Update price and volume history"""
        
        self.nvda_history.append(nvda_ohlcv)
        self.msft_history.append(msft_ohlcv)
        self.nvda_volume_history.append(nvda_ohlcv.get('volume', 1000.0))
        self.msft_volume_history.append(msft_ohlcv.get('volume', 1000.0))
        
        # Keep history manageable
        max_history = self.lookback_5min + 100
        if len(self.nvda_history) > max_history:
            self.nvda_history = self.nvda_history[-max_history:]
            self.msft_history = self.msft_history[-max_history:]
            self.nvda_volume_history = self.nvda_volume_history[-max_history:]
            self.msft_volume_history = self.msft_volume_history[-max_history:]
    
    def _get_default_features(self) -> MicroFeatures:
        """Return default features when insufficient data"""
        return MicroFeatures(
            spread=0.0, book_imbalance=0.0, vwap_deviation=0.0,
            trade_intensity=0.0, price_momentum=0.0, realized_vol=0.0,
            vol_of_vol=0.0, skewness=0.0, kurtosis=0.0,
            correlation=0.0, spread_ratio=0.0, momentum_divergence=0.0
        )
    
    def reset(self):
        """Reset state for new episode"""
        self.nvda_history.clear()
        self.msft_history.clear() 
        self.nvda_volume_history.clear()
        self.msft_volume_history.clear()
        
        logger.info("🔄 MicrostructuralFeatureExtractor reset")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for analysis"""
        return [
            'spread', 'book_imbalance', 'vwap_deviation',
            'trade_intensity', 'price_momentum', 'realized_vol',
            'vol_of_vol', 'skewness', 'kurtosis',
            'correlation', 'spread_ratio', 'momentum_divergence'
        ]
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimensionality"""
        return 12  # 12 features per asset