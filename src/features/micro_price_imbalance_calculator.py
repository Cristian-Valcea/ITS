# src/features/micro_price_imbalance_calculator.py
"""
Micro-Price Imbalance Calculator

Calculates micro-price imbalance features that capture order flow dynamics
and market microstructure effects. These features help reduce Q-value variance
by providing insights into short-term price pressure and liquidity dynamics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_calculator import BaseFeatureCalculator


class MicroPriceImbalanceCalculator(BaseFeatureCalculator):
    """
    Calculator for micro-price imbalance and order flow features.
    
    Provides microstructural features that capture:
    - Order flow imbalance
    - Price pressure dynamics
    - Liquidity-based signals
    - Market impact indicators
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize micro-price imbalance calculator.
        
        Args:
            config: Configuration containing imbalance parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # Micro-price imbalance configuration
        self.imbalance_config = config.get('micro_price_imbalance', {})
        self.windows = self.imbalance_config.get('windows', [5, 10, 20])  # Short-term windows
        self.volume_windows = self.imbalance_config.get('volume_windows', [10, 30])  # Volume analysis windows
        self.price_impact_window = self.imbalance_config.get('price_impact_window', 5)
        
        # Ensure windows are lists
        if isinstance(self.windows, int):
            self.windows = [self.windows]
        if isinstance(self.volume_windows, int):
            self.volume_windows = [self.volume_windows]
        
        # Calculate max lookback
        all_windows = self.windows + self.volume_windows + [self.price_impact_window]
        self._max_lookback = max(all_windows) + 20  # Extra buffer
        
        self.logger.info(f"Micro-Price Imbalance Calculator initialized with windows: {self.windows}, "
                        f"volume windows: {self.volume_windows}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate micro-price imbalance and order flow features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with micro-price imbalance features added
        """
        if len(data) < max(self.windows + self.volume_windows):
            self.logger.warning(f"Insufficient data for micro-price imbalance calculation. "
                              f"Need {max(self.windows + self.volume_windows)}, got {len(data)}")
            return self._add_nan_features(data)
        
        result = data.copy()
        
        try:
            # Calculate basic price and volume features
            result = self._calculate_price_features(result)
            
            # Calculate order flow imbalance proxies
            result = self._calculate_order_flow_imbalance(result)
            
            # Calculate volume-based imbalance features
            result = self._calculate_volume_imbalance(result)
            
            # Calculate price pressure and impact features
            result = self._calculate_price_pressure(result)
            
            # Calculate liquidity-based features
            result = self._calculate_liquidity_features(result)
            
            self.logger.debug(f"Micro-price imbalance calculation completed for {len(result)} rows")
            
        except Exception as e:
            self.logger.error(f"Error calculating micro-price imbalance: {e}")
            result = self._add_nan_features(data)
        
        return result
    
    def _calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price-based features for imbalance analysis."""
        result = data.copy()
        
        # Price changes and returns
        result['price_change'] = result['close'].diff()
        result['price_return'] = result['close'].pct_change()
        
        # High-Low spread (proxy for intrabar volatility)
        result['hl_spread'] = result['high'] - result['low']
        result['hl_spread_normalized'] = result['hl_spread'] / result['close']
        
        # Open-Close spread (intrabar directional movement)
        result['oc_spread'] = result['close'] - result['open']
        result['oc_spread_normalized'] = result['oc_spread'] / result['open']
        
        # Typical price and weighted close
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        result['weighted_close'] = (result['high'] + result['low'] + 2 * result['close']) / 4
        
        return result
    
    def _calculate_order_flow_imbalance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow imbalance proxies."""
        result = data.copy()
        
        # Volume-weighted price change (proxy for order flow)
        result['volume_weighted_return'] = result['price_return'] * result['volume']
        
        # Tick direction (simplified without actual tick data)
        result['tick_direction'] = np.where(
            result['price_change'] > 0, 1,
            np.where(result['price_change'] < 0, -1, 0)
        )
        
        # Volume imbalance (buy vs sell volume proxy)
        # Assume volume on upticks is buy volume, downticks is sell volume
        result['buy_volume_proxy'] = np.where(result['tick_direction'] > 0, result['volume'], 0)
        result['sell_volume_proxy'] = np.where(result['tick_direction'] < 0, result['volume'], 0)
        
        for window in self.windows:
            # Rolling order flow imbalance
            buy_vol_sum = result['buy_volume_proxy'].rolling(window=window).sum()
            sell_vol_sum = result['sell_volume_proxy'].rolling(window=window).sum()
            total_vol_sum = buy_vol_sum + sell_vol_sum
            
            result[f'order_flow_imbalance_{window}'] = (
                (buy_vol_sum - sell_vol_sum) / (total_vol_sum + 1e-8)
            )
            
            # Volume-weighted average tick direction
            result[f'vwap_tick_direction_{window}'] = (
                (result['tick_direction'] * result['volume']).rolling(window=window).sum() /
                (result['volume'].rolling(window=window).sum() + 1e-8)
            )
            
            # Order flow momentum
            result[f'order_flow_momentum_{window}'] = (
                result[f'order_flow_imbalance_{window}'].diff()
            )
        
        return result
    
    def _calculate_volume_imbalance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based imbalance features."""
        result = data.copy()
        
        for vol_window in self.volume_windows:
            # Volume moving average and ratio
            vol_ma = result['volume'].rolling(window=vol_window).mean()
            result[f'volume_ratio_{vol_window}'] = result['volume'] / (vol_ma + 1e-8)
            
            # Volume-price correlation
            result[f'volume_price_corr_{vol_window}'] = (
                result['volume'].rolling(window=vol_window)
                .corr(result['price_return'])
            )
            
            # Volume acceleration
            result[f'volume_acceleration_{vol_window}'] = (
                result['volume'].pct_change().rolling(window=vol_window).mean()
            )
            
            # Relative volume strength
            vol_std = result['volume'].rolling(window=vol_window).std()
            result[f'volume_strength_{vol_window}'] = (
                (result['volume'] - vol_ma) / (vol_std + 1e-8)
            )
        
        return result
    
    def _calculate_price_pressure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price pressure and market impact features."""
        result = data.copy()
        
        # Price impact per unit volume
        result['price_impact'] = np.abs(result['price_change']) / (result['volume'] + 1e-8)
        
        # Rolling price impact
        result[f'price_impact_ma_{self.price_impact_window}'] = (
            result['price_impact'].rolling(window=self.price_impact_window).mean()
        )
        
        # Price pressure (cumulative volume-weighted price change)
        for window in self.windows:
            cumulative_vw_return = (
                result['volume_weighted_return'].rolling(window=window).sum()
            )
            cumulative_volume = result['volume'].rolling(window=window).sum()
            
            result[f'price_pressure_{window}'] = (
                cumulative_vw_return / (cumulative_volume + 1e-8)
            )
            
            # Price momentum adjusted for volume
            price_momentum = result['price_return'].rolling(window=window).sum()
            volume_momentum = result['volume_ratio_' + str(self.volume_windows[0])].rolling(window=window).mean()
            
            result[f'volume_adjusted_momentum_{window}'] = (
                price_momentum * volume_momentum
            )
        
        # Market impact efficiency (price change per volume unit)
        result['market_impact_efficiency'] = (
            np.abs(result['price_return']) / (result['volume_ratio_' + str(self.volume_windows[0])] + 1e-8)
        )
        
        return result
    
    def _calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity-based features."""
        result = data.copy()
        
        # Liquidity proxy (inverse of price impact)
        result['liquidity_proxy'] = 1 / (result['price_impact'] + 1e-8)
        
        # Bid-ask spread proxy (using high-low range)
        result['spread_proxy'] = result['hl_spread_normalized']
        
        # Liquidity-adjusted returns
        result['liquidity_adjusted_return'] = (
            result['price_return'] * result['liquidity_proxy']
        )
        
        # Market depth proxy (volume relative to price movement)
        result['market_depth_proxy'] = (
            result['volume'] / (np.abs(result['price_change']) + 1e-8)
        )
        
        # Liquidity momentum
        for window in self.windows:
            result[f'liquidity_momentum_{window}'] = (
                result['liquidity_proxy'].rolling(window=window).mean().pct_change()
            )
            
            # Volume-liquidity interaction
            result[f'volume_liquidity_interaction_{window}'] = (
                result[f'volume_ratio_{self.volume_windows[0]}'] * 
                result['liquidity_proxy'].rolling(window=window).mean()
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
        
        # Basic price features
        features.extend([
            'price_change', 'price_return', 'hl_spread', 'hl_spread_normalized',
            'oc_spread', 'oc_spread_normalized', 'typical_price', 'weighted_close'
        ])
        
        # Order flow features
        features.extend([
            'volume_weighted_return', 'tick_direction', 'buy_volume_proxy', 'sell_volume_proxy'
        ])
        
        for window in self.windows:
            features.extend([
                f'order_flow_imbalance_{window}',
                f'vwap_tick_direction_{window}',
                f'order_flow_momentum_{window}',
                f'price_pressure_{window}',
                f'volume_adjusted_momentum_{window}',
                f'liquidity_momentum_{window}',
                f'volume_liquidity_interaction_{window}'
            ])
        
        # Volume features
        for vol_window in self.volume_windows:
            features.extend([
                f'volume_ratio_{vol_window}',
                f'volume_price_corr_{vol_window}',
                f'volume_acceleration_{vol_window}',
                f'volume_strength_{vol_window}'
            ])
        
        # Price pressure and liquidity features
        features.extend([
            'price_impact',
            f'price_impact_ma_{self.price_impact_window}',
            'market_impact_efficiency',
            'liquidity_proxy',
            'spread_proxy',
            'liquidity_adjusted_return',
            'market_depth_proxy'
        ])
        
        return features
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period required."""
        return self._max_lookback