# src/features/vwap_calculator.py
import pandas as pd
import numpy as np
from typing import List
from .base_calculator import BaseFeatureCalculator
from src.column_names import COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME


class VWAPCalculator(BaseFeatureCalculator):
    """
    Calculator for Volume Weighted Average Price (VWAP) indicators.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP and VWAP deviation.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with original data plus VWAP features
        """
        required_cols = [COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
        if not self.validate_data(data, required_cols):
            return data
            
        df = data.copy()
        window = self.config.get('window')
        group_by_day = self.config.get('group_by_day', True)
        
        try:
            vwap_series = self._calculate_vwap(
                df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], 
                df[COL_VOLUME], window, group_by_day
            )
            df['vwap'] = vwap_series
            
            # Calculate VWAP deviation
            df['vwap_deviation'] = (df[COL_CLOSE] - df['vwap']) / df['vwap']
            self.logger.info("Added 'vwap' and 'vwap_deviation' features")
            
        except Exception as e:
            df['vwap'] = self.handle_calculation_error(e, 'vwap', data.index)
            df['vwap_deviation'] = self.handle_calculation_error(
                e, 'vwap_deviation', data.index
            )
            
        return df
    
    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, window: int = None, 
                       group_by_day: bool = True) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high, low, close, volume: Price and volume series
            window: Rolling window size (None for daily/cumulative VWAP)
            group_by_day: Whether to calculate daily VWAP
            
        Returns:
            VWAP series
        """
        self.logger.debug(f"Calculating VWAP. Window: {window}, Group by Day: {group_by_day}")
        
        if not all(isinstance(s, pd.Series) for s in [high, low, close, volume]):
            self.logger.error("VWAP calculation requires pandas Series inputs")
            return pd.Series(np.nan, index=close.index, name='vwap_error')
            
        typical_price = (high + low + close) / 3
        pv = typical_price * volume

        if window is not None and window > 0:  # Rolling VWAP
            self.logger.debug(f"Calculating rolling VWAP with window {window}")
            rolling_pv_sum = pv.rolling(window=window, min_periods=1).sum()
            rolling_volume_sum = volume.rolling(window=window, min_periods=1).sum()
            vwap = (rolling_pv_sum / rolling_volume_sum).rename(f'vwap_roll_{window}')
            
        elif group_by_day:  # Daily VWAP
            self.logger.debug("Calculating daily VWAP")
            if not isinstance(close.index, pd.DatetimeIndex):
                self.logger.error("Daily VWAP requires a DatetimeIndex")
                return pd.Series(np.nan, index=close.index, name='vwap_daily_error')
            
            # Group by date and calculate cumulative VWAP within each day
            df_temp = pd.DataFrame({
                'pv': pv, 
                COL_VOLUME: volume, 
                'trading_date': close.index.date
            })
            df_temp['daily_cum_pv'] = df_temp.groupby('trading_date')['pv'].cumsum()
            df_temp['daily_cum_volume'] = df_temp.groupby('trading_date')[COL_VOLUME].cumsum()
            vwap = (df_temp['daily_cum_pv'] / df_temp['daily_cum_volume']).rename('vwap_daily')
            vwap.index = close.index
            
        else:  # Cumulative VWAP
            self.logger.warning("VWAP calculation parameters unclear. Defaulting to cumulative VWAP")
            vwap = (pv.cumsum() / volume.cumsum()).rename('vwap_cumulative')
        
        # Handle division by zero
        vwap.replace([np.inf, -np.inf], np.nan, inplace=True)
        vwap.ffill(inplace=True)
        
        return vwap
    
    def get_feature_names(self) -> List[str]:
        """Get VWAP feature names."""
        return ['vwap', 'vwap_deviation']
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period for VWAP."""
        window = self.config.get('window')
        if window is not None and window > 0:
            return window
        # For daily VWAP, return a reasonable default
        return 50