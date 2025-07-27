# src/features/ema_calculator.py
import pandas as pd
import numpy as np
from typing import List
from .base_calculator import BaseFeatureCalculator
from src.column_names import COL_CLOSE


class EMACalculator(BaseFeatureCalculator):
    """
    Calculator for Exponential Moving Average (EMA) indicators.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA indicators.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with original data plus EMA features
        """
        if not self.validate_data(data, [COL_CLOSE]):
            return data
            
        df = data.copy()
        windows = self.config.get('windows', [])
        
        # Calculate individual EMAs
        for window in windows:
            try:
                self.logger.debug(f"Calculating EMA with window {window}")
                ema_series = df[COL_CLOSE].ewm(
                    span=window, 
                    adjust=False, 
                    min_periods=window
                ).mean().rename(f'ema_{window}')
                df[f'ema_{window}'] = ema_series
                
            except Exception as e:
                df[f'ema_{window}'] = self.handle_calculation_error(
                    e, f'ema_{window}', data.index
                )
        
        # Calculate EMA difference (MACD-like) if requested
        if (self.config.get('ema_diff', False) and 
            len(windows) == 2 and 
            all(f'ema_{w}' in df.columns for w in windows)):
            
            try:
                fast_ema_col = f'ema_{windows[0]}'
                slow_ema_col = f'ema_{windows[1]}'
                df['ema_diff'] = df[fast_ema_col] - df[slow_ema_col]
                self.logger.info(f"Added 'ema_diff' feature ({fast_ema_col} - {slow_ema_col})")
                
            except Exception as e:
                df['ema_diff'] = self.handle_calculation_error(
                    e, 'ema_diff', data.index
                )
                
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get EMA feature names."""
        windows = self.config.get('windows', [])
        feature_names = [f'ema_{window}' for window in windows]
        
        if self.config.get('ema_diff', False) and len(windows) == 2:
            feature_names.append('ema_diff')
            
        return feature_names
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period for EMA."""
        windows = self.config.get('windows', [])
        return max(windows) if windows else 0