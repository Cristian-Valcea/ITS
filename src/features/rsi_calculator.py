# src/features/rsi_calculator.py
import pandas as pd
import numpy as np
import ta
from typing import List
from .base_calculator import BaseFeatureCalculator
from src.column_names import COL_CLOSE


class RSICalculator(BaseFeatureCalculator):
    """
    Calculator for Relative Strength Index (RSI) indicator.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with original data plus RSI features
        """
        if not self.validate_data(data, [COL_CLOSE]):
            return data
            
        df = data.copy()
        window = self.config.get('window', 14)
        
        try:
            self.logger.debug(f"Calculating RSI with window {window}")
            rsi_indicator = ta.momentum.RSIIndicator(
                close=df[COL_CLOSE], 
                window=window, 
                fillna=True
            )
            rsi_series = rsi_indicator.rsi()
            df[f'rsi_{window}'] = rsi_series.rename(f'rsi_{window}')
            
        except Exception as e:
            df[f'rsi_{window}'] = self.handle_calculation_error(
                e, f'rsi_{window}', data.index
            )
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get RSI feature names."""
        window = self.config.get('window', 14)
        return [f'rsi_{window}']
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period for RSI."""
        return self.config.get('window', 14)