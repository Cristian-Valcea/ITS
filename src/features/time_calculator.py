# src/features/time_calculator.py
import pandas as pd
import numpy as np
from typing import List
from .base_calculator import BaseFeatureCalculator


class TimeFeatureCalculator(BaseFeatureCalculator):
    """
    Calculator for time-based features with optional sin/cos encoding.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features.
        
        Args:
            data: DataFrame containing data with DatetimeIndex
            
        Returns:
            DataFrame with original data plus time features
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex. Cannot add time features")
            return data
            
        df = data.copy()
        
        # Handle both old and new config formats
        time_features_config = self.config.get('time_features', {})
        if isinstance(time_features_config, dict):
            # New format: time_features: {include_hour: true, use_cyclical_encoding: true}
            include_hour = time_features_config.get('include_hour', False)
            include_minute = time_features_config.get('include_minute', False)
            include_day_of_week = time_features_config.get('include_day_of_week', False)
            include_month = time_features_config.get('include_month', False)
            use_cyclical = time_features_config.get('use_cyclical_encoding', False)
            
            # Add time features based on new config
            if include_hour:
                df['hour_of_day'] = df.index.hour
                if use_cyclical:
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
                    df.drop(columns=['hour_of_day'], inplace=True)
                    
            if include_minute:
                df['minute_of_hour'] = df.index.minute
                if use_cyclical:
                    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60.0)
                    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60.0)
                    df.drop(columns=['minute_of_hour'], inplace=True)
                    
            if include_day_of_week:
                df['day_of_week'] = df.index.dayofweek
                if use_cyclical:
                    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
                    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
                    df.drop(columns=['day_of_week'], inplace=True)
                    
            if include_month:
                df['month_of_year'] = df.index.month
                if use_cyclical:
                    df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12.0)
                    df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12.0)
                    df.drop(columns=['month_of_year'], inplace=True)
                    
            return df
        else:
            # Old format: time_features: ['hour_of_day', 'minute_of_hour']
            time_features = time_features_config if isinstance(time_features_config, list) else []
            sin_cos_encode = self.config.get('sin_cos_encode', [])
        
        # Add basic time features
        if 'hour_of_day' in time_features:
            df['hour_of_day'] = df.index.hour
            self.logger.debug("Added 'hour_of_day' feature")
            
        if 'minute_of_hour' in time_features:
            df['minute_of_hour'] = df.index.minute
            self.logger.debug("Added 'minute_of_hour' feature")
            
        if 'day_of_week' in time_features:
            df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
            self.logger.debug("Added 'day_of_week' feature")
            
        if 'month_of_year' in time_features:
            df['month_of_year'] = df.index.month
            self.logger.debug("Added 'month_of_year' feature")
        
        # Apply sin/cos encoding for cyclical features
        df = self._apply_sin_cos_encoding(df, sin_cos_encode)
        
        return df
    
    def _apply_sin_cos_encoding(self, df: pd.DataFrame, sin_cos_config: List[str]) -> pd.DataFrame:
        """
        Apply sin/cos encoding to cyclical time features.
        
        Args:
            df: DataFrame with time features
            sin_cos_config: List of features to encode
            
        Returns:
            DataFrame with sin/cos encoded features
        """
        if 'hour_of_day' in sin_cos_config and 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
            df.drop(columns=['hour_of_day'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'hour_of_day'")
            
        if 'minute_of_hour' in sin_cos_config and 'minute_of_hour' in df.columns:
            df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60.0)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60.0)
            df.drop(columns=['minute_of_hour'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'minute_of_hour'")
            
        if 'day_of_week' in sin_cos_config and 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
            df.drop(columns=['day_of_week'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'day_of_week'")
            
        if 'month_of_year' in sin_cos_config and 'month_of_year' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12.0)
            df.drop(columns=['month_of_year'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'month_of_year'")
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get time feature names."""
        time_features_config = self.config.get('time_features', {})
        feature_names = []
        
        if isinstance(time_features_config, dict):
            # New format
            include_hour = time_features_config.get('include_hour', False)
            include_minute = time_features_config.get('include_minute', False)
            include_day_of_week = time_features_config.get('include_day_of_week', False)
            include_month = time_features_config.get('include_month', False)
            use_cyclical = time_features_config.get('use_cyclical_encoding', False)
            
            if include_hour:
                if use_cyclical:
                    feature_names.extend(['hour_sin', 'hour_cos'])
                else:
                    feature_names.append('hour_of_day')
                    
            if include_minute:
                if use_cyclical:
                    feature_names.extend(['minute_sin', 'minute_cos'])
                else:
                    feature_names.append('minute_of_hour')
                    
            if include_day_of_week:
                if use_cyclical:
                    feature_names.extend(['day_of_week_sin', 'day_of_week_cos'])
                else:
                    feature_names.append('day_of_week')
                    
            if include_month:
                if use_cyclical:
                    feature_names.extend(['month_sin', 'month_cos'])
                else:
                    feature_names.append('month_of_year')
        else:
            # Old format
            time_features = time_features_config if isinstance(time_features_config, list) else []
            sin_cos_encode = self.config.get('sin_cos_encode', [])
            
            # Add basic time features (if not sin/cos encoded)
            if 'hour_of_day' in time_features:
                if 'hour_of_day' in sin_cos_encode:
                    feature_names.extend(['hour_sin', 'hour_cos'])
                else:
                    feature_names.append('hour_of_day')
                    
            if 'minute_of_hour' in time_features:
                if 'minute_of_hour' in sin_cos_encode:
                    feature_names.extend(['minute_sin', 'minute_cos'])
                else:
                    feature_names.append('minute_of_hour')
                    
            if 'day_of_week' in time_features:
                if 'day_of_week' in sin_cos_encode:
                    feature_names.extend(['day_of_week_sin', 'day_of_week_cos'])
                else:
                    feature_names.append('day_of_week')
                    
            if 'month_of_year' in time_features:
                if 'month_of_year' in sin_cos_encode:
                    feature_names.extend(['month_sin', 'month_cos'])
                else:
                    feature_names.append('month_of_year')
                    
        return feature_names
    
    def get_max_lookback(self) -> int:
        """Get maximum lookback period for time features."""
        return 0  # Time features don't require historical data