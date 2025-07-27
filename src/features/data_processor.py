# src/features/data_processor.py
import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME


class DataProcessor:
    """
    Handles data processing operations including normalization, scaling, and sequencing.
    Separated from feature computation for better modularity.
    """
    
    def __init__(self, config: Dict[str, Any], scalers_dir: str, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            scalers_dir: Directory to save/load scalers
            logger: Logger instance
        """
        self.config = config
        self.scalers_dir = scalers_dir
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.scaler: Optional[StandardScaler] = None
        self.lookback_window = self.config.get('lookback_window', 1)
        
        os.makedirs(self.scalers_dir, exist_ok=True)
        
    def normalize_features(self, features_df: pd.DataFrame, fit_scaler: bool = False, 
                          symbol: str = "default") -> Optional[pd.DataFrame]:
        """
        Normalize (scale) the feature columns using StandardScaler.
        
        Args:
            features_df: DataFrame with features to normalize
            fit_scaler: If True, fits a new scaler. Otherwise, uses existing scaler
            symbol: Symbol name for naming the saved scaler file
            
        Returns:
            DataFrame with normalized features, or None if error
        """
        if features_df is None or features_df.empty:
            self.logger.error("Cannot normalize empty or None DataFrame")
            return None

        scaler_path = os.path.join(self.scalers_dir, f"{symbol}_feature_scaler.pkl")
        
        # Determine which columns to scale
        feature_cols_to_scale = self.config.get('feature_cols_to_scale', [])
        if not feature_cols_to_scale:
            # Auto-detect columns to scale (exclude OHLCV and datetime columns)
            excluded_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, 'date', 'datetime']
            feature_cols_to_scale = [
                col for col in features_df.columns 
                if col not in excluded_cols and features_df[col].dtype in [np.float64, np.int64]
            ]
            self.logger.info(f"Auto-detected feature columns to scale: {feature_cols_to_scale}")

        if not feature_cols_to_scale:
            self.logger.warning("No feature columns specified or detected for scaling. Returning data as is")
            return features_df
            
        # Validate columns exist
        missing_cols = [col for col in feature_cols_to_scale if col not in features_df.columns]
        if missing_cols:
            self.logger.error(f"Columns specified for scaling are missing: {missing_cols}")
            return None

        data_to_scale = features_df[feature_cols_to_scale].copy()

        # Fit or load scaler
        if fit_scaler:
            self.scaler = StandardScaler()
            self.logger.info(f"Fitting new StandardScaler for {symbol} on columns: {feature_cols_to_scale}")
            self.scaler.fit(data_to_scale)
            self._save_scaler(scaler_path, symbol)
        else:
            if not self._load_or_create_scaler(scaler_path, symbol, data_to_scale):
                return None

        # Transform data
        try:
            normalized_data = self.scaler.transform(data_to_scale)
            normalized_df = pd.DataFrame(
                normalized_data, 
                columns=feature_cols_to_scale, 
                index=features_df.index
            )
            
            # Combine with non-scaled columns
            final_df = features_df.copy()
            for col in feature_cols_to_scale:
                final_df[col] = normalized_df[col]
                
            self.logger.info(f"Features normalized for {symbol}")
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error transforming features for {symbol}: {e}", exc_info=True)
            return None
    
    def _save_scaler(self, scaler_path: str, symbol: str):
        """Save the scaler to disk."""
        try:
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Scaler for {symbol} saved to {scaler_path}")
        except Exception as e:
            self.logger.error(f"Error saving scaler for {symbol}: {e}")
    
    def _load_or_create_scaler(self, scaler_path: str, symbol: str, 
                              data_to_scale: pd.DataFrame) -> bool:
        """Load existing scaler or create new one as fallback."""
        if self.scaler is None:
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    self.logger.info(f"Loaded existing StandardScaler for {symbol}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error loading scaler: {e}. Fitting a new one")
                    
            # Fallback: fit new scaler
            self.logger.warning(f"Scaler not found for {symbol}. Fitting new one (may cause data leakage)")
            self.scaler = StandardScaler()
            self.scaler.fit(data_to_scale)
            
        return self.scaler is not None
    
    def create_sequences(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Transform DataFrame of features into sequences (lookback windows).
        
        Args:
            features_df: DataFrame of features (potentially normalized)
            
        Returns:
            NumPy array of sequences with shape (num_samples, lookback_window, num_features)
        """
        if self.lookback_window <= 0:
            self.logger.warning("Lookback window is <= 0. Returning data as non-sequential array")
            return features_df.values

        if features_df is None or features_df.empty:
            self.logger.error("Cannot create sequences from empty DataFrame")
            return None
        
        if len(features_df) < self.lookback_window:
            self.logger.error(
                f"Data length ({len(features_df)}) is less than lookback window "
                f"({self.lookback_window}). Cannot create sequences"
            )
            return None

        data_values = features_df.values
        num_samples = data_values.shape[0] - self.lookback_window + 1
        num_features = data_values.shape[1]
        
        sequences = np.zeros((num_samples, self.lookback_window, num_features))
        
        for i in range(num_samples):
            sequences[i] = data_values[i : i + self.lookback_window]
            
        self.logger.info(f"Created sequences of shape: {sequences.shape}")
        return sequences
    
    def prepare_observation_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for model observation.
        
        Args:
            features_df: DataFrame with all computed features
            
        Returns:
            DataFrame with only observation features
        """
        obs_feature_cols = self.config.get('observation_feature_cols', features_df.columns.tolist())
        
        # Validate observation columns exist
        missing_obs_cols = [col for col in obs_feature_cols if col not in features_df.columns]
        if missing_obs_cols:
            self.logger.error(f"Observation feature columns missing: {missing_obs_cols}")
            obs_feature_cols = features_df.columns.tolist()
            
        return features_df[obs_feature_cols]
    
    def align_price_data_with_sequences(self, price_data: pd.Series, 
                                       num_sequences: int) -> pd.Series:
        """
        Align price data with feature sequences for reward calculation.
        
        Args:
            price_data: Series of price data
            num_sequences: Number of sequences created
            
        Returns:
            Aligned price data series
        """
        if self.lookback_window > 1:
            # Price data should align with sequence endpoints
            aligned_prices = price_data.iloc[self.lookback_window - 1:]
            
            if len(aligned_prices) != num_sequences:
                self.logger.warning(
                    f"Price data length ({len(aligned_prices)}) doesn't match "
                    f"sequence count ({num_sequences}) after alignment"
                )
            return aligned_prices
        else:
            return price_data
    
    def load_scaler(self, symbol: str) -> bool:
        """
        Load scaler for live trading.
        
        Args:
            symbol: Symbol to load scaler for
            
        Returns:
            True if successful, False otherwise
        """
        scaler_path = os.path.join(self.scalers_dir, f"{symbol}_feature_scaler.pkl")
        
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler for {symbol}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading scaler for {symbol}: {e}")
                return False
        else:
            self.logger.error(f"Scaler file not found for {symbol}")
            return False
    
    def transform_single_observation(self, features_row: pd.DataFrame, 
                                   symbol: str) -> Optional[np.ndarray]:
        """
        Transform a single row of features for live trading.
        
        Args:
            features_row: Single-row DataFrame with computed features
            symbol: Symbol name for context
            
        Returns:
            Transformed feature array or None if error
        """
        if self.scaler is None:
            self.logger.error(f"Scaler not loaded for {symbol}")
            return None
            
        feature_cols_to_scale = self.config.get('feature_cols_to_scale', [])
        if not feature_cols_to_scale:
            excluded_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, 'date', 'datetime']
            feature_cols_to_scale = [
                col for col in features_row.columns 
                if col not in excluded_cols and features_row[col].dtype in [np.float64, np.int64]
            ]
        
        # Validate columns
        if not all(col in features_row.columns for col in feature_cols_to_scale):
            self.logger.error(f"Missing columns for scaling in live data for {symbol}")
            return None
            
        try:
            data_to_scale = features_row[feature_cols_to_scale]
            normalized_data = self.scaler.transform(data_to_scale)
            return normalized_data[0]  # Return 1D array for single observation
            
        except Exception as e:
            self.logger.error(f"Error transforming live features for {symbol}: {e}")
            return None