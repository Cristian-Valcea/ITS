# src/agents/feature_agent.py
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
from collections import deque

from src.column_names import COL_CLOSE
from src.features import FeatureManager, DataProcessor
from src.shared import FeatureStore, get_feature_store
from .base_agent import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Refactored FeatureAgent using modular feature computation and data processing.
    
    Responsibilities:
    1. Orchestrate feature computation using FeatureManager
    2. Handle data processing using DataProcessor
    3. Manage live trading sessions
    4. Provide clean interface for batch and live processing
    """
    
    def __init__(self, config: dict):
        super().__init__(agent_name="FeatureAgent", config=config)
        
        # Initialize directories
        self.processed_data_dir = self.config.get('data_dir_processed', 'data/processed')
        self.scalers_dir = self.config.get('scalers_dir', 'data/scalers')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Initialize modular components
        self.feature_manager = FeatureManager(config=self.config, logger=self.logger)
        self.data_processor = DataProcessor(
            config=self.config, 
            scalers_dir=self.scalers_dir, 
            logger=self.logger
        )
        
        # Initialize FeatureStore for high-performance caching
        feature_store_root = self.config.get('feature_store_root', None)
        self.feature_store = get_feature_store(root=feature_store_root, logger=self.logger)
        self.use_feature_cache = self.config.get('use_feature_cache', True)
        
        # Live trading attributes
        self.live_data_buffer = pd.DataFrame()
        self.normalized_feature_history_buffer = deque(
            maxlen=self.data_processor.lookback_window
        )
        self._is_live_session_initialized = False
        
        self._log_initialization_info()
    
    def _log_initialization_info(self):
        """Log initialization information."""
        self.logger.info(f"Processed data directory: {self.processed_data_dir}")
        self.logger.info(f"Scalers directory: {self.scalers_dir}")
        self.logger.info(f"Active feature calculators: {self.feature_manager.list_active_calculators()}")
        self.logger.info(f"Lookback window: {self.data_processor.lookback_window}")
        self.logger.info(f"Max indicator lookback: {self.feature_manager.get_max_lookback()}")
        self.logger.info(f"Feature caching enabled: {self.use_feature_cache}")
        if self.use_feature_cache:
            cache_stats = self.feature_store.get_cache_stats()
            self.logger.info(f"Feature cache stats: {cache_stats.get('total_entries', 0)} entries, "
                           f"{cache_stats.get('total_size_mb', 0)} MB")
    
    def compute_features(self, raw_data_df: pd.DataFrame, symbol: str = "default") -> Optional[pd.DataFrame]:
        """
        Compute features using the FeatureManager with FeatureStore caching.
        
        Args:
            raw_data_df: DataFrame with OHLCV data and DatetimeIndex
            symbol: Symbol name for cache key generation
            
        Returns:
            DataFrame with computed features or None if error
        """
        if not self.use_feature_cache:
            # Direct computation without caching
            return self.feature_manager.compute_features(raw_data_df)
        
        # Use FeatureStore for cached computation
        try:
            # Create feature configuration for cache key
            feature_config = {
                'active_calculators': self.feature_manager.list_active_calculators(),
                'feature_config': self.feature_manager.feature_config,
                'max_lookback': self.feature_manager.get_max_lookback()
            }
            
            # Define computation function for FeatureStore
            def compute_func(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
                return self.feature_manager.compute_features(raw_df)
            
            return self.feature_store.get_or_compute(
                symbol=symbol,
                raw_df=raw_data_df,
                config=feature_config,
                compute_func=compute_func
            )
            
        except Exception as e:
            self.logger.error(f"Error in cached feature computation: {e}")
            # Fallback to direct computation
            self.logger.info("Falling back to direct feature computation")
            return self.feature_manager.compute_features(raw_data_df)
    
    def normalize_features(self, features_df: pd.DataFrame, fit_scaler: bool = False, 
                          symbol: str = "default") -> Optional[pd.DataFrame]:
        """
        Normalize features using the DataProcessor.
        
        Args:
            features_df: DataFrame with features to normalize
            fit_scaler: Whether to fit a new scaler
            symbol: Symbol name for scaler file naming
            
        Returns:
            DataFrame with normalized features or None if error
        """
        return self.data_processor.normalize_features(features_df, fit_scaler, symbol)
    
    def create_sequences(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Create feature sequences using the DataProcessor.
        
        Args:
            features_df: DataFrame of features
            
        Returns:
            NumPy array of sequences or None if error
        """
        return self.data_processor.create_sequences(features_df)
    
    def run(self, raw_data_df: pd.DataFrame, symbol: str, 
            cache_processed_data: bool = True, fit_scaler: bool = False,
            start_date_str: str = "unknown", end_date_str: str = "unknown", 
            interval_str: str = "unknown") -> Tuple[Optional[pd.DataFrame], 
                                                   Optional[np.ndarray], 
                                                   Optional[pd.Series]]:
        """
        Main processing pipeline: compute features, normalize, and create sequences.
        
        Args:
            raw_data_df: Raw market data
            symbol: Stock symbol
            cache_processed_data: Whether to save processed data to disk
            fit_scaler: Whether to fit new scaler (True for training)
            start_date_str, end_date_str, interval_str: Metadata for caching
            
        Returns:
            Tuple of (final_features_df, feature_sequences, price_data_for_env)
        """
        self.logger.info(f"FeatureAgent processing started for {symbol}")
        
        if raw_data_df is None or raw_data_df.empty:
            self.logger.error("Raw data is empty")
            return None, None, None

        # 1. Compute features (with caching)
        features_df = self.compute_features(raw_data_df, symbol=symbol)
        if features_df is None:
            self.logger.error("Feature computation failed")
            return None, None, None

        # 2. Normalize features
        normalized_features_df = self.normalize_features(
            features_df, fit_scaler=fit_scaler, symbol=symbol
        )
        if normalized_features_df is None:
            self.logger.error("Feature normalization failed")
            return None, None, None

        # 3. Extract price data for environment (before sequencing)
        price_data_for_env = self._extract_price_data(
            normalized_features_df, features_df, raw_data_df
        )

        # 4. Prepare observation features and create sequences
        obs_features_df = self.data_processor.prepare_observation_features(normalized_features_df)
        
        if self.data_processor.lookback_window > 1:
            feature_sequences = self.create_sequences(obs_features_df)
            if feature_sequences is None:
                self.logger.error("Sequence creation failed")
                return normalized_features_df, None, price_data_for_env
                
            # Align price data with sequences
            if price_data_for_env is not None:
                price_data_for_env = self.data_processor.align_price_data_with_sequences(
                    price_data_for_env, feature_sequences.shape[0]
                )
        else:
            feature_sequences = obs_features_df.values

        # 5. Cache processed data if requested
        if cache_processed_data:
            self._cache_processed_data(
                normalized_features_df, feature_sequences, symbol,
                start_date_str, end_date_str, interval_str
            )

        self.logger.info(f"FeatureAgent processing completed for {symbol}")
        return normalized_features_df, feature_sequences, price_data_for_env
    
    def warm_feature_cache(self, raw_data_df: pd.DataFrame, symbol: str):
        """
        Warm the feature cache by pre-computing features for given data.
        Useful for offline preloading before training runs.
        
        Args:
            raw_data_df: Raw market data
            symbol: Stock symbol
        """
        if not self.use_feature_cache:
            self.logger.info("Feature caching disabled, skipping cache warming")
            return
            
        self.logger.info(f"Warming feature cache for {symbol} with {len(raw_data_df)} rows")
        
        try:
            # Create feature configuration for cache key
            feature_config = {
                'active_calculators': self.feature_manager.list_active_calculators(),
                'feature_config': self.feature_manager.feature_config,
                'max_lookback': self.feature_manager.get_max_lookback()
            }
            
            # Define computation function
            def compute_func(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
                return self.feature_manager.compute_features(raw_df)
            
            self.feature_store.warm_cache(
                symbol=symbol,
                raw_df=raw_data_df,
                config=feature_config,
                compute_func=compute_func
            )
            
            self.logger.info(f"Feature cache warming completed for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error warming feature cache for {symbol}: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get feature cache statistics."""
        if not self.use_feature_cache:
            return {'caching_disabled': True}
        return self.feature_store.get_cache_stats()
    
    def clear_feature_cache(self, symbol: Optional[str] = None):
        """
        Clear feature cache.
        
        Args:
            symbol: If provided, only clear cache for this symbol
        """
        if not self.use_feature_cache:
            self.logger.info("Feature caching disabled, nothing to clear")
            return
            
        self.feature_store.clear_cache(symbol=symbol)
        if symbol:
            self.logger.info(f"Cleared feature cache for {symbol}")
        else:
            self.logger.info("Cleared entire feature cache")
    
    def _extract_price_data(self, final_features_df: pd.DataFrame, 
                           features_df: pd.DataFrame, 
                           raw_data_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract price data for environment reward calculation."""
        if COL_CLOSE not in final_features_df.columns:
            # Try to get unscaled CLOSE from features_df or raw_data_df
            if COL_CLOSE in features_df.columns:
                return features_df.loc[final_features_df.index, COL_CLOSE].copy()
            elif COL_CLOSE in raw_data_df.columns:
                return raw_data_df.loc[final_features_df.index, COL_CLOSE].copy()
            else:
                self.logger.error("CLOSE column not found for price data extraction")
                return None
        else:
            # Use CLOSE from features_df if available (unscaled), otherwise from final_features_df
            if COL_CLOSE in features_df.columns:
                return features_df.loc[final_features_df.index, COL_CLOSE].copy()
            else:
                return final_features_df[COL_CLOSE].copy()
    
    def _cache_processed_data(self, final_features_df: pd.DataFrame, 
                             feature_sequences: Optional[np.ndarray], symbol: str,
                             start_date_str: str, end_date_str: str, interval_str: str):
        """Cache processed data to disk."""
        try:
            df_cache_path = self.get_processed_filepath(
                symbol, start_date_str, end_date_str, interval_str, "csv"
            )
            npy_cache_path = self.get_processed_filepath(
                symbol, start_date_str, end_date_str, interval_str, "npy"
            )

            os.makedirs(os.path.dirname(df_cache_path), exist_ok=True)
            
            # Save DataFrame
            final_features_df.to_csv(df_cache_path)
            self.logger.info(f"Saved processed DataFrame to {df_cache_path}")

            # Save sequences if available
            if feature_sequences is not None:
                np.save(npy_cache_path, feature_sequences)
                self.logger.info(f"Saved feature sequences to {npy_cache_path}")

        except Exception as e:
            self.logger.error(f"Error caching processed data for {symbol}: {e}")
    
    def get_processed_filepath(self, symbol: str, start_date: str, end_date: str, 
                              interval: str, data_format: str = "npy") -> str:
        """Generate consistent filepath for processed data."""
        start_f = start_date.replace('-', '').split(' ')[0]
        end_f = end_date.replace('-', '').split(' ')[0]
        interval_f = interval.replace(' ', '')
        filename = f"{symbol}_{start_f}_{end_f}_{interval_f}_features.{data_format}"
        return os.path.join(self.processed_data_dir, symbol, filename)
    
    # --- Live Trading Methods ---
    
    def initialize_live_session(self, symbol: str, 
                               historical_data_for_warmup: Optional[pd.DataFrame] = None):
        """
        Initialize live trading session.
        
        Args:
            symbol: Symbol for live trading
            historical_data_for_warmup: Optional historical data for indicator warmup
        """
        self.logger.info(f"Initializing live trading session for {symbol}")
        self._is_live_session_initialized = False

        # Load scaler
        if not self.data_processor.load_scaler(symbol):
            self.logger.error(f"Failed to load scaler for {symbol}")
            return

        # Initialize buffers
        self.live_data_buffer = pd.DataFrame()
        self.normalized_feature_history_buffer.clear()

        # Warmup with historical data if provided
        if historical_data_for_warmup is not None and not historical_data_for_warmup.empty:
            self._warmup_live_session(historical_data_for_warmup, symbol)

        self._is_live_session_initialized = True
        self.logger.info("Live trading session initialized successfully")
    
    def _warmup_live_session(self, historical_data: pd.DataFrame, symbol: str):
        """Warmup live session with historical data."""
        self.logger.info(f"Warming up with {len(historical_data)} historical bars")
        
        max_lookback = self.feature_manager.get_max_lookback()
        
        # Trim historical data to required length
        if len(historical_data) > max_lookback:
            historical_data = historical_data.iloc[-max_lookback:]
            
        self.live_data_buffer = historical_data.copy()
        
        # Pre-compute features and populate history buffer
        if len(self.live_data_buffer) >= max_lookback:
            warmup_features = self.compute_features(self.live_data_buffer, symbol=symbol)
            if warmup_features is not None and not warmup_features.empty:
                warmup_normalized = self.normalize_features(
                    warmup_features, fit_scaler=False, symbol=symbol
                )
                if warmup_normalized is not None:
                    self._populate_history_buffer(warmup_normalized)
    
    def _populate_history_buffer(self, normalized_features: pd.DataFrame):
        """Populate normalized feature history buffer from warmup data."""
        obs_features = self.data_processor.prepare_observation_features(normalized_features)
        
        # Take the most recent lookback_window observations
        num_to_take = min(len(obs_features), self.data_processor.lookback_window)
        for i in range(num_to_take):
            row_idx = -(num_to_take - i)
            self.normalized_feature_history_buffer.append(obs_features.iloc[row_idx].values)
            
        self.logger.info(f"Populated history buffer with {len(self.normalized_feature_history_buffer)} entries")
    
    def process_live_bar(self, new_bar_df: pd.DataFrame, 
                        symbol: str) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
        """
        Process a single live bar and generate observation sequence.
        
        Args:
            new_bar_df: Single-row DataFrame with new bar data
            symbol: Symbol name
            
        Returns:
            Tuple of (observation_sequence, latest_price_series)
        """
        if not self._is_live_session_initialized:
            self.logger.error("Live session not initialized")
            return None, None

        if not isinstance(new_bar_df.index, pd.DatetimeIndex) or len(new_bar_df) != 1:
            self.logger.error("New bar must be single-row DataFrame with DatetimeIndex")
            return None, None

        latest_price_series = new_bar_df.iloc[-1].copy()

        # 1. Update live data buffer
        self._update_live_buffer(new_bar_df)

        # 2. Compute features on current buffer
        current_features = self.compute_features(self.live_data_buffer, symbol=symbol)
        if current_features is None or current_features.empty:
            self.logger.warning("Feature computation on live buffer yielded no data")
            return None, latest_price_series

        # 3. Process latest features
        latest_observation = self._process_latest_features(current_features, symbol)
        if latest_observation is None:
            return None, latest_price_series

        # 4. Update history buffer and create sequence
        self.normalized_feature_history_buffer.append(latest_observation)

        if len(self.normalized_feature_history_buffer) < self.data_processor.lookback_window:
            self.logger.debug("Not enough history for full sequence")
            return None, latest_price_series

        observation_sequence = np.array(list(self.normalized_feature_history_buffer))

        # Check for NaNs
        if np.isnan(observation_sequence).any():
            self.logger.warning("NaNs detected in observation sequence")

        return observation_sequence, latest_price_series
    
    def _update_live_buffer(self, new_bar_df: pd.DataFrame):
        """Update live data buffer with new bar."""
        # Append new bar
        if pd.__version__ >= "1.4.0":
            self.live_data_buffer = pd.concat([self.live_data_buffer, new_bar_df])
        else:
            self.live_data_buffer = self.live_data_buffer.append(new_bar_df)

        # Trim buffer to reasonable size
        max_lookback = self.feature_manager.get_max_lookback()
        buffer_limit = max_lookback + 20  # Keep some extra margin
        
        if len(self.live_data_buffer) > buffer_limit:
            self.live_data_buffer = self.live_data_buffer.iloc[-buffer_limit:]
    
    def _process_latest_features(self, current_features: pd.DataFrame, 
                                symbol: str) -> Optional[np.ndarray]:
        """Process latest computed features for live trading."""
        # Get latest row of computed features
        latest_features_row = current_features.iloc[-1:]
        
        # Transform using data processor
        transformed_features = self.data_processor.transform_single_observation(
            latest_features_row, symbol
        )
        
        if transformed_features is None:
            return None
            
        # Combine with observation features
        obs_feature_cols = self.config.get('observation_feature_cols', 
                                          current_features.columns.tolist())
        
        final_observation = []
        feature_cols_to_scale = self.config.get('feature_cols_to_scale', [])
        
        for col in obs_feature_cols:
            if col in feature_cols_to_scale:
                # Use transformed (scaled) value
                col_idx = feature_cols_to_scale.index(col)
                final_observation.append(transformed_features[col_idx])
            elif col in latest_features_row.columns:
                # Use original (unscaled) value
                final_observation.append(latest_features_row[col].iloc[0])
            else:
                self.logger.error(f"Observation column '{col}' not found")
                final_observation.append(np.nan)
        
        return np.array(final_observation)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Configuration
    config = {
        'data_dir_processed': 'data/processed_test',
        'scalers_dir': 'data/scalers_test',
        'features': ['RSI', 'EMA', 'VWAP', 'Time'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True},
        'vwap': {'window': None},
        'time': {'time_features': ['hour_of_day', 'day_of_week'], 'sin_cos_encode': ['hour_of_day']},
        'lookback_window': 5,
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation'],
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation', 'hour_sin', 'hour_cos']
    }

    feature_agent = FeatureAgent(config=config)

    # Create dummy raw data
    from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME
    
    dates = pd.to_datetime([
        '2023-01-01 09:30:00', '2023-01-01 09:31:00', '2023-01-01 09:32:00', '2023-01-01 09:33:00',
        '2023-01-01 09:34:00', '2023-01-01 09:35:00', '2023-01-01 09:36:00', '2023-01-01 09:37:00',
        '2023-01-01 09:38:00', '2023-01-01 09:39:00', '2023-01-01 09:40:00', '2023-01-01 09:41:00',
        '2023-01-01 09:42:00', '2023-01-01 09:43:00', '2023-01-01 09:44:00', '2023-01-01 09:45:00',
    ])
    num_rows = len(dates)
    data = {
        COL_OPEN: np.random.rand(num_rows) * 10 + 100,
        COL_HIGH: np.random.rand(num_rows) * 5 + 105,
        COL_LOW: np.random.rand(num_rows) * 5 + 95,
        COL_CLOSE: np.random.rand(num_rows) * 10 + 100,
        COL_VOLUME: np.random.randint(100, 1000, size=num_rows) * 10
    }
    raw_df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='date'))
    raw_df[COL_HIGH] = raw_df[[COL_OPEN, COL_CLOSE]].max(axis=1) + np.random.rand(num_rows) * 2
    raw_df[COL_LOW] = raw_df[[COL_OPEN, COL_CLOSE]].min(axis=1) - np.random.rand(num_rows) * 2

    print("--- Dummy Raw Data ---")
    print(raw_df.head())

    # Run the FeatureAgent
    final_df, sequences, prices = feature_agent.run(
        raw_df, 
        symbol="DUMMY_STOCK", 
        fit_scaler=True,
        start_date_str="20230101", 
        end_date_str="20230101", 
        interval_str="1min"
    )

    if final_df is not None:
        print("\n--- Final Features DataFrame ---")
        print(final_df.head())
        print(f"Shape: {final_df.shape}")

    if sequences is not None:
        print("\n--- Feature Sequences ---")
        print(f"Shape: {sequences.shape}")

    if prices is not None:
        print("\n--- Price Data ---")
        print(f"Shape: {prices.shape}")
    
    print("\nModular FeatureAgent example run complete.")