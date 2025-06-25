# src/agents/feature_agent.py
import pandas as pd
import numpy as np
import os
# import ta # Technical Analysis library: pip install ta
from sklearn.preprocessing import StandardScaler # Example scaler
import joblib # For saving/loading scaler

from .base_agent import BaseAgent
# from ..utils.data_utils import load_data # Example utility

class FeatureAgent(BaseAgent):
    """
    FeatureAgent is responsible for:
    1. Loading raw bar data (typically fetched by DataAgent).
    2. Computing various technical indicators and features (RSI, EMA, VWAP deviation, time-of-day).
    3. Normalizing/Standardizing features.
    4. Rolling features into NumPy arrays suitable for the RL environment's observation space.
    5. Saving/Loading scalers to ensure consistent transformation between training and live trading.
    """
    def __init__(self, config: dict):
        """
        Initializes the FeatureAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'data_dir_processed': Path to save processed feature data.
                           'scalers_dir': Path to save fitted scalers.
                           'features': List of features to compute (e.g., ['RSI', 'EMA_12', 'EMA_26']).
                           'rsi': {'window': 14}
                           'ema': {'windows': [12, 26]}
                           'vwap': {} # VWAP specific params, if any
                           'time_features': ['hour', 'minute'] or False
                           'lookback_window': int, for creating sequences for LSTM/Transformer like models
        """
        super().__init__(agent_name="FeatureAgent", config=config)
        self.processed_data_dir = self.config.get('data_dir_processed', 'data/processed')
        self.scalers_dir = self.config.get('scalers_dir', 'data/scalers')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        self.feature_list = self.config.get('features', [])
        self.scaler = None # Will be initialized or loaded
        self.lookback_window = self.config.get('lookback_window', 1) # For sequential data

        self.logger.info(f"Processed data will be saved to: {self.processed_data_dir}")
        self.logger.info(f"Scalers will be saved to/loaded from: {self.scalers_dir}")
        self.logger.info(f"Features to compute: {self.feature_list}")
        self.logger.info(f"Lookback window for sequences: {self.lookback_window}")


    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        # TODO: Implement RSI calculation using `ta` library or manually.
        # from ta.momentum import RSIIndicator
        # rsi_indicator = RSIIndicator(close=series, window=window, fillna=True)
        # return rsi_indicator.rsi()
        self.logger.info(f"Calculating RSI with window {window} (using dummy implementation).")
        # Dummy implementation for skeleton
        return pd.Series(np.random.rand(len(series)) * 100, index=series.index, name=f'rsi_{window}')

    def _calculate_ema(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        # TODO: Implement EMA calculation using `ta` library or pandas.
        # from ta.trend import EMAIndicator
        # ema_indicator = EMAIndicator(close=series, window=window, fillna=True)
        # return ema_indicator.ema_indicator()
        self.logger.info(f"Calculating EMA with window {window} (using dummy implementation).")
        # Dummy implementation for skeleton
        return series.ewm(span=window, adjust=False).mean().rename(f'ema_{window}')


    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = None) -> pd.Series:
        """
        Calculates Volume Weighted Average Price (VWAP).
        If window is None, calculates daily VWAP. If window is an int, calculates rolling VWAP.
        """
        # TODO: Implement VWAP and VWAP deviation.
        # For daily VWAP: group by date and calculate cumulative (price * volume) / cumulative volume
        # For rolling VWAP: use a rolling window.
        self.logger.info(f"Calculating VWAP (using dummy implementation).")
        typical_price = (high + low + close) / 3
        if window: # Rolling VWAP
            pv = typical_price * volume
            # return (pv.rolling(window=window).sum() / volume.rolling(window=window).sum()).rename('vwap_rolling')
            # Dummy for rolling
            return pd.Series(typical_price + np.random.randn(len(typical_price)) * 0.1, index=close.index, name=f'vwap_roll_{window}')
        else: # Daily VWAP (requires grouping by day, more complex if data spans multiple days)
            # This is a simplified placeholder for daily VWAP, assuming data is for a single day or needs to be reset daily.
            # A proper daily VWAP resets at the start of each trading day.
            # pv = typical_price * volume
            # return (pv.cumsum() / volume.cumsum()).rename('vwap_daily')
            # Dummy for daily
            return pd.Series(typical_price + np.random.randn(len(typical_price)) * 0.1, index=close.index, name='vwap_daily')

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds time-based features like hour of day, minute of hour, day of week."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex. Cannot add time features.")
            return df
        
        time_feature_conf = self.config.get('time_features', [])
        if 'hour' in time_feature_conf:
            df['hour_of_day'] = df.index.hour
            self.logger.info("Added 'hour_of_day' feature.")
        if 'minute' in time_feature_conf:
            df['minute_of_hour'] = df.index.minute
            self.logger.info("Added 'minute_of_hour' feature.")
        if 'day_of_week' in time_feature_conf:
            df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
            self.logger.info("Added 'day_of_week' feature.")
        # TODO: Consider sinusoidal encoding for cyclical features (hour, minute, day_of_week)
        # E.g., df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        #       df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        return df

    def compute_features(self, raw_data_df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Computes all configured features on the raw data.

        Args:
            raw_data_df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
                                        and a DatetimeIndex.

        Returns:
            pd.DataFrame or None: DataFrame with original data and added features, or None if error.
        """
        if not isinstance(raw_data_df.index, pd.DatetimeIndex):
            self.logger.error("Raw data DataFrame must have a DatetimeIndex.")
            return None
            
        df = raw_data_df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Raw data missing one or more required columns: {required_cols}")
            return None

        # --- RSI ---
        if 'RSI' in self.feature_list and 'rsi' in self.config:
            rsi_params = self.config['rsi']
            df[f'rsi_{rsi_params.get("window", 14)}'] = self._calculate_rsi(df['close'], rsi_params.get("window", 14))

        # --- EMAs ---
        if 'EMA' in self.feature_list and 'ema' in self.config:
            ema_params = self.config['ema']
            for window in ema_params.get('windows', []):
                df[f'ema_{window}'] = self._calculate_ema(df['close'], window)
            # Example: EMA difference (MACD-like)
            if 'ema_diff' in ema_params and len(ema_params.get('windows', [])) == 2:
                fast_ema_col = f'ema_{ema_params["windows"][0]}'
                slow_ema_col = f'ema_{ema_params["windows"][1]}'
                if fast_ema_col in df and slow_ema_col in df:
                    df['ema_diff'] = df[fast_ema_col] - df[slow_ema_col]
                    self.logger.info(f"Added 'ema_diff' feature ({fast_ema_col} - {slow_ema_col}).")


        # --- VWAP & VWAP Deviation ---
        if 'VWAP' in self.feature_list and 'vwap' in self.config:
            vwap_params = self.config['vwap']
            # For simplicity, using a daily VWAP placeholder. A rolling window VWAP might also be useful.
            df['vwap'] = self._calculate_vwap(df['high'], df['low'], df['close'], df['volume'], window=vwap_params.get("window"))
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] # Percentage deviation
            self.logger.info("Added 'vwap' and 'vwap_deviation' features.")

        # --- Time Features ---
        if self.config.get('time_features', False):
            df = self._add_time_features(df)

        # TODO: Add optional order-book imbalance features if such data is available and processed.
        # This would likely come from a different data source or a more granular DataAgent output.

        # Drop rows with NaNs created by indicators with initial warm-up periods
        # Or decide on a fill strategy (e.g., fill with mean, 0, or forward fill)
        original_len = len(df)
        df.dropna(inplace=True)
        self.logger.info(f"Dropped {original_len - len(df)} rows due to NaNs after feature calculation.")
        
        if df.empty:
            self.logger.warning("DataFrame is empty after feature calculation and NaN drop. Check indicator warm-up periods and data length.")
            return None
            
        # Select only the columns that will be used as features for the model + close for reward calculation
        # The final feature set for the model might exclude OHLCV if they are not directly used.
        # This depends on the observation space definition.
        # For now, we keep them and allow selection later.
        self.logger.info(f"Computed features. DataFrame columns: {df.columns.tolist()}")
        return df

    def normalize_features(self, features_df: pd.DataFrame, fit_scaler: bool = False, symbol: str = "default") -> pd.DataFrame | None:
        """
        Normalizes (scales) the feature columns.
        Uses StandardScaler by default. Fits the scaler if `fit_scaler` is True or if no scaler is loaded.
        Saves the scaler if `fit_scaler` is True.

        Args:
            features_df (pd.DataFrame): DataFrame with features to normalize.
            fit_scaler (bool): If True, fits a new scaler. Otherwise, uses existing/loaded scaler.
            symbol (str): Symbol name, used for naming the saved scaler file.

        Returns:
            pd.DataFrame or None: DataFrame with normalized features, or None if error.
        """
        if features_df is None or features_df.empty:
            self.logger.error("Cannot normalize empty or None DataFrame.")
            return None

        scaler_path = os.path.join(self.scalers_dir, f"{symbol}_feature_scaler.pkl")
        
        feature_cols_to_scale = self.config.get('feature_cols_to_scale', [])
        if not feature_cols_to_scale: # If not specified, try to scale all non-OHLCV and non-datetime columns
            excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'datetime'] # Common exclusions
            feature_cols_to_scale = [col for col in features_df.columns if col not in excluded_cols and features_df[col].dtype in [np.float64, np.int64]]
            self.logger.info(f"Auto-detected feature columns to scale: {feature_cols_to_scale}")

        if not feature_cols_to_scale:
            self.logger.warning("No feature columns specified or detected for scaling. Returning data as is.")
            return features_df
            
        # Ensure all columns to scale are actually in the dataframe
        missing_cols = [col for col in feature_cols_to_scale if col not in features_df.columns]
        if missing_cols:
            self.logger.error(f"Columns specified for scaling are missing from DataFrame: {missing_cols}")
            return None

        data_to_scale = features_df[feature_cols_to_scale].copy()

        if fit_scaler:
            self.scaler = StandardScaler()
            self.logger.info(f"Fitting new StandardScaler for {symbol} on columns: {feature_cols_to_scale}")
            self.scaler.fit(data_to_scale)
            try:
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"Scaler for {symbol} saved to {scaler_path}")
            except Exception as e:
                self.logger.error(f"Error saving scaler for {symbol} to {scaler_path}: {e}")
        else:
            if self.scaler is None: # Try to load if not already in memory
                if os.path.exists(scaler_path):
                    try:
                        self.scaler = joblib.load(scaler_path)
                        self.logger.info(f"Loaded existing StandardScaler for {symbol} from {scaler_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading scaler from {scaler_path}: {e}. Fitting a new one.")
                        self.scaler = StandardScaler()
                        self.scaler.fit(data_to_scale) # Fallback: fit on current data
                else:
                    self.logger.warning(f"Scaler file not found at {scaler_path} and fit_scaler is False. Fitting a new one on current data (may lead to data leakage if this is test data).")
                    self.scaler = StandardScaler()
                    self.scaler.fit(data_to_scale) # Fallback: fit on current data

        if self.scaler is None:
             self.logger.error(f"Scaler for {symbol} is not available. Cannot normalize features.")
             return None

        normalized_data = self.scaler.transform(data_to_scale)
        normalized_df = pd.DataFrame(normalized_data, columns=feature_cols_to_scale, index=features_df.index)
        
        # Combine back with non-scaled columns (like 'close', 'volume' if needed by env)
        # Create a new DataFrame to avoid modifying the original features_df structure unnecessarily here
        final_df = features_df.copy()
        for col in feature_cols_to_scale:
            final_df[col] = normalized_df[col]
            
        self.logger.info(f"Features normalized for {symbol}.")
        return final_df

    def create_sequences(self, features_df: pd.DataFrame) -> np.ndarray | None:
        """
        Transforms the DataFrame of features into sequences (lookback windows).
        The output shape will be (num_samples, lookback_window, num_features).
        This is typically used for environments that expect sequential input (e.g., for LSTMs).

        Args:
            features_df (pd.DataFrame): DataFrame of features (potentially normalized).
                                        It's assumed that the columns in this DF are the ones
                                        to be included in the sequence.

        Returns:
            np.ndarray or None: NumPy array of sequences, or None if error.
        """
        if self.lookback_window <= 0:
            self.logger.warning("Lookback window is <= 0. Returning data as non-sequential NumPy array.")
            return features_df.values

        if features_df is None or features_df.empty:
            self.logger.error("Cannot create sequences from empty or None DataFrame.")
            return None
        
        if len(features_df) < self.lookback_window:
            self.logger.error(f"Data length ({len(features_df)}) is less than lookback window ({self.lookback_window}). Cannot create sequences.")
            return None

        data_values = features_df.values
        num_samples = data_values.shape[0] - self.lookback_window + 1
        num_features = data_values.shape[1]
        
        sequences = np.zeros((num_samples, self.lookback_window, num_features))
        
        for i in range(num_samples):
            sequences[i] = data_values[i : i + self.lookback_window]
            
        self.logger.info(f"Created sequences of shape: {sequences.shape}")
        return sequences
        
    def get_processed_filepath(self, symbol: str, start_date: str, end_date: str, interval: str, data_format: str = "npy") -> str:
        """Helper to generate a consistent filepath for processed data."""
        # Ensure dates are in a consistent format for filenames
        start_f = start_date.replace('-', '').split(' ')[0]
        end_f = end_date.replace('-', '').split(' ')[0]
        interval_f = interval.replace(' ', '')
        filename = f"{symbol}_{start_f}_{end_f}_{interval_f}_features.{data_format}"
        return os.path.join(self.processed_data_dir, symbol, filename)

    def run(self, raw_data_df: pd.DataFrame, symbol: str, 
            cache_processed_data: bool = True, fit_scaler: bool = False,
            # Date params are for cache naming consistency with DataAgent
            start_date_str: str = "unknown", end_date_str: str = "unknown", interval_str: str = "unknown" 
            ) -> tuple[pd.DataFrame | None, np.ndarray | None, pd.Series | None]:
        """
        Main method to process raw data: compute features, normalize, and create sequences.

        Args:
            raw_data_df (pd.DataFrame): Raw market data.
            symbol (str): Stock symbol, used for naming saved scalers and processed data.
            cache_processed_data (bool): Whether to save the processed features (before sequencing) to disk.
            fit_scaler (bool): Whether to fit a new scaler (True for training) or use existing (False for testing/live).
            start_date_str, end_date_str, interval_str (str): Metadata for naming cached processed files.

        Returns:
            tuple(pd.DataFrame | None, np.ndarray | None, pd.Series | None):
                - final_features_df: DataFrame of features (normalized, before sequencing). Retains index.
                - feature_sequences: NumPy array of feature sequences (if lookback_window > 0). Index is lost.
                                     If lookback_window <=0, this is just `final_features_df.values`.
                - price_data_for_env: Series of 'close' prices aligned with the `final_features_df` index,
                                      to be used by the environment for reward calculation.
                                      If sequencing is applied, this needs to be sliced to match the
                                      number of samples in `feature_sequences`.
        """
        self.logger.info(f"FeatureAgent run triggered for {symbol}.")
        if raw_data_df is None or raw_data_df.empty:
            self.logger.error("Raw data is empty. Cannot process features.")
            return None, None, None

        # 1. Compute Features
        features_df = self.compute_features(raw_data_df)
        if features_df is None:
            self.logger.error("Feature computation failed.")
            return None, None, None

        # 2. Normalize Features
        # The columns to scale are defined in config: 'feature_cols_to_scale'
        # If not defined, it attempts to scale all numeric columns not in OHLCV.
        normalized_features_df = self.normalize_features(features_df, fit_scaler=fit_scaler, symbol=symbol)
        if normalized_features_df is None:
            self.logger.error("Feature normalization failed.")
            return None, None, None # Or return unnormalized features_df?

        final_features_df = normalized_features_df
        
        # Extract price data (e.g., 'close' prices) for the environment
        # This should be done BEFORE sequencing, using the same index as final_features_df
        # The environment will need this to calculate P&L.
        # Make sure 'close' prices are from the original data, not scaled.
        if 'close' not in final_features_df.columns:
            self.logger.error("'close' column not found in features DataFrame. Cannot provide price data for env.")
            # Potentially, 'close' might have been in raw_data_df but not explicitly carried over or was scaled.
            # Ensure 'close' (unscaled) is available.
            # If 'close' was scaled, we should use the original 'close' from features_df (before normalization)
            # or raw_data_df, aligned with final_features_df.index.
            if 'close' in features_df: # from before normalization
                 price_data_for_env = features_df.loc[final_features_df.index, 'close'].copy()
            elif 'close' in raw_data_df: # from raw_data, aligned
                 price_data_for_env = raw_data_df.loc[final_features_df.index, 'close'].copy()
            else:
                 price_data_for_env = None # Should not happen if data pipeline is correct
        else: # 'close' is in final_features_df (might be scaled if not excluded)
            # It is safer to take 'close' from features_df (before scaling) or raw_data_df
            if 'close' in features_df:
                 price_data_for_env = features_df.loc[final_features_df.index, 'close'].copy()
            else: # Fallback, though 'close' ideally shouldn't be scaled if used for P&L
                 price_data_for_env = final_features_df['close'].copy()


        # 3. Create Sequences (if lookback_window > 1)
        # The environment usually expects only the feature values, not OHLCV unless they are features.
        # Select only the final set of features intended for the observation space before sequencing.
        obs_feature_cols = self.config.get('observation_feature_cols', final_features_df.columns.tolist())
        # Ensure all obs_feature_cols are in final_features_df
        missing_obs_cols = [col for col in obs_feature_cols if col not in final_features_df.columns]
        if missing_obs_cols:
            self.logger.error(f"Observation feature columns {missing_obs_cols} not found in the dataframe. Using all columns.")
            obs_feature_cols = final_features_df.columns.tolist()
            
        df_for_sequencing = final_features_df[obs_feature_cols]

        if self.lookback_window > 1 : # lookback_window=1 means single step, no sequence needed beyond current obs
            feature_sequences = self.create_sequences(df_for_sequencing)
            if feature_sequences is None:
                self.logger.error("Sequence creation failed.")
                return final_features_df, None, price_data_for_env # Return what we have so far

            # If sequences were created, the price_data_for_env needs to be aligned with the sequences.
            # A sequence starting at index `k` (from original data) corresponds to making a decision
            # at the end of the sequence, i.e., at index `k + lookback_window - 1`.
            # The price at this point is relevant for calculating the reward for an action taken.
            if price_data_for_env is not None:
                price_data_for_env = price_data_for_env.iloc[self.lookback_window - 1:]
                if len(price_data_for_env) != feature_sequences.shape[0]:
                    self.logger.warning(f"Mismatch in length between price data ({len(price_data_for_env)}) and feature sequences ({feature_sequences.shape[0]}) after alignment.")
                    # This might require adjustment based on how exactly rewards are calculated for sequences.
                    # Typically, the price at time `t` is used for action at `t`, and reward is based on price at `t+1`.
                    # If obs at `t` is sequence `s_t = [x_{t-L+1}, ..., x_t]`, action `a_t` is taken.
                    # Reward `r_t` may depend on `p_{t+1}`.
                    # So, `price_data_for_env` should align with the *end points* of sequences.
        elif self.lookback_window == 1: # Single observation, not a sequence over time
             feature_sequences = df_for_sequencing.values # Shape: (num_samples, num_features)
             # Price data alignment is direct
        else: # lookback_window <= 0, e.g. for non-sequential models that take current features_df.values
             feature_sequences = df_for_sequencing.values
             # Price data alignment is direct

        if cache_processed_data:
            try:
                # Save the final_features_df (DataFrame before sequencing, but normalized)
                # And/Or the feature_sequences (NumPy array)
                # For simplicity, let's save final_features_df as CSV and feature_sequences as NPY
                # This cache path should be robust using actual start/end dates of the data.
                # Using passed strings for now.
                df_cache_path = self.get_processed_filepath(symbol, start_date_str, end_date_str, interval_str, "csv")
                npy_cache_path = self.get_processed_filepath(symbol, start_date_str, end_date_str, interval_str, "npy")

                os.makedirs(os.path.dirname(df_cache_path), exist_ok=True)
                final_features_df.to_csv(df_cache_path)
                self.logger.info(f"Saved processed DataFrame features to {df_cache_path}")

                if feature_sequences is not None:
                    np.save(npy_cache_path, feature_sequences)
                    self.logger.info(f"Saved processed NumPy feature sequences to {npy_cache_path}")

            except Exception as e:
                self.logger.error(f"Error caching processed data for {symbol}: {e}")
        
        self.logger.info(f"FeatureAgent run completed for {symbol}.")
        return final_features_df, feature_sequences, price_data_for_env


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Configuration
    config = {
        'data_dir_processed': 'data/processed_test',
        'scalers_dir': 'data/scalers_test',
        'features': ['RSI', 'EMA', 'VWAP'], # Which categories of features to compute
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True}, # Compute EMA(12), EMA(26), and their difference
        'vwap': {'window': None}, # Daily VWAP (placeholder logic)
        'time_features': ['hour', 'day_of_week'], # Add hour and day_of_week
        'lookback_window': 5, # Create sequences of 5 timesteps
        # Define which columns from the feature_df should be scaled
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation', 'hour_of_day', 'day_of_week'],
        # Define which columns (after scaling) should be part of the observation sequence for the model
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation', 'hour_of_day', 'day_of_week', 'close', 'volume'] # Example
    }

    feature_agent = FeatureAgent(config=config)

    # Create dummy raw data (as DataAgent would provide)
    dates = pd.to_datetime([
        '2023-01-01 09:30:00', '2023-01-01 09:31:00', '2023-01-01 09:32:00', '2023-01-01 09:33:00',
        '2023-01-01 09:34:00', '2023-01-01 09:35:00', '2023-01-01 09:36:00', '2023-01-01 09:37:00',
        '2023-01-01 09:38:00', '2023-01-01 09:39:00', '2023-01-01 09:40:00', '2023-01-01 09:41:00',
        '2023-01-01 09:42:00', '2023-01-01 09:43:00', '2023-01-01 09:44:00', '2023-01-01 09:45:00',
    ])
    num_rows = len(dates)
    data = {
        'open': np.random.rand(num_rows) * 10 + 100,
        'high': np.random.rand(num_rows) * 5 + 105, # ensure high > open
        'low': np.random.rand(num_rows) * 5 + 95,   # ensure low < open
        'close': np.random.rand(num_rows) * 10 + 100,
        'volume': np.random.randint(100, 1000, size=num_rows) * 10
    }
    raw_df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='date'))
    raw_df['high'] = raw_df[['open', 'close']].max(axis=1) + np.random.rand(num_rows) * 2
    raw_df['low'] = raw_df[['open', 'close']].min(axis=1) - np.random.rand(num_rows) * 2

    print("--- Dummy Raw Data ---")
    print(raw_df.head())

    # Run the FeatureAgent (fit_scaler=True as this would be 'training' data)
    # Using dummy date strings for cache naming example
    final_df, sequences, prices = feature_agent.run(
        raw_df, 
        symbol="DUMMY_STOCK", 
        fit_scaler=True,
        start_date_str="20230101", 
        end_date_str="20230101", 
        interval_str="1min"
    )

    if final_df is not None:
        print("\n--- Final Features DataFrame (normalized, before sequencing) ---")
        print(final_df.head())
        print(f"Shape: {final_df.shape}")
        
        # Verify scaler was saved
        scaler_path = os.path.join(config['scalers_dir'], "DUMMY_STOCK_feature_scaler.pkl")
        if os.path.exists(scaler_path):
            print(f"Scaler saved to {scaler_path}")
        else:
            print(f"Scaler NOT saved to {scaler_path}")
            
        # Verify processed data cache
        df_cache_path = feature_agent.get_processed_filepath("DUMMY_STOCK", "20230101", "20230101", "1min", "csv")
        if os.path.exists(df_cache_path):
            print(f"Processed DataFrame cache saved to {df_cache_path}")
        else:
            print(f"Processed DataFrame cache NOT saved to {df_cache_path}")


    if sequences is not None:
        print("\n--- Feature Sequences (for RL agent observation) ---")
        print(sequences[0]) # Print the first sequence
        print(f"Shape: {sequences.shape}")
        
        npy_cache_path = feature_agent.get_processed_filepath("DUMMY_STOCK", "20230101", "20230101", "1min", "npy")
        if os.path.exists(npy_cache_path):
            print(f"NumPy sequence cache saved to {npy_cache_path}")
        else:
            print(f"NumPy sequence cache NOT saved to {npy_cache_path}")


    if prices is not None:
        print("\n--- Price Data for Environment (aligned with features) ---")
        print(prices.head())
        print(f"Shape: {prices.shape}")
        if sequences is not None and config['lookback_window'] > 1:
             # Expected length of prices should match number of sequences
            expected_len = raw_df.shape[0] - len(raw_df) + sequences.shape[0] # Simplified, depends on NaN drops
            # More directly: prices length should match sequences.shape[0] after alignment
            if len(prices) == sequences.shape[0]:
                print("Price data length matches sequence data length.")
            else:
                print(f"Price data length ({len(prices)}) MISMATCHES sequence data length ({sequences.shape[0]}). Check alignment logic.")
    
    print("\nFeatureAgent example run complete.")
