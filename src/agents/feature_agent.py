# src/agents/feature_agent.py
import pandas as pd
import numpy as np
import os
import ta # Technical Analysis library: pip install ta
from sklearn.preprocessing import StandardScaler
import joblib 

from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME

from .base_agent import BaseAgent

class FeatureAgent(BaseAgent):
    """
    FeatureAgent is responsible for:
    1. Loading raw bar data.
    2. Computing technical indicators (RSI, EMA, VWAP deviation, time-of-day).
    3. Normalizing/Standardizing features.
    4. Rolling features into NumPy arrays for the RL environment.
    5. Saving/Loading scalers.
    """
    def __init__(self, config: dict):
        super().__init__(agent_name="FeatureAgent", config=config)
        self.processed_data_dir = self.config.get('data_dir_processed', 'data/processed')
        self.scalers_dir = self.config.get('scalers_dir', 'data/scalers')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        self.feature_config = self.config.get('feature_engineering', {}) # Main key for all feature params
        # If feature_engineering is not nested, the config itself might contain the feature params
        if not self.feature_config:
            self.feature_config = self.config
        self.feature_list = self.feature_config.get('features', []) # List of features to compute
        self.scaler = None
        self.lookback_window = self.feature_config.get('lookback_window', 1) # For observation sequences

        # For live processing
        self.live_data_buffer = pd.DataFrame() # Buffer for raw OHLCV bars for live feature calculation
        self.normalized_feature_history_buffer = [] # Stores recent N rows of normalized features for sequencing
        self._max_indicator_lookback = 0 # Determined from feature configs, for live_data_buffer size
        self._is_live_session_initialized = False


        self.logger.info(f"Processed data will be saved to: {self.processed_data_dir}")
        self.logger.info(f"Scalers will be saved/loaded from: {self.scalers_dir}")
        self.logger.info(f"Features to compute based on config: {self.feature_list}")
        self.logger.info(f"Lookback window for sequences: {self.lookback_window}")

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates Relative Strength Index (RSI) using 'ta' library."""
        self.logger.debug(f"Calculating RSI with window {window}.")
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=series, window=window, fillna=True)
            rsi_series = rsi_indicator.rsi()
            return rsi_series.rename(f'rsi_{window}')
        except Exception as e:
            self.logger.error(f"Error calculating RSI with window {window}: {e}", exc_info=True)
            return pd.Series(np.nan, index=series.index, name=f'rsi_{window}')


    def _calculate_ema(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA) using pandas."""
        self.logger.debug(f"Calculating EMA with window {window}.")
        try:
            return series.ewm(span=window, adjust=False, min_periods=window).mean().rename(f'ema_{window}')
        except Exception as e:
            self.logger.error(f"Error calculating EMA with window {window}: {e}", exc_info=True)
            return pd.Series(np.nan, index=series.index, name=f'ema_{window}')


    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = None, group_by_day: bool = True) -> pd.Series:
        """
        Calculates Volume Weighted Average Price (VWAP).
        If window is None and group_by_day is True, calculates daily VWAP.
        If window is an int, calculates rolling VWAP over that window.
        """
        self.logger.debug(f"Calculating VWAP. Window: {window}, Group by Day: {group_by_day}")
        if not all(isinstance(s, pd.Series) for s in [high, low, close, volume]):
            self.logger.error("VWAP calculation requires pandas Series inputs.")
            return pd.Series(np.nan, index=close.index, name='vwap_error')
            
        typical_price = (high + low + close) / 3
        pv = typical_price * volume

        if window is not None and window > 0: # Rolling VWAP
            self.logger.debug(f"Calculating rolling VWAP with window {window}.")
            rolling_pv_sum = pv.rolling(window=window, min_periods=1).sum()
            rolling_volume_sum = volume.rolling(window=window, min_periods=1).sum()
            vwap = (rolling_pv_sum / rolling_volume_sum).rename(f'vwap_roll_{window}')
        elif group_by_day: # Daily VWAP
            self.logger.debug("Calculating daily VWAP.")
            # Requires a DatetimeIndex
            if not isinstance(close.index, pd.DatetimeIndex):
                self.logger.error("Daily VWAP requires a DatetimeIndex.")
                return pd.Series(np.nan, index=close.index, name='vwap_daily_error')
            
            # Group by date (day) and calculate cumulative sum for pv and volume within each day
            # then divide. This resets VWAP calculation at the start of each day.
            df_temp = pd.DataFrame({'pv': pv, COL_VOLUME: volume, 'trading_date': close.index.date})
            df_temp['daily_cum_pv'] = df_temp.groupby('trading_date')['pv'].cumsum()
            df_temp['daily_cum_volume'] = df_temp.groupby('trading_date')[COL_VOLUME].cumsum()
            vwap = (df_temp['daily_cum_pv'] / df_temp['daily_cum_volume']).rename('vwap_daily')
            vwap.index = close.index # Ensure original index is restored
        else:
            self.logger.warning("VWAP calculation parameters unclear (no window, not daily). Defaulting to cumulative VWAP.")
            # Cumulative VWAP over the whole series if no window and not daily
            vwap = (pv.cumsum() / volume.cumsum()).rename('vwap_cumulative')
        
        # Handle potential division by zero if volume is zero for a period
        vwap.replace([np.inf, -np.inf], np.nan, inplace=True) 
        # Forward fill NaNs that might result from zero volume, common at start of rolling/daily VWAP
        vwap.ffill(inplace=True) 
        return vwap

    def _add_time_features(self, df: pd.DataFrame, time_feature_config: list, sin_cos_config: list) -> pd.DataFrame:
        """Adds time-based features and optionally sin/cos encodes them."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex. Cannot add time features.")
            return df
        
        if 'hour_of_day' in time_feature_config:
            df['hour_of_day'] = df.index.hour
            self.logger.debug("Added 'hour_of_day' feature.")
        if 'minute_of_hour' in time_feature_config:
            df['minute_of_hour'] = df.index.minute
            self.logger.debug("Added 'minute_of_hour' feature.")
        if 'day_of_week' in time_feature_config:
            df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
            self.logger.debug("Added 'day_of_week' feature.")
        if 'month_of_year' in time_feature_config: # Added for completeness
            df['month_of_year'] = df.index.month
            self.logger.debug("Added 'month_of_year' feature.")

        # Sin/Cos encoding for cyclical features
        if 'hour_of_day' in sin_cos_config and 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
            df.drop(columns=['hour_of_day'], inplace=True) # Drop original if encoded
            self.logger.debug("Applied Sin/Cos encoding to 'hour_of_day'.")
        if 'minute_of_hour' in sin_cos_config and 'minute_of_hour' in df.columns:
            df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60.0)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60.0)
            df.drop(columns=['minute_of_hour'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'minute_of_hour'.")
        if 'day_of_week' in sin_cos_config and 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
            df.drop(columns=['day_of_week'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'day_of_week'.")
        if 'month_of_year' in sin_cos_config and 'month_of_year' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12.0)
            df.drop(columns=['month_of_year'], inplace=True)
            self.logger.debug("Applied Sin/Cos encoding to 'month_of_year'.")
            
        return df

    def compute_features(self, raw_data_df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Computes all configured features on the raw data.

        Args:
            raw_data_df (pd.DataFrame): DataFrame with COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME columns
                                        and a DatetimeIndex.

        Returns:
            pd.DataFrame or None: DataFrame with original data and added features, or None if error.
        """
        if not isinstance(raw_data_df.index, pd.DatetimeIndex):
            self.logger.error("Raw data DataFrame must have a DatetimeIndex.")
            return None
            
        df = raw_data_df.copy()
        
        # Ensure required columns exist
        required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Raw data missing one or more required columns: {required_cols}")
            return None

        # --- RSI ---
        if 'RSI' in self.feature_list and 'rsi' in self.feature_config:
            rsi_params = self.feature_config['rsi']
            df[f'rsi_{rsi_params.get("window", 14)}'] = self._calculate_rsi(df[COL_CLOSE], rsi_params.get("window", 14))

        # --- EMAs ---
        if 'EMA' in self.feature_list and 'ema' in self.feature_config:
            ema_params = self.feature_config['ema']
            for window in ema_params.get('windows', []):
                df[f'ema_{window}'] = self._calculate_ema(df[COL_CLOSE], window)
            # Example: EMA difference (MACD-like)
            if 'ema_diff' in ema_params and len(ema_params.get('windows', [])) == 2:
                fast_ema_col = f'ema_{ema_params["windows"][0]}'
                slow_ema_col = f'ema_{ema_params["windows"][1]}'
                if fast_ema_col in df and slow_ema_col in df:
                    df['ema_diff'] = df[fast_ema_col] - df[slow_ema_col]
                    self.logger.info(f"Added 'ema_diff' feature ({fast_ema_col} - {slow_ema_col}).")


        # --- VWAP & VWAP Deviation ---
        if 'VWAP' in self.feature_list and 'vwap' in self.feature_config:
            vwap_params = self.feature_config['vwap']
            # For simplicity, using a daily VWAP placeholder. A rolling window VWAP might also be useful.
            df['vwap'] = self._calculate_vwap(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], df[COL_VOLUME], window=vwap_params.get("window"))
            df['vwap_deviation'] = (df[COL_CLOSE] - df['vwap']) / df['vwap'] # Percentage deviation
            self.logger.info("Added 'vwap' and 'vwap_deviation' features.")

        # --- Time Features ---
        if 'Time' in self.feature_list and 'time_features' in self.feature_config:
            time_features = self.feature_config.get('time_features', [])
            sin_cos_encode = self.feature_config.get('sin_cos_encode', [])
            df = self._add_time_features(df, time_features, sin_cos_encode)

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
        
        feature_cols_to_scale = self.feature_config.get('feature_cols_to_scale', [])
        if not feature_cols_to_scale: # If not specified, try to scale all non-OHLCV and non-datetime columns
            excluded_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, 'date', 'datetime'] # Common exclusions
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
        
        # Combine back with non-scaled columns (like COL_CLOSE, COL_VOLUME if needed by env)
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
                - price_data_for_env: Series of CLOSE prices aligned with the `final_features_df` index,
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
        
        # Extract price data (e.g., CLOSE prices) for the environment
        # This should be done BEFORE sequencing, using the same index as final_features_df
        # The environment will need this to calculate P&L.
        # Make sure CLOSE prices are from the original data, not scaled.
        if COL_CLOSE not in final_features_df.columns:
            self.logger.error("CLOSE column not found in features DataFrame. Cannot provide price data for env.")
            # Potentially, CLOSE might have been in raw_data_df but not explicitly carried over or was scaled.
            # Ensure CLOSE (unscaled) is available.
            # If CLOSE was scaled, we should use the original CLOSE from features_df (before normalization)
            # or raw_data_df, aligned with final_features_df.index.
            if COL_CLOSE in features_df: # from before normalization
                 price_data_for_env = features_df.loc[final_features_df.index, COL_CLOSE].copy()
            elif COL_CLOSE in raw_data_df: # from raw_data, aligned
                 price_data_for_env = raw_data_df.loc[final_features_df.index, COL_CLOSE].copy()
            else:
                 price_data_for_env = None # Should not happen if data pipeline is correct
        else: # CLOSE is in final_features_df (might be scaled if not excluded)
            # It is safer to take CLOSE from features_df (before scaling) or raw_data_df
            if COL_CLOSE in features_df:
                 price_data_for_env = features_df.loc[final_features_df.index, COL_CLOSE].copy()
            else: # Fallback, though CLOSE ideally shouldn't be scaled if used for P&L
                 price_data_for_env = final_features_df[COL_CLOSE].copy()


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
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation', 'hour_of_day', 'day_of_week', COL_CLOSE, COL_VOLUME] # Example
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
        COL_OPEN: np.random.rand(num_rows) * 10 + 100,
        COL_HIGH: np.random.rand(num_rows) * 5 + 105, # ensure high > open
        COL_LOW: np.random.rand(num_rows) * 5 + 95,   # ensure low < open
        COL_CLOSE: np.random.rand(num_rows) * 10 + 100,
        COL_VOLUME: np.random.randint(100, 1000, size=num_rows) * 10
    }
    raw_df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='date'))
    raw_df[COL_HIGH] = raw_df[[COL_OPEN, COL_CLOSE]].max(axis=1) + np.random.rand(num_rows) * 2
    raw_df[COL_LOW] = raw_df[[COL_OPEN, COL_CLOSE]].min(axis=1) - np.random.rand(num_rows) * 2

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


    # --- Methods for Live Trading Feature Calculation ---

    def _determine_max_indicator_lookback(self) -> int:
        """
        Determines the maximum window size required by any configured indicator.
        This helps define the minimum size for the live_data_buffer.
        """
        max_lookback = 0
        if 'RSI' in self.feature_list and 'rsi' in self.feature_config:
            max_lookback = max(max_lookback, self.feature_config['rsi'].get("window", 14))

        if 'EMA' in self.feature_list and 'ema' in self.feature_config:
            for window in self.feature_config['ema'].get('windows', []):
                max_lookback = max(max_lookback, window)

        if 'VWAP' in self.feature_list and 'vwap' in self.feature_config:
            # Daily VWAP doesn't have a fixed lookback in the same sense, but rolling VWAP does.
            # If using rolling VWAP, its window should be considered.
            # For daily VWAP, we need enough data to cover the start of the day if the session starts mid-day.
            # This is more about ensuring the buffer can hold a full day's data if needed, or relying on warmup.
            # For simplicity, if a rolling VWAP window is given, use that. Otherwise, a default like 50-100 might be needed
            # if no other indicator has a larger window, just to give VWAP some data.
            vwap_window = self.feature_config['vwap'].get("window")
            if vwap_window: # Rolling VWAP
                max_lookback = max(max_lookback, vwap_window)
            # else for Daily VWAP, it's calculated per day, doesn't add to this numerical max_lookback for buffer length.
            # However, the buffer should ideally be long enough to establish VWAP.

        # Add other features like Bollinger Bands, MACD, etc. if they are implemented.
        # For example:
        # if 'MACD' in self.feature_list and 'macd' in self.feature_config:
        #     fast = self.feature_config['macd'].get("fast_period", 12)
        #     slow = self.feature_config['macd'].get("slow_period", 26)
        #     max_lookback = max(max_lookback, slow) # MACD depends on the slower EMA

        if max_lookback == 0: # Default if no windowed indicators are configured
            max_lookback = self.lookback_window # At least the model's observation lookback
            self.logger.warning(f"No specific indicator windows found; setting max_indicator_lookback to model's lookback_window: {max_lookback}")

        # Ensure it's at least as large as the model's observation sequence length
        self._max_indicator_lookback = max(max_lookback, self.lookback_window)
        self.logger.info(f"Determined maximum indicator lookback period: {self._max_indicator_lookback} bars.")
        return self._max_indicator_lookback

    def _load_scaler(self, symbol: str) -> bool:
        """Loads the scaler for the given symbol. Returns True if successful."""
        scaler_path = os.path.join(self.scalers_dir, f"{symbol}_feature_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Live Trading: Loaded StandardScaler for {symbol} from {scaler_path}")
                return True
            except Exception as e:
                self.logger.error(f"Live Trading: Error loading scaler for {symbol} from {scaler_path}: {e}", exc_info=True)
                self.scaler = None
                return False
        else:
            self.logger.error(f"Live Trading: Scaler file not found for {symbol} at {scaler_path}. Cannot proceed with live feature calculation.")
            self.scaler = None
            return False

    def initialize_live_session(self, symbol: str, historical_data_for_warmup: pd.DataFrame = None):
        """
        Prepares the FeatureAgent for a live trading session.
        - Loads the appropriate scaler.
        - Initializes the live_data_buffer, optionally with historical data for warming up indicators.
        - Determines the maximum lookback needed by indicators.
        """
        self.logger.info(f"Initializing live trading session for symbol: {symbol}")
        self._is_live_session_initialized = False # Reset

        if not self._load_scaler(symbol):
            return # Error already logged by _load_scaler

        self._determine_max_indicator_lookback()

        self.live_data_buffer = pd.DataFrame()
        self.normalized_feature_history_buffer = [] # Use a deque for efficiency if lookback_window is large
        # from collections import deque
        # self.normalized_feature_history_buffer = deque(maxlen=self.lookback_window)


        if historical_data_for_warmup is not None and not historical_data_for_warmup.empty:
            self.logger.info(f"Warming up live_data_buffer with {len(historical_data_for_warmup)} historical bars.")
            # Ensure warmup data has correct columns and index, similar to a live bar
            # For simplicity, assuming it's already correctly formatted.
            self.live_data_buffer = historical_data_for_warmup.copy()
            # Trim buffer if warmup data is longer than needed (max_indicator_lookback + lookback_window for safety margin)
            # The buffer needs enough history for indicators AND then enough for the model's sequence.
            # Total buffer size should be roughly max_indicator_lookback + self.lookback_window
            # Let's use _max_indicator_lookback as it already considers self.lookback_window.
            # No, _max_indicator_lookback is for *indicator calculation*.
            # The actual buffer for raw data should be at least _max_indicator_lookback.
            # Then, from the computed features, we take `self.lookback_window` for the model.
            buffer_needed_for_indicators = self._max_indicator_lookback

            if len(self.live_data_buffer) > buffer_needed_for_indicators:
                 self.live_data_buffer = self.live_data_buffer.iloc[-buffer_needed_for_indicators:]
            self.logger.info(f"live_data_buffer initialized with {len(self.live_data_buffer)} bars after warmup and trim.")

            # Optional: Pre-compute features and fill normalized_feature_history_buffer if possible
            if len(self.live_data_buffer) >= buffer_needed_for_indicators: # Ensure enough data for stable features
                self.logger.info("Pre-computing features and populating normalized_feature_history_buffer from warmup data...")
                warmup_features_df = self.compute_features(self.live_data_buffer)
                if warmup_features_df is not None and not warmup_features_df.empty:
                    # Normalize all available warmup features
                    warmup_norm_features_df = self.normalize_features(warmup_features_df, fit_scaler=False, symbol=symbol)
                    if warmup_norm_features_df is not None and not warmup_norm_features_df.empty:
                        obs_cols = self.feature_config.get('observation_feature_cols', warmup_norm_features_df.columns.tolist())
                        warmup_obs_features = warmup_norm_features_df[obs_cols]

                        # Populate the history buffer with the most recent `lookback_window` observations
                        num_to_take = min(len(warmup_obs_features), self.lookback_window)
                        for i in range(num_to_take):
                            # Append as list or 1-row DataFrame/Series
                            self.normalized_feature_history_buffer.append(warmup_obs_features.iloc[-(num_to_take - i)].values)
                        self.logger.info(f"Normalized feature history buffer populated with {len(self.normalized_feature_history_buffer)} entries from warmup data.")

        self._is_live_session_initialized = True
        self.logger.info("Live trading session initialized successfully.")


    def process_live_bar(self, new_bar_df: pd.DataFrame, symbol: str) -> tuple[np.ndarray | None, pd.Series | None]:
        """
        Processes a single new live bar to update features and generate the latest observation sequence.

        Args:
            new_bar_df (pd.DataFrame): A single-row DataFrame for the new bar.
                                       Must have a DatetimeIndex and OHLCV columns.
            symbol (str): The symbol of the bar (used for context, scaler already loaded).

        Returns:
            tuple(np.ndarray | None, pd.Series | None):
                - observation_sequence: NumPy array for the model (shape: lookback_window, num_features) or None.
                - latest_price_series: pd.Series containing the latest bar's OHLCV and timestamp, for context.
        """
        if not self._is_live_session_initialized:
            self.logger.error("Live session not initialized. Call initialize_live_session() first.")
            return None, None

        if self.scaler is None:
            self.logger.error(f"Scaler for {symbol} is not loaded. Cannot process live bar.")
            return None, None

        if not isinstance(new_bar_df.index, pd.DatetimeIndex) or len(new_bar_df) != 1:
            self.logger.error("New bar data must be a single-row DataFrame with a DatetimeIndex.")
            return None, new_bar_df.iloc[-1] if not new_bar_df.empty else None

        latest_price_series = new_bar_df.iloc[-1].copy() # For returning context

        # 1. Append to raw data buffer and trim
        # Use pd.concat instead of append if pandas version is newer
        if pd.__version__ >= "1.4.0":
            self.live_data_buffer = pd.concat([self.live_data_buffer, new_bar_df])
        else:
            self.live_data_buffer = self.live_data_buffer.append(new_bar_df)

        # Trim buffer to keep only necessary length for indicator calculation
        # Max length needed is _max_indicator_lookback (which includes model's lookback_window if larger)
        # No, _max_indicator_lookback is for indicator calculation only.
        # The buffer should be at least _max_indicator_lookback long for stable features.
        min_buffer_len_for_indicators = self._max_indicator_lookback
        if len(self.live_data_buffer) > min_buffer_len_for_indicators + 20: # Keep a bit extra margin
            self.live_data_buffer = self.live_data_buffer.iloc[-(min_buffer_len_for_indicators + 20):]

        if len(self.live_data_buffer) < min_buffer_len_for_indicators:
            self.logger.warning(f"Live data buffer size ({len(self.live_data_buffer)}) is less than "
                                f"max indicator lookback ({min_buffer_len_for_indicators}). Features might be unstable or NaN.")
            # Continue processing, but features might be NaN until buffer fills.
            # compute_features should handle this by returning NaNs or dropping rows.

        # 2. Compute features on the current buffer
        # compute_features will drop NaNs, so the output df might be shorter than live_data_buffer
        current_features_df = self.compute_features(self.live_data_buffer)
        if current_features_df is None or current_features_df.empty:
            self.logger.warning(f"Feature computation on live buffer for {symbol} yielded no data. Buffer size: {len(self.live_data_buffer)}")
            return None, latest_price_series

        # 3. Normalize the latest set of features
        # We only need to normalize the features corresponding to the *latest bar* for the model input.
        # However, the scaler expects a 2D array.
        latest_computed_features_row = current_features_df.iloc[-1:] # Keep as DataFrame for column selection

        # Ensure columns for scaling are present (as done in normalize_features)
        feature_cols_to_scale = self.feature_config.get('feature_cols_to_scale', [])
        if not feature_cols_to_scale: # Auto-detect if not specified
            excluded_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, 'date', 'datetime']
            feature_cols_to_scale = [col for col in latest_computed_features_row.columns if col not in excluded_cols and latest_computed_features_row[col].dtype in [np.float64, np.int64]]

        if not all(col in latest_computed_features_row.columns for col in feature_cols_to_scale):
            self.logger.error(f"One or more columns to scale are missing from latest computed features for {symbol}.")
            return None, latest_price_series

        data_to_scale = latest_computed_features_row[feature_cols_to_scale]

        try:
            normalized_data_values = self.scaler.transform(data_to_scale) # Input must be 2D
        except Exception as e:
            self.logger.error(f"Error transforming latest features for {symbol}: {e}", exc_info=True)
            return None, latest_price_series

        # Create a Series or 1-row DF for the latest normalized features
        latest_normalized_features = pd.Series(normalized_data_values[0], index=feature_cols_to_scale)

        # Combine with any non-scaled observation features from latest_computed_features_row
        # The final observation for the model uses columns from 'observation_feature_cols'
        obs_feature_cols = self.feature_config.get('observation_feature_cols', [])
        if not obs_feature_cols: # Default to all columns in latest_computed_features_row if not specified
            obs_feature_cols = latest_computed_features_row.columns.tolist()

        final_observation_values = []
        for col in obs_feature_cols:
            if col in latest_normalized_features.index: # It was a scaled column
                final_observation_values.append(latest_normalized_features[col])
            elif col in latest_computed_features_row.columns: # It was an unscaled column
                final_observation_values.append(latest_computed_features_row[col].iloc[0])
            else:
                self.logger.error(f"Observation column '{col}' not found in latest computed or normalized features for {symbol}. Filling with NaN.")
                final_observation_values.append(np.nan)

        latest_observation_row_np = np.array(final_observation_values)

        # 4. Update normalized feature history buffer and create sequence
        if len(self.normalized_feature_history_buffer) == self.lookback_window and self.lookback_window > 0:
            self.normalized_feature_history_buffer.pop(0) # Remove oldest if using list
        self.normalized_feature_history_buffer.append(latest_observation_row_np)
        # If using deque, append will automatically handle maxlen.

        if len(self.normalized_feature_history_buffer) < self.lookback_window:
            self.logger.debug(f"Normalized feature history buffer for {symbol} has {len(self.normalized_feature_history_buffer)} entries, "
                              f"needs {self.lookback_window} for a full sequence. Returning None for observation.")
            return None, latest_price_series # Not enough history to form a full sequence

        observation_sequence_np = np.array(self.normalized_feature_history_buffer) # Shape: (lookback_window, num_features)

        # Sanity check for NaNs in the final observation sequence
        if np.isnan(observation_sequence_np).any():
            self.logger.warning(f"NaNs detected in the final observation sequence for {symbol}. This might be due to indicator warmup or data issues.")
            # Potentially return None or let the model handle NaNs if designed to. For now, return as is.

        return observation_sequence_np, latest_price_series

    # --- End Live Trading Methods ---
