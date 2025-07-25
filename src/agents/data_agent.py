# src/agents/data_agent.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from ib_insync import IB, util, Contract, Stock # Assuming ib_insync is installed

from .base_agent import BaseAgent
# from ..utils.config_loader import load_config # Example for loading config
# Removed: from src.tools.ibkr_tools import fetch_5min_bars  # Using direct IBKR call instead
from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME
from src.shared import get_feature_store
from src.shared.market_impact import calc_market_impact_features

class DataAgent(BaseAgent):
    """
    DataAgent is responsible for:
    1. Fetching historical and live market data from Interactive Brokers.
    2. Caching the data to local storage (e.g., CSV or Parquet/Pickle).
    3. Performing basic data validation and quality checks.
    """
    def __init__(self, config: dict, ib_client: IB = None):
        """
        Initializes the DataAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'data_dir_raw': Path to save raw data.
                           'ibkr_conn': IBKR connection details (host, port, clientId, timeout_seconds).
            ib_client (ib_insync.IB, optional): An active IB Insync IB client instance.
                                                 Allows for sharing an existing connection.
        """
        super().__init__(agent_name="DataAgent", config=config)
        self.data_dir_raw = self.config.get('data_dir_raw', 'data/raw')
        os.makedirs(self.data_dir_raw, exist_ok=True)
        
        self.ib_config = self.config.get('ibkr_conn', {}) or self.config.get('ibkr_connection', {})
        self.ib = None # Will be initialized
        self.ib_connected = False
        self._ib_managed_locally = False # Flag to track if this instance manages the connection
        self.live_bar_tickers = {} # Stores symbol: RealTimeBarList object
        self.bar_update_callback = None # To be set by Orchestrator


        if ib_client and ib_client.isConnected():
            self.ib = ib_client
            self.ib_connected = True
            self.logger.info("Using provided, already connected IB client.")
        elif self.ib_config:
            self._connect_ibkr() # Attempt to connect if config is provided
        else:
            self.logger.warning("No IB client provided and no ibkr_conn config found. IBKR functionalities will be unavailable.")

        self.logger.info(f"Raw data will be saved to: {self.data_dir_raw}")

    def _connect_ibkr(self):
        """Initializes and connects to Interactive Brokers if not already connected."""
        if self.ib_connected:
            self.logger.info("IB client is already connected.")
            return

        # Safety check for missing config
        if not self.ib_config:
            self.logger.error("IBKR configuration is missing. Cannot connect to Interactive Brokers.")
            return

        # Check for simulation mode
        if self.ib_config.get('simulation_mode', False):
            self.logger.warning("IBKR simulation mode enabled. Using cached data only.")
            self.ib_connected = False  # Keep as False to use cached data
            return

        if not self.ib_config.get('host') or not self.ib_config.get('port') or self.ib_config.get('clientId') is None:
            self.logger.error("IBKR connection parameters (host, port, clientId) missing in config.")
            return

        try:
            from ib_insync import util
            import asyncio
            
            # Patch asyncio to work in threading environment
            util.patchAsyncio()
            
            self.ib = IB()
            self.logger.info(f"Attempting to connect to IBKR at {self.ib_config['host']}:{self.ib_config['port']} with clientId {self.ib_config['clientId']}...")
            
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self.ib.connect(
                host=self.ib_config['host'],
                port=self.ib_config['port'],
                clientId=self.ib_config['clientId'],
                timeout=self.ib_config.get('timeout_seconds', 10), # Default timeout 10s
                readonly=self.ib_config.get('readonly', False) # Default to read-write
            )
            self.ib_connected = True
            self._ib_managed_locally = True # This instance established the connection
            self.logger.info("Successfully connected to IBKR.")
        except ConnectionRefusedError:
            self.logger.error(f"IBKR connection refused. Please check:")
            self.logger.error(f"1. TWS/Gateway is running on {self.ib_config['host']}:{self.ib_config['port']}")
            self.logger.error(f"2. API connections are enabled in TWS (File -> Global Configuration -> API -> Settings)")
            self.logger.error(f"3. 'Enable ActiveX and Socket Clients' is checked")
            self.logger.error(f"4. Client ID {self.ib_config['clientId']} is not already in use")
            self.ib_connected = False
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}", exc_info=True)
            self.ib_connected = False

    def set_bar_update_callback(self, callback):
        """Set the callback function to be called when new bars are received."""
        self.bar_update_callback = callback
        self.logger.info("Bar update callback set")


    def disconnect_ibkr(self):
        """Disconnects from IBKR if the connection was managed by this instance."""
        if self.ib and self.ib_connected and self._ib_managed_locally:
            self.logger.info("Disconnecting from IBKR...")
            try:
                self.ib.disconnect()
                self.ib_connected = False
                self.logger.info("Successfully disconnected from IBKR.")
            except Exception as e:
                self.logger.error(f"Error during IBKR disconnection: {e}", exc_info=True)
        elif self.ib and self.ib_connected and not self._ib_managed_locally:
            self.logger.info("IBKR connection was provided externally; not disconnecting from this instance.")
        else:
            self.logger.info("IBKR not connected or connection not managed locally.")
            
    def _determine_actual_start_end_for_cache(self, end_datetime_str: str, duration_str: str) -> tuple[str, str]:
        """
        Determines the actual start and end dates for consistent cache naming.
        IBKR's durationStr can be tricky. This function aims to provide a best-effort
        standardized date range for the cache filename.
        """
        try:
            # Parse end_datetime_str (YYYYMMDD HH:MM:SS [TZ])
            # For simplicity, we'll use the date part for filename consistency.
            # A more robust solution might handle timezones if provided.
            end_dt_obj = datetime.strptime(end_datetime_str.split(" ")[0], "%Y-%m-%d")
            
            # Parse duration_str (e.g., "1 Y", "6 M", "30 D", "1000 S")
            # This is a simplified parser. ib_insync handles this internally for the request.
            # We only need to approximate for the cache filename.
            num_str, unit = duration_str.split(" ")
            num = int(num_str)
            
            if unit == 'S': start_dt_obj = end_dt_obj - timedelta(seconds=num) # Approx for filename
            elif unit == 'D': start_dt_obj = end_dt_obj - timedelta(days=num)
            elif unit == 'W': start_dt_obj = end_dt_obj - timedelta(weeks=num)
            elif unit == 'M': start_dt_obj = end_dt_obj - timedelta(days=num * 30) # Approx month
            elif unit == 'Y': start_dt_obj = end_dt_obj - timedelta(days=num * 365) # Approx year
            else:
                self.logger.warning(f"Unknown duration unit '{unit}'. Using end date for start date in cache name.")
                start_dt_obj = end_dt_obj

            return start_dt_obj.strftime("%Y%m%d"), end_dt_obj.strftime("%Y%m%d")

        except Exception as e:
            self.logger.error(f"Error parsing dates for cache filename (end: '{end_datetime_str}', dur: '{duration_str}'): {e}. Using fallback names.")
            return "UNKNOWN_START", end_datetime_str.split(" ")[0] if end_datetime_str else "UNKNOWN_END"


    def _convert_interval_for_ibkr(self, interval: str) -> str:
        """
        Convert interval format from web form to IBKR format.
        
        Args:
            interval: Interval string (e.g., "1min", "5mins", "1hour", "1day")
            
        Returns:
            IBKR-compatible interval string (e.g., "1 min", "5 mins", "1 hour", "1 day")
        """
        # Common interval mappings
        interval_map = {
            "1min": "1 min",
            "5mins": "5 mins", 
            "15mins": "15 mins",
            "30mins": "30 mins",
            "1hour": "1 hour",
            "1day": "1 day"
        }
        
        # Return mapped value or original if not found
        return interval_map.get(interval, interval)
    
    def _get_cache_filepath(self, symbol: str, start_date_str_yyyymmdd: str, end_date_str_yyyymmdd: str, bar_size_str: str, data_format: str = "csv") -> str:
        """
        Helper to generate a consistent filepath for cached data using standardized date formats.
        Args:
            start_date_str_yyyymmdd (str): Start date in YYYYMMDD format.
            end_date_str_yyyymmdd (str): End date in YYYYMMDD format.
            bar_size_str (str): Bar size string, spaces removed (e.g., "1min", "5mins").
        """
        # Sanitize bar_size_str for filename
        interval_sanitized = bar_size_str.replace(" ", "").replace(":", "")
        symbol_clean = os.path.basename(symbol.replace('/', '').replace('\\', ''))
        filename = f"{symbol_clean}_{start_date_str_yyyymmdd}_{end_date_str_yyyymmdd}_{interval_sanitized}.{data_format}"
        full_path = os.path.join(self.data_dir_raw, symbol_clean.upper(), filename)
        full_path = os.path.abspath(full_path) 
        return full_path

    def fetch_ibkr_bars(self, symbol: str, end_datetime_str: str, duration_str: str, bar_size_str: str,
                        sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD",
                        what_to_show: str = "TRADES", use_rth: bool = True,
                        data_format: str = "csv", force_fetch: bool = False,
                        primary_exchange: str = None) -> pd.DataFrame | None:
        """
        Fetches historical bar data from Interactive Brokers.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").
            end_datetime_str (str): The end date and time for the data request (YYYYMMDD HH:MM:SS [TZ]).
                                    If empty, current time is used.
            duration_str (str): The amount of time to go back from end_datetime_str
                                (e.g., "1 Y", "6 M", "10 D", "1000 S").
            bar_size_str (str): The bar size (e.g., "1 min", "5 mins", "1 day").
            what_to_show (str): Type of data (TRADES, MIDPOINT, BID, ASK). Defaults to "TRADES".
            use_rth (bool): If True, only data from regular trading hours is returned. Defaults to True.
            data_format (str): "csv" or "pkl" for saving.
            force_fetch (bool): If True, fetches data from IBKR even if a cache file exists.

        Returns:
            pd.DataFrame or None: DataFrame with bar data (datetime, open, high, low, close, volume, average, barCount)
                                  or None if fetching fails or data is empty.
        """
        self.logger.info(f"Requesting bars for {symbol}: end={end_datetime_str}, dur={duration_str}, size={bar_size_str}")

        start_date_str_yyyymmdd, end_date_str_yyyymmdd = self._determine_actual_start_end_for_cache(end_datetime_str, duration_str)

        # Clean inputs for cache filepath
        symbol_clean = symbol.replace('/', '').replace('\\', '')
        bar_size_clean = bar_size_str.replace(' ', '').replace(':', '').replace('/', '').replace('\\', '')

        cache_filepath = self._get_cache_filepath(
            symbol_clean,
            start_date_str_yyyymmdd,
            end_date_str_yyyymmdd,
            bar_size_clean,
            data_format
        )
        
        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)

        if not force_fetch: 
            if os.path.exists(cache_filepath):
                self.logger.info(f"Loading cached data from {cache_filepath}")
                try:
                    if data_format == "csv":
                        bars_df = pd.read_csv(cache_filepath, index_col='datetime', parse_dates=True)
                    elif data_format == "pkl":
                        bars_df = pd.read_pickle(cache_filepath)
                    else:
                        self.logger.warning(f"Unsupported data format '{data_format}' for loading.")
                        return None
                    if not bars_df.empty:
                        return bars_df
                    else:
                        self.logger.warning(f"Cached file {cache_filepath} is empty. Fetching new data.")
                except Exception as e:
                    self.logger.error(f"Error loading cached data from {cache_filepath}: {e}. Fetching new data.")

        # TODO: Implement actual IBKR data fetching logic using ib_insync
        # This is a placeholder and will not actually fetch data.
        # Ensure self.ib is connected before trying to use it.
        # if not self.ib_connected:
        #     self.logger.error("IBKR not connected. Cannot fetch bars.")
        #     return None
        #
        # contract = Contract() # This needs to be correctly defined (e.g., Stock(symbol, 'SMART', 'USD'))
        # contract.symbol = symbol
        # contract.secType = "STK" # Example: for stocks
        # contract.exchange = "SMART"
        # contract.currency = "USD"
        #
        # self.logger.info(f"Fetching {duration_str} of {bar_size_str} bars for {symbol} ending {end_datetime_str or 'now'}")
        # try:
        #     bars = self.ib.reqHistoricalData(
        #         contract,
        #         endDateTime=end_datetime_str,
        #         durationStr=duration_str,
        #         barSizeSetting=bar_size_str,
        #         whatToShow=what_to_show,
        #         useRTH=use_rth,
        #         formatDate=1  # UTC timestamp
        #     )
        #     if not bars:
        #         self.logger.warning(f"No data returned from IBKR for {symbol}")
        #         return None
        #
        #     bars_df = util.df(bars) # Convert to pandas DataFrame
        #     if bars_df is None or bars_df.empty:
        #         self.logger.warning(f"DataFrame is empty after fetching from IBKR for {symbol}")
        #         return None
        #
        #     # Ensure 'date' column is datetime and set as index
        #     if 'date' in bars_df.columns:
        #        bars_df['date'] = pd.to_datetime(bars_df['date'])
        #        bars_df.set_index('date', inplace=True)
        #
        #     self.logger.info(f"Successfully fetched {len(bars_df)} bars for {symbol}.")
        #
        #     # Cache the fetched data
        #     try:
        #         if data_format == "csv":
        #             bars_df.to_csv(cache_filepath)
        #         elif data_format == "pkl":
        #             bars_df.to_pickle(cache_filepath)
        #         self.logger.info(f"Cached data to {cache_filepath}")
        #     except Exception as e:
        #         self.logger.error(f"Error caching data to {cache_filepath}: {e}")
        #
        #     return bars_df
        #
        # except Exception as e:
        #     self.logger.error(f"Error fetching historical data for {symbol} from IBKR: {e}")
        #     return None
        #self.logger.warning("Actual IBKR fetching is not implemented in this skeleton.")
        # Create a dummy DataFrame for skeleton purposes

        #dates = pd.to_datetime([
        #    datetime(2023, 1, 1, 9, 30),
        #    datetime(2023, 1, 1, 9, 31),
        #    datetime(2023, 1, 1, 9, 32)
        #])
        #dummy_data = {COL_OPEN: [100, 101, 102], COL_HIGH: [102, 102, 103], COL_LOW: [99, 100, 101],
        #              COL_CLOSE: [101, 102, 103], COL_VOLUME: [1000, 1200, 1100]}
        #bars_df = pd.DataFrame(dummy_data, index=pd.DatetimeIndex(dates, name='date'))
        # Compute start and end dates for fetch_5min_bars

        # Import the direct IBKR function to use the correct interval
        from src.tools.GetIBKRData import get_ibkr_data_sync, ib
        
        # Use the actual requested interval instead of hardcoded 5min
        bars_df, _bt_feed = get_ibkr_data_sync(
            ib_instance=ib,
            ticker_symbol=symbol,
            start_date=start_date_str_yyyymmdd,
            end_date=end_date_str_yyyymmdd,
            bar_size=ibkr_interval,  # Use the converted interval (e.g., "1 min")
            what_to_show='TRADES',
            use_rth=use_rth
        )
        
        # Reset index and rename columns to match expected format
        if 'datetime' in bars_df.columns:
            bars_df = bars_df.set_index('datetime')
        elif bars_df.index.name != 'Date' and 'Date' not in bars_df.columns:
            bars_df.index.name = 'Date'
        # If your system expects the column 'Date' instead of index name:
        bars_df = bars_df.reset_index().rename(columns={'datetime': 'Date'}).set_index('Date')

        # Save dummy data to cache path for testing workflow
        try:
            if data_format == "csv":
                bars_df.to_csv(cache_filepath)
            elif data_format == "pkl":
                bars_df.to_pickle(cache_filepath)
            self.logger.info(f"Saved data to {cache_filepath}")
        except Exception as e:
            self.logger.error(f"Error caching data to {cache_filepath}: {e}")
        return bars_df


    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase for consistent processing.
        
        Args:
            df (pd.DataFrame): DataFrame with potentially mixed-case column names
            
        Returns:
            pd.DataFrame: DataFrame with standardized lowercase column names
        """
        column_mapping = {}
        for col in df.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col.lower()
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.debug(f"Standardized column names: {column_mapping}")
        
        return df

    def validate_data(self, df: pd.DataFrame, symbol: str) -> tuple[bool, pd.DataFrame]:
        """
        Performs basic validation on the fetched data and standardizes column names.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            symbol (str): Symbol for logging purposes.

        Returns:
            tuple[bool, pd.DataFrame]: (validation_success, standardized_dataframe)
        """
        if df is None or df.empty:
            self.logger.warning(f"Data validation failed for {symbol}: DataFrame is empty.")
            return False, df

        # Standardize column names before validation
        df = self.standardize_column_names(df)

        # Check for required columns (adjust based on actual IBKR output)
        # Typical columns: open, high, low, close, volume
        required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Data validation failed for {symbol}: Missing one or more required columns {required_cols}.")
            return False, df

        # Check for NaNs
        if df.isnull().values.any():
            self.logger.warning(f"Data validation failed for {symbol}: DataFrame contains NaN values.")
            # TODO: Add more sophisticated NaN handling (e.g., fill or drop based on strategy)
            return False, df

        # Check if index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(f"Data validation failed for {symbol}: Index is not a DatetimeIndex.")
            return False, df
        
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            self.logger.warning(f"Data validation failed for {symbol}: Data is not sorted chronologically. Sorting...")
            df.sort_index(inplace=True)

        # TODO: Add more checks:
        # - Outlier detection (e.g., unusually large price changes or volumes)
        # - Consistent time intervals between bars (for unformly sampled data)
        # - Volume checks (e.g., non-negative volume)

        self.logger.info(f"Data validation passed for {symbol} with {len(df)} rows.")
        return True, df

    def run(self, symbol: str, start_date: str, end_date: str, interval: str = "1min", **kwargs) -> pd.DataFrame | None:
        """
        Main method to fetch and cache data for a given symbol and date range.
        This is a simplified run method for demonstration. A more robust one would parse
        start_date and end_date to determine duration for IBKR.

        Args:
            symbol (str): Stock symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format. (Used for cache naming primarily)
            end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format for IBKR.
            interval (str): Bar interval (e.g., "1 min", "5 mins").
            **kwargs: Additional arguments for fetch_ibkr_bars.

        Returns:
            pd.DataFrame or None: The fetched and validated data, or None if an error occurs.
        """
        self.logger.info(f"DataAgent run triggered for {symbol} from {start_date} to {end_date} at {interval} interval.")

        # TODO: For a real implementation, you'd need to calculate `duration_str` for IBKR
        # based on start_date and end_date.
        # Example:
        # from_dt = datetime.strptime(start_date, "%Y-%m-%d")
        # to_dt = datetime.strptime(end_date.split(' ')[0], "%Y-%m-%d") # Assuming end_date includes time
        # delta = to_dt - from_dt
        # duration_str = f"{delta.days + 1} D" # Simplified, IBKR has max durations for certain bar sizes.
        # This part requires careful handling of IBKR API limitations.
        # For this skeleton, we'll assume `duration_str` is passed via kwargs or a default.
        
        duration_str = kwargs.get('duration_str', "30 D") # Default to 30 days for example
        data_format = kwargs.get('data_format', "csv")
        force_fetch = kwargs.get('force_fetch', False)

        # The cache_filepath in fetch_ibkr_bars needs to be robust.
        # The current _get_cache_filepath is simplified. It should use start_date and end_date
        # that accurately reflect the data content.
        # For now, fetch_ibkr_bars's cache naming is based on end_datetime_str and duration_str.
        # We pass start_date mainly for conceptual alignment with the run method's args.
        
        # Convert interval format for IBKR (e.g., "1min" -> "1 min")
        ibkr_interval = self._convert_interval_for_ibkr(interval)
        self.logger.info(f"Converted interval '{interval}' to IBKR format '{ibkr_interval}'")
        
        # Use direct IBKR call with correct interval instead of fetch_ibkr_bars
        from src.tools.GetIBKRData import get_ibkr_data_sync, ib
        
        try:
            bars_df, _bt_feed = get_ibkr_data_sync(
                ib_instance=ib,
                ticker_symbol=symbol,
                start_date=start_date,  # Use start_date directly
                end_date=end_date.split(' ')[0],  # Remove time part if present
                bar_size=ibkr_interval,  # Use the converted interval (e.g., "1 min")
                what_to_show=kwargs.get('what_to_show', 'TRADES'),
                use_rth=kwargs.get('use_rth', True)
            )
            
            # Reset index and rename columns to match expected format
            if 'datetime' in bars_df.columns:
                bars_df = bars_df.set_index('datetime')
            elif bars_df.index.name != 'Date' and 'Date' not in bars_df.columns:
                bars_df.index.name = 'Date'
                
        except Exception as e:
            self.logger.error(f"Failed to fetch data using direct IBKR call: {e}")
            return None
            
        # Save to cache with correct interval name
        if bars_df is not None and not bars_df.empty:
            # Create cache filepath with actual interval (not requested interval)
            interval_clean = ibkr_interval.replace(' ', '').replace(':', '')
            cache_filepath = self._get_cache_filepath(
                symbol,
                start_date.replace('-', ''),  # Convert to YYYYMMDD
                end_date.split(' ')[0].replace('-', ''),  # Convert to YYYYMMDD
                interval_clean,
                data_format
            )
            
            os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
            
            try:
                if data_format == "csv":
                    bars_df.to_csv(cache_filepath)
                elif data_format == "pkl":
                    bars_df.to_pickle(cache_filepath)
                self.logger.info(f"Saved data to {cache_filepath}")
            except Exception as e:
                self.logger.error(f"Error caching data to {cache_filepath}: {e}")

        if bars_df is not None:
            is_valid, standardized_df = self.validate_data(bars_df, symbol)
            if is_valid:
                self.logger.info(f"Successfully fetched and validated data for {symbol}.")
                # The path where data was actually saved is determined within fetch_ibkr_bars
                # based on its parameters. If you need the exact path here, fetch_ibkr_bars
                # should return it or it should be reconstructed identically.
                # For now, we just return the standardized DataFrame.
                return standardized_df
            else:
                self.logger.error(f"Data validation failed for {symbol}. No data returned.")
                return None
        else:
            self.logger.error(f"Failed to fetch data for {symbol}.")
            return None

    def warm_feature_cache_for_symbol(self, symbol: str, feature_agent, **fetch_kwargs):
        """
        Warm feature cache by fetching data and pre-computing features.
        This is useful for offline preloading before training runs.
        
        Args:
            symbol: Stock symbol to warm cache for
            feature_agent: FeatureAgent instance to use for feature computation
            **fetch_kwargs: Additional arguments for data fetching
        """
        self.logger.info(f"Warming feature cache for {symbol}")
        
        try:
            # Use default parameters if not provided
            end_date = fetch_kwargs.get('end_date', datetime.now().strftime("%Y%m%d %H:%M:%S"))
            duration_str = fetch_kwargs.get('duration_str', "30 D")
            interval = fetch_kwargs.get('interval', "5 mins")
            
            # Fetch raw data
            raw_data = self.fetch_ibkr_bars(
                symbol=symbol,
                end_datetime_str=end_date,
                duration_str=duration_str,
                bar_size_str=interval,
                **fetch_kwargs
            )
            
            if raw_data is not None and not raw_data.empty:
                # Warm the feature cache
                feature_agent.warm_feature_cache(raw_data, symbol)
                self.logger.info(f"Successfully warmed feature cache for {symbol}")
            else:
                self.logger.warning(f"No data available to warm cache for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error warming feature cache for {symbol}: {e}")

    def enhance_data_with_order_book(self, df: pd.DataFrame, symbol: str = "default") -> pd.DataFrame:
        """
        Enhance OHLCV data with simulated order book features for training.
        
        This method simulates order book data based on OHLC prices and adds
        market impact features that can be used for training observations.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            DataFrame with added order book columns and market impact features
        """
        if df.empty:
            return df
            
        try:
            self.logger.debug(f"Enhancing {len(df)} rows with order book data for {symbol}")
            
            # Create a copy to avoid modifying original data
            enhanced_df = df.copy()
            
            # Simulate order book levels based on OHLC data
            enhanced_df = self._simulate_order_book_levels(enhanced_df)
            
            # Calculate market impact features
            enhanced_df = self._add_market_impact_features(enhanced_df)
            
            self.logger.debug(f"Successfully enhanced data with order book features for {symbol}")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Error enhancing data with order book features: {e}")
            return df  # Return original data if enhancement fails
    
    def _simulate_order_book_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate order book levels based on OHLC data.
        
        This creates realistic bid/ask spreads and sizes based on:
        - High/Low range for spread estimation
        - Volume for size estimation
        - Random variations for realism
        """
        try:
            # Calculate mid price from OHLC
            df['mid'] = (df[COL_HIGH] + df[COL_LOW]) / 2
            
            # Estimate spread based on high-low range (as proxy for volatility)
            hl_range = df[COL_HIGH] - df[COL_LOW]
            base_spread_pct = np.clip(hl_range / df['mid'], 0.0001, 0.01)  # 0.01% to 1%
            
            # Add some randomness to spread
            np.random.seed(42)  # For reproducible results
            spread_multiplier = np.random.uniform(0.5, 2.0, len(df))
            spread_pct = base_spread_pct * spread_multiplier
            
            # Calculate bid/ask prices (Level 1)
            half_spread = df['mid'] * spread_pct / 2
            df['bid_px1'] = df['mid'] - half_spread
            df['ask_px1'] = df['mid'] + half_spread
            
            # Simulate sizes based on volume
            # Assume level 1 has 10-50% of total volume
            volume_fraction = np.random.uniform(0.1, 0.5, len(df))
            base_size = df[COL_VOLUME] * volume_fraction
            
            # Add imbalance (bid vs ask sizes)
            imbalance = np.random.uniform(-0.3, 0.3, len(df))  # -30% to +30% imbalance
            df['bid_sz1'] = base_size * (1 + imbalance)
            df['ask_sz1'] = base_size * (1 - imbalance)
            
            # Ensure positive sizes
            df['bid_sz1'] = np.maximum(df['bid_sz1'], 100)
            df['ask_sz1'] = np.maximum(df['ask_sz1'], 100)
            
            # Add deeper levels (2-5) with decreasing sizes and wider spreads
            for level in range(2, 6):
                level_spread_mult = 1 + (level - 1) * 0.5  # Increasing spread
                level_size_mult = 0.7 ** (level - 1)  # Decreasing size
                
                df[f'bid_px{level}'] = df['bid_px1'] - half_spread * (level_spread_mult - 1)
                df[f'ask_px{level}'] = df['ask_px1'] + half_spread * (level_spread_mult - 1)
                df[f'bid_sz{level}'] = df['bid_sz1'] * level_size_mult
                df[f'ask_sz{level}'] = df['ask_sz1'] * level_size_mult
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error simulating order book levels: {e}")
            return df
    
    def _add_market_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market impact features to the DataFrame.
        
        Uses the market impact calculator to add microstructure features.
        """
        try:
            # Initialize feature columns
            impact_features = ['spread_bps', 'queue_imbalance', 'impact_10k', 'kyle_lambda']
            for feature in impact_features:
                df[feature] = np.nan
            
            # Calculate features for each row
            for i in range(len(df)):
                try:
                    row = df.iloc[i]
                    mid = row['mid']
                    
                    # Get previous mid for Kyle's lambda
                    last_mid = df.iloc[i-1]['mid'] if i > 0 else None
                    
                    # Calculate market impact features
                    features = calc_market_impact_features(
                        book=row,
                        mid=mid,
                        last_mid=last_mid,
                        signed_vol=None,  # Would need trade data for this
                        notional=10_000
                    )
                    
                    # Assign features to DataFrame
                    for feature_name, value in features.items():
                        if feature_name in impact_features:
                            df.loc[df.index[i], feature_name] = value
                            
                except Exception as e:
                    self.logger.warning(f"Error calculating impact features for row {i}: {e}")
                    continue
            
            # Fill NaN values with defaults
            df['spread_bps'] = df['spread_bps'].fillna(0.0)
            df['queue_imbalance'] = df['queue_imbalance'].fillna(0.0)
            df['impact_10k'] = df['impact_10k'].fillna(0.0)
            # kyle_lambda can remain NaN
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding market impact features: {e}")
            return df

    # Removed duplicate disconnect_ibkr method - using the one defined earlier

    # --- Live Data Methods ---
    def set_bar_update_callback(self, callback: Callable[[pd.DataFrame, str], None]):
        """
        Sets the callback function to be invoked when a new live bar is received.
        The callback will receive the bar data (pd.DataFrame) and the symbol.
        """
        self.logger.info(f"Setting bar update callback to {callback}")
        self.bar_update_callback = callback

    def get_cached_data(self, symbol: str) -> pd.DataFrame:
        """
        Get cached data for simulation mode.
        Returns the most recent cached data file for the symbol.
        """
        try:
            # Look for recent cache files
            cache_dir = "cache_ibkr"
            if not os.path.exists(cache_dir):
                self.logger.warning(f"Cache directory {cache_dir} does not exist")
                return None
            
            # Find the most recent cache file for this symbol
            cache_files = [f for f in os.listdir(cache_dir) if f.startswith(symbol) and f.endswith('.pkl')]
            if not cache_files:
                self.logger.warning(f"No cached data files found for {symbol}")
                return None
            
            # Get the most recent file
            cache_files.sort(reverse=True)  # Most recent first
            cache_file = cache_files[0]
            cache_filepath = os.path.join(cache_dir, cache_file)
            
            self.logger.info(f"Loading cached data for simulation from {cache_filepath}")
            try:
                bars_df = pd.read_pickle(cache_filepath)
                
                if bars_df.empty:
                    self.logger.warning(f"Cached file {cache_filepath} is empty")
                    return None
                    
                self.logger.info(f"Loaded {len(bars_df)} rows from cache for simulation")
                return bars_df
            except Exception as pickle_error:
                self.logger.error(f"Error reading pickle file {cache_filepath}: {pickle_error}")
                # Try other cache files if this one is corrupted
                for other_file in cache_files[1:]:  # Skip the first one we just tried
                    try:
                        other_filepath = os.path.join(cache_dir, other_file)
                        self.logger.info(f"Trying alternative cache file: {other_filepath}")
                        bars_df = pd.read_pickle(other_filepath)
                        if not bars_df.empty:
                            self.logger.info(f"Successfully loaded {len(bars_df)} rows from alternative cache")
                            return bars_df
                    except Exception as e:
                        self.logger.warning(f"Alternative cache file {other_file} also failed: {e}")
                        continue
                return None
            
        except Exception as e:
            self.logger.error(f"Error loading cached data for simulation: {e}")
            return None

    def _on_bar_update_wrapper(self, symbol: str, bars: 'RealTimeBarList', hasNewBar: bool):
        """
        Internal wrapper for ib_insync's barUpdateEvent for a specific RealTimeBarList.
        This wrapper includes the symbol.
        """
        if hasNewBar and self.bar_update_callback:
            latest_bar_data = bars[-1] # Get the most recent bar

            # Convert RealTimeBar to a DataFrame consistent with historical data
            bar_df = pd.DataFrame([{
                'datetime': pd.to_datetime(latest_bar_data.time, unit='s'), # Assuming time is Unix epoch. IBKR might send datetime object.
                COL_OPEN: latest_bar_data.open_,
                COL_HIGH: latest_bar_data.high,
                COL_LOW: latest_bar_data.low,
                COL_CLOSE: latest_bar_data.close,
                COL_VOLUME: latest_bar_data.volume,
                # 'average': latest_bar_data.wap, # Uncomment if WAP is needed as 'average'
                # 'barCount': latest_bar_data.count # Uncomment if bar count is needed
            }]).set_index('datetime')

            # Ensure datetime is timezone-aware if IB returns naive datetime
            # Example: if latest_bar_data.time is a naive datetime in UTC:
            # bar_df.index = bar_df.index.tz_localize('UTC')
            # Or convert to local time if necessary, but UTC is often preferred for internal processing.

            self.logger.debug(f"New live bar for {symbol} via wrapper: {bar_df.iloc[0].to_dict()}")
            try:
                self.bar_update_callback(bar_df, symbol)
            except Exception as e:
                self.logger.error(f"Error in bar_update_callback for {symbol}: {e}", exc_info=True)
        elif not hasNewBar:
            self.logger.debug(f"Bar update event for {symbol} but hasNewBar is False. Bars: {bars[-1] if bars else 'N/A'}")


    def subscribe_live_bars(self, symbol: str, interval_seconds: int = 5, sec_type: str = "STK",
                            exchange: str = "SMART", currency: str = "USD", use_rth: bool = True) -> bool:
        """
        Subscribes to live bars for a given symbol.
        Note: IBKR's reqRealTimeBars typically provides 5-second bars.
              Aggregation to other intervals (e.g., 1 minute) must be handled by the recipient (Orchestrator/FeatureAgent).

        Args:
            symbol (str): The stock symbol.
            interval_seconds (int): Requested bar interval in seconds. IB API usually only supports 5 for realTimeBars.
            sec_type (str): Security type (e.g., "STK", "FUT", "CASH").
            exchange (str): Exchange (e.g., "SMART", "NASDAQ", "NYSE").
            currency (str): Currency (e.g., "USD").
            use_rth (bool): If True, only data from regular trading hours.

        Returns:
            bool: True if subscription was successful or already active, False otherwise.
        """
        if not self.ib_connected:
            # Check if we're in simulation mode
            if self.ib_config and self.ib_config.get('simulation_mode', False):
                self.logger.info(f"Simulation mode: Simulating live bar subscription for {symbol}.")
                return True  # Pretend subscription succeeded
            else:
                self.logger.error(f"IBKR not connected. Cannot subscribe to live bars for {symbol}.")
                return False
        if not self.bar_update_callback:
            self.logger.error(f"Bar update callback not set. Cannot subscribe to live bars for {symbol}.")
            return False

        if interval_seconds != 5:
            self.logger.warning(f"Requested live bar interval is {interval_seconds}s for {symbol}. "
                                f"IBKR reqRealTimeBars provides 5-second bars. Subscribing to 5-second bars. "
                                "Aggregation to other intervals must be handled by the application.")
            # We proceed with 5 seconds as that's what the API provides for real-time bars.

        contract_key = (symbol, sec_type, exchange, currency) # Unique key for the contract

        if contract_key in self.live_bar_tickers:
            self.logger.warning(f"Already subscribed to live bars for {contract_key}. Unsubscribe first if parameters need to change.")
            return True

        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        # For some symbols/secTypes, primaryExchange might be needed
        # contract.primaryExchange = "NASDAQ"

        self.logger.info(f"Attempting to qualify contract for {symbol}...")
        try:
            qualified_contracts = self.ib.qualifyContracts(contract)
            if not qualified_contracts:
                self.logger.error(f"Failed to qualify contract for {symbol}.")
                return False
            qualified_contract = qualified_contracts[0]
            self.logger.info(f"Contract qualified: {qualified_contract}")
        except Exception as e:
            self.logger.error(f"Exception qualifying contract for {symbol}: {e}", exc_info=True)
            return False

        self.logger.info(f"Subscribing to live 5-second bars for {qualified_contract.symbol} (RTH: {use_rth}).")

        try:
            # reqRealTimeBars returns a RealTimeBarList object
            # barSize must be 5 (seconds) for whatToShow='TRADES', 'MIDPOINT', 'BID', or 'ASK'
            ticker = self.ib.reqRealTimeBars(
                contract=qualified_contract,
                barSize=5,
                whatToShow="TRADES",
                useRTH=use_rth,
                realTimeBarsOptions=[]
            )

            self.live_bar_tickers[contract_key] = ticker

            # Register the event handler for this specific ticker using a lambda to pass the symbol
            # The symbol passed to the wrapper should be the original requested symbol for consistency
            ticker.updateEvent += lambda bars, hasNewBar: self._on_bar_update_wrapper(symbol, bars, hasNewBar)

            self.logger.info(f"Successfully subscribed to live bars for {symbol}. Ticker object: {ticker}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to live bars for {symbol}: {e}", exc_info=True)
            if contract_key in self.live_bar_tickers:
                del self.live_bar_tickers[contract_key]
            return False

    def unsubscribe_live_bars(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD"):
        """
        Unsubscribes from live bars for a given symbol contract.
        """
        contract_key = (symbol, sec_type, exchange, currency)

        if contract_key not in self.live_bar_tickers:
            self.logger.warning(f"Not currently subscribed to live bars for {contract_key}.")
            return

        ticker = self.live_bar_tickers.pop(contract_key)
        try:
            self.logger.info(f"Unsubscribing from live bars for {contract_key}...")
            self.ib.cancelRealTimeBars(ticker)

            # Attempt to remove the specific lambda handler. This can be tricky.
            # A common way is to store the handler if it needs to be removed,
            # or rely on the ticker object being garbage collected if not referenced elsewhere.
            # For simplicity, we'll not explicitly remove the lambda handler here,
            # assuming ib_insync handles it or the ticker object is no longer used.
            # If issues arise, a more robust handler management would be needed.
            # e.g., store: self.live_bar_handlers[contract_key] = handler_lambda
            # then: ticker.updateEvent -= self.live_bar_handlers[contract_key]

            self.logger.info(f"Successfully unsubscribed from live bars for {contract_key}.")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from live bars for {contract_key}: {e}", exc_info=True)
            # If cancellation failed, consider adding the ticker back or logging severity
            self.live_bar_tickers[contract_key] = ticker # Add back if failed

    # --- End Live Data Methods ---


    # --- Methods for Order Execution ---

    def _create_contract(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART",
                         currency: str = "USD", primary_exchange: Optional[str] = None) -> Optional[Contract]:
        """
        Creates and qualifies an IBKR contract.
        Returns the qualified contract or None if qualification fails.
        """
        if not self.ib_connected:
            self.logger.error("IBKR not connected. Cannot create contract.")
            return None

        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = sec_type.upper()
        contract.exchange = exchange.upper()
        contract.currency = currency.upper()
        if primary_exchange: # Often needed for US stocks to avoid ambiguity
            contract.primaryExchange = primary_exchange.upper()

        self.logger.info(f"Attempting to qualify contract: {contract}")
        try:
            qualified_contracts = self.ib.qualifyContracts(contract)
            if not qualified_contracts:
                self.logger.error(f"Failed to qualify contract for {symbol}. No qualified contracts returned.")
                return None

            qualified_contract = qualified_contracts[0] # Usually, the first one is the most specific
            self.logger.info(f"Contract qualified: {qualified_contract}")
            return qualified_contract
        except Exception as e:
            self.logger.error(f"Exception qualifying contract for {symbol}: {e}", exc_info=True)
            return None

    def place_order(self, order_details: dict) -> Optional['ib_insync.Trade']:
        """
        Places an order with Interactive Brokers.

        Args:
            order_details (dict): A dictionary containing order parameters:
                'symbol' (str): e.g., "AAPL"
                'sec_type' (str): e.g., "STK", "FUT", "OPT". Default "STK".
                'exchange' (str): e.g., "SMART", "NASDAQ". Default "SMART".
                'currency' (str): e.g., "USD". Default "USD".
                'primary_exchange' (Optional[str]): e.g., "NASDAQ" for some US stocks.
                'action' (str): "BUY" or "SELL".
                'quantity' (float): Number of shares/contracts.
                'order_type' (str): "MKT", "LMT", "STP", "STP LMT", etc.
                'limit_price' (Optional[float]): Required for LMT orders.
                'aux_price' (Optional[float]): Required for STP LMT or TRAIL orders (stop price).
                'tif' (Optional[str]): Time in Force, e.g., "DAY", "GTC". Default "DAY".
                'account' (Optional[str]): Specify account if not default.
                # Other params like goodAfterTime, goodTillDate, ocaGroup, etc. can be added.

        Returns:
            Optional[ib_insync.Trade]: The Trade object returned by IBKR, or None if order placement failed.
                                       The Trade object should be monitored for status updates.
        """
        if not self.ib_connected:
            self.logger.error("IBKR not connected. Cannot place order.")
            return None

        if not isinstance(order_details, dict):
            self.logger.error("order_details must be a dictionary.")
            return None

        required_keys = ['symbol', 'action', 'quantity', 'order_type']
        if not all(key in order_details for key in required_keys):
            self.logger.error(f"Missing one or more required keys in order_details: {required_keys}")
            return None

        contract = self._create_contract(
            symbol=order_details['symbol'],
            sec_type=order_details.get('sec_type', "STK"),
            exchange=order_details.get('exchange', "SMART"),
            currency=order_details.get('currency', "USD"),
            primary_exchange=order_details.get('primary_exchange')
        )
        if not contract:
            self.logger.error(f"Failed to create/qualify contract for order: {order_details}")
            return None

        action = order_details['action'].upper()
        quantity = float(order_details['quantity'])
        order_type_str = order_details['order_type'].upper()

        if action not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid order action: {action}. Must be 'BUY' or 'SELL'.")
            return None
        if quantity <= 0:
            self.logger.error(f"Order quantity must be positive. Got: {quantity}")
            return None

        # Create the Order object
        order = None
        if order_type_str == "MKT":
            from ib_insync import MarketOrder # Import here to keep other parts runnable if ib_insync is missing
            order = MarketOrder(action, quantity)
        elif order_type_str == "LMT":
            limit_price = order_details.get('limit_price')
            if limit_price is None:
                self.logger.error("Limit price ('limit_price') required for LMT order.")
                return None
            from ib_insync import LimitOrder
            order = LimitOrder(action, quantity, float(limit_price))
        elif order_type_str == "STP":
            stop_price = order_details.get('aux_price') # Stop orders use auxPrice for stop price
            if stop_price is None:
                self.logger.error("Stop price ('aux_price') required for STP order.")
                return None
            from ib_insync import StopOrder
            order = StopOrder(action, quantity, float(stop_price))
        # TODO: Add other common order types like STP LMT, TRAIL as needed.
        else:
            self.logger.error(f"Unsupported order type: {order_type_str}")
            return None

        # Set Time In Force (TIF)
        order.tif = order_details.get('tif', 'DAY').upper()

        # Set account if specified
        if 'account' in order_details and order_details['account']:
            order.account = order_details['account']

        # Optional: Transmit flag (default is True, order is transmitted immediately)
        # order.transmit = order_details.get('transmit', True)

        self.logger.info(f"Placing order: Contract={contract}, Order={order}")
        try:
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Order placed successfully. Trade object: {trade}")
            self.logger.info(f"  Order ID: {trade.order.orderId if trade and trade.order else 'N/A'}")
            self.logger.info(f"  Initial Order Status: {trade.orderStatus.status if trade and trade.orderStatus else 'N/A'}")
            # The caller (Orchestrator) should monitor trade.log for status updates.
            return trade
        except Exception as e:
            self.logger.error(f"Error placing order for {contract.symbol}: {e}", exc_info=True)
            return None

    # --- End Order Execution Methods ---


    # --- Methods for Portfolio and Position Tracking ---

    def get_account_summary(self, account_id: Optional[str] = None,
                            tags: str = "NetLiquidation,TotalCashValue,AvailableFunds,GrossPositionValue,RealizedPnL,UnrealizedPnL") -> Dict[str, Any]:
        """
        Fetches account summary details like NetLiquidation, TotalCashValue, etc.

        Args:
            account_id (Optional[str]): Specific account ID. If None, summary for all accessible accounts might be fetched
                                     or it defaults to the primary account associated with the connection.
                                     For simplicity, we'll assume it fetches for the default/primary if None.
            tags (str): Comma-separated string of account tags to fetch.

        Returns:
            Dict[str, Any]: A dictionary where keys are tags and values are their corresponding values.
                            Returns an empty dict if fetching fails or not connected.
        """
        if not self.ib_connected:
            self.logger.error("IBKR not connected. Cannot fetch account summary.")
            return {}

        summary_values = {}
        try:
            # ib.accountSummary() returns a list of AccountValue objects.
            # If account_id is None, it might return summaries for multiple accounts if linked.
            # We may need to filter for a specific account if multiple are returned and account_id was None.
            # For now, let's assume we are interested in the first account summary if not specified, or a specific one.

            # Requesting all tags and then filtering might be easier than subscribing one by one.
            # However, self.ib.accountValues() is for streaming.
            # self.ib.reqAccountSummary() is also for streaming.
            # For a one-time snapshot, it's often done by requesting pnl, accountSummary etc.
            # and then processing the results that arrive via event handlers.
            # A simpler way for a snapshot is to use `ib.accountSummaryAsync` or manage subscription.

            # Let's use the simpler approach of just calling `ib.accountSummary()` and processing what we get.
            # This method, when called without args, might require prior subscription or might not work as expected for snapshot.
            # A more robust way for snapshot is to use `ib.reqAccountUpdates(True, account_id or "")`
            # then access `ib.accountValues()`. Then `ib.reqAccountUpdates(False, account_id or "")`.
            # For simplicity here, let's assume `ib.accountSummary()` can provide a snapshot or is used in a context
            # where data is already streaming. `ib_insync` docs suggest `ib.accountSummary()` is for streaming.

            # Correct approach for snapshot with ib_insync:
            acc_summary_list = self.ib.accountSummary(account=account_id if account_id else "") # Get all for the account

            if not acc_summary_list:
                self.logger.warning(f"No account summary data received for account '{account_id if account_id else 'Default'}'.")
                return {}

            # Filter by requested tags
            tag_list = [t.strip() for t in tags.split(',')]

            for acc_value in acc_summary_list:
                if acc_value.tag in tag_list:
                    try:
                        # Try to convert to float if possible, else keep as string
                        summary_values[acc_value.tag] = float(acc_value.value)
                    except ValueError:
                        summary_values[acc_value.tag] = acc_value.value

            self.logger.info(f"Fetched account summary for account '{account_id if account_id else 'Default'}': {summary_values}")

        except Exception as e:
            self.logger.error(f"Error fetching account summary: {e}", exc_info=True)
            return {}

        return summary_values

    def get_portfolio_positions(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches current portfolio positions.

        Args:
            account_id (Optional[str]): Specific account ID. If None, positions for all accessible accounts are fetched.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a position.
                                 Keys include: 'symbol', 'sec_type', 'currency', 'exchange',
                                 'position', 'market_price', 'market_value', 'average_cost',
                                 'unrealized_pnl', 'realized_pnl'.
                                 Returns an empty list if fetching fails or no positions.
        """
        if not self.ib_connected:
            self.logger.error("IBKR not connected. Cannot fetch portfolio positions.")
            return []

        positions_list = []
        try:
            portfolio_items = self.ib.portfolio() # Fetches for all accounts unless IB client is bound to one
            if not portfolio_items:
                self.logger.info("No portfolio positions found.")
                return []

            for item in portfolio_items:
                # Filter by account_id if provided
                if account_id and item.account != account_id:
                    continue

                pos_data = {
                    "account": item.account,
                    "symbol": item.contract.symbol,
                    "sec_type": item.contract.secType,
                    "currency": item.contract.currency,
                    "exchange": item.contract.exchange, # Primary exchange might be item.contract.primaryExchange
                    "primary_exchange": item.contract.primaryExchange,
                    "con_id": item.contract.conId,
                    "position": item.position, # float, can be negative for short
                    "market_price": item.marketPrice,
                    "market_value": item.marketValue,
                    "average_cost": item.averageCost,
                    "unrealized_pnl": item.unrealizedPNL,
                    "realized_pnl": item.realizedPNL
                }
                positions_list.append(pos_data)

            self.logger.info(f"Fetched {len(positions_list)} portfolio positions for account(s) '{account_id if account_id else 'All'}'.")

        except Exception as e:
            self.logger.error(f"Error fetching portfolio positions: {e}", exc_info=True)
            return []

        return positions_list



    # --- End Portfolio and Position Tracking ---

# --- End of DataAgent class ---

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Configuration
    config = {
        'data_dir_raw': 'data/raw_test',
        'default_symbol': 'DUMMY'
    }

    data_agent = DataAgent(config=config)
    print("DataAgent example run complete.")
