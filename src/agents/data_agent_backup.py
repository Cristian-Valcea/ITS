# src/agents/data_agent.py
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ib_insync import IB, util, Contract, Stock # Assuming ib_insync is installed

from .base_agent import BaseAgent
# from ..utils.config_loader import load_config # Example for loading config
from src.tools.ibkr_tools import fetch_5min_bars
from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME
from src.shared import get_feature_store

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

        # Now call fetch_5min_bars with the correct arguments
        bars_df = fetch_5min_bars(symbol, start_date_str_yyyymmdd, end_date_str_yyyymmdd, use_rth=use_rth)
        #bars_df, _bt_feed = get_ibkr_data_sync(
        #    ib_instance=self.ib,
        #    ticker_symbol=symbol,
        #    start_date=start_date_str_yyyymmdd,
        #    end_date=end_date_str_yyyymmdd,
        #    bar_size='5 mins',      # exactly this string
        #    what_to_show='TRADES',  # matches your IBKR requests
        #    use_rth=use_rth
        #)
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


    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Performs basic validation on the fetched data.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            symbol (str): Symbol for logging purposes.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        if df is None or df.empty:
            self.logger.warning(f"Data validation failed for {symbol}: DataFrame is empty.")
            return False

        # Check for required columns (adjust based on actual IBKR output)
        # Typical columns: open, high, low, close, volume
        required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Data validation failed for {symbol}: Missing one or more required columns {required_cols}.")
            return False

        # Check for NaNs
        if df.isnull().values.any():
            self.logger.warning(f"Data validation failed for {symbol}: DataFrame contains NaN values.")
            # TODO: Add more sophisticated NaN handling (e.g., fill or drop based on strategy)
            return False

        # Check if index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(f"Data validation failed for {symbol}: Index is not a DatetimeIndex.")
            return False
        
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            self.logger.warning(f"Data validation failed for {symbol}: Data is not sorted chronologically. Sorting...")
            df.sort_index(inplace=True)

        # TODO: Add more checks:
        # - Outlier detection (e.g., unusually large price changes or volumes)
        # - Consistent time intervals between bars (for unformly sampled data)
        # - Volume checks (e.g., non-negative volume)

        self.logger.info(f"Data validation passed for {symbol} with {len(df)} rows.")
        return True

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
        
        bars_df = self.fetch_ibkr_bars(
            symbol=symbol,
            end_datetime_str=end_date, # This is the crucial IBKR parameter
            duration_str=duration_str, # This determines how far back from end_date
            bar_size_str=interval,
            what_to_show=kwargs.get('what_to_show', "TRADES"),
            use_rth=kwargs.get('use_rth', True),
            data_format=data_format,
            force_fetch=force_fetch
        )

        if bars_df is not None:
            if self.validate_data(bars_df, symbol):
                self.logger.info(f"Successfully fetched and validated data for {symbol}.")
                # The path where data was actually saved is determined within fetch_ibkr_bars
                # based on its parameters. If you need the exact path here, fetch_ibkr_bars
                # should return it or it should be reconstructed identically.
                # For now, we just return the DataFrame.
                return bars_df
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

    # Removed duplicate disconnect_ibkr method - using the one defined earlier

    # --- Live Data Methods ---
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Configuration (replace with actual config loading)
    config = {
        'data_dir_raw': 'data/raw_test',
        'default_symbol': 'DUMMY',
        # 'ibkr_conn': { # Optional: if DataAgent manages its own connection
        #     'host': '127.0.0.1',
        #     'port': 7497, # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway Paper, 4001 for Gateway Live
        #     'clientId': 101 
        # }
    }

    data_agent = DataAgent(config=config)

    # Example: Fetch dummy data (since IBKR connection is not live in this skeleton)
    # In a real scenario, you'd provide valid IBKR parameters.
    # The current fetch_ibkr_bars will generate dummy data if IBKR part is commented out.
    
    # Parameters for fetching
    symbol_to_fetch = config['default_symbol']
    # For IBKR, endDateTime is usually YYYYMMDD HH:MM:SS [TimeZone] or "" for current time.
    # For dummy data, we'll use a placeholder that helps form the filename.
    # The actual date range of dummy data is fixed inside fetch_ibkr_bars for now.
    end_date_for_fetch = datetime.now().strftime("%Y%m%d %H:%M:%S") # Example: current time
    
    # This duration is how far back from end_date_for_fetch.
    # The actual start date of data would be (end_date_for_fetch - duration_for_fetch).
    # This calculated start date should be used in cache naming for consistency.
    # The _get_cache_filepath method needs to be updated to reflect this.
    duration_for_fetch = "5 D" # Example: 5 days of data
    bar_interval = "1 min"

    print(f"\n--- Running DataAgent for {symbol_to_fetch} ---")
    # The 'start_date' in run() is currently for conceptual grouping/logging.
    # The actual data window is defined by 'end_date' and 'duration_str' for IBKR.
    # The cache filename should ideally reflect the true start and end of the data.
    # For this example, we'll use a placeholder start date for run() that matches the cache logic.
    # A more robust solution would derive the cache's start_date from end_date and duration.
    
    # Let's simulate a start_date for the run method's conceptual purpose.
    # The actual data fetching logic in fetch_ibkr_bars uses end_datetime_str and duration_str.
    # The cache key in _get_cache_filepath is currently simplified and needs start_date.
    # We'll pass a placeholder for now.
    
    df_bars = data_agent.run(
        symbol=symbol_to_fetch,
        start_date="20230101", # Placeholder, actual data depends on end_date & duration
        end_date=end_date_for_fetch,
        interval=bar_interval,
        duration_str=duration_for_fetch, # This is used by fetch_ibkr_bars
        force_fetch=False # Set to True to regenerate dummy data
    )

    if df_bars is not None:
        print(f"\n--- Fetched Data for {symbol_to_fetch} ---")
        print(df_bars.head())
        print(f"Data shape: {df_bars.shape}")
        
        # Example of how the cache path might be constructed for verification
        # This needs to align perfectly with _get_cache_filepath's internal logic
        # Note: This is still using the simplified cache path logic
        expected_cache_path = data_agent._get_cache_filepath(
            symbol_to_fetch, 
            "TEMP_START", # This placeholder matches the one in fetch_ibkr_bars
            end_date_for_fetch.split(' ')[0], 
            bar_interval.replace(' ', ''), 
            "csv"
        )
        print(f"Data for {symbol_to_fetch} is expected to be cached at/loaded from: {expected_cache_path}")
        if os.path.exists(expected_cache_path):
            print("Cache file exists.")
        else:
            print("Cache file does NOT exist (may be an issue with path matching or save).")

    else:
        print(f"Could not retrieve data for {symbol_to_fetch}.")

    # data_agent.disconnect_ibkr() # If connection was managed by agent
    print("\nDataAgent example run complete.")


    # --- Methods for Live Data Handling ---
    def set_bar_update_callback(self, callback: Callable[[pd.DataFrame, str], None]):
        """
        Sets the callback function to be invoked when a new live bar is received.
        The callback will receive the bar data (pd.DataFrame) and the symbol.
        """
        self.logger.info(f"Setting bar update callback to {callback}")
        self.bar_update_callback = callback

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
