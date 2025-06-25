# src/agents/data_agent.py
import os
import pandas as pd
from datetime import datetime, timedelta
from ib_insync import IB, util, Contract, Stock # Assuming ib_insync is installed

from .base_agent import BaseAgent
# from ..utils.config_loader import load_config # Example for loading config

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
        
        self.ib_config = self.config.get('ibkr_conn', {})
        self.ib = None # Will be initialized
        self.ib_connected = False
        self._ib_managed_locally = False # Flag to track if this instance manages the connection

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

        if not self.ib_config.get('host') or not self.ib_config.get('port') or self.ib_config.get('clientId') is None:
            self.logger.error("IBKR connection parameters (host, port, clientId) missing in config.")
            return

        try:
            self.ib = IB()
            self.logger.info(f"Attempting to connect to IBKR at {self.ib_config['host']}:{self.ib_config['port']} with clientId {self.ib_config['clientId']}...")
            # util.patchAsyncio() # Not always needed, depends on environment
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
            self.logger.error(f"IBKR connection refused. Ensure TWS/Gateway is running and API connections are enabled.")
            self.ib_connected = False
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}", exc_info=True)
            self.ib_connected = False


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
            end_dt_obj = datetime.strptime(end_datetime_str.split(" ")[0], "%Y%m%d")
            
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
        filename = f"{symbol}_{start_date_str_yyyymmdd}_{end_date_str_yyyymmdd}_{interval_sanitized}.{data_format}"
        return os.path.join(self.data_dir_raw, symbol.upper(), filename)

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

        # TODO: Construct cache path based on a more robust representation of start/end dates
        # For now, let's use a simplified version of the end_datetime_str and duration for caching name
        # A proper implementation would parse end_datetime_str and duration_str to determine actual start_date
        # For this skeleton, we'll simplify the cache filename.
        # This needs to be made more robust to map to actual date ranges.
        # Example: parse end_datetime_str and duration_str to get an actual start_date for filename.
        # For now, using a placeholder for cache path construction.
        # A real implementation would calculate the effective start date.
        cache_filepath = self._get_cache_filepath(symbol, "TEMP_START", end_datetime_str.split(' ')[0], bar_size_str.replace(' ', ''), data_format)
        
        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)

        if not force_fetch and os.path.exists(cache_filepath):
            self.logger.info(f"Loading cached data from {cache_filepath}")
            try:
                if data_format == "csv":
                    bars_df = pd.read_csv(cache_filepath, index_col='date', parse_dates=True)
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
        self.logger.warning("Actual IBKR fetching is not implemented in this skeleton.")
        # Create a dummy DataFrame for skeleton purposes
        dates = pd.to_datetime([
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 9, 31),
            datetime(2023, 1, 1, 9, 32)
        ])
        dummy_data = {'open': [100, 101, 102], 'high': [102, 102, 103], 'low': [99, 100, 101],
                      'close': [101, 102, 103], 'volume': [1000, 1200, 1100]}
        bars_df = pd.DataFrame(dummy_data, index=pd.DatetimeIndex(dates, name='date'))
        
        # Save dummy data to cache path for testing workflow
        try:
            if data_format == "csv":
                bars_df.to_csv(cache_filepath)
            elif data_format == "pkl":
                bars_df.to_pickle(cache_filepath)
            self.logger.info(f"Saved DUMMY data to {cache_filepath}")
        except Exception as e:
            self.logger.error(f"Error caching DUMMY data to {cache_filepath}: {e}")
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
        required_cols = ['open', 'high', 'low', 'close', 'volume']
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

    def disconnect_ibkr(self):
        """Disconnects from IBKR if connected by this agent instance."""
        # TODO: Implement disconnection logic if IBKR connection was managed by this agent.
        # if hasattr(self, 'ib') and self.ib and self.ib.isConnected() and not self.ib_client_passed_in:
        #     self.logger.info("Disconnecting from IBKR.")
        #     self.ib.disconnect()
        #     self.ib_connected = False
        pass


if __name__ == '__main__':
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
        force_fetch=True # Set to True to regenerate dummy data
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
