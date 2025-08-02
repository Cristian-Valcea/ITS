# src/gym_env/dual_ticker_data_adapter.py
"""
Dual-Ticker Data Adapter with Timestamp Alignment

Converts team's TimescaleDB data pipeline output to environment format.
Implements robust timestamp alignment with gap detection and logging.
"""

import pandas as pd
import numpy as np
import logging
import psycopg2
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timezone
import warnings


class DataQualityError(Exception):
    """Raised when data quality validation fails"""
    pass


class DualTickerDataAdapter:
    """
    Converts team's data pipeline output to dual-ticker environment format.
    
    Key Features:
    - 🔧 Robust timestamp alignment with inner join
    - 🔧 Data quality validation with detailed logging  
    - 🔧 Dropped row tracking (catches vendor gaps Day 1)
    - 🔧 Schema validation for TimescaleDB integration
    """
    
    def __init__(self, timescaledb_config: Dict[str, Any], 
                 live_trading_mode: bool = False):
        self.db_config = timescaledb_config
        self.live_trading_mode = live_trading_mode
        self.logger = logging.getLogger(f"{__name__}.DualTickerDataAdapter")
        
        # Data quality thresholds (mode-specific)
        if live_trading_mode:
            # LIVE TRADING: Strict tolerances
            self.max_missing_data_pct = 0.001  # 0.1% max missing data
            self.max_price_jump_pct = 0.10     # 10% max single-step price jump
            self.min_volume_threshold = 10000  # Higher minimum volume
            self.logger.info("🔴 LIVE TRADING MODE: Strict data quality thresholds")
        else:
            # TRAINING: Lenient tolerances
            # Note: 5% tolerance = ~10k rows for 1-year minute data (200k rows/symbol)
            self.max_missing_data_pct = 0.05  # 5% max missing data (TRAINING ONLY)
            self.max_price_jump_pct = 0.20    # 20% max single-step price jump
            self.min_volume_threshold = 1000  # Minimum daily volume
            self.logger.info("🟡 TRAINING MODE: Lenient data quality thresholds")
        
    def load_training_data(self, 
                          start_date: str, 
                          end_date: str,
                          symbols: List[str] = ['NVDA', 'MSFT'],
                          bar_size: str = '1min',
                          data_split: str = 'train') -> Dict[str, Any]:
        """
        Load aligned NVDA + MSFT data for training with professional data pipeline
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading  
            symbols: List of symbols to load (default NVDA, MSFT)
            bar_size: Bar frequency ('1min', '5min', etc.)
            data_split: Data split to load ('train', 'validation', 'test')
        
        Returns:
            Dict with aligned features, prices, and shared trading_days index
        """
        
        self.logger.info(f"🔧 Loading dual-ticker data: {symbols} from {start_date} to {end_date} (split={data_split})")
        
        # Try to load from professional pipeline first
        processed_data = self._load_processed_data(data_split, symbols)
        
        if processed_data:
            self.logger.info("✅ Using processed data from professional pipeline")
            return processed_data
        
        # Fallback to individual asset loading
        self.logger.warning("⚠️ Processed data not found, loading individual assets")
        nvda_data = self._load_asset_data('NVDA', start_date, end_date, bar_size)
        msft_data = self._load_asset_data('MSFT', start_date, end_date, bar_size)
        
        # 🔧 IMPLEMENT _align_timestamps NOW (catches vendor gaps Day 1)
        aligned_data = self._align_timestamps(nvda_data, msft_data)
        
        # Validate data quality
        self._validate_data_quality(aligned_data)
        
        return {
            'nvda_features': aligned_data['nvda']['features'],
            'nvda_prices': aligned_data['nvda']['prices'],
            'msft_features': aligned_data['msft']['features'], 
            'msft_prices': aligned_data['msft']['prices'],
            'trading_days': aligned_data['trading_days'],  # 🔧 SHARED INDEX
            'feature_names': aligned_data['feature_names']  # 🔧 COLUMN ORDER VALIDATION
        }
    
    def _load_asset_data(self, symbol: str, start_date: str, end_date: str, bar_size: str = '1min') -> pd.DataFrame:
        """Load raw data for a single asset from TimescaleDB
        
        Args:
            symbol: Asset symbol (NVDA, MSFT)
            start_date: Start date
            end_date: End date  
            bar_size: Bar frequency - passed to aggregation logic if needed
        """
        
        if self.db_config.get('mock_data', False):
            # Generate mock data for testing
            return self._generate_mock_data(symbol, start_date, end_date, bar_size)
        
        try:
            # Connect to TimescaleDB
            conn = psycopg2.connect(**self.db_config)
            
            query = f"""
            SELECT 
                timestamp,
                open, high, low, close, volume
            FROM market_data 
            WHERE symbol = %s 
            AND timestamp >= %s 
            AND timestamp <= %s
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            self.logger.info(f"📊 Loaded {len(df)} rows for {symbol} with technical indicators")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load {symbol} data: {e}")
            # Fallback to mock data
            self.logger.warning(f"Falling back to mock data for {symbol}")
            return self._generate_mock_data(symbol, start_date, end_date, bar_size)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data"""
        
        if len(df) < 50:  # Need minimum data for indicators
            # For small datasets, just add simple features
            df['rsi'] = 50.0  # Neutral RSI
            df['ema_short'] = df['close']
            df['ema_long'] = df['close']
            df['vwap'] = df['close']
            df['volatility'] = 0.02
            df['momentum'] = 0.0
            df['time_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 252)
            df['time_cos'] = np.cos(2 * np.pi * np.arange(len(df)) / 252)
            df['volume_sma'] = df['volume']
            df['price_change'] = df['close'].diff().fillna(0)
            df['returns'] = df['close'].pct_change().fillna(0)
            return df
        
        # Calculate RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Fill NaN with neutral
        
        # EMAs
        df['ema_short'] = df['close'].ewm(span=12).mean()
        df['ema_long'] = df['close'].ewm(span=26).mean()
        
        # VWAP (approximation)
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['vwap'] = df['vwap'].fillna(df['close'])
        
        # Volatility (20-period rolling std of returns)
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(window=20).std().fillna(0.02)
        
        # Momentum (10-period price change)
        df['momentum'] = df['close'].pct_change(periods=10).fillna(0)
        
        # Time features (day of year cycle)
        day_of_year = df.index.dayofyear
        df['time_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        df['time_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean().fillna(df['volume'])
        
        # Price change and returns
        df['price_change'] = df['close'].diff().fillna(0)
        df['returns'] = df['close'].pct_change().fillna(0)
        
        return df
    
    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str, bar_size: str = '1min') -> pd.DataFrame:
        """Generate mock market data for testing
        
        Args:
            symbol: Asset symbol  
            start_date: Start date
            end_date: End date
            bar_size: Bar frequency for intraday data generation
        """
        
        # Create date range (business days only)
        dates = pd.bdate_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Calculate bars per day based on bar_size
        if bar_size == '1min':
            bars_per_day = 390
        elif bar_size == '5min':
            bars_per_day = 78
        else:
            bars_per_day = 390  # Default fallback
        
        # Generate realistic price data
        np.random.seed(42 if symbol == 'NVDA' else 123)  # Reproducible but different
        
        base_price = 500.0 if symbol == 'NVDA' else 300.0
        price_changes = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Generate OHLC from close prices
        opens = prices * (1 + np.random.normal(0, 0.005, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        closes = prices
        volumes = np.random.lognormal(15, 0.5, n_days)  # Realistic volume distribution
        
        # Generate technical features
        rsi = 50 + 30 * np.sin(np.arange(n_days) * 0.1) + np.random.normal(0, 5, n_days)
        rsi = np.clip(rsi, 0, 100)
        
        ema_short = prices * (1 + np.random.normal(0, 0.01, n_days))
        ema_long = prices * (1 + np.random.normal(0, 0.02, n_days))
        vwap = prices * (1 + np.random.normal(0, 0.005, n_days))
        volatility = np.abs(np.random.normal(0.02, 0.01, n_days))
        momentum = np.random.normal(0, 0.01, n_days)
        
        # Time features
        time_sin = np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual cycle
        time_cos = np.cos(2 * np.pi * np.arange(n_days) / 252)
        
        # Volume and return features
        volume_sma = volumes * (1 + np.random.normal(0, 0.1, n_days))
        price_change = np.diff(prices, prepend=prices[0])
        returns = price_change / prices
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs, 
            'low': lows,
            'close': closes,
            'volume': volumes,
            'rsi': rsi,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'vwap': vwap,
            'volatility': volatility,
            'momentum': momentum,
            'time_sin': time_sin,
            'time_cos': time_cos,
            'volume_sma': volume_sma,
            'price_change': price_change,
            'returns': returns
        }, index=dates)
        
        self.logger.info(f"📊 Generated {len(df)} mock data rows for {symbol}")
        return df
    
    def _align_timestamps(self, nvda_data: pd.DataFrame, msft_data: pd.DataFrame) -> Dict[str, Any]:
        """
        🔧 INNER JOIN on timestamp with dropped row logging
        
        Catches vendor gaps immediately and ensures perfect alignment
        """
        
        self.logger.info("🔧 Aligning timestamps between NVDA and MSFT data...")
        
        # Get timestamps for both assets
        nvda_timestamps = set(nvda_data.index)
        msft_timestamps = set(msft_data.index)
        
        # Find common timestamps (inner join)
        common_timestamps = nvda_timestamps.intersection(msft_timestamps)
        
        # 🔧 LOG DROPPED ROWS (catches vendor gaps immediately)
        nvda_dropped = nvda_timestamps - common_timestamps
        msft_dropped = msft_timestamps - common_timestamps
        
        if nvda_dropped:
            dropped_list = sorted(list(nvda_dropped))
            self.logger.warning(f"🔧 NVDA: Dropped {len(nvda_dropped)} timestamps")
            self.logger.warning(f"🔧 First 5 dropped NVDA timestamps: {dropped_list[:5]}")
            if len(nvda_dropped) > 10:
                self.logger.warning(f"🔧 ... and {len(nvda_dropped) - 5} more NVDA timestamps")
                
        if msft_dropped:
            dropped_list = sorted(list(msft_dropped))
            self.logger.warning(f"🔧 MSFT: Dropped {len(msft_dropped)} timestamps")
            self.logger.warning(f"🔧 First 5 dropped MSFT timestamps: {dropped_list[:5]}")
            if len(msft_dropped) > 10:
                self.logger.warning(f"🔧 ... and {len(msft_dropped) - 5} more MSFT timestamps")
        
        # Check if we lost too much data
        original_nvda_count = len(nvda_timestamps)
        original_msft_count = len(msft_timestamps)
        common_count = len(common_timestamps)
        
        nvda_loss_pct = len(nvda_dropped) / original_nvda_count if original_nvda_count > 0 else 0
        msft_loss_pct = len(msft_dropped) / original_msft_count if original_msft_count > 0 else 0
        
        if nvda_loss_pct > self.max_missing_data_pct:
            self.logger.error(f"🔧 NVDA data loss too high: {nvda_loss_pct:.2%} > {self.max_missing_data_pct:.2%}")
        if msft_loss_pct > self.max_missing_data_pct:
            self.logger.error(f"🔧 MSFT data loss too high: {msft_loss_pct:.2%} > {self.max_missing_data_pct:.2%}")
        
        # Create aligned datasets
        trading_days = pd.DatetimeIndex(sorted(common_timestamps))
        
        aligned_nvda = nvda_data.reindex(trading_days)
        aligned_msft = msft_data.reindex(trading_days)
        
        # Validate no NaN values after alignment
        assert not aligned_nvda.isnull().any().any(), "NVDA data has NaN after alignment"
        assert not aligned_msft.isnull().any().any(), "MSFT data has NaN after alignment"
        
        # Extract features and prices with column order validation
        feature_columns = ['rsi', 'ema_short', 'ema_long', 'vwap', 'volatility', 'momentum', 
                          'time_sin', 'time_cos', 'volume_sma', 'price_change', 'returns', 'volume']
        
        # Critical: Assert feature column order consistency for weight transfer
        nvda_columns = list(aligned_nvda[feature_columns].columns)
        msft_columns = list(aligned_msft[feature_columns].columns)
        
        assert nvda_columns == msft_columns == feature_columns, \
            f"Feature column order mismatch: NVDA {nvda_columns} vs MSFT {msft_columns} vs expected {feature_columns}"
        
        self.logger.info(f"🔧 Feature column order validated: {len(feature_columns)} features consistent")
        
        nvda_features = aligned_nvda[feature_columns].values.astype(np.float32)
        msft_features = aligned_msft[feature_columns].values.astype(np.float32)
        
        nvda_prices = aligned_nvda['close']
        msft_prices = aligned_msft['close']
        
        self.logger.info(f"🔧 Data aligned successfully: {len(trading_days)} common timestamps")
        self.logger.info(f"🔧 Feature matrix shape: NVDA {nvda_features.shape}, MSFT {msft_features.shape}")
        
        return {
            'nvda': {
                'features': nvda_features,
                'prices': nvda_prices,
                'raw_data': aligned_nvda
            },
            'msft': {
                'features': msft_features, 
                'prices': msft_prices,
                'raw_data': aligned_msft
            },
            'trading_days': trading_days,
            'feature_names': feature_columns  # 🔧 STORE FOR WEIGHT TRANSFER VALIDATION
        }
    
    def _validate_data_quality(self, aligned_data: Dict[str, Any]) -> None:
        """Validate data quality and raise errors if issues found"""
        
        self.logger.info("🔧 Running data quality validation...")
        
        issues = []
        
        # Check minimum data points
        n_days = len(aligned_data['trading_days'])
        if n_days < 100:
            issues.append(f"Insufficient data: {n_days} days < 100 minimum")
        
        # Validate price continuity for both assets
        for symbol in ['nvda', 'msft']:
            prices = aligned_data[symbol]['prices']
            
            # Check for extreme price jumps
            price_changes = prices.pct_change().dropna()
            extreme_changes = price_changes[abs(price_changes) > self.max_price_jump_pct]
            
            if len(extreme_changes) > 0:
                self.logger.warning(f"🔧 {symbol.upper()}: {len(extreme_changes)} extreme price jumps detected")
                for date, change in extreme_changes.head().items():
                    self.logger.warning(f"🔧   {date}: {change:.2%} change")
            
            # Check for zero or negative prices
            invalid_prices = prices[prices <= 0]
            if len(invalid_prices) > 0:
                issues.append(f"{symbol.upper()}: {len(invalid_prices)} invalid prices (≤ 0)")
            
            # Check volume data
            if 'volume' in aligned_data[symbol]['raw_data'].columns:
                volumes = aligned_data[symbol]['raw_data']['volume']
                low_volume_days = volumes[volumes < self.min_volume_threshold]
                if len(low_volume_days) > n_days * 0.1:  # More than 10% low volume days
                    self.logger.warning(f"🔧 {symbol.upper()}: {len(low_volume_days)} low volume days")
        
        # Check feature matrix validity
        for symbol in ['nvda', 'msft']:
            features = aligned_data[symbol]['features']
            
            # Check for NaN or infinite values
            if np.isnan(features).any():
                issues.append(f"{symbol.upper()}: NaN values in feature matrix")
            if np.isinf(features).any():
                issues.append(f"{symbol.upper()}: Infinite values in feature matrix")
            
            # Check feature ranges (basic sanity)
            if features.shape[1] != 12:
                issues.append(f"{symbol.upper()}: Expected 12 features, got {features.shape[1]}")
        
        # Raise error if critical issues found
        if issues:
            error_msg = "Data quality validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            self.logger.error(f"🔧 {error_msg}")
            raise DataQualityError(error_msg)
        
        self.logger.info("✅ Data quality validation passed")
    
    def test_schema_and_insert(self) -> bool:
        """
        🔧 Test TimescaleDB schema and perform 10-row insert test
        
        Used by CI pipeline to validate database connectivity
        """
        
        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Test schema existence
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'market_data'
            """)
            
            if not cursor.fetchone():
                self.logger.warning("🔧 market_data table doesn't exist, creating...")
                self._create_test_schema(cursor)
            
            # Test 10-row insert
            test_data = self._generate_test_rows(10)
            
            for row in test_data:
                cursor.execute("""
                    INSERT INTO market_data 
                    (timestamp, symbol, open, high, low, close, volume, rsi)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol) DO NOTHING
                """, row)
            
            conn.commit()
            
            # Verify insert
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'TEST'")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"✅ TimescaleDB test successful: {count} test rows inserted")
            return True
            
        except Exception as e:
            self.logger.error(f"🔧 TimescaleDB test failed: {e}")
            return False
    
    def _create_test_schema(self, cursor):
        """Create basic market_data table for testing"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                rsi DOUBLE PRECISION,
                ema_short DOUBLE PRECISION,
                ema_long DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                volatility DOUBLE PRECISION,
                momentum DOUBLE PRECISION,
                time_sin DOUBLE PRECISION,
                time_cos DOUBLE PRECISION,
                volume_sma DOUBLE PRECISION,
                price_change DOUBLE PRECISION,
                returns DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Create hypertable if TimescaleDB extension is available
        try:
            cursor.execute("SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE)")
        except:
            pass  # Regular PostgreSQL without TimescaleDB
    
    def _generate_test_rows(self, n_rows: int) -> List[Tuple]:
        """Generate test data rows for schema validation"""
        from datetime import datetime, timedelta
        
        rows = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(n_rows):
            timestamp = base_time + timedelta(minutes=i)
            rows.append((
                timestamp,
                'TEST',
                100.0 + i,  # open
                101.0 + i,  # high
                99.0 + i,   # low
                100.5 + i,  # close
                1000000,    # volume
                50.0        # rsi
            ))
        
        return rows
    
    def _load_processed_data(self, data_split: str, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """
        Load processed data from professional pipeline
        
        Args:
            data_split: Data split to load ('train', 'validation', 'test')
            symbols: List of symbols to load
            
        Returns:
            Dict with aligned features and prices, or None if not found
        """
        from pathlib import Path
        
        processed_data_path = Path("data/processed")
        
        if not processed_data_path.exists():
            return None
        
        try:
            # Load processed parquet files for each symbol
            symbol_data = {}
            
            for symbol in symbols:
                filename = f"{symbol.lower()}_{data_split}_processed.parquet"
                filepath = processed_data_path / filename
                
                if not filepath.exists():
                    self.logger.warning(f"⚠️ Processed data not found: {filepath}")
                    return None
                
                df = pd.read_parquet(filepath)
                
                # Calculate technical indicators if not present
                if 'rsi' not in df.columns:
                    df = self._add_technical_indicators(df)
                
                symbol_data[symbol] = df
            
            # Align timestamps between symbols
            if len(symbol_data) == 2:
                nvda_data = symbol_data['NVDA']
                msft_data = symbol_data['MSFT']
                
                aligned_data = self._align_timestamps(nvda_data, msft_data)
                
                self.logger.info(f"✅ Loaded processed data split '{data_split}': {len(aligned_data['trading_days'])} aligned timestamps")
                
                return {
                    'nvda_features': aligned_data['nvda']['features'],
                    'nvda_prices': aligned_data['nvda']['prices'],
                    'msft_features': aligned_data['msft']['features'], 
                    'msft_prices': aligned_data['msft']['prices'],
                    'trading_days': aligned_data['trading_days'],
                    'feature_names': aligned_data['feature_names'],
                    'data_source': 'professional_pipeline',
                    'data_split': data_split
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error loading processed data: {e}")
            return None
        
        return None
    
    def get_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics for loaded data"""
        
        summary = {
            'date_range': {
                'start': data['trading_days'][0],
                'end': data['trading_days'][-1],
                'total_days': len(data['trading_days'])
            },
            'nvda': {
                'price_range': {
                    'min': float(data['nvda_prices'].min()),
                    'max': float(data['nvda_prices'].max()),
                    'mean': float(data['nvda_prices'].mean())
                },
                'feature_shape': data['nvda_features'].shape
            },
            'msft': {
                'price_range': {
                    'min': float(data['msft_prices'].min()),
                    'max': float(data['msft_prices'].max()),
                    'mean': float(data['msft_prices'].mean())
                },
                'feature_shape': data['msft_features'].shape
            }
        }
        
        return summary