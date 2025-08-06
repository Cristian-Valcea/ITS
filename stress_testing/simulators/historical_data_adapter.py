"""
Historical Data Adapter for Stress Testing

Connects to TimescaleDB to fetch real historical market data for stress testing scenarios.
Integrates with vault access and database connections as documented in VAULT_ACCESS_GUIDE.md
and dbConnections.md.
"""

import psycopg2
import pandas as pd
from typing import Dict, List, Optional, Iterator, Tuple
from datetime import datetime, date
import logging
import os
from dotenv import load_dotenv

from ..core.config import get_config


class HistoricalDataAdapter:
    """
    Adapter for accessing historical market data from TimescaleDB.
    
    Provides clean interface for stress testing scenarios to access
    real market data with proper error handling and connection management.
    """
    
    def __init__(self):
        self.config = get_config()
        self._connection = None
        
        # Load environment variables for database access
        load_dotenv()
        
        logging.info("HistoricalDataAdapter initialized")
    
    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection using environment variables."""
        if self._connection is None or self._connection.closed:
            try:
                # Use password from .env file (as documented in VAULT_ACCESS_GUIDE.md)
                db_password = os.getenv('TIMESCALE_PASSWORD')
                if not db_password:
                    raise ValueError("TIMESCALE_PASSWORD not found in environment")
                
                self._connection = psycopg2.connect(
                    host='localhost',
                    port=5432,
                    database='trading_data',
                    user='postgres',
                    password=db_password
                )
                logging.info("Database connection established")
                
            except Exception as e:
                logging.error(f"Database connection failed: {e}")
                raise
        
        return self._connection
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the database."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT DISTINCT symbol FROM minute_bars ORDER BY symbol;")
            symbols = [row[0] for row in cur.fetchall()]
            logging.info(f"Available symbols: {symbols}")
            return symbols
            
        except Exception as e:
            logging.error(f"Failed to get symbols: {e}")
            raise
        finally:
            cur.close()
    
    def get_date_range(self, symbol: str) -> Tuple[date, date]:
        """Get available date range for a symbol."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT 
                    MIN(DATE(timestamp)) as min_date,
                    MAX(DATE(timestamp)) as max_date
                FROM minute_bars 
                WHERE symbol = %s;
            """, (symbol,))
            
            result = cur.fetchone()
            if result and result[0]:
                return result[0], result[1]
            else:
                raise ValueError(f"No data found for symbol {symbol}")
                
        except Exception as e:
            logging.error(f"Failed to get date range for {symbol}: {e}")
            raise
        finally:
            cur.close()
    
    def get_trading_day_data(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        Get all minute bars for a specific trading day.
        
        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            target_date: Date to fetch data for
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM minute_bars 
                WHERE symbol = %s 
                AND DATE(timestamp) = %s
                ORDER BY timestamp;
            """, (symbol, target_date))
            
            data = cur.fetchall()
            if not data:
                raise ValueError(f"No data found for {symbol} on {target_date}")
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime if needed
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            
            df['volume'] = df['volume'].astype(int)
            
            logging.info(f"Retrieved {len(df)} bars for {symbol} on {target_date}")
            return df
            
        except Exception as e:
            logging.error(f"Failed to get trading day data for {symbol} on {target_date}: {e}")
            raise
        finally:
            cur.close()
    
    def get_flash_crash_data(self, symbol: str = 'NVDA', crash_date: str = '2023-10-17') -> pd.DataFrame:
        """
        Get data for the flash crash simulation.
        
        Returns minute-by-minute data for the specified crash date.
        """
        target_date = datetime.strptime(crash_date, '%Y-%m-%d').date()
        
        df = self.get_trading_day_data(symbol, target_date)
        
        # Add some metadata for crash analysis
        day_high = df['high'].max()
        day_low = df['low'].min()
        intraday_range = ((day_high - day_low) / day_low) * 100
        
        logging.info(f"Flash crash data: {len(df)} bars, {intraday_range:.1f}% intraday range")
        
        return df
    
    def get_high_volatility_periods(self, symbol: str, days: int = 10) -> List[Tuple[date, float]]:
        """
        Find high volatility trading days for stress testing.
        
        Args:
            symbol: Stock symbol
            days: Number of high volatility days to return
            
        Returns:
            List of (date, volatility_pct) tuples
        """
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT 
                    DATE(timestamp) as trading_date,
                    (MAX(high) - MIN(low)) / MIN(low) * 100 as intraday_range_pct,
                    COUNT(*) as bar_count
                FROM minute_bars 
                WHERE symbol = %s
                GROUP BY DATE(timestamp)
                HAVING COUNT(*) > 100  -- Ensure full trading day
                ORDER BY intraday_range_pct DESC
                LIMIT %s;
            """, (symbol, days))
            
            results = cur.fetchall()
            volatility_days = [(row[0], row[1]) for row in results]
            
            logging.info(f"Found {len(volatility_days)} high volatility days for {symbol}")
            return volatility_days
            
        except Exception as e:
            logging.error(f"Failed to get high volatility periods: {e}")
            raise
        finally:
            cur.close()
    
    def stream_minute_bars(self, symbol: str, target_date: date) -> Iterator[Dict]:
        """
        Stream minute bars one by one for real-time simulation.
        
        Yields:
            Dict with keys: timestamp, open, high, low, close, volume
        """
        df = self.get_trading_day_data(symbol, target_date)
        
        for _, row in df.iterrows():
            yield {
                'timestamp': row['timestamp'],
                'symbol': symbol,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
    
    def validate_data_availability(self) -> Dict[str, any]:
        """
        Validate that required data is available for stress testing.
        
        Returns:
            Validation report with data availability status
        """
        report = {
            'database_connected': False,
            'symbols_available': [],
            'flash_crash_data_available': False,
            'total_bars': 0,
            'date_range': None,
            'validation_passed': False
        }
        
        try:
            # Test database connection
            conn = self._get_connection()
            report['database_connected'] = True
            
            # Get available symbols
            symbols = self.get_available_symbols()
            report['symbols_available'] = symbols
            
            # Check NVDA data availability
            if 'NVDA' in symbols:
                min_date, max_date = self.get_date_range('NVDA')
                report['date_range'] = (min_date, max_date)
                
                # Check flash crash date
                crash_date = datetime.strptime('2023-10-17', '%Y-%m-%d').date()
                if min_date <= crash_date <= max_date:
                    crash_data = self.get_trading_day_data('NVDA', crash_date)
                    if len(crash_data) > 100:  # Ensure sufficient data
                        report['flash_crash_data_available'] = True
                
                # Get total bar count
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM minute_bars WHERE symbol = 'NVDA';")
                report['total_bars'] = cur.fetchone()[0]
                cur.close()
            
            # Overall validation
            report['validation_passed'] = (
                report['database_connected'] and
                len(report['symbols_available']) > 0 and
                report['flash_crash_data_available'] and
                report['total_bars'] > 100000  # Ensure substantial dataset
            )
            
            logging.info(f"Data validation: {'PASSED' if report['validation_passed'] else 'FAILED'}")
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            report['error'] = str(e)
        
        return report
    
    def close(self):
        """Close database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            logging.info("Database connection closed")


# Global instance for easy access
_global_adapter: Optional[HistoricalDataAdapter] = None


def get_historical_data_adapter() -> HistoricalDataAdapter:
    """Get the global historical data adapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = HistoricalDataAdapter()
    return _global_adapter