#!/usr/bin/env python3
"""
🎯 POLYGON HISTORICAL DATA FETCHER - INSTITUTIONAL GRADE
Fetches 1-minute bars for NVDA & MSFT from 2022-01-03 to 2025-07-31
Designed for the gold-standard V3 training pipeline
"""

import os
import sys
import json
import time
import logging
import requests
import argparse
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolygonHistoricalFetcher:
    """Institutional-grade historical data fetcher for V3 training"""
    
    def __init__(self):
        # Get API key
        try:
            self.api_key = SecretsHelper.get_polygon_api_key()
            logger.info("✅ Polygon API key retrieved")
        except Exception as e:
            logger.error(f"❌ Failed to get Polygon API key: {e}")
            raise
        
        # Polygon API configuration
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.rate_limit_delay = 12  # 5 calls/minute for Starter plan
        
        # Database configuration
        try:
            db_password = SecretsHelper.get_timescaledb_password()
            self.db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_data',
                'user': 'postgres',
                'password': db_password
            }
            logger.info("✅ Database config retrieved")
        except Exception as e:
            logger.error(f"❌ Failed to get database config: {e}")
            raise
        
        # Symbols for dual-ticker system
        self.symbols = ["NVDA", "MSFT"]
    
    def fetch_minute_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch 1-minute bars for a symbol over date range"""
        
        url = f"{self.base_url}/{symbol}/range/1/minute/{start_date}/{end_date}"
        
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000  # Maximum allowed
        }
        
        all_data = []
        
        try:
            logger.info(f"📡 Fetching {symbol} 1-min data: {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    bars = data['results']
                    logger.info(f"✅ {symbol}: Retrieved {len(bars)} minute bars")
                    
                    # Convert to standard format
                    for bar in bars:
                        timestamp = datetime.fromtimestamp(bar['t'] / 1000)
                        
                        # Only include market hours (9:30 AM - 4:00 PM ET)
                        if 9.5 <= timestamp.hour + timestamp.minute/60 <= 16:
                            all_data.append({
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'open': bar['o'],
                                'high': bar['h'],
                                'low': bar['l'],
                                'close': bar['c'],
                                'volume': bar['v']
                            })
                    
                    logger.info(f"✅ {symbol}: {len(all_data)} market-hours bars processed")
                    
                else:
                    logger.warning(f"⚠️ {symbol}: No data - {data.get('status', 'unknown')}")
                    
            elif response.status_code == 429:
                logger.error(f"❌ Rate limit exceeded for {symbol}")
                logger.info("⏳ Waiting 60 seconds for rate limit reset...")
                time.sleep(60)
                return self.fetch_minute_data(symbol, start_date, end_date)  # Retry
                
            else:
                logger.error(f"❌ HTTP {response.status_code} for {symbol}: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} data: {e}")
        
        return all_data
    
    def fetch_date_range_chunked(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch data in chunks to handle large date ranges"""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        chunk_size_days = 30  # 30-day chunks to stay within API limits
        
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size_days), end_dt)
            
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = current_end.strftime('%Y-%m-%d')
            
            logger.info(f"📊 Fetching {symbol} chunk: {chunk_start_str} to {chunk_end_str}")
            
            chunk_data = self.fetch_minute_data(symbol, chunk_start_str, chunk_end_str)
            all_data.extend(chunk_data)
            
            # Rate limiting between chunks
            if current_end < end_dt:
                logger.info(f"⏳ Rate limiting: waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
            
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"🎯 {symbol}: Total {len(all_data)} bars fetched")
        return all_data
    
    def insert_to_timescaledb(self, data: List[Dict]) -> int:
        """Insert data into TimescaleDB"""
        
        if not data:
            return 0
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create table if not exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS minute_bars (
                symbol VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DECIMAL(10,4) NOT NULL,
                high DECIMAL(10,4) NOT NULL,
                low DECIMAL(10,4) NOT NULL,
                close DECIMAL(10,4) NOT NULL,
                volume BIGINT NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            
            -- Create hypertable if not exists
            SELECT create_hypertable('minute_bars', 'timestamp', if_not_exists => TRUE);
            
            -- Create index for symbol queries
            CREATE INDEX IF NOT EXISTS idx_minute_bars_symbol_time 
            ON minute_bars (symbol, timestamp DESC);
            """
            
            cursor.execute(create_table_sql)
            
            # Insert data using COPY for performance
            insert_sql = """
            INSERT INTO minute_bars (symbol, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
            """
            
            # Prepare data for insertion
            insert_data = [
                (
                    row['symbol'],
                    row['timestamp'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                )
                for row in data
            ]
            
            cursor.executemany(insert_sql, insert_data)
            conn.commit()
            
            rows_inserted = cursor.rowcount
            logger.info(f"💾 Inserted {rows_inserted} rows into TimescaleDB")
            
            cursor.close()
            conn.close()
            
            return rows_inserted
            
        except Exception as e:
            logger.error(f"❌ Database insertion failed: {e}")
            return 0
    
    def fetch_and_store_historical_data(self, start_date: str, end_date: str) -> Dict[str, int]:
        """Main method to fetch and store historical data"""
        
        logger.info(f"🎯 HISTORICAL DATA FETCH: {start_date} to {end_date}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏱️ Timeframe: 1-minute bars")
        
        results = {}
        
        for i, symbol in enumerate(self.symbols):
            if i > 0:
                logger.info(f"⏳ Rate limiting between symbols: {self.rate_limit_delay}s")
                time.sleep(self.rate_limit_delay)
            
            # Fetch data
            symbol_data = self.fetch_date_range_chunked(symbol, start_date, end_date)
            
            if symbol_data:
                # Store in database
                rows_inserted = self.insert_to_timescaledb(symbol_data)
                results[symbol] = rows_inserted
                logger.info(f"✅ {symbol}: {rows_inserted} rows stored")
            else:
                logger.error(f"❌ {symbol}: No data fetched")
                results[symbol] = 0
        
        return results
    
    def verify_data_coverage(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        """Verify data coverage in TimescaleDB"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            results = {}
            
            for symbol in self.symbols:
                query = """
                SELECT 
                    COUNT(*) as total_bars,
                    MIN(timestamp) as first_bar,
                    MAX(timestamp) as last_bar,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days
                FROM minute_bars 
                WHERE symbol = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
                """
                
                cursor.execute(query, (symbol, start_date, end_date))
                row = cursor.fetchone()
                
                results[symbol] = {
                    'total_bars': row[0],
                    'first_bar': row[1],
                    'last_bar': row[2],
                    'trading_days': row[3]
                }
                
                logger.info(f"📊 {symbol}: {row[0]} bars, {row[3]} trading days")
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Data verification failed: {e}")
            return {}

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Polygon Historical Data Fetcher for V3 Training")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data coverage")
    
    args = parser.parse_args()
    
    logger.info("🎯 POLYGON HISTORICAL FETCHER - INSTITUTIONAL GRADE")
    logger.info("=" * 60)
    
    fetcher = PolygonHistoricalFetcher()
    
    if args.verify_only:
        logger.info("🔍 Verifying existing data coverage...")
        results = fetcher.verify_data_coverage(args.start, args.end)
        
        total_bars = sum(r['total_bars'] for r in results.values())
        logger.info(f"📊 Total bars in database: {total_bars:,}")
        
        return 0
    
    # Fetch and store data
    start_time = time.time()
    
    results = fetcher.fetch_and_store_historical_data(args.start, args.end)
    
    fetch_time = time.time() - start_time
    
    # Summary
    total_rows = sum(results.values())
    logger.info(f"\n🎉 HISTORICAL DATA FETCH COMPLETED!")
    logger.info(f"   ⏱️ Total time: {fetch_time/3600:.1f} hours")
    logger.info(f"   📊 Total rows: {total_rows:,}")
    logger.info(f"   💾 Data stored in TimescaleDB")
    
    # Verify coverage
    logger.info(f"\n🔍 Verifying data coverage...")
    verification = fetcher.verify_data_coverage(args.start, args.end)
    
    for symbol, stats in verification.items():
        logger.info(f"   {symbol}: {stats['total_bars']:,} bars, {stats['trading_days']} days")
    
    return 0

if __name__ == "__main__":
    exit(main())