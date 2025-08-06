#!/usr/bin/env python3
"""
Upload MSFT Historical Data to Database
Fetch MSFT data from 2022-2024 and upload directly to secure database
"""

import os
import sys
import json
import time
import logging
import requests
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MSFTDataUploader:
    """Upload MSFT data directly to secure database"""
    
    def __init__(self):
        # Get secure credentials from vault
        try:
            self.api_key = SecretsHelper.get_polygon_api_key()
            self.db_password = SecretsHelper.get_timescaledb_password()
            logger.info("✅ Retrieved secure credentials from vault")
        except Exception as e:
            logger.error(f"❌ Failed to get credentials: {e}")
            raise
        
        # API configuration
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.rate_limit_delay = 12  # 5 calls/minute for Starter plan
        
        # Database configuration with secure password
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': self.db_password
        }
    
    def fetch_msft_chunk(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch MSFT minute data for date range"""
        
        url = f"{self.base_url}/MSFT/range/1/minute/{start_date}/{end_date}"
        
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            logger.info(f"📡 Fetching MSFT data: {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    bars = data['results']
                    
                    # Convert to standard format (market hours only)
                    processed_bars = []
                    for bar in bars:
                        timestamp = datetime.fromtimestamp(bar['t'] / 1000)
                        
                        # Only include market hours (9:30 AM - 4:00 PM ET)
                        hour_minute = timestamp.hour + timestamp.minute/60
                        if 9.5 <= hour_minute <= 16:
                            processed_bars.append({
                                'symbol': 'MSFT',
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'open': float(bar['o']),
                                'high': float(bar['h']),
                                'low': float(bar['l']),
                                'close': float(bar['c']),
                                'volume': int(bar['v'])
                            })
                    
                    logger.info(f"✅ MSFT: Retrieved {len(bars)} bars, {len(processed_bars)} market-hours bars")
                    return processed_bars
                else:
                    logger.error(f"❌ API Error: {data}")
                    return []
            else:
                logger.error(f"❌ HTTP Error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Fetch error: {e}")
            return []
    
    def upload_to_database(self, data: List[Dict]) -> bool:
        """Upload data directly to secure database"""
        
        if not data:
            logger.warning("⚠️ No data to upload")
            return False
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Prepare insert statement
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
            
            # Bulk insert
            cur.executemany(insert_sql, insert_data)
            conn.commit()
            
            rows_inserted = cur.rowcount
            logger.info(f"💾 Uploaded {rows_inserted} MSFT bars to database")
            
            cur.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database upload failed: {e}")
            return False
    
    def upload_full_msft_history(self):
        """Upload complete MSFT history from 2022-2024"""
        
        logger.info("🚀 MSFT HISTORICAL DATA UPLOAD")
        logger.info("=" * 50)
        
        # Date ranges for chunked fetching (30-day chunks)
        start_date = datetime(2022, 1, 3)
        end_date = datetime(2024, 12, 31)
        
        current_date = start_date
        total_uploaded = 0
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=30), end_date)
            
            start_str = current_date.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')
            
            # Fetch chunk
            chunk_data = self.fetch_msft_chunk(start_str, end_str)
            
            if chunk_data:
                # Upload to database
                success = self.upload_to_database(chunk_data)
                if success:
                    total_uploaded += len(chunk_data)
                    logger.info(f"✅ Uploaded chunk: {len(chunk_data)} bars (Total: {total_uploaded})")
                else:
                    logger.error(f"❌ Failed to upload chunk {start_str} to {end_str}")
                    return False
            
            # Rate limiting
            if current_date < end_date:
                logger.info(f"⏳ Rate limiting: waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
            
            current_date = chunk_end + timedelta(days=1)
        
        logger.info(f"🎉 MSFT upload complete! Total bars uploaded: {total_uploaded}")
        return True

def main():
    """Main execution"""
    
    uploader = MSFTDataUploader()
    
    try:
        success = uploader.upload_full_msft_history()
        
        if success:
            # Verify upload and show final dual-ticker status
            conn = psycopg2.connect(**uploader.db_config)
            cur = conn.cursor()
            
            # Get counts for both symbols
            cur.execute("""
                SELECT symbol, COUNT(*) as count, MIN(timestamp), MAX(timestamp)
                FROM minute_bars 
                WHERE symbol IN ('NVDA', 'MSFT')
                GROUP BY symbol
                ORDER BY symbol;
            """)
            results = cur.fetchall()
            
            logger.info(f"📊 Final dual-ticker database status:")
            total_records = 0
            for row in results:
                logger.info(f"  {row[0]}: {row[1]:,} records from {row[2]} to {row[3]}")
                total_records += row[1]
            
            logger.info(f"  📈 Total dual-ticker records: {total_records:,}")
            logger.info(f"  🎯 Ready for 48K training: {'✅ YES' if total_records > 50000 else '❌ NO'}")
            
            conn.close()
            
            print("✅ MSFT DATA UPLOAD: SUCCESS")
            print(f"✅ DUAL-TICKER CACHE: {total_records:,} total records")
        else:
            print("❌ MSFT DATA UPLOAD: FAILED")
            
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        print("❌ MSFT DATA UPLOAD: FAILED")

if __name__ == "__main__":
    main()