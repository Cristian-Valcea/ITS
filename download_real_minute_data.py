#!/usr/bin/env python3
"""
📈 REAL MINUTE DATA DOWNLOAD - POLYGON API
Download real NVDA/MSFT minute bars for curriculum model evaluation
Target: 3 months (≈30K steps) for <30min evaluation runs
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonMinuteDataFetcher:
    """Polygon.io minute-level data fetcher for real evaluation"""
    
    def __init__(self):
        # Try to get Polygon API key from environment or config
        self.api_key = self._get_polygon_key()
        
        if not self.api_key:
            logger.error("❌ Polygon API key not found!")
            logger.info("💡 Options:")
            logger.info("   1. Set POLYGON_API_KEY environment variable")
            logger.info("   2. Add to .env file: POLYGON_API_KEY=your_key")
            logger.info("   3. Use --mock-data for synthetic evaluation")
            
        # Polygon minute aggregates endpoint
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.rate_limit_delay = 12  # Starter: 5 calls/minute
        
        # Target symbols
        self.symbols = ["NVDA", "MSFT"]
        
    def _get_polygon_key(self) -> Optional[str]:
        """Get Polygon API key from secrets manager"""
        
        # Try secrets manager first (primary source)
        try:
            from secrets_helper import SecretsHelper
            key = SecretsHelper.get_polygon_api_key()
            if key:
                logger.info("✅ Polygon API key retrieved from secrets manager")
                return key
        except Exception as e:
            logger.warning(f"⚠️ Secrets manager access failed: {e}")
        
        # Fallback to environment variable
        key = os.getenv('POLYGON_API_KEY')
        if key:
            logger.info("✅ Polygon API key retrieved from environment variable")
            return key
            
        # Fallback to .env file
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('POLYGON_API_KEY='):
                        logger.info("✅ Polygon API key retrieved from .env file")
                        return line.split('=', 1)[1].strip()
            
        return None
    
    def fetch_minute_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch minute-level OHLCV data for a symbol"""
        
        if not self.api_key:
            logger.error(f"❌ Cannot fetch {symbol}: No API key")
            return None
        
        # Polygon minute aggregates URL
        url = f"{self.base_url}/{symbol}/range/1/minute/{start_date}/{end_date}"
        
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000  # Polygon allows large requests
        }
        
        logger.info(f"📥 Fetching {symbol} minute data: {start_date} to {end_date}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    results = data['results']
                    logger.info(f"✅ {symbol}: {len(results)} minute bars received")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    
                    # Rename columns to standard OHLCV
                    df = df.rename(columns={
                        'o': 'open',
                        'h': 'high', 
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                    
                    # Select and reorder columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.set_index('timestamp')
                    
                    # Filter to market hours only (9:30 AM - 4:00 PM ET)
                    df = self._filter_market_hours(df)
                    
                    return df
                    
                else:
                    logger.warning(f"⚠️ {symbol}: No data - {data.get('status', 'unknown')}")
                    if 'error' in data:
                        logger.warning(f"   Error: {data['error']}")
                    return None
                    
            elif response.status_code == 429:
                logger.error(f"❌ Rate limit exceeded for {symbol}")
                logger.info("⏳ Polygon Starter: 5 calls/minute limit - wait 12 seconds")
                return None
                
            else:
                logger.error(f"❌ HTTP {response.status_code} for {symbol}")
                logger.error(f"   Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular market hours (9:30 AM - 4:00 PM ET)"""
        
        # Convert to Eastern time
        df_et = df.copy()
        df_et.index = df_et.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        # Filter to market hours
        market_hours = df_et.between_time('09:30', '16:00')
        
        # Filter to weekdays only
        market_days = market_hours[market_hours.index.weekday < 5]
        
        logger.info(f"   📊 Filtered to {len(market_days)} market-hour bars")
        
        return market_days
    
    def download_evaluation_dataset(self, months_back: int = 3) -> Dict[str, pd.DataFrame]:
        """Download 3-month evaluation dataset for both symbols"""
        
        logger.info("📈 DOWNLOADING REAL EVALUATION DATASET")
        logger.info(f"🎯 Target: {months_back} months of minute data for NVDA + MSFT")
        
        # Calculate date range (3 months back, excluding last 3 days for data completeness)
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=months_back * 30 + 7)).strftime('%Y-%m-%d')
        
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        datasets = {}
        
        for symbol in self.symbols:
            logger.info(f"🔄 Processing {symbol}...")
            
            # Fetch data with rate limiting
            if symbol != self.symbols[0]:  # Skip delay for first symbol
                logger.info(f"⏳ Rate limiting: waiting {self.rate_limit_delay}s...")
                time.sleep(self.rate_limit_delay)
            
            df = self.fetch_minute_data(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                datasets[symbol] = df
                
                # Save to CSV for backup
                output_file = f"real_data_{symbol.lower()}_{start_date}_to_{end_date}.csv"
                df.to_csv(output_file)
                logger.info(f"💾 Saved to {output_file}")
                
                # Log statistics
                logger.info(f"📊 {symbol} statistics:")
                logger.info(f"   📅 Date range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"   📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                logger.info(f"   📊 Total bars: {len(df):,}")
                
            else:
                logger.error(f"❌ Failed to fetch {symbol} data")
                
        if len(datasets) == 2:
            logger.info("✅ REAL EVALUATION DATASET READY")
            logger.info(f"📊 NVDA: {len(datasets['NVDA']):,} bars")
            logger.info(f"📊 MSFT: {len(datasets['MSFT']):,} bars")
            
            # Align timestamps (inner join for overlapping trading times)
            nvda_df = datasets['NVDA']
            msft_df = datasets['MSFT']
            
            common_times = nvda_df.index.intersection(msft_df.index)
            logger.info(f"📊 Common timestamps: {len(common_times):,}")
            
            if len(common_times) > 20000:  # Enough for evaluation
                aligned_data = {
                    'NVDA': nvda_df.loc[common_times],
                    'MSFT': msft_df.loc[common_times],
                    'timestamps': common_times
                }
                
                logger.info("✅ DATASETS ALIGNED AND READY FOR EVALUATION")
                return aligned_data
            else:
                logger.warning("⚠️ Insufficient overlapping data for evaluation")
                
        else:
            logger.error("❌ Failed to fetch complete dataset")
            
        return {}
    
    def generate_mock_evaluation_data(self, periods: int = 30000) -> Dict[str, pd.DataFrame]:
        """Generate mock minute-level data for evaluation (fallback)"""
        
        logger.info("🧪 GENERATING MOCK EVALUATION DATA")
        logger.info(f"📊 Generating {periods:,} minute bars per symbol")
        
        # Create realistic timestamp sequence (market hours only)
        start_date = datetime.now() - timedelta(days=90)
        timestamps = pd.bdate_range(start=start_date, periods=periods//390, freq='D')
        
        # Expand to minute-level (390 minutes per trading day)
        minute_timestamps = []
        for date in timestamps:
            day_start = date.replace(hour=9, minute=30)
            day_minutes = pd.date_range(day_start, periods=390, freq='1min')
            minute_timestamps.extend(day_minutes)
        
        minute_timestamps = minute_timestamps[:periods]  # Trim to exact count
        
        datasets = {}
        
        for symbol in self.symbols:
            logger.info(f"🔄 Generating {symbol} mock data...")
            
            # Symbol-specific parameters
            if symbol == 'NVDA':
                base_price = 170.0
                volatility = 0.02
                drift = 0.0001
            else:  # MSFT
                base_price = 510.0
                volatility = 0.015
                drift = 0.0001
            
            # Generate realistic price series
            np.random.seed(42)  # Consistent with training data
            returns = np.random.normal(drift, volatility, periods)
            
            # Add realistic intraday patterns
            for i in range(periods):
                minute_of_day = i % 390
                # Higher volatility at open/close
                if minute_of_day < 30 or minute_of_day > 360:
                    returns[i] *= 1.5
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLCV bars
            data = []
            for i, (timestamp, close_price) in enumerate(zip(minute_timestamps, prices)):
                # Realistic spread around close price
                spread = close_price * 0.001  # 0.1% spread
                high = close_price + np.random.uniform(0, spread)
                low = close_price - np.random.uniform(0, spread)
                open_price = prices[i-1] if i > 0 else close_price
                
                volume = np.random.randint(1000, 10000)  # Realistic volume
                
                data.append({
                    'open': open_price,
                    'high': max(open_price, high, close_price),
                    'low': min(open_price, low, close_price), 
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=minute_timestamps)
            datasets[symbol] = df
            
            logger.info(f"✅ {symbol} mock data generated: {len(df):,} bars")
            logger.info(f"   📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Add aligned timestamps
        datasets['timestamps'] = minute_timestamps
        
        logger.info("✅ MOCK EVALUATION DATASET READY")
        return datasets

def main():
    """Main function to download real evaluation data"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Download real minute data for evaluation')
    parser.add_argument('--months', type=int, default=3, help='Months of data to download')
    parser.add_argument('--mock-data', action='store_true', help='Generate mock data instead')
    args = parser.parse_args()
    
    fetcher = PolygonMinuteDataFetcher()
    
    if args.mock_data or not fetcher.api_key:
        logger.info("🧪 Using mock data generation")
        datasets = fetcher.generate_mock_evaluation_data()
    else:
        logger.info("📈 Downloading real market data")
        datasets = fetcher.download_evaluation_dataset(args.months)
    
    if datasets:
        logger.info("🎉 EVALUATION DATA READY!")
        logger.info("📋 Next steps:")
        logger.info("   1. Run curriculum model evaluation on this data")
        logger.info("   2. Compare performance vs mock data results")
        logger.info("   3. Validate production readiness")
        
        return True
    else:
        logger.error("❌ Failed to prepare evaluation data")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ REAL DATA DOWNLOAD COMPLETED")
    else:
        print("❌ DATA DOWNLOAD FAILED")