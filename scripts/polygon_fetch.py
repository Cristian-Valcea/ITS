#!/usr/bin/env python3
"""
Polygon Data Fetching Script - Updated from Alpha Vantage
Fetches NVDA + MSFT daily data from Polygon.io Starter ($29/month)
"""

import os
import sys
import json
import time
import logging
import requests
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our working secrets manager
from secrets_helper import SecretsHelper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """Polygon.io data fetcher for NVDA + MSFT dual-ticker system"""
    
    def __init__(self, output_dir: str = "raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get Polygon API key from secure vault
        try:
            self.api_key = SecretsHelper.get_polygon_api_key()
            logger.info("✅ Polygon API key retrieved from secure vault")
        except Exception as e:
            logger.error(f"❌ Failed to get Polygon API key: {e}")
            logger.info("💡 Use --mock-data to generate test data instead")
            self.api_key = None
        
        # Polygon API configuration
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.rate_limit_delay = 12  # 5 calls/minute = one call every 12 seconds
        
        # Default symbols for dual-ticker system
        self.symbols = ["NVDA", "MSFT"]
    
    def fetch_daily_data(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch daily OHLCV data for a symbol"""
        
        if not self.api_key:
            return None
        
        url = f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}"
        
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 1000  # Polygon Starter allows up to 1000 results
        }
        
        try:
            logger.info(f"📡 Fetching {symbol} data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    bars = data['results']
                    logger.info(f"✅ {symbol}: Retrieved {len(bars)} daily bars")
                    
                    # Convert to standard format
                    formatted_data = []
                    for bar in bars:
                        formatted_data.append({
                            'timestamp': datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            'open': bar['o'],
                            'high': bar['h'],  
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v']
                        })
                    
                    return {
                        'symbol': symbol,
                        'data': formatted_data,
                        'count': len(formatted_data),
                        'source': 'polygon_starter',
                        'timeframe': 'daily'
                    }
                else:
                    logger.warning(f"⚠️ {symbol}: No data returned - {data.get('status', 'unknown status')}")
                    return None
                    
            elif response.status_code == 429:
                logger.error(f"❌ Rate limit exceeded for {symbol}")
                logger.info("⏳ Polygon Starter: 5 calls/minute limit")
                return None
                
            else:
                logger.error(f"❌ HTTP {response.status_code} for {symbol}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} data: {e}")
            return None
    
    def fetch_dual_ticker_data(self, mock_data: bool = False, days_back: int = 30, seed: int = 42) -> Dict[str, any]:
        """Fetch NVDA + MSFT data (main interface matching alpha_vantage_fetch.py)"""
        
        if mock_data or not self.api_key:
            logger.info("🧪 Generating mock data for NVDA + MSFT")
            return self._generate_mock_data(days_back, seed)
        
        # Calculate date range  
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')  # Avoid weekend/delay issues
        start_date = (datetime.now() - timedelta(days=days_back + 7)).strftime('%Y-%m-%d')
        
        results = {
            'fetch_time': datetime.now().isoformat(),
            'date_range': {'start': start_date, 'end': end_date},
            'source': 'polygon_starter_api',
            'symbols': {}
        }
        
        # Fetch each symbol with rate limiting
        for i, symbol in enumerate(self.symbols):
            if i > 0:  # Rate limiting between calls
                logger.info(f"⏳ Rate limiting: waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
            
            symbol_data = self.fetch_daily_data(symbol, start_date, end_date)
            
            if symbol_data:
                results['symbols'][symbol] = symbol_data
                logger.info(f"✅ {symbol}: {symbol_data['count']} bars retrieved")
            else:
                logger.error(f"❌ {symbol}: Failed to fetch data")
                results['symbols'][symbol] = {'error': 'fetch_failed'}
        
        return results
    
    def _generate_mock_data(self, days_back: int = 30, seed: int = 42) -> Dict[str, any]:
        """Generate realistic mock data for testing with deterministic RNG"""
        
        # Set deterministic seed for reproducible mock data
        np.random.seed(seed)
        
        logger.info(f"🧪 Generating {days_back} days of mock data (seed={seed})")
        
        # Realistic starting prices
        prices = {"NVDA": 170.0, "MSFT": 510.0}
        
        results = {
            'fetch_time': datetime.now().isoformat(),
            'date_range': {
                'start': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d')
            },
            'source': 'mock_data_generator',
            'symbols': {}
        }
        
        for symbol in self.symbols:
            base_price = prices[symbol]
            data = []
            
            for i in range(days_back):
                date = datetime.now() - timedelta(days=days_back - i)
                
                # Skip weekends
                if date.weekday() >= 5:
                    continue
                
                # Generate realistic OHLCV
                daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
                
                open_price = base_price * (1 + daily_change)
                high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
                close_price = open_price + np.random.normal(0, open_price * 0.015)
                
                # Ensure OHLC relationships
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = int(np.random.normal(50000000 if symbol == "NVDA" else 20000000, 10000000))
                volume = max(volume, 1000000)  # Minimum volume
                
                data.append({
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
                
                base_price = close_price  # Next day starts from previous close
            
            results['symbols'][symbol] = {
                'symbol': symbol,
                'data': data,
                'count': len(data),
                'source': 'mock_generator',
                'timeframe': 'daily'
            }
            
            logger.info(f"🧪 {symbol}: Generated {len(data)} mock bars")
        
        return results
    
    def save_data(self, data: Dict, format: str = 'json') -> List[str]:
        """Save fetched data to files"""
        
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            # Save complete dataset
            filename = f"polygon_dual_ticker_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            saved_files.append(str(filepath))
            logger.info(f"💾 Saved complete dataset: {filename}")
            
            # Save individual symbol files
            for symbol in data.get('symbols', {}):
                symbol_data = data['symbols'][symbol]
                if 'data' in symbol_data:
                    symbol_filename = f"polygon_{symbol.lower()}_{timestamp}.json"
                    symbol_filepath = self.output_dir / symbol_filename
                    
                    with open(symbol_filepath, 'w') as f:
                        json.dump(symbol_data, f, indent=2)
                    
                    saved_files.append(str(symbol_filepath))
                    logger.info(f"💾 Saved {symbol} data: {symbol_filename}")
        
        elif format.lower() == 'csv':
            # Save as CSV files
            for symbol in data.get('symbols', {}):
                symbol_data = data['symbols'][symbol]
                if 'data' in symbol_data:
                    df = pd.DataFrame(symbol_data['data'])
                    filename = f"polygon_{symbol.lower()}_{timestamp}.csv"
                    filepath = self.output_dir / filename
                    
                    df.to_csv(filepath, index=False)
                    saved_files.append(str(filepath))
                    logger.info(f"💾 Saved {symbol} CSV: {filename}")
        
        return saved_files
    
    def validate_data_quality(self, data: Dict) -> Dict[str, any]:
        """Basic data quality validation"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_checked': 0,
            'total_bars': 0,
            'quality_issues': []
        }
        
        for symbol in data.get('symbols', {}):
            symbol_data = data['symbols'][symbol]
            
            if 'data' not in symbol_data:
                validation_results['quality_issues'].append(f"{symbol}: No data available")
                continue
            
            validation_results['symbols_checked'] += 1
            bars = symbol_data['data']
            validation_results['total_bars'] += len(bars)
            
            # Check for basic issues
            if len(bars) == 0:
                validation_results['quality_issues'].append(f"{symbol}: Empty dataset")
                continue
            
            # Check OHLC relationships
            invalid_ohlc = 0
            for bar in bars:
                if not (bar['low'] <= bar['open'] <= bar['high'] and 
                       bar['low'] <= bar['close'] <= bar['high']):
                    invalid_ohlc += 1
            
            if invalid_ohlc > 0:
                validation_results['quality_issues'].append(
                    f"{symbol}: {invalid_ohlc} bars with invalid OHLC relationships"
                )
            
            logger.info(f"✅ {symbol}: {len(bars)} bars validated")
        
        return validation_results

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Polygon Data Fetcher for Dual-Ticker System")
    parser.add_argument("--mock-data", action="store_true", help="Generate mock data instead of API calls")
    parser.add_argument("--output-dir", default="raw", help="Output directory (default: raw)")
    parser.add_argument("--format", choices=['json', 'csv'], default='json', help="Output format")
    parser.add_argument("--days-back", type=int, default=30, help="Days of historical data (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic mock data generation (default: 42)")
    parser.add_argument("--validate", action="store_true", help="Run data quality validation")
    
    args = parser.parse_args()
    
    logger.info("🚀 Polygon Data Fetcher Starting")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📊 Format: {args.format}")
    logger.info(f"📅 Days back: {args.days_back}")
    
    # Initialize fetcher
    fetcher = PolygonDataFetcher(args.output_dir)
    
    # Fetch data
    data = fetcher.fetch_dual_ticker_data(
        mock_data=args.mock_data,
        days_back=args.days_back,
        seed=args.seed
    )
    
    if not data or not data.get('symbols'):
        logger.error("❌ No data retrieved")
        return 1
    
    # Validate if requested
    if args.validate:
        logger.info("🔍 Running data quality validation...")
        validation = fetcher.validate_data_quality(data)
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   - Symbols checked: {validation['symbols_checked']}")
        logger.info(f"   - Total bars: {validation['total_bars']}")
        
        if validation['quality_issues']:
            logger.warning("⚠️ Quality issues found:")
            for issue in validation['quality_issues']:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ No quality issues detected")
    
    # Save data
    saved_files = fetcher.save_data(data, args.format)
    
    logger.info("🎉 Data fetching completed successfully!")
    logger.info(f"📁 Files saved: {len(saved_files)}")
    for file in saved_files:
        logger.info(f"   - {file}")
    
    return 0

if __name__ == "__main__":
    exit(main())