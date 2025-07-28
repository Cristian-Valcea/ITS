#!/usr/bin/env python3
"""
Alpha Vantage Data Fetching Prototype
Fetches NVDA + MSFT 1-minute data and saves to raw/
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpha_vantage_client import AlphaVantageClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionPrototype:
    """Prototype for NVDA + MSFT data ingestion"""
    
    def __init__(self, output_dir: str = "raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Alpha Vantage client
        try:
            self.client = AlphaVantageClient()
            logger.info("✅ Alpha Vantage client initialized successfully")
        except ValueError as e:
            logger.error(f"❌ Alpha Vantage client failed: {e}")
            logger.info("💡 Set ALPHA_VANTAGE_KEY environment variable or use --mock-data")
            self.client = None
    
    def fetch_dual_ticker_data(self, mock_data: bool = False) -> Dict[str, any]:
        """Fetch NVDA + MSFT 1-minute data"""
        
        if mock_data or not self.client:
            logger.info("🧪 Generating mock data for NVDA + MSFT")
            return self._generate_mock_data()
        
        logger.info("📡 Fetching live data from Alpha Vantage...")
        
        try:
            # Fetch dual-ticker quotes (fast)
            quotes = self.client.get_dual_ticker_quotes()
            logger.info(f"✅ Quotes fetched: {len(quotes)} symbols")
            
            # Fetch dual-ticker bars (slower due to rate limiting)
            bars = self.client.get_dual_ticker_bars('1min')
            logger.info(f"✅ Bars fetched: {len(bars)} symbols")
            
            return {
                'quotes': quotes,
                'bars': bars,
                'timestamp': datetime.now().isoformat(),
                'source': 'alpha_vantage_live'
            }
            
        except Exception as e:
            logger.error(f"❌ Live data fetch failed: {e}")
            logger.info("🧪 Falling back to mock data")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, any]:
        """Generate realistic mock data for testing"""
        import pandas as pd
        import numpy as np
        
        # Generate timestamps for today's trading hours (9:30 AM - 4:00 PM ET)
        base_date = datetime.now().date()
        start_time = datetime.combine(base_date, datetime.strptime("09:30", "%H:%M").time())
        
        # Generate 390 minutes of 1-minute bars (6.5 hours)
        timestamps = [start_time + timedelta(minutes=i) for i in range(390)]
        
        symbols_data = {}
        
        for symbol in ['NVDA', 'MSFT']:
            # Base prices (realistic values)
            base_price = 485.0 if symbol == 'NVDA' else 415.0
            
            # Generate realistic OHLCV data
            prices = []
            current_price = base_price
            
            for i in range(390):
                # Small random walk
                change = np.random.normal(0, 0.002)  # 0.2% volatility per minute
                current_price *= (1 + change)
                
                # Generate OHLC around current price
                high = current_price * (1 + abs(np.random.normal(0, 0.001)))
                low = current_price * (1 - abs(np.random.normal(0, 0.001)))
                open_price = prices[-1]['close'] if prices else current_price
                close_price = current_price
                
                # Ensure OHLC relationships
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                # Volume (realistic for 1-minute bars)
                volume = int(np.random.lognormal(8, 1))  # ~3000 average volume
                
                prices.append({
                    'timestamp': timestamps[i].isoformat(),
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            # Current quote
            last_price = prices[-1]['close']
            quote = {
                'symbol': symbol,
                'price': last_price,
                'change': round(last_price - base_price, 2),
                'change_percent': round((last_price - base_price) / base_price * 100, 2),
                'volume': sum(p['volume'] for p in prices),
                'timestamp': timestamps[-1].isoformat()
            }
            
            symbols_data[symbol] = {
                'quote': quote,
                'bars': prices
            }
        
        return {
            'quotes': {symbol: data['quote'] for symbol, data in symbols_data.items()},
            'bars': {symbol: data['bars'] for symbol, data in symbols_data.items()},
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_data_generator'
        }
    
    def save_data(self, data: Dict, filename_prefix: str = None) -> Tuple[str, str]:
        """Save fetched data to files"""
        
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"dual_ticker_{timestamp}"
        
        # Save as JSON
        json_file = self.output_dir / f"{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for easy inspection
        csv_file = self.output_dir / f"{filename_prefix}.csv"
        
        # Flatten bars data for CSV
        csv_rows = []
        for symbol, bars in data['bars'].items():
            for bar in bars:
                row = {'symbol': symbol, **bar}
                csv_rows.append(row)
        
        import pandas as pd
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"✅ Data saved:")
        logger.info(f"  📄 JSON: {json_file}")
        logger.info(f"  📊 CSV: {csv_file}")
        
        return str(json_file), str(csv_file)
    
    def validate_data_quality(self, data: Dict) -> Dict[str, any]:
        """Basic data quality validation"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {}
        }
        
        # Check 1: Both symbols present
        expected_symbols = {'NVDA', 'MSFT'}
        actual_symbols = set(data['bars'].keys())
        symbols_check = expected_symbols == actual_symbols
        
        validation_results['checks']['symbols_present'] = {
            'status': 'PASS' if symbols_check else 'FAIL',
            'expected': list(expected_symbols),
            'actual': list(actual_symbols)
        }
        
        # Check 2: Data completeness
        completeness_results = {}
        for symbol in expected_symbols:
            if symbol in data['bars']:
                bar_count = len(data['bars'][symbol])
                expected_count = 390  # Full trading day
                completeness_pct = bar_count / expected_count
                
                completeness_results[symbol] = {
                    'bar_count': bar_count,
                    'expected_count': expected_count,
                    'completeness_pct': completeness_pct,
                    'status': 'PASS' if completeness_pct >= 0.95 else 'FAIL'
                }
        
        validation_results['checks']['data_completeness'] = completeness_results
        
        # Check 3: OHLC relationships
        ohlc_results = {}
        for symbol in expected_symbols:
            if symbol in data['bars']:
                bars = data['bars'][symbol]
                ohlc_violations = 0
                
                for bar in bars:
                    # Check: high >= low, high >= open, high >= close, low <= open, low <= close
                    if not (bar['high'] >= bar['low'] and 
                           bar['high'] >= bar['open'] and 
                           bar['high'] >= bar['close'] and
                           bar['low'] <= bar['open'] and 
                           bar['low'] <= bar['close']):
                        ohlc_violations += 1
                
                ohlc_results[symbol] = {
                    'total_bars': len(bars),
                    'violations': ohlc_violations,
                    'violation_pct': ohlc_violations / len(bars) if bars else 0,
                    'status': 'PASS' if ohlc_violations == 0 else 'FAIL'
                }
        
        validation_results['checks']['ohlc_relationships'] = ohlc_results
        
        # Overall status
        all_checks = []
        all_checks.append(validation_results['checks']['symbols_present']['status'])
        
        for symbol_result in validation_results['checks']['data_completeness'].values():
            all_checks.append(symbol_result['status'])
        
        for symbol_result in validation_results['checks']['ohlc_relationships'].values():
            all_checks.append(symbol_result['status'])
        
        if all(check == 'PASS' for check in all_checks):
            validation_results['overall_status'] = 'PASS'
        else:
            validation_results['overall_status'] = 'FAIL'
        
        return validation_results
    
    def run_ingestion(self, mock_data: bool = False, validate: bool = True) -> Dict[str, any]:
        """Run complete data ingestion process"""
        
        logger.info("🚀 Starting dual-ticker data ingestion prototype")
        start_time = time.time()
        
        # Fetch data
        data = self.fetch_dual_ticker_data(mock_data=mock_data)
        
        # Save data
        json_file, csv_file = self.save_data(data)
        
        # Validate data quality
        validation_results = None
        if validate:
            validation_results = self.validate_data_quality(data)
            
            # Save validation results
            validation_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"✅ Validation results: {validation_results['overall_status']}")
            logger.info(f"  📋 Report: {validation_file}")
        
        elapsed_time = time.time() - start_time
        
        # Summary
        summary = {
            'status': 'SUCCESS',
            'elapsed_time_seconds': elapsed_time,
            'data_source': data['source'],
            'files_created': {
                'json': json_file,
                'csv': csv_file,
                'validation': str(validation_file) if validate else None
            },
            'data_summary': {
                'symbols': list(data['bars'].keys()),
                'total_bars': sum(len(bars) for bars in data['bars'].values()),
                'timestamp_range': {
                    'start': min(min(bar['timestamp'] for bar in bars) for bars in data['bars'].values()),
                    'end': max(max(bar['timestamp'] for bar in bars) for bars in data['bars'].values())
                }
            },
            'validation_status': validation_results['overall_status'] if validate else 'SKIPPED'
        }
        
        logger.info(f"🎉 Data ingestion complete!")
        logger.info(f"  ⏱️  Time: {elapsed_time:.2f} seconds")
        logger.info(f"  📊 Bars: {summary['data_summary']['total_bars']}")
        logger.info(f"  ✅ Status: {summary['validation_status']}")
        
        return summary

def main():
    """CLI interface for data ingestion prototype"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha Vantage Data Ingestion Prototype")
    parser.add_argument('--mock-data', action='store_true', 
                       help='Use mock data instead of live Alpha Vantage API')
    parser.add_argument('--output-dir', default='raw',
                       help='Output directory for raw data files')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip data quality validation')
    
    args = parser.parse_args()
    
    # Create ingestion prototype
    ingester = DataIngestionPrototype(output_dir=args.output_dir)
    
    try:
        # Run ingestion
        summary = ingester.run_ingestion(
            mock_data=args.mock_data,
            validate=not args.no_validate
        )
        
        # Print summary
        print("\n📊 INGESTION SUMMARY")
        print("=" * 40)
        print(f"Status: {summary['status']}")
        print(f"Data Source: {summary['data_source']}")
        print(f"Total Bars: {summary['data_summary']['total_bars']}")
        print(f"Validation: {summary['validation_status']}")
        print(f"Time: {summary['elapsed_time_seconds']:.2f}s")
        print(f"\nFiles created:")
        for file_type, file_path in summary['files_created'].items():
            if file_path:
                print(f"  {file_type}: {file_path}")
        
        # Exit code based on validation
        if summary['validation_status'] == 'PASS':
            sys.exit(0)
        elif summary['validation_status'] == 'FAIL':
            sys.exit(1)
        else:
            sys.exit(0)  # SKIPPED validation
            
    except Exception as e:
        logger.error(f"💥 Ingestion failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()