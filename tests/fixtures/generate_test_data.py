#!/usr/bin/env python3
"""
Generate test fixtures for dual-ticker CI pipeline
Creates 5 rows per symbol (NVDA + MSFT) for fast CI execution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def create_ohlc_bar(open_price, close_price, volatility=0.02):
    """Create valid OHLC bar ensuring high >= max(open,close) and low <= min(open,close)"""
    high = max(open_price, close_price) * (1 + volatility)
    low = min(open_price, close_price) * (1 - volatility)
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price
    }

def create_dual_ticker_test_fixtures():
    """Generate NVDA + MSFT test data (5 rows per symbol)"""
    
    # Create fixtures directory
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)
    
    # Generate 5 rows per symbol (fast CI execution)
    timestamps = pd.date_range(
        '2025-01-27 09:30:00', 
        periods=5, 
        freq='5min', 
        tz='UTC'
    )
    
    # NVDA test data (using realistic NVDA price levels)
    nvda_opens = [875.0, 876.5, 878.2, 877.0, 879.8]
    nvda_closes = [876.5, 878.2, 877.0, 879.8, 881.2]
    
    nvda_ohlc_data = []
    for open_price, close_price in zip(nvda_opens, nvda_closes):
        nvda_ohlc_data.append(create_ohlc_bar(open_price, close_price, volatility=0.015))
    
    nvda_data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'NVDA',
        'open': [bar['open'] for bar in nvda_ohlc_data],
        'high': [bar['high'] for bar in nvda_ohlc_data],
        'low': [bar['low'] for bar in nvda_ohlc_data],
        'close': [bar['close'] for bar in nvda_ohlc_data],
        'volume': [2500, 2650, 2800, 2600, 2750],
        'rsi': [58.0, 60.0, 62.0, 59.0, 61.0],
        'ema_short': [876.0, 877.2, 877.8, 878.5, 879.8],
        'ema_long': [874.5, 875.0, 875.8, 876.5, 877.2],
        'vwap': [875.8, 877.1, 878.5, 878.0, 880.2]
    })
    
    # MSFT test data
    msft_opens = [300.0, 300.5, 301.2, 301.0, 301.8]
    msft_closes = [300.5, 301.2, 301.0, 301.8, 302.0]
    
    msft_ohlc_data = []
    for open_price, close_price in zip(msft_opens, msft_closes):
        msft_ohlc_data.append(create_ohlc_bar(open_price, close_price, volatility=0.012))
    
    msft_data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'MSFT',
        'open': [bar['open'] for bar in msft_ohlc_data],
        'high': [bar['high'] for bar in msft_ohlc_data],
        'low': [bar['low'] for bar in msft_ohlc_data],
        'close': [bar['close'] for bar in msft_ohlc_data],
        'volume': [800, 850, 900, 825, 875],
        'rsi': [52.0, 54.0, 56.0, 53.0, 55.0],
        'ema_short': [300.2, 300.7, 301.0, 301.3, 301.7],
        'ema_long': [299.8, 300.0, 300.2, 300.4, 300.6],
        'vwap': [300.1, 300.6, 301.1, 301.2, 301.8]
    })
    
    # Add time features (cyclical encoding)
    for df in [nvda_data, msft_data]:
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.minute / 60)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Combine datasets
    combined_data = pd.concat([nvda_data, msft_data], ignore_index=True)
    
    # Save in multiple formats for flexibility
    combined_data.to_parquet(fixtures_dir / 'dual_ticker_sample.parquet', index=False)
    combined_data.to_csv(fixtures_dir / 'dual_ticker_sample.csv', index=False)
    
    # Save individual symbol data
    nvda_data.to_parquet(fixtures_dir / 'nvda_sample.parquet', index=False)
    msft_data.to_parquet(fixtures_dir / 'msft_sample.parquet', index=False)
    
    # Create metadata file
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_rows': len(combined_data),
        'nvda_rows': len(nvda_data),
        'msft_rows': len(msft_data),
        'time_range': {
            'start': timestamps[0].isoformat(),
            'end': timestamps[-1].isoformat(),
            'frequency': '5min'
        },
        'columns': list(combined_data.columns),
        'purpose': 'CI smoke tests - fast execution'
    }
    
    import json
    with open(fixtures_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'✅ Generated test fixtures:')
    print(f'  📊 Total rows: {len(combined_data)} ({len(nvda_data)} NVDA + {len(msft_data)} MSFT)')
    print(f'  📁 Files created:')
    print(f'    - dual_ticker_sample.parquet')
    print(f'    - dual_ticker_sample.csv') 
    print(f'    - nvda_sample.parquet')
    print(f'    - msft_sample.parquet')
    print(f'    - metadata.json')
    print(f'  🎯 Ready for CI smoke tests!')
    
    return combined_data

if __name__ == '__main__':
    create_dual_ticker_test_fixtures()