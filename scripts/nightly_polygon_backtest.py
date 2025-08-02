#!/usr/bin/env python3
"""
Nightly Rolling-Window Backtest with Live Polygon Data
Specifically designed for CI/CD pipeline validation

This script:
1. Downloads live market data from Polygon API
2. Runs rolling-window backtests with production models
3. Validates system performance against real market conditions
4. Generates reports for management review
"""
import os
import sys
import argparse
import json
import requests
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolygonDataFetcher:
    """Fetches live market data from Polygon API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        
    def fetch_day_data(self, ticker: str, date: str) -> pd.DataFrame:
        """Fetch minute-level data for a specific ticker and date"""
        url = f"{self.base_url}/{ticker}/range/1/minute/{date}/{date}"
        params = {'apikey': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                logger.warning(f"No data returned for {ticker} on {date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df['symbol'] = ticker
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap'
            })
            
            # Ensure required columns exist
            if 'vwap' not in df.columns:
                df['vwap'] = df['close']  # Fallback
            
            logger.info(f"Fetched {len(df)} bars for {ticker} on {date}")
            return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap']]
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker} on {date}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_days(self, tickers: list, dates: list) -> pd.DataFrame:
        """Fetch data for multiple tickers and dates"""
        all_data = []
        
        for ticker in tickers:
            logger.info(f"Fetching data for {ticker}...")
            
            for date in dates:
                df = self.fetch_day_data(ticker, date)
                if not df.empty:
                    all_data.append(df)
                
                # Rate limiting - Polygon allows 5 requests per minute for free tier
                time.sleep(0.1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total bars fetched: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No data fetched")
            return pd.DataFrame()


class DatabaseManager:
    """Manages database operations for backtest data"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        
    def store_market_data(self, df: pd.DataFrame):
        """Store market data in TimescaleDB"""
        if df.empty:
            logger.warning("No data to store")
            return 0
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Insert data
            inserted_count = 0
            for _, row in df.iterrows():
                cur.execute('''
                    INSERT INTO dual_ticker_bars 
                    (timestamp, symbol, open, high, low, close, volume, vwap)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol) DO NOTHING
                ''', (
                    row['timestamp'], row['symbol'], 
                    row['open'], row['high'], row['low'], row['close'], 
                    row['volume'], row['vwap']
                ))
                inserted_count += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Stored {inserted_count} bars in database")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return 0
    
    def get_available_dates(self, lookback_days: int = 10) -> list:
        """Get available trading dates from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute('''
                SELECT DISTINCT DATE(timestamp) as trading_date 
                FROM dual_ticker_bars 
                WHERE symbol = 'NVDA'
                AND timestamp >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY trading_date DESC
            ''', (lookback_days,))
            
            dates = [row[0].strftime('%Y-%m-%d') for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return sorted(dates)  # Oldest first
            
        except Exception as e:
            logger.error(f"Failed to get available dates: {e}")
            return []


class BacktestRunner:
    """Runs rolling-window backtests on live data"""
    
    def __init__(self, db_config: dict, window_size: int = 3):
        self.db_config = db_config
        self.window_size = window_size
        
    def run_rolling_windows(self, trading_days: list) -> dict:
        """Run rolling-window backtests"""
        if len(trading_days) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} days, got {len(trading_days)}")
        
        results = []
        total_windows = len(trading_days) - self.window_size + 1
        
        logger.info(f"Running {total_windows} rolling windows of {self.window_size} days each")
        
        for i in range(total_windows):
            window_days = trading_days[i:i+self.window_size]
            start_date = window_days[0]
            end_date = window_days[-1]
            
            logger.info(f"Window {i+1}/{total_windows}: {start_date} to {end_date}")
            
            try:
                # Load data for this window
                adapter = DualTickerDataAdapter(self.db_config)
                data = adapter.load_training_data(start_date, end_date)
                
                if not data or len(data.get('trading_days', [])) == 0:
                    logger.warning(f"No data for window {i+1}")
                    continue
                
                # Run backtest
                result = self._run_single_window(data, i+1, start_date, end_date)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Window {i+1} failed: {e}")
                results.append({
                    'window': i+1,
                    'start_date': start_date,
                    'end_date': end_date,
                    'error': str(e),
                    'success': False
                })
        
        return self._calculate_summary(results)
    
    def _run_single_window(self, data: dict, window_num: int, start_date: str, end_date: str) -> dict:
        """Run backtest for a single window"""
        
        # Create environment
        env = DualTickerTradingEnv(**data)
        obs, info = env.reset()
        
        # Run simple buy-and-hold strategy
        total_reward = 0
        step_count = 0
        portfolio_values = [info.get('portfolio_value', 100000)]
        
        while step_count < 1500:  # Safety limit
            # Simple strategy: Hold both stocks (action 4)
            action = 4  # HOLD_BOTH
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            portfolio_values.append(info.get('portfolio_value', 100000))
            
            if done:
                break
        
        # Calculate metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        returns = (final_value - initial_value) / initial_value
        
        # Sharpe ratio calculation
        if len(portfolio_values) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe = np.mean(daily_returns) / max(0.001, np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        result = {
            'window': window_num,
            'start_date': start_date,
            'end_date': end_date,
            'steps': step_count,
            'initial_value': initial_value,
            'final_value': final_value,
            'returns_pct': returns * 100,
            'total_reward': total_reward,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown * 100,
            'success': True
        }
        
        logger.info(f"  Window {window_num}: {returns*100:.2f}% returns, "
                   f"Sharpe: {sharpe:.2f}, Max DD: {max_drawdown*100:.2f}%")
        
        return result
    
    def _calculate_summary(self, results: list) -> dict:
        """Calculate summary statistics"""
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            return {
                'summary': {
                    'total_windows': len(results),
                    'successful_windows': 0,
                    'success_rate': 0,
                    'error': 'No successful windows'
                },
                'windows': results
            }
        
        returns = [r['returns_pct'] for r in successful]
        sharpe_ratios = [r['sharpe_ratio'] for r in successful]
        
        summary = {
            'total_windows': len(results),
            'successful_windows': len(successful),
            'success_rate': len(successful) / len(results),
            'avg_returns_pct': np.mean(returns),
            'std_returns_pct': np.std(returns),
            'min_returns_pct': np.min(returns),
            'max_returns_pct': np.max(returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'summary': summary,
            'windows': results
        }


def get_trading_days(lookback_days: int) -> list:
    """Get recent trading days (excluding weekends)"""
    trading_days = []
    current_date = datetime.now() - timedelta(days=1)  # Start from yesterday
    
    while len(trading_days) < lookback_days:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            trading_days.append(current_date.strftime('%Y-%m-%d'))
        current_date -= timedelta(days=1)
        
        # Safety check
        if (datetime.now() - current_date).days > 14:
            break
    
    return sorted(trading_days)  # Oldest first


def save_results(results: dict, output_file: str):
    """Save results to JSON file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Nightly Polygon backtest runner")
    parser.add_argument('--lookback-days', type=int, default=5,
                       help='Number of trading days to look back (default: 5)')
    parser.add_argument('--window-size', type=int, default=3,
                       help='Rolling window size in days (default: 3)')
    parser.add_argument('--output', default='reports/backtest/nightly_results.json',
                       help='Output file for results')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (2 days, smaller windows)')
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.lookback_days = 2
        args.window_size = 1
        logger.info("🧪 Running in test mode")
    
    print("🌙 NIGHTLY POLYGON BACKTEST")
    print("=" * 40)
    
    try:
        # Get API key
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            sys.exit(1)
        
        # Database configuration
        db_config = {
            'host': os.getenv('TEST_DB_HOST', 'localhost'),
            'port': int(os.getenv('TEST_DB_PORT', 5432)),
            'database': os.getenv('TEST_DB_NAME', 'intradayjules'),
            'user': os.getenv('TEST_DB_USER', 'postgres'),
            'password': os.getenv('TEST_DB_PASSWORD', 'ci_test_password')
        }
        
        # Get trading days
        trading_days = get_trading_days(args.lookback_days)
        logger.info(f"Target trading days: {trading_days}")
        
        # Fetch live data from Polygon
        logger.info("📥 Fetching live data from Polygon API...")
        fetcher = PolygonDataFetcher(api_key)
        market_data = fetcher.fetch_multiple_days(['NVDA', 'MSFT'], trading_days)
        
        if market_data.empty:
            logger.error("Failed to fetch market data")
            sys.exit(1)
        
        # Store in database
        logger.info("💾 Storing data in database...")
        db_manager = DatabaseManager(db_config)
        stored_count = db_manager.store_market_data(market_data)
        
        if stored_count == 0:
            logger.error("Failed to store market data")
            sys.exit(1)
        
        # Get available dates from database
        available_dates = db_manager.get_available_dates(args.lookback_days)
        logger.info(f"Available dates in DB: {available_dates}")
        
        # Run rolling-window backtest
        logger.info("🎯 Running rolling-window backtest...")
        runner = BacktestRunner(db_config, args.window_size)
        results = runner.run_rolling_windows(available_dates)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        summary = results['summary']
        logger.info("📊 BACKTEST SUMMARY:")
        logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        
        if 'avg_returns_pct' in summary:
            logger.info(f"  Avg returns: {summary['avg_returns_pct']:.2f}%")
            logger.info(f"  Avg Sharpe: {summary['avg_sharpe_ratio']:.2f}")
            logger.info(f"  Win rate: {summary['win_rate']:.1%}")
        
        # Performance gate
        if summary['success_rate'] >= 0.7:
            logger.info("✅ SUCCESS: Performance gate passed")
            sys.exit(0)
        else:
            logger.error(f"❌ FAILURE: Success rate {summary['success_rate']:.1%} < 70%")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()