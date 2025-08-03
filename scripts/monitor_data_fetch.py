#!/usr/bin/env python3
"""
📊 MONITOR DATA FETCH PROGRESS
Real-time monitoring of historical data fetch for gold standard training
"""

import os
import sys
import time
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

from secrets_helper import SecretsHelper

def check_data_progress():
    """Check current data fetch progress"""
    
    try:
        # Database config
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'minute_bars'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("📊 Table 'minute_bars' not yet created")
            return None
        
        # Get data statistics
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as total_bars,
                MIN(timestamp) as first_bar,
                MAX(timestamp) as last_bar,
                COUNT(DISTINCT DATE(timestamp)) as trading_days
            FROM minute_bars 
            GROUP BY symbol
            ORDER BY symbol;
        """)
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return None

def check_fetch_process():
    """Check if fetch process is still running"""
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'polygon_historical_fetch.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    return True, pid
        
        return False, None
        
    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False, None

def estimate_completion_time(current_bars: int, target_bars: int, start_time: datetime):
    """Estimate completion time based on current progress"""
    
    if current_bars == 0:
        return "Unknown"
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    bars_per_second = current_bars / elapsed_time
    
    remaining_bars = target_bars - current_bars
    remaining_seconds = remaining_bars / bars_per_second if bars_per_second > 0 else 0
    
    remaining_hours = remaining_seconds / 3600
    
    completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
    
    return f"{remaining_hours:.1f}h (ETA: {completion_time.strftime('%H:%M')})"

def main():
    """Monitor data fetch progress"""
    
    print("📊 MONITORING HISTORICAL DATA FETCH")
    print("=" * 50)
    
    # Estimated targets (rough estimates)
    target_bars_per_symbol = 1800000  # ~1.8M bars per symbol for 3+ years
    total_target_bars = target_bars_per_symbol * 2  # NVDA + MSFT
    
    # Assume fetch started around 20:01
    fetch_start_time = datetime.now().replace(hour=20, minute=1, second=0, microsecond=0)
    
    while True:
        print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')} - Checking progress...")
        
        # Check if process is running
        process_running, pid = check_fetch_process()
        
        if process_running:
            print(f"✅ Fetch process running (PID: {pid})")
        else:
            print("⚠️ Fetch process not found - may have completed or failed")
        
        # Check data progress
        data_results = check_data_progress()
        
        if data_results is None:
            print("📊 No data available yet")
        else:
            print(f"📊 Current data in TimescaleDB:")
            
            total_bars = 0
            for symbol, bars, first_bar, last_bar, trading_days in data_results:
                total_bars += bars
                print(f"   {symbol}: {bars:,} bars ({trading_days} days)")
                print(f"      Range: {first_bar} to {last_bar}")
            
            print(f"   Total: {total_bars:,} bars")
            
            # Progress calculation
            if total_bars > 0:
                progress_pct = (total_bars / total_target_bars) * 100
                print(f"   Progress: {progress_pct:.1f}% of estimated target")
                
                # ETA calculation
                eta = estimate_completion_time(total_bars, total_target_bars, fetch_start_time)
                print(f"   ETA: {eta}")
        
        # Check if we should continue monitoring
        if not process_running and data_results and len(data_results) == 2:
            # Process finished and we have data for both symbols
            print(f"\n🎉 DATA FETCH APPEARS COMPLETE!")
            
            total_bars = sum(result[1] for result in data_results)
            print(f"   📊 Total bars: {total_bars:,}")
            print(f"   🎯 Ready to start V3 training!")
            
            break
        
        # Wait before next check
        print("⏳ Waiting 30 seconds for next check...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        import traceback
        traceback.print_exc()