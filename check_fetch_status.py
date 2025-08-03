#!/usr/bin/env python3
"""Quick status check for the data fetch process"""

import os
import sys
import time
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

from secrets_helper import SecretsHelper

def test_polygon_api():
    """Test if Polygon API is accessible"""
    try:
        api_key = SecretsHelper.get_polygon_api_key()
        
        # Test with a simple call
        url = "https://api.polygon.io/v2/aggs/ticker/NVDA/range/1/minute/2025-08-01/2025-08-01"
        params = {'apikey': api_key, 'limit': 1}
        
        print("🧪 Testing Polygon API connectivity...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API accessible - Status: {data.get('status', 'unknown')}")
            if 'results' in data:
                print(f"   📊 Sample data available: {len(data['results'])} bars")
            return True
        elif response.status_code == 429:
            print("⚠️ Rate limit hit - this is expected during bulk fetch")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def check_process_details():
    """Check process details"""
    import subprocess
    
    try:
        # Get process info
        result = subprocess.run(['ps', '-p', '15936', '-o', 'pid,ppid,etime,pcpu,pmem,cmd'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                header = lines[0]
                process_info = lines[1]
                print("📊 Process Details:")
                print(f"   {header}")
                print(f"   {process_info}")
                
                # Parse elapsed time
                parts = process_info.split()
                if len(parts) >= 3:
                    elapsed = parts[2]  # ETIME column
                    print(f"   ⏱️ Running for: {elapsed}")
                
                return True
        else:
            print("❌ Process 15936 not found")
            return False
            
    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False

def estimate_progress():
    """Estimate progress based on time"""
    
    # Assume started at 20:01
    start_time = datetime.now().replace(hour=20, minute=1, second=0, microsecond=0)
    elapsed = datetime.now() - start_time
    elapsed_minutes = elapsed.total_seconds() / 60
    
    print(f"⏱️ Estimated elapsed time: {elapsed_minutes:.1f} minutes")
    
    # With rate limiting (12s between calls) and 30-day chunks
    # Each symbol needs ~40 chunks (3.5 years / 30 days)
    # Each chunk takes ~12s + API call time (~1s) = ~13s
    # Total per symbol: 40 * 13s = 520s = 8.7 minutes
    # Both symbols: ~17.4 minutes
    
    estimated_total_minutes = 17.4
    progress_pct = (elapsed_minutes / estimated_total_minutes) * 100
    
    print(f"📊 Estimated progress: {progress_pct:.1f}%")
    
    if progress_pct < 100:
        remaining_minutes = estimated_total_minutes - elapsed_minutes
        print(f"⏳ Estimated remaining: {remaining_minutes:.1f} minutes")
    else:
        print("🤔 Should be complete - may be in database insertion phase")

def main():
    print("🔍 FETCH STATUS CHECK")
    print("=" * 40)
    print(f"🕐 Current time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Check if process exists
    if not check_process_details():
        return
    
    print()
    
    # Test API connectivity
    api_ok = test_polygon_api()
    print()
    
    # Estimate progress
    estimate_progress()
    print()
    
    if api_ok:
        print("✅ System appears healthy - fetch likely in progress")
        print("💡 The process may be in rate-limiting delays between API calls")
        print("💡 Database table will be created when first data is inserted")
    else:
        print("⚠️ API issues detected - may need intervention")

if __name__ == "__main__":
    main()