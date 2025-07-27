#!/usr/bin/env python3
"""
Real-time log monitor for IntradayJules live trading
"""
import time
import os
import sys
from datetime import datetime

def monitor_log_file(log_file):
    """Monitor a log file for new entries"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📡 Monitoring: {log_file}")
    print("🔍 Watching for live trading activity...")
    print("=" * 60)
    
    # Get initial file size
    with open(log_file, 'r') as f:
        f.seek(0, 2)  # Go to end of file
        last_position = f.tell()
    
    try:
        while True:
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                
                if new_lines:
                    for line in new_lines:
                        line = line.strip()
                        if any(keyword in line.lower() for keyword in [
                            'live trading', 'ibkr connection', 'account summary', 
                            'historical bars', 'orchestrator', 'live_trading',
                            'connecting to ibkr', 'fetching initial', 'warmup'
                        ]):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] {line}")
                    
                    last_position = f.tell()
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

def main():
    log_file = "logs/data_provisioning.log"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print("🔍 IntradayJules Live Log Monitor")
    print("=" * 40)
    print("💡 This will show live trading related log entries in real-time")
    print("💡 Press Ctrl+C to stop monitoring")
    print()
    
    monitor_log_file(log_file)

if __name__ == "__main__":
    main()