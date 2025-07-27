#!/usr/bin/env python3
"""
Script to help you find and view IntradayJules logs
"""
import os
import glob
import subprocess
import sys
from datetime import datetime

def find_log_files():
    """Find all log files in the project"""
    log_patterns = [
        "*.log",
        "logs/*.log", 
        "test_logs/**/*.log",
        "src/**/*.log"
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern, recursive=True))
    
    return log_files

def show_recent_logs():
    """Show recent log files"""
    log_files = find_log_files()
    
    if not log_files:
        print("❌ No log files found in the project directory")
        print("\n💡 The logs might be going to:")
        print("   1. Console output (what you see when running the server)")
        print("   2. System logs (Windows Event Viewer)")
        print("   3. Temporary files")
        return
    
    print(f"📋 Found {len(log_files)} log files:")
    print("=" * 50)
    
    for log_file in sorted(log_files, key=os.path.getmtime, reverse=True):
        stat = os.stat(log_file)
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"📄 {log_file}")
        print(f"   Size: {size:,} bytes")
        print(f"   Modified: {modified}")
        print()

def tail_latest_log():
    """Show the tail of the most recent log file"""
    log_files = find_log_files()
    
    if not log_files:
        print("❌ No log files found")
        return
    
    # Get the most recently modified log file
    latest_log = max(log_files, key=os.path.getmtime)
    
    print(f"📄 Showing last 50 lines of: {latest_log}")
    print("=" * 60)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def show_live_trading_logs():
    """Show logs related to live trading"""
    print("🔍 Searching for live trading logs...")
    
    # Search for live trading related log entries
    search_terms = [
        "LIVE TRADING",
        "live_trading",
        "IBKR connection",
        "account summary",
        "historical bars",
        "OrchestratorAgent"
    ]
    
    log_files = find_log_files()
    found_entries = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for term in search_terms:
                        if term.lower() in line.lower():
                            found_entries.append((log_file, i+1, line.strip()))
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
    
    if found_entries:
        print(f"✅ Found {len(found_entries)} live trading log entries:")
        print("=" * 60)
        for log_file, line_num, line in found_entries[-20:]:  # Show last 20
            print(f"📄 {log_file}:{line_num}")
            print(f"   {line}")
            print()
    else:
        print("❌ No live trading logs found in files")
        print("\n💡 The live trading logs are likely in the console output.")
        print("   When you run the server, watch the terminal for log messages.")

def main():
    print("🔍 IntradayJules Log Viewer")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            show_recent_logs()
        elif command == "tail":
            tail_latest_log()
        elif command == "live":
            show_live_trading_logs()
        else:
            print(f"❌ Unknown command: {command}")
    else:
        print("📋 Available commands:")
        print("   python view_logs.py list  - List all log files")
        print("   python view_logs.py tail  - Show tail of latest log")
        print("   python view_logs.py live  - Search for live trading logs")
        print()
        
        # Show a quick summary
        show_recent_logs()
        print("\n🔍 Searching for live trading logs...")
        show_live_trading_logs()

if __name__ == "__main__":
    main()