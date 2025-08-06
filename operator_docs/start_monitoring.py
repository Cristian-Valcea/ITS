#!/usr/bin/env python3
"""
Start Monitoring System
Simple script for operators to start the complete monitoring stack
"""

import os
import sys
import time
import logging

# Add src to path
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from risk_governor.prometheus_monitoring import setup_monitoring

def main():
    print("🚀 Starting Production Risk Governor Monitoring...")
    
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/monitoring.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Start monitoring system
        monitoring = setup_monitoring(
            prometheus_port=8000,
            slack_webhook=None  # Configure if available
        )
        
        print("✅ Monitoring system started successfully!")
        print("📊 Prometheus metrics: http://localhost:8000/metrics")
        print("📈 System ready for trading operations")
        print("🔍 Health checks will run every 30 seconds")
        print("")
        print("Press Ctrl+C to stop monitoring...")
        
        # Keep running
        try:
            while True:
                time.sleep(30)
                print(f"💓 Monitoring heartbeat: {time.strftime('%H:%M:%S')}")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down monitoring system...")
            monitoring.stop()
            print("✅ Monitoring stopped successfully")
        
    except Exception as e:
        print(f"❌ Failed to start monitoring: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())