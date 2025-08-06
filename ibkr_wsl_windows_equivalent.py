#!/usr/bin/env python3
"""
EXACT WSL equivalent of the Windows IBKR test script
Using canonical WSL fix: 172.24.32.1 instead of 127.0.0.1
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ib_insync import *

def test_ibkr_connection():
    """
    Test IBKR connection by connecting, retrieving market data, and placing a test order.
    WSL version - EXACT copy of Windows script with IP change.
    """
    ib = IB()
    
    try:
        # Connect to IBKR - ONLY CHANGE: Use Windows host IP from WSL
        print("Connecting to IBKR...")
        ib.connect('172.24.32.1', 7497, clientId=1)  # 172.24.32.1 instead of 127.0.0.1
        print("✓ Connected successfully")
        
        # Test market data request
        print("\nTesting market data request...")
        stock = Stock('MSFT', 'SMART', 'USD')
        ticker = ib.reqMktData(stock, '', False, False)
        
        # Wait for data to populate
        ib.sleep(2)
        
        if ticker.last is not None:
            print(f"✓ Market data received - Last price: ${ticker.last}")
        else:
            print("⚠ Market data request successful but no price data available")
            print("  (This may be normal outside market hours)")
        
        # Test order placement (use paper trading account!)
        print("\nTesting order placement...")
        order = MarketOrder('BUY', 1)
        trade = ib.placeOrder(stock, order)
        
        # Wait for order status
        ib.sleep(2)
        
        print(f"✓ Order placed - Status: {trade.orderStatus.status}")
        
        # Cancel the order to avoid accidental execution
        if trade.orderStatus.status in ['Submitted', 'PreSubmitted']:
            print("Cancelling test order...")
            ib.cancelOrder(order)
            ib.sleep(1)
            print("✓ Test order cancelled")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
        
    finally:
        # Always disconnect
        if ib.isConnected():
            ib.disconnect()
            print("✓ Disconnected from IBKR")

def main():
    """
    Main function to run the IBKR connection test.
    WSL version - EXACT copy of Windows script.
    """
    print("IBKR Connection Tester - WSL Version")
    print("=" * 40)
    print("🔧 Using Canonical WSL Fix: 172.24.32.1")
    print("⚠ WARNING: Make sure you're using a paper trading account!")
    print("⚠ This test will place and cancel a small market order.")
    print()
    
    print("Running automatic test (no confirmation needed)...")
    print()
    
    success = test_ibkr_connection()
    
    if success:
        print("\n✅ Connection test PASSED")
        print("🎯 WSL Canonical Fix working perfectly!")
    else:
        print("\n❌ Connection test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()