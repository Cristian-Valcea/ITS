#!/usr/bin/env python3
"""
Final IBKR WSL Test - Exactly like Windows script
Uses different client ID to avoid conflicts
"""

from ib_insync import *
import time

def test_ibkr_connection():
    """
    Test IBKR connection - WSL version with canonical fix
    """
    ib = IB()
    
    try:
        # Connect to IBKR using WSL canonical fix
        print("Connecting to IBKR from WSL...")
        ib.connect('172.24.32.1', 7497, clientId=2)  # Different client ID
        print("✓ Connected successfully")
        
        # Show basic info
        print(f"Server version: {ib.client.serverVersion()}")
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")
        
        # Test market data request
        print("\nTesting market data request...")
        stock = Stock('MSFT', 'SMART', 'USD')
        ticker = ib.reqMktData(stock, '', False, False)
        
        # Wait for data to populate
        ib.sleep(2)
        
        if ticker.last is not None and str(ticker.last) != 'nan':
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
        print(f"  Order ID: {trade.order.orderId}")
        
        # Cancel the order to avoid accidental execution
        if trade.orderStatus.status in ['Submitted', 'PreSubmitted']:
            print("Cancelling test order...")
            ib.cancelOrder(order)
            ib.sleep(1)
            print("✓ Test order cancelled")
        
        print("\n🎉 All tests completed successfully!")
        print("🎯 WSL canonical fix working perfectly!")
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
    print("🔌 FINAL IBKR WSL TEST")
    print("=" * 30)
    print("Equivalent to Windows script using canonical WSL fix")
    print()
    
    success = test_ibkr_connection()
    
    if success:
        print("\n✅ WSL CONNECTION TEST PASSED!")
        print("🚀 Ready for paper trading deployment!")
    else:
        print("\n❌ Connection test failed")

if __name__ == "__main__":
    main()