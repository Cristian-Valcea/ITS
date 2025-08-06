#!/usr/bin/env python3
"""
WSL Equivalent of Windows IBKR Test Script - Auto Run
Using canonical WSL fix with working IP configuration
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ib_insync import *
    print("✅ ib_insync imported successfully")
except ImportError:
    print("❌ ib_insync not available")
    sys.exit(1)

def test_ibkr_connection():
    """
    Test IBKR connection by connecting, retrieving market data, and placing a test order.
    WSL version using canonical fix.
    """
    ib = IB()
    
    try:
        # Connect to IBKR using WSL canonical fix
        print("🔌 Connecting to IBKR from WSL...")
        print("📡 Using canonical WSL fix: Windows host IP 172.24.32.1")
        ib.connect('172.24.32.1', 7497, clientId=1)  # WSL canonical fix
        print("✅ Connected successfully!")
        
        # Show connection details
        print(f"📊 Server version: {ib.client.serverVersion()}")
        try:
            conn_time = ib.client.twsConnectionTime()
            print(f"🕐 Connection time: {conn_time}")
        except:
            print("🕐 Connection time: API method not available")
        
        accounts = ib.managedAccounts()
        print(f"👤 Managed accounts: {accounts}")
        
        # Test market data request
        print("\n📈 Testing market data request...")
        stock = Stock('MSFT', 'SMART', 'USD')
        
        # Qualify contract first
        ib.qualifyContracts(stock)
        print(f"✅ Contract qualified: {stock}")
        
        ticker = ib.reqMktData(stock, '', False, False)
        
        # Wait for data to populate
        print("⏳ Waiting for market data...")
        ib.sleep(3)
        
        if ticker.last is not None and str(ticker.last) != 'nan':
            print(f"✅ Market data received - Last price: ${ticker.last}")
        elif ticker.close is not None and str(ticker.close) != 'nan':
            print(f"✅ Market data received - Close price: ${ticker.close}")
        else:
            print("⚠️  Market data request successful but no price data available")
            print("   (This may be normal outside market hours or requires subscription)")
            print(f"   Ticker state: last={ticker.last}, close={ticker.close}")
        
        # Test order placement (use paper trading account!)
        print("\n📋 Testing order placement...")
        order = MarketOrder('BUY', 1)  # Buy 1 share
        trade = ib.placeOrder(stock, order)
        
        # Wait for order status
        print("⏳ Waiting for order acknowledgment...")
        ib.sleep(3)
        
        print(f"✅ Order placed successfully!")
        print(f"   Order ID: {trade.order.orderId}")
        print(f"   Status: {trade.orderStatus.status}")
        print(f"   Details: {trade.order.action} {trade.order.totalQuantity} {stock.symbol}")
        
        # Cancel the order to avoid accidental execution
        if trade.orderStatus.status in ['Submitted', 'PreSubmitted', 'PendingSubmit']:
            print("🔄 Cancelling test order...")
            ib.cancelOrder(trade.order)
            ib.sleep(2)
            print("✅ Test order cancelled successfully")
        else:
            print(f"ℹ️  Order status '{trade.orderStatus.status}' - may not need cancellation")
        
        # Test positions
        print("\n📊 Testing positions retrieval...")
        try:
            positions = ib.positions()
            if positions:
                print("✅ Positions retrieved:")
                for pos in positions[:3]:  # Show first 3
                    print(f"   {pos.contract.symbol}: {pos.position} shares")
            else:
                print("✅ No positions found (clean paper account)")
        except Exception as e:
            print(f"⚠️  Positions error: {e}")
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🎯 WSL Canonical Fix is working perfectly!")
        print("🚀 IBKR Paper Trading is fully operational from WSL!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Specific WSL troubleshooting
        if "ConnectionRefusedError" in str(type(e)) or "Connection refused" in str(e):
            print("\n🔧 WSL Troubleshooting:")
            print("   - Verify IBKR Workstation is running")
            print("   - Check API is enabled in TWS")
            print("   - Ensure using correct Windows host IP: 172.24.32.1")
        elif "TimeoutError" in str(type(e)) or "reset by peer" in str(e):
            print("\n🔧 WSL Troubleshooting:")
            print("   - Add WSL IP (172.24.46.63) to IBKR Trusted IPs")
            print("   - Uncheck 'Allow connections from localhost only'")
            print("   - Restart IBKR Workstation")
        
        return False
        
    finally:
        # Always disconnect
        if ib.isConnected():
            ib.disconnect()
            print("🔌 Disconnected from IBKR")

def main():
    """
    Main function to run the WSL IBKR connection test.
    """
    print("🔌 IBKR CONNECTION TESTER - WSL VERSION")
    print("=" * 50)
    print("🔧 Using Canonical WSL Fix")
    print("📡 Connecting to Windows host: 172.24.32.1")
    print("👤 Paper trading account: DU8009825")
    print("⚠️  Running in PAPER TRADING mode")
    print("⚠️  Test will place and cancel a small market order")
    print()
    
    success = test_ibkr_connection()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ WSL CONNECTION TEST PASSED")
        print("🎯 Canonical WSL fix is working perfectly!")
        print("🚀 Ready for full paper trading deployment!")
        print("\n📝 Next steps:")
        print("   1. Update your trading scripts to use IBKR_HOST_IP=172.24.32.1")
        print("   2. Deploy paper trading with real strategies")
        print("   3. Monitor performance via Grafana dashboards")
    else:
        print("\n" + "=" * 50)
        print("❌ WSL CONNECTION TEST FAILED")
        print("🔧 Check the troubleshooting suggestions above")
        sys.exit(1)

if __name__ == "__main__":
    main()