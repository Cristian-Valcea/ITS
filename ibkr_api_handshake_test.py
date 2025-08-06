#!/usr/bin/env python3
"""
IBKR API Handshake Test - Detailed diagnosis of API connection issues
"""

import sys
import time
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ib_insync import IB, Stock
    IB_AVAILABLE = True
except ImportError:
    print("❌ ib_insync not available")
    sys.exit(1)

def detailed_connection_test():
    """Perform detailed connection test with verbose logging"""
    print("🔍 DETAILED IBKR API HANDSHAKE TEST")
    print("=" * 50)
    
    host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
    
    print(f"📡 Target: {host}:{port} (Client ID: {client_id})")
    print(f"🐧 WSL IP: {os.getenv('IBKR_WSL_IP', 'Unknown')}")
    print()
    
    # Enable verbose logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    ib = IB()
    
    try:
        print("🔌 Step 1: Initiating connection...")
        print(f"   Connecting to {host}:{port} with clientId={client_id}")
        
        # Try connection with extended timeout
        ib.connect(host, port, clientId=client_id, timeout=30)
        
        print("✅ Step 2: Socket connection established!")
        print(f"   Connection status: {ib.isConnected()}")
        
        if ib.isConnected():
            print("✅ Step 3: API handshake successful!")
            
            # Get server info
            try:
                server_version = ib.client.serverVersion()
                print(f"   📊 Server version: {server_version}")
            except:
                print("   ⚠️  Could not get server version")
            
            # Get connection time
            try:
                connection_time = ib.client.connectionTime()
                print(f"   ⏰ Connection time: {connection_time}")
            except:
                print("   ⚠️  Could not get connection time")
            
            # Test account access
            print("\n🔍 Step 4: Testing account access...")
            try:
                accounts = ib.managedAccounts()
                if accounts:
                    print(f"   ✅ Managed accounts: {accounts}")
                    
                    # Test account summary
                    account = accounts[0]
                    print(f"   📊 Testing account summary for: {account}")
                    
                    # Request account summary
                    summary = ib.accountSummary()
                    if summary:
                        print(f"   ✅ Account summary received ({len(summary)} items)")
                        
                        # Show key account info
                        for item in summary[:5]:  # Show first 5 items
                            print(f"      {item.tag}: {item.value} {item.currency}")
                    else:
                        print("   ⚠️  No account summary data")
                        
                else:
                    print("   ❌ No managed accounts found")
                    
            except Exception as e:
                print(f"   ❌ Account access failed: {e}")
            
            # Test market data access
            print("\n🔍 Step 5: Testing market data access...")
            try:
                print("   📈 Requesting AAPL market data...")
                stock = Stock('AAPL', 'SMART', 'USD')
                ib.qualifyContracts(stock)
                print(f"   ✅ Contract qualified: {stock}")
                
                ticker = ib.reqMktData(stock, '', False, False)
                print(f"   📊 Market data request sent...")
                
                # Wait for data
                ib.sleep(3)
                
                if ticker.last and str(ticker.last) != 'nan':
                    print(f"   ✅ Live market data: AAPL ${ticker.last:.2f}")
                elif ticker.close and str(ticker.close) != 'nan':
                    print(f"   ✅ Close price: AAPL ${ticker.close:.2f}")
                else:
                    print("   ⚠️  No market data (may be after hours)")
                    
                # Cancel market data
                ib.cancelMktData(stock)
                
            except Exception as e:
                print(f"   ❌ Market data test failed: {e}")
            
            # Test order placement capability
            print("\n🔍 Step 6: Testing order placement capability...")
            try:
                from ib_insync import MarketOrder
                
                # Create a test order (but don't place it)
                stock = Stock('AAPL', 'SMART', 'USD')
                order = MarketOrder('BUY', 1)
                
                print("   ✅ Order objects created successfully")
                print(f"   📋 Test order: {order.action} {order.totalQuantity} {stock.symbol}")
                print("   ℹ️  (Order not actually placed - just testing capability)")
                
            except Exception as e:
                print(f"   ❌ Order capability test failed: {e}")
            
            print("\n🎉 CONNECTION TEST SUCCESSFUL!")
            print("   Your IBKR API connection is working properly!")
            
            return True
            
        else:
            print("❌ Step 3: API handshake failed!")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide specific guidance based on error type
        if "TimeoutError" in str(type(e)):
            print("\n💡 TIMEOUT ERROR DIAGNOSIS:")
            print("   • Socket connection established but API handshake timed out")
            print("   • This usually means IBKR API is not properly configured")
            print("   • Check: API enabled, correct port, trusted IPs")
            
        elif "ConnectionRefusedError" in str(type(e)):
            print("\n💡 CONNECTION REFUSED DIAGNOSIS:")
            print("   • IBKR Workstation is not running or not listening on port")
            print("   • Check: TWS/Gateway running, correct port number")
            
        elif "OSError" in str(type(e)) and "113" in str(e):
            print("\n💡 NO ROUTE TO HOST DIAGNOSIS:")
            print("   • Network routing issue or incorrect IP address")
            print("   • Check: IP address, network connectivity")
            
        return False
        
    finally:
        if ib.isConnected():
            print("\n🔌 Disconnecting...")
            ib.disconnect()
            print("   ✅ Disconnected cleanly")

def main():
    """Run detailed API handshake test"""
    
    print("This test will perform a detailed diagnosis of your IBKR API connection.")
    print("It will test each step of the connection process individually.")
    print()
    
    try:
        response = input("Proceed with detailed test? (y/N): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return
    
    print()
    success = detailed_connection_test()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 RESULT: IBKR API CONNECTION IS WORKING!")
        print("\n🚀 NEXT STEPS:")
        print("   1. Your IBKR setup is correct")
        print("   2. You can now run real AI trading")
        print("   3. Try: python working_ai_trader.py")
        
    else:
        print("❌ RESULT: IBKR API CONNECTION NEEDS CONFIGURATION")
        print("\n🔧 REQUIRED ACTIONS:")
        print("   1. Open IBKR Paper Trading Workstation")
        print("   2. Enable API: File → Global Configuration → API → Settings")
        print("   3. Add WSL IP to Trusted IPs")
        print("   4. Restart IBKR Workstation")
        print("   5. Run this test again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)