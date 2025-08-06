#!/usr/bin/env python3
"""
Debug IBKR Connection - Detailed logging and diagnostics
"""

import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ib_insync import IB, Stock
    from ib_insync.util import logToConsole
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("❌ ib_insync not available")
    sys.exit(1)

# Enable detailed ib_insync logging
logToConsole(logging.DEBUG)

def debug_connection():
    """Debug IBKR connection with detailed logging"""
    
    print("🔍 DEBUG IBKR CONNECTION")
    print("=" * 50)
    
    host = "172.24.32.1"
    port = 7497
    client_id = 1
    
    print(f"📡 Connecting to: {host}:{port} (Client ID: {client_id})")
    print(f"🔧 With detailed logging enabled...")
    print()
    
    try:
        ib = IB()
        
        # Set up event handlers for debugging
        def onConnected():
            print("🎉 onConnected event fired!")
            
        def onDisconnected():
            print("💔 onDisconnected event fired!")
            
        def onError(reqId, errorCode, errorString, contract):
            print(f"❌ onError: reqId={reqId}, code={errorCode}, msg={errorString}")
            if contract:
                print(f"   Contract: {contract}")
        
        ib.connectedEvent += onConnected
        ib.disconnectedEvent += onDisconnected
        ib.errorEvent += onError
        
        print("🔄 Attempting connection with timeout=30s...")
        ib.connect(host, port, clientId=client_id, timeout=30)
        
        print("✅ Connection established!")
        print(f"📊 Server version: {ib.client.serverVersion()}")
        print(f"🕐 Connection time: {ib.client.twsConnectionTime()}")
        
        # Wait a bit to see if we get any errors
        print("⏳ Waiting 5 seconds for any error messages...")
        ib.sleep(5)
        
        # Try to get account summary
        print("📊 Requesting account summary...")
        try:
            account_summary = ib.accountSummary()
            if account_summary:
                print("✅ Account summary received:")
                for item in account_summary[:5]:  # Show first 5 items
                    print(f"   {item.tag}: {item.value} {item.currency}")
            else:
                print("⚠️  No account summary received")
        except Exception as e:
            print(f"❌ Account summary error: {e}")
        
        # Try to get managed accounts
        print("👤 Requesting managed accounts...")
        try:
            accounts = ib.managedAccounts()
            if accounts:
                print(f"✅ Managed accounts: {accounts}")
            else:
                print("⚠️  No managed accounts received")
        except Exception as e:
            print(f"❌ Managed accounts error: {e}")
        
        # Try a simple market data request
        print("📈 Testing market data request...")
        try:
            contract = Stock('AAPL', 'SMART', 'USD')
            ib.qualifyContracts(contract)
            print(f"✅ Contract qualified: {contract}")
            
            ticker = ib.reqMktData(contract, '', False, False)
            print(f"📊 Market data requested for {contract.symbol}")
            
            # Wait for data
            ib.sleep(5)
            
            if ticker.last and ticker.last > 0:
                print(f"✅ {contract.symbol} Last: ${ticker.last:.2f}")
            elif ticker.close and ticker.close > 0:
                print(f"✅ {contract.symbol} Close: ${ticker.close:.2f}")
            else:
                print(f"⚠️  No market data for {contract.symbol}")
                
        except Exception as e:
            print(f"❌ Market data error: {e}")
        
        print("🔌 Disconnecting...")
        ib.disconnect()
        print("✅ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def check_ibkr_settings():
    """Display checklist for IBKR settings"""
    print("\n🔧 IBKR WORKSTATION SETTINGS CHECKLIST")
    print("=" * 50)
    print("Please verify these settings in IBKR Workstation:")
    print()
    print("1. 📊 Paper Trading Mode:")
    print("   - Title bar should show 'Paper Trading'")
    print("   - Account should start with 'DU' (demo/paper)")
    print()
    print("2. 🔌 API Configuration (File → Global Configuration → API):")
    print("   - ✅ Enable ActiveX and Socket Clients")
    print("   - ✅ Socket port: 7497")
    print("   - ✅ Master API client ID: 0 (or leave blank)")
    print("   - ✅ Read-Only API: NO (unchecked)")
    print("   - ✅ Trusted IPs: 172.24.46.63 (your WSL IP)")
    print()
    print("3. 🔐 Security Settings:")
    print("   - ✅ Bypass Order Precautions for API orders: YES")
    print("   - ✅ Allow connections from localhost: YES")
    print()
    print("4. 🔄 After Changes:")
    print("   - ✅ Click OK")
    print("   - ✅ Restart TWS completely")
    print("   - ✅ Log back into Paper Trading mode")
    print()
    print("5. 📊 Status Verification:")
    print("   - API status should show 'Listening on port 7497'")
    print("   - No error messages in TWS message area")

def main():
    """Main debug function"""
    
    if not IB_AVAILABLE:
        print("❌ ib_insync not available")
        return 1
    
    print("🔍 This script provides detailed debugging for IBKR connection")
    print("🔍 It will show all API communication and error messages")
    print()
    
    success = debug_connection()
    
    if not success:
        check_ibkr_settings()
        return 1
    
    print("\n✅ DEBUG COMPLETE - Connection working!")
    return 0

if __name__ == "__main__":
    exit(main())