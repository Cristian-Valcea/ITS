#!/usr/bin/env python3
"""
IBKR Connection Hold Test - Keep connection open to see IBKR prompts
"""

import sys
import time
import os
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ib_insync import IB
    IB_AVAILABLE = True
except ImportError:
    print("❌ ib_insync not available")
    sys.exit(1)

def connection_hold_test():
    """Hold connection open to trigger IBKR prompts"""
    
    host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    client_id = 999  # Use unique client ID
    
    print("🔍 IBKR CONNECTION HOLD TEST")
    print("=" * 40)
    print(f"📡 Target: {host}:{port}")
    print(f"🆔 Client ID: {client_id}")
    print()
    print("This test will:")
    print("1. Connect to IBKR and hold the connection")
    print("2. Give you time to see any IBKR prompts")
    print("3. Wait for you to accept the connection")
    print()
    
    ib = IB()
    
    try:
        print("🔌 Connecting to IBKR...")
        print("👀 WATCH YOUR IBKR WORKSTATION FOR POPUPS!")
        print()
        
        # Connect with longer timeout
        ib.connect(host, port, clientId=client_id, timeout=30)
        
        if ib.isConnected():
            print("🎉 CONNECTION SUCCESSFUL!")
            print()
            
            # Get connection info
            try:
                server_version = ib.client.serverVersion()
                print(f"📊 Server version: {server_version}")
            except:
                print("⚠️  Could not get server version")
            
            try:
                accounts = ib.managedAccounts()
                print(f"👤 Accounts: {accounts}")
            except:
                print("⚠️  Could not get accounts")
            
            print()
            print("✅ IBKR API is working correctly!")
            print("🎯 You can now run real AI trading!")
            
            return True
            
        else:
            print("❌ Connection not confirmed")
            return False
            
    except Exception as e:
        error_type = type(e).__name__
        print(f"❌ Connection failed: {error_type}: {e}")
        
        if "TimeoutError" in error_type:
            print()
            print("💡 TIMEOUT DIAGNOSIS:")
            print("   • Connection attempt timed out")
            print("   • This usually means:")
            print("     - IBKR rejected the connection")
            print("     - No popup appeared (API not enabled)")
            print("     - Popup appeared but was not accepted")
            print()
            print("🔧 NEXT STEPS:")
            print("   1. Check if you saw any popup in IBKR")
            print("   2. If no popup: Enable API in IBKR settings")
            print("   3. If popup appeared: Click Accept and try again")
            
        return False
        
    finally:
        if ib.isConnected():
            print("\n🔌 Disconnecting...")
            ib.disconnect()
            print("✅ Disconnected")

def main():
    """Main test function"""
    
    print("🚨 IMPORTANT: Make sure IBKR Workstation is open and visible!")
    print("You need to watch for connection prompts.")
    print()
    
    try:
        response = input("Ready to test? (y/N): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return
    
    print()
    success = connection_hold_test()
    
    if success:
        print("\n🎉 SUCCESS! IBKR connection is working!")
        print("🚀 Ready for AI trading!")
    else:
        print("\n🔧 Connection needs configuration")
        print("💡 Run: python ibkr_config_diagnosis.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)