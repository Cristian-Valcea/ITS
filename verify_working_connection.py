#!/usr/bin/env python3
"""
Verify the working IBKR connection from WSL
"""

from ib_insync import IB
import time

def quick_test():
    print("🔌 Quick IBKR Connection Verification")
    print("=" * 40)
    
    ib = IB()
    
    try:
        print("Connecting to 172.24.32.1:7497...")
        ib.connect('172.24.32.1', 7497, clientId=3, timeout=10)
        
        if ib.isConnected():
            print("✅ CONNECTION SUCCESSFUL!")
            
            # Get basic info
            server_version = ib.client.serverVersion()
            print(f"📊 Server version: {server_version}")
            
            accounts = ib.managedAccounts()
            print(f"👤 Accounts: {accounts}")
            
            print("🎉 WSL Canonical Fix is working!")
            return True
        else:
            print("❌ Not connected")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("✅ Disconnected")

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎯 CANONICAL WSL FIX CONFIRMED WORKING!")
        print("📝 Configuration: IBKR_HOST_IP=172.24.32.1")
        print("🚀 Ready for production paper trading!")
    else:
        print("\n❌ Connection verification failed")