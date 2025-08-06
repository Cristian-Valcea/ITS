#!/usr/bin/env python3
"""
Quick IBKR Setup Verification - Run after configuring IBKR Workstation
"""

import sys
import os
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.append('src')

def quick_verification():
    """Quick verification of IBKR setup"""
    print("🔍 QUICK IBKR SETUP VERIFICATION")
    print("=" * 40)
    
    try:
        from brokers.ib_gateway import IBGatewayClient
        
        print("📡 Testing connection...")
        client = IBGatewayClient()
        
        # Test real connection
        success = client.test_real_connection()
        
        if success:
            print("\n🎉 SUCCESS! IBKR API is working correctly!")
            print("\n✅ What this means:")
            print("   • IBKR Workstation is properly configured")
            print("   • API access is enabled and working")
            print("   • Your WSL IP is trusted")
            print("   • Paper trading account is accessible")
            
            print("\n🚀 You can now run:")
            print("   python working_ai_trader.py")
            print("\n   Your AI will place real orders in IBKR Paper Trading!")
            
            return True
            
        else:
            print("\n❌ IBKR API connection still not working")
            print("\n🔧 Please check:")
            print("   1. IBKR Workstation is running in Paper Trading mode")
            print("   2. API is enabled (File → Global Configuration → API)")
            print("   3. WSL IP (172.24.46.63) is in Trusted IPs")
            print("   4. You restarted IBKR Workstation after changes")
            print("   5. Green 'API' indicator shows in status bar")
            
            print("\n💡 Run for detailed guidance:")
            print("   python ibkr_config_diagnosis.py")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("This will quickly test if your IBKR setup is working.")
    print("Make sure you've configured IBKR Workstation first.")
    print()
    
    try:
        response = input("Proceed with verification? (y/N): ")
        if response.lower() != 'y':
            print("Verification cancelled.")
            return
    except KeyboardInterrupt:
        print("\nVerification cancelled.")
        return
    
    print()
    success = quick_verification()
    
    if success:
        print("\n🎯 READY FOR AI TRADING!")
    else:
        print("\n🔧 CONFIGURATION NEEDED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)