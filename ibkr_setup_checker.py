#!/usr/bin/env python3
"""
IBKR Setup Checker - Diagnose IBKR Paper Trading Workstation setup
"""

import os
import sys
import socket
import subprocess
import time
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    load_dotenv(env_file)
except ImportError:
    pass

def check_network_connectivity():
    """Check basic network connectivity to IBKR host"""
    print("🌐 NETWORK CONNECTIVITY CHECK")
    print("=" * 40)
    
    host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    
    print(f"   Target: {host}:{port}")
    
    # Test port connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"   ✅ Port {port} is OPEN and accepting connections")
            return True
        else:
            print(f"   ❌ Port {port} is CLOSED or unreachable")
            print(f"      Error code: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ Network test failed: {e}")
        return False

def check_wsl_configuration():
    """Check WSL network configuration"""
    print("\n🐧 WSL CONFIGURATION CHECK")
    print("=" * 40)
    
    try:
        # Get WSL IP address
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        wsl_ip = result.stdout.strip().split()[0] if result.stdout else "Unknown"
        
        print(f"   WSL IP Address: {wsl_ip}")
        
        # Get Windows host IP (default gateway)
        result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
        if result.stdout:
            gateway_line = result.stdout.strip()
            gateway_ip = gateway_line.split()[2] if len(gateway_line.split()) > 2 else "Unknown"
            print(f"   Windows Host IP: {gateway_ip}")
            
            expected_host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
            if gateway_ip == expected_host:
                print(f"   ✅ Configuration matches: {expected_host}")
            else:
                print(f"   ⚠️  Configuration mismatch:")
                print(f"      Expected: {expected_host}")
                print(f"      Detected: {gateway_ip}")
                print(f"      Consider updating IBKR_HOST_IP to {gateway_ip}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ WSL configuration check failed: {e}")
        return False

def check_ibkr_api_connection():
    """Test actual IBKR API connection"""
    print("\n🔌 IBKR API CONNECTION TEST")
    print("=" * 40)
    
    try:
        sys.path.append('src')
        from brokers.ib_gateway import IBGatewayClient
        
        print("   Creating IBKR client...")
        client = IBGatewayClient()
        
        print(f"   Attempting connection to {client.host}:{client.port}...")
        connected = client.connect()
        
        if connected and not client.simulation_mode:
            print("   ✅ REAL IBKR CONNECTION SUCCESSFUL!")
            
            # Test account info
            try:
                account_info = client.get_account_info()
                print(f"   📊 Account: {account_info.get('account_id', 'Unknown')}")
                print(f"   💰 Buying Power: ${account_info.get('buying_power', 0):,.2f}")
                print(f"   🎭 Mode: {account_info.get('mode', 'Unknown')}")
            except Exception as e:
                print(f"   ⚠️  Account info failed: {e}")
            
            client.disconnect()
            return True
            
        elif connected and client.simulation_mode:
            print("   ⚠️  Connected in SIMULATION mode")
            print("   💡 This means IBKR Workstation is not properly configured")
            client.disconnect()
            return False
            
        else:
            print("   ❌ Connection failed completely")
            return False
            
    except Exception as e:
        print(f"   ❌ API connection test failed: {e}")
        return False

def print_setup_instructions():
    """Print detailed setup instructions"""
    print("\n📋 IBKR PAPER TRADING WORKSTATION SETUP")
    print("=" * 50)
    
    wsl_ip = os.getenv('IBKR_WSL_IP', '172.24.46.63')
    
    print("""
🎯 REQUIRED IBKR WORKSTATION CONFIGURATION:

1. 📱 LAUNCH IBKR WORKSTATION
   • Open Interactive Brokers Workstation on Windows
   • Switch to PAPER TRADING mode (important!)
   • Verify "Paper" appears in title bar

2. ⚙️ ENABLE API ACCESS
   • Go to: File → Global Configuration → API → Settings
   • ✅ Enable ActiveX and Socket Clients: CHECKED
   • ✅ Socket port: 7497
   • ✅ Master API client ID: 0 (or leave BLANK)
   • ✅ Read-Only API: UNCHECKED (must allow read/write)
   • ✅ Allow connections from localhost: CHECKED

3. 🔒 ADD TRUSTED IP ADDRESS
   • In the same API Settings window:
   • ✅ Trusted IPs: Add this IP address:""")
    
    print(f"      {wsl_ip}")
    
    print("""
4. 🛡️ SECURITY SETTINGS
   • Go to: File → Global Configuration → API → Precautions
   • ✅ Bypass Order Precautions for API orders: CHECKED
   • (This allows automated trading)

5. 🔄 RESTART WORKSTATION
   • Close IBKR Workstation completely
   • Reopen and switch back to Paper Trading mode
   • Verify API status shows "Listening on port 7497"

6. ✅ VERIFY SETUP
   • Look for "API" indicator in bottom status bar
   • Should show green "API" with port 7497
   • If red, API is not properly enabled

💡 TROUBLESHOOTING TIPS:
   • Make sure Windows Firewall allows IBKR Workstation
   • Ensure you're in PAPER TRADING mode (not live)
   • The trusted IP must match your WSL IP exactly
   • Restart Workstation after any configuration changes
""")

def main():
    """Run comprehensive IBKR setup check"""
    print("🔍 IBKR PAPER TRADING SETUP CHECKER")
    print("=" * 50)
    print("Checking your IBKR Paper Trading Workstation setup...")
    print()
    
    # Run all checks
    checks = [
        ("Network Connectivity", check_network_connectivity),
        ("WSL Configuration", check_wsl_configuration),
        ("IBKR API Connection", check_ibkr_api_connection),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"   ❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP CHECK SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {check_name}")
    
    print(f"\n🎯 Overall Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 EXCELLENT! Your IBKR setup is working correctly!")
        print("   You can now run real AI trading with IBKR Paper Trading.")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Run: python working_ai_trader.py")
        print("   2. Watch orders appear in IBKR Workstation")
        print("   3. Monitor real paper trading activity")
        
    elif results.get("Network Connectivity", False):
        print("\n⚠️  IBKR Workstation is running but API is not configured properly.")
        print_setup_instructions()
        
    else:
        print("\n❌ IBKR Workstation is not running or not accessible.")
        print("\n💡 IMMEDIATE ACTIONS:")
        print("   1. Start IBKR Paper Trading Workstation on Windows")
        print("   2. Switch to Paper Trading mode")
        print("   3. Run this checker again")
        print_setup_instructions()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)