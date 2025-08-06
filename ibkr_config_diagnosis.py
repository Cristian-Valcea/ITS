#!/usr/bin/env python3
"""
IBKR Configuration Diagnosis - Specific guidance for API setup
"""

import os
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_network_info():
    """Get WSL network information"""
    try:
        # Get WSL IP
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        wsl_ip = result.stdout.strip().split()[0] if result.stdout else "Unknown"
        
        # Get Windows host IP
        result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
        gateway_ip = "Unknown"
        if result.stdout:
            parts = result.stdout.strip().split()
            if len(parts) > 2:
                gateway_ip = parts[2]
        
        return wsl_ip, gateway_ip
    except:
        return "Unknown", "Unknown"

def main():
    """Provide specific IBKR configuration guidance"""
    
    print("🔧 IBKR API CONFIGURATION DIAGNOSIS")
    print("=" * 50)
    
    # Get network info
    wsl_ip, gateway_ip = get_network_info()
    configured_ip = os.getenv('IBKR_HOST_IP', '172.24.32.1')
    
    print(f"📡 Network Information:")
    print(f"   WSL IP Address: {wsl_ip}")
    print(f"   Windows Host IP: {gateway_ip}")
    print(f"   Configured Target: {configured_ip}")
    print()
    
    # Check if IPs match
    if gateway_ip != "Unknown" and configured_ip != gateway_ip:
        print("⚠️  IP MISMATCH DETECTED!")
        print(f"   Your .env file has: {configured_ip}")
        print(f"   But Windows host is: {gateway_ip}")
        print(f"   Consider updating IBKR_HOST_IP to {gateway_ip}")
        print()
    
    print("🎯 DIAGNOSIS: Socket connects but API handshake times out")
    print("   This means IBKR Workstation is running but API is misconfigured")
    print()
    
    print("📋 REQUIRED IBKR WORKSTATION CONFIGURATION:")
    print("=" * 50)
    print()
    
    print("1. 🖥️  OPEN IBKR WORKSTATION")
    print("   • Launch Interactive Brokers Workstation on Windows")
    print("   • Make sure you're in PAPER TRADING mode")
    print("   • Look for 'Paper' in the title bar")
    print()
    
    print("2. ⚙️  ENABLE API ACCESS")
    print("   • Go to: File → Global Configuration → API → Settings")
    print("   • Make these exact changes:")
    print()
    print("   ✅ Enable ActiveX and Socket Clients: ☑️  CHECKED")
    print("   ✅ Socket port: 7497")
    print("   ✅ Master API client ID: 0 (or leave BLANK)")
    print("   ✅ Read-Only API: ☐ UNCHECKED (must allow read/write)")
    print("   ✅ Allow connections from localhost: ☑️  CHECKED")
    print()
    
    print("3. 🔒 ADD TRUSTED IP ADDRESS")
    print("   • In the same API Settings window:")
    print("   • Find 'Trusted IPs' section")
    print(f"   • Add this EXACT IP address: {wsl_ip}")
    print("   • Click 'Add' or press Enter")
    print("   • Verify it appears in the list")
    print()
    
    print("4. 🛡️  SECURITY SETTINGS")
    print("   • Go to: File → Global Configuration → API → Precautions")
    print("   • Find 'Bypass Order Precautions for API orders'")
    print("   • ✅ Bypass Order Precautions for API orders: ☑️  CHECKED")
    print("   • (This allows automated trading)")
    print()
    
    print("5. 🔄 RESTART WORKSTATION")
    print("   • Close IBKR Workstation COMPLETELY")
    print("   • Wait 10 seconds")
    print("   • Reopen IBKR Workstation")
    print("   • Switch back to Paper Trading mode")
    print()
    
    print("6. ✅ VERIFY API STATUS")
    print("   • Look at the bottom status bar of IBKR Workstation")
    print("   • You should see a GREEN 'API' indicator")
    print("   • It should show 'Listening on port 7497'")
    print("   • If it's RED or missing, API is not enabled")
    print()
    
    print("🚨 COMMON MISTAKES TO AVOID:")
    print("=" * 30)
    print("❌ Using Live Trading mode instead of Paper Trading")
    print("❌ Wrong port number (7496 is live, 7497 is paper)")
    print("❌ Forgetting to add WSL IP to Trusted IPs")
    print("❌ Not restarting Workstation after changes")
    print("❌ Having Read-Only API enabled")
    print("❌ Windows Firewall blocking connections")
    print()
    
    print("🧪 AFTER CONFIGURATION:")
    print("=" * 25)
    print("Run this test to verify:")
    print("   python src/brokers/ib_gateway.py --test real-test --verbose")
    print()
    print("You should see:")
    print("   ✅ Real IBKR connection successful!")
    print("   👤 Accounts: ['DU8009825']")
    print("   📊 Server version: 176")
    print()
    
    print("💡 TROUBLESHOOTING TIPS:")
    print("=" * 25)
    print("• If still timing out: Check Windows Firewall")
    print("• If connection refused: Check IBKR is running")
    print("• If wrong account: Make sure you're in Paper mode")
    print("• If permission denied: Check API precautions")
    print()
    
    print("🎯 KEY POINTS:")
    print(f"   • Your WSL IP: {wsl_ip} (add to Trusted IPs)")
    print(f"   • Target IP: {configured_ip} (Windows host)")
    print("   • Port: 7497 (Paper Trading)")
    print("   • Mode: Paper Trading (not Live)")
    print()
    
    print("Once configured, you'll be able to run real AI trading!")

if __name__ == "__main__":
    main()