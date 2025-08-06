#!/usr/bin/env python3
"""
Comprehensive IBKR Troubleshooting
Tests various scenarios and provides detailed diagnostics
"""

import socket
import time
import subprocess
import sys
import os

def test_basic_connectivity():
    """Test basic TCP connectivity"""
    print("🔌 BASIC CONNECTIVITY TEST")
    print("-" * 30)
    
    host = "172.24.32.1"
    port = 7497
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ TCP connection to {host}:{port} successful")
            return True
        else:
            print(f"❌ TCP connection to {host}:{port} failed: {result}")
            return False
    except Exception as e:
        print(f"❌ TCP connection error: {e}")
        return False

def test_telnet_connection():
    """Test telnet-style connection"""
    print("\n🔗 TELNET-STYLE CONNECTION TEST")
    print("-" * 30)
    
    host = "172.24.32.1"
    port = 7497
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        print(f"✅ Socket connected to {host}:{port}")
        
        # Try to send a simple message and see what happens
        try:
            # Send a basic message (this might get rejected)
            sock.send(b"API\0")
            time.sleep(1)
            
            # Try to receive response
            sock.settimeout(2)
            response = sock.recv(1024)
            print(f"📨 Received response: {response}")
            
        except socket.timeout:
            print("⏰ No response received (timeout)")
        except Exception as e:
            print(f"📨 Communication error: {e}")
        
        sock.close()
        print("🔌 Socket closed")
        return True
        
    except Exception as e:
        print(f"❌ Socket connection failed: {e}")
        return False

def check_windows_firewall():
    """Check for potential Windows firewall issues"""
    print("\n🛡️ WINDOWS FIREWALL CHECK")
    print("-" * 30)
    
    print("🔍 Checking if Windows Firewall might be blocking:")
    print("   - WSL subnet: 172.24.0.0/16")
    print("   - Your WSL IP: 172.24.46.63")
    print("   - Target port: 7497")
    print()
    print("💡 To check Windows Firewall:")
    print("   1. Open Windows Defender Firewall")
    print("   2. Check 'Allow an app through firewall'")
    print("   3. Look for 'Trader Workstation' or 'TWS'")
    print("   4. Ensure both Private and Public are checked")
    print()
    print("🔧 Alternative: Temporarily disable Windows Firewall to test")

def check_ibkr_status():
    """Provide IBKR status checklist"""
    print("\n📊 IBKR WORKSTATION STATUS CHECKLIST")
    print("-" * 30)
    
    checklist = [
        "Is TWS showing 'Paper Trading' in the title?",
        "Is the account ID starting with 'DU' (demo)?",
        "Is there a green 'Connected' indicator?",
        "Are there any error messages in the message area?",
        "Is the API status showing 'Listening on port 7497'?",
        "Did you restart TWS after changing API settings?",
        "Are you logged into the correct paper trading account?"
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"   {i}. {item}")
    
    print("\n🔍 To check API status in TWS:")
    print("   - Look for API status indicator (usually bottom right)")
    print("   - Should show 'API: Listening on port 7497'")
    print("   - If red or showing error, restart TWS")

def check_alternative_ports():
    """Check if other IBKR ports are available"""
    print("\n🔍 ALTERNATIVE PORT CHECK")
    print("-" * 30)
    
    host = "172.24.32.1"
    ports_to_test = [
        (7497, "TWS Paper Trading"),
        (7496, "TWS Live Trading"),
        (4002, "IB Gateway Paper"),
        (4001, "IB Gateway Live"),
        (7462, "TWS Alternative"),
    ]
    
    for port, description in ports_to_test:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"   ✅ {port} ({description}) - OPEN")
            else:
                print(f"   ❌ {port} ({description}) - CLOSED")
        except:
            print(f"   ❌ {port} ({description}) - ERROR")

def provide_next_steps():
    """Provide next steps based on findings"""
    print("\n🎯 NEXT STEPS BASED ON FINDINGS")
    print("-" * 30)
    
    print("1. 🔧 IF TCP CONNECTION WORKS BUT API FAILS:")
    print("   → This is an IBKR API configuration issue")
    print("   → Check ALL API settings in TWS")
    print("   → Try different TWS version if available")
    print()
    
    print("2. 🛡️ IF TCP CONNECTION FAILS:")
    print("   → Windows Firewall is likely blocking")
    print("   → Check firewall settings")
    print("   → Try temporarily disabling firewall")
    print()
    
    print("3. 🔄 IF OTHER PORTS ARE OPEN:")
    print("   → Try connecting to those ports")
    print("   → Update IBKR_PORT in .env file")
    print("   → Test with different port")
    print()
    
    print("4. 📞 IBKR SUPPORT OPTIONS:")
    print("   → Check IBKR API documentation")
    print("   → Contact IBKR technical support")
    print("   → Check IBKR forums for similar issues")
    print()
    
    print("5. 🎭 FALLBACK OPTION:")
    print("   → Use simulation mode for now")
    print("   → Continue with paper trading development")
    print("   → Fix IBKR connection later")

def main():
    """Run comprehensive troubleshooting"""
    
    print("🔍 COMPREHENSIVE IBKR TROUBLESHOOTING")
    print("=" * 50)
    print("This script will test various connection scenarios")
    print("and provide detailed diagnostics for IBKR connection issues.")
    print()
    
    # Test basic connectivity
    tcp_works = test_basic_connectivity()
    
    # Test telnet-style connection
    if tcp_works:
        telnet_works = test_telnet_connection()
    
    # Check firewall
    check_windows_firewall()
    
    # Check IBKR status
    check_ibkr_status()
    
    # Check alternative ports
    check_alternative_ports()
    
    # Provide next steps
    provide_next_steps()
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY:")
    if tcp_works:
        print("✅ TCP connectivity works - Issue is with IBKR API configuration")
        print("🔧 Focus on TWS API settings and restart TWS")
    else:
        print("❌ TCP connectivity fails - Issue is with network/firewall")
        print("🛡️ Focus on Windows Firewall and network settings")
    
    print("\n💡 IMMEDIATE ACTION:")
    print("1. Check the items in the IBKR Status Checklist above")
    print("2. If TCP works, focus on API settings in TWS")
    print("3. If TCP fails, check Windows Firewall")
    print("4. Consider using simulation mode as fallback")

if __name__ == "__main__":
    main()