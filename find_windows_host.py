#!/usr/bin/env python3
"""
Find Windows Host IP from WSL
Tests various methods to find the correct Windows host IP
"""

import subprocess
import socket
import os

def get_wsl_host_ip():
    """Get Windows host IP from WSL"""
    methods = []
    
    # Method 1: Default gateway
    try:
        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gateway = result.stdout.split()[2]
            methods.append(("Default Gateway", gateway))
    except:
        pass
    
    # Method 2: /etc/resolv.conf nameserver
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if line.startswith('nameserver'):
                    nameserver = line.split()[1]
                    methods.append(("Nameserver", nameserver))
                    break
    except:
        pass
    
    # Method 3: WSL_HOST environment variable (if set)
    wsl_host = os.environ.get('WSL_HOST')
    if wsl_host:
        methods.append(("WSL_HOST env", wsl_host))
    
    # Method 4: Common WSL patterns
    try:
        # Get our IP and derive host IP
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            our_ip = result.stdout.strip().split()[0]
            # WSL typically uses .1 as the Windows host
            ip_parts = our_ip.split('.')
            if len(ip_parts) == 4:
                host_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.1"
                methods.append(("Derived Host (.1)", host_ip))
    except:
        pass
    
    return methods

def test_port_connectivity(host, port=7497):
    """Test if port is open on host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    print("🔍 FINDING WINDOWS HOST IP FROM WSL")
    print("=" * 50)
    
    # Get our WSL IP
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        wsl_ip = result.stdout.strip().split()[0]
        print(f"📍 WSL IP: {wsl_ip}")
    except:
        wsl_ip = "unknown"
        print("📍 WSL IP: unknown")
    
    print(f"🎯 Testing port 7497 connectivity...\n")
    
    # Get potential host IPs
    host_methods = get_wsl_host_ip()
    
    working_ips = []
    
    for method, ip in host_methods:
        print(f"📡 Testing {method}: {ip}")
        if test_port_connectivity(ip, 7497):
            print(f"   ✅ Port 7497 OPEN on {ip}")
            working_ips.append(ip)
        else:
            print(f"   ❌ Port 7497 CLOSED on {ip}")
    
    print(f"\n" + "=" * 50)
    
    if working_ips:
        print(f"🎯 WORKING IPs FOUND:")
        for ip in working_ips:
            print(f"   ✅ {ip}")
        
        best_ip = working_ips[0]
        print(f"\n🎯 RECOMMENDED IP: {best_ip}")
        print(f"\n📝 UPDATE COMMANDS:")
        print(f"   1. Update .env file:")
        print(f"      IBKR_HOST_IP={best_ip}")
        print(f"\n   2. Add to IBKR Trusted IPs:")
        print(f"      WSL IP: {wsl_ip}")
        print(f"      Host IP: {best_ip}")
        
    else:
        print("❌ NO WORKING IPs FOUND")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure IBKR Workstation is running")
        print("2. Check API is enabled (File → Global Config → API)")
        print("3. Verify port 7497 is set for paper trading")
        print("4. Add these IPs to IBKR Trusted IPs:")
        for method, ip in host_methods:
            print(f"   - {ip} ({method})")
        print(f"   - {wsl_ip} (WSL IP)")

if __name__ == "__main__":
    main()