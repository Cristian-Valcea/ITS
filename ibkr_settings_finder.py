#!/usr/bin/env python3
"""
IBKR Settings Finder - Help locate the correct API settings
"""

def print_ibkr_settings_guide():
    """Print comprehensive guide for finding IBKR API settings"""
    
    print("🔍 IBKR API SETTINGS LOCATION GUIDE")
    print("=" * 60)
    print()
    
    print("📍 PRIMARY LOCATION:")
    print("   File → Global Configuration → API → Settings")
    print()
    
    print("🔧 ESSENTIAL SETTINGS TO FIND:")
    print("=" * 40)
    
    print("\n1. ✅ BASIC API SETTINGS:")
    print("   □ Enable ActiveX and Socket Clients")
    print("   □ Socket port: 7497")
    print("   □ Master API client ID: 0 (or blank)")
    print("   □ Trusted IPs: 172.24.46.63")
    print()
    
    print("2. 🚨 READ-ONLY SETTING (CRITICAL):")
    print("   □ Read-Only API: UNCHECKED")
    print("   □ OR: 'Create API message log file': Can be checked")
    print("   □ OR: Look for any 'Read Only' checkbox - UNCHECK it")
    print()
    
    print("3. 🔐 SECURITY/ORDER SETTINGS:")
    print("   Look in these locations:")
    print("   □ Same API Settings page")
    print("   □ File → Global Configuration → Trading")
    print("   □ File → Global Configuration → Orders")
    print("   □ Account → Settings → Trading Permissions")
    print()
    
    print("   Settings to look for (may have different names):")
    print("   □ 'Bypass Order Precautions for API orders'")
    print("   □ 'Allow API orders'")
    print("   □ 'Enable API order placement'")
    print("   □ 'API order confirmations': Disable/Bypass")
    print("   □ 'Precautionary settings for API': Disable")
    print()
    
    print("📋 ALTERNATIVE LOCATIONS TO CHECK:")
    print("=" * 40)
    print("1. File → Global Configuration → Trading")
    print("2. File → Global Configuration → Orders")
    print("3. Account → Settings (if available)")
    print("4. Trading → Order Defaults")
    print("5. Configure → Account Settings")
    print()
    
    print("🎯 WHAT TO DO IF SETTING NOT FOUND:")
    print("=" * 40)
    print("1. The setting might not exist in your TWS version")
    print("2. It might be enabled by default")
    print("3. Try connecting without it first")
    print("4. Check TWS version - newer versions may have different settings")
    print()
    
    print("🔄 STEP-BY-STEP PROCESS:")
    print("=" * 40)
    print("1. Focus on the ESSENTIAL settings first:")
    print("   - Enable ActiveX and Socket Clients ✅")
    print("   - Socket port: 7497 ✅")
    print("   - Read-Only API: UNCHECKED ✅")
    print("   - Trusted IPs: 172.24.46.63 ✅")
    print()
    print("2. Click OK and restart TWS")
    print("3. Test connection with debug script")
    print("4. If still fails, look for order/security settings")
    print()
    
    print("🧪 TEST AFTER EACH CHANGE:")
    print("   python debug_ibkr_connection.py")
    print()
    
    print("📞 COMMON TWS VERSIONS & SETTINGS:")
    print("=" * 40)
    print("• TWS 10.19+: Settings usually in API tab")
    print("• TWS 10.12-10.18: May be in Trading tab")
    print("• Older versions: Check Account settings")
    print()
    
    print("🎯 PRIORITY ORDER:")
    print("1. Enable ActiveX and Socket Clients")
    print("2. Set port to 7497")
    print("3. Add WSL IP to Trusted IPs")
    print("4. UNCHECK any 'Read-Only' options")
    print("5. Restart TWS completely")
    print("6. Test connection")
    print("7. If fails, look for order precaution settings")

def main():
    print_ibkr_settings_guide()
    
    print("\n" + "=" * 60)
    print("🎯 IMMEDIATE ACTION PLAN:")
    print("1. Check the ESSENTIAL settings first")
    print("2. Don't worry about 'Bypass Order Precautions' for now")
    print("3. Test connection after each change")
    print("4. Report back what settings you can find")
    print("=" * 60)

if __name__ == "__main__":
    main()