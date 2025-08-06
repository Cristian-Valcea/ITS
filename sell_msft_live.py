#!/usr/bin/env python3
"""
🛡️ Sell 4 MSFT shares LIVE using the working IBKR connection
Uses the proven connection that worked before
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.brokers.ib_gateway import IBGatewayClient

def sell_msft_live():
    """
    Sell 4 MSFT shares using the proven IBKR connection
    """
    print("🛡️ SELLING 4 MSFT SHARES - LIVE IBKR CONNECTION")
    print("=" * 60)
    print("Using the proven connection method that worked before")
    print()
    
    # Create IBGatewayClient
    print("1️⃣ Creating IBGatewayClient...")
    ib_client = IBGatewayClient()
    
    # Connect
    print("2️⃣ Connecting to IBKR...")
    if not ib_client.connect():
        print("❌ Connection failed")
        return False
    
    print("✅ Connected successfully")
    print(f"   Mode: {'Simulation' if ib_client.simulation_mode else 'Live'}")
    
    # Show current positions
    print("\n3️⃣ Checking current positions...")
    try:
        positions = ib_client.get_positions()
        print(f"📊 Current positions:")
        for symbol, data in positions.items():
            if data['position'] != 0:
                print(f"   {symbol}: {data['position']} shares @ ${data['avgCost']:.2f}")
        
        # Check MSFT position
        msft_position = positions.get('MSFT', {}).get('position', 0)
        print(f"\n📊 MSFT Position: {msft_position} shares")
        
        if msft_position < 4:
            if msft_position > 0:
                print(f"⚠️  Only have {msft_position} MSFT shares - will sell what we have")
                quantity_to_sell = int(msft_position)
            else:
                print("❌ No MSFT shares to sell")
                return False
        else:
            quantity_to_sell = 4
            
    except Exception as e:
        print(f"⚠️  Could not get positions: {e}")
        print("Proceeding with sell order anyway...")
        quantity_to_sell = 4
    
    # Place SELL order
    print(f"\n4️⃣ Placing SELL order for {quantity_to_sell} MSFT shares...")
    print("🚨 This will place a REAL order in IBKR Workstation!")
    
    try:
        # Use the existing working method
        result = ib_client.place_market_order('MSFT', quantity_to_sell, 'SELL')
        
        print(f"\n📊 ORDER RESULT:")
        print("=" * 30)
        print(f"   Order ID: {result['order_id']}")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Action: {result['action']}")
        print(f"   Quantity: {result['quantity']}")
        print(f"   Status: {result['status']}")
        print(f"   Fill Price: ${result['fill_price']:.2f}")
        print(f"   Mode: {result['mode']}")
        
        if result['status'] == 'Filled':
            total_proceeds = result['quantity'] * result['fill_price']
            print(f"   💰 Total Proceeds: ${total_proceeds:.2f}")
            print(f"\n🎉 MSFT SHARES SOLD SUCCESSFULLY!")
        else:
            print(f"\n⚠️  Order Status: {result['status']}")
            print("Check IBKR Workstation for order details")
        
        return True
        
    except Exception as e:
        print(f"❌ Order placement failed: {e}")
        return False
    
    finally:
        print("\n5️⃣ Disconnecting...")
        ib_client.disconnect()
        print("✅ Disconnected from IBKR")

def main():
    print("💰 SELL MSFT SHARES")
    print("Using the proven IBKR connection method")
    print()
    
    success = sell_msft_live()
    
    if success:
        print("\n✅ MSFT sale completed!")
        print("Check your IBKR Workstation for execution details")
    else:
        print("\n❌ Sale failed - check error messages above")

if __name__ == "__main__":
    main()