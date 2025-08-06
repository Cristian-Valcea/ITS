#!/usr/bin/env python3
"""
🛡️ Direct MSFT Sale - Same method that worked for buying
Uses the exact same approach as the successful test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ib_insync import *

def sell_msft_direct():
    """
    Direct MSFT sale using the same method that worked for buying
    """
    print("🛡️ DIRECT MSFT SALE")
    print("=" * 40)
    print("Using the exact method that successfully placed buy orders")
    print()
    
    ib = IB()
    
    try:
        # Connect using the working IP
        print("🔌 Connecting to IBKR...")
        print("   Host: 172.24.32.1 (Windows host IP)")
        print("   Port: 7497 (Paper trading)")
        
        ib.connect('172.24.32.1', 7497, clientId=6, timeout=15)
        print("✅ Connected successfully!")
        
        # Show connection info
        print(f"📊 Server version: {ib.client.serverVersion()}")
        accounts = ib.managedAccounts()
        print(f"👤 Accounts: {accounts}")
        
        # Check positions first
        print("\n📊 Checking current positions...")
        positions = ib.positions()
        
        msft_position = 0
        for pos in positions:
            if pos.contract.symbol == 'MSFT':
                msft_position = pos.position
                print(f"   MSFT: {pos.position} shares @ ${pos.avgCost:.2f}")
                break
        
        if msft_position == 0:
            print("❌ No MSFT shares found in positions")
            print("   (The shares may have been sold already or are in a different account)")
            return False
        
        print(f"\n🎯 Found {msft_position} MSFT shares to sell")
        
        # Create MSFT contract
        print("\n📋 Creating MSFT contract...")
        stock = Stock('MSFT', 'SMART', 'USD')
        
        # Qualify contract
        ib.qualifyContracts(stock)
        print(f"✅ Contract qualified: {stock}")
        
        # Create SELL order
        quantity_to_sell = min(4, int(abs(msft_position)))  # Sell up to 4 shares
        print(f"\n💰 Creating SELL order for {quantity_to_sell} MSFT shares...")
        
        order = MarketOrder('SELL', quantity_to_sell)
        print(f"📋 Order: {order.action} {order.totalQuantity} shares")
        
        # Place the order
        print("\n🚨 PLACING REAL SELL ORDER...")
        trade = ib.placeOrder(stock, order)
        order_id = trade.order.orderId
        
        print(f"✅ Order placed with ID: {order_id}")
        print("👀 Monitoring order status...")
        
        # Monitor for longer time to see all status changes
        for i in range(20):  # 20 seconds
            ib.sleep(1)
            current_status = trade.orderStatus.status
            filled_qty = trade.orderStatus.filled
            avg_price = trade.orderStatus.avgFillPrice
            
            print(f"   [{i+1:2d}s] Status: {current_status}", end="")
            if filled_qty > 0:
                print(f" | Filled: {filled_qty} @ ${avg_price:.2f}", end="")
            print()
            
            # Check if order is complete
            if current_status in ['Filled', 'Cancelled', 'ApiCancelled']:
                break
        
        # Final status
        final_status = trade.orderStatus.status
        final_filled = trade.orderStatus.filled
        final_price = trade.orderStatus.avgFillPrice or 0
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Order ID: {order_id}")
        print(f"   Final Status: {final_status}")
        print(f"   Filled Quantity: {final_filled}")
        print(f"   Average Price: ${final_price:.2f}")
        
        if final_status == 'Filled':
            proceeds = final_filled * final_price
            print(f"   💰 Total Proceeds: ${proceeds:.2f}")
            print(f"\n🎉 MSFT SHARES SOLD SUCCESSFULLY!")
        else:
            print(f"\n⚠️  Order not filled: {final_status}")
            print("Check IBKR Workstation for details")
        
        return final_status == 'Filled'
        
    except Exception as e:
        print(f"❌ Error during sale: {e}")
        return False
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n🔌 Disconnected from IBKR")

def main():
    print("💰 DIRECT MSFT SALE TEST")
    print("Same proven method that successfully bought MSFT")
    print()
    
    success = sell_msft_direct()
    
    if success:
        print("\n✅ MSFT SALE COMPLETED!")
        print("🎯 The proven IBKR connection method works for both buying and selling")
    else:
        print("\n⚠️  Sale not completed - check status above")

if __name__ == "__main__":
    main()