#!/usr/bin/env python3
"""
📊 Check ALL positions and account details in IBKR
"""

from ib_insync import *

def check_account_details():
    """Check complete account status"""
    
    print("📊 COMPLETE IBKR ACCOUNT CHECK")
    print("=" * 50)
    
    ib = IB()
    
    try:
        # Connect
        print("🔌 Connecting...")
        ib.connect('172.24.32.1', 7497, clientId=7, timeout=15)
        print("✅ Connected!")
        
        # Account info
        accounts = ib.managedAccounts()
        print(f"\n👤 Managed Accounts: {accounts}")
        
        # Account summary
        if accounts:
            account = accounts[0]
            print(f"\n💰 Account Summary for {account}:")
            
            try:
                summary = ib.accountSummary(account)
                for item in summary:
                    if item.tag in ['NetLiquidation', 'TotalCashValue', 'AvailableFunds', 'BuyingPower']:
                        print(f"   {item.tag}: {item.value} {item.currency}")
            except Exception as e:
                print(f"   Could not get account summary: {e}")
        
        # All positions
        print(f"\n📊 ALL POSITIONS:")
        positions = ib.positions()
        
        if positions:
            print(f"   Found {len(positions)} position(s):")
            for pos in positions:
                print(f"   📈 {pos.contract.symbol}: {pos.position} shares")
                print(f"      Average Cost: ${pos.avgCost:.2f}")
                print(f"      Market Value: ${pos.marketValue:.2f}")
                print(f"      Unrealized P&L: ${pos.unrealizedPNL:.2f}")
                print()
        else:
            print("   ❌ No positions found")
        
        # Portfolio items
        print(f"\n💼 PORTFOLIO ITEMS:")
        try:
            portfolio = ib.portfolio()
            if portfolio:
                print(f"   Found {len(portfolio)} portfolio item(s):")
                for item in portfolio:
                    print(f"   📊 {item.contract.symbol}: {item.position} shares")
                    print(f"      Market Price: ${item.marketPrice:.2f}")
                    print(f"      Market Value: ${item.marketValue:.2f}")
                    print()
            else:
                print("   ❌ No portfolio items found")
        except Exception as e:
            print(f"   Could not get portfolio: {e}")
        
        # Open orders
        print(f"\n📋 OPEN ORDERS:")
        trades = ib.openTrades()
        if trades:
            print(f"   Found {len(trades)} open order(s):")
            for trade in trades:
                print(f"   🔄 Order {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}")
                print(f"      Status: {trade.orderStatus.status}")
        else:
            print("   ✅ No open orders")
        
        # Recent executions
        print(f"\n⚡ RECENT EXECUTIONS (last 24h):")
        try:
            from datetime import datetime, timedelta
            yesterday = datetime.now() - timedelta(days=1)
            executions = ib.executions()
            
            if executions:
                recent_executions = [ex for ex in executions if ex.time >= yesterday.strftime('%Y%m%d')]
                if recent_executions:
                    print(f"   Found {len(recent_executions)} recent execution(s):")
                    for exec in recent_executions[-5:]:  # Last 5
                        print(f"   💥 {exec.contract.symbol}: {exec.execution.side} {exec.execution.shares} @ ${exec.execution.price:.2f}")
                        print(f"      Time: {exec.execution.time}")
                else:
                    print("   ❌ No recent executions")
            else:
                print("   ❌ No executions found")
        except Exception as e:
            print(f"   Could not get executions: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n🔌 Disconnected")

def main():
    print("🔍 COMPREHENSIVE ACCOUNT INSPECTION")
    print("Let's see what's actually in the IBKR paper trading account")
    print()
    
    success = check_account_details()
    
    if success:
        print("\n✅ Account inspection complete!")
        print("Now we can see exactly what positions exist")

if __name__ == "__main__":
    main()