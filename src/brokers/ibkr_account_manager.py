#!/usr/bin/env python3
"""
🧹 IBKR Account Manager
Comprehensive paper trading account management and reset functionality
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ib_insync import IB, MarketOrder, Contract, Position, Trade, util
import pandas as pd

logger = logging.getLogger(__name__)

class IBKRAccountManager:
    """
    🛡️ IBKR Paper Trading Account Manager
    
    Features:
    - Check current positions and P&L
    - Flatten all positions (clean slate)
    - Cancel all pending orders
    - Account reset for clean testing
    - Position and trade monitoring
    """
    
    def __init__(self, host='172.24.32.1', port=7497, client_id=1):
        """Initialize IBKR Account Manager"""
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
    
    def connect(self) -> bool:
        """Connect to IBKR with canonical WSL method"""
        
        try:
            if not self.connected:
                logger.info(f"🔌 Connecting to IBKR: {self.host}:{self.port}")
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
                self.connected = True
                
                # Get account info
                accounts = self.ib.managedAccounts()
                server_version = self.ib.client.serverVersion()
                
                logger.info(f"✅ Connected to IBKR")
                logger.info(f"   Server version: {server_version}")
                logger.info(f"   Accounts: {accounts}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        
        if self.connected:
            try:
                self.ib.disconnect()
                self.connected = False
                logger.info("🔌 Disconnected from IBKR")
            except Exception as e:
                logger.error(f"⚠️ Disconnect error: {e}")
    
    def get_current_positions(self) -> List[Dict]:
        """Get all current positions with details"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        positions = []
        ibkr_positions = self.ib.positions()
        
        logger.info(f"📊 Found {len(ibkr_positions)} positions")
        
        for pos in ibkr_positions:
            if pos.position != 0:  # Only non-zero positions
                position_data = {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': pos.marketPrice,
                    'market_value': pos.marketValue,
                    'unrealized_pnl': pos.unrealizedPNL,
                    'realized_pnl': pos.realizedPNL,
                    'contract': pos.contract
                }
                positions.append(position_data)
                
                logger.info(f"   {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f} "
                          f"(Market: ${pos.marketPrice:.2f}, P&L: ${pos.unrealizedPNL:.2f})")
        
        return positions
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open/pending orders"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        orders = []
        open_orders = self.ib.openOrders()
        
        logger.info(f"📋 Found {len(open_orders)} open orders")
        
        for order in open_orders:
            order_data = {
                'order_id': order.orderId,
                'symbol': order.contract.symbol,
                'action': order.action,
                'quantity': order.totalQuantity,
                'order_type': order.orderType,
                'limit_price': getattr(order, 'lmtPrice', None),
                'status': order.orderStatus.status if hasattr(order, 'orderStatus') else 'Unknown'
            }
            orders.append(order_data)
            
            logger.info(f"   Order {order.orderId}: {order.action} {order.totalQuantity} "
                      f"{order.contract.symbol} ({order.orderType})")
        
        return orders
    
    def cancel_all_orders(self) -> bool:
        """Cancel all pending orders"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        try:
            logger.info("🚫 Cancelling all open orders...")
            
            # Global cancel first
            self.ib.reqGlobalCancel()
            util.sleep(2)  # Allow cancellations to process
            
            # Individual order cancellations for any remaining
            open_orders = self.ib.openOrders()
            cancelled_count = 0
            
            for order in open_orders:
                try:
                    self.ib.cancelOrder(order)
                    cancelled_count += 1
                    logger.info(f"   ✅ Cancelled order {order.orderId}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to cancel order {order.orderId}: {e}")
            
            util.sleep(2)  # Allow final cancellations
            
            # Verify all orders cancelled
            remaining_orders = self.ib.openOrders()
            
            if len(remaining_orders) == 0:
                logger.info(f"✅ All orders cancelled successfully ({cancelled_count} orders)")
                return True
            else:
                logger.warning(f"⚠️ {len(remaining_orders)} orders still pending after cancellation")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to cancel orders: {e}")
            return False
    
    def flatten_all_positions(self) -> bool:
        """Flatten all positions to zero (clean slate)"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        try:
            logger.info("📉 Flattening all positions...")
            
            positions = self.ib.positions()
            non_zero_positions = [pos for pos in positions if pos.position != 0]
            
            if len(non_zero_positions) == 0:
                logger.info("✅ No positions to flatten - account already clean")
                return True
            
            logger.info(f"   Found {len(non_zero_positions)} positions to flatten")
            
            flatten_orders = []
            
            for pos in non_zero_positions:
                symbol = pos.contract.symbol
                quantity = pos.position
                
                if quantity != 0:
                    # Determine opposite action
                    action = 'SELL' if quantity > 0 else 'BUY'
                    abs_quantity = abs(quantity)
                    
                    logger.info(f"   🔄 Flattening {symbol}: {action} {abs_quantity} (current: {quantity})")
                    
                    # Create market order to flatten
                    order = MarketOrder(action, abs_quantity)
                    
                    try:
                        trade = self.ib.placeOrder(pos.contract, order)
                        flatten_orders.append(trade)
                        logger.info(f"   ✅ Flatten order placed: {symbol} {action} {abs_quantity}")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Failed to place flatten order for {symbol}: {e}")
            
            # Wait for fills
            if flatten_orders:
                logger.info("⏳ Waiting for flatten orders to fill...")
                util.sleep(5)  # Allow time for fills
                
                # Check fill status
                filled_count = 0
                for trade in flatten_orders:
                    if trade.orderStatus.status in ['Filled', 'PartiallyFilled']:
                        filled_count += 1
                
                logger.info(f"📊 Flatten orders: {filled_count}/{len(flatten_orders)} filled")
            
            # Verify positions are flat
            util.sleep(2)  # Allow position updates
            final_positions = self.ib.positions()
            remaining_positions = [pos for pos in final_positions if pos.position != 0]
            
            if len(remaining_positions) == 0:
                logger.info("✅ All positions successfully flattened")
                return True
            else:
                logger.warning(f"⚠️ {len(remaining_positions)} positions still remain:")
                for pos in remaining_positions:
                    logger.warning(f"   {pos.contract.symbol}: {pos.position}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to flatten positions: {e}")
            return False
    
    def reset_paper_account(self) -> bool:
        """Complete account reset: cancel orders + flatten positions"""
        
        logger.info("🧹 STARTING PAPER ACCOUNT RESET")
        logger.info("=" * 50)
        
        success = True
        
        # Step 1: Cancel all orders
        logger.info("Step 1: Cancel all pending orders")
        if not self.cancel_all_orders():
            logger.error("❌ Failed to cancel all orders")
            success = False
        
        # Step 2: Flatten all positions
        logger.info("\nStep 2: Flatten all positions")
        if not self.flatten_all_positions():
            logger.error("❌ Failed to flatten all positions")
            success = False
        
        # Step 3: Verification
        logger.info("\nStep 3: Verification")
        util.sleep(3)  # Allow final updates
        
        final_positions = self.get_current_positions()
        final_orders = self.get_open_orders()
        
        logger.info(f"📊 Final state:")
        logger.info(f"   Positions: {len(final_positions)}")
        logger.info(f"   Open orders: {len(final_orders)}")
        
        if len(final_positions) == 0 and len(final_orders) == 0:
            logger.info("✅ PAPER ACCOUNT RESET SUCCESSFUL - CLEAN SLATE ACHIEVED")
            success = True
        else:
            logger.error("❌ PAPER ACCOUNT RESET INCOMPLETE")
            success = False
        
        logger.info("=" * 50)
        return success
    
    def get_pnl_summary(self) -> Dict:
        """Get comprehensive P&L summary"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        positions = self.get_current_positions()
        
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_positions': len(positions),
            'total_market_value': 0.0,
            'total_unrealized_pnl': 0.0,
            'total_realized_pnl': 0.0,
            'positions': positions
        }
        
        for pos in positions:
            summary['total_market_value'] += pos.get('market_value', 0.0)
            summary['total_unrealized_pnl'] += pos.get('unrealized_pnl', 0.0)
            summary['total_realized_pnl'] += pos.get('realized_pnl', 0.0)
        
        return summary
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        trades = []
        recent_trades = self.ib.trades()
        
        # Sort by most recent first
        recent_trades.sort(key=lambda t: getattr(t.log[-1], 'time', datetime.min) if t.log else datetime.min, reverse=True)
        
        for trade in recent_trades[:limit]:
            if trade.log:  # Has execution data
                trade_data = {
                    'order_id': trade.order.orderId,
                    'symbol': trade.contract.symbol,
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'filled_quantity': trade.orderStatus.filled,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'status': trade.orderStatus.status,
                    'time': trade.log[-1].time if trade.log else None
                }
                trades.append(trade_data)
        
        return trades

def main():
    """Command-line interface for account management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="IBKR Account Manager")
    parser.add_argument('--positions', action='store_true', help='Show current positions')
    parser.add_argument('--orders', action='store_true', help='Show open orders')
    parser.add_argument('--cancel-orders', action='store_true', help='Cancel all orders')
    parser.add_argument('--flatten', action='store_true', help='Flatten all positions')
    parser.add_argument('--reset', action='store_true', help='Full account reset')
    parser.add_argument('--pnl', action='store_true', help='Show P&L summary')
    parser.add_argument('--trades', action='store_true', help='Show recent trades')
    
    args = parser.parse_args()
    
    # Create account manager
    manager = IBKRAccountManager()
    
    try:
        if not manager.connect():
            print("❌ Failed to connect to IBKR")
            return 1
        
        if args.positions:
            positions = manager.get_current_positions()
            print(f"\n📊 CURRENT POSITIONS ({len(positions)} total):")
            if positions:
                for pos in positions:
                    print(f"   {pos['symbol']}: {pos['position']} @ ${pos['avg_cost']:.2f} "
                          f"(P&L: ${pos['unrealized_pnl']:.2f})")
            else:
                print("   ✅ No positions")
        
        if args.orders:
            orders = manager.get_open_orders()
            print(f"\n📋 OPEN ORDERS ({len(orders)} total):")
            if orders:
                for order in orders:
                    print(f"   {order['order_id']}: {order['action']} {order['quantity']} "
                          f"{order['symbol']} ({order['status']})")
            else:
                print("   ✅ No open orders")
        
        if args.cancel_orders:
            print("\n🚫 CANCELLING ALL ORDERS...")
            success = manager.cancel_all_orders()
            print(f"   {'✅ Success' if success else '❌ Failed'}")
        
        if args.flatten:
            print("\n📉 FLATTENING ALL POSITIONS...")
            success = manager.flatten_all_positions()
            print(f"   {'✅ Success' if success else '❌ Failed'}")
        
        if args.reset:
            print("\n🧹 FULL ACCOUNT RESET...")
            success = manager.reset_paper_account()
            print(f"   {'✅ Success' if success else '❌ Failed'}")
        
        if args.pnl:
            pnl = manager.get_pnl_summary()
            print(f"\n💰 P&L SUMMARY:")
            print(f"   Total Positions: {pnl['total_positions']}")
            print(f"   Market Value: ${pnl['total_market_value']:.2f}")
            print(f"   Unrealized P&L: ${pnl['total_unrealized_pnl']:.2f}")
            print(f"   Realized P&L: ${pnl['total_realized_pnl']:.2f}")
        
        if args.trades:
            trades = manager.get_recent_trades()
            print(f"\n📈 RECENT TRADES ({len(trades)} shown):")
            for trade in trades:
                print(f"   {trade['symbol']}: {trade['action']} {trade['filled_quantity']} "
                      f"@ ${trade['avg_fill_price']:.2f} ({trade['status']})")
        
        # If no specific action, show summary
        if not any([args.positions, args.orders, args.cancel_orders, args.flatten, 
                   args.reset, args.pnl, args.trades]):
            print("\n📊 ACCOUNT SUMMARY:")
            positions = manager.get_current_positions()
            orders = manager.get_open_orders()
            print(f"   Positions: {len(positions)}")
            print(f"   Open Orders: {len(orders)}")
            print(f"   Account Status: {'🟢 Clean' if len(positions) == 0 and len(orders) == 0 else '🟡 Active'}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    finally:
        manager.disconnect()

if __name__ == "__main__":
    import sys
    sys.exit(main())