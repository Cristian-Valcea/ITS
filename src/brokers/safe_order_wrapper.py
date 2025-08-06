#!/usr/bin/env python3
"""
🛡️ Safe Order Wrapper for IBGatewayClient
Enhances existing order placement with proper monitoring
"""

import logging
from typing import Dict
from .ib_gateway import IBGatewayClient
from .order_monitor import OrderStatusMonitor

logger = logging.getLogger(__name__)

class SafeOrderWrapper:
    """
    🛡️ Safety wrapper that enhances IBGatewayClient with proper order monitoring
    
    Fixes the scary blind trading issue by adding real-time monitoring
    to existing order placement methods.
    """
    
    def __init__(self, ib_client: IBGatewayClient):
        self.ib_client = ib_client
        self.monitor = OrderStatusMonitor()
        
    def place_market_order_safely(self, symbol: str, quantity: int, action: str = 'BUY') -> Dict:
        """
        🛡️ SAFE market order placement with real-time monitoring
        
        Wraps the existing place_market_order with proper status tracking
        """
        
        # Safety check
        if not self.ib_client.connected:
            raise ValueError("IBGatewayClient not connected")
        
        # Log intent
        intent = f"{action} {quantity} {symbol}"
        logger.warning(f"🚨 PLACING SAFE ORDER: {intent}")
        print(f"\n🚨 SAFE ORDER PLACEMENT: {intent}")
        
        if self.ib_client.simulation_mode:
            # Use existing simulation logic
            return self.ib_client.place_market_order(symbol, quantity, action)
        
        # ENHANCED LIVE ORDER PLACEMENT
        try:
            # Get contract
            if symbol not in self.ib_client.contracts:
                raise ValueError(f"Contract not available for {symbol}")
            
            contract = self.ib_client.contracts[symbol]
            
            # Create order using ib_insync directly (bypass the 1-second wait)
            from ib_insync import MarketOrder
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib_client.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            print(f"✅ Order placed with ID: {order_id}")
            logger.warning(f"Order {order_id} placed: {intent}")
            
            # 🛡️ CRITICAL: Use proper monitoring instead of 1-second wait
            monitoring_result = self.monitor.monitor_order_realtime(trade, timeout_seconds=30)
            
            # Return enhanced result
            return {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'MKT',
                'status': monitoring_result['final_status'],
                'fill_price': monitoring_result['fill_price'],
                'filled_quantity': monitoring_result['filled_quantity'],
                'is_live': monitoring_result['is_live'],
                'is_filled': monitoring_result['is_filled'],
                'timestamp': monitoring_result['status_history'][0]['timestamp'] if monitoring_result['status_history'] else None,
                'mode': 'live_monitored',
                'monitoring_result': monitoring_result
            }
            
        except Exception as e:
            logger.error(f"Safe order placement failed: {e}")
            raise
    
    def place_limit_order_safely(self, symbol: str, quantity: int, price: float, action: str = 'BUY') -> Dict:
        """
        🛡️ SAFE limit order placement with real-time monitoring
        """
        
        # Safety check
        if not self.ib_client.connected:
            raise ValueError("IBGatewayClient not connected")
        
        # Log intent
        intent = f"{action} {quantity} {symbol} @ ${price}"
        logger.warning(f"🚨 PLACING SAFE LIMIT ORDER: {intent}")
        print(f"\n🚨 SAFE LIMIT ORDER PLACEMENT: {intent}")
        
        if self.ib_client.simulation_mode:
            # Use existing simulation logic
            return self.ib_client.place_limit_order(symbol, quantity, price, action)
        
        # ENHANCED LIVE ORDER PLACEMENT
        try:
            # Get contract
            if symbol not in self.ib_client.contracts:
                raise ValueError(f"Contract not available for {symbol}")
            
            contract = self.ib_client.contracts[symbol]
            
            # Create order using ib_insync directly
            from ib_insync import LimitOrder
            order = LimitOrder(action, quantity, price)
            
            # Place order
            trade = self.ib_client.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            print(f"✅ Limit order placed with ID: {order_id}")
            logger.warning(f"Limit order {order_id} placed: {intent}")
            
            # 🛡️ CRITICAL: Use proper monitoring
            monitoring_result = self.monitor.monitor_order_realtime(trade, timeout_seconds=30)
            
            # Return enhanced result
            return {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'LMT',
                'limit_price': price,
                'status': monitoring_result['final_status'],
                'fill_price': monitoring_result['fill_price'],
                'filled_quantity': monitoring_result['filled_quantity'],
                'is_live': monitoring_result['is_live'],
                'is_filled': monitoring_result['is_filled'],
                'timestamp': monitoring_result['status_history'][0]['timestamp'] if monitoring_result['status_history'] else None,
                'mode': 'live_monitored',
                'monitoring_result': monitoring_result
            }
            
        except Exception as e:
            logger.error(f"Safe limit order placement failed: {e}")
            raise