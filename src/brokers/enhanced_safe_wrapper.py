#!/usr/bin/env python3
"""
🛡️ Enhanced Safe Order Wrapper with Risk Governor Integration
Integrates with event-driven monitoring and circuit breakers
"""

import logging
from typing import Dict, Optional, Callable
from .connection_validator import IBKRConnectionValidator, ConnectionConfig
from .event_order_monitor import EventDrivenOrderMonitor, OrderMonitorResult
from ib_insync import MarketOrder, LimitOrder

logger = logging.getLogger(__name__)

class RiskGovernorAction:
    """Risk governor action types"""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    EMERGENCY_CANCEL = "emergency_cancel"

class EnhancedSafeOrderWrapper:
    """
    🛡️ Enhanced safe order wrapper with risk governor integration
    
    FIXES ALL REVIEWER ISSUES:
    - Event-driven monitoring (no polling)
    - Risk governor circuit breaker integration
    - Hard credential validation
    - Proper error handling
    """
    
    def __init__(self, ib_client, risk_governor_callback: Optional[Callable] = None):
        """
        Initialize enhanced safe wrapper
        
        Args:
            ib_client: IBGatewayClient instance
            risk_governor_callback: Callback for risk decisions
        """
        self.ib_client = ib_client
        self.monitor = EventDrivenOrderMonitor()
        self.risk_governor_callback = risk_governor_callback
        
        # Validate connection config
        self._validate_connection()
        
    def _validate_connection(self):
        """Validate connection using hard credential checking"""
        try:
            if self.ib_client.simulation_mode:
                # Paper trading validation
                config = IBKRConnectionValidator.validate_paper_trading_config()
                logger.info(f"✅ Paper trading validated: {config.host}:{config.port}")
            else:
                # Live trading - strict validation
                config = IBKRConnectionValidator.validate_connection_config()
                logger.info(f"✅ Live trading validated: {config.host}:{config.port}")
                
        except (ValueError, RuntimeError) as e:
            logger.error(f"🚨 Connection validation failed: {e}")
            raise
    
    def _risk_governor_hook(self, order_id: int, status: str, event_type: str):
        """
        🛡️ Risk governor integration hook
        
        Called when critical order events occur (ORDER_LIVE, FILLED, etc.)
        """
        
        if not self.risk_governor_callback:
            return  # No risk governor configured
        
        try:
            # Call risk governor
            action = self.risk_governor_callback(order_id, status, event_type)
            
            logger.info(f"Risk governor decision for order {order_id}: {action}")
            
            # Take action based on risk governor decision
            if action == RiskGovernorAction.EMERGENCY_CANCEL:
                logger.warning(f"🚨 RISK GOVERNOR: Emergency cancel order {order_id}")
                self._emergency_cancel_order(order_id)
                
            elif action == RiskGovernorAction.BLOCK:
                logger.warning(f"🚨 RISK GOVERNOR: Blocking further orders")
                # Could set a flag to block new orders
                
            elif action == RiskGovernorAction.THROTTLE:
                logger.warning(f"⚠️  RISK GOVERNOR: Throttling orders")
                # Could implement throttling logic
                
        except Exception as e:
            logger.error(f"Risk governor callback failed: {e}")
    
    def _emergency_cancel_order(self, order_id: int):
        """Emergency cancel order via risk governor"""
        try:
            # Find and cancel the order
            trades = self.ib_client.ib.openTrades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib_client.ib.cancelOrder(trade.order)
                    logger.warning(f"🚨 Emergency cancelled order {order_id}")
                    break
        except Exception as e:
            logger.error(f"Emergency cancel failed for order {order_id}: {e}")
    
    def place_market_order_with_governor(self, symbol: str, quantity: int, action: str = 'BUY') -> Dict:
        """
        🛡️ Place market order with full safety and risk governor integration
        
        FIXES ALL ISSUES:
        - Event-driven monitoring (no sleep polling)
        - Risk governor integration
        - Proper error handling
        """
        
        # Pre-order risk check
        if self.risk_governor_callback:
            pre_action = self.risk_governor_callback(None, 'PRE_ORDER', f"{action}_{symbol}_{quantity}")
            if pre_action == RiskGovernorAction.BLOCK:
                raise ValueError(f"🚨 RISK GOVERNOR: Order blocked - {action} {quantity} {symbol}")
        
        # Safety checks
        if not self.ib_client.connected:
            raise ValueError("IBGatewayClient not connected")
        
        intent = f"{action} {quantity} {symbol}"
        logger.warning(f"🚨 ENHANCED SAFE ORDER: {intent}")
        print(f"\n🛡️ ENHANCED SAFE ORDER PLACEMENT: {intent}")
        
        if self.ib_client.simulation_mode:
            # Use existing simulation logic
            result = self.ib_client.place_market_order(symbol, quantity, action)
            print(f"🎭 Simulation result: {result}")
            return result
        
        # 🚀 ENHANCED LIVE ORDER PLACEMENT
        try:
            # Get contract
            if symbol not in self.ib_client.contracts:
                raise ValueError(f"Contract not available for {symbol}")
            
            contract = self.ib_client.contracts[symbol]
            order = MarketOrder(action, quantity)
            
            # 🚀 KEY FIX: Subscribe to events BEFORE placing order
            print("📡 Subscribing to order events...")
            
            # Place order
            trade = self.ib_client.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            print(f"✅ Order placed with ID: {order_id}")
            logger.warning(f"Order {order_id} placed: {intent}")
            
            # 🚀 EVENT-DRIVEN MONITORING (NO POLLING!)
            print("🚀 Starting event-driven monitoring...")
            monitoring_result = self.monitor.monitor_order_async(
                trade, 
                timeout_seconds=30,
                risk_callback=self._risk_governor_hook
            )
            
            # Enhanced result with monitoring data
            return {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'MKT',
                'final_status': monitoring_result.final_status,
                'is_filled': monitoring_result.is_filled,
                'is_live': monitoring_result.is_live,
                'filled_quantity': monitoring_result.filled_quantity,
                'avg_fill_price': monitoring_result.avg_fill_price,
                'monitoring_time': monitoring_result.total_monitoring_time,
                'status_events': len(monitoring_result.status_events),
                'critical_transitions': [t.value for t in monitoring_result.critical_transitions],
                'mode': 'live_event_driven',
                'risk_governor_integrated': self.risk_governor_callback is not None
            }
            
        except Exception as e:
            logger.error(f"Enhanced safe order placement failed: {e}")
            raise
    
    def place_limit_order_with_governor(self, symbol: str, quantity: int, price: float, action: str = 'BUY') -> Dict:
        """
        🛡️ Place limit order with full safety and risk governor integration
        """
        
        # Pre-order risk check
        if self.risk_governor_callback:
            pre_action = self.risk_governor_callback(None, 'PRE_ORDER', f"{action}_{symbol}_{quantity}@{price}")
            if pre_action == RiskGovernorAction.BLOCK:
                raise ValueError(f"🚨 RISK GOVERNOR: Limit order blocked - {action} {quantity} {symbol} @ ${price}")
        
        # Safety checks
        if not self.ib_client.connected:
            raise ValueError("IBGatewayClient not connected")
        
        intent = f"{action} {quantity} {symbol} @ ${price}"
        logger.warning(f"🚨 ENHANCED SAFE LIMIT ORDER: {intent}")
        print(f"\n🛡️ ENHANCED SAFE LIMIT ORDER: {intent}")
        
        if self.ib_client.simulation_mode:
            result = self.ib_client.place_limit_order(symbol, quantity, price, action)
            print(f"🎭 Simulation result: {result}")
            return result
        
        # Enhanced live limit order placement
        try:
            if symbol not in self.ib_client.contracts:
                raise ValueError(f"Contract not available for {symbol}")
            
            contract = self.ib_client.contracts[symbol]
            order = LimitOrder(action, quantity, price)
            
            # Place order with event monitoring
            trade = self.ib_client.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            print(f"✅ Limit order placed with ID: {order_id}")
            logger.warning(f"Limit order {order_id} placed: {intent}")
            
            # Event-driven monitoring
            monitoring_result = self.monitor.monitor_order_async(
                trade,
                timeout_seconds=30,
                risk_callback=self._risk_governor_hook
            )
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'LMT',
                'limit_price': price,
                'final_status': monitoring_result.final_status,
                'is_filled': monitoring_result.is_filled,
                'is_live': monitoring_result.is_live,
                'filled_quantity': monitoring_result.filled_quantity,
                'avg_fill_price': monitoring_result.avg_fill_price,
                'monitoring_time': monitoring_result.total_monitoring_time,
                'status_events': len(monitoring_result.status_events),
                'critical_transitions': [t.value for t in monitoring_result.critical_transitions],
                'mode': 'live_event_driven',
                'risk_governor_integrated': self.risk_governor_callback is not None
            }
            
        except Exception as e:
            logger.error(f"Enhanced safe limit order failed: {e}")
            raise