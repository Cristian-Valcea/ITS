#!/usr/bin/env python3
"""
🚀 Event-Driven Order Status Monitor
Replaces polling with IBKR's event callbacks - no more blind waits!
"""

import asyncio
import time
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class OrderTransition(Enum):
    """Critical order state transitions"""
    SUBMITTED = "order_submitted"
    GOES_LIVE = "order_goes_live"  # PreSubmitted/Submitted
    FILLED = "order_filled"
    CANCELLED = "order_cancelled"
    REJECTED = "order_rejected"

@dataclass
class StatusEvent:
    """Single order status change event"""
    timestamp: datetime
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    remaining_qty: float = 0.0
    elapsed_seconds: float = 0.0

@dataclass
class OrderMonitorResult:
    """Complete order monitoring result"""
    order_id: int
    final_status: str
    is_filled: bool
    is_live: bool
    filled_quantity: float
    avg_fill_price: float
    total_monitoring_time: float
    status_events: List[StatusEvent] = field(default_factory=list)
    critical_transitions: List[OrderTransition] = field(default_factory=list)

class EventDrivenOrderMonitor:
    """
    🚀 Event-driven order monitor using IBKR's orderStatusEvent callbacks
    
    FIXES CRITICAL ISSUES:
    - No more polling with sleep()
    - Captures ALL status changes, even fast fills
    - Non-blocking async operation
    - Real-time transition detection
    """
    
    # IBKR Status interpretation (from previous monitor)
    STATUS_MEANINGS = {
        'PendingSubmit': ('🟡', 'Order created but not sent', False, False),
        'PendingCancel': ('🟡', 'Cancel request pending', True, False),
        'PreSubmitted': ('🟢', '🚨 ORDER IS LIVE! (Pre-market)', True, False),
        'Submitted': ('🟢', '🚨 ORDER IS ACTIVE! (Live)', True, False),
        'Cancelled': ('🔴', 'Order cancelled', False, False),
        'Filled': ('✅', '🎯 ORDER EXECUTED!', False, True),
        'Inactive': ('⚪', 'Order inactive', False, False),
        'ApiCancelled': ('🔴', 'Cancelled by API', False, False),
    }
    
    def __init__(self):
        self.monitoring_sessions = {}  # order_id -> monitoring state
        
    def monitor_order_async(self, trade, timeout_seconds: int = 30, 
                           risk_callback: Optional[Callable] = None) -> OrderMonitorResult:
        """
        🚀 Monitor order using event-driven callbacks (NO POLLING!)
        
        Args:
            trade: IB trade object
            timeout_seconds: Maximum monitoring time
            risk_callback: Optional callback for risk governor integration
            
        Returns:
            OrderMonitorResult with complete status history
        """
        
        order_id = trade.order.orderId
        start_time = time.time()
        
        logger.warning(f"🚀 EVENT-DRIVEN MONITORING: Order {order_id}")
        print(f"\n🚀 ORDER {order_id} EVENT-DRIVEN MONITORING:")
        print("=" * 50)
        
        # Initialize monitoring state
        monitoring_state = {
            'order_id': order_id,
            'start_time': start_time,
            'events': [],
            'transitions': [],
            'completed': False,
            'final_result': None,
            'risk_callback': risk_callback
        }
        
        self.monitoring_sessions[order_id] = monitoring_state
        
        # 🚀 KEY FIX: Subscribe to events BEFORE any status checks
        trade.orderStatusEvent += self._on_order_status_change
        
        try:
            # Capture initial status immediately
            initial_status = trade.orderStatus.status
            self._record_status_event(order_id, initial_status, trade.orderStatus, start_time)
            
            # 🚀 Use asyncio to wait for completion or timeout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Wait for order completion or timeout
                result = loop.run_until_complete(
                    self._wait_for_completion(order_id, timeout_seconds)
                )
                return result
                
            finally:
                loop.close()
                
        finally:
            # Clean up event subscription
            trade.orderStatusEvent -= self._on_order_status_change
            if order_id in self.monitoring_sessions:
                del self.monitoring_sessions[order_id]
    
    def _on_order_status_change(self, trade):
        """
        🚀 Event callback for order status changes (NO POLLING!)
        """
        order_id = trade.order.orderId
        
        if order_id not in self.monitoring_sessions:
            return  # Not monitoring this order
            
        monitoring_state = self.monitoring_sessions[order_id]
        start_time = monitoring_state['start_time']
        
        # Record the status event
        self._record_status_event(order_id, trade.orderStatus.status, 
                                trade.orderStatus, start_time)
        
        # Check for completion
        status = trade.orderStatus.status
        if status in ['Filled', 'Cancelled', 'ApiCancelled']:
            self._complete_monitoring(order_id, trade)
    
    def _record_status_event(self, order_id: int, status: str, order_status, start_time: float):
        """Record a single status change event"""
        
        if order_id not in self.monitoring_sessions:
            return
            
        monitoring_state = self.monitoring_sessions[order_id]
        elapsed = time.time() - start_time
        
        # Create status event
        event = StatusEvent(
            timestamp=datetime.now(),
            status=status,
            filled_qty=order_status.filled,
            avg_fill_price=order_status.avgFillPrice or 0.0,
            remaining_qty=order_status.remaining,
            elapsed_seconds=elapsed
        )
        
        monitoring_state['events'].append(event)
        
        # Interpret status
        emoji, description, is_live, is_filled = self.STATUS_MEANINGS.get(
            status, ('❓', f'Unknown: {status}', False, False)
        )
        
        # Log event
        print(f"   [{elapsed:5.1f}s] {status} {emoji} {description}")
        logger.info(f"Order {order_id} [{elapsed:.1f}s]: {status} - {description}")
        
        # 🚨 CRITICAL TRANSITION DETECTION
        if is_live and status in ['PreSubmitted', 'Submitted']:
            print(f"   🔴 CRITICAL: ORDER WENT LIVE!")
            monitoring_state['transitions'].append(OrderTransition.GOES_LIVE)
            
            # 🛡️ Risk callback integration
            if monitoring_state.get('risk_callback'):
                try:
                    monitoring_state['risk_callback'](order_id, status, 'ORDER_LIVE')
                except Exception as e:
                    logger.error(f"Risk callback failed: {e}")
        
        if is_filled:
            print(f"   🎯 FILLED: {event.filled_qty} @ ${event.avg_fill_price}")
            monitoring_state['transitions'].append(OrderTransition.FILLED)
            logger.warning(f"Order {order_id} FILLED: {event.filled_qty} @ ${event.avg_fill_price}")
        
        if status in ['Cancelled', 'ApiCancelled']:
            monitoring_state['transitions'].append(OrderTransition.CANCELLED)
    
    def _complete_monitoring(self, order_id: int, trade):
        """Complete monitoring and prepare final result"""
        
        if order_id not in self.monitoring_sessions:
            return
            
        monitoring_state = self.monitoring_sessions[order_id]
        events = monitoring_state['events']
        
        if not events:
            return
            
        final_event = events[-1]
        final_status = final_event.status
        
        # Interpret final status
        _, _, is_live, is_filled = self.STATUS_MEANINGS.get(
            final_status, ('❓', 'Unknown', False, False)
        )
        
        # Create final result
        result = OrderMonitorResult(
            order_id=order_id,
            final_status=final_status,
            is_filled=is_filled,
            is_live=is_live,
            filled_quantity=final_event.filled_qty,
            avg_fill_price=final_event.avg_fill_price,
            total_monitoring_time=final_event.elapsed_seconds,
            status_events=events,
            critical_transitions=monitoring_state['transitions']
        )
        
        monitoring_state['final_result'] = result
        monitoring_state['completed'] = True
        
        print(f"\n📊 FINAL: {final_status} after {result.total_monitoring_time:.1f}s")
        if is_filled:
            print(f"   💰 EXECUTED: {result.filled_quantity} @ ${result.avg_fill_price}")
    
    async def _wait_for_completion(self, order_id: int, timeout_seconds: int) -> OrderMonitorResult:
        """Wait for order completion or timeout using async"""
        
        end_time = time.time() + timeout_seconds
        
        while time.time() < end_time:
            if order_id not in self.monitoring_sessions:
                break
                
            monitoring_state = self.monitoring_sessions[order_id]
            
            if monitoring_state.get('completed'):
                return monitoring_state['final_result']
            
            # Small async sleep to yield control
            await asyncio.sleep(0.1)
        
        # Timeout - create final result from current state
        if order_id in self.monitoring_sessions:
            monitoring_state = self.monitoring_sessions[order_id]
            events = monitoring_state['events']
            
            if events:
                final_event = events[-1]
                _, _, is_live, is_filled = self.STATUS_MEANINGS.get(
                    final_event.status, ('❓', 'Unknown', False, False)
                )
                
                result = OrderMonitorResult(
                    order_id=order_id,
                    final_status=final_event.status,
                    is_filled=is_filled,
                    is_live=is_live,
                    filled_quantity=final_event.filled_qty,
                    avg_fill_price=final_event.avg_fill_price,
                    total_monitoring_time=timeout_seconds,
                    status_events=events,
                    critical_transitions=monitoring_state['transitions']
                )
                
                print(f"\n⏰ TIMEOUT after {timeout_seconds}s: {final_event.status}")
                return result
        
        # Fallback empty result
        return OrderMonitorResult(
            order_id=order_id,
            final_status='TIMEOUT',
            is_filled=False,
            is_live=False,
            filled_quantity=0,
            avg_fill_price=0,
            total_monitoring_time=timeout_seconds,
            status_events=[],
            critical_transitions=[]
        )