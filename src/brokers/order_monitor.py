#!/usr/bin/env python3
"""
🛡️ IBKR Order Status Monitor
Fixes the CRITICAL safety issue where orders are placed without proper monitoring
"""

import time
import logging
from typing import Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderStatusMonitor:
    """
    🚨 CRITICAL SAFETY: Properly monitor IBKR order status
    
    This fixes the scary issue where:
    - Orders are placed but status is misinterpreted  
    - "PreSubmitted" was treated as failure when it means LIVE ORDER
    - Real money trades happen without awareness
    """
    
    # IBKR Order Status Reference (CRITICAL KNOWLEDGE)
    STATUS_MEANINGS = {
        'PendingSubmit': ('🟡', 'Order created but not sent to exchange', False, False),
        'PendingCancel': ('🟡', 'Cancel request sent but not confirmed', True, False),
        'PreSubmitted': ('🟢', '⚠️ ORDER IS LIVE! Waiting for market open', True, False),
        'Submitted': ('🟢', '🚨 ORDER IS ACTIVE! Live in the market', True, False),
        'Cancelled': ('🔴', 'Order cancelled', False, False),
        'Filled': ('✅', '🎯 ORDER EXECUTED! Position changed', False, True),
        'Inactive': ('⚪', 'Order inactive (outside trading hours)', False, False),
        'ApiCancelled': ('🔴', 'Cancelled by API', False, False),
        'ApiPending': ('🟡', 'API processing order', True, False)
    }
    
    @classmethod
    def interpret_status(cls, status: str) -> Tuple[str, str, bool, bool]:
        """
        🚨 CRITICAL: Properly interpret IBKR order status
        Returns: (emoji, description, is_live, is_filled)
        """
        return cls.STATUS_MEANINGS.get(status, ('❓', f'Unknown: {status}', False, False))
    
    @classmethod
    def monitor_order_realtime(cls, trade, timeout_seconds: int = 30, check_interval: float = 1.0) -> Dict:
        """
        🛡️ ULTRA-SAFE: Monitor order in real-time with proper interpretation
        
        This is the fix for the scary blind trading issue!
        """
        
        order_id = trade.order.orderId
        start_time = time.time()
        
        logger.warning(f"🚨 MONITORING ORDER {order_id} - REAL-TIME TRACKING")
        print(f"\n🚨 ORDER {order_id} REAL-TIME MONITORING:")
        print("=" * 50)
        
        status_history = []
        
        for check_num in range(int(timeout_seconds / check_interval)):
            current_status = trade.orderStatus.status
            emoji, description, is_live, is_filled = cls.interpret_status(current_status)
            
            # Log each status check
            elapsed = time.time() - start_time
            check_msg = f"[{elapsed:5.1f}s] {current_status} {emoji} {description}"
            print(f"   {check_msg}")
            logger.info(f"Order {order_id}: {check_msg}")
            
            status_history.append({
                'timestamp': datetime.now().isoformat(),
                'elapsed': elapsed,
                'status': current_status,
                'is_live': is_live,
                'is_filled': is_filled
            })
            
            # CRITICAL ALERTS
            if is_live:
                print(f"   🔴 ALERT: ORDER IS LIVE IN MARKET!")
                logger.warning(f"Order {order_id} is LIVE in market: {current_status}")
            
            if is_filled:
                filled_qty = trade.orderStatus.filled
                fill_price = trade.orderStatus.avgFillPrice
                print(f"   🎯 ORDER FILLED: {filled_qty} shares @ ${fill_price}")
                logger.warning(f"Order {order_id} FILLED: {filled_qty} @ ${fill_price}")
                break
            
            if current_status in ['Cancelled', 'ApiCancelled']:
                print(f"   🛑 ORDER CANCELLED: {current_status}")
                logger.info(f"Order {order_id} cancelled: {current_status}")
                break
            
            time.sleep(check_interval)
        
        # Final status
        final_status = trade.orderStatus.status
        final_emoji, final_desc, final_live, final_filled = cls.interpret_status(final_status)
        
        result = {
            'order_id': order_id,
            'final_status': final_status,
            'final_description': final_desc,
            'is_live': final_live,
            'is_filled': final_filled,
            'filled_quantity': trade.orderStatus.filled if final_filled else 0,
            'fill_price': trade.orderStatus.avgFillPrice if final_filled else 0,
            'status_history': status_history,
            'monitoring_duration': time.time() - start_time
        }
        
        print(f"\n📊 FINAL STATUS: {final_status} {final_emoji} {final_desc}")
        if final_filled:
            print(f"   💰 EXECUTION: {result['filled_quantity']} @ ${result['fill_price']}")
        
        logger.warning(f"Order {order_id} monitoring complete: {final_status}")
        
        return result