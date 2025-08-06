"""
End-of-Day Position Management System
Automatically flattens positions before market close with multiple safety layers
"""

import time
import logging
import threading
import schedule  
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import os
import sys

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk_governor.broker_adapter import BrokerExecutionManager, OrderStatus

@dataclass
class EODConfig:
    """End-of-day configuration"""
    warning_time: str = "15:50"      # First warning
    flatten_time: str = "15:55"      # Start flattening positions
    hard_cutoff: str = "15:59:30"    # Cancel all orders, force MOC
    market_close: str = "16:00"      # Market officially closes
    
    # Retry configuration
    max_flatten_attempts: int = 3
    flatten_retry_interval: int = 60  # seconds
    
    # Safety margins
    min_time_to_flatten: int = 300    # 5 minutes minimum
    emergency_flatten_threshold: int = 180  # 3 minutes for emergency

class EODPositionManager:
    """
    Manages end-of-day position flattening with multiple safety layers
    
    Timeline:
    15:50 - Warning notification
    15:55 - Begin systematic position flattening
    15:57 - Retry any failed flattens
    15:59:30 - Hard cutoff: cancel pending, force MOC orders
    16:00 - Market close
    """
    
    def __init__(self, broker_manager: BrokerExecutionManager, config: EODConfig = None):
        self.broker_manager = broker_manager
        self.config = config or EODConfig()
        self.logger = logging.getLogger("EODPositionManager")
        
        # State tracking
        self.eod_active = False
        self.positions_flattened = False
        self.warning_sent = False
        self.last_flatten_attempt = 0
        self.flatten_attempts = 0
        
        # Threading
        self.monitor_thread = None
        self.shutdown_event = threading.Event()
        
        # Callbacks for notifications
        self.warning_callback = None
        self.flatten_callback = None
        self.emergency_callback = None
        
        self.logger.info("EOD Position Manager initialized")
    
    def start_monitor(self):
        """Start the EOD monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.shutdown_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("EOD monitor started")
    
    def stop_monitor(self):
        """Stop the EOD monitoring thread"""
        self.shutdown_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("EOD monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                current_time_min = current_time[:5]  # HH:MM
                
                # Warning phase
                if not self.warning_sent and current_time_min >= self.config.warning_time:
                    self._send_warning()
                
                # Flatten phase
                if not self.positions_flattened and current_time_min >= self.config.flatten_time:
                    self._initiate_flatten()
                
                # Retry flatten if needed
                if (self.eod_active and not self.positions_flattened and 
                    time.time() - self.last_flatten_attempt > self.config.flatten_retry_interval):
                    self._retry_flatten()
                
                # Hard cutoff phase
                if current_time >= self.config.hard_cutoff:
                    self._emergency_flatten()
                    break  # Exit monitoring after hard cutoff
                
                # Check if market is closed
                if current_time_min >= self.config.market_close:
                    self.logger.info("Market closed, EOD monitoring complete")
                    break
                
                time.sleep(10)  # Check every 10 seconds during EOD period
                
            except Exception as e:
                self.logger.error(f"Error in EOD monitor loop: {e}")
                time.sleep(30)  # Wait longer after errors
    
    def _send_warning(self):
        """Send warning notification"""
        self.warning_sent = True
        message = f"EOD Warning: Position flattening will begin at {self.config.flatten_time}"
        
        self.logger.warning(message)
        
        if self.warning_callback:
            try:
                self.warning_callback(message)
            except Exception as e:
                self.logger.error(f"Warning callback error: {e}")
    
    def _initiate_flatten(self):
        """Initiate systematic position flattening"""
        self.eod_active = True
        self.last_flatten_attempt = time.time()
        self.flatten_attempts += 1
        
        self.logger.info("Initiating EOD position flattening")
        
        try:
            # Get current positions from broker
            open_orders = self.broker_manager.broker.get_open_orders()
            
            if not open_orders:
                self.positions_flattened = True
                self.logger.info("No open positions to flatten")
                return
            
            # Attempt to flatten each position
            flatten_results = []
            
            for order in open_orders:
                try:
                    # Cancel existing order first
                    self.broker_manager.broker.cancel_order(order.order_id)
                    
                    # Place opposite order to flatten
                    flatten_size = -order.quantity  # Opposite direction
                    
                    result = self.broker_manager.execute_safe_order(
                        symbol=order.symbol,
                        position_increment=flatten_size * self._estimate_price(order.symbol),
                        current_price=self._estimate_price(order.symbol),
                        max_latency_ms=2000.0  # Allow more time for EOD orders
                    )
                    
                    flatten_results.append({
                        "symbol": order.symbol,
                        "original_quantity": order.quantity,
                        "flatten_quantity": flatten_size,
                        "success": result["success"],
                        "message": result["message"]
                    })
                    
                    if result["success"]:
                        self.logger.info(f"Flattened {order.symbol}: {order.quantity} -> 0")
                    else:
                        self.logger.error(f"Failed to flatten {order.symbol}: {result['message']}")
                
                except Exception as e:
                    self.logger.error(f"Error flattening {order.symbol}: {e}")
                    flatten_results.append({
                        "symbol": order.symbol,
                        "success": False,
                        "message": str(e)
                    })
            
            # Check if all positions were successfully flattened
            successful_flattens = sum(1 for r in flatten_results if r["success"])
            total_positions = len(flatten_results)
            
            if successful_flattens == total_positions:
                self.positions_flattened = True
                self.logger.info(f"Successfully flattened all {total_positions} positions")
            else:
                self.logger.warning(f"Only flattened {successful_flattens}/{total_positions} positions")
            
            # Callback notification
            if self.flatten_callback:
                try:
                    self.flatten_callback(flatten_results)
                except Exception as e:
                    self.logger.error(f"Flatten callback error: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in flatten initiation: {e}")
    
    def _retry_flatten(self):
        """Retry flattening if previous attempts failed"""
        if self.flatten_attempts >= self.config.max_flatten_attempts:
            self.logger.error(f"Max flatten attempts ({self.config.max_flatten_attempts}) reached")
            return
        
        self.logger.info(f"Retrying position flatten (attempt {self.flatten_attempts + 1})")
        self._initiate_flatten()
    
    def _emergency_flatten(self):
        """Emergency flattening at hard cutoff"""
        self.logger.warning("Emergency flatten activated at hard cutoff")
        
        try:
            # Cancel ALL pending orders immediately
            open_orders = self.broker_manager.broker.get_open_orders()
            
            for order in open_orders:
                self.broker_manager.broker.cancel_order(order.order_id)
                self.logger.info(f"Cancelled order {order.order_id}")
            
            # Place MOC (Market-on-Close) orders for any remaining positions
            # Note: This would need to be implemented based on the actual broker API
            emergency_orders = self._place_emergency_moc_orders()
            
            self.logger.warning(f"Placed {len(emergency_orders)} emergency MOC orders")
            
            # Callback notification
            if self.emergency_callback:
                try:
                    self.emergency_callback({
                        "cancelled_orders": len(open_orders),
                        "emergency_orders": len(emergency_orders),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    self.logger.error(f"Emergency callback error: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in emergency flatten: {e}")
    
    def _place_emergency_moc_orders(self) -> List[Dict]:
        """Place Market-on-Close orders for emergency flattening"""
        # This is a simplified implementation
        # Real implementation would query actual positions from broker
        
        emergency_orders = []
        
        # For now, just ensure all pending orders are cancelled
        # In a real system, this would place MOC orders for any net positions
        
        return emergency_orders
    
    def _estimate_price(self, symbol: str) -> float:
        """Estimate current price for position calculations"""
        # Simplified price estimation
        # Real implementation would use real-time market data
        
        price_estimates = {
            "MSFT": 420.0,
            "AAPL": 195.0,
            "NVDA": 450.0,
            "TSLA": 250.0
        }
        
        return price_estimates.get(symbol, 100.0)
    
    def force_flatten_now(self) -> Dict:
        """Manual trigger for immediate position flattening"""
        self.logger.warning("Manual flatten triggered")
        
        self.eod_active = True
        self._initiate_flatten()
        
        return {
            "status": "initiated",
            "timestamp": time.time(),
            "positions_flattened": self.positions_flattened
        }
    
    def get_eod_status(self) -> Dict:
        """Get current EOD status"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        return {
            "current_time": current_time,
            "eod_active": self.eod_active,
            "warning_sent": self.warning_sent,
            "positions_flattened": self.positions_flattened,
            "flatten_attempts": self.flatten_attempts,
            "last_flatten_attempt": self.last_flatten_attempt,
            "config": {
                "warning_time": self.config.warning_time,
                "flatten_time": self.config.flatten_time,
                "hard_cutoff": self.config.hard_cutoff
            }
        }
    
    def set_callbacks(self, 
                     warning_callback=None, 
                     flatten_callback=None, 
                     emergency_callback=None):
        """Set notification callbacks"""
        self.warning_callback = warning_callback
        self.flatten_callback = flatten_callback  
        self.emergency_callback = emergency_callback
        
        self.logger.info("EOD callbacks configured")

class EODScheduler:
    """
    Scheduler for EOD operations with timezone awareness
    Handles market holidays and schedule adjustments
    """
    
    def __init__(self, eod_manager: EODPositionManager):
        self.eod_manager = eod_manager
        self.logger = logging.getLogger("EODScheduler")
        
        # Market calendar (simplified - in production use a proper market calendar)
        self.market_holidays = [
            "2025-01-01",  # New Year's Day
            "2025-07-04",  # Independence Day
            "2025-12-25",  # Christmas
            # Add more holidays as needed
        ]
        
        self.schedule_active = False
    
    def start_daily_schedule(self):
        """Start daily EOD schedule"""
        if self.schedule_active:
            return
        
        # Schedule EOD monitoring to start each weekday
        schedule.every().monday.at("15:45").do(self._start_eod_if_market_day)
        schedule.every().tuesday.at("15:45").do(self._start_eod_if_market_day)
        schedule.every().wednesday.at("15:45").do(self._start_eod_if_market_day)
        schedule.every().thursday.at("15:45").do(self._start_eod_if_market_day)
        schedule.every().friday.at("15:45").do(self._start_eod_if_market_day)
        
        self.schedule_active = True
        self.logger.info("EOD daily schedule activated")
        
        # Start schedule runner thread
        schedule_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        schedule_thread.start()
    
    def _run_scheduler(self):
        """Run the schedule checker"""
        while self.schedule_active:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _start_eod_if_market_day(self):
        """Start EOD monitoring if it's a market day"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in self.market_holidays:
            self.logger.info(f"Market holiday today ({today}), skipping EOD")
            return
        
        # Check if market is actually open today
        # (In production, use a proper market calendar API)
        
        self.logger.info(f"Starting EOD monitoring for {today}")
        self.eod_manager.start_monitor()
    
    def stop_schedule(self):
        """Stop the daily schedule"""
        schedule.clear()
        self.schedule_active = False
        self.logger.info("EOD schedule stopped")

# Example usage and integration
def create_eod_system(broker_manager: BrokerExecutionManager) -> EODPositionManager:
    """Create complete EOD system with monitoring and scheduling"""
    
    # Create EOD manager
    eod_config = EODConfig(
        warning_time="15:50",
        flatten_time="15:55", 
        hard_cutoff="15:59:30"
    )
    
    eod_manager = EODPositionManager(broker_manager, eod_config)
    
    # Set up callbacks
    def warning_notification(message):
        logging.getLogger("EOD").warning(f"📢 {message}")
        # Could send Slack/email notifications here
    
    def flatten_notification(results):
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        logging.getLogger("EOD").info(f"📊 Flattened {successful}/{total} positions")
    
    def emergency_notification(data):
        logging.getLogger("EOD").critical(f"🚨 Emergency flatten: {data}")
        # Could trigger alerts/escalations here
    
    eod_manager.set_callbacks(warning_notification, flatten_notification, emergency_notification)
    
    # Create scheduler
    scheduler = EODScheduler(eod_manager)
    scheduler.start_daily_schedule()
    
    return eod_manager