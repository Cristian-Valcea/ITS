"""
Production Broker Execution Adapter with Chaos Testing
Handles IBKR paper API with retry logic and graceful failure modes
"""

import time
import logging
import random
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timezone
import threading
import queue

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"

@dataclass
class Order:
    order_id: str
    symbol: str
    quantity: float  # Positive = buy, negative = sell
    order_type: str  # "MKT", "LMT", "MOC"
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    timestamp: float = 0.0
    reject_reason: Optional[str] = None

@dataclass
class ExecutionReport:
    order_id: str
    executed_size: float
    execution_price: float  
    commission: float
    status: OrderStatus
    timestamp: float
    error_message: Optional[str] = None

class IBKRPaperAdapter:
    """
    IBKR Paper Trading API adapter with chaos testing and retry logic
    Simulates real broker behavior including random rejections
    """
    
    def __init__(self, chaos_mode: bool = False, rejection_rate: float = 0.01):
        self.logger = logging.getLogger("IBKRPaperAdapter")
        self.chaos_mode = chaos_mode
        self.rejection_rate = rejection_rate
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.execution_queue = queue.Queue()
        
        # Rate limiting (IBKR has limits)
        self.last_order_time = 0.0
        self.min_order_interval = 0.1  # 100ms between orders
        self.orders_per_second_limit = 5
        self.recent_orders = []
        
        # Connection simulation
        self.connected = True
        self.connection_failures = 0
        
        # Market hours
        self.market_open_time = "09:30"
        self.market_close_time = "16:00"
        
        self.logger.info(f"IBKR Paper Adapter initialized (chaos_mode={chaos_mode})")
    
    def place_order(self, 
                   symbol: str, 
                   quantity: float, 
                   order_type: str = "MKT",
                   limit_price: Optional[float] = None,
                   timeout: float = 5.0) -> Tuple[bool, ExecutionReport]:
        """
        Place order with retry logic and chaos testing
        
        Returns:
            (success, execution_report)
        """
        start_time = time.time()
        
        try:
            # Generate order ID
            self.order_counter += 1
            order_id = f"ORD_{int(time.time())}_{self.order_counter}"
            
            # Rate limiting check
            if not self._check_rate_limits():
                return False, ExecutionReport(
                    order_id=order_id,
                    executed_size=0.0,
                    execution_price=0.0,
                    commission=0.0,
                    status=OrderStatus.REJECTED,
                    timestamp=time.time(),
                    error_message="Rate limit exceeded"
                )
            
            # Market hours check
            if not self._is_market_open():
                return False, ExecutionReport(
                    order_id=order_id,
                    executed_size=0.0,
                    execution_price=0.0,
                    commission=0.0,
                    status=OrderStatus.REJECTED,
                    timestamp=time.time(),
                    error_message="Market closed"
                )
            
            # Connection check
            if not self._check_connection():
                return False, ExecutionReport(
                    order_id=order_id,
                    executed_size=0.0,
                    execution_price=0.0,
                    commission=0.0,
                    status=OrderStatus.REJECTED,
                    timestamp=time.time(),
                    error_message="Connection lost"
                )
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                timestamp=time.time()
            )
            
            # Chaos testing - random rejections
            if self.chaos_mode and random.random() < self.rejection_rate:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "CHAOS_TEST_REJECTION"
                
                self.orders[order_id] = order
                
                return False, ExecutionReport(
                    order_id=order_id,
                    executed_size=0.0,
                    execution_price=0.0,
                    commission=0.0,
                    status=OrderStatus.REJECTED,
                    timestamp=time.time(),
                    error_message="Chaos test rejection"
                )
            
            # Simulate order processing delay
            processing_delay = random.uniform(0.1, 0.5)  # 100-500ms
            time.sleep(processing_delay)
            
            # Execute order (paper trading simulation)
            execution_price = self._simulate_execution_price(symbol, order_type, limit_price)
            commission = self._calculate_commission(abs(quantity), execution_price)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            
            self.orders[order_id] = order
            
            # Create execution report
            execution_report = ExecutionReport(
                order_id=order_id,
                executed_size=quantity,
                execution_price=execution_price,
                commission=commission,
                status=OrderStatus.FILLED,
                timestamp=time.time()
            )
            
            total_time = time.time() - start_time
            self.logger.info(f"Order {order_id} executed: {quantity} {symbol} @ ${execution_price:.2f} (latency: {total_time*1000:.1f}ms)")
            
            return True, execution_report
            
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return False, ExecutionReport(
                order_id="ERROR",
                executed_size=0.0,
                execution_price=0.0,
                commission=0.0,
                status=OrderStatus.REJECTED,
                timestamp=time.time(),
                error_message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.logger.info(f"Order {order_id} cancelled")
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [order for order in self.orders.values() 
                if order.status == OrderStatus.PENDING]
    
    def flatten_all_positions(self) -> List[ExecutionReport]:
        """Emergency flatten all positions (MOC orders)"""
        # In real implementation, this would query current positions
        # For now, simulate closing any remaining orders
        reports = []
        
        for order in self.get_open_orders():
            success, report = self.place_order(
                symbol=order.symbol,
                quantity=-order.quantity,  # Opposite direction
                order_type="MOC"
            )
            if success:
                reports.append(report)
        
        return reports
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Minimum interval between orders
        if current_time - self.last_order_time < self.min_order_interval:
            return False
        
        # Orders per second limit
        self.recent_orders = [t for t in self.recent_orders if current_time - t < 1.0]
        if len(self.recent_orders) >= self.orders_per_second_limit:
            return False
        
        # Update tracking
        self.last_order_time = current_time
        self.recent_orders.append(current_time)
        
        return True
    
    def _is_market_open(self) -> bool:
        """Check if market is open (simplified)"""
        # In real implementation, use market calendar
        current_time = datetime.now().strftime("%H:%M")
        return self.market_open_time <= current_time <= self.market_close_time
    
    def _check_connection(self) -> bool:
        """Simulate connection health checks"""
        if self.chaos_mode and random.random() < 0.001:  # 0.1% chance of connection loss
            self.connected = False
            self.connection_failures += 1
            self.logger.warning("Simulated connection loss")
            
        # Simulate reconnection after 1 second
        if not self.connected:
            time.sleep(1.0)
            self.connected = True
            self.logger.info("Connection restored")
        
        return self.connected
    
    def _simulate_execution_price(self, 
                                 symbol: str, 
                                 order_type: str, 
                                 limit_price: Optional[float]) -> float:
        """Simulate realistic execution prices"""
        
        # Base prices for common symbols
        base_prices = {
            "MSFT": 420.0,
            "AAPL": 195.0,
            "NVDA": 450.0,
            "TSLA": 250.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        if order_type == "LMT" and limit_price:
            # Limit orders execute at limit price (simplified)
            return limit_price
        else:
            # Market orders with realistic slippage
            slippage = random.uniform(-0.002, 0.002)  # ±0.2% slippage
            return base_price * (1 + slippage)
    
    def _calculate_commission(self, shares: float, price: float) -> float:
        """Calculate IBKR-style commissions with real-time estimation"""
        # IBKR tiered pricing: $0.0035 per share, $0.35 minimum
        per_share_cost = 0.0035
        min_commission = 0.35
        
        commission = max(abs(shares) * per_share_cost, min_commission)
        return commission
    
    def estimate_trade_cost(self, position_increment: float, current_price: float) -> Dict:
        """Real-time trade cost estimation for risk budget gating"""
        if abs(position_increment) < 0.01:
            return {
                "shares": 0,
                "commission": 0.0,
                "total_cost": 0.0,
                "cost_per_dollar": 0.0
            }
        
        # Convert position increment to shares
        shares = abs(position_increment) / current_price
        shares = round(shares, 0)  # Whole shares for stocks
        
        # Calculate commission
        commission = self._calculate_commission(shares, current_price)
        
        # Total trade cost includes commission
        total_cost = commission
        
        # Cost per dollar of position (for turnover gating)
        cost_per_dollar = total_cost / abs(position_increment) if position_increment != 0 else 0
        
        return {
            "shares": shares,
            "commission": commission,
            "total_cost": total_cost,
            "cost_per_dollar": cost_per_dollar,
            "position_value": abs(position_increment)
        }

class BrokerExecutionManager:
    """
    High-level broker execution manager with retry logic and safety features
    Integrates with risk governor for safe order execution
    """
    
    def __init__(self, chaos_mode: bool = False):
        self.logger = logging.getLogger("BrokerExecutionManager")
        
        # Initialize broker adapter
        self.broker = IBKRPaperAdapter(chaos_mode=chaos_mode)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Start with 1 second
        self.backoff_multiplier = 2.0
        
        # Safety features
        self.daily_order_count = 0
        self.daily_commission_cost = 0.0
        self.daily_effective_cost = 0.0  # Track real trading costs
        self.max_daily_orders = 500
        self.max_daily_commission = 100.0
        self.max_daily_effective_cost = 50.0  # Limit on real trading costs
        
        # EOD management
        self.eod_flatten_time = "15:55"
        self.hard_cutoff_time = "15:59:30"
        self.eod_thread = None
        self.shutdown_flag = threading.Event()
        
        self.logger.info("Broker Execution Manager initialized")
    
    def execute_safe_order(self, 
                          symbol: str, 
                          position_increment: float,
                          current_price: float,
                          max_latency_ms: float = 1000.0) -> Dict:
        """
        Execute order with safety checks and retry logic
        Integrates with risk governor's position increment
        """
        start_time = time.time()
        
        try:
            # Get real-time cost estimation
            cost_estimate = self.broker.estimate_trade_cost(position_increment, current_price)
            
            if cost_estimate["shares"] < 1:  # Less than 1 share
                return {
                    "success": True,
                    "executed_size": 0.0,
                    "execution_price": current_price,
                    "commission": 0.0,
                    "effective_cost": 0.0,
                    "message": "Position increment too small to execute",
                    "latency_ms": 0.0
                }
            
            # Safety checks including cost limits
            if not self._pre_execution_checks_with_costs(cost_estimate, current_price):
                return {
                    "success": False,
                    "executed_size": 0.0,
                    "execution_price": 0.0,
                    "commission": 0.0,
                    "effective_cost": 0.0,
                    "message": "Pre-execution safety check failed (cost limits)",
                    "latency_ms": 0.0
                }
            
            shares = cost_estimate["shares"]
            
            # Execute with retries
            for attempt in range(self.max_retries):
                success, execution_report = self.broker.place_order(
                    symbol=symbol,
                    quantity=shares,
                    order_type="MKT"
                )
                
                if success:
                    # Update daily tracking
                    self.daily_order_count += 1
                    self.daily_commission_cost += execution_report.commission
                    self.daily_effective_cost += cost_estimate["total_cost"]
                    
                    latency = (time.time() - start_time) * 1000
                    
                    return {
                        "success": True,
                        "executed_size": execution_report.executed_size,
                        "execution_price": execution_report.execution_price,
                        "commission": execution_report.commission,
                        "effective_cost": cost_estimate["total_cost"],
                        "cost_per_dollar": cost_estimate["cost_per_dollar"],
                        "order_id": execution_report.order_id,
                        "message": "Order executed successfully",
                        "latency_ms": latency
                    }
                
                # Retry with exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.backoff_multiplier ** attempt)
                    self.logger.warning(f"Order attempt {attempt + 1} failed, retrying in {delay}s")
                    time.sleep(delay)
            
            # All retries failed
            return {
                "success": False,
                "executed_size": 0.0,
                "execution_price": 0.0,
                "commission": 0.0,
                "message": f"Order failed after {self.max_retries} attempts",
                "latency_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Execute safe order error: {e}")
            return {
                "success": False,
                "executed_size": 0.0,
                "execution_price": 0.0,
                "commission": 0.0,
                "message": f"Execution error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    def _position_increment_to_shares(self, increment: float, price: float) -> float:
        """Convert dollar position increment to shares"""
        if abs(increment) < 0.01:
            return 0.0
        
        shares = increment / price
        
        # Round to avoid fractional shares for stocks
        return round(shares, 0)
    
    def _pre_execution_checks_with_costs(self, cost_estimate: Dict, price: float) -> bool:
        """Enhanced pre-execution safety checks with real-time cost estimation"""
        
        # Daily order limit
        if self.daily_order_count >= self.max_daily_orders:
            self.logger.error(f"Daily order limit exceeded: {self.daily_order_count}")
            return False
        
        # Daily commission limit
        if self.daily_commission_cost + cost_estimate["commission"] > self.max_daily_commission:
            self.logger.error(f"Daily commission limit would be exceeded: {self.daily_commission_cost + cost_estimate['commission']:.2f}")
            return False
        
        # Daily effective cost limit (new)
        if self.daily_effective_cost + cost_estimate["total_cost"] > self.max_daily_effective_cost:
            self.logger.error(f"Daily effective cost limit would be exceeded: {self.daily_effective_cost + cost_estimate['total_cost']:.2f}")
            return False
        
        # Cost efficiency check - prevent excessive cost ratios
        if cost_estimate["cost_per_dollar"] > 0.05:  # 5% max cost ratio
            self.logger.error(f"Trade cost ratio too high: {cost_estimate['cost_per_dollar']:.1%}")
            return False
        
        # Market hours check
        if not self.broker._is_market_open():
            self.logger.error("Market is closed")
            return False
        
        return True
    
    def _pre_execution_checks(self, shares: float, price: float) -> bool:
        """Legacy pre-execution checks (for backward compatibility)"""
        cost_estimate = self.broker.estimate_trade_cost(shares * price, price)
        return self._pre_execution_checks_with_costs(cost_estimate, price)
    
    def start_eod_monitor(self):
        """Start end-of-day monitoring thread"""
        if self.eod_thread is None or not self.eod_thread.is_alive():
            self.eod_thread = threading.Thread(target=self._eod_monitor_loop)
            self.eod_thread.daemon = True
            self.eod_thread.start()
            self.logger.info("EOD monitor thread started")
    
    def _eod_monitor_loop(self):
        """Monitor for end-of-day and auto-flatten positions"""
        while not self.shutdown_flag.is_set():
            current_time = datetime.now().strftime("%H:%M")
            
            # Auto-flatten at 15:55
            if current_time >= self.eod_flatten_time:
                self.logger.info("EOD auto-flatten triggered")
                reports = self.broker.flatten_all_positions()
                self.logger.info(f"Flattened {len(reports)} positions")
                break
            
            # Hard cutoff at 15:59:30  
            if current_time >= self.hard_cutoff_time:
                self.logger.warning("Hard cutoff triggered - cancelling all orders")
                for order in self.broker.get_open_orders():
                    self.broker.cancel_order(order.order_id)
                break
            
            time.sleep(30)  # Check every 30 seconds
    
    def shutdown(self):
        """Shutdown broker manager"""
        self.shutdown_flag.set()
        if self.eod_thread:
            self.eod_thread.join(timeout=5.0)
        self.logger.info("Broker Execution Manager shutdown")
    
    def get_daily_stats(self) -> Dict:
        """Get daily execution statistics with enhanced cost tracking"""
        return {
            "daily_order_count": self.daily_order_count,
            "daily_commission_cost": self.daily_commission_cost,
            "daily_effective_cost": self.daily_effective_cost,
            "max_daily_orders": self.max_daily_orders,
            "max_daily_commission": self.max_daily_commission,
            "max_daily_effective_cost": self.max_daily_effective_cost,
            "orders_remaining": self.max_daily_orders - self.daily_order_count,
            "commission_budget_remaining": self.max_daily_commission - self.daily_commission_cost,
            "effective_cost_budget_remaining": self.max_daily_effective_cost - self.daily_effective_cost,
            "average_cost_per_order": self.daily_effective_cost / max(1, self.daily_order_count)
        }