#!/usr/bin/env python3
"""
📊 IB EXECUTOR SERVICE  
Executes approved trading orders through Interactive Brokers paper trading
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import redis
from dataclasses import dataclass
from enum import Enum

# IB Gateway integration
from src.brokers.ib_gateway import IBGatewayClient as IBGateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradingOrder:
    timestamp: float
    action: int
    action_name: str
    confidence: float
    risk_approved: bool
    risk_reason: str
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    error_message: Optional[str] = None

class IBExecutorService:
    """Service to execute trading orders via IB Gateway"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.ib_gateway = None
        self.is_connected = False
        
        # Order tracking
        self.pending_orders: Dict[str, TradingOrder] = {}
        self.completed_orders: List[TradingOrder] = []
        
        # Execution statistics
        self.execution_stats = {
            "total_orders": 0,
            "successful_fills": 0,
            "failed_orders": 0,
            "cancelled_orders": 0,
            "connection_errors": 0,
            "last_execution": None,
            "avg_fill_time": 0.0
        }
        
        # Current positions (tracked locally)
        self.current_positions = {
            "NVDA": 0.0,
            "MSFT": 0.0,
            "cash": 100000.0
        }
        
    async def initialize_ib_connection(self):
        """Initialize connection to IB Gateway"""
        
        logger.info("📊 Initializing IB Gateway connection")
        
        try:
            # Initialize IB Gateway (paper trading)
            self.ib_gateway = IBGateway(
                host="127.0.0.1",
                port=7497,  # Paper trading port
                client_id=1
            )
            
            # Connect with retry logic
            for attempt in range(3):
                try:
                    success = await asyncio.to_thread(self.ib_gateway.connect)
                    if success:
                        self.is_connected = True
                        logger.info("✅ IB Gateway connected successfully")
                        
                        # Test connection with account info
                        await self.verify_connection()
                        return True
                    else:
                        logger.warning(f"🔄 Connection attempt {attempt + 1} failed")
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ Connection attempt {attempt + 1} error: {e}")
                    await asyncio.sleep(5)
            
            logger.error("❌ Failed to connect to IB Gateway after 3 attempts")
            return False
            
        except Exception as e:
            logger.error(f"❌ IB Gateway initialization error: {e}")
            self.execution_stats["connection_errors"] += 1
            return False
    
    async def verify_connection(self):
        """Verify IB Gateway connection and get account info"""
        
        try:
            if not self.ib_gateway:
                return False
                
            # Get account summary (paper trading account)
            account_info = await asyncio.to_thread(self.ib_gateway.get_account_summary)
            
            if account_info:
                logger.info("📊 IB Gateway account verified")
                
                # Update cash position if available
                if 'TotalCashValue' in account_info:
                    self.current_positions['cash'] = float(account_info['TotalCashValue'])
                    logger.info(f"   Cash: ${self.current_positions['cash']:,.2f}")
                
                return True
            else:
                logger.warning("⚠️ Could not verify IB account")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection verification error: {e}")
            return False
    
    def action_to_orders(self, action: int, confidence: float) -> List[Dict[str, Any]]:
        """Convert model action to IB orders"""
        
        # Action mapping to position changes
        action_map = {
            0: {"NVDA": 0, "MSFT": 0},      # HOLD_BOTH
            1: {"NVDA": 1, "MSFT": 0},      # BUY_NVDA_HOLD_MSFT  
            2: {"NVDA": -1, "MSFT": 0},     # SELL_NVDA_HOLD_MSFT
            3: {"NVDA": 0, "MSFT": 1},      # HOLD_NVDA_BUY_MSFT
            4: {"NVDA": 1, "MSFT": 1},      # BUY_BOTH
            5: {"NVDA": -1, "MSFT": 1},     # SELL_NVDA_BUY_MSFT
            6: {"NVDA": 0, "MSFT": -1},     # HOLD_NVDA_SELL_MSFT
            7: {"NVDA": 1, "MSFT": -1},     # BUY_NVDA_SELL_MSFT
            8: {"NVDA": -1, "MSFT": -1}     # SELL_BOTH
        }
        
        position_changes = action_map.get(action, {"NVDA": 0, "MSFT": 0})
        
        orders = []
        
        # Create orders for each symbol with position change
        for symbol, change in position_changes.items():
            if change != 0:
                # Determine order size based on confidence and risk
                base_size = 10  # Base position size (shares)
                confidence_multiplier = max(0.5, confidence)  # Min 50% of base size
                order_size = int(base_size * confidence_multiplier)
                
                order = {
                    "symbol": symbol,
                    "action": "BUY" if change > 0 else "SELL",
                    "order_type": "MKT",  # Market order
                    "quantity": order_size,
                    "time_in_force": "DAY"
                }
                orders.append(order)
        
        return orders
    
    async def execute_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single order via IB Gateway"""
        
        try:
            if not self.is_connected or not self.ib_gateway:
                raise Exception("IB Gateway not connected")
            
            logger.info(f"📊 Executing order: {order_data['action']} {order_data['quantity']} {order_data['symbol']}")
            
            # Place order via IB Gateway
            order_result = await asyncio.to_thread(
                self.ib_gateway.place_order,
                symbol=order_data["symbol"],
                action=order_data["action"],
                order_type=order_data["order_type"],
                quantity=order_data["quantity"],
                time_in_force=order_data.get("time_in_force", "DAY")
            )
            
            if order_result and order_result.get("success"):
                logger.info(f"✅ Order submitted: {order_result.get('order_id')}")
                return {
                    "success": True,
                    "order_id": order_result.get("order_id"),
                    "status": "submitted",
                    "message": "Order submitted successfully"
                }
            else:
                error_msg = order_result.get("error", "Unknown error") if order_result else "No response from IB"
                logger.error(f"❌ Order failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "status": "rejected"
                }
                
        except Exception as e:
            logger.error(f"❌ Order execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "rejected"
            }
    
    async def process_trading_order(self, order: TradingOrder):
        """Process a complete trading order (multiple IB orders if needed)"""
        
        logger.info(f"📊 Processing trading order: {order.action_name}")
        
        try:
            # Convert action to IB orders
            ib_orders = self.action_to_orders(order.action, order.confidence)
            
            if not ib_orders:
                logger.info("ℹ️ No position changes required (HOLD action)")
                order.status = OrderStatus.FILLED
                order.fill_time = datetime.now()
                return
            
            # Execute each IB order
            execution_results = []
            
            for ib_order in ib_orders:
                result = await self.execute_order(ib_order)
                execution_results.append(result)
                
                if result["success"]:
                    # Update local position tracking
                    symbol = ib_order["symbol"]
                    quantity = ib_order["quantity"] if ib_order["action"] == "BUY" else -ib_order["quantity"]
                    self.current_positions[symbol] += quantity
                    
                    logger.info(f"📈 Position updated: {symbol} = {self.current_positions[symbol]}")
                
                # Small delay between orders
                await asyncio.sleep(0.5)
            
            # Determine overall order status
            successful_orders = [r for r in execution_results if r["success"]]
            
            if len(successful_orders) == len(ib_orders):
                order.status = OrderStatus.FILLED
                order.fill_time = datetime.now()
                logger.info("✅ All orders executed successfully")
                self.execution_stats["successful_fills"] += 1
                
            elif successful_orders:
                order.status = OrderStatus.FILLED  # Partial fill still counts as filled
                order.fill_time = datetime.now()
                logger.warning(f"⚠️ Partial execution: {len(successful_orders)}/{len(ib_orders)} orders filled")
                self.execution_stats["successful_fills"] += 1
                
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = "All orders failed"
                logger.error("❌ All orders failed")
                self.execution_stats["failed_orders"] += 1
            
            self.execution_stats["last_execution"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"❌ Error processing trading order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.execution_stats["failed_orders"] += 1
    
    async def process_orders(self):
        """Main loop to process trading orders from Redis"""
        
        logger.info("📊 IB Executor service started")
        
        # Initialize IB connection
        connection_success = await self.initialize_ib_connection()
        if not connection_success:
            logger.error("❌ Cannot start executor - IB Gateway connection failed")
            return
        
        logger.info("✅ IB Executor ready for orders")
        
        while True:
            try:
                # Read orders from Redis stream
                messages = self.redis_client.xread({'trading:orders': '$'}, count=1, block=1000)
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self.process_single_order(fields)
                        
            except Exception as e:
                logger.error(f"❌ Error in order processing loop: {e}")
                
                # Try to reconnect if connection lost
                if not self.is_connected:
                    logger.info("🔄 Attempting to reconnect to IB Gateway")
                    await self.initialize_ib_connection()
                
                await asyncio.sleep(5)
    
    async def process_single_order(self, order_data: Dict[str, str]):
        """Process a single order from Redis stream"""
        
        try:
            # Parse order data
            trading_order = TradingOrder(
                timestamp=float(order_data.get('timestamp', time.time())),
                action=int(order_data.get('action', 0)),
                action_name=order_data.get('action_name', 'UNKNOWN'),
                confidence=float(order_data.get('confidence', 0.0)),
                risk_approved=order_data.get('risk_approved', 'false').lower() == 'true',
                risk_reason=order_data.get('risk_reason', '')
            )
            
            # Generate order ID
            trading_order.order_id = f"ord_{int(time.time())}_{trading_order.action}"
            
            # Track order
            self.pending_orders[trading_order.order_id] = trading_order
            self.execution_stats["total_orders"] += 1
            
            # Process the order
            await self.process_trading_order(trading_order)
            
            # Move to completed orders
            self.completed_orders.append(trading_order)
            if trading_order.order_id in self.pending_orders:
                del self.pending_orders[trading_order.order_id]
            
            # Publish execution result
            await self.publish_execution_result(trading_order)
            
            # Push metrics
            self.push_execution_metrics()
            
        except Exception as e:
            logger.error(f"❌ Error processing single order: {e}")
    
    async def publish_execution_result(self, order: TradingOrder):
        """Publish order execution result"""
        
        try:
            result_data = {
                "order_id": order.order_id,
                "timestamp": order.timestamp,
                "action": order.action,
                "action_name": order.action_name,
                "status": order.status.value,
                "fill_time": order.fill_time.isoformat() if order.fill_time else None,
                "error_message": order.error_message,
                "positions": json.dumps(self.current_positions)
            }
            
            # Publish to execution results stream
            self.redis_client.xadd("trading:executions", result_data)
            
            logger.debug(f"📊 Execution result published: {order.status.value}")
            
        except Exception as e:
            logger.error(f"❌ Error publishing execution result: {e}")
    
    def push_execution_metrics(self):
        """Push execution metrics to monitoring"""
        
        try:
            metrics_data = {
                "ib_executor_orders_total": self.execution_stats["total_orders"],
                "ib_executor_fills_total": self.execution_stats["successful_fills"],
                "ib_executor_failures_total": self.execution_stats["failed_orders"],
                "ib_executor_connection_errors": self.execution_stats["connection_errors"],
                "ib_positions_nvda": self.current_positions["NVDA"],
                "ib_positions_msft": self.current_positions["MSFT"],
                "ib_cash_balance": self.current_positions["cash"],
                "ib_connected": 1 if self.is_connected else 0
            }
            
            # Store in Redis
            self.redis_client.hset("execution_metrics", mapping=metrics_data)
            
        except Exception as e:
            logger.error(f"❌ Error pushing execution metrics: {e}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        
        return {
            "connected": self.is_connected,
            "pending_orders": len(self.pending_orders),
            "completed_orders": len(self.completed_orders),
            "current_positions": self.current_positions,
            "statistics": self.execution_stats,
            "last_updated": datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        
        logger.info("📊 Shutting down IB Executor service")
        
        try:
            if self.ib_gateway and self.is_connected:
                await asyncio.to_thread(self.ib_gateway.disconnect)
                logger.info("✅ IB Gateway disconnected")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main function to run IB executor service"""
    
    executor = IBExecutorService()
    
    try:
        await executor.process_orders()
    except KeyboardInterrupt:
        logger.info("📊 IB Executor service stopped by user")
    except Exception as e:
        logger.error(f"❌ IB Executor service error: {e}")
    finally:
        await executor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())