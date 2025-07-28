#!/usr/bin/env python3
"""
Order Management System (OMS) Models
Enhanced order and position tracking for dual-ticker trading
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by the OMS"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"          # Order created, not yet submitted
    SUBMITTED = "submitted"      # Order sent to broker
    ACCEPTED = "accepted"        # Order accepted by broker
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"           # Order completely executed
    CANCELLED = "cancelled"     # Order cancelled
    REJECTED = "rejected"       # Order rejected by broker
    EXPIRED = "expired"         # Order expired

class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class OrderFill:
    """Represents a fill (execution) of an order"""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    source: str = "unknown"  # broker source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'source': self.source
        }

@dataclass
class Order:
    """Enhanced order model for dual-ticker trading"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Execution tracking
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    total_commission: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)
    
    # Metadata
    strategy_id: str = "dual_ticker_rl"
    source: str = "rl_agent"
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields"""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled or cancelled)"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                              OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_completed(self) -> bool:
        """Check if order is completed (filled, cancelled, rejected, expired)"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage (0.0 to 1.0)"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    def add_fill(self, fill: OrderFill) -> bool:
        """Add a fill to this order"""
        try:
            # Validate fill
            if fill.symbol != self.symbol:
                logger.error(f"Fill symbol {fill.symbol} doesn't match order symbol {self.symbol}")
                return False
            
            if fill.side != self.side:
                logger.error(f"Fill side {fill.side} doesn't match order side {self.side}")
                return False
            
            if fill.quantity <= 0:
                logger.error(f"Invalid fill quantity: {fill.quantity}")
                return False
            
            # Check if fill would overfill the order
            if self.filled_quantity + fill.quantity > self.quantity:
                logger.error(f"Fill quantity {fill.quantity} would overfill order "
                           f"({self.filled_quantity + fill.quantity} > {self.quantity})")
                return False
            
            # Add fill
            fill.order_id = self.order_id
            self.fills.append(fill)
            
            # Update order state
            old_filled_quantity = self.filled_quantity
            self.filled_quantity += fill.quantity
            self.remaining_quantity = self.quantity - self.filled_quantity
            self.total_commission += fill.commission
            
            # Update average fill price
            if self.filled_quantity > 0:
                total_value = (old_filled_quantity * self.average_fill_price + 
                             fill.quantity * fill.price)
                self.average_fill_price = total_value / self.filled_quantity
            
            # Update status
            if self.remaining_quantity == 0:
                self.status = OrderStatus.FILLED
                self.filled_at = fill.timestamp
            elif self.filled_quantity > 0:
                self.status = OrderStatus.PARTIALLY_FILLED
            
            logger.info(f"Fill added to order {self.order_id}: {fill.quantity} @ {fill.price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add fill to order {self.order_id}: {e}")
            return False
    
    def cancel(self, reason: str = "user_requested") -> bool:
        """Cancel the order"""
        if not self.is_active:
            logger.warning(f"Cannot cancel order {self.order_id} in status {self.status}")
            return False
        
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.now()
        self.error_message = reason
        
        logger.info(f"Order {self.order_id} cancelled: {reason}")
        return True
    
    def reject(self, reason: str) -> bool:
        """Reject the order"""
        if self.status != OrderStatus.SUBMITTED:
            logger.warning(f"Cannot reject order {self.order_id} in status {self.status}")
            return False
        
        self.status = OrderStatus.REJECTED
        self.error_message = reason
        
        logger.error(f"Order {self.order_id} rejected: {reason}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'total_commission': self.total_commission,
            'fills': [fill.to_dict() for fill in self.fills],
            'strategy_id': self.strategy_id,
            'source': self.source,
            'broker_order_id': self.broker_order_id,
            'error_message': self.error_message,
            'fill_percentage': self.fill_percentage,
            'is_active': self.is_active,
            'is_completed': self.is_completed
        }

@dataclass
class Position:
    """Enhanced position model for dual-ticker portfolio tracking"""
    symbol: str = ""
    quantity: int = 0  # Positive = long, negative = short, 0 = flat
    average_price: float = 0.0
    market_price: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Cost basis tracking
    cost_basis: float = 0.0
    total_commission: float = 0.0
    
    # Position metadata
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    trades_count: int = 0
    
    @property
    def side(self) -> PositionSide:
        """Get position side"""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def market_value(self) -> float:
        """Get current market value of position"""
        return self.quantity * self.market_price
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no shares)"""
        return self.quantity == 0
    
    def update_market_price(self, new_price: float) -> bool:
        """Update market price and recalculate unrealized P&L"""
        try:
            if new_price <= 0:
                logger.error(f"Invalid market price for {self.symbol}: {new_price}")
                return False
            
            self.market_price = new_price
            self.last_updated = datetime.now()
            
            # Recalculate unrealized P&L
            if not self.is_flat:
                self.unrealized_pnl = (self.market_price - self.average_price) * self.quantity
                self.total_pnl = self.realized_pnl + self.unrealized_pnl
            else:
                self.unrealized_pnl = 0.0
                self.total_pnl = self.realized_pnl
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update market price for {self.symbol}: {e}")
            return False
    
    def apply_fill(self, fill: OrderFill) -> bool:
        """Apply an order fill to update position"""
        try:
            if fill.symbol != self.symbol:
                logger.error(f"Fill symbol {fill.symbol} doesn't match position symbol {self.symbol}")
                return False
            
            old_quantity = self.quantity
            fill_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            new_quantity = old_quantity + fill_quantity
            
            # Calculate realized P&L for position reductions
            if old_quantity != 0 and ((old_quantity > 0 and fill_quantity < 0) or 
                                     (old_quantity < 0 and fill_quantity > 0)):
                # Position reduction - calculate realized P&L
                reduction_quantity = min(abs(old_quantity), abs(fill_quantity))
                if old_quantity > 0:  # Reducing long position
                    realized_pnl_per_share = fill.price - self.average_price
                else:  # Reducing short position
                    realized_pnl_per_share = self.average_price - fill.price
                
                self.realized_pnl += realized_pnl_per_share * reduction_quantity
            
            # Update position
            if new_quantity == 0:
                # Position closed
                self.quantity = 0
                self.average_price = 0.0
                self.cost_basis = 0.0
                self.unrealized_pnl = 0.0
            else:
                # Position opened or increased
                if old_quantity == 0 or (old_quantity > 0) == (fill_quantity > 0):
                    # New position or same-side addition
                    total_cost = abs(old_quantity) * self.average_price + abs(fill_quantity) * fill.price
                    self.quantity = new_quantity
                    self.average_price = total_cost / abs(new_quantity)
                    self.cost_basis = abs(new_quantity) * self.average_price
                    
                    if self.opened_at is None:
                        self.opened_at = fill.timestamp
                else:
                    # Opposite-side addition (remaining after partial close)
                    self.quantity = new_quantity
                    self.average_price = fill.price
                    self.cost_basis = abs(new_quantity) * self.average_price
                    self.opened_at = fill.timestamp
            
            # Update commission and trade count
            self.total_commission += fill.commission
            self.trades_count += 1
            self.last_updated = fill.timestamp
            
            # Recalculate total P&L
            self.total_pnl = self.realized_pnl + self.unrealized_pnl
            
            logger.info(f"Position {self.symbol} updated: {self.quantity} @ {self.average_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply fill to position {self.symbol}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'side': self.side.value,
            'average_price': self.average_price,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'total_commission': self.total_commission,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'last_updated': self.last_updated.isoformat(),
            'trades_count': self.trades_count,
            'is_flat': self.is_flat
        }

class OMSTracker:
    """Order Management System tracker for dual-ticker trading"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.supported_symbols = {'NVDA', 'MSFT'}
        
        # Initialize positions for supported symbols
        for symbol in self.supported_symbols:
            self.positions[symbol] = Position(symbol=symbol)
    
    def create_order(self, symbol: str, side: OrderSide, quantity: int, 
                    order_type: OrderType = OrderType.MARKET, 
                    price: Optional[float] = None) -> Optional[Order]:
        """Create a new order"""
        try:
            if symbol not in self.supported_symbols:
                logger.error(f"Unsupported symbol: {symbol}")
                return None
            
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return None
            
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
                logger.error(f"Price required for {order_type} orders")
                return None
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price
            )
            
            self.orders[order.order_id] = order
            logger.info(f"Order created: {order.order_id} ({symbol} {side.value} {quantity})")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None
    
    def submit_order(self, order_id: str) -> bool:
        """Submit order to broker"""
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} not in pending status: {order.status}")
            return False
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        logger.info(f"Order submitted: {order_id}")
        return True
    
    def add_fill(self, order_id: str, quantity: int, price: float, 
                commission: float = 0.0, source: str = "broker") -> bool:
        """Add a fill to an order and update position"""
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        # Create fill
        fill = OrderFill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            source=source
        )
        
        # Add fill to order
        if not order.add_fill(fill):
            return False
        
        # Update position
        if order.symbol in self.positions:
            if not self.positions[order.symbol].apply_fill(fill):
                logger.error(f"Failed to update position for {order.symbol}")
                return False
        
        logger.info(f"Fill processed: {order_id} {quantity} @ {price:.2f}")
        return True
    
    def cancel_order(self, order_id: str, reason: str = "user_requested") -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        return self.orders[order_id].cancel(reason)
    
    def update_market_prices(self, prices: Dict[str, float]) -> bool:
        """Update market prices for all positions"""
        try:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_market_price(price)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update market prices: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        total_value = 0.0
        total_pnl = 0.0
        total_commission = 0.0
        active_positions = 0
        
        position_summaries = {}
        
        for symbol, position in self.positions.items():
            position_summaries[symbol] = position.to_dict()
            
            if not position.is_flat:
                active_positions += 1
                total_value += abs(position.market_value)
            
            total_pnl += position.total_pnl
            total_commission += position.total_commission
        
        # Order statistics
        active_orders = sum(1 for order in self.orders.values() if order.is_active)
        completed_orders = sum(1 for order in self.orders.values() if order.is_completed)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_commission': total_commission,
                'active_positions': active_positions,
                'net_pnl': total_pnl - total_commission
            },
            'positions': position_summaries,
            'orders': {
                'total_orders': len(self.orders),
                'active_orders': active_orders,
                'completed_orders': completed_orders
            }
        }
    
    def get_trading_summary(self, start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get trading activity summary"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        
        # Filter orders by time
        recent_orders = [
            order for order in self.orders.values()
            if order.created_at >= start_time
        ]
        
        # Calculate statistics
        total_orders = len(recent_orders)
        filled_orders = len([o for o in recent_orders if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in recent_orders if o.status == OrderStatus.CANCELLED])
        
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
        
        # Volume statistics
        total_volume = sum(order.filled_quantity for order in recent_orders)
        total_notional = sum(
            order.filled_quantity * order.average_fill_price 
            for order in recent_orders if order.filled_quantity > 0
        )
        
        return {
            'period_start': start_time.isoformat(),
            'period_end': datetime.now().isoformat(),
            'orders': {
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'cancelled_orders': cancelled_orders,
                'fill_rate': fill_rate
            },
            'volume': {
                'total_shares': total_volume,
                'total_notional': total_notional
            }
        }