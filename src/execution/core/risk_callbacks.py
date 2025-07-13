"""
Risk Callbacks Core Module

Contains risk management callbacks and functions for the execution system.
This module handles:
- Pre-trade risk checks
- Position size throttling
- Risk event handlers
- Real-time risk monitoring

This is an internal module - use src.execution.OrchestratorAgent for public API.

RISK CONFIGURATION DEFAULTS:
- max_position_size: 1000 shares
- max_order_size: 100 shares  
- max_spread_bps: 50 basis points (0.5%)
- max_volatility: 0.02 (2%)
- max_volume_participation: 0.1 (10% of ADV)
- min_order_size: 1 share
- max_daily_loss: 1000 (currency units)
- max_position_concentration: 0.1 (10% of portfolio)
"""

import logging
from typing import Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import IntEnum
import pandas as pd
import asyncio

# Risk event types and reason codes
class KillSwitchReason(IntEnum):
    """Enumeration of kill switch reason codes for high-performance audit."""
    RISK_BREACH = 1
    DAILY_LOSS_LIMIT = 2
    POSITION_LIMIT = 3
    CONCENTRATION_LIMIT = 4
    MARKET_VOLATILITY = 5
    SYSTEM_ERROR = 6
    MANUAL_STOP = 7
    CONNECTIVITY_LOSS = 8


def pre_trade_check(
    event: Dict[str, Any],
    risk_config: Dict[str, Any],
    current_positions: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Perform comprehensive pre-trade risk checks before executing a trade.
    
    Args:
        event: Trading event details
        risk_config: Risk configuration parameters
        current_positions: Current portfolio positions
        logger: Optional logger instance
        
    Returns:
        Tuple of (is_allowed, reason_if_blocked)
    """
    logger = logger or logging.getLogger(__name__)
    
    symbol = event.get('symbol')
    action = event.get('action')
    shares = event.get('shares', 0)
    
    # 1. Basic parameter validation
    if not symbol or not action or shares <= 0:
        return False, "Invalid trade parameters"
    
    if action not in ['BUY', 'SELL']:
        return False, f"Unknown action: {action}"
    
    # 2. Position size limits
    max_position_size = risk_config.get('max_position_size', 1000)
    current_position = current_positions.get(symbol, {}).get('shares', 0)
    
    if action == 'BUY':
        new_position = current_position + shares
    else:  # SELL
        new_position = current_position - shares
        
    if abs(new_position) > max_position_size:
        return False, f"Position size would exceed limit: {abs(new_position)} > {max_position_size}"
    
    # 3. Order size limits
    max_order_size = risk_config.get('max_order_size', 500)
    if shares > max_order_size:
        return False, f"Order size exceeds limit: {shares} > {max_order_size}"
    
    # 4. Daily loss limit check (if current P&L available)
    current_pnl = current_positions.get('_daily_pnl', 0)
    max_daily_loss = risk_config.get('max_daily_loss', 1000)
    if current_pnl < -max_daily_loss:
        return False, f"Daily loss limit exceeded: {current_pnl} < -{max_daily_loss}"
    
    # 5. Position concentration check (if portfolio value available)
    portfolio_value = current_positions.get('_total_value', 0)
    if portfolio_value > 0:
        # Estimate position value - price should come from market data or risk_agent
        # IMPORTANT: Default price of 100 is a fallback assumption
        # In production, inject last trade price from market data feed
        estimated_price = event.get('price', 100)  # TODO: Use real market price
        if estimated_price == 100:
            logger.warning(f"Using default price fallback (100) for concentration check on {symbol}")
        
        position_value = abs(new_position) * estimated_price
        max_concentration = risk_config.get('max_position_concentration', 0.1)
        concentration = position_value / portfolio_value
        
        if concentration > max_concentration:
            return False, f"Position concentration too high: {concentration:.2%} > {max_concentration:.2%}"
    
    # 6. Market hours check
    if risk_config.get('check_market_hours', True):
        # Simplified market hours check (should be more sophisticated)
        current_hour = datetime.now().hour
        market_open = risk_config.get('market_open_hour', 9)
        market_close = risk_config.get('market_close_hour', 16)
        
        if not (market_open <= current_hour < market_close):
            return False, f"Trading outside market hours: {current_hour}:00"
    
    return True, None


def throttle_size(
    order: Dict[str, Any],
    risk_config: Dict[str, Any],
    market_conditions: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Throttle order size based on risk parameters and market conditions.
    
    Args:
        order: Original order details
        risk_config: Risk configuration parameters
        market_conditions: Current market conditions (spread, volatility, volume)
        logger: Optional logger instance
        
    Returns:
        Modified order with potentially adjusted size
    """
    logger = logger or logging.getLogger(__name__)
    
    original_shares = order.get('shares', 0)
    throttled_shares = original_shares
    
    # 1. Apply market condition-based throttling first (most restrictive)
    if market_conditions:
        # Limit based on available volume (most restrictive)
        avg_volume = market_conditions.get('avg_volume', float('inf'))
        max_volume_pct = risk_config.get('max_volume_participation', 0.1)  # 10%
        max_shares_by_volume = int(avg_volume * max_volume_pct)
        if throttled_shares > max_shares_by_volume:
            throttled_shares = max_shares_by_volume
            logger.info(f"Order size limited by volume participation: {max_volume_pct:.1%}")
        
        # Reduce size based on spread
        spread = market_conditions.get('spread', 0)
        max_spread = risk_config.get('max_spread_bps', 50)  # 50 basis points
        if spread > max_spread:
            spread_factor = max_spread / spread
            throttled_shares = int(throttled_shares * spread_factor)
            logger.info(f"Order size reduced due to wide spread: {spread} bps")
        
        # Reduce size based on volatility
        volatility = market_conditions.get('volatility', 0)
        max_volatility = risk_config.get('max_volatility', 0.02)  # 2%
        if volatility > max_volatility:
            vol_factor = max_volatility / volatility
            throttled_shares = int(throttled_shares * vol_factor)
            logger.info(f"Order size reduced due to high volatility: {volatility:.2%}")
    
    # 2. Apply absolute size limits (secondary constraint)
    max_order_size = risk_config.get('max_order_size', 100)
    throttled_shares = min(throttled_shares, max_order_size)
    
    # Ensure minimum viable size
    min_order_size = risk_config.get('min_order_size', 1)
    throttled_shares = max(throttled_shares, min_order_size)
    
    # Log throttling if occurred
    if throttled_shares != original_shares:
        logger.info(f"Order size throttled from {original_shares} to {throttled_shares}")
        
    # Return modified order
    modified_order = order.copy()
    modified_order['shares'] = throttled_shares
    modified_order['original_shares'] = original_shares
    modified_order['throttled'] = throttled_shares != original_shares
    modified_order['throttle_reason'] = _get_throttle_reason(original_shares, throttled_shares, market_conditions, risk_config)
    
    # 🔍 DIAGNOSTIC: Log post-throttle shares
    logger.debug(f"🔍 Post-throttle shares: {throttled_shares} (original: {original_shares})")
    
    return modified_order


def _get_throttle_reason(original: int, throttled: int, market_conditions: Dict, risk_config: Dict) -> str:
    """Get human-readable reason for throttling."""
    if original == throttled:
        return "No throttling applied"
    
    reasons = []
    
    # Check volume participation limit
    if market_conditions:
        avg_volume = market_conditions.get('avg_volume', float('inf'))
        max_volume_pct = risk_config.get('max_volume_participation', 0.1)
        max_shares_by_volume = int(avg_volume * max_volume_pct)
        if original > max_shares_by_volume:
            reasons.append("volume participation")
    
    # Check other limits
    if throttled <= risk_config.get('max_order_size', 100):
        reasons.append("size limit")
    if market_conditions.get('spread', 0) > risk_config.get('max_spread_bps', 50):
        reasons.append("wide spread")
    if market_conditions.get('volatility', 0) > risk_config.get('max_volatility', 0.02):
        reasons.append("high volatility")
    
    return ", ".join(reasons) if reasons else "risk management"


def check_daily_loss_limit(
    current_pnl: float,
    risk_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if daily loss limit has been exceeded.
    
    TODO: This function exists but is not called from execution loop yet.
    Consider integrating into pre_trade_check or calling separately.
    
    Args:
        current_pnl: Current daily P&L
        risk_config: Risk configuration parameters
        logger: Optional logger instance
        
    Returns:
        Tuple of (trading_allowed, reason_if_blocked)
    """
    logger = logger or logging.getLogger(__name__)
    
    max_daily_loss = risk_config.get('max_daily_loss', 1000)
    
    if current_pnl < -max_daily_loss:
        reason = f"Daily loss limit exceeded: {current_pnl} < -{max_daily_loss}"
        logger.warning(reason)
        return False, reason
        
    return True, None


def check_position_concentration(
    symbol: str,
    new_position_value: float,
    total_portfolio_value: float,
    risk_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if position concentration limits are respected.
    
    TODO: This function exists but is not called from execution loop yet.
    Consider integrating into pre_trade_check or calling separately.
    
    Args:
        symbol: Trading symbol
        new_position_value: Value of the new position
        total_portfolio_value: Total portfolio value
        risk_config: Risk configuration parameters
        logger: Optional logger instance
        
    Returns:
        Tuple of (is_allowed, reason_if_blocked)
    """
    logger = logger or logging.getLogger(__name__)
    
    if total_portfolio_value <= 0:
        return True, None  # Can't calculate concentration
        
    max_concentration = risk_config.get('max_position_concentration', 0.1)  # 10%
    concentration = abs(new_position_value) / total_portfolio_value
    
    if concentration > max_concentration:
        reason = f"Position concentration too high for {symbol}: {concentration:.2%} > {max_concentration:.2%}"
        logger.warning(reason)
        return False, reason
        
    return True, None


class RiskEventHandler:
    """
    Thread-safe handler for risk-related events during execution.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the risk event handler.
        
        Args:
            config: Risk configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.event_callbacks: Dict[str, Callable] = {}
        self._callback_lock = asyncio.Lock()  # Thread safety for callbacks
        
    async def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for a specific risk event type."""
        async with self._callback_lock:
            self.event_callbacks[event_type] = callback
        
    async def handle_risk_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a risk event by calling the appropriate callback."""
        async with self._callback_lock:
            callback = self.event_callbacks.get(event_type)
        
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                self.logger.error(f"Risk event callback failed for {event_type}: {e}")
        else:
            self.logger.warning(f"No callback registered for risk event: {event_type}")
            
    def emergency_stop(self, reason: str, reason_code: int = 0, 
                      symbol_id: int = 0, position_size: int = 0, pnl_cents: int = 0) -> None:
        """
        Trigger an emergency stop of trading with ultra-low latency audit.
        
        Args:
            reason: Human-readable reason (for logging)
            reason_code: Numeric reason code for high-perf audit
            symbol_id: Symbol identifier
            position_size: Current position size
            pnl_cents: Current P&L in cents
        """
        # CRITICAL PATH: Log to high-performance audit system first
        try:
            # Try to import high-performance audit module
            try:
                from .high_perf_audit import audit_kill_switch
            except ImportError:
                # Fallback to standard logging if high-perf audit not available
                self.logger.warning("High-performance audit module not available, using standard logging")
                audit_kill_switch = None
            
            # Map reason to code if not provided
            if reason_code == 0:
                reason_lower = reason.lower()
                if 'loss' in reason_lower:
                    reason_code = KillSwitchReason.DAILY_LOSS_LIMIT
                elif 'position' in reason_lower:
                    reason_code = KillSwitchReason.POSITION_LIMIT
                elif 'concentration' in reason_lower:
                    reason_code = KillSwitchReason.CONCENTRATION_LIMIT
                elif 'volatility' in reason_lower:
                    reason_code = KillSwitchReason.MARKET_VOLATILITY
                elif 'error' in reason_lower:
                    reason_code = KillSwitchReason.SYSTEM_ERROR
                elif 'manual' in reason_lower:
                    reason_code = KillSwitchReason.MANUAL_STOP
                elif 'connectivity' in reason_lower:
                    reason_code = KillSwitchReason.CONNECTIVITY_LOSS
                else:
                    reason_code = KillSwitchReason.RISK_BREACH
            
            # Ultra-fast audit logging (sub-microsecond) if available
            if audit_kill_switch:
                audit_kill_switch(reason_code, symbol_id, position_size, pnl_cents)
            
        except Exception as e:
            # Don't let audit failure block emergency stop, but log the issue
            self.logger.warning(f"Audit logging failed during emergency stop: {e}")
        
        # Standard logging (can be slower)
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Handle risk event (can be slower) - create task to avoid blocking
        try:
            try:
                loop = asyncio.get_running_loop()  # Python 3.7+ preferred method
            except RuntimeError:
                # No event loop running - skip async handling
                self.logger.warning("Emergency stop event handling skipped - no event loop")
                return
            
            loop.create_task(self.handle_risk_event("emergency_stop", {
                "reason": reason, 
                "reason_code": reason_code,
                "symbol_id": symbol_id,
                "position_size": position_size,
                "pnl_cents": pnl_cents,
                "timestamp": datetime.now()
            }))
        except Exception as e:
            # Don't let event handling failure block emergency stop
            self.logger.warning(f"Emergency stop event handling failed: {e}")