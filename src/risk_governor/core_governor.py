"""
Production Risk Governor - Core Implementation
Phase 1: PositionSize + DrawDown governors for MSFT single-symbol trading

Key Requirements:
- ATR-scaled position increments (not fixed $100)
- Cumulative turnover tracking
- <5ms latency per decision
- Absolute hard stops that cannot be breached
- 100% unit test coverage
"""

import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class TradingAction(Enum):
    """5-action discrete trading actions from Stairways V4"""
    SELL_AGGRESSIVE = 0
    SELL_CONSERVATIVE = 1  
    HOLD = 2
    BUY_CONSERVATIVE = 3
    BUY_AGGRESSIVE = 4

@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations"""
    symbol: str
    current_position: float  # Current position size in $
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    current_price: float
    daily_turnover: float  # Cumulative turnover today
    max_daily_drawdown: float  # Peak-to-trough today

@dataclass
class RiskLimits:
    """Absolute risk limits that governors cannot override"""
    max_intraday_loss: float = 100.0  # $100 absolute stop
    max_position_notional: float = 500.0  # $500 max position
    max_daily_turnover: float = 2000.0  # $2000 max daily turnover
    max_single_trade: float = 50.0  # $50 max single trade
    
class PositionSizeGovernor:
    """
    ATR-scaled position size governor with turnover tracking
    Prevents single-bar kills and excessive churning
    """
    
    def __init__(self, symbol: str = "MSFT"):
        self.symbol = symbol
        self.atr_lookback = 20  # 20-period ATR
        self.atr_multiplier = 0.5  # Max position = 0.5 * ATR
        self.max_base_increment = 50.0  # Base max increment
        self.recent_prices = []
        
        # Intraday ATR for gap handling
        self.intraday_atr_window = 390  # Full trading day in minutes
        self.intraday_prices = []
        self.use_intraday_atr = False  # Enable during market open/close
        
        self.logger = logging.getLogger(f"PositionGovernor_{symbol}")
        
    def update_atr(self, high: float, low: float, close: float, prev_close: float, 
                   timestamp: float = None):
        """Update ATR calculation with new bar data"""
        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        # Regular ATR (daily/weekly timeframe)
        if len(self.recent_prices) >= self.atr_lookback:
            self.recent_prices.pop(0)
        self.recent_prices.append(true_range)
        
        # Intraday ATR (390-bar rolling window for gap handling)
        current_time = timestamp or time.time()
        self.intraday_prices.append({
            'true_range': true_range,
            'timestamp': current_time
        })
        
        # Maintain intraday window size
        if len(self.intraday_prices) > self.intraday_atr_window:
            self.intraday_prices.pop(0)
        
        # Enable intraday ATR during market open/close periods
        self._update_atr_mode(current_time)
        
    def get_current_atr(self) -> float:
        """Get current ATR value"""
        # Use intraday ATR during market open/close for gap protection
        if self.use_intraday_atr and len(self.intraday_prices) >= 10:
            return np.mean([p['true_range'] for p in self.intraday_prices])
        
        # Regular ATR for normal trading hours
        if len(self.recent_prices) < 5:
            return 2.0  # Conservative fallback for MSFT
        return np.mean(self.recent_prices)
    
    def _update_atr_mode(self, current_time: float):
        """Update ATR mode based on time of day"""
        from datetime import datetime
        
        current_hour = datetime.fromtimestamp(current_time).strftime("%H:%M")
        
        # Market open period (9:30-10:00) and close period (15:30-16:00)
        open_period = "09:30" <= current_hour <= "10:00"
        close_period = "15:30" <= current_hour <= "16:00"
        
        previous_mode = self.use_intraday_atr
        self.use_intraday_atr = open_period or close_period
        
        if self.use_intraday_atr != previous_mode:
            mode_name = "intraday" if self.use_intraday_atr else "regular"
            self.logger.info(f"Switched to {mode_name} ATR mode at {current_hour}")
    
    def get_atr_status(self) -> Dict:
        """Get current ATR calculation status"""
        return {
            "regular_atr": np.mean(self.recent_prices) if len(self.recent_prices) >= 5 else 2.0,
            "intraday_atr": np.mean([p['true_range'] for p in self.intraday_prices]) if len(self.intraday_prices) >= 10 else 2.0,
            "using_intraday": self.use_intraday_atr,
            "regular_samples": len(self.recent_prices),
            "intraday_samples": len(self.intraday_prices),
            "current_atr": self.get_current_atr()
        }
    
    def calculate_max_increment(self, current_price: float) -> float:
        """Calculate ATR-scaled maximum position increment"""
        atr = self.get_current_atr()
        atr_scaled_max = self.atr_multiplier * atr * current_price / 100  # Scale to dollars
        return min(atr_scaled_max, self.max_base_increment)
    
    def filter_action(self, 
                     raw_action: TradingAction, 
                     portfolio: PortfolioState,
                     risk_limits: RiskLimits) -> Tuple[float, str]:
        """
        Convert discrete action to safe position increment
        Returns: (position_increment, reason)
        """
        start_time = time.perf_counter()
        
        try:
            # Calculate ATR-scaled max increment
            max_increment = self.calculate_max_increment(portfolio.current_price)
            
            # Map discrete actions to position increments
            if raw_action == TradingAction.BUY_AGGRESSIVE:
                desired_increment = max_increment
            elif raw_action == TradingAction.BUY_CONSERVATIVE:
                desired_increment = max_increment * 0.5
            elif raw_action == TradingAction.SELL_AGGRESSIVE:
                desired_increment = -max_increment
            elif raw_action == TradingAction.SELL_CONSERVATIVE:
                desired_increment = -max_increment * 0.5
            else:  # HOLD
                desired_increment = 0.0
            
            # Apply risk limit checks
            safe_increment = self._apply_risk_limits(
                desired_increment, portfolio, risk_limits
            )
            
            # Log decision
            latency = (time.perf_counter() - start_time) * 1000  # ms
            reason = f"ATR_scaled_max={max_increment:.2f}, latency={latency:.2f}ms"
            
            if latency > 5.0:
                self.logger.warning(f"Governor latency {latency:.2f}ms exceeds 5ms budget")
            
            return safe_increment, reason
            
        except Exception as e:
            self.logger.error(f"Error in position filter: {e}")
            return 0.0, f"ERROR: {str(e)}"
    
    def _apply_risk_limits(self, 
                          desired_increment: float,
                          portfolio: PortfolioState, 
                          risk_limits: RiskLimits) -> float:
        """Apply absolute risk limits to desired increment"""
        
        # 1. Single trade size limit
        if abs(desired_increment) > risk_limits.max_single_trade:
            desired_increment = np.sign(desired_increment) * risk_limits.max_single_trade
        
        # 2. Position notional limit
        new_position = portfolio.current_position + desired_increment
        if abs(new_position) > risk_limits.max_position_notional:
            # Reduce increment to stay within limit
            if new_position > 0:
                desired_increment = risk_limits.max_position_notional - portfolio.current_position
            else:
                desired_increment = -risk_limits.max_position_notional - portfolio.current_position
            desired_increment = max(0, desired_increment)  # No negative positions for now
        
        # 3. Daily turnover limit
        new_turnover = portfolio.daily_turnover + abs(desired_increment)
        if new_turnover > risk_limits.max_daily_turnover:
            # Reduce increment to stay within turnover limit
            remaining_turnover = risk_limits.max_daily_turnover - portfolio.daily_turnover
            if remaining_turnover <= 0:
                desired_increment = 0.0
            else:
                desired_increment = np.sign(desired_increment) * min(
                    abs(desired_increment), remaining_turnover
                )
        
        return desired_increment

class DrawdownGovernor:
    """
    Multi-zone drawdown governor with real-time price feeds
    Prevents cliff scenarios through graduated response
    """
    
    def __init__(self, symbol: str = "MSFT"):
        self.symbol = symbol
        self.yellow_zone = 0.05   # 5% DD triggers caution
        self.red_zone = 0.08      # 8% DD triggers position reduction  
        self.hard_stop = 0.10     # 10% DD forces flat (absolute limit)
        self.logger = logging.getLogger(f"DrawdownGovernor_{symbol}")
        
    def filter_action(self,
                     position_increment: float,
                     portfolio: PortfolioState,
                     risk_limits: RiskLimits,
                     top_of_book_mid: Optional[float] = None) -> Tuple[float, str]:
        """
        Apply drawdown-based position filtering
        Uses top-of-book mid price if available to avoid lag
        """
        start_time = time.perf_counter()
        
        try:
            # Use top-of-book price if available, else portfolio price
            current_price = top_of_book_mid or portfolio.current_price
            
            # Calculate current drawdown using real-time price
            if portfolio.current_position != 0:
                # Calculate P&L from position change
                price_change = current_price - portfolio.current_price
                position_pnl = price_change * abs(portfolio.current_position) / 100.0  # Scale to dollars
                if portfolio.current_position < 0:  # Short position
                    position_pnl = -position_pnl
            else:
                position_pnl = 0
                
            total_pnl = portfolio.realized_pnl + portfolio.unrealized_pnl + position_pnl
            current_loss = abs(min(0, total_pnl))
            
            # Calculate DD as percentage of assumed starting capital (1000)
            starting_capital = 1000.0
            current_dd = current_loss / starting_capital
            
            # Apply graduated response based on DD zone
            if current_dd >= self.hard_stop:
                # HARD STOP: Force flat immediately
                safe_increment = -portfolio.current_position  # Close all positions
                reason = f"HARD_STOP: DD={current_dd:.3f} >= {self.hard_stop:.3f}"
                
            elif current_dd >= self.red_zone:
                # RED ZONE: Only allow position reductions
                if position_increment > 0:
                    safe_increment = 0.0  # Block position increases
                    reason = f"RED_ZONE: DD={current_dd:.3f}, blocking increases"
                else:
                    safe_increment = position_increment  # Allow reductions
                    reason = f"RED_ZONE: DD={current_dd:.3f}, allowing reduction"
                    
            elif current_dd >= self.yellow_zone:
                # YELLOW ZONE: Half normal position size
                safe_increment = position_increment * 0.5
                reason = f"YELLOW_ZONE: DD={current_dd:.3f}, 50% sizing"
                
            else:
                # NORMAL: No restrictions
                safe_increment = position_increment
                reason = f"NORMAL: DD={current_dd:.3f}"
            
            # Final check against absolute loss limit
            projected_loss = abs(min(0, total_pnl + safe_increment))
            if projected_loss > risk_limits.max_intraday_loss:
                safe_increment = 0.0
                reason += f" | LOSS_LIMIT: projected={projected_loss:.2f}"
            
            # Log latency
            latency = (time.perf_counter() - start_time) * 1000
            if latency > 5.0:
                self.logger.warning(f"DD Governor latency {latency:.2f}ms exceeds budget")
            
            return safe_increment, reason
            
        except Exception as e:
            self.logger.error(f"Error in drawdown filter: {e}")
            return 0.0, f"ERROR: {str(e)}"

class ProductionRiskGovernor:
    """
    Master governor coordinating all risk layers
    Single entry point for production trading system
    """
    
    def __init__(self, symbol: str = "MSFT"):
        self.symbol = symbol
        self.position_governor = PositionSizeGovernor(symbol)
        self.drawdown_governor = DrawdownGovernor(symbol)
        self.risk_limits = RiskLimits()
        self.logger = logging.getLogger(f"RiskGovernor_{symbol}")
        
    def filter_trading_action(self,
                            raw_action: TradingAction,
                            portfolio: PortfolioState,
                            market_data: Dict,
                            top_of_book_mid: Optional[float] = None) -> Dict:
        """
        Master filter: Raw action -> Safe executable increment
        Returns complete decision audit trail
        """
        start_time = time.perf_counter()
        
        try:
            # Update ATR with latest market data
            if 'high' in market_data and 'low' in market_data:
                self.position_governor.update_atr(
                    market_data['high'],
                    market_data['low'], 
                    market_data['close'],
                    market_data.get('prev_close', market_data['close'])
                )
            
            # Layer 1: Position Size Governor
            position_increment, pos_reason = self.position_governor.filter_action(
                raw_action, portfolio, self.risk_limits
            )
            
            # Layer 2: Drawdown Governor  
            safe_increment, dd_reason = self.drawdown_governor.filter_action(
                position_increment, portfolio, self.risk_limits, top_of_book_mid
            )
            
            total_latency = (time.perf_counter() - start_time) * 1000
            
            # Return complete audit trail
            return {
                'raw_action': raw_action.name,
                'safe_increment': safe_increment,
                'position_reason': pos_reason,
                'drawdown_reason': dd_reason,
                'total_latency_ms': total_latency,
                'risk_limits_applied': safe_increment != position_increment,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Master filter error: {e}")
            return {
                'raw_action': raw_action.name if raw_action else 'UNKNOWN',
                'safe_increment': 0.0,
                'position_reason': f"ERROR: {str(e)}",
                'drawdown_reason': "ERROR_FALLBACK",
                'total_latency_ms': 999.0,
                'risk_limits_applied': True,
                'timestamp': time.time()
            }