"""
PnL Tracker Core Module

Contains portfolio and P&L tracking logic extracted from OrchestratorAgent.
This module handles:
- Portfolio state synchronization
- P&L calculations and tracking
- Position management
- Net liquidation value updates

This is an internal module - use src.execution.OrchestratorAgent for public API.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd

# Fee engine imports
try:
    from ...shared.fee_schedule import get_cme_fee_schedule, FeeSchedule
    FEE_ENGINE_AVAILABLE = True
except ImportError:
    FEE_ENGINE_AVAILABLE = False

# Prometheus metrics for fee tracking
try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
    
    FEES_TOTAL = Counter(
        'trading_fees_total_usd',
        'Total trading fees paid in USD',
        ['symbol', 'venue']
    )
    
    FEE_PER_CONTRACT = Histogram(
        'trading_fee_per_contract_usd',
        'Fee per contract in USD',
        ['symbol', 'venue'],
        buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    )
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    FEES_TOTAL = MockMetric()
    FEE_PER_CONTRACT = MockMetric()


class PnLTracker:
    """
    Core P&L and portfolio tracking system.
    
    Handles portfolio state management, P&L calculations,
    and position tracking for the trading system.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the P&L tracker.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_state: Dict[str, Any] = {}
        self.pnl_history: List[Dict[str, Any]] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Fee tracking
        self.fee_accumulator: float = 0.0
        self.fees_by_symbol: Dict[str, float] = {}
        self.volume_ytd: Dict[str, int] = {}  # Year-to-date volume for tiered fees
        
        # Initialize fee schedule
        self.fee_schedule: Optional[FeeSchedule] = None
        if FEE_ENGINE_AVAILABLE:
            try:
                self.fee_schedule = get_cme_fee_schedule()
                self.logger.info(f"Fee engine initialized for venue: {self.fee_schedule.venue}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize fee engine: {e}")
        else:
            self.logger.warning("Fee engine not available - fees will not be calculated")
        
        # Check simulation mode
        ibkr_config = self.config.get('ibkr_conn', {}) or self.config.get('ibkr_connection', {})
        self.simulation_mode = ibkr_config.get('simulation_mode', False)
    
    def initialize_portfolio_state(self, initial_capital: float) -> bool:
        """
        Initialize the portfolio state.
        
        Args:
            initial_capital: Starting capital amount
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.portfolio_state = {
                'net_liquidation': initial_capital,
                'total_cash_value': initial_capital,
                'positions': {},
                'open_orders': {},
                'last_update': datetime.now(),
                'initial_capital': initial_capital,
                'total_pnl': 0.0,
                'daily_pnl': 0.0
            }
            
            self.logger.info(f"Portfolio state initialized with capital: ${initial_capital:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing portfolio state: {e}")
            return False
    
    def _apply_fees(self, trade: Dict[str, Any]) -> float:
        """
        Apply trading fees to a trade and update cash position.
        
        Args:
            trade: Trade dictionary with keys: symbol, qty, side, price
                  - symbol: Trading symbol (e.g., 'MES', 'MNQ')
                  - qty: Number of contracts (positive for buy, negative for sell)
                  - side: +1 for buy, -1 for sell
                  - price: Execution price
        
        Returns:
            Total fee applied in USD
        """
        if not self.fee_schedule:
            return 0.0
        
        try:
            symbol = trade['symbol']
            quantity = abs(trade['qty'])  # Always positive for fee calculation
            
            # Get current YTD volume for tiered pricing
            current_volume = self.volume_ytd.get(symbol, 0)
            
            # Calculate fee
            total_fee = self.fee_schedule.calculate_total_fee(
                symbol=symbol,
                quantity=quantity,
                volume_ytd=current_volume
            )
            
            # Update cash position (reduce by fee amount)
            if 'total_cash_value' in self.portfolio_state:
                self.portfolio_state['total_cash_value'] -= total_fee
            
            # Update fee tracking
            self.fee_accumulator += total_fee
            self.fees_by_symbol[symbol] = self.fees_by_symbol.get(symbol, 0) + total_fee
            self.volume_ytd[symbol] = current_volume + quantity
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                FEES_TOTAL.labels(symbol=symbol, venue=self.fee_schedule.venue).inc(total_fee)
                fee_per_contract = total_fee / quantity if quantity > 0 else 0
                FEE_PER_CONTRACT.labels(symbol=symbol, venue=self.fee_schedule.venue).observe(fee_per_contract)
            
            # Log fee application
            fee_per_contract = total_fee / quantity if quantity > 0 else 0
            self.logger.debug(
                f"Applied fees: {symbol} {quantity} contracts × ${fee_per_contract:.2f} = ${total_fee:.2f}"
            )
            
            # Record fee event in P&L history
            self.record_pnl_event(
                symbol=symbol,
                event_type='fee',
                amount=-total_fee,  # Negative because it reduces P&L
                details={
                    'quantity': quantity,
                    'fee_per_contract': fee_per_contract,
                    'volume_ytd': self.volume_ytd[symbol],
                    'venue': self.fee_schedule.venue
                }
            )
            
            return total_fee
            
        except Exception as e:
            self.logger.error(f"Error applying fees for trade {trade}: {e}")
            return 0.0
    
    def on_fill(self, trade: Dict[str, Any]) -> None:
        """
        Process a trade fill - apply fees and update positions.
        
        Args:
            trade: Trade dictionary with execution details
        """
        try:
            # Apply fees first (reduces cash)
            fee_applied = self._apply_fees(trade)
            
            # Update position
            self._update_position(trade)
            
            # Update P&L
            self._update_pnl(trade)
            
            # Log the fill
            symbol = trade.get('symbol', 'UNKNOWN')
            qty = trade.get('qty', 0)
            price = trade.get('price', 0)
            
            self.logger.info(
                f"Fill processed: {symbol} {qty} @ ${price:.2f} "
                f"(Fee: ${fee_applied:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing fill: {e}")
    
    def _update_position(self, trade: Dict[str, Any]) -> None:
        """Update position based on trade execution."""
        try:
            symbol = trade['symbol']
            qty = trade['qty']
            price = trade['price']
            
            # Update portfolio state positions
            if 'positions' not in self.portfolio_state:
                self.portfolio_state['positions'] = {}
            
            current_position = self.portfolio_state['positions'].get(symbol, 0)
            new_position = current_position + qty
            self.portfolio_state['positions'][symbol] = new_position
            
            # Update detailed position tracking
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'shares': 0,
                    'avg_cost': 0,
                    'market_price': price,
                    'market_value': 0,
                    'unrealized_pnl': 0,
                    'updated_at': datetime.now()
                }
            
            pos = self.positions[symbol]
            old_shares = pos['shares']
            old_avg_cost = pos['avg_cost']
            
            # Calculate new average cost
            if new_position == 0:
                # Position closed
                pos['avg_cost'] = 0
            elif old_shares == 0:
                # New position
                pos['avg_cost'] = price
            elif (old_shares > 0 and qty > 0) or (old_shares < 0 and qty < 0):
                # Adding to position
                total_cost = (old_shares * old_avg_cost) + (qty * price)
                pos['avg_cost'] = total_cost / new_position
            # For reducing position, keep old average cost
            
            pos['shares'] = new_position
            pos['market_price'] = price
            pos['market_value'] = new_position * price
            pos['unrealized_pnl'] = new_position * (price - pos['avg_cost'])
            pos['updated_at'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def _update_pnl(self, trade: Dict[str, Any]) -> None:
        """Update P&L calculations based on trade."""
        try:
            # This would typically calculate realized P&L for position changes
            # For now, we'll update the portfolio state timestamp
            self.portfolio_state['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating P&L: {e}")
    
    def get_fee_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive fee summary.
        
        Returns:
            Dictionary with fee statistics and breakdown
        """
        return {
            'total_fees': self.fee_accumulator,
            'fees_by_symbol': self.fees_by_symbol.copy(),
            'volume_ytd': self.volume_ytd.copy(),
            'fee_engine_available': self.fee_schedule is not None,
            'venue': self.fee_schedule.venue if self.fee_schedule else None,
            'average_fee_per_contract': (
                self.fee_accumulator / sum(self.volume_ytd.values())
                if sum(self.volume_ytd.values()) > 0 else 0
            )
        }
    
    def get_fee_impact_analysis(self) -> Dict[str, Any]:
        """
        Analyze the impact of fees on performance.
        
        Returns:
            Dictionary with fee impact metrics
        """
        try:
            total_pnl = self.calculate_pnl().get('total_pnl', 0)
            gross_pnl = total_pnl + self.fee_accumulator  # P&L before fees
            
            fee_impact = {
                'gross_pnl': gross_pnl,
                'net_pnl': total_pnl,
                'total_fees': self.fee_accumulator,
                'fee_impact_pct': (
                    (self.fee_accumulator / abs(gross_pnl)) * 100
                    if abs(gross_pnl) > 0 else 0
                ),
                'fee_drag_bps': (
                    (self.fee_accumulator / self.portfolio_state.get('initial_capital', 100000)) * 10000
                ),
                'total_volume': sum(self.volume_ytd.values()),
                'symbols_traded': len(self.volume_ytd)
            }
            
            return fee_impact
            
        except Exception as e:
            self.logger.error(f"Error calculating fee impact: {e}")
            return {}
        
    def synchronize_portfolio_state_with_broker(
        self, 
        symbol_traded: Optional[str] = None,
        data_agent=None
    ) -> None:
        """
        Synchronize portfolio state with broker.
        
        Args:
            symbol_traded: Optional symbol that was just traded
            data_agent: Optional data agent for broker communication
        """
        try:
            if self.simulation_mode:
                # In simulation mode, just update timestamp
                self.portfolio_state['last_update'] = datetime.now()
                self.logger.debug("Portfolio state synchronized (simulation mode)")
                return
                
            # Real mode - would query IBKR for actual positions and cash
            self.logger.debug("Synchronizing portfolio state with broker")
            
            # TODO: Implement actual broker synchronization
            # This would involve:
            # 1. Query broker for current positions
            # 2. Query broker for current cash balance
            # 3. Update portfolio_state with actual values
            
        except Exception as e:
            self.logger.error(f"Error synchronizing portfolio state: {e}")
        
    def update_net_liquidation_and_risk_agent(self, risk_agent, current_prices: Optional[Dict[str, float]] = None) -> None:
        """
        Update net liquidation value and notify risk agent.
        
        Args:
            risk_agent: Risk management agent
            current_prices: Optional dictionary of current prices
        """
        try:
            # Calculate net liquidation value
            total_cash = self.portfolio_state.get('total_cash_value', 0)
            positions_value = 0
            
            for symbol, shares in self.portfolio_state.get('positions', {}).items():
                try:
                    # Get current price for position valuation
                    if current_prices and symbol in current_prices:
                        current_price = current_prices[symbol]
                    else:
                        current_price = 100.0  # Placeholder - in real implementation would query market data
                    positions_value += shares * current_price
                except Exception as e:
                    self.logger.warning(f"Error valuing position {symbol}: {e}")
            
            net_liquidation = total_cash + positions_value
            self.portfolio_state['net_liquidation'] = net_liquidation
            
            # Update risk agent
            try:
                risk_agent.update_portfolio_state(self.portfolio_state)
            except Exception as e:
                self.logger.warning(f"Failed to update risk agent: {e}")
                
            self.logger.debug(f"Updated net liquidation: ${net_liquidation:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating net liquidation: {e}")
        
    def update_position(
        self,
        symbol: str,
        shares: float,
        avg_cost: float,
        market_price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update position information for a symbol."""
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.positions:
            self.positions[symbol] = {}
            
        self.positions[symbol].update({
            'shares': shares,
            'avg_cost': avg_cost,
            'market_price': market_price,
            'market_value': shares * market_price,
            'unrealized_pnl': shares * (market_price - avg_cost),
            'updated_at': timestamp
        })
        
    def calculate_realized_pnl(
        self,
        symbol: str,
        shares_traded: float,
        execution_price: float,
        avg_cost: float
    ) -> float:
        """Calculate realized P&L for a trade."""
        return shares_traded * (execution_price - avg_cost)
        
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio."""
        total_market_value = sum(
            pos.get('market_value', 0) for pos in self.positions.values()
        )
        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) for pos in self.positions.values()
        )
        
        return {
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions': self.positions.copy(),
            'updated_at': datetime.now()
        }
        
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a specific symbol."""
        return self.positions.get(symbol)
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return self.portfolio_state.copy()
    
    def calculate_pnl(self, symbol: Optional[str] = None, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate current P&L including fee impact.
        
        Args:
            symbol: Optional specific symbol to calculate P&L for
            current_prices: Optional dictionary of current prices for positions
            
        Returns:
            Dictionary containing P&L metrics including fee breakdown
        """
        try:
            if not self.portfolio_state:
                return {
                    'total_pnl': 0.0, 'unrealized_pnl': 0.0, 'realized_pnl': 0.0,
                    'gross_pnl': 0.0, 'total_fees': 0.0
                }
            
            total_cash = self.portfolio_state.get('total_cash_value', 0)
            positions_value = 0.0
            
            # Calculate positions value
            if current_prices:
                for pos_symbol, shares in self.portfolio_state.get('positions', {}).items():
                    if symbol is None or pos_symbol == symbol:
                        current_price = current_prices.get(pos_symbol, 100.0)  # Default price if not available
                        positions_value += shares * current_price
            
            net_liquidation = total_cash + positions_value
            initial_capital = self.portfolio_state.get('initial_capital', 100000)
            total_pnl = net_liquidation - initial_capital
            
            # Calculate fee-adjusted metrics
            symbol_fees = (
                self.fees_by_symbol.get(symbol, 0) if symbol 
                else self.fee_accumulator
            )
            gross_pnl = total_pnl + symbol_fees  # P&L before fees
            
            pnl_metrics = {
                'total_pnl': total_pnl,  # Net P&L (after fees)
                'gross_pnl': gross_pnl,  # P&L before fees
                'total_fees': symbol_fees,
                'fee_impact_pct': (
                    (symbol_fees / abs(gross_pnl)) * 100 
                    if abs(gross_pnl) > 0 else 0
                ),
                'total_pnl_pct': (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0,
                'gross_pnl_pct': (gross_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0,
                'net_liquidation': net_liquidation,
                'cash_value': total_cash,
                'positions_value': positions_value,
                'unrealized_pnl': positions_value,  # Simplified - would need cost basis for accurate calculation
                'realized_pnl': total_cash - initial_capital,  # Simplified
                'fee_drag_bps': (symbol_fees / initial_capital) * 10000 if initial_capital > 0 else 0
            }
            
            return pnl_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return {
                'total_pnl': 0.0, 'unrealized_pnl': 0.0, 'realized_pnl': 0.0,
                'gross_pnl': 0.0, 'total_fees': 0.0
            }
        
    def get_pnl_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get P&L history, optionally filtered by symbol."""
        if symbol is None:
            return self.pnl_history.copy()
        return [
            entry for entry in self.pnl_history
            if entry.get('symbol') == symbol
        ]
        
    def record_pnl_event(
        self,
        symbol: str,
        event_type: str,
        amount: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a P&L event in the history."""
        event = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'event_type': event_type,
            'amount': amount,
            'details': details or {}
        }
        self.pnl_history.append(event)