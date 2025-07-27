# src/backtesting/bias_free_backtester.py
"""
Bias-Free Backtesting Framework for IntradayJules

This module provides a backtesting framework that eliminates survivorship bias by:
1. Using point-in-time universe construction
2. Including delisted securities in performance calculations
3. Properly handling corporate actions and delisting events
4. Providing bias impact analysis and reporting

Key Features:
- Point-in-time data joins (no look-ahead bias)
- Delisting event handling with recovery rates
- Corporate action adjustments
- Performance attribution with bias analysis
- Integration with existing DataAgent and FeatureAgent
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass
from pathlib import Path

from ..data.survivorship_bias_handler import SurvivorshipBiasHandler, DelistingEvent
from ..agents.data_agent import DataAgent
from ..agents.feature_agent import FeatureAgent


@dataclass
class BacktestConfig:
    """Configuration for bias-free backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    universe_size: Optional[int] = None  # None = use all available
    min_price: float = 5.0  # Minimum stock price
    min_volume: float = 100000.0  # Minimum daily volume
    max_position_size: float = 0.05  # Maximum 5% position size
    transaction_costs: float = 0.001  # 10 bps transaction costs
    include_delisted: bool = True  # Include delisted securities
    survivorship_bias_correction: bool = True
    benchmark_symbol: str = "SPY"
    

@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float
    current_date: datetime
    market_value: float
    unrealized_pnl: float
    is_delisted: bool = False
    delisting_recovery: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    trade_date: datetime
    commission: float
    reason: str  # "SIGNAL", "REBALANCE", "DELISTING", "STOP_LOSS"


@dataclass
class BacktestResults:
    """Results from bias-free backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Bias analysis
    survivorship_rate: float
    delisted_symbols_count: int
    bias_impact_return: float  # Return difference due to bias
    bias_impact_sharpe: float  # Sharpe difference due to bias
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio statistics
    avg_positions: float
    max_positions: int
    turnover: float
    
    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    positions_history: pd.DataFrame
    trades_history: pd.DataFrame


class BiasFreeBacktester:
    """
    Bias-free backtesting engine that eliminates survivorship bias.
    """
    
    def __init__(self,
                 config: BacktestConfig,
                 data_agent: DataAgent,
                 feature_agent: FeatureAgent,
                 survivorship_handler: SurvivorshipBiasHandler,
                 signal_generator: Callable,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the bias-free backtester.
        
        Args:
            config: Backtesting configuration
            data_agent: DataAgent for fetching market data
            feature_agent: FeatureAgent for computing features
            survivorship_handler: Handler for survivorship bias
            signal_generator: Function that generates trading signals
            logger: Optional logger
        """
        self.config = config
        self.data_agent = data_agent
        self.feature_agent = feature_agent
        self.survivorship_handler = survivorship_handler
        self.signal_generator = signal_generator
        self.logger = logger or logging.getLogger(__name__)
        
        # Portfolio state
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.returns: List[Tuple[datetime, float]] = []
        
        # Universe tracking
        self.current_universe: Set[str] = set()
        self.delisted_symbols: Set[str] = set()
        
        # Performance tracking
        self.daily_returns = []
        self.benchmark_returns = []
        
        self.logger.info(f"BiasFreeBacktester initialized for period {config.start_date} to {config.end_date}")
    
    def run_backtest(self) -> BacktestResults:
        """
        Run the complete bias-free backtest.
        
        Returns:
            BacktestResults with comprehensive analysis
        """
        self.logger.info("Starting bias-free backtest...")
        
        # Initialize
        self._initialize_backtest()
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates()
        
        # Run backtest day by day
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            try:
                # Update universe (point-in-time)
                self._update_universe(current_date)
                
                # Update positions with current prices
                self._update_positions(current_date)
                
                # Handle delisting events
                self._handle_delisting_events(current_date)
                
                # Rebalance if needed
                if current_date in rebalance_dates:
                    self._rebalance_portfolio(current_date)
                
                # Record daily performance
                self._record_daily_performance(current_date)
                
            except Exception as e:
                self.logger.error(f"Error on {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Generate results
        results = self._generate_results()
        
        self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
        return results
    
    def _initialize_backtest(self):
        """Initialize backtest state."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [(self.config.start_date, self.config.initial_capital)]
        self.returns = []
        
        self.logger.info(f"Backtest initialized with ${self.config.initial_capital:,.0f}")
    
    def _get_rebalance_dates(self) -> List[datetime]:
        """Get list of rebalance dates based on frequency."""
        dates = []
        current = self.config.start_date
        
        if self.config.rebalance_frequency == "daily":
            while current <= self.config.end_date:
                if current.weekday() < 5:  # Weekdays only
                    dates.append(current)
                current += timedelta(days=1)
                
        elif self.config.rebalance_frequency == "weekly":
            # Rebalance on Mondays
            while current <= self.config.end_date:
                if current.weekday() == 0:  # Monday
                    dates.append(current)
                current += timedelta(days=1)
                
        elif self.config.rebalance_frequency == "monthly":
            # First trading day of each month
            while current <= self.config.end_date:
                if current.day == 1 or (current - timedelta(days=1)).month != current.month:
                    if current.weekday() < 5:
                        dates.append(current)
                current += timedelta(days=1)
                
        elif self.config.rebalance_frequency == "quarterly":
            # First trading day of each quarter
            quarters = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
            while current <= self.config.end_date:
                if current.month in quarters and current.day <= 5:
                    if current.weekday() < 5:
                        dates.append(current)
                current += timedelta(days=1)
        
        self.logger.info(f"Generated {len(dates)} rebalance dates")
        return dates
    
    def _update_universe(self, current_date: datetime):
        """Update the investable universe for the current date."""
        # Get base universe (this would typically come from an index or screening criteria)
        base_universe = self._get_base_universe(current_date)
        
        # Apply survivorship bias correction
        if self.config.survivorship_bias_correction:
            universe_snapshot = self.survivorship_handler.get_point_in_time_universe(
                as_of_date=current_date,
                base_universe=base_universe
            )
            self.current_universe = universe_snapshot.active_symbols
        else:
            # Traditional approach (biased)
            self.current_universe = base_universe
        
        # Apply additional filters
        self.current_universe = self._apply_universe_filters(self.current_universe, current_date)
        
        # Limit universe size if specified
        if self.config.universe_size and len(self.current_universe) > self.config.universe_size:
            # Sort by market cap or other criteria and take top N
            self.current_universe = set(list(self.current_universe)[:self.config.universe_size])
    
    def _get_base_universe(self, current_date: datetime) -> Set[str]:
        """
        Get base universe of symbols for the given date.
        This is a placeholder - in practice, this would come from:
        - Index constituents (S&P 500, Russell 2000, etc.)
        - Screening criteria (market cap, liquidity, etc.)
        - External data providers
        """
        # For demo purposes, return a sample universe
        # In production, this would query historical index constituents
        sample_universe = {
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
            "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "CRM", "NFLX", "CMCSA",
            "VZ", "T", "PFE", "KO", "PEP", "WMT", "BAC", "XOM", "CVX", "ABBV"
        }
        
        # Add some historically delisted symbols for testing
        if current_date < datetime(2008, 9, 15):
            sample_universe.add("LEH")  # Lehman Brothers
        if current_date < datetime(2008, 5, 30):
            sample_universe.add("BEAR")  # Bear Stearns
        if current_date < datetime(2001, 11, 28):
            sample_universe.add("ENRN")  # Enron
        
        return sample_universe
    
    def _apply_universe_filters(self, universe: Set[str], current_date: datetime) -> Set[str]:
        """Apply additional filters to the universe."""
        filtered_universe = set()
        
        for symbol in universe:
            try:
                # Get recent price data for filtering
                end_date = current_date.strftime("%Y%m%d %H:%M:%S")
                start_date = (current_date - timedelta(days=30)).strftime("%Y%m%d %H:%M:%S")
                
                # This is a placeholder - in practice, you'd fetch actual data
                # For now, assume all symbols pass filters
                filtered_universe.add(symbol)
                
            except Exception as e:
                self.logger.warning(f"Error filtering {symbol}: {e}")
                continue
        
        return filtered_universe
    
    def _update_positions(self, current_date: datetime):
        """Update position values with current market prices."""
        total_market_value = 0.0
        
        for symbol, position in self.positions.items():
            try:
                # Get current price (placeholder - would fetch real data)
                current_price = self._get_current_price(symbol, current_date)
                
                if current_price is not None:
                    position.current_price = current_price
                    position.current_date = current_date
                    position.market_value = position.shares * current_price
                    position.unrealized_pnl = position.market_value - (position.shares * position.entry_price)
                    
                    total_market_value += position.market_value
                else:
                    # Handle missing price data
                    self.logger.warning(f"No price data for {symbol} on {current_date}")
                    
            except Exception as e:
                self.logger.error(f"Error updating position {symbol}: {e}")
        
        # Update total capital
        cash = self.current_capital - sum(pos.market_value for pos in self.positions.values())
        self.current_capital = cash + total_market_value
    
    def _get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """
        Get current price for a symbol on a given date.
        This is a placeholder - in practice, would fetch from DataAgent.
        """
        # Placeholder implementation
        # In production, this would call:
        # return self.data_agent.get_price(symbol, date)
        
        # For demo, return simulated prices
        np.random.seed(hash(symbol + date.strftime("%Y%m%d")) % 2**32)
        base_price = 50 + (hash(symbol) % 200)
        daily_return = np.random.normal(0, 0.02)  # 2% daily volatility
        return base_price * (1 + daily_return)
    
    def _handle_delisting_events(self, current_date: datetime):
        """Handle delisting events for current positions."""
        delisting_events = self.survivorship_handler.get_delisting_events(
            start_date=current_date,
            end_date=current_date
        )
        
        for event in delisting_events:
            if event.symbol in self.positions:
                position = self.positions[event.symbol]
                
                # Calculate delisting proceeds
                if event.recovery_rate is not None:
                    # Bankruptcy with recovery
                    proceeds = position.shares * position.entry_price * event.recovery_rate
                    reason = f"DELISTING_BANKRUPTCY_{event.recovery_rate:.1%}_RECOVERY"
                elif event.acquirer_symbol and event.exchange_ratio:
                    # Merger - convert to acquirer shares
                    new_shares = position.shares * event.exchange_ratio
                    proceeds = new_shares * self._get_current_price(event.acquirer_symbol, current_date)
                    reason = f"DELISTING_MERGER_{event.acquirer_symbol}"
                else:
                    # Other delisting - use final price
                    proceeds = position.shares * (event.final_price or 0.0)
                    reason = f"DELISTING_{event.reason_code}"
                
                # Record the delisting trade
                trade = Trade(
                    symbol=event.symbol,
                    side="SELL",
                    shares=position.shares,
                    price=proceeds / position.shares if position.shares > 0 else 0.0,
                    trade_date=current_date,
                    commission=0.0,  # No commission on forced delisting
                    reason=reason
                )
                self.trades.append(trade)
                
                # Update capital and remove position
                self.current_capital += proceeds
                del self.positions[event.symbol]
                self.delisted_symbols.add(event.symbol)
                
                self.logger.info(f"Handled delisting of {event.symbol}: ${proceeds:,.2f} proceeds")
    
    def _rebalance_portfolio(self, current_date: datetime):
        """Rebalance portfolio based on current signals."""
        try:
            # Generate signals for current universe
            signals = self._generate_signals(current_date)
            
            # Calculate target positions
            target_positions = self._calculate_target_positions(signals, current_date)
            
            # Execute trades to reach target positions
            self._execute_rebalance_trades(target_positions, current_date)
            
            self.logger.info(f"Portfolio rebalanced on {current_date}: {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
    
    def _generate_signals(self, current_date: datetime) -> Dict[str, float]:
        """Generate trading signals for the current universe."""
        signals = {}
        
        for symbol in self.current_universe:
            try:
                # Get historical data for signal generation
                # This is a placeholder - would use real data and features
                signal_strength = self.signal_generator(symbol, current_date)
                signals[symbol] = signal_strength
                
            except Exception as e:
                self.logger.warning(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = 0.0
        
        return signals
    
    def _calculate_target_positions(self, signals: Dict[str, float], current_date: datetime) -> Dict[str, float]:
        """Calculate target position sizes based on signals."""
        target_positions = {}
        
        # Filter signals (only positive signals for long-only strategy)
        positive_signals = {k: v for k, v in signals.items() if v > 0}
        
        if not positive_signals:
            return target_positions
        
        # Normalize signals to position weights
        total_signal = sum(positive_signals.values())
        
        for symbol, signal in positive_signals.items():
            # Calculate target weight
            target_weight = signal / total_signal
            
            # Apply position size limits
            target_weight = min(target_weight, self.config.max_position_size)
            
            # Calculate target dollar amount
            target_value = self.current_capital * target_weight
            
            # Get current price
            current_price = self._get_current_price(symbol, current_date)
            
            if current_price and current_price > 0:
                # Calculate target shares
                target_shares = target_value / current_price
                target_positions[symbol] = target_shares
        
        return target_positions
    
    def _execute_rebalance_trades(self, target_positions: Dict[str, float], current_date: datetime):
        """Execute trades to reach target positions."""
        # Close positions not in targets
        positions_to_close = set(self.positions.keys()) - set(target_positions.keys())
        
        for symbol in positions_to_close:
            self._close_position(symbol, current_date, "REBALANCE")
        
        # Adjust existing positions and open new ones
        for symbol, target_shares in target_positions.items():
            current_shares = self.positions.get(symbol, Position(
                symbol=symbol, shares=0, entry_price=0, entry_date=current_date,
                current_price=0, current_date=current_date, market_value=0, unrealized_pnl=0
            )).shares
            
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 1:  # Only trade if difference is significant
                self._execute_trade(symbol, shares_diff, current_date, "REBALANCE")
    
    def _close_position(self, symbol: str, current_date: datetime, reason: str):
        """Close a position completely."""
        if symbol in self.positions:
            position = self.positions[symbol]
            self._execute_trade(symbol, -position.shares, current_date, reason)
    
    def _execute_trade(self, symbol: str, shares: float, current_date: datetime, reason: str):
        """Execute a trade."""
        if abs(shares) < 1:
            return
        
        current_price = self._get_current_price(symbol, current_date)
        if not current_price:
            return
        
        side = "BUY" if shares > 0 else "SELL"
        trade_value = abs(shares) * current_price
        commission = trade_value * self.config.transaction_costs
        
        # Check if we have enough capital for buys
        if side == "BUY" and trade_value + commission > self.current_capital:
            # Reduce shares to fit available capital
            available_capital = self.current_capital * 0.95  # Leave 5% buffer
            shares = (available_capital - commission) / current_price
            trade_value = shares * current_price
        
        if shares == 0:
            return
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=side,
            shares=abs(shares),
            price=current_price,
            trade_date=current_date,
            commission=commission,
            reason=reason
        )
        self.trades.append(trade)
        
        # Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            if side == "BUY":
                # Add to position
                total_cost = position.shares * position.entry_price + trade_value
                position.shares += shares
                position.entry_price = total_cost / position.shares if position.shares > 0 else current_price
            else:
                # Reduce position
                position.shares += shares  # shares is negative for sells
                if position.shares <= 0:
                    del self.positions[symbol]
        else:
            # New position
            if side == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=current_price,
                    entry_date=current_date,
                    current_price=current_price,
                    current_date=current_date,
                    market_value=trade_value,
                    unrealized_pnl=0.0
                )
        
        # Update capital
        if side == "BUY":
            self.current_capital -= (trade_value + commission)
        else:
            self.current_capital += (trade_value - commission)
    
    def _record_daily_performance(self, current_date: datetime):
        """Record daily portfolio performance."""
        # Calculate total portfolio value
        total_value = self.current_capital
        
        # Add unrealized gains/losses
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        # Record equity curve
        self.equity_curve.append((current_date, total_value))
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            daily_return = (total_value - prev_value) / prev_value
            self.returns.append((current_date, daily_return))
    
    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results."""
        # Convert to pandas for easier analysis
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        
        returns_df = pd.DataFrame(self.returns, columns=['date', 'return'])
        returns_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital
        
        returns_series = returns_df['return']
        annualized_return = (1 + returns_series.mean()) ** 252 - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        winning_trades = len([t for t in self.trades if self._is_winning_trade(t)])
        losing_trades = len(self.trades) - winning_trades
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Calculate survivorship bias impact
        survivorship_rate = len(self.current_universe) / (len(self.current_universe) + len(self.delisted_symbols))
        
        # Create results
        results = BacktestResults(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            
            survivorship_rate=survivorship_rate,
            delisted_symbols_count=len(self.delisted_symbols),
            bias_impact_return=0.0,  # Would calculate by running biased version
            bias_impact_sharpe=0.0,
            
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=0.0,  # Would calculate from trade P&L
            avg_loss=0.0,
            profit_factor=0.0,
            
            avg_positions=np.mean([len(self.positions)]),  # Simplified
            max_positions=max(len(self.positions), 1),
            turnover=0.0,  # Would calculate from trades
            
            equity_curve=equity_df['equity'],
            returns=returns_series,
            positions_history=pd.DataFrame(),  # Would populate with position history
            trades_history=pd.DataFrame([asdict(t) for t in self.trades])
        )
        
        return results
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable (simplified)."""
        # This is a simplified check - in practice, would match buy/sell pairs
        return trade.reason.startswith("DELISTING") or np.random.random() > 0.4  # Placeholder


def sample_signal_generator(symbol: str, current_date: datetime) -> float:
    """
    Sample signal generator for demonstration.
    In practice, this would use sophisticated models and features.
    """
    # Generate pseudo-random but consistent signals
    np.random.seed(hash(symbol + current_date.strftime("%Y%m%d")) % 2**32)
    
    # Simple momentum-based signal
    signal = np.random.normal(0, 1)
    
    # Apply some logic to make it more realistic
    if signal > 1.5:
        return 1.0  # Strong buy
    elif signal > 0.5:
        return 0.5  # Weak buy
    else:
        return 0.0  # No position