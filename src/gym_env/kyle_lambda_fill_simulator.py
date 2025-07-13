# src/gym_env/kyle_lambda_fill_simulator.py
"""
Kyle Lambda Fill Price Simulator for Backtesting.

Integrates Kyle's Lambda market impact model to provide realistic fill prices
instead of mid-price fills, improving backtest accuracy at size.

Uses the same Kyle-λ calculation as RiskAgent for consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import logging

try:
    from ..risk.calculators.kyle_lambda_calculator import KyleLambdaCalculator
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from risk.calculators.kyle_lambda_calculator import KyleLambdaCalculator


class KyleLambdaFillSimulator:
    """
    Fill price simulator using Kyle's Lambda market impact model.
    
    Provides realistic fill prices that account for market impact,
    replacing naive mid-price fills in backtesting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Kyle Lambda fill simulator.
        
        Args:
            config: Configuration dictionary with Kyle Lambda parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default configuration - ENHANCED MARKET IMPACT
        default_config = {
            'lookback_period': 50,
            'min_periods': 10,
            'impact_decay': 0.7,       # 🔥 REDUCED: More persistent impact (was 0.9)
            'bid_ask_spread_bps': 5.0,  # Default 5 bps spread
            'min_impact_bps': 0.5,     # Minimum impact (0.5 bps)
            'max_impact_bps': 100.0,   # 🚨 DOUBLED: Higher impact cap (was 50 bps)
            'temporary_impact_decay': 0.5,  # Temporary impact decay factor
            'enable_bid_ask_bounce': True,  # Add bid-ask bounce effect
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize Kyle Lambda calculator
        kyle_config = {
            'lookback_period': self.config['lookback_period'],
            'min_periods': self.config['min_periods'],
            'impact_decay': self.config['impact_decay']
        }
        self.kyle_calculator = KyleLambdaCalculator(kyle_config)
        
        # Market data buffers
        self.price_history = deque(maxlen=self.config['lookback_period'])
        self.order_flow_history = deque(maxlen=self.config['lookback_period'])
        self.volume_history = deque(maxlen=self.config['lookback_period'])
        
        # Current market state
        self.current_kyle_lambda = 0.0
        self.current_spread_bps = self.config['bid_ask_spread_bps']
        self.last_fill_impact = 0.0
        
        # Performance tracking
        self.fill_count = 0
        self.total_impact_bps = 0.0
        
        self.logger.info(f"KyleLambdaFillSimulator initialized with config: {self.config}")
    
    def update_market_data(self, 
                          price: float, 
                          volume: float = None, 
                          timestamp: pd.Timestamp = None) -> None:
        """
        Update market data for Kyle Lambda calculation.
        
        Args:
            price: Current market price (mid-price)
            volume: Trading volume (optional)
            timestamp: Current timestamp (optional)
        """
        # Calculate price change
        if len(self.price_history) > 0:
            price_change = price - self.price_history[-1]
        else:
            price_change = 0.0
        
        # Add to history
        self.price_history.append(price)
        
        # Use volume or default
        if volume is None:
            volume = 1000.0  # Default volume
        self.volume_history.append(volume)
        
        # For order flow, we need to track net buying/selling pressure
        # In backtesting, we don't have real order flow, so we'll estimate
        # from price changes and volume
        estimated_order_flow = price_change * volume if price_change != 0 else 0.0
        self.order_flow_history.append(estimated_order_flow)
        
        # Update Kyle Lambda if we have enough data
        if len(self.price_history) >= self.config['min_periods']:
            self._update_kyle_lambda()
    
    def _update_kyle_lambda(self) -> None:
        """Update Kyle Lambda calculation with current market data."""
        try:
            # Prepare data for Kyle Lambda calculator
            price_changes = np.array([
                self.price_history[i] - self.price_history[i-1] 
                for i in range(1, len(self.price_history))
            ])
            
            order_flows = np.array(list(self.order_flow_history)[1:])  # Skip first element
            
            # Calculate Kyle Lambda
            data = {
                'price_changes': price_changes,
                'order_flows': order_flows
            }
            
            result = self.kyle_calculator.calculate(data)
            
            if result.values:
                self.current_kyle_lambda = result.values.get('kyle_lambda', 0.0)
                
                # Log significant changes
                if abs(self.current_kyle_lambda) > 1e-6:
                    self.logger.debug(f"Kyle Lambda updated: {self.current_kyle_lambda:.8f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update Kyle Lambda: {e}")
            self.current_kyle_lambda = 0.0
    
    def calculate_fill_price(self, 
                           mid_price: float,
                           trade_size: float,
                           side: str,
                           current_volume: float = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate realistic fill price using Kyle Lambda model.
        
        Args:
            mid_price: Current mid-market price
            trade_size: Size of trade (positive for buy, negative for sell)
            side: Trade side ("buy" or "sell")
            current_volume: Current market volume (optional)
        
        Returns:
            Tuple of (fill_price, impact_info)
        """
        self.fill_count += 1
        
        # Determine trade direction and size
        if side.lower() in ['buy', 'long', 'bot']:
            trade_direction = 1
            abs_trade_size = abs(trade_size)
        elif side.lower() in ['sell', 'short', 'sld']:
            trade_direction = -1
            abs_trade_size = abs(trade_size)
        else:
            raise ValueError(f"Invalid trade side: {side}")
        
        # Calculate market impact components
        impact_info = self._calculate_market_impact(
            mid_price, abs_trade_size, trade_direction, current_volume
        )
        
        # Calculate fill price
        total_impact_bps = impact_info['total_impact_bps']
        impact_price_adjustment = mid_price * (total_impact_bps / 10000.0)
        
        # Apply impact in the direction that hurts the trader
        fill_price = mid_price + (trade_direction * impact_price_adjustment)
        
        # Ensure fill price is positive
        fill_price = max(fill_price, mid_price * 0.01)  # Minimum 1% of mid price
        
        # Update tracking
        self.total_impact_bps += abs(total_impact_bps)
        self.last_fill_impact = total_impact_bps
        
        # Enhanced impact info
        impact_info.update({
            'mid_price': mid_price,
            'fill_price': fill_price,
            'trade_size': trade_size,
            'trade_direction': trade_direction,
            'price_improvement_bps': -total_impact_bps,  # Negative because it's cost
            'fill_count': self.fill_count
        })
        
        # 🔍 ENHANCED LOGGING: Show market impact in action
        if total_impact_bps > 1.0:  # Log significant impacts
            self.logger.info(
                f"🔍 KYLE LAMBDA IMPACT: Mid=${mid_price:.4f} → Fill=${fill_price:.4f} "
                f"(Impact: {total_impact_bps:.2f}bps, Size: {abs_trade_size:.0f} shares, Side: {side})"
            )
        else:
            self.logger.debug(
                f"Fill calculated: mid={mid_price:.4f}, fill={fill_price:.4f}, "
                f"impact={total_impact_bps:.2f}bps, size={abs_trade_size:.0f}"
            )
        
        return fill_price, impact_info
    
    def _calculate_market_impact(self, 
                               mid_price: float,
                               trade_size: float,
                               direction: int,
                               volume: float = None) -> Dict[str, Any]:
        """
        Calculate market impact components using Kyle Lambda model.
        
        Args:
            mid_price: Current mid price
            trade_size: Absolute trade size
            direction: Trade direction (1 for buy, -1 for sell)
            volume: Current volume (optional)
        
        Returns:
            Dictionary with impact breakdown
        """
        # Base bid-ask spread impact
        spread_impact_bps = self.current_spread_bps / 2.0  # Half spread
        
        # Kyle Lambda permanent impact
        if self.current_kyle_lambda > 0 and trade_size > 0:
            # Kyle Lambda gives price impact per unit of order flow
            # Convert trade size to order flow equivalent
            notional_value = trade_size * mid_price
            
            # Calculate permanent impact
            permanent_impact = self.current_kyle_lambda * notional_value
            permanent_impact_bps = (permanent_impact / mid_price) * 10000.0
            
            # Apply bounds
            permanent_impact_bps = np.clip(
                permanent_impact_bps,
                self.config['min_impact_bps'],
                self.config['max_impact_bps']
            )
        else:
            permanent_impact_bps = self.config['min_impact_bps']
        
        # Temporary impact (square root law approximation)
        if volume is None:
            volume = np.mean(list(self.volume_history)) if self.volume_history else 1000.0
        
        if volume > 0:
            participation_rate = trade_size / volume
            temporary_impact_bps = self.config['temporary_impact_decay'] * np.sqrt(participation_rate) * 10.0
            temporary_impact_bps = min(temporary_impact_bps, 20.0)  # Cap at 20 bps
        else:
            temporary_impact_bps = 0.0
        
        # Bid-ask bounce (random component)
        if self.config['enable_bid_ask_bounce']:
            bounce_impact_bps = np.random.uniform(-0.5, 0.5)  # Random ±0.5 bps
        else:
            bounce_impact_bps = 0.0
        
        # Total impact
        total_impact_bps = spread_impact_bps + permanent_impact_bps + temporary_impact_bps + bounce_impact_bps
        
        return {
            'spread_impact_bps': spread_impact_bps,
            'permanent_impact_bps': permanent_impact_bps,
            'temporary_impact_bps': temporary_impact_bps,
            'bounce_impact_bps': bounce_impact_bps,
            'total_impact_bps': total_impact_bps,
            'kyle_lambda': self.current_kyle_lambda,
            'participation_rate': participation_rate if volume > 0 else 0.0,
            'volume': volume,
            'notional_value': trade_size * mid_price
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        if self.fill_count == 0:
            return {
                'fill_count': 0,
                'average_impact_bps': 0.0,
                'total_impact_bps': 0.0,
                'current_kyle_lambda': 0.0
            }
        
        return {
            'fill_count': self.fill_count,
            'average_impact_bps': self.total_impact_bps / self.fill_count,
            'total_impact_bps': self.total_impact_bps,
            'current_kyle_lambda': self.current_kyle_lambda,
            'last_fill_impact_bps': self.last_fill_impact,
            'data_points': len(self.price_history),
            'spread_bps': self.current_spread_bps
        }
    
    def reset(self) -> None:
        """Reset simulator state for new episode."""
        self.price_history.clear()
        self.order_flow_history.clear()
        self.volume_history.clear()
        
        self.current_kyle_lambda = 0.0
        self.current_spread_bps = self.config['bid_ask_spread_bps']
        self.last_fill_impact = 0.0
        
        self.fill_count = 0
        self.total_impact_bps = 0.0
        
        self.logger.debug("KyleLambdaFillSimulator reset")


class FillPriceSimulatorFactory:
    """Factory for creating fill price simulators."""
    
    @staticmethod
    def create_kyle_lambda_simulator(config: Dict[str, Any] = None) -> KyleLambdaFillSimulator:
        """Create Kyle Lambda fill simulator with configuration."""
        return KyleLambdaFillSimulator(config)
    
    @staticmethod
    def create_simple_impact_simulator(spread_bps: float = 5.0, 
                                     impact_bps_per_million: float = 2.0) -> 'SimpleFillSimulator':
        """Create simple linear impact simulator."""
        config = {
            'spread_bps': spread_bps,
            'impact_bps_per_million': impact_bps_per_million
        }
        return SimpleFillSimulator(config)


class SimpleFillSimulator:
    """
    Simple linear impact fill simulator for comparison.
    
    Uses linear impact model: impact = spread/2 + size * impact_rate
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spread_bps = config.get('spread_bps', 5.0)
        self.impact_bps_per_million = config.get('impact_bps_per_million', 2.0)
        self.fill_count = 0
        self.total_impact_bps = 0.0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update_market_data(self, price: float, volume: float = None, timestamp: pd.Timestamp = None) -> None:
        """Update market data (no-op for simple simulator)."""
        pass
    
    def calculate_fill_price(self, 
                           mid_price: float,
                           trade_size: float,
                           side: str,
                           current_volume: float = None) -> Tuple[float, Dict[str, Any]]:
        """Calculate fill price using simple linear impact."""
        self.fill_count += 1
        
        # Determine direction
        direction = 1 if side.lower() in ['buy', 'long', 'bot'] else -1
        abs_trade_size = abs(trade_size)
        
        # Calculate impact
        spread_impact_bps = self.spread_bps / 2.0
        size_impact_bps = (abs_trade_size * mid_price / 1_000_000) * self.impact_bps_per_million
        total_impact_bps = spread_impact_bps + size_impact_bps
        
        # Apply impact
        impact_price_adjustment = mid_price * (total_impact_bps / 10000.0)
        fill_price = mid_price + (direction * impact_price_adjustment)
        
        # Update tracking
        self.total_impact_bps += total_impact_bps
        
        impact_info = {
            'mid_price': mid_price,
            'fill_price': fill_price,
            'total_impact_bps': total_impact_bps,
            'spread_impact_bps': spread_impact_bps,
            'size_impact_bps': size_impact_bps,
            'trade_size': trade_size,
            'fill_count': self.fill_count
        }
        
        return fill_price, impact_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'fill_count': self.fill_count,
            'average_impact_bps': self.total_impact_bps / max(self.fill_count, 1),
            'total_impact_bps': self.total_impact_bps,
            'spread_bps': self.spread_bps,
            'impact_rate': self.impact_bps_per_million
        }
    
    def reset(self) -> None:
        """Reset simulator state."""
        self.fill_count = 0
        self.total_impact_bps = 0.0


__all__ = [
    'KyleLambdaFillSimulator',
    'SimpleFillSimulator', 
    'FillPriceSimulatorFactory'
]