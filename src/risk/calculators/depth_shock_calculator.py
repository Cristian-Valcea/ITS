# src/risk/calculators/depth_shock_calculator.py
"""
Depth-at-Price Shock Calculator - HIGH priority sensor

Estimates price impact if positions were liquidated immediately
by sweeping through order book depth.

Priority: HIGH
Latency Target: <150µs
Action: THROTTLE when liquidation impact becomes excessive
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base_calculator import VectorizedCalculator, RiskCalculationResult, RiskMetricType


class DepthShockCalculator(VectorizedCalculator):
    """
    Depth-at-Price Shock Calculator - Liquidation impact estimation.
    
    Estimates the price impact if current positions were liquidated immediately
    by sweeping through available order book depth.
    
    Formula: Impact = Σ(volume_i × price_impact_i) / total_volume
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.DEPTH_SHOCK
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.max_impact_threshold = self.config.get('max_impact_threshold', 0.10)  # 10% max
        self.depth_levels = self.config.get('depth_levels', 5)  # Number of order book levels
        self.impact_model = self.config.get('impact_model', 'linear')  # linear, sqrt, log
        
        if not 0.01 <= self.max_impact_threshold <= 0.50:
            raise ValueError("max_impact_threshold must be between 1% and 50%")
        if self.depth_levels < 1:
            raise ValueError("depth_levels must be at least 1")
        if self.impact_model not in ['linear', 'sqrt', 'log']:
            raise ValueError("impact_model must be 'linear', 'sqrt', or 'log'")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['positions', 'current_prices', 'order_book_depth']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate depth shock with vectorized operations.
        
        Args:
            data: Must contain 'positions', 'current_prices', and 'order_book_depth'
            
        Returns:
            RiskCalculationResult with depth shock metrics
        """
        positions = data['positions']
        current_prices = data['current_prices']
        order_book_depth = data.get('order_book_depth', {})
        
        if not positions:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'max_impact': 0.0, 'total_impact': 0.0},
                metadata={'no_positions': True}
            )
        
        # Calculate impact for each position
        position_impacts = []
        total_notional = 0.0
        
        for symbol, position_size in positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            position_notional = abs(position_size * current_price)
            total_notional += position_notional
            
            # Calculate impact for this position
            impact = self._calculate_position_impact(
                symbol, abs(position_size), current_price, order_book_depth
            )
            
            position_impacts.append({
                'symbol': symbol,
                'position_size': position_size,
                'notional': position_notional,
                'impact': impact,
                'impact_dollars': impact * position_notional
            })
        
        if not position_impacts:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'max_impact': 0.0, 'total_impact': 0.0},
                metadata={'no_valid_positions': True}
            )
        
        # Calculate aggregate metrics (vectorized)
        impacts = np.array([p['impact'] for p in position_impacts])
        notionals = np.array([p['notional'] for p in position_impacts])
        impact_dollars = np.array([p['impact_dollars'] for p in position_impacts])
        
        # Weighted average impact
        if total_notional > 0:
            weighted_avg_impact = np.sum(impact_dollars) / total_notional
        else:
            weighted_avg_impact = 0.0
        
        # Maximum single position impact
        max_impact = np.max(impacts) if len(impacts) > 0 else 0.0
        
        # Portfolio-level impact (assuming simultaneous liquidation)
        portfolio_impact = self._calculate_portfolio_impact(position_impacts)
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'max_impact': float(max_impact),
                'weighted_avg_impact': float(weighted_avg_impact),
                'portfolio_impact': float(portfolio_impact),
                'total_impact_dollars': float(np.sum(impact_dollars)),
                'positions_at_risk': int(np.sum(impacts > 0.02)),  # Positions with >2% impact
                'worst_position': self._get_worst_position(position_impacts)
            },
            metadata={
                'total_positions': len(position_impacts),
                'total_notional': float(total_notional),
                'impact_model': self.impact_model,
                'vectorized': True
            }
        )
    
    def _calculate_position_impact(self, symbol: str, shares_to_sell: float, 
                                 current_price: float, order_book_depth: Dict[str, Any]) -> float:
        """Calculate price impact for a single position."""
        # Get order book data for this symbol
        depth_data = order_book_depth.get(symbol, {})
        
        if not depth_data:
            # Use simplified impact model based on position size
            return self._estimate_impact_from_size(shares_to_sell, current_price)
        
        # Use order book depth if available
        bid_levels = depth_data.get('bids', [])
        
        if not bid_levels:
            return self._estimate_impact_from_size(shares_to_sell, current_price)
        
        # Calculate impact by sweeping through bid levels
        return self._calculate_order_book_impact(shares_to_sell, bid_levels, current_price)
    
    def _estimate_impact_from_size(self, shares: float, price: float) -> float:
        """Estimate impact using simplified model when order book data unavailable."""
        notional = shares * price
        
        # Simple impact models based on notional size
        if self.impact_model == 'linear':
            # Linear: 1% impact per $1M notional
            impact = min(notional / 1_000_000 * 0.01, self.max_impact_threshold)
        elif self.impact_model == 'sqrt':
            # Square root: impact grows slower for large trades
            impact = min(np.sqrt(notional / 1_000_000) * 0.01, self.max_impact_threshold)
        elif self.impact_model == 'log':
            # Logarithmic: even slower growth for large trades
            if notional > 100_000:
                impact = min(np.log(notional / 100_000) * 0.005, self.max_impact_threshold)
            else:
                impact = 0.001  # Minimal impact for small trades
        else:
            impact = 0.01  # Default 1% impact
        
        return float(impact)
    
    def _calculate_order_book_impact(self, shares_to_sell: float, 
                                   bid_levels: List[tuple], current_price: float) -> float:
        """Calculate impact using actual order book depth."""
        remaining_shares = shares_to_sell
        total_proceeds = 0.0
        
        for price, size in bid_levels[:self.depth_levels]:
            if remaining_shares <= 0:
                break
            
            shares_at_level = min(remaining_shares, size)
            total_proceeds += shares_at_level * price
            remaining_shares -= shares_at_level
        
        if shares_to_sell > 0:
            # Calculate average execution price
            shares_executed = shares_to_sell - remaining_shares
            if shares_executed > 0:
                avg_price = total_proceeds / shares_executed
                impact = (current_price - avg_price) / current_price
                return max(0.0, min(impact, self.max_impact_threshold))
        
        return 0.0
    
    def _calculate_portfolio_impact(self, position_impacts: List[Dict[str, Any]]) -> float:
        """Calculate portfolio-level impact assuming simultaneous liquidation."""
        if not position_impacts:
            return 0.0
        
        # Simple aggregation: weighted average with correlation adjustment
        total_notional = sum(p['notional'] for p in position_impacts)
        
        if total_notional <= 0:
            return 0.0
        
        # Weighted impact
        weighted_impact = sum(p['impact_dollars'] for p in position_impacts) / total_notional
        
        # Add correlation penalty (simultaneous liquidation is worse)
        correlation_penalty = 1.2 if len(position_impacts) > 1 else 1.0
        
        portfolio_impact = weighted_impact * correlation_penalty
        
        return min(portfolio_impact, self.max_impact_threshold)
    
    def _get_worst_position(self, position_impacts: List[Dict[str, Any]]) -> Optional[str]:
        """Get the symbol with the worst impact."""
        if not position_impacts:
            return None
        
        worst_position = max(position_impacts, key=lambda x: x['impact'])
        return worst_position['symbol']