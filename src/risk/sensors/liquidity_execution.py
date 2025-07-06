# src/risk/sensors/liquidity_execution.py
"""
Liquidity & Execution Risk Sensors - "Can I unwind before the market notices?"

These sensors detect when position sizes become too large relative to market liquidity,
creating execution risk and market impact concerns.

Sensors:
1. ADVParticipationSensor - % of daily volume needed to exit
2. DepthAtPriceShockSensor - Price impact of immediate liquidation
3. KyleLambdaSensor - Market impact slope (Kyle's lambda)
"""

import numpy as np
from typing import Dict, Any, List

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class ADVParticipationSensor(BaseSensor):
    """
    Average Daily Volume (ADV) Participation Sensor.
    
    Measures what percentage of the average daily volume would be needed
    to exit current positions. High participation rates indicate liquidity risk.
    
    Formula: ADV_Participation = |position_size| / (20_day_ADV)
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.LIQUIDITY_EXECUTION
    
    def _get_data_requirements(self) -> List[str]:
        return ['positions', 'daily_volumes', 'current_prices']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute maximum ADV participation across all positions."""
        positions = data.get('positions', {})
        daily_volumes = data.get('daily_volumes', {})
        current_prices = data.get('current_prices', {})
        
        if not positions:
            return 0.0
        
        max_participation = 0.0
        
        for symbol, position_size in positions.items():
            if symbol not in daily_volumes or symbol not in current_prices:
                continue
            
            # Get 20-day average daily volume
            volumes = daily_volumes.get(symbol, [])
            if len(volumes) == 0:
                continue
            
            avg_daily_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            current_price = current_prices[symbol]
            
            # Calculate position value
            position_value = abs(position_size * current_price)
            
            # Calculate daily volume value (assuming average price)
            daily_volume_value = avg_daily_volume * current_price
            
            if daily_volume_value > 0:
                participation = position_value / daily_volume_value
                max_participation = max(max_participation, participation)
        
        return max_participation
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality."""
        base_confidence = super()._compute_confidence(value, data)
        
        daily_volumes = data.get('daily_volumes', {})
        positions = data.get('positions', {})
        
        # Check data coverage
        covered_positions = 0
        total_positions = len(positions)
        
        for symbol in positions.keys():
            if symbol in daily_volumes and len(daily_volumes[symbol]) >= 5:
                covered_positions += 1
        
        if total_positions > 0:
            coverage_ratio = covered_positions / total_positions
            coverage_bonus = coverage_ratio * 0.2
        else:
            coverage_bonus = 0.0
        
        confidence = base_confidence + coverage_bonus
        return max(0.0, min(1.0, confidence))


class DepthAtPriceShockSensor(BaseSensor):
    """
    Depth-at-Price Shock Sensor.
    
    Estimates the price impact if positions were liquidated immediately
    by sweeping through the order book depth.
    
    Formula: Price_Impact = Σ(volume_i * price_impact_i) / total_volume
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.LIQUIDITY_EXECUTION
    
    def _get_data_requirements(self) -> List[str]:
        return ['positions', 'order_book_depth', 'current_prices']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute maximum price impact across all positions."""
        positions = data.get('positions', {})
        order_book_depth = data.get('order_book_depth', {})
        current_prices = data.get('current_prices', {})
        
        if not positions:
            return 0.0
        
        max_impact = 0.0
        
        for symbol, position_size in positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Get order book depth (simplified model)
            depth_data = order_book_depth.get(symbol, {})
            
            if not depth_data:
                # Use simplified impact model based on position size
                # Assume 1% impact per $1M position value
                position_value = abs(position_size * current_price)
                estimated_impact = position_value / 1_000_000 * 0.01
                max_impact = max(max_impact, estimated_impact)
                continue
            
            # Calculate impact using order book depth
            impact = self._calculate_order_book_impact(
                abs(position_size), depth_data, current_price
            )
            max_impact = max(max_impact, impact)
        
        return max_impact
    
    def _calculate_order_book_impact(self, shares_to_sell: float, 
                                   depth_data: Dict[str, Any], 
                                   current_price: float) -> float:
        """Calculate price impact using order book depth."""
        # Simplified order book impact model
        # In practice, this would use real bid/ask depth data
        
        # Get bid levels (for selling)
        bid_levels = depth_data.get('bids', [])
        
        if not bid_levels:
            # Fallback to simple impact model
            return min(0.1, shares_to_sell / 10000 * 0.01)  # 1% per 10k shares
        
        remaining_shares = shares_to_sell
        total_proceeds = 0.0
        
        for price, size in bid_levels:
            if remaining_shares <= 0:
                break
            
            shares_at_level = min(remaining_shares, size)
            total_proceeds += shares_at_level * price
            remaining_shares -= shares_at_level
        
        if shares_to_sell > 0:
            average_price = total_proceeds / (shares_to_sell - remaining_shares)
            impact = (current_price - average_price) / current_price
            return max(0.0, impact)
        
        return 0.0
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on order book data quality."""
        base_confidence = super()._compute_confidence(value, data)
        
        order_book_depth = data.get('order_book_depth', {})
        positions = data.get('positions', {})
        
        # Check if we have real order book data
        real_data_count = 0
        for symbol in positions.keys():
            if symbol in order_book_depth and order_book_depth[symbol]:
                real_data_count += 1
        
        if len(positions) > 0:
            real_data_ratio = real_data_count / len(positions)
            data_quality_bonus = real_data_ratio * 0.3
        else:
            data_quality_bonus = 0.0
        
        confidence = base_confidence + data_quality_bonus
        return max(0.0, min(1.0, confidence))


class KyleLambdaSensor(BaseSensor):
    """
    Kyle's Lambda Sensor - Market impact slope.
    
    Measures the permanent price impact per unit of order flow.
    High lambda values indicate that even small trades will move the market.
    
    Formula: Lambda = Cov(price_change, order_flow) / Var(order_flow)
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.LIQUIDITY_EXECUTION
    
    def _get_data_requirements(self) -> List[str]:
        return ['price_changes', 'order_flows', 'timestamps']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute Kyle's lambda from price changes and order flows."""
        price_changes = np.array(data.get('price_changes', []))
        order_flows = np.array(data.get('order_flows', []))
        
        if len(price_changes) < 10 or len(order_flows) < 10:
            return 0.0
        
        # Ensure arrays are same length
        min_length = min(len(price_changes), len(order_flows))
        price_changes = price_changes[-min_length:]
        order_flows = order_flows[-min_length:]
        
        # Calculate Kyle's lambda
        if np.var(order_flows) == 0:
            return 0.0
        
        lambda_kyle = np.cov(price_changes, order_flows)[0, 1] / np.var(order_flows)
        
        return abs(lambda_kyle)
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and correlation strength."""
        base_confidence = super()._compute_confidence(value, data)
        
        price_changes = np.array(data.get('price_changes', []))
        order_flows = np.array(data.get('order_flows', []))
        
        # Higher confidence with more data points
        min_length = min(len(price_changes), len(order_flows))
        if min_length >= 50:
            sample_bonus = 0.2
        elif min_length >= 20:
            sample_bonus = 0.1
        else:
            sample_bonus = 0.0
        
        # Check correlation strength
        if min_length >= 10:
            try:
                correlation = np.corrcoef(price_changes[-min_length:], 
                                       order_flows[-min_length:])[0, 1]
                if abs(correlation) > 0.3:
                    correlation_bonus = 0.1
                else:
                    correlation_bonus = 0.0
            except:
                correlation_bonus = 0.0
        else:
            correlation_bonus = 0.0
        
        confidence = base_confidence + sample_bonus + correlation_bonus
        return max(0.0, min(1.0, confidence))