# src/risk/calculators/adv_participation_calculator.py
"""
ADV Participation Calculator - LOW priority sensor

Calculates what percentage of Average Daily Volume would be needed
to exit current positions.

Priority: LOW (monitor lane)
Latency Target: <500µs
Action: MONITOR when participation becomes excessive
"""

import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class ADVParticipationCalculator(BaseRiskCalculator):
    """
    Average Daily Volume Participation Calculator.
    
    Measures what percentage of the average daily volume would be needed
    to exit current positions. High participation rates indicate liquidity risk.
    
    Formula: ADV_Participation = |position_size| / (N_day_ADV)
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.ADV_PARTICIPATION
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        self.adv_lookback_days = self.config.get('adv_lookback_days', 20)
        self.participation_threshold = self.config.get('participation_threshold', 0.20)  # 20%
        self.min_volume_days = self.config.get('min_volume_days', 5)
        
        if self.adv_lookback_days < 1:
            raise ValueError("adv_lookback_days must be at least 1")
        if not 0.01 <= self.participation_threshold <= 1.0:
            raise ValueError("participation_threshold must be between 1% and 100%")
        if self.min_volume_days < 1:
            raise ValueError("min_volume_days must be at least 1")
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['positions', 'daily_volumes', 'current_prices']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate ADV participation rates.
        
        Args:
            data: Must contain 'positions', 'daily_volumes', and 'current_prices'
            
        Returns:
            RiskCalculationResult with participation metrics
        """
        positions = data['positions']
        daily_volumes = data['daily_volumes']
        current_prices = data['current_prices']
        
        if not positions:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'max_participation': 0.0, 'weighted_participation': 0.0},
                metadata={'no_positions': True}
            )
        
        # Calculate participation for each position
        participation_data = []
        total_position_value = 0.0
        
        for symbol, position_size in positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            position_value = abs(position_size * current_price)
            total_position_value += position_value
            
            # Calculate ADV participation for this position
            participation = self._calculate_symbol_participation(
                symbol, abs(position_size), current_price, daily_volumes
            )
            
            participation_data.append({
                'symbol': symbol,
                'position_size': position_size,
                'position_value': position_value,
                'participation': participation,
                'adv_available': self._get_symbol_adv(symbol, daily_volumes)
            })
        
        if not participation_data:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'max_participation': 0.0, 'weighted_participation': 0.0},
                metadata={'no_valid_positions': True}
            )
        
        # Calculate aggregate metrics
        participations = [p['participation'] for p in participation_data]
        position_values = [p['position_value'] for p in participation_data]
        
        # Maximum single position participation
        max_participation = max(participations) if participations else 0.0
        
        # Value-weighted average participation
        if total_position_value > 0:
            weighted_participation = sum(
                p['participation'] * p['position_value'] 
                for p in participation_data
            ) / total_position_value
        else:
            weighted_participation = 0.0
        
        # Identify high-participation positions
        high_participation_positions = [
            p for p in participation_data 
            if p['participation'] > self.participation_threshold
        ]
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'max_participation': float(max_participation),
                'weighted_participation': float(weighted_participation),
                'positions_over_threshold': len(high_participation_positions),
                'worst_position_symbol': self._get_worst_position(participation_data),
                'total_liquidity_risk': float(self._calculate_total_liquidity_risk(participation_data)),
                'diversification_benefit': float(self._calculate_diversification_benefit(participation_data))
            },
            metadata={
                'total_positions': len(participation_data),
                'total_position_value': float(total_position_value),
                'adv_lookback_days': self.adv_lookback_days,
                'participation_threshold': self.participation_threshold
            }
        )
    
    def _calculate_symbol_participation(self, symbol: str, position_size: float, 
                                     current_price: float, daily_volumes: Dict[str, List[float]]) -> float:
        """Calculate ADV participation for a single symbol."""
        if symbol not in daily_volumes:
            return 0.0
        
        volumes = daily_volumes[symbol]
        if len(volumes) < self.min_volume_days:
            return 0.0
        
        # Use recent volume data
        recent_volumes = volumes[-self.adv_lookback_days:] if len(volumes) > self.adv_lookback_days else volumes
        
        # Calculate average daily volume
        avg_daily_volume = np.mean(recent_volumes)
        
        if avg_daily_volume <= 0:
            return 1.0  # Assume 100% participation if no volume data
        
        # Calculate participation rate
        participation = position_size / avg_daily_volume
        
        return min(participation, 1.0)  # Cap at 100%
    
    def _get_symbol_adv(self, symbol: str, daily_volumes: Dict[str, List[float]]) -> float:
        """Get average daily volume for a symbol."""
        if symbol not in daily_volumes:
            return 0.0
        
        volumes = daily_volumes[symbol]
        if len(volumes) < self.min_volume_days:
            return 0.0
        
        recent_volumes = volumes[-self.adv_lookback_days:] if len(volumes) > self.adv_lookback_days else volumes
        return float(np.mean(recent_volumes))
    
    def _get_worst_position(self, participation_data: List[Dict[str, Any]]) -> str:
        """Get symbol with highest participation rate."""
        if not participation_data:
            return ""
        
        worst_position = max(participation_data, key=lambda x: x['participation'])
        return worst_position['symbol']
    
    def _calculate_total_liquidity_risk(self, participation_data: List[Dict[str, Any]]) -> float:
        """Calculate total portfolio liquidity risk score."""
        if not participation_data:
            return 0.0
        
        # Simple risk score: sum of squared participations (penalizes concentration)
        risk_score = sum(p['participation'] ** 2 for p in participation_data)
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _calculate_diversification_benefit(self, participation_data: List[Dict[str, Any]]) -> float:
        """Calculate diversification benefit (lower is better for risk)."""
        if len(participation_data) <= 1:
            return 0.0
        
        # Calculate Herfindahl index for position concentration
        total_value = sum(p['position_value'] for p in participation_data)
        
        if total_value <= 0:
            return 0.0
        
        # Calculate concentration index
        concentration_index = sum(
            (p['position_value'] / total_value) ** 2 
            for p in participation_data
        )
        
        # Diversification benefit = 1 - concentration (higher is better)
        diversification_benefit = 1.0 - concentration_index
        
        return diversification_benefit