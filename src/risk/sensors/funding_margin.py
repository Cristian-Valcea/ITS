# src/risk/sensors/funding_margin.py
"""
Funding & Margin Risk Sensors - "When does margin call knock?"

These sensors monitor funding and margin requirements to prevent margin calls
and liquidity crunches.

Sensors:
1. TimeToMarginExhaustionSensor - Time until margin call under stress
2. LiquidityAtRiskSensor - Cash needed for initial + variation margin
3. HaircutSensitivitySensor - Impact of changing haircuts on margin
"""

import numpy as np
from typing import Dict, Any, List

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class TimeToMarginExhaustionSensor(BaseSensor):
    """
    Time-to-Margin Exhaustion Sensor.
    
    Calculates how long until margin exhaustion under adverse scenarios.
    Uses 3-sigma adverse moves and current bleed rate.
    
    Formula: TTM = (current_equity - maintenance_margin) / daily_bleed_rate
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.FUNDING_MARGIN
    
    def _get_data_requirements(self) -> List[str]:
        return ['current_equity', 'maintenance_margin', 'daily_pnl_history', 'positions']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute time to margin exhaustion in days."""
        current_equity = float(data.get('current_equity', 0))
        maintenance_margin = float(data.get('maintenance_margin', 0))
        daily_pnl_history = np.array(data.get('daily_pnl_history', []))
        
        if current_equity <= maintenance_margin:
            return 0.0  # Already at margin call
        
        if len(daily_pnl_history) < 5:
            return float('inf')  # Insufficient data
        
        # Calculate current margin buffer
        margin_buffer = current_equity - maintenance_margin
        
        # Calculate daily bleed rate (negative PnL trend)
        recent_pnl = daily_pnl_history[-20:] if len(daily_pnl_history) >= 20 else daily_pnl_history
        
        # Use 3-sigma adverse scenario
        mean_daily_pnl = np.mean(recent_pnl)
        std_daily_pnl = np.std(recent_pnl)
        
        # 3-sigma adverse daily PnL
        adverse_daily_pnl = mean_daily_pnl - 3 * std_daily_pnl
        
        if adverse_daily_pnl >= 0:
            return float('inf')  # No adverse scenario
        
        # Time to margin exhaustion
        time_to_exhaustion = margin_buffer / abs(adverse_daily_pnl)
        
        return max(0.0, time_to_exhaustion)
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality and stability."""
        base_confidence = super()._compute_confidence(value, data)
        
        daily_pnl_history = np.array(data.get('daily_pnl_history', []))
        
        # Higher confidence with more PnL history
        if len(daily_pnl_history) >= 30:
            history_bonus = 0.2
        elif len(daily_pnl_history) >= 10:
            history_bonus = 0.1
        else:
            history_bonus = 0.0
        
        # Check PnL stability (lower volatility = higher confidence)
        if len(daily_pnl_history) >= 10:
            pnl_volatility = np.std(daily_pnl_history)
            mean_abs_pnl = np.mean(np.abs(daily_pnl_history))
            
            if mean_abs_pnl > 0:
                volatility_ratio = pnl_volatility / mean_abs_pnl
                if volatility_ratio < 1.0:
                    stability_bonus = 0.1
                else:
                    stability_bonus = 0.0
            else:
                stability_bonus = 0.0
        else:
            stability_bonus = 0.0
        
        confidence = base_confidence + history_bonus + stability_bonus
        return max(0.0, min(1.0, confidence))


class LiquidityAtRiskSensor(BaseSensor):
    """
    Liquidity-at-Risk (LaR) Sensor.
    
    Calculates cash needed to meet initial and variation margin requirements
    under stress scenarios.
    
    Formula: LaR = max(initial_margin_requirement, variation_margin_stress) - available_cash
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.FUNDING_MARGIN
    
    def _get_data_requirements(self) -> List[str]:
        return ['positions', 'available_cash', 'margin_requirements', 'portfolio_volatility']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute liquidity at risk."""
        available_cash = float(data.get('available_cash', 0))
        margin_requirements = data.get('margin_requirements', {})
        portfolio_volatility = float(data.get('portfolio_volatility', 0.02))
        positions = data.get('positions', {})
        
        # Calculate initial margin requirement
        initial_margin = 0.0
        for symbol, position_size in positions.items():
            symbol_margin = margin_requirements.get(symbol, {})
            margin_rate = symbol_margin.get('initial_margin_rate', 0.1)  # 10% default
            position_value = abs(position_size) * symbol_margin.get('price', 100)  # Default price
            initial_margin += position_value * margin_rate
        
        # Calculate variation margin under stress (3-sigma move)
        portfolio_value = sum(
            abs(size) * margin_requirements.get(symbol, {}).get('price', 100)
            for symbol, size in positions.items()
        )
        
        stress_move = 3 * portfolio_volatility  # 3-sigma daily move
        variation_margin_stress = portfolio_value * stress_move
        
        # Total margin requirement under stress
        total_margin_requirement = max(initial_margin, variation_margin_stress)
        
        # Liquidity at risk
        liquidity_at_risk = max(0.0, total_margin_requirement - available_cash)
        
        return liquidity_at_risk
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data completeness."""
        base_confidence = super()._compute_confidence(value, data)
        
        margin_requirements = data.get('margin_requirements', {})
        positions = data.get('positions', {})
        
        # Check data completeness
        covered_positions = 0
        for symbol in positions.keys():
            if symbol in margin_requirements:
                covered_positions += 1
        
        if len(positions) > 0:
            coverage_ratio = covered_positions / len(positions)
            coverage_bonus = coverage_ratio * 0.2
        else:
            coverage_bonus = 0.0
        
        # Check if we have portfolio volatility data
        portfolio_volatility = data.get('portfolio_volatility')
        if portfolio_volatility is not None and portfolio_volatility > 0:
            volatility_bonus = 0.1
        else:
            volatility_bonus = 0.0
        
        confidence = base_confidence + coverage_bonus + volatility_bonus
        return max(0.0, min(1.0, confidence))


class HaircutSensitivitySensor(BaseSensor):
    """
    Haircut Sensitivity Sensor.
    
    Measures sensitivity to changes in repo haircuts, which can suddenly
    increase margin requirements during market stress.
    
    Formula: Haircut_Sensitivity = Σ(position_value_i * Δhaircut_i)
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.FUNDING_MARGIN
    
    def _get_data_requirements(self) -> List[str]:
        return ['positions', 'current_haircuts', 'stressed_haircuts', 'position_values']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute haircut sensitivity."""
        positions = data.get('positions', {})
        current_haircuts = data.get('current_haircuts', {})
        stressed_haircuts = data.get('stressed_haircuts', {})
        position_values = data.get('position_values', {})
        
        total_sensitivity = 0.0
        
        for symbol, position_size in positions.items():
            # Get position value
            if symbol in position_values:
                position_value = position_values[symbol]
            else:
                # Estimate position value (fallback)
                position_value = abs(position_size) * 100  # Assume $100 per share
            
            # Get haircut change
            current_haircut = current_haircuts.get(symbol, 0.05)  # 5% default
            stressed_haircut = stressed_haircuts.get(symbol, current_haircut * 2)  # Double in stress
            
            haircut_change = stressed_haircut - current_haircut
            
            # Calculate sensitivity for this position
            position_sensitivity = position_value * haircut_change
            total_sensitivity += position_sensitivity
        
        return total_sensitivity
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on haircut data availability."""
        base_confidence = super()._compute_confidence(value, data)
        
        positions = data.get('positions', {})
        current_haircuts = data.get('current_haircuts', {})
        stressed_haircuts = data.get('stressed_haircuts', {})
        
        # Check haircut data coverage
        current_coverage = len([s for s in positions.keys() if s in current_haircuts])
        stressed_coverage = len([s for s in positions.keys() if s in stressed_haircuts])
        
        if len(positions) > 0:
            avg_coverage = (current_coverage + stressed_coverage) / (2 * len(positions))
            coverage_bonus = avg_coverage * 0.3
        else:
            coverage_bonus = 0.0
        
        confidence = base_confidence + coverage_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with haircut context."""
        return (f"{self.sensor_name}: ${value:,.0f} sensitivity "
                f"(threshold: ${threshold:,.0f}) → {action.value}")