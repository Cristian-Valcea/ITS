# src/risk/sensors/counterparty.py
"""
Counterparty & Settlement Risk Sensors - "Who can fail me?"

These sensors monitor counterparty exposure and settlement risks that could
result in losses even if market positions are profitable.

Sensors:
1. CorrelationAdjustedPFESensor - Potential Future Exposure with correlation
2. HerstattWindowSensor - Settlement risk during time zone gaps
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone

from .base_sensor import BaseSensor, FailureMode, SensorPriority


class CorrelationAdjustedPFESensor(BaseSensor):
    """
    Correlation-Adjusted Potential Future Exposure (PFE) Sensor.
    
    Calculates potential future exposure to counterparties, adjusted for
    correlation between counterparty credit risk and asset values.
    
    Formula: Adj_PFE = PFE * (1 + correlation_factor * stress_multiplier)
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.COUNTERPARTY_SETTLEMENT
    
    def _get_data_requirements(self) -> List[str]:
        return ['counterparty_exposures', 'asset_correlations', 'credit_spreads']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute correlation-adjusted PFE."""
        counterparty_exposures = data.get('counterparty_exposures', {})
        asset_correlations = data.get('asset_correlations', {})
        credit_spreads = data.get('credit_spreads', {})
        
        if not counterparty_exposures:
            return 0.0
        
        total_adjusted_pfe = 0.0
        
        for counterparty, exposure_data in counterparty_exposures.items():
            # Base PFE (potential future exposure)
            base_pfe = exposure_data.get('pfe', 0.0)
            
            if base_pfe <= 0:
                continue
            
            # Get correlation between counterparty and assets
            correlation = asset_correlations.get(counterparty, 0.0)
            
            # Get credit spread (proxy for default probability)
            credit_spread = credit_spreads.get(counterparty, 0.01)  # 1% default
            
            # Calculate stress multiplier based on credit spread
            stress_multiplier = min(2.0, credit_spread * 100)  # Cap at 2x
            
            # Adjust PFE for correlation
            # Positive correlation increases risk (wrong-way risk)
            correlation_adjustment = 1.0 + abs(correlation) * stress_multiplier
            
            adjusted_pfe = base_pfe * correlation_adjustment
            total_adjusted_pfe += adjusted_pfe
        
        return total_adjusted_pfe
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on data quality."""
        base_confidence = super()._compute_confidence(value, data)
        
        counterparty_exposures = data.get('counterparty_exposures', {})
        asset_correlations = data.get('asset_correlations', {})
        credit_spreads = data.get('credit_spreads', {})
        
        # Check data completeness
        total_counterparties = len(counterparty_exposures)
        if total_counterparties == 0:
            return 0.0
        
        correlation_coverage = len([cp for cp in counterparty_exposures.keys() 
                                  if cp in asset_correlations])
        credit_coverage = len([cp for cp in counterparty_exposures.keys() 
                             if cp in credit_spreads])
        
        avg_coverage = (correlation_coverage + credit_coverage) / (2 * total_counterparties)
        coverage_bonus = avg_coverage * 0.3
        
        confidence = base_confidence + coverage_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with counterparty context."""
        return (f"{self.sensor_name}: ${value:,.0f} adjusted PFE "
                f"(threshold: ${threshold:,.0f}) → {action.value}")


class HerstattWindowSensor(BaseSensor):
    """
    Herstatt Window Sensor.
    
    Monitors settlement risk during time zone gaps when one leg of a trade
    has settled but the other hasn't (named after Herstatt Bank failure).
    
    Formula: Herstatt_Risk = unsettled_notional / CLS_cutoff_proximity
    """
    
    def _get_failure_mode(self) -> FailureMode:
        return FailureMode.COUNTERPARTY_SETTLEMENT
    
    def _get_data_requirements(self) -> List[str]:
        return ['unsettled_trades', 'current_time', 'settlement_times']
    
    def _compute_sensor_value(self, data: Dict[str, Any]) -> float:
        """Compute Herstatt window risk."""
        unsettled_trades = data.get('unsettled_trades', [])
        current_time = data.get('current_time', datetime.now(timezone.utc))
        settlement_times = data.get('settlement_times', {})
        
        if not unsettled_trades:
            return 0.0
        
        total_risk = 0.0
        
        for trade in unsettled_trades:
            trade_id = trade.get('trade_id')
            notional = trade.get('notional', 0.0)
            currency_pair = trade.get('currency_pair', 'UNKNOWN')
            settlement_date = trade.get('settlement_date')
            
            if not settlement_date or notional <= 0:
                continue
            
            # Check if trade is in Herstatt window
            risk_factor = self._calculate_herstatt_risk_factor(
                current_time, settlement_date, currency_pair, settlement_times
            )
            
            trade_risk = notional * risk_factor
            total_risk += trade_risk
        
        return total_risk
    
    def _calculate_herstatt_risk_factor(self, current_time: datetime, 
                                      settlement_date: datetime,
                                      currency_pair: str,
                                      settlement_times: Dict[str, Any]) -> float:
        """Calculate risk factor based on settlement timing."""
        # Simplified Herstatt risk calculation
        
        # Get settlement windows for currency pair
        pair_info = settlement_times.get(currency_pair, {})
        
        # Default settlement windows (hours UTC)
        default_windows = {
            'USD': (13, 21),  # New York: 8 AM - 4 PM EST
            'EUR': (7, 15),   # Frankfurt: 8 AM - 4 PM CET  
            'JPY': (0, 8),    # Tokyo: 9 AM - 5 PM JST
            'GBP': (8, 16),   # London: 8 AM - 4 PM GMT
        }
        
        # Extract currencies from pair (e.g., 'EURUSD' -> 'EUR', 'USD')
        if len(currency_pair) >= 6:
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:6]
        else:
            return 0.0  # Unknown format
        
        base_window = pair_info.get(base_currency, default_windows.get(base_currency, (0, 24)))
        quote_window = pair_info.get(quote_currency, default_windows.get(quote_currency, (0, 24)))
        
        # Check if current time is in Herstatt window
        current_hour = current_time.hour
        
        # Risk is highest when one market is closed and other is open
        base_open = base_window[0] <= current_hour <= base_window[1]
        quote_open = quote_window[0] <= current_hour <= quote_window[1]
        
        if base_open and not quote_open:
            return 0.8  # High risk
        elif quote_open and not base_open:
            return 0.8  # High risk
        elif not base_open and not quote_open:
            return 0.3  # Medium risk (both closed)
        else:
            return 0.1  # Low risk (both open)
    
    def _compute_confidence(self, value: float, data: Dict[str, Any]) -> float:
        """Compute confidence based on settlement data quality."""
        base_confidence = super()._compute_confidence(value, data)
        
        unsettled_trades = data.get('unsettled_trades', [])
        settlement_times = data.get('settlement_times', {})
        
        # Higher confidence with more complete trade data
        complete_trades = 0
        for trade in unsettled_trades:
            if all(key in trade for key in ['trade_id', 'notional', 'currency_pair', 'settlement_date']):
                complete_trades += 1
        
        if len(unsettled_trades) > 0:
            completeness_ratio = complete_trades / len(unsettled_trades)
            completeness_bonus = completeness_ratio * 0.2
        else:
            completeness_bonus = 0.0
        
        # Bonus for having settlement time data
        if settlement_times:
            settlement_bonus = 0.1
        else:
            settlement_bonus = 0.0
        
        confidence = base_confidence + completeness_bonus + settlement_bonus
        return max(0.0, min(1.0, confidence))
    
    def _format_message(self, value: float, threshold: float, action) -> str:
        """Format message with settlement context."""
        current_time = datetime.now(timezone.utc)
        return (f"{self.sensor_name}: ${value:,.0f} Herstatt risk "
                f"at {current_time.strftime('%H:%M UTC')} "
                f"(threshold: ${threshold:,.0f}) → {action.value}")