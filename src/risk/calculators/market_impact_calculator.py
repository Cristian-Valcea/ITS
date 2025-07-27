"""
Market Impact Risk Calculator

Monitors market microstructure conditions and triggers risk actions
when liquidity conditions deteriorate.

Features monitored:
- Spread width (basis points)
- Queue imbalance
- Market impact for standard notional
- Kyle's lambda (price impact slope)

Action: THROTTLE when market impact becomes excessive
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_calculator import BaseRiskCalculator
from ..event_types import RiskEvent, RiskEventType


class MarketImpactRiskCalculator(BaseRiskCalculator):
    """
    Market Impact Risk Calculator - Microstructure risk monitoring.
    
    Monitors order book conditions and market impact metrics to detect
    when trading should be throttled due to poor liquidity conditions.
    
    Formula: Multiple thresholds on spread_bps, queue_imbalance, impact_10k
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("market_impact", config)
        
        # Configuration parameters
        self.max_spread_bps = self.config.get('max_spread_bps', 50.0)  # 0.5% max spread
        self.max_impact_threshold = self.config.get('max_impact_threshold', 0.001)  # 0.1% max impact
        self.min_queue_balance = self.config.get('min_queue_balance', -0.8)  # -80% max imbalance
        self.max_queue_balance = self.config.get('max_queue_balance', 0.8)   # +80% max imbalance
        self.kyle_lambda_threshold = self.config.get('kyle_lambda_threshold', 1e-6)  # Kyle's lambda limit
        
        # Validation
        if not 1.0 <= self.max_spread_bps <= 1000.0:
            raise ValueError("max_spread_bps must be between 1 and 1000 bps")
        
        if not 0.0001 <= self.max_impact_threshold <= 0.05:
            raise ValueError("max_impact_threshold must be between 0.01% and 5%")
        
        if not -1.0 <= self.min_queue_balance <= self.max_queue_balance <= 1.0:
            raise ValueError("queue balance thresholds must be between -1 and 1")
    
    def calculate(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate market impact risk metrics.
        
        Args:
            market_data: Dictionary containing market impact features:
                - spread_bps: Bid-ask spread in basis points
                - queue_imbalance: Order book imbalance [-1, 1]
                - impact_10k: Market impact for 10k notional
                - kyle_lambda: Kyle's lambda (optional)
        
        Returns:
            Risk calculation result
        """
        try:
            # Extract market impact features
            spread_bps = market_data.get('spread_bps', 0.0)
            queue_imbalance = market_data.get('queue_imbalance', 0.0)
            impact_10k = market_data.get('impact_10k', 0.0)
            kyle_lambda = market_data.get('kyle_lambda', np.nan)
            
            # Initialize risk flags
            spread_risk = False
            impact_risk = False
            imbalance_risk = False
            kyle_risk = False
            
            # Check spread risk
            if spread_bps > self.max_spread_bps:
                spread_risk = True
            
            # Check market impact risk
            if impact_10k > self.max_impact_threshold:
                impact_risk = True
            
            # Check queue imbalance risk
            if queue_imbalance < self.min_queue_balance or queue_imbalance > self.max_queue_balance:
                imbalance_risk = True
            
            # Check Kyle's lambda risk (if available)
            if not np.isnan(kyle_lambda) and kyle_lambda > self.kyle_lambda_threshold:
                kyle_risk = True
            
            # Determine overall risk level
            risk_flags = [spread_risk, impact_risk, imbalance_risk, kyle_risk]
            num_risks = sum(risk_flags)
            
            if num_risks == 0:
                risk_level = "LOW"
                action = "ALLOW"
            elif num_risks == 1:
                risk_level = "MEDIUM"
                action = "WARN"
            else:
                risk_level = "HIGH"
                action = "THROTTLE"
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(
                spread_bps, queue_imbalance, impact_10k, kyle_lambda
            )
            
            # Create risk message
            risk_details = []
            if spread_risk:
                risk_details.append(f"Wide spread: {spread_bps:.1f} bps > {self.max_spread_bps:.1f}")
            if impact_risk:
                risk_details.append(f"High impact: {impact_10k:.4f} > {self.max_impact_threshold:.4f}")
            if imbalance_risk:
                risk_details.append(f"Queue imbalance: {queue_imbalance:.2f}")
            if kyle_risk:
                risk_details.append(f"High Kyle's lambda: {kyle_lambda:.2e}")
            
            message = f"Market impact risk: {risk_level}"
            if risk_details:
                message += f" ({'; '.join(risk_details)})"
            
            return {
                'calculator': self.name,
                'risk_level': risk_level,
                'action': action,
                'risk_score': risk_score,
                'message': message,
                'values': {
                    'spread_bps': spread_bps,
                    'queue_imbalance': queue_imbalance,
                    'impact_10k': impact_10k,
                    'kyle_lambda': kyle_lambda,
                    'spread_risk': spread_risk,
                    'impact_risk': impact_risk,
                    'imbalance_risk': imbalance_risk,
                    'kyle_risk': kyle_risk,
                    'num_risk_flags': num_risks
                },
                'metadata': {
                    'max_spread_bps': self.max_spread_bps,
                    'max_impact_threshold': self.max_impact_threshold,
                    'queue_balance_range': [self.min_queue_balance, self.max_queue_balance],
                    'kyle_lambda_threshold': self.kyle_lambda_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact risk: {e}")
            return {
                'calculator': self.name,
                'risk_level': 'UNKNOWN',
                'action': 'BLOCK',
                'risk_score': 100.0,
                'message': f"Market impact calculation error: {str(e)}",
                'values': {},
                'metadata': {}
            }
    
    def _calculate_risk_score(
        self, 
        spread_bps: float, 
        queue_imbalance: float, 
        impact_10k: float, 
        kyle_lambda: float
    ) -> float:
        """
        Calculate a composite risk score from 0-100.
        
        Args:
            spread_bps: Spread in basis points
            queue_imbalance: Queue imbalance [-1, 1]
            impact_10k: Market impact for 10k notional
            kyle_lambda: Kyle's lambda
            
        Returns:
            Risk score from 0 (low risk) to 100 (high risk)
        """
        try:
            # Spread component (0-40 points)
            spread_component = min(40.0, (spread_bps / self.max_spread_bps) * 40.0)
            
            # Impact component (0-30 points)
            impact_component = min(30.0, (impact_10k / self.max_impact_threshold) * 30.0)
            
            # Imbalance component (0-20 points)
            abs_imbalance = abs(queue_imbalance)
            max_abs_imbalance = max(abs(self.min_queue_balance), abs(self.max_queue_balance))
            imbalance_component = min(20.0, (abs_imbalance / max_abs_imbalance) * 20.0)
            
            # Kyle's lambda component (0-10 points)
            kyle_component = 0.0
            if not np.isnan(kyle_lambda) and kyle_lambda > 0:
                kyle_component = min(10.0, (kyle_lambda / self.kyle_lambda_threshold) * 10.0)
            
            # Total score
            total_score = spread_component + impact_component + imbalance_component + kyle_component
            
            return min(100.0, total_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk score: {e}")
            return 50.0  # Default moderate risk
    
    def get_required_data_keys(self) -> list:
        """Return list of required data keys for this calculator."""
        return ['spread_bps', 'queue_imbalance', 'impact_10k']
    
    def get_optional_data_keys(self) -> list:
        """Return list of optional data keys for this calculator."""
        return ['kyle_lambda']
    
    def validate_config(self) -> bool:
        """Validate calculator configuration."""
        try:
            # Check required parameters exist
            required_params = ['max_spread_bps', 'max_impact_threshold']
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter ranges
            if not 1.0 <= self.max_spread_bps <= 1000.0:
                self.logger.error("max_spread_bps must be between 1 and 1000 bps")
                return False
            
            if not 0.0001 <= self.max_impact_threshold <= 0.05:
                self.logger.error("max_impact_threshold must be between 0.01% and 5%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this calculator."""
        return {
            'max_spread_bps': {
                'type': 'float',
                'default': 50.0,
                'min': 1.0,
                'max': 1000.0,
                'description': 'Maximum allowed spread in basis points'
            },
            'max_impact_threshold': {
                'type': 'float',
                'default': 0.001,
                'min': 0.0001,
                'max': 0.05,
                'description': 'Maximum allowed market impact (fraction)'
            },
            'min_queue_balance': {
                'type': 'float',
                'default': -0.8,
                'min': -1.0,
                'max': 0.0,
                'description': 'Minimum allowed queue imbalance'
            },
            'max_queue_balance': {
                'type': 'float',
                'default': 0.8,
                'min': 0.0,
                'max': 1.0,
                'description': 'Maximum allowed queue imbalance'
            },
            'kyle_lambda_threshold': {
                'type': 'float',
                'default': 1e-6,
                'min': 1e-9,
                'max': 1e-3,
                'description': 'Maximum allowed Kyle\'s lambda'
            }
        }