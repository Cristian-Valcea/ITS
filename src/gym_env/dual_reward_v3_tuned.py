#!/usr/bin/env python3
"""
🎯 DUAL-TICKER REWARD SYSTEM V3 - TUNED VERSION
Modified V3 reward system with tuned hold-bonus and ticket-cost weights

TUNING OBJECTIVE: Increase trading activity while preserving core performance
- Hold bonus: 0.01 → 0.0005 (20x reduction)
- Ticket cost: $0.50 → $0.20 (60% reduction)

Based on frozen V3 specification but with adjustable trading incentives.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Breakdown of V3 reward components for analysis (SAME AS V3)"""
    risk_free_nav_change: float      # PnL minus cash yield
    embedded_impact: float           # Kyle lambda impact (includes turnover)
    downside_semi_variance: float    # Penalty for negative swings only
    kelly_bonus: float               # Natural log bonus for good returns
    position_decay_penalty: float    # Penalty for holding positions during OFF periods
    turnover_penalty: float          # Turnover penalty with OFF-period kicker
    size_penalty: float              # Soft position-size cap penalty
    hold_bonus: float                # Bonus for doing nothing when α≈0 (TUNED)
    action_change_penalty: float     # Penalty for changing actions frequently
    ticket_cost: float               # Fixed per-trade ticket cost (TUNED)
    total_reward: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'risk_free_nav_change': self.risk_free_nav_change,
            'embedded_impact': self.embedded_impact,
            'downside_semi_variance': self.downside_semi_variance,
            'kelly_bonus': self.kelly_bonus,
            'position_decay_penalty': self.position_decay_penalty,
            'turnover_penalty': self.turnover_penalty,
            'size_penalty': self.size_penalty,
            'hold_bonus': self.hold_bonus,
            'action_change_penalty': self.action_change_penalty,
            'ticket_cost': self.ticket_cost,
            'total_reward': self.total_reward
        }

class DualTickerRewardV3Tuned:
    """
    V3 reward system with tuned weights for increased trading activity
    
    CHANGES FROM V3:
    - hold_bonus_weight: 0.01 → 0.0005 (20x reduction)
    - ticket_cost_per_trade: 0.50 → 0.20 (60% reduction)
    
    All other components remain identical to preserve core performance.
    """
    
    def __init__(
        self,
        # Core V3 parameters (UNCHANGED)
        base_impact_bp: float = 68.0,
        impact_exponent: float = 0.5,
        risk_free_rate_annual: float = 0.05,
        
        # Component weights (MOSTLY UNCHANGED)
        downside_penalty_weight: float = 2.0,
        kelly_bonus_weight: float = 0.5,
        position_decay_weight: float = 0.1,
        turnover_penalty_weight: float = 0.05,
        size_penalty_weight: float = 0.02,
        action_change_penalty_weight: float = 0.005,
        
        # 🎯 TUNED WEIGHTS FOR INCREASED TRADING
        hold_bonus_weight: float = 0.0005,  # REDUCED from 0.01
        ticket_cost_per_trade: float = 0.20  # REDUCED from 0.50
    ):
        """
        Initialize V3 tuned reward system
        
        Args:
            Core parameters unchanged from V3
            hold_bonus_weight: TUNED - reduced to discourage excessive holding
            ticket_cost_per_trade: TUNED - reduced to make trading cheaper
        """
        # Store configuration
        self.version = "3.0.0-tuned"
        self.tuning_date = "2025-08-02"
        self.base_version = "v3_gold_standard_400k_20250802_202736"
        
        # Core V3 parameters (UNCHANGED)
        self.base_impact_bp = base_impact_bp
        self.impact_exponent = impact_exponent
        self.risk_free_rate_annual = risk_free_rate_annual
        
        # Component weights (MOSTLY UNCHANGED)
        self.downside_penalty_weight = downside_penalty_weight
        self.kelly_bonus_weight = kelly_bonus_weight
        self.position_decay_weight = position_decay_weight
        self.turnover_penalty_weight = turnover_penalty_weight
        self.size_penalty_weight = size_penalty_weight
        self.action_change_penalty_weight = action_change_penalty_weight
        
        # 🎯 TUNED WEIGHTS
        self.hold_bonus_weight = hold_bonus_weight
        self.ticket_cost_per_trade = ticket_cost_per_trade
        
        # Convert annual risk-free rate to per-minute
        self.risk_free_rate_per_minute = self.risk_free_rate_annual / (252 * 6.5 * 60)
        
        logger.info(f"🎯 V3 Tuned Reward System initialized - v{self.version}")
        logger.info(f"   Hold bonus: {self.hold_bonus_weight} (reduced from 0.01)")
        logger.info(f"   Ticket cost: ${self.ticket_cost_per_trade} (reduced from $0.50)")
        logger.info(f"   Base version: {self.base_version}")
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        nvda_position: int,
        msft_position: int,
        nvda_trade: int,
        msft_trade: int,
        nvda_price: float,
        msft_price: float,
        nvda_alpha: float,
        msft_alpha: float,
        action: int,
        previous_action: int
    ) -> RewardComponents:
        """
        Calculate V3 tuned reward with modified trading incentives
        
        LOGIC: Identical to V3 except for hold_bonus and ticket_cost weights
        
        Args:
            Same as V3 reward system
            
        Returns:
            RewardComponents with tuned hold_bonus and ticket_cost
        """
        
        # 1. Risk-free NAV change (UNCHANGED)
        raw_nav_change = portfolio_value - previous_portfolio_value
        risk_free_yield = previous_portfolio_value * self.risk_free_rate_per_minute
        risk_free_nav_change = raw_nav_change - risk_free_yield
        
        # 2. Embedded impact cost using Kyle lambda model (UNCHANGED)
        nvda_notional = abs(nvda_trade) * nvda_price
        msft_notional = abs(msft_trade) * msft_price
        total_turnover = nvda_notional + msft_notional
        
        if total_turnover > 0:
            impact_factor = self.base_impact_bp * (total_turnover / 100000) ** self.impact_exponent
            embedded_impact = -(impact_factor / 10000) * total_turnover
        else:
            embedded_impact = 0.0
        
        # 3. Downside semi-variance penalty (UNCHANGED)
        if risk_free_nav_change < 0:
            downside_semi_variance = -self.downside_penalty_weight * (risk_free_nav_change ** 2)
        else:
            downside_semi_variance = 0.0
        
        # 4. Kelly bonus for positive returns (UNCHANGED)
        if risk_free_nav_change > 0:
            kelly_bonus = self.kelly_bonus_weight * np.log(1 + risk_free_nav_change / previous_portfolio_value)
        else:
            kelly_bonus = 0.0
        
        # 5. Position decay penalty during low alpha periods (UNCHANGED)
        avg_alpha = abs(nvda_alpha) + abs(msft_alpha)
        if avg_alpha < 0.1:
            total_position_notional = abs(nvda_position) * nvda_price + abs(msft_position) * msft_price
            position_decay_penalty = -self.position_decay_weight * (total_position_notional / 100000)
        else:
            position_decay_penalty = 0.0
        
        # 6. Turnover penalty with alpha adjustment (UNCHANGED)
        if avg_alpha < 0.1:
            turnover_penalty = -self.turnover_penalty_weight * 2.0 * total_turnover / 100000
        else:
            turnover_penalty = -self.turnover_penalty_weight * total_turnover / 100000
        
        # 7. Soft position size penalty (UNCHANGED)
        max_position_value = 500 * max(nvda_price, msft_price)
        nvda_position_value = abs(nvda_position) * nvda_price
        msft_position_value = abs(msft_position) * msft_price
        
        nvda_size_penalty = 0.0
        msft_size_penalty = 0.0
        
        if nvda_position_value > max_position_value:
            nvda_size_penalty = -self.size_penalty_weight * (nvda_position_value - max_position_value) / 100000
        
        if msft_position_value > max_position_value:
            msft_size_penalty = -self.size_penalty_weight * (msft_position_value - max_position_value) / 100000
        
        size_penalty = nvda_size_penalty + msft_size_penalty
        
        # 8. Hold bonus when alpha is near zero (🎯 TUNED - REDUCED WEIGHT)
        if action == 4 and avg_alpha < 0.05:  # Action 4 = Hold, Hold
            hold_bonus = self.hold_bonus_weight  # REDUCED from 0.01 to 0.0005
        else:
            hold_bonus = 0.0
        
        # 9. Action change penalty (UNCHANGED)
        if action != previous_action:
            action_change_penalty = -self.action_change_penalty_weight
        else:
            action_change_penalty = 0.0
        
        # 10. Ticket cost for trades (🎯 TUNED - REDUCED COST)
        num_trades = (1 if nvda_trade != 0 else 0) + (1 if msft_trade != 0 else 0)
        ticket_cost = -num_trades * self.ticket_cost_per_trade / 100000  # REDUCED cost
        
        # Total reward (SAME FORMULA)
        total_reward = (
            risk_free_nav_change +
            embedded_impact +
            downside_semi_variance +
            kelly_bonus +
            position_decay_penalty +
            turnover_penalty +
            size_penalty +
            hold_bonus +
            action_change_penalty +
            ticket_cost
        )
        
        # Return detailed breakdown
        return RewardComponents(
            risk_free_nav_change=risk_free_nav_change,
            embedded_impact=embedded_impact,
            downside_semi_variance=downside_semi_variance,
            kelly_bonus=kelly_bonus,
            position_decay_penalty=position_decay_penalty,
            turnover_penalty=turnover_penalty,
            size_penalty=size_penalty,
            hold_bonus=hold_bonus,
            action_change_penalty=action_change_penalty,
            ticket_cost=ticket_cost,
            total_reward=total_reward
        )
    
    def get_tuning_info(self) -> Dict[str, Any]:
        """
        Get tuning information and comparison with V3
        
        Returns:
            Dictionary with tuning details
        """
        return {
            'version': self.version,
            'tuning_date': self.tuning_date,
            'base_version': self.base_version,
            'tuning_changes': {
                'hold_bonus_weight': {
                    'original': 0.01,
                    'tuned': self.hold_bonus_weight,
                    'change_factor': 0.01 / self.hold_bonus_weight,
                    'objective': 'Reduce holding incentive'
                },
                'ticket_cost_per_trade': {
                    'original': 0.50,
                    'tuned': self.ticket_cost_per_trade,
                    'change_factor': 0.50 / self.ticket_cost_per_trade,
                    'objective': 'Make trading cheaper'
                }
            },
            'unchanged_components': [
                'base_impact_bp', 'impact_exponent', 'risk_free_rate_annual',
                'downside_penalty_weight', 'kelly_bonus_weight', 
                'position_decay_weight', 'turnover_penalty_weight',
                'size_penalty_weight', 'action_change_penalty_weight'
            ],
            'expected_outcomes': {
                'increased_trading_frequency': True,
                'reduced_holding_percentage': True,
                'maintained_profitability': True,
                'acceptable_sharpe_degradation': True
            }
        }

# Tuning constants
TUNING_CONSTANTS = {
    'ORIGINAL_HOLD_BONUS': 0.01,
    'TUNED_HOLD_BONUS': 0.0005,
    'HOLD_BONUS_REDUCTION_FACTOR': 20.0,
    
    'ORIGINAL_TICKET_COST': 0.50,
    'TUNED_TICKET_COST': 0.20,
    'TICKET_COST_REDUCTION_FACTOR': 2.5,
    
    'TARGET_TRADES_PER_EPISODE': 25,
    'TARGET_HOLDING_PERCENTAGE': 60,
    'ACCEPTABLE_SHARPE_RANGE': [0.3, 1.2]
}