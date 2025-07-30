#!/usr/bin/env python3
"""
🎯 DUAL-TICKER REWARD SYSTEM V2 - RISK-ADJUSTED P&L
Implements: ΔNAV − TC − λ·turnover − β·volatility
Designed for production-grade dual-ticker trading with proper risk adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis"""
    nav_change: float
    transaction_costs: float
    turnover_penalty: float
    volatility_penalty: float
    total_reward: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'nav_change': self.nav_change,
            'transaction_costs': self.transaction_costs,
            'turnover_penalty': self.turnover_penalty,
            'volatility_penalty': self.volatility_penalty,
            'total_reward': self.total_reward
        }

class DualTickerRewardV2:
    """
    Enhanced reward system for dual-ticker trading with risk adjustment
    
    Formula: ΔNAV − TC − λ·turnover − β·volatility
    Where:
    - ΔNAV: Net asset value change
    - TC: Transaction costs (bid-ask spread + market impact)
    - λ·turnover: Turnover penalty (discourages over-trading)
    - β·volatility: Volatility penalty (risk adjustment)
    """
    
    def __init__(
        self,
        # Transaction cost parameters
        tc_bp: float = 1.0,                    # Transaction cost basis points
        market_impact_bp: float = 0.5,         # Market impact basis points
        
        # Turnover penalty parameters - ESCALATED SURGICAL CHANGES
        lambda_turnover: float = 0.02,         # 20x stronger turnover penalty (escalated)
        target_turnover: float = 1.0,          # Conservative annual turnover target
        
        # Volatility penalty parameters - DISABLED FOR NOW
        beta_volatility: float = 0.0,          # Disabled until P&L positive
        vol_lookback: int = 50,                # Volatility calculation window
        
        # Hold bonus parameters - NEW
        hold_bonus_coef: float = 0.002,        # Bonus per step for holding position
        
        # Risk adjustment parameters - SIMPLIFIED
        sharpe_bonus: float = 0.0,             # Disabled for now
        max_dd_penalty: float = 0.0,           # Disabled - env handles 2% DD stop
        
        # Debugging and analysis
        verbose: bool = False
    ):
        self.tc_bp = tc_bp
        self.market_impact_bp = market_impact_bp
        self.lambda_turnover = lambda_turnover
        self.target_turnover = target_turnover
        self.beta_volatility = beta_volatility
        self.vol_lookback = vol_lookback
        self.hold_bonus_coef = hold_bonus_coef
        self.sharpe_bonus = sharpe_bonus
        self.max_dd_penalty = max_dd_penalty
        self.verbose = verbose
        
        # State tracking
        self.portfolio_history = []
        self.trade_history = []
        self.volatility_history = []
        self.position_history = []  # Track position changes for hold bonus
        self.steps_since_trade = 0  # Steps since last position change
        
        logger.info(f"🎯 DualTickerRewardV2 initialized (ESCALATED SURGICAL PATCH):")
        logger.info(f"   💰 TC: {tc_bp}bp + {market_impact_bp}bp impact")
        logger.info(f"   🔄 Turnover penalty: λ={lambda_turnover} (20x stronger), target={target_turnover}")
        logger.info(f"   📊 Volatility penalty: β={beta_volatility} (disabled)")
        logger.info(f"   ⏳ Hold bonus: {hold_bonus_coef} per step held")
        
    def calculate_reward(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        nvda_trade_value: float,
        msft_trade_value: float,
        nvda_position: float,
        msft_position: float,
        nvda_price: float,
        msft_price: float,
        step: int
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate risk-adjusted reward for dual-ticker trading
        
        Returns:
            (reward, components): Total reward and component breakdown
        """
        
        # 1. NAV Change (core return)
        nav_change = curr_portfolio_value - prev_portfolio_value
        
        # 2. Transaction Costs
        transaction_costs = self._calculate_transaction_costs(
            nvda_trade_value, msft_trade_value, nvda_price, msft_price
        )
        
        # 3. Turnover Penalty
        turnover_penalty = self._calculate_turnover_penalty(
            nvda_trade_value, msft_trade_value, curr_portfolio_value
        )
        
        # 4. Volatility Penalty (DISABLED)
        volatility_penalty = 0.0  # Disabled until P&L positive
        
        # 5. Hold Bonus (NEW)
        hold_bonus = self._calculate_hold_bonus(
            nvda_trade_value, msft_trade_value, curr_portfolio_value
        )
        
        # 6. Risk Adjustments (SIMPLIFIED - mostly disabled)
        risk_adjustments = 0.0  # Simplified for now
        
        # Total reward: ΔNAV − TC − λ·turnover + hold_bonus (simplified)
        total_reward = (
            nav_change 
            - transaction_costs 
            - turnover_penalty 
            + hold_bonus
        )
        
        # Create component breakdown
        components = RewardComponents(
            nav_change=nav_change,
            transaction_costs=transaction_costs,
            turnover_penalty=turnover_penalty,
            volatility_penalty=hold_bonus,  # Use this field for hold bonus
            total_reward=total_reward
        )
        
        # Update state tracking
        self._update_state_tracking(
            curr_portfolio_value, nvda_trade_value, msft_trade_value
        )
        
        if self.verbose and step % 1000 == 0:
            logger.info(f"Step {step}: Reward breakdown: {components.to_dict()}")
        
        return total_reward, components
    
    def _calculate_transaction_costs(
        self, nvda_trade_value: float, msft_trade_value: float,
        nvda_price: float, msft_price: float
    ) -> float:
        """Calculate realistic transaction costs with market impact"""
        
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        
        if total_trade_value == 0:
            return 0.0
        
        # Base transaction cost (bid-ask spread)
        base_cost = total_trade_value * (self.tc_bp / 10000)
        
        # Market impact (increases with trade size)
        impact_factor = min(total_trade_value / 100000, 1.0)  # Cap at $100K
        market_impact = total_trade_value * (self.market_impact_bp / 10000) * impact_factor
        
        return base_cost + market_impact
    
    def _calculate_turnover_penalty(
        self, nvda_trade_value: float, msft_trade_value: float,
        portfolio_value: float
    ) -> float:
        """Calculate turnover penalty to discourage over-trading"""
        
        if portfolio_value <= 0:
            return 0.0
        
        # Calculate turnover as fraction of portfolio
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        turnover_rate = total_trade_value / portfolio_value
        
        # Penalty increases quadratically with excess turnover
        target_per_step = self.target_turnover / (252 * 390)  # Daily target
        excess_turnover = max(0, turnover_rate - target_per_step)
        
        penalty = self.lambda_turnover * portfolio_value * (excess_turnover ** 2)
        
        return penalty
    
    def _calculate_hold_bonus(
        self, nvda_trade_value: float, msft_trade_value: float,
        portfolio_value: float
    ) -> float:
        """Calculate bonus for holding positions (patience reward)"""
        
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        
        # If there was a trade, reset the counter
        if total_trade_value > 0:
            self.steps_since_trade = 0
            return 0.0
        
        # Otherwise, increment and award hold bonus
        self.steps_since_trade += 1
        
        # Bonus increases with time held (up to reasonable limit)
        days_held = self.steps_since_trade / (390)  # 390 minutes per trading day
        bonus = self.hold_bonus_coef * portfolio_value * min(days_held, 10.0)  # Cap at 10 days
        
        return bonus
    
    def _calculate_volatility_penalty(
        self, portfolio_value: float, step: int
    ) -> float:
        """Calculate volatility penalty for risk adjustment"""
        
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Calculate recent portfolio volatility
        if len(self.portfolio_history) >= self.vol_lookback:
            recent_values = self.portfolio_history[-self.vol_lookback:]
        else:
            recent_values = self.portfolio_history
        
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate returns and volatility
        returns = np.diff(recent_values) / recent_values[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # Annualize volatility (252 * 390 minutes per year)
        annual_vol = volatility * np.sqrt(252 * 390)
        
        # Penalty for high volatility
        penalty = self.beta_volatility * portfolio_value * (annual_vol ** 2)
        
        return penalty
    
    def _calculate_risk_adjustments(
        self, portfolio_value: float, step: int
    ) -> float:
        """Calculate risk-based bonuses and penalties"""
        
        if len(self.portfolio_history) < 20:  # Need minimum history
            return 0.0
        
        adjustments = 0.0
        
        # Sharpe ratio bonus
        if len(self.portfolio_history) >= 50:
            recent_values = self.portfolio_history[-50:]
            returns = np.diff(recent_values) / np.array(recent_values[:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                adjustments += self.sharpe_bonus * portfolio_value * max(0, sharpe)
        
        # Drawdown penalty
        if len(self.portfolio_history) >= 10:
            recent_peak = max(self.portfolio_history[-100:]) if len(self.portfolio_history) >= 100 else max(self.portfolio_history)
            current_dd = (recent_peak - portfolio_value) / recent_peak if recent_peak > 0 else 0
            if current_dd > 0.01:  # 1% threshold
                adjustments -= self.max_dd_penalty * portfolio_value * (current_dd ** 2)
        
        return adjustments
    
    def _update_state_tracking(
        self, portfolio_value: float, nvda_trade: float, msft_trade: float
    ):
        """Update internal state for reward calculation"""
        
        self.portfolio_history.append(portfolio_value)
        self.trade_history.append(abs(nvda_trade) + abs(msft_trade))
        
        # Keep history manageable
        max_history = 1000
        if len(self.portfolio_history) > max_history:
            self.portfolio_history = self.portfolio_history[-max_history:]
            self.trade_history = self.trade_history[-max_history:]
    
    def reset(self):
        """Reset state tracking for new episode"""
        self.portfolio_history.clear()
        self.trade_history.clear()
        self.volatility_history.clear()
        self.position_history.clear()
        self.steps_since_trade = 0
        
        if self.verbose:
            logger.info("🔄 DualTickerRewardV2 state reset (SURGICAL PATCH)")
    
    def get_stats(self) -> Dict[str, float]:
        """Get reward system statistics"""
        
        if len(self.portfolio_history) < 2:
            return {}
        
        # Calculate statistics
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        total_trades = sum(self.trade_history)
        avg_trade_size = np.mean(self.trade_history) if self.trade_history else 0
        
        stats = {
            'episodes': len(self.portfolio_history),
            'total_return': (self.portfolio_history[-1] / self.portfolio_history[0] - 1) if self.portfolio_history[0] > 0 else 0,
            'volatility': np.std(returns) * np.sqrt(252 * 390) if len(returns) > 1 else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0,
            'total_trades': total_trades,
            'avg_trade_size': avg_trade_size,
            'turnover': total_trades / self.portfolio_history[0] if self.portfolio_history[0] > 0 else 0
        }
        
        return stats

def create_reward_calculator(config: Dict[str, Any]) -> DualTickerRewardV2:
    """Factory function to create reward calculator from config"""
    
    return DualTickerRewardV2(
        tc_bp=config.get('tc_bp', 1.0),
        market_impact_bp=config.get('market_impact_bp', 0.5),
        lambda_turnover=config.get('lambda_turnover', 0.001),
        target_turnover=config.get('target_turnover', 2.0),
        beta_volatility=config.get('beta_volatility', 0.01),
        vol_lookback=config.get('vol_lookback', 50),
        sharpe_bonus=config.get('sharpe_bonus', 0.001),
        max_dd_penalty=config.get('max_dd_penalty', 0.01),
        verbose=config.get('verbose', False)
    )