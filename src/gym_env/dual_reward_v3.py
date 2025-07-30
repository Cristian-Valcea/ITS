#!/usr/bin/env python3
"""
🎯 DUAL-TICKER REWARD SYSTEM V3 - STRUCTURAL REDESIGN
Makes the cheapest strategy to stay flat unless genuine alpha is sensed
Formula: risk-free ΔNAV - embedded_impact - downside_semi_variance + kelly_bonus
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Breakdown of V3 reward components for analysis"""
    risk_free_nav_change: float      # PnL minus cash yield
    embedded_impact: float           # Kyle lambda impact (includes turnover)
    downside_semi_variance: float    # Penalty for negative swings only
    kelly_bonus: float               # Natural log bonus for good returns
    position_decay_penalty: float    # Penalty for holding positions during OFF periods
    turnover_penalty: float          # Turnover penalty with OFF-period kicker
    size_penalty: float              # Soft position-size cap penalty
    hold_bonus: float                # Bonus for doing nothing when α≈0
    action_change_penalty: float     # Penalty for changing actions frequently
    ticket_cost: float               # Fixed per-trade ticket cost
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

class DualTickerRewardV3:
    """
    V3 reward system with structural redesign to prevent cost-blind trading
    
    Core Philosophy: Make doing nothing the cheapest strategy unless there's genuine alpha
    
    Formula: risk_free_ΔNAV - embedded_impact - downside_penalty + kelly_bonus
    """
    
    def __init__(
        self,
        # Risk-free baseline
        risk_free_rate_annual: float = 0.05,   # 5% annual risk-free rate
        
        # Impact model parameters (Kyle lambda embedded)
        base_impact_bp: float = 68.0,          # Calibrated for alpha signal extraction
        impact_exponent: float = 0.5,          # sqrt scaling for impact
        adv_scaling: float = 40000000.0,       # NVDA ADV ~40M shares (~$6.8B)
        
        # Turnover penalty parameters
        lambda_turnover_on: float = 0.0030,    # Further increased base turnover penalty during ON periods
        lambda_turnover_off_multiplier: float = 4.0,  # Further increased kicker multiplier for OFF periods
        
        # Risk penalty parameters
        downside_penalty_coef: float = 0.01,   # Penalty coefficient for negative swings
        downside_lookback: int = 50,           # Lookback for downside calculation
        
        # Kelly bonus parameters
        kelly_bonus_coef: float = 0.001,       # Kelly log bonus coefficient
        kelly_floor: float = 0.001,            # Floor to prevent log(0)
        
        # Position decay parameters (for piecewise curriculum)
        position_decay_penalty: float = 0.50,  # Increased penalty for holding during OFF periods
        
        # Position sizing parameters
        max_notional_ratio: float = 0.30,      # 30% NAV soft position limit
        size_penalty_kappa: float = 2.0e-4,   # Quadratic penalty coefficient for excess size
        
        # Turnover reduction parameters (reviewer recommendations - ENHANCED)
        hold_bonus_coef: float = 2e-4,         # Increased bonus for doing nothing when α≈0
        action_change_cost_bp: float = 12.5,   # Increased action change cost (was 7.5bp)
        ticket_cost_usd: float = 25.0,         # Increased fixed ticket cost per trade (was $15)
        
        # Time step parameters
        step_minutes: float = 1.0,             # Minutes per step (for risk-free scaling)
        
        # Debugging
        verbose: bool = False
    ):
        self.risk_free_rate_annual = risk_free_rate_annual
        self.base_impact_bp = base_impact_bp
        self.impact_exponent = impact_exponent
        self.adv_scaling = adv_scaling
        self.lambda_turnover_on = lambda_turnover_on
        self.lambda_turnover_off_multiplier = lambda_turnover_off_multiplier
        self.downside_penalty_coef = downside_penalty_coef
        self.downside_lookback = downside_lookback
        self.kelly_bonus_coef = kelly_bonus_coef
        self.kelly_floor = kelly_floor
        self.position_decay_penalty = position_decay_penalty
        self.max_notional_ratio = max_notional_ratio
        self.size_penalty_kappa = size_penalty_kappa
        self.hold_bonus_coef = hold_bonus_coef
        self.action_change_cost_bp = action_change_cost_bp
        self.ticket_cost_usd = ticket_cost_usd
        self.step_minutes = step_minutes
        self.verbose = verbose
        
        # State tracking
        self.portfolio_history = []
        self.trade_history = []
        self.return_history = []
        self.prev_action = None  # Track previous action for action-change penalty
        
        # Calculate per-step risk-free rate
        minutes_per_year = 525600  # 365.25 * 24 * 60
        self.risk_free_rate_per_step = risk_free_rate_annual * step_minutes / minutes_per_year
        
        logger.info(f"🎯 DualTickerRewardV3 initialized (STRUCTURAL REDESIGN):")
        logger.info(f"   📊 Risk-free rate: {risk_free_rate_annual:.1%} annual ({self.risk_free_rate_per_step*100:.6f}% per {step_minutes}min)")
        logger.info(f"   💥 Impact model: {base_impact_bp}bp base × |ΔQ|^{impact_exponent}, ADV={adv_scaling/1e6:.0f}M shares")
        logger.info(f"   📉 Downside penalty: {downside_penalty_coef} × semi-variance")
        logger.info(f"   🎲 Kelly bonus: {kelly_bonus_coef} × log(1 + return)")
        logger.info(f"   🎯 Philosophy: Cheapest strategy = do nothing unless genuine alpha")
    
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
        step: int,
        action: Optional[int] = None,  # Current action for change penalty
        features: Optional[np.ndarray] = None  # Add features to detect OFF periods
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate V3 reward with structural redesign
        
        Returns:
            (reward, components): Total reward and component breakdown
        """
        
        # 1. Risk-Free Adjusted NAV Change
        raw_nav_change = curr_portfolio_value - prev_portfolio_value
        risk_free_yield = prev_portfolio_value * self.risk_free_rate_per_step
        risk_free_nav_change = raw_nav_change - risk_free_yield
        
        # 2. Embedded Impact (Kyle lambda style, includes all trading costs)
        embedded_impact = self._calculate_embedded_impact(
            nvda_trade_value, msft_trade_value, nvda_price, msft_price
        )
        
        # 3. Downside Semi-Variance Penalty
        downside_penalty = self._calculate_downside_penalty(
            curr_portfolio_value, prev_portfolio_value
        )
        
        # 4. Kelly Bonus (natural log scaling)
        kelly_bonus = self._calculate_kelly_bonus(
            curr_portfolio_value, prev_portfolio_value
        )
        
        # 5. Position Decay Penalty (for piecewise curriculum OFF periods)
        position_decay_penalty = self._calculate_position_decay_penalty(
            nvda_position, msft_position, features
        )
        
        # 6. Turnover Penalty with OFF-period kicker
        turnover_penalty = self._calculate_turnover_penalty(
            nvda_trade_value, msft_trade_value, curr_portfolio_value, features
        )
        
        # 7. Soft Position-Size Cap (30% NAV quadratic penalty)
        size_penalty = self._calculate_size_penalty(
            nvda_position, msft_position, curr_portfolio_value
        )
        
        # 8. HOLD Bonus (reviewer recommendation: make doing nothing better)
        hold_bonus = self._calculate_hold_bonus(
            nvda_trade_value, msft_trade_value, curr_portfolio_value, features
        )
        
        # 9. Action Change Penalty (reviewer recommendation: penalize frequent changes)
        action_change_penalty = self._calculate_action_change_penalty(
            nvda_trade_value, msft_trade_value, nvda_price, msft_price, action
        )
        
        # 10. Fixed Ticket Cost (reviewer recommendation: flat fee per trade)
        ticket_cost = self._calculate_ticket_cost(
            nvda_trade_value, msft_trade_value, curr_portfolio_value
        )
        
        # Total V3 reward (enhanced with reviewer recommendations)
        total_reward = (
            risk_free_nav_change 
            - embedded_impact 
            - downside_penalty 
            + kelly_bonus
            - position_decay_penalty
            - turnover_penalty
            - size_penalty
            + hold_bonus              # Make doing nothing better
            - action_change_penalty   # Penalize frequent action changes
            - ticket_cost            # Fixed cost per trade
        )
        
        # Create component breakdown
        components = RewardComponents(
            risk_free_nav_change=risk_free_nav_change,
            embedded_impact=embedded_impact,
            downside_semi_variance=downside_penalty,
            kelly_bonus=kelly_bonus,
            position_decay_penalty=position_decay_penalty,
            turnover_penalty=turnover_penalty,
            size_penalty=size_penalty,
            hold_bonus=hold_bonus,
            action_change_penalty=action_change_penalty,
            ticket_cost=ticket_cost,
            total_reward=total_reward
        )
        
        # Update state tracking
        self._update_state_tracking(curr_portfolio_value, prev_portfolio_value)
        
        if self.verbose and step % 1000 == 0:
            logger.info(f"Step {step}: V3 reward breakdown: {components.to_dict()}")
        
        return total_reward, components
    
    def _calculate_embedded_impact(
        self, nvda_trade_value: float, msft_trade_value: float,
        nvda_price: float, msft_price: float
    ) -> float:
        """Calculate embedded impact using Kyle lambda model with share-based scaling"""
        
        if nvda_trade_value == 0 and msft_trade_value == 0:
            return 0.0
        
        total_impact = 0.0
        
        # NVDA impact (if any)
        if nvda_trade_value != 0:
            nvda_shares = abs(nvda_trade_value) / nvda_price
            normalized_nvda_size = nvda_shares / self.adv_scaling  # Shares/ADV
            nvda_impact_multiplier = normalized_nvda_size ** self.impact_exponent
            nvda_impact = abs(nvda_trade_value) * (self.base_impact_bp / 10000) * (1 + 3 * nvda_impact_multiplier)
            total_impact += nvda_impact
        
        # MSFT impact (if any) - using same ADV assumption for simplicity
        if msft_trade_value != 0:
            msft_shares = abs(msft_trade_value) / msft_price
            normalized_msft_size = msft_shares / (self.adv_scaling * 0.75)  # MSFT ADV ~30M vs NVDA 40M
            msft_impact_multiplier = normalized_msft_size ** self.impact_exponent
            msft_impact = abs(msft_trade_value) * (self.base_impact_bp / 10000) * (1 + 3 * msft_impact_multiplier)
            total_impact += msft_impact
        
        return total_impact
    
    def _calculate_downside_penalty(
        self, curr_portfolio: float, prev_portfolio: float
    ) -> float:
        """Calculate penalty for downside semi-variance only"""
        
        if len(self.return_history) < 5:  # Need minimum history
            return 0.0
        
        # Calculate recent returns
        recent_returns = self.return_history[-min(self.downside_lookback, len(self.return_history)):]
        
        # Only penalize negative returns (semi-variance)
        negative_returns = [r for r in recent_returns if r < 0]
        
        if len(negative_returns) < 2:
            return 0.0
        
        # Semi-variance calculation
        mean_negative = np.mean(negative_returns)
        semi_variance = np.mean([(r - mean_negative) ** 2 for r in negative_returns])
        
        # Penalty scales with current portfolio value
        penalty = self.downside_penalty_coef * curr_portfolio * semi_variance
        
        return penalty
    
    def _calculate_kelly_bonus(
        self, curr_portfolio: float, prev_portfolio: float
    ) -> float:
        """Calculate Kelly-style log bonus for good returns"""
        
        if prev_portfolio <= 0:
            return 0.0
        
        # Calculate return
        portfolio_return = (curr_portfolio - prev_portfolio) / prev_portfolio
        
        # Only apply Kelly bonus for positive returns
        if portfolio_return <= 0:
            return 0.0
        
        # Kelly bonus: log(1 + return) with floor to prevent log(0)
        adjusted_return = max(portfolio_return, self.kelly_floor)
        log_return = np.log(1 + adjusted_return)
        
        # Scale bonus
        bonus = self.kelly_bonus_coef * prev_portfolio * log_return
        
        return bonus
    
    def _calculate_position_decay_penalty(
        self, nvda_position: float, msft_position: float, features: Optional[np.ndarray] = None
    ) -> float:
        """Calculate penalty for holding positions during piecewise OFF periods (with hard clamp backup)"""
        
        if features is None or len(features) < 14:
            # No OFF period indicator available, no penalty
            return 0.0
        
        # Extract OFF period indicator from features (14th feature, index 13)
        off_period_indicator = features[13] if len(features) > 13 else 0.0
        
        if off_period_indicator <= 0.0:
            # Not in OFF period, no penalty
            return 0.0
        
        # Calculate absolute position size (as USD value)
        total_position_value = abs(nvda_position) + abs(msft_position)
        
        if total_position_value <= 0:
            # No position, no penalty
            return 0.0
        
        # Apply moderate penalty (hard clamp should prevent most violations)
        penalty = self.position_decay_penalty * total_position_value * off_period_indicator
        
        if self.verbose:
            logger.info(f"🔄 Position decay penalty: {penalty:.2f} (OFF={off_period_indicator:.1f}, pos=${total_position_value:.0f})")
        
        return penalty
    
    def _calculate_turnover_penalty(
        self, nvda_trade_value: float, msft_trade_value: float, 
        nav: float, features: Optional[np.ndarray] = None
    ) -> float:
        """Calculate turnover penalty with OFF-period kicker (×2.5 when α ≈ 0)"""
        
        if nav <= 0:
            return 0.0
        
        # Calculate total turnover as fraction of NAV
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        if total_trade_value <= 0:
            return 0.0
            
        turnover_ratio = total_trade_value / nav
        
        # Determine if we're in OFF period (α ≈ 0)
        alpha_mag = 0.0
        if features is not None and len(features) >= 13:
            alpha_signal = features[12]  # Alpha signal feature
            alpha_mag = abs(alpha_signal)
        
        # Apply kicker: base λ for ON periods, ×2.5 for OFF periods
        if alpha_mag < 1e-6:
            # OFF period - apply kicker
            lambda_turnover = self.lambda_turnover_on * self.lambda_turnover_off_multiplier
            period_type = "OFF"
        else:
            # ON period - base penalty
            lambda_turnover = self.lambda_turnover_on
            period_type = "ON"
        
        penalty = lambda_turnover * turnover_ratio
        
        if self.verbose and total_trade_value > 0:
            logger.info(f"🔄 Turnover penalty: {penalty:.4f} ({period_type}, λ={lambda_turnover:.4f}, turnover={turnover_ratio:.3f})")
        
        return penalty
    
    def _calculate_size_penalty(
        self, nvda_position: float, msft_position: float, nav: float
    ) -> float:
        """Calculate quadratic penalty for positions exceeding 30% NAV soft limit"""
        
        if nav <= 0:
            return 0.0
        
        # Calculate total notional exposure as fraction of NAV
        total_notional = abs(nvda_position) + abs(msft_position)
        notional_ratio = total_notional / nav
        
        # Apply quadratic penalty if above 30% NAV threshold
        if notional_ratio <= self.max_notional_ratio:
            return 0.0
        
        # Quadratic penalty: κ × (notional_ratio - threshold)²
        excess_ratio = notional_ratio - self.max_notional_ratio
        penalty = self.size_penalty_kappa * nav * (excess_ratio ** 2)
        
        if self.verbose and penalty > 0:
            logger.info(f"📏 Size penalty: {penalty:.4f} (notional={notional_ratio:.2%}, threshold={self.max_notional_ratio:.2%})")
        
        return penalty
    
    def _calculate_hold_bonus(
        self, nvda_trade_value: float, msft_trade_value: float, 
        nav: float, features: Optional[np.ndarray] = None
    ) -> float:
        """Calculate bonus for doing nothing when α≈0 (reviewer recommendation)"""
        
        if nav <= 0:
            return 0.0
        
        # Check if we're doing nothing (no trades)
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        if total_trade_value > 1.0:  # Any meaningful trade
            return 0.0
        
        # Check if alpha is near zero (OFF period or weak signal)
        alpha_mag = 0.0
        if features is not None and len(features) >= 13:
            alpha_signal = features[12]  # Alpha signal feature
            alpha_mag = abs(alpha_signal)
        
        # Bonus only when doing nothing AND alpha is weak
        if alpha_mag < 1e-4:  # Very weak alpha signal
            bonus = self.hold_bonus_coef * nav
            if self.verbose and bonus > 0:
                logger.info(f"🎁 HOLD bonus: {bonus:.4f} (α≈0, no trade)")
            return bonus
        
        return 0.0
    
    def _calculate_action_change_penalty(
        self, nvda_trade_value: float, msft_trade_value: float,
        nvda_price: float, msft_price: float, action: Optional[int] = None
    ) -> float:
        """Calculate penalty for changing actions frequently (reviewer recommendation)"""
        
        if action is None or self.prev_action is None:
            # Store current action for next time
            self.prev_action = action
            return 0.0
        
        # No penalty if action unchanged
        if action == self.prev_action:
            self.prev_action = action
            return 0.0
        
        # Calculate penalty: ψ = 0.5 × tc_bp × |ΔQ|
        total_trade_value = abs(nvda_trade_value) + abs(msft_trade_value)
        if total_trade_value <= 1.0:
            self.prev_action = action
            return 0.0
        
        # Apply action change cost (50% of normal transaction cost)
        if self.action_change_cost_bp > 0:
            penalty = (self.action_change_cost_bp / 10000) * total_trade_value
            if self.verbose and penalty > 0:
                logger.info(f"🔄 Action change penalty: {penalty:.4f} (prev={self.prev_action}, curr={action})")
        else:
            penalty = 0.0
        
        self.prev_action = action
        return penalty
    
    def _calculate_ticket_cost(
        self, nvda_trade_value: float, msft_trade_value: float, nav: float
    ) -> float:
        """Calculate fixed ticket cost per trade (reviewer recommendation)"""
        
        if nav <= 0:
            return 0.0
        
        # Count number of trades (NVDA and/or MSFT)
        num_trades = 0
        if abs(nvda_trade_value) > 1.0:
            num_trades += 1
        if abs(msft_trade_value) > 1.0:
            num_trades += 1
        
        if num_trades == 0:
            return 0.0
        
        # Fixed cost per trade, normalized by NAV
        total_ticket_cost = num_trades * self.ticket_cost_usd
        normalized_cost = total_ticket_cost / nav
        
        if self.verbose and normalized_cost > 0:
            logger.info(f"🎫 Ticket cost: ${total_ticket_cost:.0f} ({num_trades} trades, {normalized_cost:.6f} of NAV)")
        
        return normalized_cost
    
    def _update_state_tracking(self, curr_portfolio: float, prev_portfolio: float):
        """Update internal state for reward calculations"""
        
        self.portfolio_history.append(curr_portfolio)
        
        # Calculate and store return
        if prev_portfolio > 0:
            portfolio_return = (curr_portfolio - prev_portfolio) / prev_portfolio
            self.return_history.append(portfolio_return)
        
        # Keep history manageable
        max_history = 1000
        if len(self.portfolio_history) > max_history:
            self.portfolio_history = self.portfolio_history[-max_history:]
            self.return_history = self.return_history[-max_history:]
    
    def reset(self):
        """Reset state tracking for new episode"""
        self.portfolio_history.clear()
        self.trade_history.clear()
        self.return_history.clear()
        self.prev_action = None  # Reset action tracking
        
        if self.verbose:
            logger.info("🔄 DualTickerRewardV3 state reset (STRUCTURAL REDESIGN)")
    
    def get_stats(self) -> Dict[str, float]:
        """Get reward system statistics"""
        
        if len(self.return_history) < 2:
            return {}
        
        returns = np.array(self.return_history)
        negative_returns = returns[returns < 0]
        
        stats = {
            'episodes': len(self.portfolio_history),
            'total_return': (self.portfolio_history[-1] / self.portfolio_history[0] - 1) if len(self.portfolio_history) > 1 and self.portfolio_history[0] > 0 else 0,
            'mean_return': np.mean(returns),
            'volatility': np.std(returns) * np.sqrt(252 * 390) if len(returns) > 1 else 0,
            'downside_volatility': np.std(negative_returns) * np.sqrt(252 * 390) if len(negative_returns) > 1 else 0,
            'downside_frequency': len(negative_returns) / len(returns) if len(returns) > 0 else 0,
            'kelly_criterion': np.mean(returns) / np.var(returns) if len(returns) > 1 and np.var(returns) > 0 else 0
        }
        
        return stats

def create_reward_calculator_v3(config: Dict[str, Any]) -> DualTickerRewardV3:
    """Factory function to create V3 reward calculator from config"""
    
    return DualTickerRewardV3(
        risk_free_rate_annual=config.get('risk_free_rate_annual', 0.05),
        base_impact_bp=config.get('base_impact_bp', 68.0),
        impact_exponent=config.get('impact_exponent', 0.5),
        adv_scaling=config.get('adv_scaling', 40000000.0),
        downside_penalty_coef=config.get('downside_penalty_coef', 0.01),
        downside_lookback=config.get('downside_lookback', 50),
        kelly_bonus_coef=config.get('kelly_bonus_coef', 0.001),
        kelly_floor=config.get('kelly_floor', 0.001),
        position_decay_penalty=config.get('position_decay_penalty', 0.20),
        step_minutes=config.get('step_minutes', 1.0),
        verbose=config.get('verbose', False)
    )