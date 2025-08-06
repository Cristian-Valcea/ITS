#!/usr/bin/env python3
"""
🎯 REFINED REWARD SYSTEM FOR REAL DATA TRAINING
Implements your specific suggestions:
1. Normalized P&L rewards (bounded [-1,1])
2. Stepwise holding bonuses 
3. Smooth penalty curves with tanh
4. Exploration bonuses with decay
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Structured reward breakdown for analysis"""
    normalized_pnl: float = 0.0
    holding_bonus: float = 0.0
    smoothed_penalty: float = 0.0
    exploration_bonus: float = 0.0
    directional_bonus: float = 0.0
    early_exit_tax: float = 0.0
    time_bonus: float = 0.0
    completion_bonus: float = 0.0
    major_completion_bonus: float = 0.0
    total_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'normalized_pnl': self.normalized_pnl,
            'holding_bonus': self.holding_bonus, 
            'smoothed_penalty': self.smoothed_penalty,
            'exploration_bonus': self.exploration_bonus,
            'directional_bonus': self.directional_bonus,
            'early_exit_tax': self.early_exit_tax,
            'time_bonus': self.time_bonus,
            'completion_bonus': self.completion_bonus,
            'major_completion_bonus': self.major_completion_bonus,
            'total_reward': self.total_reward
        }

class RefinedRewardSystem:
    """
    Refined reward system implementing your specific suggestions for better learning
    on real market data
    """
    
    def __init__(
        self,
        # Core parameters
        initial_capital: float = 10000.0,
        
        # Normalized P&L parameters  
        pnl_epsilon: float = 1000.0,        # Min value for normalization denominator
        
        # Holding bonus parameters
        holding_alpha: float = 0.01,        # α ≈ 0.01 for holding bonus
        holding_lookback_k: int = 5,        # Lookback steps for holding bonus
        holding_min_ret: float = 0.0,       # Minimum return threshold
        
        # Penalty smoothing parameters
        penalty_beta: float = 0.5,          # β ≈ 0.5 for smooth penalties
        
        # Exploration parameters
        exploration_coef: float = 0.05,     # Initial exploration bonus
        exploration_decay: float = 0.9999,  # Decay per step
        
        # Directional bonus parameters  
        directional_weight: float = 0.1,    # Weight for directional correctness
        
        # Early-exit tax parameters
        early_exit_tax: float = 0.0,         # Penalty for short episodes
        min_episode_length: int = 80,        # Minimum episode length threshold
        
        # Corrective incentive parameters (Phase 2 fix)
        time_bonus: float = 0.0,             # Per-step bonus for staying in market
        time_bonus_threshold: int = 60,      # Start time bonus at this step
        completion_bonus: float = 0.0,       # Bonus for completing long episodes
        completion_threshold: int = 80,      # Threshold for completion bonus
        
        # V2 Enhanced incentive parameters
        secondary_tax: float = 0.0,          # Secondary tax for intermediate lengths
        secondary_threshold: int = 70,       # Secondary tax threshold
        major_completion_bonus: float = 0.0, # Major bonus for very long episodes
        major_threshold: int = 80,           # Major completion threshold
        
        verbose: bool = True
    ):
        self.initial_capital = initial_capital
        self.pnl_epsilon = pnl_epsilon
        self.holding_alpha = holding_alpha
        self.holding_lookback_k = holding_lookback_k
        self.holding_min_ret = holding_min_ret
        self.penalty_beta = penalty_beta
        self.exploration_coef = exploration_coef
        self.exploration_decay = exploration_decay
        self.directional_weight = directional_weight
        self.early_exit_tax = early_exit_tax
        self.min_episode_length = min_episode_length
        self.time_bonus = time_bonus
        self.time_bonus_threshold = time_bonus_threshold
        self.completion_bonus = completion_bonus
        self.completion_threshold = completion_threshold
        self.secondary_tax = secondary_tax
        self.secondary_threshold = secondary_threshold
        self.major_completion_bonus = major_completion_bonus
        self.major_threshold = major_threshold
        self.verbose = verbose
        
        # State tracking
        self.peak_portfolio_abs = initial_capital  # Track |P_peak| for normalization
        self.current_exploration_coef = exploration_coef
        self.step_count = 0
        
        # Holding bonus tracking
        self.portfolio_history = []  # Track portfolio values for lookback
        self.position_history = []   # Track positions for lookback
        self.holding_bonus_triggers = 0
        self.steps_in_position = 0
        
        if self.verbose:
            logger.info(f"🎯 Refined Reward System initialized")
            logger.info(f"   Normalized P&L: ε={pnl_epsilon}")
            logger.info(f"   Holding bonus: α={holding_alpha}, lookback={holding_lookback_k}")
            logger.info(f"   Penalty smoothing: β={penalty_beta}")
            logger.info(f"   Exploration coef: {exploration_coef}")
    
    def reset_episode(self):
        """Reset for new episode"""
        self.step_count = 0
        # Reset episode-specific tracking
        self.portfolio_history = []
        self.position_history = []
        self.holding_bonus_triggers = 0
        self.steps_in_position = 0
        # Keep peak portfolio value across episodes for normalization stability
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        nvda_position: float,
        msft_position: float,
        action: int,
        drawdown_pct: float = 0.0
    ) -> RewardComponents:
        """
        Calculate refined reward using your specific formulations
        
        Args:
            portfolio_value: Current portfolio value  
            previous_portfolio_value: Previous portfolio value
            nvda_position: Current NVDA position
            msft_position: Current MSFT position
            action: Action taken (0-4, where 4=Hold Both)
            drawdown_pct: Current drawdown percentage [0,1]
            
        Returns:
            RewardComponents with detailed breakdown
        """
        
        self.step_count += 1
        components = RewardComponents()
        
        # Update history tracking
        self.portfolio_history.append(portfolio_value)
        total_position = nvda_position + msft_position
        self.position_history.append(total_position)
        
        # Keep only lookback_k + 1 history
        if len(self.portfolio_history) > self.holding_lookback_k + 1:
            self.portfolio_history.pop(0)
            self.position_history.pop(0)
        
        # 1. NORMALIZED P&L REWARD: r_t = ΔP_t / max(|P_peak|, ε)
        pnl_change = portfolio_value - previous_portfolio_value
        
        # Update peak portfolio absolute value
        if abs(portfolio_value) > self.peak_portfolio_abs:
            self.peak_portfolio_abs = abs(portfolio_value)
        
        # Normalize P&L change
        normalization_denominator = max(self.peak_portfolio_abs, self.pnl_epsilon)
        components.normalized_pnl = pnl_change / normalization_denominator
        
        # 2. IMPROVED HOLDING BONUS: Support long & short, lookback, positivity gate
        in_position = abs(total_position) > 0
        if in_position:
            self.steps_in_position += 1
            
            # Check if we have enough history for lookback
            if len(self.portfolio_history) >= self.holding_lookback_k + 1:
                # Calculate lookback return
                price_t = portfolio_value
                price_t_k = self.portfolio_history[-(self.holding_lookback_k + 1)]
                
                if price_t_k > 0:  # Avoid division by zero
                    lookback_ret = (price_t / price_t_k) - 1.0
                    
                    # Align return with position direction
                    aligned = np.sign(total_position) * lookback_ret
                    
                    # Reward only when in-the-money over lookback period
                    if aligned > self.holding_min_ret:
                        components.holding_bonus = self.holding_alpha * aligned
                        self.holding_bonus_triggers += 1
        
        # 3. SMOOTH PENALTY: r_t -= β·tanh(drawdown)  
        if drawdown_pct > 0:
            components.smoothed_penalty = -self.penalty_beta * np.tanh(drawdown_pct)
        
        # 4. EXPLORATION BONUS (decaying)
        if action != 4:  # Non-hold actions get exploration bonus
            components.exploration_bonus = self.current_exploration_coef
        
        # Decay exploration coefficient per step
        self.current_exploration_coef *= self.exploration_decay
        
        # 5. DIRECTIONAL CORRECTNESS BONUS (simplified)
        # Bonus if we're positioned correctly for the P&L direction
        if pnl_change != 0:
            total_position = nvda_position + msft_position
            # Reward if position direction matches P&L direction
            if (total_position > 0 and pnl_change > 0) or (total_position < 0 and pnl_change < 0):
                components.directional_bonus = self.directional_weight * abs(pnl_change) / normalization_denominator
        
        # 6. TIME BONUS (Corrective incentive - per-step bonus for staying in market)
        if self.time_bonus > 0 and self.step_count >= self.time_bonus_threshold:
            components.time_bonus = self.time_bonus
        
        # TOTAL REWARD
        components.total_reward = (
            components.normalized_pnl +
            components.holding_bonus +
            components.smoothed_penalty +
            components.exploration_bonus + 
            components.directional_bonus +
            components.early_exit_tax +
            components.time_bonus +
            components.completion_bonus +
            components.major_completion_bonus
        )
        
        return components
    
    def apply_early_exit_tax(self, episode_length: int) -> Tuple[float, float, float]:
        """
        Apply early-exit tax and completion bonuses based on episode length (V2 enhanced)
        
        Args:
            episode_length: Length of the completed episode
            
        Returns:
            Tuple of (early_exit_tax, completion_bonus, major_completion_bonus)
        """
        early_exit_tax = 0.0
        completion_bonus = 0.0
        major_completion_bonus = 0.0
        
        # V2 Dual tax structure
        if episode_length < self.min_episode_length:
            # Primary tax for very short episodes
            if self.early_exit_tax > 0:
                early_exit_tax = -self.early_exit_tax
                if self.verbose:
                    logger.info(f"🚨 Primary early-exit tax: Episode length {episode_length} < {self.min_episode_length}, penalty: {early_exit_tax}")
        elif episode_length < self.secondary_threshold:
            # Secondary tax for intermediate episodes
            if self.secondary_tax > 0:
                early_exit_tax = -self.secondary_tax
                if self.verbose:
                    logger.info(f"⚠️ Secondary early-exit tax: Episode length {episode_length} < {self.secondary_threshold}, penalty: {early_exit_tax}")
        
        # V2 Dual completion bonus structure
        if episode_length >= self.major_threshold and self.major_completion_bonus > 0:
            # Major bonus for very long episodes
            major_completion_bonus = self.major_completion_bonus
            if self.verbose:
                logger.info(f"🏆 Major completion bonus: Episode length {episode_length} ≥ {self.major_threshold}, bonus: {major_completion_bonus}")
        
        if episode_length >= self.completion_threshold and self.completion_bonus > 0:
            # Regular completion bonus
            completion_bonus = self.completion_bonus
            if self.verbose:
                logger.info(f"🎉 Completion bonus: Episode length {episode_length} ≥ {self.completion_threshold}, bonus: {completion_bonus}")
        
        return early_exit_tax, completion_bonus, major_completion_bonus
    
    def get_stats(self) -> Dict[str, float]:
        """Get current reward system statistics"""
        return {
            'peak_portfolio_abs': self.peak_portfolio_abs,
            'current_exploration_coef': self.current_exploration_coef,
            'normalization_denominator': max(self.peak_portfolio_abs, self.pnl_epsilon),
            'step_count': self.step_count,
            'holding_bonus_triggers': self.holding_bonus_triggers,
            'steps_in_position': self.steps_in_position,
            'holding_trigger_rate': self.holding_bonus_triggers / max(self.steps_in_position, 1)
        }