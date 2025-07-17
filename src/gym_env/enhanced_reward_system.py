# src/gym_env/enhanced_reward_system.py
"""
Enhanced Reward System for High-Frequency Trading Environment.

Addresses the core issue: reward tied directly to realized P&L minus cost
results in near-zero or negative rewards that provide weak learning signals.

This module implements a hybrid reward system that combines:
1. Directional signal rewards (Option A)
2. Behavioral shaping rewards (Option B) 
3. Multi-timeframe reward aggregation
4. Adaptive reward scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import logging


class EnhancedRewardCalculator:
    """
    Enhanced reward calculator for high-frequency trading environments.
    
    Combines multiple reward components to provide stronger learning signals
    while maintaining alignment with actual profitability.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default configuration
        default_config = {
            # Core reward weights
            'realized_pnl_weight': 1.0,        # Weight for actual P&L
            'directional_weight': 0.3,         # Weight for directional signal
            'behavioral_weight': 0.2,          # Weight for behavioral shaping
            
            # Directional signal parameters
            'directional_scaling': 0.001,      # Scale factor for directional rewards
            'min_price_change_bps': 0.5,       # Minimum price change to reward (0.5 bps)
            
            # Behavioral shaping parameters
            'flip_flop_penalty': 0.001,        # Penalty for position changes
            'holding_bonus': 0.0001,           # Small bonus for maintaining position
            'correct_direction_bonus': 0.1,    # Bonus for correct directional trades
            'wrong_direction_penalty': 0.1,    # Penalty for wrong directional trades
            
            # Multi-timeframe aggregation
            'enable_multi_timeframe': True,    # Enable multi-timeframe rewards
            'short_window': 5,                 # Short-term window (5 minutes)
            'medium_window': 15,               # Medium-term window (15 minutes)
            'long_window': 60,                 # Long-term window (1 hour)
            
            # Adaptive scaling
            'enable_adaptive_scaling': True,   # Enable adaptive reward scaling
            'target_reward_magnitude': 0.01,   # Target reward magnitude
            'scaling_window': 100,             # Window for scaling calculation
            'min_scaling_factor': 0.1,         # Minimum scaling factor
            'max_scaling_factor': 10.0,        # Maximum scaling factor
        }
        
        self.config = {**default_config, **(config or {})}
        
        # State tracking
        self.price_history = deque(maxlen=max(
            self.config['short_window'],
            self.config['medium_window'], 
            self.config['long_window']
        ))
        self.position_history = deque(maxlen=self.config['long_window'])
        self.reward_history = deque(maxlen=self.config['scaling_window'])
        
        # Previous state for behavioral analysis
        self.previous_position = 0
        self.previous_price = None
        
        # Adaptive scaling state
        self.current_scaling_factor = 1.0
        
        self.logger.info(f"EnhancedRewardCalculator initialized with config: {self.config}")
    
    def calculate_reward(self,
                        realized_pnl: float,
                        transaction_cost: float,
                        current_position: int,
                        current_price: float,
                        portfolio_value: float,
                        step_info: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate enhanced reward using hybrid approach.
        
        Args:
            realized_pnl: Realized P&L from trades
            transaction_cost: Transaction costs incurred
            current_position: Current position (-1, 0, 1)
            current_price: Current market price
            portfolio_value: Current portfolio value
            step_info: Additional step information
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Update state tracking
        self.price_history.append(current_price)
        self.position_history.append(current_position)
        
        # Initialize reward components
        reward_components = {}
        
        # 1. CORE REWARD: Net P&L (existing system)
        net_pnl = realized_pnl - transaction_cost
        core_reward = net_pnl * self.config['realized_pnl_weight']
        reward_components['core_pnl'] = core_reward
        
        # 2. DIRECTIONAL SIGNAL REWARD (Option A Enhanced)
        directional_reward = self._calculate_directional_reward(
            current_position, current_price
        )
        reward_components['directional'] = directional_reward
        
        # 3. BEHAVIORAL SHAPING REWARD (Option B Enhanced)
        behavioral_reward = self._calculate_behavioral_reward(
            current_position, realized_pnl, transaction_cost
        )
        reward_components['behavioral'] = behavioral_reward
        
        # 4. MULTI-TIMEFRAME REWARD (if enabled)
        if self.config['enable_multi_timeframe']:
            multi_timeframe_reward = self._calculate_multi_timeframe_reward(
                current_position
            )
            reward_components['multi_timeframe'] = multi_timeframe_reward
        else:
            reward_components['multi_timeframe'] = 0.0
        
        # 5. COMBINE ALL COMPONENTS
        total_reward = (
            reward_components['core_pnl'] +
            reward_components['directional'] * self.config['directional_weight'] +
            reward_components['behavioral'] * self.config['behavioral_weight'] +
            reward_components['multi_timeframe']
        )
        
        # 6. ADAPTIVE SCALING (if enabled)
        if self.config['enable_adaptive_scaling']:
            total_reward = self._apply_adaptive_scaling(total_reward)
            reward_components['scaling_factor'] = self.current_scaling_factor
        
        # Update state for next step
        self.previous_position = current_position
        self.previous_price = current_price
        self.reward_history.append(total_reward)
        
        # Detailed breakdown for monitoring
        reward_breakdown = {
            **reward_components,
            'total_reward': total_reward,
            'raw_pnl': realized_pnl,
            'transaction_cost': transaction_cost,
            'net_pnl': net_pnl
        }
        
        return total_reward, reward_breakdown
    
    def _calculate_directional_reward(self, current_position: int, current_price: float) -> float:
        """Calculate directional signal reward (Option A Enhanced)."""
        if self.previous_price is None or len(self.price_history) < 2:
            return 0.0
        
        # Calculate price change
        price_change = current_price - self.previous_price
        price_change_bps = (price_change / self.previous_price) * 10000
        
        # Only reward if price change is significant
        if abs(price_change_bps) < self.config['min_price_change_bps']:
            return 0.0
        
        # Directional reward: position * price_change
        # Positive when position aligns with price movement
        directional_reward = current_position * price_change * self.config['directional_scaling']
        
        return directional_reward
    
    def _calculate_behavioral_reward(self, 
                                   current_position: int, 
                                   realized_pnl: float,
                                   transaction_cost: float) -> float:
        """Calculate behavioral shaping reward (Option B Enhanced)."""
        behavioral_reward = 0.0
        
        # 1. Flip-flop penalty
        if self.previous_position != current_position:
            position_change = abs(current_position - self.previous_position)
            flip_flop_penalty = -self.config['flip_flop_penalty'] * position_change
            behavioral_reward += flip_flop_penalty
        
        # 2. Holding bonus (small reward for maintaining position)
        if current_position != 0 and current_position == self.previous_position:
            behavioral_reward += self.config['holding_bonus']
        
        # 3. Correct/wrong direction bonus/penalty
        if realized_pnl != 0:  # Only when we have actual trades
            if realized_pnl > transaction_cost:  # Profitable trade
                behavioral_reward += self.config['correct_direction_bonus']
            else:  # Unprofitable trade
                behavioral_reward -= self.config['wrong_direction_penalty']
        
        return behavioral_reward
    
    def _calculate_multi_timeframe_reward(self, current_position: int) -> float:
        """Calculate multi-timeframe reward aggregation."""
        if len(self.price_history) < self.config['short_window']:
            return 0.0
        
        multi_timeframe_reward = 0.0
        
        # Calculate returns over different timeframes
        timeframes = [
            ('short', self.config['short_window'], 0.1),
            ('medium', self.config['medium_window'], 0.05),
            ('long', self.config['long_window'], 0.02)
        ]
        
        for name, window, weight in timeframes:
            if len(self.price_history) >= window:
                start_price = self.price_history[-window]
                end_price = self.price_history[-1]
                timeframe_return = (end_price - start_price) / start_price
                
                # Reward alignment with timeframe trend
                alignment_reward = current_position * timeframe_return * weight
                multi_timeframe_reward += alignment_reward
        
        return multi_timeframe_reward
    
    def _apply_adaptive_scaling(self, reward: float) -> float:
        """Apply adaptive reward scaling to maintain target magnitude."""
        if len(self.reward_history) < 10:  # Need some history
            return reward
        
        # Calculate recent reward statistics
        recent_rewards = list(self.reward_history)[-50:]  # Last 50 rewards
        reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 1.0
        reward_mean_abs = np.mean(np.abs(recent_rewards))
        
        # Adjust scaling factor to maintain target magnitude
        if reward_mean_abs > 0:
            target_scaling = self.config['target_reward_magnitude'] / reward_mean_abs
            
            # Smooth scaling factor updates
            alpha = 0.1  # Smoothing factor
            self.current_scaling_factor = (
                alpha * target_scaling + 
                (1 - alpha) * self.current_scaling_factor
            )
            
            # Apply bounds
            self.current_scaling_factor = np.clip(
                self.current_scaling_factor,
                self.config['min_scaling_factor'],
                self.config['max_scaling_factor']
            )
        
        return reward * self.current_scaling_factor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward calculator statistics."""
        if not self.reward_history:
            return {}
        
        recent_rewards = list(self.reward_history)
        
        return {
            'reward_count': len(recent_rewards),
            'reward_mean': np.mean(recent_rewards),
            'reward_std': np.std(recent_rewards),
            'reward_min': np.min(recent_rewards),
            'reward_max': np.max(recent_rewards),
            'current_scaling_factor': self.current_scaling_factor,
            'price_history_length': len(self.price_history),
            'position_history_length': len(self.position_history)
        }
    
    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.price_history.clear()
        self.position_history.clear()
        self.reward_history.clear()
        
        self.previous_position = 0
        self.previous_price = None
        self.current_scaling_factor = 1.0
        
        self.logger.debug("EnhancedRewardCalculator reset")


def create_enhanced_reward_calculator(config: Dict[str, Any] = None) -> EnhancedRewardCalculator:
    """Factory function to create enhanced reward calculator."""
    return EnhancedRewardCalculator(config)