"""
VolatilityPenalty - Online variance tracking with Welford algorithm

Implements efficient online calculation of rolling return volatility
using Welford's algorithm for numerical stability. Provides volatility-based
penalty for reward shaping.
"""

import logging
import math
from collections import deque
from typing import Dict, Any


class WelfordVariance:
    """
    Online variance calculation using Welford's algorithm.
    
    Numerically stable and memory-efficient for rolling windows.
    """
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
    
    def update(self, value: float) -> float:
        """
        Update with new value and return current standard deviation.
        
        Args:
            value: New return value
            
        Returns:
            Current standard deviation (volatility)
        """
        # If window is full, remove oldest value
        if len(self.values) == self.window_size:
            old_value = self.values[0]
            self._remove_value(old_value)
        
        # Add new value
        self.values.append(value)
        self._add_value(value)
        
        return self.get_std()
    
    def _add_value(self, value: float):
        """Add value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def _remove_value(self, value: float):
        """Remove value using reverse Welford's algorithm."""
        if self.count <= 1:
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0
            return
        
        delta = value - self.mean
        self.mean = (self.mean * self.count - value) / (self.count - 1)
        delta2 = value - self.mean
        self.m2 -= delta * delta2
        self.count -= 1
    
    def get_variance(self) -> float:
        """Get current variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        return math.sqrt(self.get_variance())
    
    def reset(self):
        """Reset all state."""
        self.values.clear()
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


class VolatilityPenalty:
    """
    Volatility-based penalty calculator for reward shaping.
    
    Uses Welford's algorithm to efficiently track rolling return volatility
    and applies configurable penalty weight.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize volatility penalty calculator.
        
        Args:
            config: Configuration dict containing:
                - window_size: Rolling window size (default: 60)
                - penalty_lambda: Penalty weight (default: 0.25)
        """
        self.logger = logging.getLogger("VolatilityPenalty")
        
        self.window_size = config.get('window_size', 60)
        self.penalty_lambda = config.get('penalty_lambda', 0.25)
        
        # Initialize Welford variance calculator
        self.welford = WelfordVariance(self.window_size)
        
        # Tracking for statistics
        self.total_penalty = 0.0
        self.step_count = 0
        
        self.logger.info(f"VolatilityPenalty initialized - window: {self.window_size}, "
                        f"lambda: {self.penalty_lambda}")
    
    def update(self, step_return: float) -> float:
        """
        Update with new return and calculate penalty.
        
        Args:
            step_return: Step return as decimal (e.g., 0.01 for 1%)
            
        Returns:
            Penalty value to subtract from reward
        """
        # Update volatility calculation
        current_volatility = self.welford.update(step_return)
        
        # Calculate penalty: λ * σ
        penalty = self.penalty_lambda * current_volatility
        
        # Track statistics
        self.total_penalty += penalty
        self.step_count += 1
        
        # Log significant volatility events
        if current_volatility > 0.05:  # 5% volatility threshold
            self.logger.debug(f"High volatility detected: {current_volatility:.4f} "
                            f"(penalty: {penalty:.4f})")
        
        return penalty
    
    def get_current_volatility(self) -> float:
        """Get current rolling volatility."""
        return self.welford.get_std()
    
    def get_average_volatility(self) -> float:
        """Get average volatility over the episode."""
        if self.step_count == 0:
            return 0.0
        
        # Calculate average from total penalty and lambda
        return (self.total_penalty / self.step_count) / self.penalty_lambda if self.penalty_lambda > 0 else 0.0
    
    def get_total_penalty(self) -> float:
        """Get total penalty applied during episode."""
        return self.total_penalty
    
    def update_lambda(self, new_lambda: float):
        """
        Update penalty weight (used by curriculum scheduler).
        
        Args:
            new_lambda: New penalty weight
        """
        old_lambda = self.penalty_lambda
        self.penalty_lambda = new_lambda
        self.logger.info(f"Updated penalty lambda: {old_lambda:.3f} → {new_lambda:.3f}")
    
    def reset(self):
        """Reset state for new episode."""
        self.welford.reset()
        self.total_penalty = 0.0
        self.step_count = 0
        
        self.logger.debug("VolatilityPenalty reset for new episode")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed statistics for logging."""
        return {
            'current_volatility': self.get_current_volatility(),
            'average_volatility': self.get_average_volatility(),
            'total_penalty': self.total_penalty,
            'penalty_lambda': self.penalty_lambda,
            'window_size': self.window_size,
            'step_count': self.step_count
        }