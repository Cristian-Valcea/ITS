"""
Action Penalty Wrapper

Applies penalties for specific trading behaviors to encourage
more disciplined and thoughtful trading decisions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List
import logging


class ActionPenaltyWrapper(gym.Wrapper):
    """
    Applies penalties for undesirable trading behaviors.
    
    Penalties include:
    - Frequent action changes (ping-ponging)
    - Trading too close to market open/close
    - Excessive position flipping
    - Trading without sufficient market signal
    
    This wrapper helps encourage:
    - More deliberate trading decisions
    - Reduced noise trading
    - Better market timing
    - Consistent trading strategies
    """
    
    def __init__(self, 
                 env: gym.Env,
                 action_change_penalty: float = 0.001,
                 ping_pong_penalty: float = 0.005,
                 rapid_flip_penalty: float = 0.01,
                 ping_pong_window: int = 5,
                 enable_timing_penalties: bool = False,
                 market_open_penalty_window: int = 30,
                 market_close_penalty_window: int = 30):
        """
        Initialize action penalty wrapper.
        
        Args:
            env: Base trading environment
            action_change_penalty: Penalty for changing actions (L2 norm)
            ping_pong_penalty: Additional penalty for rapid back-and-forth trading
            rapid_flip_penalty: Penalty for multiple position flips in short time
            ping_pong_window: Window size to detect ping-pong behavior
            enable_timing_penalties: Enable penalties for poor market timing
            market_open_penalty_window: Steps to penalize trading after market open
            market_close_penalty_window: Steps to penalize trading before market close
        """
        super().__init__(env)
        self.action_change_penalty = action_change_penalty
        self.ping_pong_penalty = ping_pong_penalty
        self.rapid_flip_penalty = rapid_flip_penalty
        self.ping_pong_window = ping_pong_window
        self.enable_timing_penalties = enable_timing_penalties
        self.market_open_penalty_window = market_open_penalty_window
        self.market_close_penalty_window = market_close_penalty_window
        
        self.logger = logging.getLogger("ActionPenaltyWrapper")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Track action history for penalty calculation
        self.action_history: List[int] = []
        self.position_history: List[int] = []
        self.step_count = 0
        
        # Penalty tracking
        self.total_action_change_penalty = 0.0
        self.total_ping_pong_penalty = 0.0
        self.total_rapid_flip_penalty = 0.0
        self.total_timing_penalty = 0.0
        
        self.logger.info(f"Action penalty wrapper active: "
                        f"change={action_change_penalty}, "
                        f"ping_pong={ping_pong_penalty}, "
                        f"rapid_flip={rapid_flip_penalty}")
    
    def step(self, action):
        """Execute action and apply penalties."""
        self.step_count += 1
        
        # Execute base action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Calculate penalties
        penalty = self._calculate_penalties(action)
        
        # Apply penalty to reward
        original_reward = reward
        reward -= penalty
        
        # Update history
        self.action_history.append(action)
        if hasattr(self.env, 'current_position'):
            self.position_history.append(self.env.current_position)
        
        # Limit history size
        if len(self.action_history) > self.ping_pong_window * 2:
            self.action_history.pop(0)
            if self.position_history:
                self.position_history.pop(0)
        
        # Add penalty info to step info
        info.update({
            'action_penalty_total': penalty,
            'action_change_penalty': self._get_last_action_change_penalty(),
            'ping_pong_penalty': self._get_last_ping_pong_penalty(),
            'rapid_flip_penalty': self._get_last_rapid_flip_penalty(),
            'timing_penalty': self._get_last_timing_penalty() if self.enable_timing_penalties else 0.0,
            'original_reward': original_reward,
            'penalized_reward': reward,
            'action_penalty_stats': self.get_penalty_stats()
        })
        
        return obs, reward, done, truncated, info
    
    def _calculate_penalties(self, action: int) -> float:
        """Calculate total penalty for current action."""
        total_penalty = 0.0
        
        # Action change penalty
        action_change_pen = self._calculate_action_change_penalty(action)
        total_penalty += action_change_pen
        self.total_action_change_penalty += action_change_pen
        
        # Ping-pong penalty
        ping_pong_pen = self._calculate_ping_pong_penalty(action)
        total_penalty += ping_pong_pen
        self.total_ping_pong_penalty += ping_pong_pen
        
        # Rapid flip penalty
        rapid_flip_pen = self._calculate_rapid_flip_penalty()
        total_penalty += rapid_flip_pen
        self.total_rapid_flip_penalty += rapid_flip_pen
        
        # Timing penalty (if enabled)
        if self.enable_timing_penalties:
            timing_pen = self._calculate_timing_penalty(action)
            total_penalty += timing_pen
            self.total_timing_penalty += timing_pen
        
        return total_penalty
    
    def _calculate_action_change_penalty(self, action: int) -> float:
        """Penalty for changing actions (L2 norm)."""
        if len(self.action_history) == 0:
            return 0.0
        
        last_action = self.action_history[-1]
        if action != last_action:
            penalty = self.action_change_penalty * (action - last_action) ** 2
            self.logger.debug(f"Action change penalty: {penalty:.6f} "
                            f"(action {last_action} → {action})")
            return penalty
        
        return 0.0
    
    def _calculate_ping_pong_penalty(self, action: int) -> float:
        """Penalty for ping-pong trading behavior."""
        if len(self.action_history) < self.ping_pong_window:
            return 0.0
        
        # Check for alternating pattern in recent history
        recent_actions = self.action_history[-self.ping_pong_window:] + [action]
        
        # Count action changes in window
        changes = sum(1 for i in range(1, len(recent_actions)) 
                     if recent_actions[i] != recent_actions[i-1])
        
        # Penalize if too many changes (ping-ponging)
        if changes >= self.ping_pong_window // 2:
            penalty = self.ping_pong_penalty * changes
            self.logger.debug(f"Ping-pong penalty: {penalty:.6f} "
                            f"({changes} changes in {self.ping_pong_window} steps)")
            return penalty
        
        return 0.0
    
    def _calculate_rapid_flip_penalty(self) -> float:
        """Penalty for rapid position flips."""
        if len(self.position_history) < 3:
            return 0.0
        
        # Check for rapid position changes
        recent_positions = self.position_history[-3:]
        
        # Look for position flip pattern (e.g., -1 → 0 → 1 or 1 → 0 → -1)
        if len(set(recent_positions)) == 3:  # All different positions
            penalty = self.rapid_flip_penalty
            self.logger.debug(f"Rapid flip penalty: {penalty:.6f} "
                            f"(positions: {recent_positions})")
            return penalty
        
        return 0.0
    
    def _calculate_timing_penalty(self, action: int) -> float:
        """Penalty for trading at poor market times."""
        if action == 1:  # No penalty for HOLD
            return 0.0
        
        # Get total episode steps from environment
        total_steps = getattr(self.env, 'max_episode_steps', 390)
        
        # Penalty for trading too early (market open)
        if self.step_count <= self.market_open_penalty_window:
            penalty = 0.001 * (self.market_open_penalty_window - self.step_count) / self.market_open_penalty_window
            return penalty
        
        # Penalty for trading too late (market close)
        steps_to_close = total_steps - self.step_count
        if steps_to_close <= self.market_close_penalty_window:
            penalty = 0.001 * (self.market_close_penalty_window - steps_to_close) / self.market_close_penalty_window
            return penalty
        
        return 0.0
    
    def _get_last_action_change_penalty(self) -> float:
        """Get the last action change penalty applied."""
        if len(self.action_history) < 2:
            return 0.0
        return self.action_change_penalty * (self.action_history[-1] - self.action_history[-2]) ** 2
    
    def _get_last_ping_pong_penalty(self) -> float:
        """Get the last ping-pong penalty applied."""
        # This is a simplified version - in practice, you'd store the last penalty
        return 0.0
    
    def _get_last_rapid_flip_penalty(self) -> float:
        """Get the last rapid flip penalty applied."""
        return 0.0
    
    def _get_last_timing_penalty(self) -> float:
        """Get the last timing penalty applied."""
        return 0.0
    
    def reset(self, **kwargs):
        """Reset penalty tracking."""
        self.action_history.clear()
        self.position_history.clear()
        self.step_count = 0
        
        # Reset penalty counters
        self.total_action_change_penalty = 0.0
        self.total_ping_pong_penalty = 0.0
        self.total_rapid_flip_penalty = 0.0
        self.total_timing_penalty = 0.0
        
        return self.env.reset(**kwargs)
    
    def get_penalty_stats(self) -> Dict[str, Any]:
        """Get penalty statistics."""
        total_penalty = (self.total_action_change_penalty + 
                        self.total_ping_pong_penalty + 
                        self.total_rapid_flip_penalty + 
                        self.total_timing_penalty)
        
        return {
            'total_action_change_penalty': self.total_action_change_penalty,
            'total_ping_pong_penalty': self.total_ping_pong_penalty,
            'total_rapid_flip_penalty': self.total_rapid_flip_penalty,
            'total_timing_penalty': self.total_timing_penalty,
            'total_penalty': total_penalty,
            'avg_penalty_per_step': total_penalty / max(1, self.step_count),
            'action_history_length': len(self.action_history),
            'position_history_length': len(self.position_history)
        }