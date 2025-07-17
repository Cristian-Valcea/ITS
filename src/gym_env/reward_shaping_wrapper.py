# src/gym_env/reward_shaping_wrapper.py
"""
Reward Shaping Wrapper for applying risk penalties and other reward modifications.

This wrapper allows callbacks and other components to modify the reward signal
without directly modifying the environment's step() function.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that allows external components to modify rewards.
    
    This enables callbacks to inject risk penalties, bonuses, or other
    reward modifications without changing the core environment.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Storage for reward modifiers
        self.reward_modifiers = []
        self.step_penalties = {}  # Store penalties for current step
        self.cumulative_penalties = 0.0
        self.penalty_count = 0
        
        # Track reward components for debugging
        self.last_base_reward = 0.0
        self.last_total_penalty = 0.0
        self.last_shaped_reward = 0.0
        
        self.logger.info("RewardShapingWrapper initialized")
    
    def add_reward_modifier(self, modifier_func: Callable[[float, Dict[str, Any]], float], name: str = None):
        """
        Add a reward modifier function.
        
        Args:
            modifier_func: Function that takes (base_reward, info) and returns modified reward
            name: Optional name for the modifier (for logging)
        """
        modifier_name = name or f"modifier_{len(self.reward_modifiers)}"
        self.reward_modifiers.append((modifier_func, modifier_name))
        self.logger.info(f"Added reward modifier: {modifier_name}")
    
    def add_step_penalty(self, penalty_name: str, penalty_value: float):
        """
        Add a penalty for the current step.
        
        This is called by callbacks or other components to inject penalties.
        
        Args:
            penalty_name: Name/type of penalty
            penalty_value: Penalty amount (positive values are subtracted from reward)
        """
        if penalty_value > 0:
            self.step_penalties[penalty_name] = penalty_value
            self.logger.debug(f"Added step penalty '{penalty_name}': {penalty_value:.6f}")
    
    def clear_step_penalties(self):
        """Clear penalties for the current step."""
        self.step_penalties.clear()
    
    def step(self, action):
        """Step the environment and apply reward shaping."""
        # Clear previous step penalties
        self.clear_step_penalties()
        
        # Step the base environment
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Store base reward for debugging
        self.last_base_reward = base_reward
        
        # Start with base reward
        shaped_reward = base_reward
        total_penalty = 0.0
        
        # Apply step penalties (from callbacks)
        for penalty_name, penalty_value in self.step_penalties.items():
            shaped_reward -= penalty_value
            total_penalty += penalty_value
            self.logger.debug(f"Applied penalty '{penalty_name}': -{penalty_value:.6f}")
        
        # Apply reward modifiers
        for modifier_func, modifier_name in self.reward_modifiers:
            try:
                old_reward = shaped_reward
                shaped_reward = modifier_func(shaped_reward, info)
                modifier_change = shaped_reward - old_reward
                if abs(modifier_change) > 1e-8:  # Only log significant changes
                    self.logger.debug(f"Modifier '{modifier_name}' changed reward by: {modifier_change:+.6f}")
            except Exception as e:
                self.logger.error(f"Reward modifier '{modifier_name}' failed: {e}")
        
        # Update tracking
        self.last_total_penalty = total_penalty
        self.last_shaped_reward = shaped_reward
        
        if total_penalty > 0:
            self.cumulative_penalties += total_penalty
            self.penalty_count += 1
        
        # Add reward shaping info to step info
        info['reward_shaping'] = {
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'total_penalty': total_penalty,
            'step_penalties': dict(self.step_penalties),
            'cumulative_penalties': self.cumulative_penalties,
            'penalty_count': self.penalty_count
        }
        
        # Log significant reward changes
        reward_change = shaped_reward - base_reward
        if abs(reward_change) > 1e-6:
            self.logger.debug(
                f"Reward shaped: {base_reward:.6f} → {shaped_reward:.6f} "
                f"(change: {reward_change:+.6f}, penalty: {total_penalty:.6f})"
            )
        
        return observation, shaped_reward, terminated, truncated, info
    
    def get_penalty_stats(self) -> Dict[str, Any]:
        """Get statistics about applied penalties."""
        avg_penalty = self.cumulative_penalties / max(self.penalty_count, 1)
        return {
            'cumulative_penalties': self.cumulative_penalties,
            'penalty_count': self.penalty_count,
            'average_penalty': avg_penalty,
            'last_base_reward': self.last_base_reward,
            'last_total_penalty': self.last_total_penalty,
            'last_shaped_reward': self.last_shaped_reward
        }


class RiskPenaltyRewardShaper:
    """
    Risk penalty reward shaper that can be used with RewardShapingWrapper.
    
    This integrates with RiskAdvisor to apply risk-based penalties.
    """
    
    def __init__(self, risk_advisor, penalty_weight: float = 0.1):
        self.risk_advisor = risk_advisor
        self.penalty_weight = penalty_weight
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track penalty statistics
        self.total_penalties = 0.0
        self.penalty_count = 0
        
        self.logger.info(f"RiskPenaltyRewardShaper initialized with weight: {penalty_weight}")
    
    def __call__(self, base_reward: float, info: Dict[str, Any]) -> float:
        """
        Apply risk penalty to the reward.
        
        Args:
            base_reward: Original reward from environment
            info: Step info dictionary
            
        Returns:
            Modified reward with risk penalty applied
        """
        try:
            # Get current observation from info if available
            obs = info.get('observation')
            if obs is None:
                return base_reward
            
            # Convert observation to dict format for risk advisor
            obs_dict = self._convert_obs_to_dict(obs)
            
            # Evaluate risk
            risk = self.risk_advisor.evaluate(obs_dict)
            
            # Calculate penalty based on drawdown velocity
            penalty = self.penalty_weight * risk.get('drawdown_velocity', 0)
            
            if penalty > 0:
                self.total_penalties += penalty
                self.penalty_count += 1
                
                self.logger.debug(f"Risk penalty applied: {penalty:.6f}")
                
                # Log periodic statistics
                if self.penalty_count % 100 == 0:
                    avg_penalty = self.total_penalties / self.penalty_count
                    self.logger.info(f"Risk penalties: {self.penalty_count} applied, avg: {avg_penalty:.6f}")
                
                return base_reward - penalty
            
        except Exception as e:
            self.logger.error(f"Risk penalty calculation failed: {e}")
        
        return base_reward
    
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        return {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }


class FunctionalRiskPenaltyCallback:
    """
    Functional risk penalty callback that works with RewardShapingWrapper.
    
    This callback injects penalties into the wrapper instead of just logging them.
    """
    
    def __init__(self, reward_wrapper: RewardShapingWrapper, risk_advisor, penalty_weight: float = 0.1):
        self.reward_wrapper = reward_wrapper
        self.risk_advisor = risk_advisor
        self.penalty_weight = penalty_weight
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track statistics
        self.total_penalties = 0.0
        self.penalty_count = 0
        
        self.logger.info(f"FunctionalRiskPenaltyCallback initialized with weight: {penalty_weight}")
    
    def on_step(self, observation: np.ndarray) -> None:
        """
        Called at each step to evaluate and inject risk penalties.
        
        Args:
            observation: Current observation from environment
        """
        try:
            # Convert observation to dict format for risk advisor
            obs_dict = self._convert_obs_to_dict(observation)
            
            # Evaluate risk
            risk = self.risk_advisor.evaluate(obs_dict)
            
            # Calculate penalty based on drawdown velocity
            penalty = self.penalty_weight * risk.get('drawdown_velocity', 0)
            
            if penalty > 0:
                # Inject penalty into reward wrapper
                self.reward_wrapper.add_step_penalty('risk_penalty', penalty)
                
                # Update statistics
                self.total_penalties += penalty
                self.penalty_count += 1
                
                self.logger.debug(f"Risk penalty injected: {penalty:.6f}")
                
                # Log periodic statistics
                if self.penalty_count % 100 == 0:
                    avg_penalty = self.total_penalties / self.penalty_count
                    self.logger.info(f"Risk penalties injected: {self.penalty_count}, avg: {avg_penalty:.6f}")
        
        except Exception as e:
            self.logger.error(f"Risk penalty injection failed: {e}")
    
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        return {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get penalty statistics."""
        avg_penalty = self.total_penalties / max(self.penalty_count, 1)
        return {
            'total_penalties': self.total_penalties,
            'penalty_count': self.penalty_count,
            'average_penalty': avg_penalty
        }


# Example usage functions
def create_risk_shaped_environment(base_env: gym.Env, risk_advisor, penalty_weight: float = 0.1) -> RewardShapingWrapper:
    """
    Create an environment with risk-based reward shaping.
    
    Args:
        base_env: Base trading environment
        risk_advisor: Risk advisor for penalty calculation
        penalty_weight: Weight for risk penalties
        
    Returns:
        Wrapped environment with risk-based reward shaping
    """
    # Wrap environment with reward shaping
    wrapped_env = RewardShapingWrapper(base_env)
    
    # Add risk penalty modifier
    risk_shaper = RiskPenaltyRewardShaper(risk_advisor, penalty_weight)
    wrapped_env.add_reward_modifier(risk_shaper, "risk_penalty")
    
    return wrapped_env


def integrate_with_callback(wrapped_env: RewardShapingWrapper, risk_advisor, penalty_weight: float = 0.1) -> FunctionalRiskPenaltyCallback:
    """
    Create a functional callback that injects penalties into the wrapped environment.
    
    Args:
        wrapped_env: RewardShapingWrapper instance
        risk_advisor: Risk advisor for penalty calculation
        penalty_weight: Weight for risk penalties
        
    Returns:
        Callback that can be used during training
    """
    return FunctionalRiskPenaltyCallback(wrapped_env, risk_advisor, penalty_weight)