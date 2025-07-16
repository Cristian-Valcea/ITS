"""
Risk Environment Wrappers

Lightweight Gym wrappers that interface with the RiskManager
to provide risk-aware observations and reward shaping.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import logging

from ...risk.controls.risk_manager import RiskManager


class RiskObsWrapper(gym.ObservationWrapper):
    """
    Appends risk features to the observation space.
    
    Adds normalized risk metrics (volatility, drawdown, position size)
    to the observation vector for risk-aware decision making.
    """
    
    def __init__(self, env: gym.Env, risk_manager: RiskManager):
        """
        Initialize risk observation wrapper.
        
        Args:
            env: Base trading environment
            risk_manager: RiskManager instance for risk feature extraction
        """
        super().__init__(env)
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("RiskObsWrapper")
        
        # Extend observation space to include risk features
        # Original observation space + 5 risk features (volatility, drawdown, position_fraction, notional_exposure, position_size)
        original_shape = env.observation_space.shape
        
        # Handle both 1D (flattened) and 2D observation spaces
        if len(original_shape) == 1:
            # Already flattened - just add 5 features
            new_shape = (original_shape[0] + 5,)
        else:
            # Multi-dimensional - flatten first, then add 5 features
            flattened_size = np.prod(original_shape)
            new_shape = (flattened_size + 5,)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=new_shape,
            dtype=np.float32
        )
        
        self.logger.info(f"Extended observation space: {original_shape} → {new_shape} (added 5 risk features: volatility, drawdown, position_fraction, notional_exposure, position_size)")
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Append risk features to observation.
        
        Args:
            obs: Original observation from environment
            
        Returns:
            Extended observation with risk features
        """
        # Get risk features from RiskManager
        risk_features = self.risk_manager.get_risk_features()
        
        # Get current position and portfolio info from environment
        current_position = getattr(self.env, 'current_position', 0)
        max_position = getattr(self.env, 'max_position', 100)  # Reasonable default
        portfolio_value = getattr(self.env, 'portfolio_value', 100000.0)
        current_price = getattr(self.env, 'current_price', 100.0)  # Fallback price
        
        # Calculate enhanced position features
        position_fraction = current_position / max_position if max_position > 0 else 0.0
        notional_exposure = (current_position * current_price) / portfolio_value if portfolio_value > 0 else 0.0
        
        # Create enhanced risk feature vector (5 features)
        risk_vector = np.array([
            risk_features['volatility'],      # Current return volatility (0-1)
            risk_features['drawdown_pct'],    # Current drawdown percentage (0-1)
            position_fraction,                # Position as fraction of max position (-1 to 1)
            notional_exposure,                # Notional exposure as fraction of portfolio (0-1)
            float(current_position)           # Raw position size (for network to learn scaling)
        ], dtype=np.float32)
        
        # Flatten observation if needed (to handle multi-dimensional obs)
        if obs.ndim > 1:
            obs_flat = obs.flatten()
        else:
            obs_flat = obs
        
        # Concatenate flattened observation with risk features
        extended_obs = np.concatenate([obs_flat, risk_vector])
        
        return extended_obs


class VolatilityPenaltyReward(gym.RewardWrapper):
    """
    Applies volatility-based penalty to rewards.
    
    Subtracts λ * σ_return from the base reward to encourage
    stable trading strategies and discourage excessive volatility.
    """
    
    def __init__(self, env: gym.Env, risk_manager: RiskManager):
        """
        Initialize volatility penalty reward wrapper.
        
        Args:
            env: Base trading environment
            risk_manager: RiskManager instance for volatility penalty calculation
        """
        super().__init__(env)
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("VolatilityPenaltyReward")
        
        # Track penalty statistics
        self.total_base_reward = 0.0
        self.total_penalty = 0.0
        self.step_count = 0
        
        self.logger.info("VolatilityPenaltyReward wrapper initialized")
    
    def reward(self, reward: float) -> float:
        """
        Apply volatility penalty to base reward.
        
        Args:
            reward: Base reward from environment
            
        Returns:
            Modified reward with volatility penalty applied
        """
        # Get current portfolio state for RiskManager
        portfolio_value = getattr(self.env, 'portfolio_value', 100000.0)
        current_position = getattr(self.env, 'current_position', 0)
        
        # Calculate step return
        prev_value = getattr(self, '_prev_portfolio_value', portfolio_value)
        step_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        self._prev_portfolio_value = portfolio_value
        
        # Get current timestamp
        current_dt = getattr(self.env, 'current_dt', None)
        
        # Prepare info dict for RiskManager
        risk_info = {
            'timestamp': current_dt,
            'portfolio_value': portfolio_value,
            'position': current_position,
            'step_return': step_return
        }
        
        # Get risk assessment from RiskManager
        risk_result = self.risk_manager.step(risk_info)
        volatility_penalty = risk_result['volatility_penalty']
        
        # Apply penalty to reward
        modified_reward = reward - volatility_penalty
        
        # Track statistics
        self.total_base_reward += reward
        self.total_penalty += volatility_penalty
        self.step_count += 1
        
        # Log significant penalties
        if volatility_penalty > 0.1:  # Log penalties > 0.1
            self.logger.debug(f"High volatility penalty: {volatility_penalty:.4f} "
                            f"(base_reward: {reward:.4f}, final: {modified_reward:.4f})")
        
        # Check for termination
        if risk_result['should_terminate']:
            self.logger.warning("Risk-based episode termination triggered")
            # Environment will handle termination in next step
        
        return modified_reward
    
    def step(self, action):
        """Override step to handle episode completion."""
        obs, reward, done, truncated, info = super().step(action)
        
        # If episode is ending, process curriculum advancement
        if done or truncated:
            final_portfolio_value = getattr(self.env, 'portfolio_value', 100000.0)
            episode_summary = self.risk_manager.episode_completed(final_portfolio_value)
            
            # Add curriculum info to episode info
            info['risk_episode_summary'] = episode_summary
            
            # Log curriculum advancement
            if episode_summary.get('curriculum_advanced', False):
                stage_info = episode_summary.get('curriculum_stage', {})
                self.logger.info(f"🎓 Curriculum advanced to stage {stage_info.get('current_stage', '?')}: "
                               f"{stage_info.get('stage_name', 'unknown')}")
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset wrapper state for new episode.
        
        Returns:
            Initial observation and info dict
        """
        # Reset RiskManager for new episode
        obs, info = self.env.reset(**kwargs)
        
        # Get initial portfolio value
        initial_value = getattr(self.env, 'portfolio_value', 100000.0)
        self.risk_manager.reset_episode(initial_value)
        self._prev_portfolio_value = initial_value
        
        # Reset statistics
        self.total_base_reward = 0.0
        self.total_penalty = 0.0
        self.step_count = 0
        
        self.logger.debug(f"Episode reset - initial portfolio value: ${initial_value:,.2f}")
        
        return obs, info
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics for logging."""
        avg_base_reward = self.total_base_reward / self.step_count if self.step_count > 0 else 0.0
        avg_penalty = self.total_penalty / self.step_count if self.step_count > 0 else 0.0
        
        return {
            'total_base_reward': self.total_base_reward,
            'total_penalty': self.total_penalty,
            'avg_base_reward': avg_base_reward,
            'avg_penalty': avg_penalty,
            'penalty_ratio': self.total_penalty / abs(self.total_base_reward) if self.total_base_reward != 0 else 0.0,
            'step_count': self.step_count
        }