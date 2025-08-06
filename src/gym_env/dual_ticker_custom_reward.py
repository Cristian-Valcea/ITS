#!/usr/bin/env python3
"""
🎯 DUAL-TICKER CUSTOM REWARD ENVIRONMENT
Clean shim that inherits V3Enhanced but replaces internal reward with RefinedRewardSystem.

DESIGN PRINCIPLE: Minimal invasive surgery
- Inherits ALL V3Enhanced features (fills, slippage, regime detection, etc.)
- Overrides ONLY the reward calculation in step()
- Preserves observation space, termination logic, vectorization compatibility
- Maintains diagnostics pipeline through info dict
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import logging

from .dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from .refined_reward_system import RefinedRewardSystem

logger = logging.getLogger(__name__)

class DualTickerTradingEnvV3CustomReward(DualTickerTradingEnvV3Enhanced):
    """
    Identical to V3Enhanced *except* internal reward replaced by RefinedRewardSystem
    
    INHERITANCE STRATEGY:
    - All V3Enhanced functionality preserved
    - Only step() method overridden to swap reward calculation
    - RefinedRewardSystem parameters configurable via refined_reward_kwargs
    - Component diagnostics maintained in info dict
    """
    
    def __init__(
        self,
        *args,
        refined_reward_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize with V3Enhanced base + RefinedRewardSystem overlay
        
        Args:
            *args, **kwargs: Passed to V3Enhanced parent
            refined_reward_kwargs: Parameters for RefinedRewardSystem
        """
        # Build RefinedRewardSystem with proven parameters from 10K probe FIRST
        default_rr_args = {
            'initial_capital': 10000.0,
            'pnl_epsilon': 750.0,           # PROVEN from probe
            'holding_alpha': 0.05,          # PROVEN from probe  
            'penalty_beta': 0.10,           # PROVEN from probe
            'exploration_coef': 0.05,       # PROVEN from probe
            'exploration_decay': 0.9999,    # PROVEN from probe
            'holding_lookback_k': 5,        # PROVEN from probe
            'holding_min_ret': 0.0,         # PROVEN from probe
            'verbose': True
        }
        
        # Override with user-supplied parameters
        if refined_reward_kwargs:
            default_rr_args.update(refined_reward_kwargs)
        
        # Initialize RefinedRewardSystem
        self._refined_reward = RefinedRewardSystem(**default_rr_args)
        
        # Track for diagnostics
        self._custom_reward_active = True
        self._episode_reward_components = []
        
        # Initialize parent V3Enhanced environment AFTER setting up reward system
        super().__init__(*args, **kwargs)
        
        logger.info("🎯 Custom Reward Environment initialized")
        logger.info(f"   RefinedRewardSystem parameters: {default_rr_args}")
        logger.info("   V3Enhanced features preserved: regime detection, controller, etc.")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step with RefinedRewardSystem replacing internal V3Enhanced reward
        
        OVERRIDE STRATEGY:
        1. Call parent step() to get obs, termination, info (discard internal reward)
        2. Extract state information from info dict
        3. Calculate custom reward using RefinedRewardSystem
        4. Inject reward components into info for diagnostics
        5. Return (obs, custom_reward, terminated, truncated, enhanced_info)
        """
        # Call parent step() - get everything except reward
        obs, internal_reward, terminated, truncated, info = super().step(action)
        
        # Extract state information for RefinedRewardSystem
        portfolio_value = info.get('portfolio_value', self.initial_capital)
        previous_portfolio_value = getattr(self, '_prev_portfolio_value', self.initial_capital)
        nvda_position = info.get('nvda_position', 0.0)
        msft_position = info.get('msft_position', 0.0)
        
        # Calculate drawdown percentage
        drawdown_pct = max(0, (self.initial_capital - portfolio_value) / self.initial_capital)
        
        # Calculate custom reward using RefinedRewardSystem
        reward_components = self._refined_reward.calculate_reward(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            nvda_position=nvda_position,
            msft_position=msft_position,
            action=action,
            drawdown_pct=drawdown_pct
        )
        
        # Use refined reward instead of internal reward
        custom_reward = reward_components.total_reward
        
        # Enhance info dict with reward diagnostics
        info.update({
            'refined_reward_components': reward_components.to_dict(),
            'internal_reward': internal_reward,  # Keep for comparison
            'custom_reward_active': True,
            'reward_system': 'RefinedRewardSystem'
        })
        
        # Store for next step
        self._prev_portfolio_value = portfolio_value
        
        # Track episode components for diagnostics
        self._episode_reward_components.append(reward_components)
        
        # Log episode summary on termination
        if terminated or truncated:
            self._log_episode_reward_summary()
            self._episode_reward_components = []  # Reset for next episode
        
        return obs, custom_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with RefinedRewardSystem state reset"""
        obs, info = super().reset(**kwargs)
        
        # Reset RefinedRewardSystem episode state
        self._refined_reward.reset_episode()
        self._prev_portfolio_value = self.initial_capital
        self._episode_reward_components = []
        
        # Add custom reward info
        info.update({
            'custom_reward_active': True,
            'reward_system': 'RefinedRewardSystem'
        })
        
        return obs, info
    
    def _log_episode_reward_summary(self):
        """Log episode reward component summary for diagnostics"""
        if not self._episode_reward_components:
            return
        
        # Calculate episode totals
        total_pnl_reward = sum(rc.normalized_pnl for rc in self._episode_reward_components)
        total_holding_bonus = sum(rc.holding_bonus for rc in self._episode_reward_components)
        total_penalty = sum(rc.smoothed_penalty for rc in self._episode_reward_components)
        total_exploration = sum(rc.exploration_bonus for rc in self._episode_reward_components)
        total_reward = sum(rc.total_reward for rc in self._episode_reward_components)
        
        episode_length = len(self._episode_reward_components)
        
        # Log summary every 50 episodes
        episode_count = getattr(self, '_episode_count', 0) + 1
        self._episode_count = episode_count
        
        if episode_count % 50 == 0:
            logger.info(f"📊 Episode {episode_count} Reward Summary:")
            logger.info(f"   Total Reward: {total_reward:.3f}")
            logger.info(f"   PnL Reward: {total_pnl_reward:.3f}")
            logger.info(f"   Holding Bonus: {total_holding_bonus:.4f}")
            logger.info(f"   Smoothed Penalty: {total_penalty:.3f}")
            logger.info(f"   Exploration Bonus: {total_exploration:.4f}")
            logger.info(f"   Episode Length: {episode_length} steps")
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics"""
        stats = self._refined_reward.get_stats()
        stats.update({
            'custom_reward_active': self._custom_reward_active,
            'reward_system': 'RefinedRewardSystem',
            'episode_count': getattr(self, '_episode_count', 0)
        })
        return stats