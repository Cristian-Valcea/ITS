"""
Cooldown Trading Wrapper

Enforces minimum time between trades to prevent over-trading
and encourage more deliberate trading decisions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import logging


class CooldownWrapper(gym.Wrapper):
    """
    Enforces a cooldown period after each trade.
    
    During cooldown, only HOLD actions (action=1) are allowed.
    Any attempt to trade during cooldown is converted to HOLD.
    
    This wrapper helps prevent:
    - Rapid-fire trading that leads to excessive turnover
    - Ping-ponging between positions
    - Noise-driven micro-adjustments
    """
    
    def __init__(self, env: gym.Env, cooldown_steps: int = 0):
        """
        Initialize cooldown wrapper.
        
        Args:
            env: Base trading environment
            cooldown_steps: Number of steps to wait after a trade (0 = disabled)
        """
        super().__init__(env)
        self.cooldown_steps = cooldown_steps
        self.steps_since_trade = cooldown_steps + 1  # Start ready to trade
        self.logger = logging.getLogger("CooldownWrapper")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Track original and modified actions for debugging
        self.original_action = None
        self.modified_action = None
        
        if cooldown_steps > 0:
            self.logger.info(f"Cooldown wrapper active: {cooldown_steps} steps between trades")
        else:
            self.logger.info("Cooldown wrapper disabled (cooldown_steps=0)")
    
    def step(self, action):
        """Execute action with cooldown enforcement."""
        self.original_action = action
        
        # Check if we're in cooldown period
        in_cooldown = self.steps_since_trade < self.cooldown_steps
        
        if in_cooldown and action != 1:  # 1 = HOLD action
            # Force HOLD during cooldown
            self.modified_action = 1
            self.logger.debug(f"Step {self.steps_since_trade}/{self.cooldown_steps}: "
                            f"Forcing HOLD (original action: {action})")
            action = 1
        else:
            self.modified_action = action
        
        # Execute the (possibly modified) action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update cooldown counter
        if action != 1:  # Trade executed (not HOLD)
            self.steps_since_trade = 0
            self.logger.debug(f"Trade executed, cooldown reset")
        else:
            self.steps_since_trade += 1
        
        # Add cooldown info to step info
        info.update({
            'cooldown_active': in_cooldown,
            'steps_since_trade': self.steps_since_trade,
            'cooldown_steps_remaining': max(0, self.cooldown_steps - self.steps_since_trade),
            'action_modified_by_cooldown': self.original_action != self.modified_action
        })
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset cooldown state."""
        self.steps_since_trade = self.cooldown_steps + 1  # Start ready to trade
        self.original_action = None
        self.modified_action = None
        
        return self.env.reset(**kwargs)
    
    def is_in_cooldown(self) -> bool:
        """Check if currently in cooldown period."""
        return self.steps_since_trade < self.cooldown_steps
    
    def get_cooldown_info(self) -> Dict[str, Any]:
        """Get current cooldown status."""
        return {
            'cooldown_steps': self.cooldown_steps,
            'steps_since_trade': self.steps_since_trade,
            'in_cooldown': self.is_in_cooldown(),
            'steps_remaining': max(0, self.cooldown_steps - self.steps_since_trade)
        }