"""
Size Limit Trading Wrapper

Enforces position size constraints and risk limits to prevent
excessive concentration and leverage.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import logging


class SizeLimitWrapper(gym.Wrapper):
    """
    Enforces position size limits and risk constraints.
    
    This wrapper:
    - Limits maximum position size as % of portfolio
    - Prevents trades that would exceed risk limits
    - Converts forbidden trades to HOLD actions
    - Provides position size feedback in info dict
    
    This helps prevent:
    - Over-concentration in single positions
    - Excessive leverage
    - Portfolio risk limit breaches
    """
    
    def __init__(self, 
                 env: gym.Env, 
                 max_position_pct: float = 0.25,
                 max_portfolio_risk_pct: float = 0.05,
                 min_cash_reserve_pct: float = 0.10):
        """
        Initialize size limit wrapper.
        
        Args:
            env: Base trading environment
            max_position_pct: Maximum position size as % of portfolio (0.25 = 25%)
            max_portfolio_risk_pct: Maximum portfolio risk as % of NAV (0.05 = 5%)
            min_cash_reserve_pct: Minimum cash reserve as % of portfolio (0.10 = 10%)
        """
        super().__init__(env)
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.logger = logging.getLogger("SizeLimitWrapper")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Track violations for monitoring
        self.violation_count = 0
        self.total_steps = 0
        
        self.logger.info(f"Size limits: position={max_position_pct:.1%}, "
                        f"risk={max_portfolio_risk_pct:.1%}, "
                        f"cash_reserve={min_cash_reserve_pct:.1%}")
    
    def step(self, action):
        """Execute action with size limit enforcement."""
        self.total_steps += 1
        original_action = action
        
        # Get current environment state before action
        current_obs = self.env.get_current_observation() if hasattr(self.env, 'get_current_observation') else None
        current_position = getattr(self.env, 'current_position', 0)
        portfolio_value = getattr(self.env, 'portfolio_value', getattr(self.env, 'initial_capital', 100000))
        current_price = getattr(self.env, 'current_price', 100.0)
        
        # Check if proposed action violates size limits
        action_valid, violation_reason = self._validate_action(
            action, current_position, portfolio_value, current_price
        )
        
        if not action_valid:
            # Convert to HOLD action
            action = 1
            self.violation_count += 1
            self.logger.debug(f"Size limit violation: {violation_reason}. "
                            f"Converting action {original_action} → HOLD")
        
        # Execute the (possibly modified) action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add size limit info to step info
        info.update({
            'size_limit_violated': not action_valid,
            'violation_reason': violation_reason if not action_valid else None,
            'action_modified_by_size_limit': original_action != action,
            'violation_rate': self.violation_count / self.total_steps if self.total_steps > 0 else 0.0,
            'position_size_pct': self._calculate_position_size_pct(current_position, portfolio_value, current_price),
            'max_position_pct': self.max_position_pct
        })
        
        return obs, reward, done, truncated, info
    
    def _validate_action(self, action: int, current_position: int, 
                        portfolio_value: float, current_price: float) -> Tuple[bool, str]:
        """
        Validate if action respects size limits.
        
        Returns:
            (is_valid, violation_reason)
        """
        # HOLD action is always valid
        if action == 1:
            return True, ""
        
        # Calculate new position after action
        if action == 0:  # Sell/Short
            new_position = current_position - 1
        elif action == 2:  # Buy/Long
            new_position = current_position + 1
        else:
            return True, ""  # Unknown action, let environment handle
        
        # Check position size limit
        new_position_size_pct = self._calculate_position_size_pct(
            new_position, portfolio_value, current_price
        )
        
        if abs(new_position_size_pct) > self.max_position_pct:
            return False, f"Position size {new_position_size_pct:.1%} > limit {self.max_position_pct:.1%}"
        
        # Check cash reserve requirement
        required_cash = abs(new_position) * current_price
        cash_reserve_pct = (portfolio_value - required_cash) / portfolio_value
        
        if cash_reserve_pct < self.min_cash_reserve_pct:
            return False, f"Cash reserve {cash_reserve_pct:.1%} < minimum {self.min_cash_reserve_pct:.1%}"
        
        # Check portfolio risk limit (simplified as position concentration)
        if abs(new_position_size_pct) > self.max_portfolio_risk_pct:
            return False, f"Portfolio risk {new_position_size_pct:.1%} > limit {self.max_portfolio_risk_pct:.1%}"
        
        return True, ""
    
    def _calculate_position_size_pct(self, position: int, portfolio_value: float, price: float) -> float:
        """Calculate position size as percentage of portfolio value."""
        if portfolio_value <= 0:
            return 0.0
        
        position_value = abs(position) * price
        return position_value / portfolio_value
    
    def reset(self, **kwargs):
        """Reset size limit tracking."""
        self.violation_count = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)
    
    def get_size_limit_stats(self) -> Dict[str, Any]:
        """Get size limit violation statistics."""
        return {
            'max_position_pct': self.max_position_pct,
            'max_portfolio_risk_pct': self.max_portfolio_risk_pct,
            'min_cash_reserve_pct': self.min_cash_reserve_pct,
            'violation_count': self.violation_count,
            'total_steps': self.total_steps,
            'violation_rate': self.violation_count / self.total_steps if self.total_steps > 0 else 0.0
        }