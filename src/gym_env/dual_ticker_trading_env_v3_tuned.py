#!/usr/bin/env python3
"""
🎯 DUAL-TICKER TRADING ENVIRONMENT V3 - TUNED VERSION
V3 environment with tuned reward weights for increased trading activity

TUNING OBJECTIVE: Increase trading frequency while preserving core performance
- Uses DualTickerRewardV3Tuned with reduced hold bonus and ticket costs
- Identical to V3 environment except for reward system
- Designed for warm-start retraining from existing V3 model
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gym_env.dual_reward_v3_tuned import DualTickerRewardV3Tuned

logger = logging.getLogger(__name__)

class DualTickerTradingEnvV3Tuned(gym.Env):
    """
    V3 Dual-Ticker Trading Environment with tuned reward weights
    
    IDENTICAL TO V3 EXCEPT:
    - Uses DualTickerRewardV3Tuned instead of DualTickerRewardV3
    - Tuned hold_bonus_weight: 0.01 → 0.0005
    - Tuned ticket_cost_per_trade: 0.50 → 0.20
    
    OBJECTIVE: Encourage more trading activity without breaking core logic
    """
    
    metadata = {
        'render.modes': ['human'],
        'version': '3.0.0-tuned',
        'tuning_date': '2025-08-02',
        'base_version': 'v3_gold_standard_400k_20250802_202736'
    }
    
    def __init__(
        self,
        # Data parameters (SAME AS V3)
        processed_feature_data: np.ndarray,
        processed_price_data: np.ndarray,
        trading_days: np.ndarray,
        
        # Environment parameters (SAME AS V3)
        initial_capital: float = 100000,
        lookback_window: int = 50,
        max_episode_steps: int = 1000,
        max_daily_drawdown_pct: float = 0.02,
        max_position_size: int = 500,
        transaction_cost_pct: float = 0.0001,
        
        # Core V3 reward parameters (UNCHANGED)
        base_impact_bp: float = 68.0,
        impact_exponent: float = 0.5,
        risk_free_rate_annual: float = 0.05,
        
        # 🎯 TUNED REWARD WEIGHTS
        hold_bonus_weight: float = 0.001,  # MODERATE reduction from 0.01 (10x)
        ticket_cost_per_trade: float = 0.50,  # UNCHANGED from original
        
        # Other reward weights (UNCHANGED)
        downside_penalty_weight: float = 2.0,
        kelly_bonus_weight: float = 0.5,
        position_decay_weight: float = 0.1,
        turnover_penalty_weight: float = 0.05,
        size_penalty_weight: float = 0.02,
        action_change_penalty_weight: float = 0.005,
        
        # Alpha signal parameters (SAME AS V3)
        alpha_mode: str = "live_replay",  # Default to live replay for tuning
        alpha_strength: float = 0.1,
        alpha_persistence: float = 0.5,
        alpha_on_probability: float = 0.6,
        
        # Logging
        log_trades: bool = True,  # Enable for tuning analysis
        verbose: bool = True
    ):
        """
        Initialize V3 Tuned Environment
        
        Args:
            Same as V3 environment except for tuned reward weights
        """
        super().__init__()
        
        # Store tuning configuration
        self.version = "3.0.0-tuned"
        self.tuning_date = "2025-08-02"
        self.base_version = "v3_gold_standard_400k_20250802_202736"
        
        # Data validation (SAME AS V3)
        assert processed_feature_data.shape[1] == 26, f"Expected 26 features, got {processed_feature_data.shape[1]}"
        assert processed_price_data.shape[1] == 4, f"Expected 4 price columns, got {processed_price_data.shape[1]}"
        assert len(processed_feature_data) == len(processed_price_data) == len(trading_days), "Data length mismatch"
        
        # Store data
        self.feature_data = processed_feature_data
        self.price_data = processed_price_data
        self.trading_days = trading_days
        self.n_timesteps = len(self.feature_data)
        
        # Environment parameters (SAME AS V3)
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.max_episode_steps = max_episode_steps
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_position_size = max_position_size
        self.transaction_cost_pct = transaction_cost_pct
        
        # Initialize V3 TUNED reward system
        self.reward_system = DualTickerRewardV3Tuned(
            base_impact_bp=base_impact_bp,
            impact_exponent=impact_exponent,
            risk_free_rate_annual=risk_free_rate_annual,
            downside_penalty_weight=downside_penalty_weight,
            kelly_bonus_weight=kelly_bonus_weight,
            position_decay_weight=position_decay_weight,
            turnover_penalty_weight=turnover_penalty_weight,
            size_penalty_weight=size_penalty_weight,
            action_change_penalty_weight=action_change_penalty_weight,
            # 🎯 TUNED WEIGHTS
            hold_bonus_weight=hold_bonus_weight,
            ticket_cost_per_trade=ticket_cost_per_trade
        )
        
        # Alpha signal configuration
        self.alpha_mode = alpha_mode
        self.alpha_strength = alpha_strength
        self.alpha_persistence = alpha_persistence
        self.alpha_on_probability = alpha_on_probability
        
        # Logging
        self.log_trades = log_trades
        self.verbose = verbose
        
        # Action and observation spaces (IDENTICAL TO V3)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        if self.verbose:
            logger.info(f"🎯 V3 Tuned Environment initialized - v{self.version}")
            logger.info(f"   Hold bonus: {hold_bonus_weight} (reduced from 0.01)")
            logger.info(f"   Ticket cost: ${ticket_cost_per_trade} (reduced from $0.50)")
            logger.info(f"   Data: {self.n_timesteps:,} timesteps")
            logger.info(f"   Base version: {self.base_version}")
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action (IDENTICAL TO V3)"""
        nvda_action = action // 3
        msft_action = action % 3
        return nvda_action, msft_action
    
    def _get_observation(self) -> np.ndarray:
        """Get observation (IDENTICAL TO V3)"""
        if self.current_step < self.lookback_window:
            obs = np.zeros(26, dtype=np.float32)
            available_steps = self.current_step + 1
            obs[:available_steps*26//self.lookback_window] = self.feature_data[
                max(0, self.current_step-available_steps+1):self.current_step+1
            ].flatten()[:available_steps*26//self.lookback_window]
        else:
            obs = self.feature_data[self.current_step].astype(np.float32)
        
        return obs
    
    def _generate_alpha_signals(self) -> Tuple[float, float]:
        """Generate alpha signals (IDENTICAL TO V3)"""
        if self.alpha_mode == "real":
            return 0.0, 0.0
        
        elif self.alpha_mode == "persistent":
            if not hasattr(self, '_persistent_alpha'):
                self._persistent_alpha = (
                    np.random.uniform(-self.alpha_strength, self.alpha_strength),
                    np.random.uniform(-self.alpha_strength, self.alpha_strength)
                )
            return self._persistent_alpha
        
        elif self.alpha_mode == "piecewise":
            if np.random.random() < self.alpha_on_probability:
                return (
                    np.random.uniform(-self.alpha_strength, self.alpha_strength),
                    np.random.uniform(-self.alpha_strength, self.alpha_strength)
                )
            else:
                return 0.0, 0.0
        
        elif self.alpha_mode == "live_replay":
            return (
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1)
            )
        
        else:
            return 0.0, 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute environment step (IDENTICAL TO V3 EXCEPT REWARD CALCULATION)
        """
        # Decode action
        nvda_action, msft_action = self._decode_action(action)
        
        # Get current prices
        nvda_price = self.price_data[self.current_step, 0]
        msft_price = self.price_data[self.current_step, 2]
        
        # Calculate position changes
        target_nvda_pos = (nvda_action - 1) * self.max_position_size
        target_msft_pos = (msft_action - 1) * self.max_position_size
        
        nvda_trade = target_nvda_pos - self.nvda_position
        msft_trade = target_msft_pos - self.msft_position
        
        # Execute trades
        nvda_trade_cost = abs(nvda_trade) * nvda_price * self.transaction_cost_pct
        msft_trade_cost = abs(msft_trade) * msft_price * self.transaction_cost_pct
        
        self.cash -= nvda_trade * nvda_price + msft_trade * msft_price
        self.cash -= nvda_trade_cost + msft_trade_cost
        
        self.nvda_position = target_nvda_pos
        self.msft_position = target_msft_pos
        
        # Calculate portfolio value
        portfolio_value = (
            self.cash + 
            self.nvda_position * nvda_price + 
            self.msft_position * msft_price
        )
        
        # Generate alpha signals
        nvda_alpha, msft_alpha = self._generate_alpha_signals()
        
        # 🎯 CALCULATE REWARD USING TUNED SYSTEM
        reward_components = self.reward_system.calculate_reward(
            portfolio_value=portfolio_value,
            previous_portfolio_value=self.previous_portfolio_value,
            nvda_position=self.nvda_position,
            msft_position=self.msft_position,
            nvda_trade=nvda_trade,
            msft_trade=msft_trade,
            nvda_price=nvda_price,
            msft_price=msft_price,
            nvda_alpha=nvda_alpha,
            msft_alpha=msft_alpha,
            action=action,
            previous_action=self.previous_action
        )
        
        reward = reward_components.total_reward
        
        # Update state
        self.previous_portfolio_value = portfolio_value
        self.previous_action = action
        self.current_step += 1
        self.episode_step += 1
        
        # Check termination conditions (IDENTICAL TO V3)
        done = False
        
        if self.episode_step >= self.max_episode_steps:
            done = True
        
        if self.current_step >= self.n_timesteps - 1:
            done = True
        
        daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
        if daily_return < -self.max_daily_drawdown_pct:
            done = True
            if self.verbose:
                logger.warning(f"Episode terminated: Daily drawdown {daily_return:.2%} exceeded limit")
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dictionary (ENHANCED FOR TUNING ANALYSIS)
        info = {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'nvda_position': self.nvda_position,
            'msft_position': self.msft_position,
            'nvda_price': nvda_price,
            'msft_price': msft_price,
            'nvda_alpha': nvda_alpha,
            'msft_alpha': msft_alpha,
            'reward_components': reward_components.to_dict(),
            'episode_step': self.episode_step,
            'current_step': self.current_step,
            # 🎯 TUNING ANALYSIS
            'tuning_info': {
                'traded_this_step': nvda_trade != 0 or msft_trade != 0,
                'holding_action': action == 4,
                'position_change': nvda_trade != 0 or msft_trade != 0,
                'total_position_value': abs(self.nvda_position) * nvda_price + abs(self.msft_position) * msft_price
            }
        }
        
        return observation, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment (IDENTICAL TO V3)"""
        if seed is not None:
            np.random.seed(seed)
        
        max_start = max(0, self.n_timesteps - self.max_episode_steps - self.lookback_window)
        self.current_step = np.random.randint(self.lookback_window, max_start)
        
        self.cash = self.initial_capital
        self.nvda_position = 0
        self.msft_position = 0
        self.previous_portfolio_value = self.initial_capital
        self.previous_action = 4  # Hold, Hold
        self.episode_step = 0
        
        if hasattr(self, '_persistent_alpha'):
            delattr(self, '_persistent_alpha')
        
        return self._get_observation(), {}
    
    def render(self, mode: str = 'human') -> None:
        """Render environment (IDENTICAL TO V3)"""
        if mode == 'human':
            portfolio_value = (
                self.cash + 
                self.nvda_position * self.price_data[self.current_step, 0] + 
                self.msft_position * self.price_data[self.current_step, 2]
            )
            
            print(f"Step {self.episode_step}: Portfolio ${portfolio_value:,.2f} "
                  f"(NVDA: {self.nvda_position}, MSFT: {self.msft_position})")
    
    def close(self) -> None:
        """Clean up environment resources"""
        pass
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """
        Get summary of tuning changes and current state
        
        Returns:
            Dictionary with tuning information
        """
        return {
            'version': self.version,
            'tuning_date': self.tuning_date,
            'base_version': self.base_version,
            'reward_system_info': self.reward_system.get_tuning_info(),
            'environment_changes': {
                'observation_space': 'unchanged',
                'action_space': 'unchanged',
                'core_logic': 'unchanged',
                'reward_calculation': 'tuned weights only'
            }
        }

# Export for compatibility
ACTION_MAPPING = {
    0: "Short NVDA, Short MSFT",
    1: "Short NVDA, Hold MSFT", 
    2: "Short NVDA, Long MSFT",
    3: "Hold NVDA, Short MSFT",
    4: "Hold NVDA, Hold MSFT",
    5: "Hold NVDA, Long MSFT",
    6: "Long NVDA, Short MSFT",
    7: "Long NVDA, Hold MSFT",
    8: "Long NVDA, Long MSFT"
}