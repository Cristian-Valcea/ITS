#!/usr/bin/env python3
"""
🚀 DUAL-TICKER TRADING ENVIRONMENT V3 - STAIRWAYS TO HEAVEN ENHANCED
V3 environment enhanced with Stairways to Heaven dual-lane controller and market regime intelligence

ENHANCEMENT OBJECTIVE: Reduce excessive holding (80%+ hold rate) through intelligent frequency control
- Integrates DualLaneController for trading frequency optimization
- Adds MarketRegimeDetector for context-aware regime intelligence
- Preserves 26-dimensional observation space (reviewer requirement)
- Controller modifies hold bonus based on current trading behavior and market regime
- All core V3 logic preserved with enhanced reward calculation only

STAIRWAYS TO HEAVEN V3.0 - PHASE 2 IMPLEMENTATION
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

# Import Stairways to Heaven components
from controller import DualLaneController
from market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

class DualTickerTradingEnvV3Enhanced(gym.Env):
    """
    V3 Dual-Ticker Trading Environment Enhanced with Stairways to Heaven Controller
    
    ENHANCEMENTS OVER V3:
    - DualLaneController: Fast/slow lane proportional control for trading frequency
    - MarketRegimeDetector: Z-score normalized regime intelligence 
    - Controller modifies hold bonus based on holding behavior and market regime
    - 26-dimensional observation space preserved (reviewer requirement)
    - All core V3 logic unchanged except reward calculation enhancement
    
    OBJECTIVE: Reduce 80%+ hold rate through intelligent frequency optimization
    while preserving V3's proven risk management and portfolio performance.
    """
    
    metadata = {
        'render.modes': ['human'],
        'version': '3.0.0-enhanced',
        'enhancement_date': '2025-08-03',
        'base_version': 'v3_gold_standard_400k_20250802_202736',
        'stairways_version': 'v3.0'
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
        
        # Base reward weights (SAME AS V3 TUNED)
        hold_bonus_weight: float = 0.015,  # Base value before controller enhancement (INCREASED for stronger signal)
        ticket_cost_per_trade: float = 0.50,
        downside_penalty_weight: float = 2.0,
        kelly_bonus_weight: float = 0.5,
        position_decay_weight: float = 0.1,
        turnover_penalty_weight: float = 0.05,
        size_penalty_weight: float = 0.02,
        action_change_penalty_weight: float = 0.005,
        
        # Alpha signal parameters (SAME AS V3)
        alpha_mode: str = "live_replay",
        alpha_strength: float = 0.1,
        alpha_persistence: float = 0.5,
        alpha_on_probability: float = 0.6,
        
        # 🚀 STAIRWAYS TO HEAVEN ENHANCEMENT PARAMETERS
        enable_controller: bool = True,  # Enable dual-lane controller
        enable_regime_detection: bool = True,  # Enable market regime intelligence
        controller_target_hold_rate: float = 0.65,  # Target 65% hold rate (down from 80%+)
        bootstrap_days: int = 50,  # Days for regime detector bootstrap
        
        # R5 FIX: Dynamic target parameters
        enable_dynamic_target: bool = False,  # Enable dynamic target adjustment
        dynamic_target_bounds: Tuple[float, float] = (0.4, 0.75),  # Target bounds [40%, 75%]
        
        # Logging
        log_trades: bool = True,
        verbose: bool = True
    ):
        """
        Initialize V3 Enhanced Environment with Stairways to Heaven components
        
        Args:
            Same as V3 environment plus Stairways enhancement parameters
        """
        super().__init__()
        
        # Store enhancement configuration
        self.version = "3.0.0-enhanced"
        self.enhancement_date = "2025-08-03"
        self.base_version = "v3_gold_standard_400k_20250802_202736"
        self.stairways_version = "v3.0"
        
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
        
        # Store base reward configuration
        self.base_hold_bonus_weight = hold_bonus_weight
        
        # Initialize V3 reward system (will be enhanced by controller)
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
            hold_bonus_weight=hold_bonus_weight,  # Base value, enhanced by controller
            ticket_cost_per_trade=ticket_cost_per_trade
        )
        
        # 🚀 INITIALIZE STAIRWAYS TO HEAVEN COMPONENTS
        self.enable_controller = enable_controller
        self.enable_regime_detection = enable_regime_detection
        self.controller_target_hold_rate = controller_target_hold_rate
        
        # R5 FIX: Dynamic target configuration
        self.enable_dynamic_target = enable_dynamic_target
        self.dynamic_target_bounds = dynamic_target_bounds
        
        # Initialize dual-lane controller
        if self.enable_controller:
            self.controller = DualLaneController(base_hold_bonus=hold_bonus_weight)
            if verbose:
                logger.info(f"🎛️ Dual-lane controller initialized (base_bonus={hold_bonus_weight})")
        else:
            self.controller = None
        
        # Initialize market regime detector  
        if self.enable_regime_detection:
            self.regime_detector = MarketRegimeDetector(bootstrap_days=bootstrap_days)
            if verbose:
                logger.info(f"📊 Market regime detector initialized ({bootstrap_days}-day bootstrap)")
        else:
            self.regime_detector = None
        
        # Regime intelligence state
        self.regime_bootstrapped = False
        self.current_regime_score = 0.0
        
        # Trading frequency tracking for controller
        self.recent_actions = []  # Track recent actions for hold rate calculation
        self.hold_rate_window = 100  # Window for hold rate calculation
        
        # Alpha signal configuration
        self.alpha_mode = alpha_mode
        self.alpha_strength = alpha_strength
        self.alpha_persistence = alpha_persistence
        self.alpha_on_probability = alpha_on_probability
        
        # Logging
        self.log_trades = log_trades
        self.verbose = verbose
        
        # Action and observation spaces (IDENTICAL TO V3 - REVIEWER REQUIREMENT)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),  # REVIEWER CRITICAL: 26-dim preserved
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        if self.verbose:
            logger.info(f"🚀 V3 Stairways Enhanced Environment initialized - v{self.version}")
            logger.info(f"   Base version: {self.base_version}")
            logger.info(f"   Stairways version: {self.stairways_version}")
            logger.info(f"   Controller enabled: {self.enable_controller}")
            logger.info(f"   Regime detection enabled: {self.enable_regime_detection}")
            logger.info(f"   Target hold rate: {self.controller_target_hold_rate:.1%}")
            logger.info(f"   Data: {self.n_timesteps:,} timesteps")
    
    def _bootstrap_regime_detector(self) -> bool:
        """
        Bootstrap market regime detector with historical data.
        
        Returns:
            bool: True if bootstrap successful, False otherwise
        """
        if not self.enable_regime_detection or self.regime_bootstrapped:
            return True
        
        try:
            # Extract NVDA and MSFT symbols for bootstrap
            symbols = ["NVDA", "MSFT"]
            
            # Attempt bootstrap with fallback
            success = self.regime_detector.bootstrap_from_history_with_fallback(
                symbols=symbols, 
                days=self.regime_detector.bootstrap_days
            )
            
            if success:
                self.regime_bootstrapped = True
                if self.verbose:
                    logger.info(f"✅ Regime detector bootstrapped successfully")
                    health = self.regime_detector.get_detector_health()
                    logger.info(f"   Bootstrap progress: {health['bootstrap_progress']:.1%}")
                    logger.info(f"   Source: {health['bootstrap_source']}")
            else:
                if self.verbose:
                    logger.warning(f"⚠️ Regime detector bootstrap failed - using neutral regime")
            
            return success
            
        except Exception as e:
            if self.verbose:
                logger.error(f"🚨 Regime detector bootstrap error: {e}")
            return False
    
    def _calculate_current_hold_rate(self) -> float:
        """
        Calculate current hold rate from recent actions.
        
        Returns:
            float: Current hold rate [0, 1], where 1.0 = 100% holding
        """
        if len(self.recent_actions) == 0:
            return 0.5  # Neutral assumption
        
        # Count hold actions (action 4 = Hold NVDA, Hold MSFT)
        hold_actions = sum(1 for action in self.recent_actions if action == 4)
        return hold_actions / len(self.recent_actions)
    
    def _calculate_hold_error(self) -> float:
        """
        Calculate hold error for controller input.
        
        Returns:
            float: Hold rate error [-1, 1] where:
                   Positive = holding too much (need more trading)
                   Negative = trading too much (need more holding)
        """
        current_hold_rate = self._calculate_current_hold_rate()
        
        # R5 FIX: Dynamic target adjustment with bounds clamping
        target_hold_rate = self.controller_target_hold_rate
        if self.enable_dynamic_target:
            # Adjust target based on regime score: target = 0.55 + 0.1 * score
            dynamic_target = 0.55 + 0.1 * self.current_regime_score
            # Clamp to bounds [0.4, 0.75] to prevent gate violations
            target_hold_rate = np.clip(dynamic_target, self.dynamic_target_bounds[0], self.dynamic_target_bounds[1])
        
        hold_error = target_hold_rate - current_hold_rate
        
        # Normalize to [-1, 1] range
        return np.clip(hold_error * 2.0, -1.0, 1.0)
    
    def _update_regime_intelligence(self) -> float:
        """
        Update market regime intelligence and return current regime score.
        
        Returns:
            float: Current regime score [-3, 3]
        """
        if not self.enable_regime_detection or not self.regime_bootstrapped:
            return 0.0  # Neutral regime
        
        try:
            # Calculate market metrics from current data
            current_idx = self.current_step
            
            if current_idx < 20:  # Need minimum data for calculations
                return 0.0
            
            # Extract recent price data for momentum/volatility calculation
            nvda_prices = self.price_data[max(0, current_idx-20):current_idx+1, 0]
            msft_prices = self.price_data[max(0, current_idx-20):current_idx+1, 2]
            
            # Calculate momentum (20-period rate of change)
            if len(nvda_prices) >= 20:
                nvda_momentum = (nvda_prices[-1] - nvda_prices[-20]) / nvda_prices[-20]
                msft_momentum = (msft_prices[-1] - msft_prices[-20]) / msft_prices[-20]
                momentum = (nvda_momentum + msft_momentum) / 2
            else:
                momentum = 0.0
            
            # Calculate volatility (rolling std of returns)
            if len(nvda_prices) >= 10:
                nvda_returns = np.diff(nvda_prices) / nvda_prices[:-1]
                msft_returns = np.diff(msft_prices) / msft_prices[:-1]
                nvda_vol = np.std(nvda_returns[-10:]) if len(nvda_returns) >= 10 else 0.0
                msft_vol = np.std(msft_returns[-10:]) if len(msft_returns) >= 10 else 0.0
                volatility = (nvda_vol + msft_vol) / 2
            else:
                volatility = 0.0
            
            # Calculate divergence (correlation breakdown)
            if len(nvda_prices) >= 10:
                nvda_returns = np.diff(nvda_prices) / nvda_prices[:-1]
                msft_returns = np.diff(msft_prices) / msft_prices[:-1]
                
                if len(nvda_returns) >= 10 and len(msft_returns) >= 10:
                    correlation = np.corrcoef(nvda_returns[-10:], msft_returns[-10:])[0, 1]
                    if not np.isnan(correlation):
                        divergence = 1.0 - abs(correlation)  # Higher divergence = less correlation
                    else:
                        divergence = 0.5  # Neutral
                else:
                    divergence = 0.5  # Neutral
            else:
                divergence = 0.5  # Neutral
            
            # Update regime detector with current market state
            regime_score = self.regime_detector.calculate_regime_score(
                momentum=momentum,
                volatility=volatility,
                divergence=divergence
            )
            
            self.current_regime_score = regime_score
            return regime_score
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ Regime intelligence update failed: {e}")
            return 0.0  # Neutral fallback
    
    def _enhance_hold_bonus_with_controller(self, base_hold_bonus: float) -> float:
        """
        Enhance hold bonus using dual-lane controller and regime intelligence.
        
        Args:
            base_hold_bonus (float): Base hold bonus from reward system
            
        Returns:
            float: Enhanced hold bonus accounting for trading frequency and market regime
        """
        if not self.enable_controller:
            return base_hold_bonus  # No enhancement
        
        try:
            # Calculate current hold error
            hold_error = self._calculate_hold_error()
            
            # Update regime intelligence
            regime_score = self._update_regime_intelligence()
            
            # Use controller to compute enhanced bonus
            enhanced_bonus = self.controller.compute_bonus(
                hold_error=hold_error,
                regime_score=regime_score
            )
            
            return enhanced_bonus
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ Controller enhancement failed: {e}")
            return base_hold_bonus  # Fallback to base bonus
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action (IDENTICAL TO V3)"""
        nvda_action = action // 3
        msft_action = action % 3
        return nvda_action, msft_action
    
    def _get_observation(self) -> np.ndarray:
        """Get observation (IDENTICAL TO V3 - REVIEWER REQUIREMENT: 26-dim preserved)"""
        if self.current_step < self.lookback_window:
            obs = np.zeros(26, dtype=np.float32)
            available_steps = self.current_step + 1
            obs[:available_steps*26//self.lookback_window] = self.feature_data[
                max(0, self.current_step-available_steps+1):self.current_step+1
            ].flatten()[:available_steps*26//self.lookback_window]
        else:
            obs = self.feature_data[self.current_step].astype(np.float32)
        
        # REVIEWER CRITICAL: Return exactly 26 dimensions (no regime features in observation)
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
        Execute environment step with Stairways to Heaven enhancements.
        
        Core V3 logic preserved with controller-enhanced reward calculation.
        """
        # Track recent actions for hold rate calculation
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.hold_rate_window:
            self.recent_actions.pop(0)  # Keep only recent window
        
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
        
        # Execute trades (IDENTICAL TO V3)
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
        
        # 🚀 STAIRWAYS ENHANCEMENT: Enhance hold bonus with controller
        base_hold_bonus = self.reward_system.hold_bonus_weight
        enhanced_hold_bonus = self._enhance_hold_bonus_with_controller(base_hold_bonus)
        
        # Temporarily update reward system with enhanced hold bonus
        original_hold_bonus = self.reward_system.hold_bonus_weight
        self.reward_system.hold_bonus_weight = enhanced_hold_bonus
        
        # Calculate reward using enhanced system
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
        
        # Restore original hold bonus weight
        self.reward_system.hold_bonus_weight = original_hold_bonus
        
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
        
        # Get next observation (REVIEWER CRITICAL: 26-dim preserved)
        observation = self._get_observation()
        
        # Enhanced info dictionary with Stairways intelligence
        current_hold_rate = self._calculate_current_hold_rate()
        hold_error = self._calculate_hold_error()
        
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
            # 🚀 STAIRWAYS TO HEAVEN INTELLIGENCE
            'stairways_info': {
                'controller_enabled': self.enable_controller,
                'regime_detection_enabled': self.enable_regime_detection,
                'regime_bootstrapped': self.regime_bootstrapped,
                'current_hold_rate': current_hold_rate,
                'target_hold_rate': self.controller_target_hold_rate,
                'hold_error': hold_error,
                'current_regime_score': self.current_regime_score,
                'base_hold_bonus': base_hold_bonus,
                'enhanced_hold_bonus': enhanced_hold_bonus,
                'hold_bonus_enhancement': enhanced_hold_bonus - base_hold_bonus,
                'recent_actions_count': len(self.recent_actions),
                'controller_health': self.controller.get_controller_health() if self.controller else None,
                'regime_detector_health': self.regime_detector.get_detector_health() if self.regime_detector else None
            },
            # Legacy tuning info (for compatibility)
            'tuning_info': {
                'traded_this_step': nvda_trade != 0 or msft_trade != 0,
                'holding_action': action == 4,
                'position_change': nvda_trade != 0 or msft_trade != 0,
                'total_position_value': abs(self.nvda_position) * nvda_price + abs(self.msft_position) * msft_price
            }
        }
        
        return observation, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment with Stairways initialization"""
        if seed is not None:
            np.random.seed(seed)
        
        # Core reset (IDENTICAL TO V3)
        max_start = max(0, self.n_timesteps - self.max_episode_steps - self.lookback_window)
        
        # Ensure valid range for randint
        if max_start <= self.lookback_window:
            # Not enough data for random start - use minimum valid position
            self.current_step = self.lookback_window
            if self.verbose:
                logger.warning(f"⚠️ Limited data: using fixed start position {self.current_step}")
        else:
            self.current_step = np.random.randint(self.lookback_window, max_start)
        
        self.cash = self.initial_capital
        self.nvda_position = 0
        self.msft_position = 0
        self.previous_portfolio_value = self.initial_capital
        self.previous_action = 4  # Hold, Hold
        self.episode_step = 0
        
        if hasattr(self, '_persistent_alpha'):
            delattr(self, '_persistent_alpha')
        
        # 🚀 STAIRWAYS RESET: Initialize controller and regime detection
        self.recent_actions = []
        self.current_regime_score = 0.0
        
        # Reset controller state
        if self.controller:
            self.controller.reset_state()
        
        # Bootstrap regime detector (once per environment lifetime)
        if not self.regime_bootstrapped:
            self._bootstrap_regime_detector()
        
        return self._get_observation(), {}
    
    def render(self, mode: str = 'human') -> None:
        """Enhanced render with Stairways information"""
        if mode == 'human':
            portfolio_value = (
                self.cash + 
                self.nvda_position * self.price_data[self.current_step, 0] + 
                self.msft_position * self.price_data[self.current_step, 2]
            )
            
            current_hold_rate = self._calculate_current_hold_rate()
            
            print(f"Step {self.episode_step}: Portfolio ${portfolio_value:,.2f} "
                  f"(NVDA: {self.nvda_position}, MSFT: {self.msft_position}) "
                  f"Hold Rate: {current_hold_rate:.1%} "
                  f"Regime: {self.current_regime_score:.2f}")
    
    def close(self) -> None:
        """Clean up environment resources"""
        pass
    
    def get_stairways_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of Stairways to Heaven enhancements.
        
        Returns:
            Dictionary with enhancement information and current state
        """
        controller_health = self.controller.get_controller_health() if self.controller else None
        regime_health = self.regime_detector.get_detector_health() if self.regime_detector else None
        
        return {
            'version': self.version,
            'enhancement_date': self.enhancement_date,
            'base_version': self.base_version,
            'stairways_version': self.stairways_version,
            'enhancement_status': {
                'controller_enabled': self.enable_controller,
                'regime_detection_enabled': self.enable_regime_detection,
                'regime_bootstrapped': self.regime_bootstrapped,
                'target_hold_rate': self.controller_target_hold_rate,
                'current_hold_rate': self._calculate_current_hold_rate(),
                'hold_error': self._calculate_hold_error(),
                'current_regime_score': self.current_regime_score
            },
            'component_health': {
                'controller': controller_health,
                'regime_detector': regime_health
            },
            'environment_preservation': {
                'observation_space': '26-dimensional (preserved)',
                'action_space': '9-discrete (preserved)',
                'core_logic': 'V3 unchanged',
                'reward_enhancement': 'Controller-modified hold bonus only'
            },
            'performance_metrics': {
                'recent_actions_tracked': len(self.recent_actions),
                'hold_rate_window': self.hold_rate_window,
                'base_hold_bonus': self.base_hold_bonus_weight,
                'bootstrap_days': self.regime_detector.bootstrap_days if self.regime_detector else None
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