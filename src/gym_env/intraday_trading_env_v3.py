#!/usr/bin/env python3
"""
🎯 INTRADAY TRADING ENVIRONMENT V3 - STRUCTURAL REDESIGN
Clean environment implementation with V3 reward system integration
Philosophy: Cheapest strategy = do nothing unless genuine alpha
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
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gym_env.dual_reward_v3 import DualTickerRewardV3, RewardComponents

logger = logging.getLogger(__name__)

class IntradayTradingEnvV3(gym.Env):
    """
    Clean intraday trading environment with V3 structural reward redesign
    
    Features:
    - V3 reward system prevents cost-blind trading
    - Single-ticker for simplicity (NVDA focus)
    - Discrete action space: [0=Sell, 1=Hold, 2=Buy]
    - Risk-free adjusted rewards with embedded impact model
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(
        self,
        processed_feature_data: np.ndarray,  # Market features (steps, features)
        price_data: pd.Series,               # Price data aligned with features
        initial_capital: float = 100000.0,
        max_daily_drawdown_pct: float = 0.02,
        transaction_cost_pct: float = 0.0,     # No legacy cost - V3 handles via embedded impact
        max_episode_steps: Optional[int] = None,
        log_trades: bool = False,
        # V3 reward parameters
        risk_free_rate_annual: float = 0.05,
        base_impact_bp: float = 20.0,
        impact_exponent: float = 0.5,
        verbose: bool = False
    ):
        super().__init__()
        
        # Environment setup
        self.processed_feature_data = processed_feature_data.astype(np.float32)
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.transaction_cost_pct = transaction_cost_pct
        self.max_episode_steps = max_episode_steps or len(processed_feature_data)
        self.log_trades = log_trades
        self.verbose = verbose
        
        # Initialize V3 reward system
        self.reward_calculator = DualTickerRewardV3(
            risk_free_rate_annual=risk_free_rate_annual,
            base_impact_bp=base_impact_bp,
            impact_exponent=impact_exponent,
            verbose=verbose
        )
        
        # Validate data alignment
        if len(self.processed_feature_data) != len(self.price_data):
            raise ValueError(f"Feature data ({len(self.processed_feature_data)}) and price data ({len(self.price_data)}) must have same length")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=Sell, 1=Hold, 2=Buy
        
        # Observation: features + position one-hot (short/flat/long) + drawdown
        n_features = self.processed_feature_data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features + 3 + 1,),  # +3 for position one-hot, +1 for drawdown
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.position_quantity = 0.0  # Number of shares held
        self.total_trade_value = 0.0  # Cumulative trade value for turnover tracking
        self.peak_portfolio_value = initial_capital
        self.trade_history = []
        
        logger.info(f"🎯 IntradayTradingEnvV3 initialized:")
        logger.info(f"   📊 Data: {len(self.processed_feature_data)} steps, {n_features} features")
        logger.info(f"   💰 Capital: ${initial_capital:,.0f}")
        logger.info(f"   🛡️ Max DD: {max_daily_drawdown_pct:.1%}")
        logger.info(f"   🎯 V3 reward: Risk-free baseline with embedded impact")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.position_quantity = 0.0
        self.total_trade_value = 0.0
        self.peak_portfolio_value = self.initial_capital
        self.trade_history = []
        
        # Reset V3 reward calculator
        self.reward_calculator.reset()
        
        # Return initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        if self.verbose:
            logger.info(f"🔄 Environment reset - Episode starting at step 0")
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        if self.current_step >= len(self.processed_feature_data) - 1:
            # Episode finished
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()
        
        # Store previous state for reward calculation
        prev_portfolio_value = self.portfolio_value
        prev_position = self.position_quantity
        
        # Apply hard inventory clamp during OFF periods
        action = self._apply_action_mask(action)
        
        # Execute action
        trade_value = self._execute_action(action)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Calculate V3 reward with current features for position decay
        current_features = self.processed_feature_data[self.current_step] if self.current_step < len(self.processed_feature_data) else None
        reward, reward_components = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            curr_portfolio_value=self.portfolio_value,
            nvda_trade_value=trade_value,
            msft_trade_value=0.0,  # Single ticker environment
            nvda_position=self.position_quantity,
            msft_position=0.0,
            nvda_price=self.price_data.iloc[self.current_step],
            msft_price=510.0,  # Dummy price for MSFT
            step=self.current_step,
            features=current_features
        )
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps - 1
        
        # Log trades if enabled
        if self.log_trades and abs(trade_value) > 1.0:
            self._log_trade(action, trade_value, reward, reward_components)
        
        # Move to next step
        self.current_step += 1
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'reward_breakdown': reward_components.to_dict(),
            'trade_value': trade_value,
            'v3_reward': reward
        })
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action_mask(self, action: int) -> int:
        """Apply hard inventory clamp during OFF periods - restrict to {HOLD, SELL} only"""
        
        # Get current alpha signal strength from features
        if self.current_step < len(self.processed_feature_data):
            current_features = self.processed_feature_data[self.current_step]
            if len(current_features) >= 14:
                off_period_indicator = current_features[13]  # OFF period indicator
                alpha_signal = current_features[12]  # Alpha signal
                
                # Hard clamp: if in OFF period (α ≈ 0), restrict to flatten-only actions
                if off_period_indicator > 0.5 or abs(alpha_signal) < 1e-6:
                    # During OFF periods, restrict actions:
                    # Action 0 = SELL, Action 1 = HOLD, Action 2 = BUY
                    
                    if self.position_quantity > 0:
                        # Have long position -> only allow SELL or HOLD
                        if action == 2:  # BUY attempted
                            action = 0  # Force SELL instead
                            if self.verbose:
                                logger.info(f"🚫 OFF period: BUY blocked → SELL (flatten long)")
                    elif self.position_quantity < 0:
                        # Have short position -> only allow BUY (to cover) or HOLD  
                        if action == 0:  # SELL attempted
                            action = 2  # Force BUY to cover short instead
                            if self.verbose:
                                logger.info(f"🚫 OFF period: SELL blocked → BUY (cover short)")
                    else:
                        # No position -> only allow HOLD
                        if action != 1:  # Any trade attempted
                            action = 1  # Force HOLD
                            if self.verbose:
                                logger.info(f"🚫 OFF period: Trade blocked → HOLD (stay flat)")
        
        return action
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return trade value"""
        
        current_price = self.price_data.iloc[self.current_step]
        
        # Calculate target position based on action (smaller size to let V3 breathe)
        POSITION_SIZE = 250.0  # Reduced from 1000 to avoid slamming impact model
        
        if action == 0:  # Sell/Short
            target_position = -POSITION_SIZE
        elif action == 1:  # Hold
            target_position = self.position_quantity  # No change
        else:  # action == 2, Buy/Long
            target_position = POSITION_SIZE
        
        # Calculate trade quantity and value
        trade_quantity = target_position - self.position_quantity
        
        # Apply ΔQ clamp: limit trade size to 15% NAV per step
        max_trade_nav_ratio = 0.15  # 15% NAV per step limit
        max_trade_value = self.portfolio_value * max_trade_nav_ratio
        max_trade_quantity = max_trade_value / current_price
        
        # Clamp trade quantity if it exceeds limit
        if abs(trade_quantity) > max_trade_quantity:
            clamped_quantity = np.sign(trade_quantity) * max_trade_quantity
            if self.verbose:
                logger.info(f"🔒 ΔQ clamp: {trade_quantity:.0f} → {clamped_quantity:.0f} shares (15% NAV limit)")
            trade_quantity = clamped_quantity
            # Recalculate target position after clamping
            target_position = self.position_quantity + trade_quantity
        
        trade_value = abs(trade_quantity * current_price)
        
        # Apply transaction costs
        transaction_cost = trade_value * self.transaction_cost_pct
        
        # Update cash and position
        self.cash -= (trade_quantity * current_price + transaction_cost)
        self.position_quantity = target_position
        self.total_trade_value += trade_value
        
        return trade_value
    
    def _update_portfolio_value(self):
        """Update portfolio value based on current positions"""
        current_price = self.price_data.iloc[self.current_step]
        position_value = self.position_quantity * current_price
        self.portfolio_value = self.cash + position_value
        
        # Update peak for drawdown calculation  
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to risk limits"""
        
        # Check drawdown limit
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if current_drawdown > self.max_daily_drawdown_pct:
                if self.verbose:
                    logger.warning(f"🛑 Episode terminated: Drawdown {current_drawdown:.2%} > {self.max_daily_drawdown_pct:.2%}")
                return True
        
        # Check for portfolio bankruptcy
        if self.portfolio_value <= 1000:  # Minimum portfolio threshold
            if self.verbose:
                logger.warning(f"🛑 Episode terminated: Portfolio value ${self.portfolio_value:.2f} too low")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Market features
        features = self.processed_feature_data[self.current_step].copy()
        
        # Position one-hot encoding (short/flat/long)
        position_onehot = np.zeros(3, dtype=np.float32)
        if self.position_quantity < -50:  # Short
            position_onehot[0] = 1.0
        elif self.position_quantity > 50:  # Long
            position_onehot[2] = 1.0
        else:  # Flat
            position_onehot[1] = 1.0
        
        # Add current drawdown as observable feature
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0
        drawdown_feature = np.array([current_drawdown], dtype=np.float32)
        
        # Concatenate features, position one-hot, and drawdown
        obs = np.concatenate([features, position_onehot, drawdown_feature]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        current_price = self.price_data.iloc[self.current_step] if self.current_step < len(self.price_data) else 0
        
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_quantity': self.position_quantity,
            'position_value': self.position_quantity * current_price,
            'current_price': current_price,
            'total_return': (self.portfolio_value / self.initial_capital - 1),
            'current_drawdown': (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0,
            'total_trade_value': self.total_trade_value
        }
    
    def _log_trade(self, action: int, trade_value: float, reward: float, components: RewardComponents):
        """Log trade details"""
        action_names = ['SELL', 'HOLD', 'BUY']
        current_price = self.price_data.iloc[self.current_step]
        
        trade_record = {
            'step': self.current_step,
            'action': action_names[action],
            'price': current_price,
            'trade_value': trade_value,
            'position': self.position_quantity,
            'portfolio': self.portfolio_value,
            'reward': reward,
            'components': components.to_dict()
        }
        
        self.trade_history.append(trade_record)
        
        if self.verbose:
            logger.info(f"Step {self.current_step:4d}: {action_names[action]:4s} @${current_price:6.2f} | "
                       f"Portfolio ${self.portfolio_value:8,.0f} | Reward {reward:8.2f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics"""
        if len(self.trade_history) == 0:
            return {}
        
        total_trades = len([t for t in self.trade_history if t['trade_value'] > 1.0])
        total_return = (self.portfolio_value / self.initial_capital - 1)
        
        stats = {
            'total_steps': self.current_step,
            'total_trades': total_trades,
            'total_return': total_return,
            'final_portfolio': self.portfolio_value,
            'max_drawdown': (self.peak_portfolio_value - min(t['portfolio'] for t in self.trade_history)) / self.peak_portfolio_value if self.trade_history else 0,
            'total_trade_value': self.total_trade_value,
            'v3_reward_stats': self.reward_calculator.get_stats()
        }
        
        return stats
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_price = self.price_data.iloc[self.current_step] if self.current_step < len(self.price_data) else 0
            print(f"Step {self.current_step:4d} | Price ${current_price:6.2f} | "
                  f"Position {self.position_quantity:6.0f} | Portfolio ${self.portfolio_value:8,.0f}")

def create_v3_environment(config: Dict[str, Any]) -> IntradayTradingEnvV3:
    """Factory function to create V3 environment from config"""
    
    return IntradayTradingEnvV3(
        processed_feature_data=config['processed_feature_data'],
        price_data=config['price_data'],
        initial_capital=config.get('initial_capital', 100000.0),
        max_daily_drawdown_pct=config.get('max_daily_drawdown_pct', 0.02),
        transaction_cost_pct=config.get('transaction_cost_pct', 0.0001),
        max_episode_steps=config.get('max_episode_steps', None),
        log_trades=config.get('log_trades', False),
        risk_free_rate_annual=config.get('risk_free_rate_annual', 0.05),
        base_impact_bp=config.get('base_impact_bp', 20.0),
        impact_exponent=config.get('impact_exponent', 0.5),
        verbose=config.get('verbose', False)
    )