# src/gym_env/dual_ticker_trading_env.py
"""
Dual-Ticker Trading Environment (NVDA + MSFT)

Phase 1: No cross-asset risk; suitable for transfer-learning bootstrap.

PHILOSOPHY: Two independent position inventories (NVDA + MSFT)
NO portfolio intelligence (correlation, beta) yet
Focus: Prove basic dual-ticker mechanics work
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import logging
import torch
from typing import Optional, Dict, Any, Tuple, List

# Import turnover penalty calculator
from src.gym_env.components.turnover_penalty import TurnoverPenaltyCalculator


class DualTickerTradingEnv(gym.Env):
    """
    A Gymnasium-compatible environment for dual-ticker trading (NVDA + MSFT).
    
    **Phase 1 Implementation**:
    - Two independent position inventories
    - No cross-asset risk calculations
    - Simple P&L addition approach
    - 26-dimensional observation space
    - 9-action portfolio matrix
    
    **Observation Space**:
    [12 NVDA features + 1 NVDA position + 12 MSFT features + 1 MSFT position] = 26 dims
    
    **Action Space**:
    Discrete(9): 3x3 matrix of (NVDA_action, MSFT_action) combinations
    0=SELL_BOTH, 1=SELL_NVDA_HOLD_MSFT, ..., 8=BUY_BOTH
    
    **Reward Function**:
    Simple addition: NVDA_P&L + MSFT_P&L - Transaction_Costs
    """
    
    metadata = {'render_modes': ['human', 'logs'], 'render_fps': 1}
    
    # 9-action portfolio matrix
    ACTION_MAPPINGS = {
        0: (-1, -1),  # SELL_BOTH
        1: (-1,  0),  # SELL_NVDA_HOLD_MSFT
        2: (-1,  1),  # SELL_NVDA_BUY_MSFT
        3: ( 0, -1),  # HOLD_NVDA_SELL_MSFT
        4: ( 0,  0),  # HOLD_BOTH (neutral)
        5: ( 0,  1),  # HOLD_NVDA_BUY_MSFT
        6: ( 1, -1),  # BUY_NVDA_SELL_MSFT
        7: ( 1,  0),  # BUY_NVDA_HOLD_MSFT
        8: ( 1,  1),  # BUY_BOTH
    }
    
    # Greppable action constants (prevents AAPL/NVDA confusion)
    ACTION_SELL_BOTH = 0
    ACTION_SELL_NVDA_HOLD_MSFT = 1
    ACTION_SELL_NVDA_BUY_MSFT = 2
    ACTION_HOLD_NVDA_SELL_MSFT = 3
    ACTION_HOLD_BOTH = 4
    ACTION_HOLD_NVDA_BUY_MSFT = 5
    ACTION_BUY_NVDA_SELL_MSFT = 6
    ACTION_BUY_NVDA_HOLD_MSFT = 7
    ACTION_BUY_BOTH = 8
    
    ACTION_DESCRIPTIONS = {
        ACTION_SELL_BOTH: "SELL_BOTH", 
        ACTION_SELL_NVDA_HOLD_MSFT: "SELL_NVDA_HOLD_MSFT", 
        ACTION_SELL_NVDA_BUY_MSFT: "SELL_NVDA_BUY_MSFT",
        ACTION_HOLD_NVDA_SELL_MSFT: "HOLD_NVDA_SELL_MSFT", 
        ACTION_HOLD_BOTH: "HOLD_BOTH", 
        ACTION_HOLD_NVDA_BUY_MSFT: "HOLD_NVDA_BUY_MSFT",
        ACTION_BUY_NVDA_SELL_MSFT: "BUY_NVDA_SELL_MSFT", 
        ACTION_BUY_NVDA_HOLD_MSFT: "BUY_NVDA_HOLD_MSFT", 
        ACTION_BUY_BOTH: "BUY_BOTH"
    }
    
    def __init__(self,
                 nvda_data: NDArray[np.float32],  # [N, 12] market features
                 msft_data: NDArray[np.float32],  # [N, 12] market features
                 nvda_prices: pd.Series,          # NVDA close prices
                 msft_prices: pd.Series,          # MSFT close prices
                 trading_days: pd.DatetimeIndex,  # 🔧 SHARED INDEX
                 initial_capital: float = 100000.0,
                 tc_bp: float = 2.0,              # 🔧 TRANSACTION COST BASIS POINTS (wider spreads during earnings)
                 trade_penalty_bp: float = 5.0,   # 🔧 ADDITIONAL PENALTY FOR OVER-TRADING (aggressive friction)
                 turnover_bp: float = 1.0,        # 🔧 TURNOVER PENALTY (rolling 30-step window)
                 hold_action_bonus: float = 0.01, # 🔧 BONUS FOR HOLDING POSITIONS
                 action_repeat_penalty: float = 0.002, # 🔧 PENALTY FOR CHANGING ACTIONS
                 high_water_mark_reward: float = 0.001, # 🔧 HIGH-WATER MARK: Reward staying above peak
                 reward_scaling: float = 0.1,     # 🔧 REWARD SCALING (10x increase for better learning)
                 bar_size: str = '1min',          # 🔧 CONFIGURABLE BAR SIZE
                 max_daily_drawdown_pct: float = 0.02,  # Legacy parameter (kept for compatibility)
                 training_drawdown_pct: float = 0.07,    # 🔧 TRAINING: Allow 7% drawdown for exploration
                 evaluation_drawdown_pct: float = 0.02,  # 🔧 EVALUATION: Strict 2% for risk control
                 is_training: bool = True,               # 🔧 MODE FLAG: True=training, False=evaluation
                 log_trades: bool = True,
                 **kwargs):
        
        # 🔧 ASSERT TIMESTAMP ALIGNMENT (catches off-by-one immediately)
        assert nvda_prices.index.equals(trading_days), \
            f"NVDA prices index mismatch: {len(nvda_prices)} vs {len(trading_days)}"
        assert msft_prices.index.equals(trading_days), \
            f"MSFT prices index mismatch: {len(msft_prices)} vs {len(trading_days)}"
        assert len(nvda_data) == len(trading_days), \
            f"NVDA data length mismatch: {len(nvda_data)} vs {len(trading_days)}"
        assert len(msft_data) == len(trading_days), \
            f"MSFT data length mismatch: {len(msft_data)} vs {len(trading_days)}"
        
        # Store aligned data
        self.nvda_data = nvda_data
        self.msft_data = msft_data
        self.nvda_prices = nvda_prices
        self.msft_prices = msft_prices
        self.trading_days = trading_days
        
        # Environment parameters
        self.initial_capital = initial_capital
        self.max_daily_drawdown_pct = max_daily_drawdown_pct  # Legacy (kept for compatibility)
        self.training_drawdown_pct = training_drawdown_pct    # 🔧 TRAINING: 7% exploration limit
        self.evaluation_drawdown_pct = evaluation_drawdown_pct # 🔧 EVALUATION: 2% risk limit
        self.is_training = is_training                        # 🔧 MODE FLAG
        self.log_trades = log_trades
        
        # 🔧 CONFIGURABLE TRANSACTION COSTS
        self.tc_bp = tc_bp / 10000.0  # Convert basis points to decimal
        self.trade_penalty_bp = trade_penalty_bp / 10000.0  # Additional over-trading penalty
        self.turnover_bp = turnover_bp / 10000.0  # Turnover penalty
        self.hold_action_bonus = hold_action_bonus  # Bonus for holding
        self.action_repeat_penalty = action_repeat_penalty  # Penalty for action changes
        self.high_water_mark_reward = high_water_mark_reward  # 🔧 HIGH-WATER MARK reward coefficient
        
        # 🔧 CONFIGURABLE BAR SIZE & SCALING
        self.bar_size = bar_size
        self.bars_per_day = self._calculate_bars_per_day(bar_size)
        
        # 🔧 REWARD SCALING (tuned for single-ticker PPO)
        self.reward_scaling = reward_scaling
        
        # Observation space: 26 dimensions
        # [12 NVDA features + 1 NVDA position + 12 MSFT features + 1 MSFT position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        
        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)
        
        # State variables
        self.current_step = 0
        self.max_steps = len(trading_days) - 1
        
        # Position tracking
        self.nvda_position = 0      # -1, 0, 1
        self.msft_position = 0      # -1, 0, 1
        self.prev_nvda_position = 0
        self.prev_msft_position = 0
        
        # Portfolio tracking
        self.portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        self.total_trades = 0
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.DualTickerEnv")
        if self.log_trades:
            self.trade_log = []
        
        # 🔧 TURNOVER PENALTY CALCULATOR (rolling 30-step window)
        if self.turnover_bp > 0:
            # Create a quiet logger for turnover penalty (reduce spam)
            turnover_logger = logging.getLogger(f"{__name__}.TurnoverPenalty")
            turnover_logger.setLevel(logging.WARNING)  # Only show warnings/errors
            
            self.turnover_calculator = TurnoverPenaltyCalculator(
                portfolio_value_getter=lambda: self.portfolio_value,
                episode_length_getter=lambda: 30,  # 30-step rolling window
                target_ratio=0.02,  # 2% target turnover
                weight_factor=self.turnover_bp,  # Use turnover_bp as weight
                curve_sharpness=25.0,
                curve="sigmoid",
                logger=turnover_logger  # Use quiet logger
            )
            self.turnover_history = []  # Track position changes for rolling window
        else:
            self.turnover_calculator = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.nvda_position = 0
        self.msft_position = 0
        self.prev_nvda_position = 0
        self.prev_msft_position = 0
        
        # Reset portfolio tracking
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.total_trades = 0
        
        # 🔧 DAILY TRADE CAP TRACKING
        self.trade_count_today = 0
        self.current_trading_day = None
        
        # 🔧 ACTION TRACKING FOR PENALTIES/BONUSES
        self.prev_action = None
        
        # 🔧 RESET TURNOVER HISTORY
        if self.turnover_calculator:
            self.turnover_history = []
        
        if self.log_trades:
            self.trade_log = []
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "nvda_position": self.nvda_position,
            "msft_position": self.msft_position,
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
            "trading_day": self.trading_days[self.current_step]
        }
        
        return observation, info
    
    def step(self, action: int):
        """Execute one trading step"""
        
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has already ended")
        
        # Store previous positions for transaction cost calculation
        self.prev_nvda_position = self.nvda_position
        self.prev_msft_position = self.msft_position
        
        # Decode action
        nvda_action, msft_action = self._decode_action(action)
        
        # Update positions independently
        self.nvda_position = nvda_action
        self.msft_position = msft_action
        
        # Calculate P&L and costs
        nvda_pnl, msft_pnl, total_cost = self._calculate_reward_components()
        
        # Update portfolio value
        portfolio_change = nvda_pnl + msft_pnl - total_cost
        self.portfolio_value += portfolio_change
        
        # Track peak for drawdown calculation
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # Calculate reward with daily trade cap penalty
        total_reward = portfolio_change * self.reward_scaling
        
        # 🔧 DAILY TRADE CAP PENALTY (50 trades/day limit)
        if self.trade_count_today > 50:
            trade_cap_penalty = 0.005 * (self.trade_count_today - 50)  # 0.5¢ per extra trade on $100 stock
            total_reward -= trade_cap_penalty
        
        # 🔧 TURNOVER PENALTY (rolling 30-step window)
        turnover_penalty = 0.0
        if self.turnover_calculator:
            # Calculate position changes for this step
            nvda_change = abs(self.nvda_position - self.prev_nvda_position)
            msft_change = abs(self.msft_position - self.prev_msft_position)
            step_turnover = nvda_change + msft_change
            
            # Add to rolling history
            self.turnover_history.append(step_turnover)
            if len(self.turnover_history) > 30:  # Keep only last 30 steps
                self.turnover_history.pop(0)
            
            # Calculate penalty based on rolling window turnover
            total_turnover = sum(self.turnover_history)
            if total_turnover > 0:
                turnover_penalty = self.turnover_calculator.compute_penalty(total_turnover, self.current_step)
                total_reward += turnover_penalty  # turnover_penalty is already negative
        
        # 🔧 HOLD ACTION BONUS (reward for holding positions)
        if action == 4:  # HOLD_BOTH action
            total_reward += self.hold_action_bonus
        
        # 🔧 ACTION REPEAT PENALTY (discourage rapid action changes)
        if self.prev_action is not None and action != self.prev_action:
            total_reward -= self.action_repeat_penalty
        
        # 🔧 HIGH-WATER MARK REWARD (encourage making and keeping gains)
        if self.peak_portfolio_value > 0:  # Avoid division by zero
            equity_ratio = self.portfolio_value / self.peak_portfolio_value
            hwm_reward = self.high_water_mark_reward * (equity_ratio - 1.0)
            total_reward += hwm_reward
        
        # Update previous action
        self.prev_action = action
        
        # Check for termination conditions
        done = self._check_termination()
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation (if not done)
        observation = self._get_observation() if not done else np.zeros(26, dtype=np.float32)
        
        # Count trades
        trades_this_step = 0
        if abs(self.nvda_position - self.prev_nvda_position) > 0:
            self.total_trades += 1
            trades_this_step += 1
        if abs(self.msft_position - self.prev_msft_position) > 0:
            self.total_trades += 1
            trades_this_step += 1
        
        # 🔧 DAILY TRADE CAP TRACKING (Calendar-based reset)
        current_date = self.trading_days[self.current_step].date()
        if not hasattr(self, 'prev_date') or self.prev_date != current_date:
            # New trading day - reset counter
            self.trade_count_today = 0
            self.prev_date = current_date
        
        self.trade_count_today += trades_this_step
        
        # 🔧 DETAILED INFO DICT (saves re-computation for tests/tensorboard)
        info = {
            "nvda_pnl": nvda_pnl,
            "msft_pnl": msft_pnl,
            "transaction_costs": total_cost,  # For backward compatibility
            "total_cost": total_cost,
            "portfolio_change": portfolio_change,
            "nvda_position": self.nvda_position,
            "msft_position": self.msft_position,
            "prev_nvda_position": self.prev_nvda_position,
            "prev_msft_position": self.prev_msft_position,
            "action_description": self.ACTION_DESCRIPTIONS[action],
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "drawdown": (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value,
            "total_trades": self.total_trades,
            "trade_count_today": self.trade_count_today,  # 🔧 Daily trade tracking
            "turnover_penalty": turnover_penalty,  # 🔧 Turnover penalty tracking
            "step": self.current_step,
            "trading_day": self.trading_days[self.current_step] if self.current_step < len(self.trading_days) else None,
            "trading_days_index": self.trading_days  # Exposed for downstream alignment
        }
        
        # Log trade if enabled
        if self.log_trades and (abs(nvda_action - self.prev_nvda_position) > 0 or 
                               abs(msft_action - self.prev_msft_position) > 0):
            self._log_trade(action, info)
        
        # Gymnasium API compatibility: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False  # We don't use truncation in trading environments
        return observation, total_reward, terminated, truncated, info
    
    def _get_observation(self):
        """Build 26-dimensional observation vector"""
        if self.current_step >= len(self.nvda_data):
            # Return zeros if we've exceeded data length
            return np.zeros(26, dtype=np.float32)
            
        # Get market features for current timestep
        nvda_features = self.nvda_data[self.current_step]  # 12 dims
        msft_features = self.msft_data[self.current_step]  # 12 dims
        
        # Build observation: [NVDA_features, NVDA_position, MSFT_features, MSFT_position]
        observation = np.concatenate([
            nvda_features,           # 12 NVDA market features
            [self.nvda_position],    # 1 NVDA position (-1, 0, 1)
            msft_features,           # 12 MSFT market features  
            [self.msft_position]     # 1 MSFT position (-1, 0, 1)
        ])
        
        return observation.astype(np.float32)
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Convert action ID to (nvda_action, msft_action)"""
        if action not in self.ACTION_MAPPINGS:
            raise ValueError(f"Invalid action ID: {action}. Must be 0-8.")
        return self.ACTION_MAPPINGS[action]
    
    def _calculate_reward_components(self) -> Tuple[float, float, float]:
        """Calculate individual P&L components"""
        
        # Calculate transaction costs (always applies when positions change)
        total_cost = self._calculate_transaction_costs()
        
        if self.current_step == 0:
            # No price change on first step, but transaction costs still apply
            return 0.0, 0.0, total_cost
        
        # Price changes from previous step
        nvda_price_change = (self.nvda_prices.iloc[self.current_step] - 
                            self.nvda_prices.iloc[self.current_step-1])
        msft_price_change = (self.msft_prices.iloc[self.current_step] - 
                            self.msft_prices.iloc[self.current_step-1])
        
        # P&L from positions (position * price_change)
        nvda_pnl = self.nvda_position * nvda_price_change
        msft_pnl = self.msft_position * msft_price_change
        
        return nvda_pnl, msft_pnl, total_cost
    
    def _calculate_transaction_costs(self) -> float:
        """
        Calculate transaction costs based on position changes.
        
        FIXED: Now correctly charges based on position size changes,
        not just the stock price per trade.
        
        Includes both:
        1. Standard transaction costs (tc_bp)
        2. Over-trading penalty (trade_penalty_bp)
        
        Returns:
            float: Total transaction costs for this step
        """
        nvda_change = abs(self.nvda_position - self.prev_nvda_position)
        msft_change = abs(self.msft_position - self.prev_msft_position)
        
        # Standard transaction costs
        nvda_cost = nvda_change * abs(self.nvda_prices.iloc[self.current_step]) * self.tc_bp
        msft_cost = msft_change * abs(self.msft_prices.iloc[self.current_step]) * self.tc_bp
        
        # Additional over-trading penalty
        trade_penalty = (nvda_change + msft_change) * self.trade_penalty_bp * \
                       (abs(self.nvda_prices.iloc[self.current_step]) + abs(self.msft_prices.iloc[self.current_step])) / 2
        
        return nvda_cost + msft_cost + trade_penalty
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # End of data
        if self.current_step >= self.max_steps:
            return True
        
        # 🔧 ADAPTIVE DRAWDOWN LIMIT: Training vs Evaluation
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        drawdown_limit = self.training_drawdown_pct if self.is_training else self.evaluation_drawdown_pct
        mode_str = "TRAINING" if self.is_training else "EVALUATION"
        
        if current_drawdown > drawdown_limit:
            self.logger.warning(f"Episode terminated ({mode_str}): Drawdown {current_drawdown:.3f} > {drawdown_limit:.3f}")
            return True
        
        return False
    
    def set_training_mode(self, is_training: bool):
        """🔧 Switch between training and evaluation modes"""
        self.is_training = is_training
        mode_str = "TRAINING" if is_training else "EVALUATION"
        drawdown_limit = self.training_drawdown_pct if is_training else self.evaluation_drawdown_pct
        self.logger.info(f"🔧 Mode switched to {mode_str} (drawdown limit: {drawdown_limit:.1%})")
    
    def _log_trade(self, action: int, info: Dict):
        """Log trade details"""
        trade_record = {
            'step': self.current_step,
            'trading_day': info['trading_day'],
            'action': action,
            'action_description': info['action_description'],
            'nvda_position': info['nvda_position'],
            'msft_position': info['msft_position'],
            'nvda_pnl': info['nvda_pnl'],
            'msft_pnl': info['msft_pnl'],
            'total_cost': info['total_cost'],
            'portfolio_value': info['portfolio_value'],
            'drawdown': info['drawdown']
        }
        
        self.trade_log.append(trade_record)
        
        if len(self.trade_log) % 100 == 0:
            self.logger.info(f"Completed {len(self.trade_log)} trades, Portfolio: ${info['portfolio_value']:.2f}")
    
    def _calculate_bars_per_day(self, bar_size: str) -> int:
        """Calculate bars per trading day based on bar size
        
        Args:
            bar_size: Bar size string (e.g., '1min', '5min', '15min', '1h')
            
        Returns:
            Number of bars per 6.5-hour trading day
        """
        # 6.5 hours = 390 minutes per trading day
        trading_minutes_per_day = 390
        
        if bar_size == '1min':
            return 390
        elif bar_size == '5min':
            return 78  # 390 / 5
        elif bar_size == '15min':
            return 26  # 390 / 15
        elif bar_size == '30min':
            return 13  # 390 / 30
        elif bar_size == '1h':
            return 7   # 390 / 60 (rounded up for partial hour)
        else:
            # Parse custom intervals (e.g., '2min', '10min')
            import re
            match = re.match(r'(\d+)(min|h)', bar_size)
            if match:
                value, unit = int(match.group(1)), match.group(2)
                if unit == 'min':
                    # 🔧 REVIEWER FIX: Ensure 390 % bar_minutes == 0 for clean division
                    if trading_minutes_per_day % value != 0:
                        raise ValueError(f"Bar size {bar_size} does not divide evenly into 390-minute trading day. "
                                       f"Valid intervals: 1, 2, 3, 5, 6, 10, 13, 15, 26, 30, 39, 65, 78, 130, 195, 390 minutes")
                    return trading_minutes_per_day // value
                elif unit == 'h':
                    hour_minutes = value * 60
                    if trading_minutes_per_day % hour_minutes != 0:
                        raise ValueError(f"Bar size {bar_size} does not divide evenly into 6.5-hour trading day. "
                                       f"Valid hour intervals: 1h, 2h, 3h, 6h (note: 1h gives 6.5 bars)")
                    return trading_minutes_per_day // hour_minutes
            
            # Default fallback with warning
            self.logger.warning(f"Unknown bar_size '{bar_size}', defaulting to 1min (390 bars/day)")
            return 390
    
    def get_trade_log(self) -> List[Dict]:
        """Get complete trade log"""
        return self.trade_log if self.log_trades else []
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary"""
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        max_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
        return {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'peak_portfolio_value': self.peak_portfolio_value,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': self.total_trades,
            'final_nvda_position': self.nvda_position,
            'final_msft_position': self.msft_position,
            'total_steps': self.current_step
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            summary = self.get_portfolio_summary()
            print(f"\n=== Dual-Ticker Trading Environment ===")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"NVDA Position: {self.nvda_position}, MSFT Position: {self.msft_position}")
            print(f"Portfolio Value: ${summary['final_portfolio_value']:.2f}")
            print(f"Total Return: {summary['total_return_pct']:.2f}%")
            print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
            print(f"Total Trades: {summary['total_trades']}")
        
        elif mode == 'logs':
            return self.get_trade_log()
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'trade_log'):
            self.trade_log.clear()