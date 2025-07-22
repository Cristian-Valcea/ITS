# src/gym_env/intraday_trading_env.py
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
from numpy.typing import NDArray  # for type hints
import pandas as pd
import logging
import json
import os
import time
import hashlib
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

# --------------------------------------------------------------------------- #
# λ-multiplier management: Sigmoid Risk Gain Schedule
# --------------------------------------------------------------------------- #
# λ = λ₀ · (1 + tanh(k · excess))
# This provides smooth scaling: gentle for small breaches, aggressive for large ones
SIGMOID_MAX_MULT     = 8.0       # Maximum multiplier (8× cap instead of brutal 20×)
SIGMOID_STEEPNESS    = 150.0     # Controls sigmoid steepness (k parameter)
LAMBDA_DECAY_FACTOR  = 0.99      # Time-based decay factor
ABS_MAX_LAMBDA       = 500000.0  # Absolute ceiling to prevent base_lambda explosions

# --------------------------------------------------------------------------- #
# Baseline Reset Logic: Escape DD Purgatory
# --------------------------------------------------------------------------- #
BASELINE_RESET_BUFFER_FACTOR = 0.5   # Reset when equity > baseline + soft_dd/2
BASELINE_RESET_FLAT_STEPS = 200      # Reset after N steps of flat performance



try:
    from .kyle_lambda_fill_simulator import KyleLambdaFillSimulator, FillPriceSimulatorFactory
    from ..risk.advanced_reward_shaping import AdvancedRewardShaper
    from .enhanced_reward_system import EnhancedRewardCalculator
    from .components.turnover_penalty import TurnoverPenaltyCalculator, TurnoverPenaltyFactory
    from .components.curriculum_scheduler import CurriculumScheduler
    from .institutional_safeguards import InstitutionalSafeguards
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from kyle_lambda_fill_simulator import KyleLambdaFillSimulator, FillPriceSimulatorFactory
    from enhanced_reward_system import EnhancedRewardCalculator
    from components.turnover_penalty import TurnoverPenaltyCalculator, TurnoverPenaltyFactory
    from components.curriculum_scheduler import CurriculumScheduler
    from institutional_safeguards import InstitutionalSafeguards
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from risk.advanced_reward_shaping import AdvancedRewardShaper

class IntradayTradingEnv(gym.Env):
    """
    A Gymnasium-compatible environment for simulating intraday trading.

    **Observation Space:**
    Consists of market features (e.g., normalized RSI, EMA, VWAP deviation, time features)
    PLUS current portfolio position (e.g., -1 for short, 0 for flat, 1 for long).
    If `lookback_window > 1`, observations are sequences of shape (lookback_window, num_market_features + 1).
    Otherwise, observations are flat arrays of shape (num_market_features + 1,).

    **Action Space:**
    Discrete: 0 (Sell/Short), 1 (Hold/Stay Flat), 2 (Buy/Long).
    (Could be extended to include position sizing, e.g., trade fixed quantity or % of capital).

    **Reward Function:**
    Change in portfolio value (realized P&L + unrealized P&L of open positions).
    Penalties for transaction costs, excessive turnover, or risk limit breaches.

    **Episode Termination:**
    - End of the trading data.
    - Max daily drawdown limit reached.
    - Optional: Hourly turnover cap breached (could terminate or penalize heavily).
    """
    metadata = {'render_modes': ['human', 'logs'], 'render_fps': 1}

    def __init__(self,
                 processed_feature_data: NDArray[np.float32], # Market features only
                 price_data: pd.Series, # Unscaled close prices for P&L, aligned with feature_data
                 initial_capital: float = 100000.0,
                 lookback_window: int = 1,
                 max_daily_drawdown_pct: float = 0.02,
                 hourly_turnover_cap: float = 5.0,
                 transaction_cost_pct: float = 0.001,
                 reward_scaling: float = 1.0,
                 max_episode_steps: int = None,
                 log_trades: bool = True,
                 turnover_penalty_factor: float = 0.0001, # REDUCED: Dynamic penalty per unit of excess turnover (% of portfolio)
                 position_sizing_pct_capital: float = 0.25, # Pct of capital to use for sizing (legacy)
                 equity_scaling_factor: float = 0.02, # k factor for equity-scaled sizing: shares = k * portfolio_value / price
                 trade_cooldown_steps: int = 5, # Number of steps to wait after a trade (INCREASED to stop ping-ponging)
                 terminate_on_turnover_breach: bool = False, # Terminate if turnover cap breached significantly
                 turnover_termination_threshold_multiplier: float = 2.0, # e.g. 2x the cap
                 # Enhanced turnover enforcement parameters
                 turnover_exponential_penalty_factor: float = 0.001, # REDUCED: Huber-shaped penalty factor
                 turnover_termination_penalty_pct: float = 0.05, # Additional penalty on termination (% of portfolio)
                 # New parameters for Kyle Lambda fill simulation
                 enable_kyle_lambda_fills: bool = True, # Enable Kyle Lambda fill price simulation
                 fill_simulator_config: dict = None, # Configuration for fill simulator
                 volume_data: pd.Series = None, # Optional volume data for better impact modeling
                 # Action change penalty to discourage ping-ponging
                 action_change_penalty_factor: float = 2.5, # L2 penalty factor for action changes (INCREASED to stop ping-ponging)
                 max_same_action_repeat: int = 3, # Maximum consecutive same actions before penalty (abort 0→2→0 spirals)
                 # Reward shaping parameters
                 turnover_bonus_threshold: float = 0.8, # Bonus when turnover < 80% of cap
                 turnover_bonus_factor: float = 0.001, # Bonus amount per step when under threshold
                 # Advanced reward shaping configuration
                 advanced_reward_config: Dict[str, Any] = None, # Configuration for advanced reward shaping
                 # Emergency fix parameters
                 use_emergency_reward_fix: bool = False, # Use simplified reward function
                 emergency_holding_bonus: float = 0.1, # Bonus for not trading (scaled by portfolio value)
                 emergency_transaction_cost_pct: float = 0.0001, # Reduced transaction cost for training
                 # Fixed turnover penalty system
                 use_turnover_penalty: bool = False, # Use fixed turnover penalty system
                 turnover_target_ratio: float = 0.02, # Target turnover ratio (dimensionless)
                 turnover_weight_factor: float = 0.02, # Penalty weight: % of NAV
                 turnover_curve_sharpness: float = 25.0, # Sigmoid curve sharpness
                 turnover_penalty_type: str = 'sigmoid', # 'sigmoid' or 'softplus'
                 # PPO-specific reward scaling
                 ppo_reward_scaling: bool = True, # Enable PPO-friendly reward scaling
                 ppo_scale_factor: float = 1000.0, # Divide rewards by (portfolio_value / this factor)
                 # Curriculum learning
                 curriculum: Dict[str, Any] = None, # Curriculum learning configuration
                 # Phase 1: Institutional safeguards
                 institutional_safeguards_config: Dict[str, Any] = None, # Phase 1 safeguards configuration
                 # Dynamic penalty lambda schedule
                 dynamic_lambda_schedule: bool = False, # Enable dynamic lambda scheduling
                 lambda_start: float = 10.0, # Starting lambda value
                 lambda_end: float = 75.0, # Ending lambda value
                 lambda_schedule_steps: int = 25000, # Steps over which to linearly increase lambda
                 global_step_counter: int = 0, # Global step counter for lambda scheduling
                 # DD baseline reset mechanism
                 dd_baseline_reset_enabled: bool = False, # Enable DD baseline reset
                 dd_recovery_threshold_pct: float = 0.005, # +0.5% equity recovery threshold
                 dd_reset_timeout_steps: int = 800, # Reset baseline after 800 steps regardless
                 purgatory_escape_threshold_pct: float = 0.015, # Purgatory escape threshold as % of baseline (1.5% default)
                 # Positive recovery bonus
                 recovery_bonus_enabled: bool = False, # Enable positive recovery bonus
                 recovery_bonus_amount: float = 0.2, # Bonus reward when excess DD < 0 (above baseline)
                 # Early-warning logger
                 early_warning_enabled: bool = False, # Enable early-warning logger
                 early_warning_threshold_pct: float = 0.005, # 0.5% excess DD threshold for warning
                 early_warning_duration_steps: int = 50 # Warn if above threshold for > 50 steps
                 ):
        """
        Args:
            processed_feature_data (np.ndarray): Market feature data (excluding position).
            price_data (pd.Series): Actual (unscaled) closing prices.
            initial_capital (float): Starting capital.
            lookback_window (int): Number of past timesteps in observation.
            max_daily_drawdown_pct (float): Max daily loss percentage.
            hourly_turnover_cap (float): Max (traded_value / capital) per hour.
            transaction_cost_pct (float): Cost per transaction.
            reward_scaling (float): Multiplier for P&L reward.
            max_episode_steps (int): Max steps per episode.
            log_trades (bool): Whether to log trades.
            turnover_penalty_factor (float): Factor to penalize excess turnover.
            position_sizing_pct_capital (float): Percentage of current capital to use for sizing new positions (legacy method).
            equity_scaling_factor (float): k factor for equity-scaled position sizing: shares = k * portfolio_value / price.
            trade_cooldown_steps (int): Minimum number of steps to wait before executing another trade.
            terminate_on_turnover_breach (bool): Whether to terminate the episode if hourly turnover cap is breached by a significant margin.
            turnover_termination_threshold_multiplier (float): Multiplier for `hourly_turnover_cap` to determine termination threshold.
            turnover_exponential_penalty_factor (float): Factor for quadratic penalty on excess turnover (makes penalty grow exponentially).
            turnover_termination_penalty_pct (float): Additional penalty applied on termination due to turnover breach (as % of portfolio value).
            enable_kyle_lambda_fills (bool): Whether to use Kyle Lambda fill price simulation instead of mid-price fills.
            fill_simulator_config (dict): Configuration for the fill price simulator.
            volume_data (pd.Series): Optional volume data for better market impact modeling.
            action_change_penalty_factor (float): L2 penalty factor for action changes to discourage ping-ponging.
            turnover_bonus_threshold (float): Threshold as fraction of hourly_turnover_cap for bonus (e.g., 0.8 = 80%).
            turnover_bonus_factor (float): Bonus amount per step when turnover is below threshold.
        """
        super().__init__()
        self.logger = logging.getLogger(f"RLTradingPlatform.Env.IntradayTradingEnv")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        self.TRADE_LOG_LEVEL = logging.DEBUG   # switch to INFO for verbose runs
        
        if not isinstance(processed_feature_data, np.ndarray):
            self.logger.error("processed_feature_data must be a NumPy array.")
            raise ValueError("processed_feature_data must be a NumPy array.")
        if not isinstance(price_data, pd.Series):
            self.logger.error("price_data must be a pandas Series.")
            raise ValueError("price_data must be a pandas Series.")
        if len(processed_feature_data) != len(price_data):
            self.logger.error(f"Length mismatch: features ({len(processed_feature_data)}) vs prices ({len(price_data)}).")
            raise ValueError("processed_feature_data and price_data must have the same length.")

        self.market_feature_data = processed_feature_data # Store market features separately
        self.price_data = price_data
        self.dates = price_data.index

        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.hourly_turnover_cap = hourly_turnover_cap
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else len(self.market_feature_data)
        self.log_trades_flag = log_trades
        self.turnover_penalty_factor = turnover_penalty_factor
        self.position_sizing_pct_capital = position_sizing_pct_capital
        self.equity_scaling_factor = equity_scaling_factor
        self.trade_cooldown_steps = trade_cooldown_steps
        self.terminate_on_turnover_breach = terminate_on_turnover_breach
        self.turnover_termination_threshold_multiplier = turnover_termination_threshold_multiplier
        self.turnover_exponential_penalty_factor = turnover_exponential_penalty_factor
        self.turnover_termination_penalty_pct = turnover_termination_penalty_pct
        self.action_change_penalty_factor = action_change_penalty_factor
        self.max_same_action_repeat = max_same_action_repeat
        self.turnover_bonus_threshold = turnover_bonus_threshold
        self.turnover_bonus_factor = turnover_bonus_factor
        
        # Emergency fix parameters
        self.use_emergency_reward_fix = use_emergency_reward_fix
        self.emergency_holding_bonus = emergency_holding_bonus
        self.emergency_transaction_cost_pct = emergency_transaction_cost_pct
        
        # Fixed turnover penalty system
        self.use_turnover_penalty = use_turnover_penalty
        self.turnover_target_ratio = turnover_target_ratio
        self.turnover_weight_factor = turnover_weight_factor
        self.turnover_curve_sharpness = turnover_curve_sharpness
        self.turnover_penalty_type = turnover_penalty_type
        
        # PPO-specific reward scaling
        self.ppo_reward_scaling = ppo_reward_scaling
        self.ppo_scale_factor = ppo_scale_factor
        
        # Curriculum learning
        self.curriculum_scheduler = None
        if curriculum:
            self.curriculum_scheduler = CurriculumScheduler(curriculum)
            # Update initial target ratio from curriculum
            if self.curriculum_scheduler.enabled:
                self.turnover_target_ratio = self.curriculum_scheduler.get_current_target_ratio()
                self.logger.info(f"🎓 Curriculum scheduler enabled - starting target: {self.turnover_target_ratio:.3f}")
        
        # Phase 1: Institutional Safeguards
        self.institutional_safeguards = None
        if institutional_safeguards_config:
            self.institutional_safeguards = InstitutionalSafeguards(institutional_safeguards_config)
            self.logger.info("🛡️ Phase 1 Institutional Safeguards enabled")
        
        # Dynamic penalty lambda schedule with parameter validation
        self.dynamic_lambda_schedule = dynamic_lambda_schedule
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_schedule_steps = lambda_schedule_steps
        self.global_step_counter = global_step_counter
        
        # Parameter sanity checks
        self._validate_lambda_parameters()
        
        # Advanced lambda scaling
        self.current_lambda_multiplier = 1.0  # Exponential multiplier for breaches
        self.lambda_reset_counter = 0  # Cool-down counter (unused in time-based decay)
        # Note: Removed step-based reset counters - now using time-based decay
        
        # Comprehensive lambda system logging
        if self.dynamic_lambda_schedule:
            self._log_lambda_system_startup()
        
        # DD baseline reset mechanism
        self.dd_baseline_reset_enabled = dd_baseline_reset_enabled
        self.dd_recovery_threshold_pct = dd_recovery_threshold_pct
        self.dd_reset_timeout_steps = dd_reset_timeout_steps
        self.purgatory_escape_threshold_pct = purgatory_escape_threshold_pct
        # DD baseline tracking variables (initialized in reset())
        self.dd_baseline_value = None
        self.dd_baseline_step = None
        self.steps_since_dd_baseline = 0
        if self.dd_baseline_reset_enabled:
            self.logger.info(f"🔄 Enhanced DD baseline reset enabled:")
            self.logger.info(f"   Method 1: Purgatory escape when equity > baseline + {self.purgatory_escape_threshold_pct:.1%}")
            self.logger.info(f"   Method 2: Flat timeout after {BASELINE_RESET_FLAT_STEPS} steps")
            self.logger.info(f"   Method 3: Legacy recovery at +{dd_recovery_threshold_pct:.1%} equity gain")
        
        # Positive recovery bonus
        self.recovery_bonus_enabled = recovery_bonus_enabled
        self.recovery_bonus_amount = recovery_bonus_amount
        if self.recovery_bonus_enabled:
            self.logger.info(f"💰 Recovery bonus enabled: +{recovery_bonus_amount} reward when above baseline")
        
        # Early-warning logger
        self.early_warning_enabled = early_warning_enabled
        self.early_warning_threshold_pct = early_warning_threshold_pct
        self.early_warning_duration_steps = early_warning_duration_steps
        # Early-warning tracking variables (initialized in reset())
        self.warning_excess_steps = 0
        self.last_warning_step = -1000  # Prevent spam on first warning
        if self.early_warning_enabled:
            self.logger.info(f"⚠️ Early-warning logger enabled: >{early_warning_threshold_pct:.1%} excess for >{early_warning_duration_steps} steps")
        
        # Initialize turnover tracking
        self.total_traded_value = 0.0
        self.episode_start_time = None
        self.episode_length_steps = 0
        
        # Initialize turnover penalty calculator (will be created in reset())
        self.turnover_penalty_calculator = None
        
        # Track consecutive drawdown steps to prevent infinite loops
        self.consecutive_drawdown_steps = 0
        self.max_consecutive_drawdown_steps = 200  # Increase limit during stabilization (was 50)
        
        # Log reward system status
        if self.use_turnover_penalty:
            self.logger.info(f"🎯 FIXED TURNOVER PENALTY SYSTEM ENABLED - Target: {self.turnover_target_ratio:.1%}, Weight: {self.turnover_weight_factor:.1%} NAV, Sharpness: {self.turnover_curve_sharpness}, Type: {self.turnover_penalty_type}")
        elif self.use_emergency_reward_fix:
            self.logger.info(f"🚨 EMERGENCY REWARD FIX ENABLED - Transaction cost: {self.emergency_transaction_cost_pct:.6f}, Holding bonus: {self.emergency_holding_bonus}")
        else:
            self.logger.info("Using standard reward system")
        
        # Kyle Lambda fill simulation setup
        self.enable_kyle_lambda_fills = enable_kyle_lambda_fills
        self.volume_data = volume_data
        
        if self.enable_kyle_lambda_fills:
            # Initialize Kyle Lambda fill simulator
            self.fill_simulator = FillPriceSimulatorFactory.create_kyle_lambda_simulator(
                config=fill_simulator_config
            )
            self.logger.info("Kyle Lambda fill price simulation enabled")
        else:
            self.fill_simulator = None
            self.logger.info("Using mid-price fills (Kyle Lambda simulation disabled)")
        
        # Initialize Advanced Reward Shaping
        if advanced_reward_config and advanced_reward_config.get('enabled', False):
            self.advanced_reward_shaper = AdvancedRewardShaper(advanced_reward_config)
            self.logger.info("🎯 Advanced Reward Shaping enabled")
        else:
            self.advanced_reward_shaper = None
            self.logger.info("Advanced Reward Shaping disabled")
        
        # Initialize Enhanced Reward Calculator
        enhanced_reward_config = advanced_reward_config.get('enhanced_reward_system', {}) if advanced_reward_config else {}
        if enhanced_reward_config.get('enabled', False):
            self.enhanced_reward_calculator = EnhancedRewardCalculator(enhanced_reward_config)
            self.logger.info("🚀 Enhanced Reward System enabled - addresses high-frequency reward noise")
        else:
            self.enhanced_reward_calculator = None
            self.logger.info("Enhanced Reward System disabled - using traditional P&L rewards")

        if not (0.01 <= self.position_sizing_pct_capital <= 1.0):
            self.logger.warning(f"position_sizing_pct_capital ({self.position_sizing_pct_capital}) is outside the recommended range [0.01, 1.0]. Clamping to 0.25.")
            self.position_sizing_pct_capital = 0.25
        
        if self.trade_cooldown_steps < 0:
            self.logger.warning(f"trade_cooldown_steps ({self.trade_cooldown_steps}) is negative. Setting to 0.")
            self.trade_cooldown_steps = 0
        
        if self.turnover_termination_threshold_multiplier <= 1.0 and self.terminate_on_turnover_breach:
            self.logger.warning(f"turnover_termination_threshold_multiplier ({self.turnover_termination_threshold_multiplier}) should be > 1.0. Setting to 2.0.")
            self.turnover_termination_threshold_multiplier = 2.0


        # Action Space
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1, 1: 0, 2: 1} # Sell, Hold, Buy -> Position -1, 0, 1

        # Observation Space: market features + 1 for current position
        if self.lookback_window > 1:
            if self.market_feature_data.ndim != 3: # (samples, window, features)
                raise ValueError("Market feature data shape mismatch for lookback_window > 1.")
            self.num_market_features = self.market_feature_data.shape[2]
            obs_shape = (self.lookback_window, self.num_market_features + 1) # +1 for position
        else: # lookback_window <= 1
            if self.market_feature_data.ndim != 2: # (samples, features)
                raise ValueError("Market feature data shape mismatch for lookback_window <= 1.")
            self.num_market_features = self.market_feature_data.shape[1]
            obs_shape = (self.num_market_features + 1,) # +1 for position
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        self.logger.info(f"Observation space: {self.observation_space} (Shape: {obs_shape})")
        self.logger.info(f"Action space: {self.action_space}")
        
        # Note: 2D observation shape (lookback_window, features) is intentional for time-series data
        # SB3 warning about "unconventional shape" can be safely ignored for trading environments

        # Internal state variables
        self.current_step = 0
        self.current_position = 0  # -1 short, 0 flat, 1 long
        self.current_capital = self.initial_capital # Cash component
        self.portfolio_value = self.initial_capital # Total value (cash + position_value)
        self.entry_price = 0.0
        self.position_quantity = 0.0 # Number of shares/contracts held
        
        self.daily_pnl = 0.0 # Realized P&L for the day
        self.start_of_day_portfolio_value = self.initial_capital
        self.last_date = None
        
        # Track net P&L after fees for reward calculation
        self.net_pnl_this_step = 0.0
        self.total_fees_this_step = 0.0
        
        # Episode-level metrics for monitoring and drift detection
        self.episode_total_fees = 0.0
        self.episode_total_turnover = 0.0
        self.episode_total_trades = 0
        self.episode_realized_pnl = 0.0
        self.episode_start_time = None
        self.episode_end_time = None

        self.hourly_traded_value = 0.0 # Sum of absolute value of trades in current hour (rolling window)
        self.trades_this_hour = [] # Stores (timestamp, trade_value, shares_traded)
        self.steps_since_last_trade = 0 

        if self.log_trades_flag:
            self.trade_log = []
        self.portfolio_history = [] # To log portfolio value at each step
        
        # Action tracking for diagnostics
        from collections import Counter
        self.action_counter = Counter()  # Track action distribution
        self.position_history = []  # Track position changes for diagnostics
        self.previous_action = 1  # Initialize to HOLD action (1) for first step

        self.reset()


    def _get_observation(self) -> NDArray[np.float32]:
        """
        Constructs the observation: market features + current position.
        
        CRITICAL: Ensure market_feature_data was properly lagged during preprocessing
        to prevent look-ahead bias. All indicators (RSI, EMA, etc.) at time t should
        use only data from times <= t, not future information.
        """
        # Add boundary protection for market feature data access
        if self.current_step >= len(self.market_feature_data):
            self.logger.warning(f"🚨 BOUNDARY PROTECTION: Step {self.current_step} >= market data length {len(self.market_feature_data)}, using last available data")
            market_obs_part = self.market_feature_data[-1].astype(np.float32)  # Use last available data
        else:
            market_obs_part = self.market_feature_data[self.current_step].astype(np.float32)
        position_feature = float(self.current_position) # Current position (-1, 0, or 1)

        if self.lookback_window > 1:
            # Append position as a new feature to each step in the lookback window
            # market_obs_part shape: (lookback_window, num_market_features)
            # position_feature_reshaped shape: (lookback_window, 1)
            position_feature_reshaped = np.full((self.lookback_window, 1), position_feature, dtype=np.float32)
            obs = np.concatenate((market_obs_part, position_feature_reshaped), axis=1)
        else:
            # Append position to the flat feature vector
            # market_obs_part shape: (num_market_features,)
            obs = np.append(market_obs_part, position_feature).astype(np.float32)
        
        return obs

    def _calculate_emergency_reward(self, realized_pnl: float, transaction_cost: float, 
                                  position_changed: bool) -> float:
        """
        🚨 EMERGENCY REWARD FIX: Simplified reward function focused on profitability
        
        This function addresses the core issue of reward function pollution by:
        1. Focusing purely on P&L after transaction costs
        2. Providing holding bonus to discourage overtrading
        3. Using reduced transaction costs for training
        4. Scaling rewards appropriately
        
        Args:
            realized_pnl (float): Realized P&L from position changes
            transaction_cost (float): Transaction cost for this step
            position_changed (bool): Whether position changed this step
            
        Returns:
            float: Simple reward focused on profitability
        """
        # Use reduced transaction cost for training if emergency fix is enabled
        if self.use_emergency_reward_fix:
            # Recalculate transaction cost with reduced rate
            if position_changed:
                # Get current price and trade details
                current_price = self._get_current_price()
                trade_value = abs(getattr(self, 'shares_traded_last_step', 0)) * current_price
                transaction_cost = trade_value * self.emergency_transaction_cost_pct
        
        # Core reward: P&L after transaction costs
        core_reward = realized_pnl - transaction_cost
        
        # Holding bonus: encourage NOT trading
        holding_bonus = 0.0
        if not position_changed:
            # Scale holding bonus by portfolio value to maintain consistency
            holding_bonus = self.emergency_holding_bonus * (self.portfolio_value / self.initial_capital)
        
        # Simple reward calculation
        total_reward = core_reward + holding_bonus
        
        # Scale to reasonable magnitude (convert to cents)
        scaled_reward = total_reward * 100
        
        # Log emergency reward details
        if self.use_emergency_reward_fix:
            self.logger.debug(
                f"🚨 EMERGENCY REWARD: Step {self.current_step}, "
                f"P&L: ${realized_pnl:.4f}, Cost: ${transaction_cost:.4f}, "
                f"Core: ${core_reward:.4f}, Holding: ${holding_bonus:.4f}, "
                f"Total: ${total_reward:.4f}, Scaled: {scaled_reward:.4f}"
            )
        
        return scaled_reward

    def _initialize_turnover_penalty_calculator(self) -> None:
        """Initialize the turnover penalty calculator component."""
        # 🎯 FIXED: Now includes episode_length normalization (your correct approach)
        self.turnover_penalty_calculator = TurnoverPenaltyCalculator(
            portfolio_value_getter=lambda: self.portfolio_value,
            episode_length_getter=lambda: getattr(self, 'max_episode_steps', 390),
            target_ratio=getattr(self, 'turnover_target_ratio', 0.02),
            weight_factor=getattr(self, 'turnover_weight_factor', 0.02),
            curve_sharpness=getattr(self, 'turnover_curve_sharpness', 25.0),
            curve=getattr(self, 'turnover_penalty_type', 'sigmoid'),
            logger=self.logger
        )
        
        self.logger.info(f"🎯 Turnover penalty calculator initialized: {self.turnover_penalty_calculator}")

    def _calculate_sigmoid_multiplier(self, excess_pct: float) -> float:
        """
        Calculate lambda multiplier using sigmoid gain schedule:
        λ = λ₀ · (1 + tanh(k · excess))
        
        This provides smooth, continuous scaling:
        - Small breaches: gentle increase
        - Large breaches: aggressive scaling
        - Natural cap at SIGMOID_MAX_MULT without brutal cliff
        
        Args:
            excess_pct: Drawdown excess as decimal (e.g., 0.005 for 0.5%)
            
        Returns:
            Multiplier value between 1.0 and SIGMOID_MAX_MULT
        """
        import math
        
        if excess_pct <= 0:
            return 1.0  # No excess, no multiplier
        
        # Modified sigmoid formula for gentler scaling:
        # Use tanh(k · excess) directly, but with offset to start at 1.0
        # This gives more gradual scaling for small excesses
        tanh_value = math.tanh(SIGMOID_STEEPNESS * excess_pct)
        
        # Scale tanh output [0, 1] to [1, SIGMOID_MAX_MULT]
        # For small excess_pct, tanh ≈ k·excess_pct (linear approximation)
        # This gives gentler initial scaling
        multiplier = 1.0 + (SIGMOID_MAX_MULT - 1.0) * tanh_value
        
        return multiplier

    def _get_current_penalty_lambda(self, excess_pct: float = 0.0) -> float:
        """Calculate current penalty lambda with exponential scaling for breaches."""
        if not self.dynamic_lambda_schedule:
            # Use static lambda from institutional safeguards or fallback
            if self.institutional_safeguards:
                return self.institutional_safeguards.penalty_lambda
            return 2500.0
        
        # Calculate base lambda from linear schedule (0.0 to 1.0)
        progress = min(self.global_step_counter / self.lambda_schedule_steps, 1.0)
        base_lambda = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        
        # Apply time-based decay first (gradual forgiveness)
        old_multiplier = self.current_lambda_multiplier
        self.current_lambda_multiplier = max(self.current_lambda_multiplier * LAMBDA_DECAY_FACTOR, 1.0)
        
        # Calculate sigmoid multiplier based on current excess
        sigmoid_multiplier = self._calculate_sigmoid_multiplier(excess_pct)
        
        # Use the maximum of decayed multiplier and sigmoid multiplier
        # This ensures we respond to current risk while maintaining memory of past breaches
        self.current_lambda_multiplier = max(self.current_lambda_multiplier, sigmoid_multiplier)
        
        # Emit telemetry for multiplier changes
        if hasattr(self.logger, 'record'):
            self.logger.record('lambda/sigmoid_multiplier', sigmoid_multiplier)
            self.logger.record('lambda/excess_pct', excess_pct * 100)
            if sigmoid_multiplier > old_multiplier:
                self.logger.record('lambda/multiplier_growth', (sigmoid_multiplier - old_multiplier) / old_multiplier)
        
        # Log significant changes
        if abs(self.current_lambda_multiplier - old_multiplier) > 0.1:
            if self.current_lambda_multiplier > old_multiplier:
                self.logger.warning(
                    f"🌊 Sigmoid lambda scaling: {old_multiplier:.2f}x → {self.current_lambda_multiplier:.2f}x "
                    f"(excess: {excess_pct*100:.2f}%, sigmoid: {sigmoid_multiplier:.2f}x)"
                )
            elif old_multiplier > 2.0 and self.current_step % 500 == 0:
                self.logger.info(f"⏳ Lambda decay: {old_multiplier:.2f}x → {self.current_lambda_multiplier:.2f}x (cooling down)")
        
        # Apply absolute ceiling protection
        penalty_lambda = min(base_lambda * self.current_lambda_multiplier, ABS_MAX_LAMBDA)
        
        # Log if absolute ceiling is hit (this should be very rare)
        if penalty_lambda == ABS_MAX_LAMBDA and base_lambda * self.current_lambda_multiplier > ABS_MAX_LAMBDA:
            self.logger.error(
                "🚨 ABSOLUTE LAMBDA CEILING HIT: %.0f (base=%.1f × mult=%.2fx = %.0f) → CAPPED at %.0f",
                penalty_lambda, base_lambda, self.current_lambda_multiplier, 
                base_lambda * self.current_lambda_multiplier, ABS_MAX_LAMBDA
            )
        
        # Emit telemetry for TensorBoard monitoring
        if hasattr(self.logger, 'record'):
            self.logger.record('lambda/multiplier', self.current_lambda_multiplier)
            self.logger.record('lambda/base_lambda', base_lambda)
            self.logger.record('lambda/penalty_lambda', penalty_lambda)
            self.logger.record('lambda/ceiling_hit', 1.0 if penalty_lambda == ABS_MAX_LAMBDA else 0.0)
        
        return penalty_lambda
    
    def _validate_lambda_parameters(self) -> None:
        """Validate lambda schedule parameters and log warnings for suspicious values."""
        if not self.dynamic_lambda_schedule:
            return
            
        # Check for common configuration errors
        issues = []
        
        if self.lambda_start <= 0:
            issues.append(f"lambda_start={self.lambda_start} (should be > 0)")
            
        if self.lambda_end <= 0:
            issues.append(f"lambda_end={self.lambda_end} (should be > 0)")
            
        if self.lambda_schedule_steps <= 0:
            issues.append(f"lambda_schedule_steps={self.lambda_schedule_steps} (should be > 0)")
            
        if self.lambda_start > self.lambda_end:
            issues.append(f"lambda_start ({self.lambda_start}) > lambda_end ({self.lambda_end}) - decreasing schedule")
            
        # Check for unrealistic values
        if self.lambda_start > 100000:
            issues.append(f"lambda_start={self.lambda_start} seems very high (typical: 1000-10000)")
            
        if self.lambda_end > 100000:
            issues.append(f"lambda_end={self.lambda_end} seems very high (typical: 1000-10000)")
            
        if self.lambda_schedule_steps > 1000000:
            issues.append(f"lambda_schedule_steps={self.lambda_schedule_steps} seems very high (typical: 10000-100000)")
            
        # Log issues
        if issues:
            self.logger.warning("⚠️ LAMBDA PARAMETER ISSUES DETECTED:")
            for issue in issues:
                self.logger.warning(f"   • {issue}")
        else:
            self.logger.info("✅ Lambda parameters validated successfully")
    
    def _log_lambda_system_startup(self) -> None:
        """Log comprehensive lambda system configuration at startup."""
        self.logger.info("=" * 60)
        self.logger.info("🔧 LAMBDA PENALTY SYSTEM CONFIGURATION")
        self.logger.info("=" * 60)
        
        # Basic schedule parameters
        self.logger.info(f"📈 Dynamic Schedule: {self.lambda_start:.1f} → {self.lambda_end:.1f} over {self.lambda_schedule_steps:,} steps")
        
        # Calculate progression rate
        if self.lambda_schedule_steps > 0:
            total_change = self.lambda_end - self.lambda_start
            change_per_1k_steps = (total_change / self.lambda_schedule_steps) * 1000
            self.logger.info(f"📊 Progression Rate: {change_per_1k_steps:+.1f} per 1,000 steps")
        
        # Sigmoid scaling parameters
        self.logger.info(f"🌊 Sigmoid Risk Gain: λ = λ₀ · (1 + tanh({SIGMOID_STEEPNESS:.0f} · excess))")
        self.logger.info(f"🛡️ Sigmoid Cap: {SIGMOID_MAX_MULT:.1f}x maximum (smooth, not brutal)")
        self.logger.info(f"🚨 Absolute Lambda Ceiling: {ABS_MAX_LAMBDA:,.0f} (ultimate safety limit)")
        self.logger.info(f"⏳ Time Decay: {LAMBDA_DECAY_FACTOR:.3f} factor ({(1-LAMBDA_DECAY_FACTOR)*100:.1f}% per step)")
        
        # Calculate key timelines
        import math
        half_life = math.log(0.5) / math.log(LAMBDA_DECAY_FACTOR)
        # Calculate sigmoid behavior examples
        example_excesses = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1%, 2%
        sigmoid_examples = [self._calculate_sigmoid_multiplier(exc) for exc in example_excesses]
        
        self.logger.info(f"⏱️ Decay Half-Life: {half_life:.0f} steps ({SIGMOID_MAX_MULT:.1f}x → {SIGMOID_MAX_MULT/2:.1f}x)")
        self.logger.info(f"📈 Sigmoid Examples:")
        for exc, mult in zip(example_excesses, sigmoid_examples):
            self.logger.info(f"   {exc*100:.1f}% excess → {mult:.2f}x multiplier")
        
        # Current state
        progress = min(self.global_step_counter / self.lambda_schedule_steps, 1.0) if self.lambda_schedule_steps > 0 else 0
        current_base = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        self.logger.info(f"🎯 Current State: Step {self.global_step_counter:,} ({progress:.1%} progress)")
        self.logger.info(f"🎯 Current Base λ: {current_base:.1f} (multiplier: {self.current_lambda_multiplier:.2f}x)")
        self.logger.info(f"🎯 Effective λ: {current_base * self.current_lambda_multiplier:.1f}")
        
        self.logger.info("=" * 60)
    
    # Note: Removed _update_lambda_reset_counter - now using time-based decay in _get_current_penalty_lambda

    def _calculate_turnover_penalty(self) -> float:
        """
        🎯 NEW TURNOVER PENALTY SYSTEM: Uses modular TurnoverPenaltyCalculator component
        
        This system addresses overtrading by:
        1. Using a dedicated, well-tested component for penalty calculation
        2. Providing clean separation of concerns
        3. Supporting multiple penalty curve types and configurations
        4. Enabling easy testing and validation
        
        Returns:
            float: Turnover penalty (negative value)
        """
        if not self.use_turnover_penalty:
            self.logger.debug("🎯 Turnover penalty DISABLED - returning 0.0")
            return 0.0
        
        self.logger.debug(f"🎯 _calculate_turnover_penalty CALLED - Step {self.current_step}")
        
        # Initialize calculator if not already done
        if self.turnover_penalty_calculator is None:
            self.logger.info("🎯 Initializing turnover penalty calculator...")
            self._initialize_turnover_penalty_calculator()
        
        # Portfolio value is now dynamically retrieved - no manual update needed
        
        # Calculate penalty using rolling window (390 steps = 1 trading day)
        rolling_turnover_total = sum(self.rolling_turnover_window) if hasattr(self, 'rolling_turnover_window') else self.total_traded_value
        penalty = self.turnover_penalty_calculator.compute_penalty(rolling_turnover_total, step=self.current_step)
        
        # 🔧 PERFORMANCE: Reduce logging frequency (every 100 steps instead of 50)
        if self.current_step % 100 == 0 or (self.total_traded_value > 1000 and self.current_step % 200 == 0):
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"🎯 COMPONENT LOGGING - Step {self.current_step}, Total Traded: ${self.total_traded_value:.0f}")
                self.turnover_penalty_calculator.log_debug(self.total_traded_value, step=self.current_step, penalty=penalty)
        
        self.logger.debug(f"🎯 _calculate_turnover_penalty RETURNING: {penalty:.4f}")
        return penalty

    def _calculate_turnover_reward(self, realized_pnl: float, transaction_cost: float, 
                                 position_changed: bool) -> float:
        """
        🎯 TURNOVER-BASED REWARD SYSTEM + HOLD BONUS
        
        This replaces the emergency fix with a more sophisticated approach:
        1. Core P&L reward (after transaction costs)
        2. Smooth turnover penalty based on normalized trading frequency
        3. 🎁 HOLD BONUS: Reward staying flat (0.2% daily bonus)
        4. Proper scaling and logging
        
        Args:
            realized_pnl (float): Realized P&L from position changes
            transaction_cost (float): Transaction cost for this step
            position_changed (bool): Whether position changed this step
            
        Returns:
            float: Total reward including turnover penalty and hold bonus
        """
        # Core reward: P&L after transaction costs
        core_reward = realized_pnl - transaction_cost
        
        # Turnover penalty: smooth, normalized penalty
        turnover_penalty = self._calculate_turnover_penalty()
        
        # 🎁 HOLD BONUS: Reward for staying flat (not changing position)
        hold_bonus = 0.0
        if not position_changed and hasattr(self, '_last_action') and self._last_action == 1:  # Action 1 = HOLD
            # 0.2% daily bonus = 0.002 * NAV / 390 steps per day
            hold_bonus = 0.002 * self.portfolio_value / 390
        
        # Total reward
        total_reward = core_reward + turnover_penalty + hold_bonus
        
        # Scale to reasonable magnitude (convert to cents)
        scaled_reward = total_reward * 100
        
        # 🔧 CRITICAL: Clip rewards to prevent gradient explosion
        scaled_reward = np.clip(scaled_reward, -10000, 10000)
        
        # Enhanced logging for turnover system
        if self.current_step % 50 == 0:  # Log every 50 steps
            self.logger.debug(
                f"🎯 TURNOVER REWARD: Step {self.current_step}, "
                f"P&L: ${realized_pnl:.4f}, Cost: ${transaction_cost:.4f}, "
                f"Core: ${core_reward:.4f}, Turnover Penalty: ${turnover_penalty:.4f}, "
                f"Hold Bonus: ${hold_bonus:.4f}, "
                f"Total: ${total_reward:.4f}, Scaled: {scaled_reward:.4f}"
            )
        
        return scaled_reward

    def _get_current_price(self) -> float:
        """Get current mid-market price with boundary protection."""
        # Add boundary protection to prevent index out of bounds
        if self.current_step >= len(self.price_data):
            self.logger.warning(f"🚨 BOUNDARY PROTECTION: Step {self.current_step} >= data length {len(self.price_data)}, using last available price")
            return self.price_data.iloc[-1]  # Return last available price
        return self.price_data.iloc[self.current_step]
    
    def _compute_unrealized_pnl(self) -> float:
        """
        Compute unrealized P&L for current position.
        
        Returns:
            float: Unrealized P&L in dollars
        """
        if self.current_position == 0 or self.position_quantity == 0:
            return 0.0
        
        current_price = self._get_current_price()
        
        if self.current_position == 1:  # Long position
            return (current_price - self.entry_price) * self.position_quantity
        elif self.current_position == -1:  # Short position
            return (self.entry_price - current_price) * self.position_quantity
        
        return 0.0
    
    def _apply_transaction_fee(self, shares: float, price: float, fee_type: str = "") -> float:
        """Apply transaction fee and track for reward calculation."""
        # Use emergency transaction cost rate if emergency fix is enabled
        if self.use_emergency_reward_fix:
            fee_rate = self.emergency_transaction_cost_pct
        else:
            fee_rate = self.transaction_cost_pct
            
        fee_amount = fee_rate * shares * price
        self.current_capital -= fee_amount
        self.total_fees_this_step += fee_amount
        self.episode_total_fees += fee_amount  # Track episode-level fees
        
        # Log the fee
        if fee_type:
            fee_rate_display = fee_rate if self.use_emergency_reward_fix else self.transaction_cost_pct
            emergency_marker = "🚨 EMERGENCY " if self.use_emergency_reward_fix else ""
            self.logger.log(self.TRADE_LOG_LEVEL, f"🔍 {emergency_marker}FEE CALC ({fee_type}): Shares={shares:.2f}, Price={price:.4f}, Rate={fee_rate_display:.6f} -> Fee=${fee_amount:.4f}")
        
        return fee_amount
    
    def _get_fill_price(self, mid_price: float, trade_size: float, side: str) -> tuple:
        """
        Get realistic fill price using Kyle Lambda simulation or mid-price.
        
        Args:
            mid_price: Current mid-market price
            trade_size: Size of trade (number of shares)
            side: Trade side ("buy" or "sell")
        
        Returns:
            Tuple of (fill_price, impact_info)
        """
        if self.enable_kyle_lambda_fills and self.fill_simulator:
            # Get current volume if available
            current_volume = None
            if self.volume_data is not None and self.current_step < len(self.volume_data):
                current_volume = self.volume_data.iloc[self.current_step]
            
            # Calculate fill price with market impact
            fill_price, impact_info = self.fill_simulator.calculate_fill_price(
                mid_price=mid_price,
                trade_size=trade_size,
                side=side,
                current_volume=current_volume
            )
            
            return fill_price, impact_info
        else:
            # Use mid-price fill (original behavior)
            impact_info = {
                'mid_price': mid_price,
                'fill_price': mid_price,
                'total_impact_bps': 0.0,
                'kyle_lambda_enabled': False
            }
            return mid_price, impact_info

    def _update_portfolio_value(self):
        """Updates total portfolio value based on current position and price."""
        current_price = self._get_current_price()
        position_market_value = 0.0
        if self.current_position == 1: # Long
            position_market_value = self.position_quantity * current_price
        elif self.current_position == -1: # Short
            # Value of short position = entry_value - (current_price - entry_price) * quantity
            # Or, more simply: cash + (entry_price - current_price) * quantity (if entry_price was for short)
            # For P&L calculation, it's (entry_price - current_price) * quantity.
            # Here, we track total portfolio value.
            # If shorting gives cash: cash_at_entry = initial_cash + entry_price * quantity
            # current_liability = current_price * quantity
            # For simplicity, let's assume short position value is (entry_price - current_price) * quantity added to capital.
            # This needs careful accounting based on broker model.
            # A common way: Portfolio Value = Cash + Unrealized P&L of open short position
            # Unrealized P&L for short = (self.entry_price - current_price) * self.position_quantity
            position_market_value = (self.entry_price - current_price) * self.position_quantity if self.position_quantity > 0 else 0
        
        # Portfolio value calculation using mark-to-market approach:
        # - For longs: cash + market value of shares
        # - For shorts: cash + unrealized P&L (entry_price - current_price) * quantity
        # - For flat: just cash
        
        if self.current_position == 1: # Long
            self.portfolio_value = self.current_capital + (self.position_quantity * current_price)
        elif self.current_position == -1: # Short
            # Mark-to-market: current_capital + unrealized P&L from short position
            # Unrealized P&L for short = (entry_price - current_price) * quantity
            self.portfolio_value = self.current_capital + (self.entry_price - current_price) * self.position_quantity
        else: # Flat
            self.portfolio_value = self.current_capital
            
    def _handle_new_day(self):
        current_date = self.dates[self.current_step].date()
        if self.last_date is None or current_date != self.last_date:
            previous_day_realized_pnl = self.daily_pnl # Store before reset
            self.logger.info(f"New trading day: {current_date}. Previous day REALIZED P&L: {previous_day_realized_pnl:.2f}")
            self.daily_pnl = 0.0 # Reset daily *realized* P&L counter
            
            # 🚨 DRAWDOWN FIX: Don't reset baseline if in drawdown (prevents counter reset)
            current_drawdown_pct = (self.start_of_day_portfolio_value - self.portfolio_value) / self.start_of_day_portfolio_value \
                                   if self.start_of_day_portfolio_value > 0 else 0
            
            if current_drawdown_pct <= self.max_daily_drawdown_pct:
                # Only reset baseline if not in drawdown
                self.start_of_day_portfolio_value = self.portfolio_value # Portfolio value at start of new day
                self.consecutive_drawdown_steps = 0  # Reset counter when baseline resets
                self.logger.info(f"Baseline reset to ${self.portfolio_value:.2f} (no drawdown)")
            else:
                # Keep existing baseline to maintain drawdown calculation
                self.logger.warning(f"Baseline NOT reset - in drawdown {current_drawdown_pct*100:.2f}% (baseline: ${self.start_of_day_portfolio_value:.2f})")
            
            self.last_date = current_date
            # Hourly turnover tracking resets implicitly via _update_hourly_trades if it spans days,
            # but should be explicitly cleared daily for cleaner accounting if desired.
            self.trades_this_hour.clear() 
            self.hourly_traded_value = 0.0
            
            # Reset daily turnover tracking for daily turnover penalty
            self.total_traded_value = 0.0
            self.daily_turnover_reset = True  # Flag for TensorBoard logging
            self.logger.info(f"Daily turnover reset to 0.0 for new trading day: {current_date}")

    def reset(self, seed=None, options=None):
        # Log action histogram from previous episode (if any)
        if hasattr(self, 'action_counter') and sum(self.action_counter.values()) > 0:
            total_actions = sum(self.action_counter.values())
            action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            histogram_str = ", ".join([f"{action_names.get(action, action)}: {count}" for action, count in sorted(self.action_counter.items())])
            self.logger.info(f"🎯 ACTION HISTOGRAM (Total: {total_actions}): {histogram_str}")
        
        super().reset(seed=seed)
        
        # Set reproducible state and save metadata
        if seed is not None:
            self.set_reproducible_state(seed=seed)
        
        # Save run metadata for reproducibility (first episode only)
        if not hasattr(self, '_metadata_saved'):
            run_id = f"episode_{int(time.time())}"
            self._save_run_metadata(run_id=run_id, additional_metadata={'episode_seed': seed})
            self._metadata_saved = True

        self.current_step = 0
        self.current_position = 0
        self.current_capital = self.initial_capital # Cash
        self.portfolio_value = self.initial_capital # Total value
        self.entry_price = 0.0
        self.position_quantity = 0.0 # Number of shares/units

        self.daily_pnl = 0.0 # Realized P&L
        self.start_of_day_portfolio_value = self.initial_capital
        self.last_date = None
        
        # Reset episode-level metrics
        self.episode_total_fees = 0.0
        self.episode_total_turnover = 0.0
        self.episode_total_trades = 0
        self.episode_realized_pnl = 0.0
        self.episode_start_time = self.dates[0] if len(self.dates) > 0 else None
        
        # Reset turnover tracking for new system
        self.total_traded_value = 0.0
        self.daily_turnover_reset = False  # Initialize reset flag
        
        # 🔧 ROLLING TURNOVER WINDOW: Track last 390 trades instead of daily reset
        from collections import deque
        self.rolling_turnover_window = deque(maxlen=390)  # 390 steps = 1 trading day
        self.episode_length_steps = len(self.price_data) if hasattr(self, 'price_data') else 0
        
        # Reset turnover penalty calculator for new episode
        if self.use_turnover_penalty:
            if self.turnover_penalty_calculator is not None:
                self.turnover_penalty_calculator.reset_history()
            # Reinitialize with correct episode length
            self._initialize_turnover_penalty_calculator()
        
        self.hourly_traded_value = 0.0
        self.trades_this_hour = []
        self.steps_since_last_trade = self.trade_cooldown_steps # Initialize to allow immediate trade if cooldown is >0

        if self.log_trades_flag:
            self.trade_log = []
        self.portfolio_history = [self.initial_capital]
        
        # Reset action counter for new episode
        from collections import Counter
        self.action_counter = Counter()
        self.position_history = []  # Reset position tracking
        self.previous_action = 1  # Reset to HOLD action for new episode
        self.same_action_count = 0  # Reset same action counter

        # Reset fill simulator
        if self.enable_kyle_lambda_fills and self.fill_simulator:
            self.fill_simulator.reset()
        
        # Reset advanced reward shaping
        if self.advanced_reward_shaper:
            self.advanced_reward_shaper.reset()
            # Initialize tracking variables for advanced reward shaping
            self.returns_history = []
            self.portfolio_values_history = [self.initial_capital]
            self.volatility_history = []
        
        # Reset enhanced reward calculator
        if self.enhanced_reward_calculator:
            self.enhanced_reward_calculator.reset()

        self._handle_new_day() 
        
        observation = self._get_observation()
        info = self._get_info()
        
        # Handle curriculum progression (if enabled)
        if hasattr(self, '_episode_total_reward') and self.curriculum_scheduler:
            stage_advanced = self.curriculum_scheduler.on_episode_end(self._episode_total_reward)
            if stage_advanced:
                # Update target ratio from curriculum
                new_target = self.curriculum_scheduler.get_current_target_ratio()
                self.turnover_target_ratio = new_target
                self.logger.info(f"🎓 Curriculum advanced! New target ratio: {new_target:.3f}")
                
                # Recreate turnover penalty calculator with new target
                if self.turnover_penalty_calculator:
                    self._initialize_turnover_penalty_calculator()
        
        # Reset episode reward tracking
        self._episode_total_reward = 0.0
        
        # Reset consecutive drawdown counter
        self.consecutive_drawdown_steps = 0
        
        # Initialize DD baseline reset tracking
        if self.dd_baseline_reset_enabled:
            self.dd_baseline_value = self.initial_capital
            self.dd_baseline_step = 0
            self.steps_since_dd_baseline = 0
            self.logger.debug(f"🔄 DD baseline initialized: ${self.dd_baseline_value:.2f} at step {self.dd_baseline_step}")
        
        # Initialize lambda scaling tracking
        if self.dynamic_lambda_schedule:
            self.current_lambda_multiplier = 1.0  # Reset multiplier each episode
            # Note: No reset counter needed - using time-based decay
        
        # Initialize early-warning tracking
        if self.early_warning_enabled:
            self.warning_excess_steps = 0
            self.last_warning_step = -1000  # Prevent spam on first warning
        
        self.logger.debug(f"Environment reset. Initial portfolio value: {self.initial_capital:.2f}")
        return observation, info

    def _update_hourly_trades(self, current_timestamp: pd.Timestamp):
        """Removes trades older than 1 hour for turnover calculation."""
        one_hour_ago = current_timestamp - pd.Timedelta(hours=1)
        # Filter out old trades and recalculate hourly_traded_value
        new_trades_this_hour = []
        self.hourly_traded_value = 0
        for ts, trade_val, shares in self.trades_this_hour:
            if ts >= one_hour_ago:
                new_trades_this_hour.append((ts, trade_val, shares))
                self.hourly_traded_value += trade_val
        self.trades_this_hour = new_trades_this_hour


    def step(self, action: int):
        # Increment global step counter for dynamic lambda scheduling
        if self.dynamic_lambda_schedule:
            self.global_step_counter += 1
        
        # Convert action to int if it's a single-element array or numpy scalar
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = int(action.item())
            else:
                raise ValueError(f"Action array has more than one element: {action}")
        elif isinstance(action, np.generic):  # Handles numpy scalar types
            action = int(action)
        assert isinstance(action, (int, np.integer)), f"Action must be int, got {type(action)}"
        
        # Track action for diagnostics and hold bonus
        self.action_counter[action] += 1
        self._last_action = action  # Track for HOLD bonus calculation
        
        desired_position_signal = self._action_map[action] # -1 (Sell), 0 (Hold), 1 (Buy)
        current_price = self._get_current_price()
        
        # Add boundary protection for timestamp access
        if self.current_step >= len(self.dates):
            self.logger.warning(f"🚨 BOUNDARY PROTECTION: Step {self.current_step} >= dates length {len(self.dates)}, episode should end")
            # Force episode termination by returning done=True
            obs = self._get_observation()
            return obs, -1000.0, True, True, {"boundary_exceeded": True}
        
        timestamp = self.dates[self.current_step]

        # End-of-day flat rule: Force position=0 at 15:55 to cut overnight risk
        if hasattr(timestamp, 'time') and timestamp.time() >= pd.Timestamp('15:55').time():
            if self.current_position != 0:
                self.logger.info(f"🕐 END-OF-DAY FLAT RULE: Forcing position to 0 at {timestamp.time()} "
                               f"(was {self.current_position})")
                desired_position_signal = 0  # Force flat position
        
        # Update fill simulator with current market data
        if self.enable_kyle_lambda_fills and self.fill_simulator:
            current_volume = None
            if self.volume_data is not None and self.current_step < len(self.volume_data):
                current_volume = self.volume_data.iloc[self.current_step]
            
            self.fill_simulator.update_market_data(
                price=current_price,
                volume=current_volume,
                timestamp=timestamp
            )

        # Store portfolio value before action
        portfolio_value_before_action = self.portfolio_value
        
        # Reset step-level tracking for reward calculation
        self.net_pnl_this_step = 0.0
        self.total_fees_this_step = 0.0
        
        # Update hourly turnover tracking every step (not just on trades)
        self._update_hourly_trades(timestamp)
        
        realized_pnl_this_step = 0.0
        transaction_executed = False
        trade_value_executed = 0.0 # Absolute monetary value of stock traded
        shares_traded_this_step = 0.0
        
        # Track shares traded for emergency reward calculation
        self.shares_traded_last_step = 0.0

        # --- Execute Trade ---
        # This logic assumes full capital deployment or fixed quantity per trade.
        # For simplicity, let's assume we trade a fixed quantity or go all-in/all-out.
        # If going all-in, quantity depends on capital and price.
        # Let's refine to a fixed quantity for now, e.g. 1 unit of underlying.
        # This needs to be configured or made more flexible (e.g. % of capital).
        # For now, let's assume self.position_quantity is how many shares we hold/short.
        # And a decision to buy/sell means changing this quantity.
        # If current_position is 0, and action is Buy (1), we buy, quantity becomes X.
        # If current_position is 1 (long X shares), and action is Sell (-1), we sell X, then short X. (Flip)
        # If current_position is 1 (long X shares), and action is Hold (0), we do nothing to quantity.
        # This is complex. Let's simplify: desired_position_signal is the *target state*.

        # --- Cooldown Check (before incrementing counter for strict timing) ---
        if self.steps_since_last_trade < self.trade_cooldown_steps and desired_position_signal != self.current_position:
            self.logger.log(self.TRADE_LOG_LEVEL, f"🕐 TRADE COOLDOWN: Step {self.current_step}, {self.steps_since_last_trade}/{self.trade_cooldown_steps} bars since last trade. Forcing HOLD (Action {action} → HOLD).")
            desired_position_signal = self.current_position # Force hold
            # Action itself is not changed, only its effect for changing position.
        
        # Increment cooldown counter AFTER the check for strict wall-clock timing
        self.steps_since_last_trade += 1

        if desired_position_signal != self.current_position:
            transaction_executed = True
            self.steps_since_last_trade = 0 # Reset cooldown counter
            
            # 1. Exit current position if any (calculate realized P&L)
            if self.current_position == 1: # Was long, now closing or flipping to short
                # Get fill price for selling
                exit_fill_price, exit_impact_info = self._get_fill_price(
                    mid_price=current_price,
                    trade_size=self.position_quantity,
                    side="sell"
                )
                
                realized_pnl_this_step = (exit_fill_price - self.entry_price) * self.position_quantity
                self.current_capital += (self.position_quantity * exit_fill_price) # Add sale proceeds to cash
                
                # Apply transaction fee and track for reward calculation
                fee_amount = self._apply_transaction_fee(self.position_quantity, exit_fill_price, "LONG EXIT")
                expected_fee = self.position_quantity * exit_fill_price * 0.001  # Expected: shares * price * 0.001
                self.logger.debug(f"Expected fee: ${expected_fee:.4f}")
                trade_value_executed += self.position_quantity * exit_fill_price
                shares_traded_this_step += self.position_quantity
                
                impact_bps = exit_impact_info.get('total_impact_bps', 0.0)
                self.logger.debug(f"Closed LONG: {self.position_quantity} @ {exit_fill_price:.4f} (mid: {current_price:.4f}, impact: {impact_bps:.1f}bps). Entry: {self.entry_price:.4f}. P&L: {realized_pnl_this_step:.2f}")
                
            elif self.current_position == -1: # Was short, now closing or flipping to long
                # Get fill price for buying to cover
                cover_fill_price, cover_impact_info = self._get_fill_price(
                    mid_price=current_price,
                    trade_size=self.position_quantity,
                    side="buy"
                )
                
                realized_pnl_this_step = (self.entry_price - cover_fill_price) * self.position_quantity
                self.current_capital -= (self.position_quantity * cover_fill_price) # Cost to buy back
                
                # Apply transaction fee and track for reward calculation
                fee_amount = self._apply_transaction_fee(self.position_quantity, cover_fill_price, "SHORT COVER")
                trade_value_executed += self.position_quantity * cover_fill_price
                shares_traded_this_step += self.position_quantity
                
                impact_bps = cover_impact_info.get('total_impact_bps', 0.0)
                self.logger.debug(f"Covered SHORT: {self.position_quantity} @ {cover_fill_price:.4f} (mid: {current_price:.4f}, impact: {impact_bps:.1f}bps). Entry: {self.entry_price:.4f}. P&L: {realized_pnl_this_step:.2f}")

            self.daily_pnl += realized_pnl_this_step # Accumulate daily *realized* P&L
            self.position_quantity = 0 # Flat after exiting

            # 2. Enter new position if not going flat
            if desired_position_signal == 1: # Buy to go Long
                # Equity-scaled position sizing: shares = k * portfolio_value / price
                # This ensures impact and fees scale proportionally with account size
                target_shares = self.equity_scaling_factor * self.portfolio_value / current_price
                self.position_quantity = np.floor(target_shares)
                
                self.logger.debug(f"🔍 EQUITY SCALING (LONG): k={self.equity_scaling_factor:.3f}, Portfolio=${self.portfolio_value:.2f}, Price=${current_price:.2f} → Target={target_shares:.1f} → Shares={self.position_quantity:.0f}")
                
                # Check if we can afford the position
                required_capital = self.position_quantity * current_price
                if self.current_capital >= required_capital and self.position_quantity > 0:
                    # Position is affordable
                    pass
                elif self.current_capital > current_price:
                    # Fallback: at least 1 share if we have enough capital
                    self.position_quantity = 1
                else:
                    # Cannot afford any position
                    self.position_quantity = 0

                if self.position_quantity > 0:
                    # Get fill price for buying
                    entry_fill_price, entry_impact_info = self._get_fill_price(
                        mid_price=current_price,
                        trade_size=self.position_quantity,
                        side="buy"
                    )
                    
                    self.entry_price = entry_fill_price
                    self.current_capital -= (self.position_quantity * entry_fill_price) # Cash used
                    
                    # Apply transaction fee and track for reward calculation
                    fee_amount = self._apply_transaction_fee(self.position_quantity, entry_fill_price, "LONG ENTRY")
                    trade_value_executed += self.position_quantity * entry_fill_price
                    shares_traded_this_step += self.position_quantity
                    
                    impact_bps = entry_impact_info.get('total_impact_bps', 0.0)
                    self.logger.debug(f"Opened LONG: {self.position_quantity} @ {entry_fill_price:.4f} (mid: {current_price:.4f}, impact: {impact_bps:.1f}bps)")
                else: # Cannot afford to go long
                    desired_position_signal = 0 # Force flat

            elif desired_position_signal == -1: # Sell to go Short
                # Equity-scaled position sizing: shares = k * portfolio_value / price
                # This ensures impact and fees scale proportionally with account size
                target_shares = self.equity_scaling_factor * self.portfolio_value / current_price
                self.position_quantity = np.floor(target_shares)
                
                self.logger.debug(f"🔍 EQUITY SCALING (SHORT): k={self.equity_scaling_factor:.3f}, Portfolio=${self.portfolio_value:.2f}, Price=${current_price:.2f} → Target={target_shares:.1f} → Shares={self.position_quantity:.0f}")
                
                # Check if we have sufficient capital for margin requirements (simplified)
                if self.current_capital > 0 and self.position_quantity > 0:
                    # Position is feasible
                    pass
                elif self.current_capital > current_price:
                    # Fallback: at least 1 share if we have enough capital
                    self.position_quantity = 1
                else:
                    # Cannot afford any position
                    self.position_quantity = 0

                if self.position_quantity > 0:
                    # Get fill price for selling short
                    entry_fill_price, entry_impact_info = self._get_fill_price(
                        mid_price=current_price,
                        trade_size=self.position_quantity,
                        side="sell"
                    )
                    
                    self.entry_price = entry_fill_price
                    self.current_capital += (self.position_quantity * entry_fill_price) # Cash received from shorting (simplified)
                    
                    # Apply transaction fee and track for reward calculation
                    fee_amount = self._apply_transaction_fee(self.position_quantity, entry_fill_price, "SHORT ENTRY")
                    trade_value_executed += self.position_quantity * entry_fill_price
                    shares_traded_this_step += self.position_quantity
                    
                    impact_bps = entry_impact_info.get('total_impact_bps', 0.0)
                    self.logger.debug(f"Opened SHORT: {self.position_quantity} @ {entry_fill_price:.4f} (mid: {current_price:.4f}, impact: {impact_bps:.1f}bps)")
                else: # Cannot short (e.g. no capital for margin, though simplified here)
                    desired_position_signal = 0 # Force flat
            
            self.current_position = desired_position_signal
            if self.current_position == 0: # If ended up flat
                self.entry_price = 0.0
                self.position_quantity = 0.0
        
        # 🔍 DIAGNOSTIC: Track position changes
        self.position_history.append({
            'step': self.current_step,
            'position_signal': self.current_position,
            'position_quantity': self.position_quantity,
            'action': action
        })
        
        # Update portfolio value after any transactions
        self._update_portfolio_value()
        
        # REWARD CALCULATION: Enhanced system addresses high-frequency noise issues
        gross_pnl_change = (self.portfolio_value - portfolio_value_before_action)
        self.net_pnl_this_step = gross_pnl_change  # Net P&L already includes fees (they reduce portfolio value)
        
        # 🟢 TRAINING HEARTBEAT - Log every 100 steps to confirm training is active
        if self.current_step % 100 == 0:
            self.logger.info(f"🟢 TRAINING ACTIVE - Step {self.current_step}, Portfolio: ${self.portfolio_value:.2f}, PnL: ${gross_pnl_change:.2f}, Action: {action}")
        
        # 🎯 NEW TURNOVER PENALTY SYSTEM: Use normalized turnover penalty if enabled
        if self.use_turnover_penalty:
            reward = self._calculate_turnover_reward(
                realized_pnl=realized_pnl_this_step,
                transaction_cost=self.total_fees_this_step,
                position_changed=transaction_executed
            )
            
            # Store turnover reward breakdown for monitoring
            turnover_penalty = self._calculate_turnover_penalty()
            
            # Get normalized turnover from rolling window (390 steps = 1 trading day)
            rolling_turnover_total = sum(self.rolling_turnover_window)
            if self.turnover_penalty_calculator is not None:
                current_portfolio_value = self.turnover_penalty_calculator._get_current_portfolio_value()
                # Rolling turnover ratio: rolling_turnover / portfolio_value (dimensionless ratio)
                normalized_turnover = rolling_turnover_total / (current_portfolio_value + 1e-6)
            else:
                # Fallback calculation for rolling turnover ratio
                normalized_turnover = rolling_turnover_total / (self.portfolio_value + 1e-6)
            
            self._last_reward_breakdown = {
                'turnover_system': True,
                'realized_pnl': realized_pnl_this_step,
                'transaction_cost': self.total_fees_this_step,
                'position_changed': transaction_executed,
                'total_reward': reward,
                'core_pnl': realized_pnl_this_step - self.total_fees_this_step,
                'turnover_penalty': turnover_penalty,
                'normalized_turnover': normalized_turnover,
                'target_turnover': self.turnover_target_ratio
            }
            
            self.logger.debug(
                f"🎯 TURNOVER REWARD ACTIVE: Step {self.current_step}, "
                f"P&L: ${realized_pnl_this_step:.4f}, Cost: ${self.total_fees_this_step:.4f}, "
                f"Turnover: {normalized_turnover:.4f}, Reward: {reward:.4f}"
            )
        
        # 🚨 EMERGENCY REWARD FIX: Use simplified reward function if enabled (fallback)
        elif self.use_emergency_reward_fix:
            reward = self._calculate_emergency_reward(
                realized_pnl=realized_pnl_this_step,
                transaction_cost=self.total_fees_this_step,
                position_changed=transaction_executed
            )
            
            # Store simple reward breakdown for monitoring
            self._last_reward_breakdown = {
                'emergency_fix': True,
                'realized_pnl': realized_pnl_this_step,
                'transaction_cost': self.total_fees_this_step,
                'position_changed': transaction_executed,
                'total_reward': reward,
                'core_pnl': realized_pnl_this_step - self.total_fees_this_step,
                'holding_bonus': 0.0 if transaction_executed else self.emergency_holding_bonus * (self.portfolio_value / self.initial_capital) * 100
            }
            
            self.logger.debug(
                f"🚨 EMERGENCY REWARD ACTIVE: Step {self.current_step}, "
                f"P&L: ${realized_pnl_this_step:.4f}, Cost: ${self.total_fees_this_step:.4f}, "
                f"Reward: {reward:.4f}"
            )
        
        # Use Enhanced Reward System if enabled (addresses high-frequency reward noise)
        elif self.enhanced_reward_calculator:
            reward, reward_breakdown = self.enhanced_reward_calculator.calculate_reward(
                realized_pnl=realized_pnl_this_step,
                transaction_cost=self.total_fees_this_step,
                current_position=self.current_position,
                current_price=current_price,
                portfolio_value=self.portfolio_value,
                step_info={'step': self.current_step, 'timestamp': timestamp}
            )
            
            # Store reward breakdown for monitoring
            self._last_reward_breakdown = reward_breakdown
            
            self.logger.debug(
                f"🚀 ENHANCED REWARD: Total={reward:.6f} "
                f"(Core P&L: {reward_breakdown.get('core_pnl', 0):.6f}, "
                f"Directional: {reward_breakdown.get('directional', 0):.6f}, "
                f"Behavioral: {reward_breakdown.get('behavioral', 0):.6f})"
            )
        else:
            # Traditional reward system (net P&L only)
            reward = self.net_pnl_this_step
            self._last_reward_breakdown = {
                'core_pnl': reward,
                'directional': 0.0,
                'behavioral': 0.0,
                'multi_timeframe': 0.0,
                'total_reward': reward,
                'raw_pnl': realized_pnl_this_step,
                'transaction_cost': self.total_fees_this_step,
                'net_pnl': self.net_pnl_this_step
            }
        
        # ADVANCED REWARD SHAPING: Apply cutting-edge risk-aware reward modifications
        if self.advanced_reward_shaper:
            # Calculate required metrics for advanced reward shaping
            self.portfolio_values_history.append(self.portfolio_value)
            
            # Calculate current return
            if len(self.portfolio_values_history) > 1:
                current_return = (self.portfolio_value - self.portfolio_values_history[-2]) / self.portfolio_values_history[-2]
                self.returns_history.append(current_return)
            else:
                current_return = 0.0
                self.returns_history.append(0.0)
            
            # Calculate rolling volatility (if enough history)
            if len(self.returns_history) >= 10:  # Minimum for volatility calculation
                recent_returns = np.array(self.returns_history[-60:])  # Last 60 steps (1 hour)
                current_volatility = np.std(recent_returns) * np.sqrt(252 * 390)  # Annualized volatility
            else:
                current_volatility = 0.0
            
            # Calculate current drawdown
            if len(self.portfolio_values_history) > 1:
                peak_value = max(self.portfolio_values_history)
                current_drawdown = (peak_value - self.portfolio_value) / peak_value
            else:
                current_drawdown = 0.0
            
            # Apply advanced reward shaping
            shaped_reward, shaping_info = self.advanced_reward_shaper.shape_reward(
                base_reward=reward,
                pnl=self.net_pnl_this_step,
                current_return=current_return,
                volatility=current_volatility,
                drawdown=current_drawdown
            )
            
            # Update reward with shaped version
            reward = shaped_reward
            
            # Store volatility penalty for monitoring
            self._last_vol_penalty = shaping_info.get('volatility_penalty', 0.0)
            
            # Log advanced reward shaping details (debug level)
            if shaping_info.get('total_shaping', 0) != 0:
                self.logger.debug(f"🎯 ADVANCED REWARD SHAPING: Step {self.current_step}")
                self.logger.debug(f"   Base reward: ${shaping_info.get('base_reward', 0):.6f}")
                self.logger.debug(f"   Shaped reward: ${shaping_info.get('shaped_reward', 0):.6f}")
                self.logger.debug(f"   Total shaping: ${shaping_info.get('total_shaping', 0):.6f}")
                if 'lagrangian_penalty' in shaping_info:
                    self.logger.debug(f"   Lagrangian penalty: ${shaping_info['lagrangian_penalty']:.6f} (λ={shaping_info.get('lambda_value', 0):.4f})")
                if 'sharpe_reward' in shaping_info:
                    self.logger.debug(f"   Sharpe reward: ${shaping_info['sharpe_reward']:.6f} (Sharpe={shaping_info.get('current_sharpe', 0):.4f})")
                if 'cvar_penalty' in shaping_info:
                    self.logger.debug(f"   CVaR penalty: ${shaping_info['cvar_penalty']:.6f} (CVaR={shaping_info.get('cvar', 0):.4f})")
        
        # Add turnover bonus when below threshold
        current_turnover_ratio = self.hourly_traded_value / max(self.start_of_day_portfolio_value, 1.0)
        turnover_threshold = self.hourly_turnover_cap * self.turnover_bonus_threshold
        
        if current_turnover_ratio < turnover_threshold:
            turnover_bonus = self.turnover_bonus_factor * self.start_of_day_portfolio_value
            reward += turnover_bonus
            self.logger.debug(f"🎁 TURNOVER BONUS: Step {self.current_step}, Turnover: {current_turnover_ratio:.3f} < {turnover_threshold:.3f}, Bonus: ${turnover_bonus:.6f}")
        
        self.logger.debug(f"💰 REWARD BREAKDOWN: Step {self.current_step}, Gross P&L: ${gross_pnl_change:.6f}, Fees: ${self.total_fees_this_step:.6f}, Net P&L: ${self.net_pnl_this_step:.6f}")
        
        # Apply L2 penalty for action changes to discourage ping-ponging
        # Scale penalty consistently with reward to prevent domination
        action_change_penalty = 0.0
        if self.action_change_penalty_factor > 0:
            action_change_penalty = self.action_change_penalty_factor * ((action - self.previous_action) ** 2) * self.reward_scaling
            reward -= action_change_penalty
            if action != self.previous_action:
                # Log at INFO level for significant penalties (was DEBUG)
                log_level = logging.INFO if action_change_penalty > 0.1 else logging.DEBUG
                self.logger.log(log_level, f"🔄 [ActionChangePenalty] Step {self.current_step}, Action {self.previous_action}→{action}, Penalty: ${action_change_penalty:.2f} (factor: {self.action_change_penalty_factor})")
        
        # Track same action repeats and apply penalty for excessive repetition
        same_action_penalty = 0.0
        if action == self.previous_action:
            self.same_action_count += 1
            # Apply penalty if exceeding max_same_action_repeat
            if self.same_action_count >= self.max_same_action_repeat:
                same_action_penalty = 0.02 * self.reward_scaling  # Fixed penalty for excessive repetition
                reward -= same_action_penalty
                self.logger.info(f"🔄 [SameActionPenalty] Step {self.current_step}, Action {action} repeated {self.same_action_count} times, Penalty: ${same_action_penalty:.2f}")
        else:
            self.same_action_count = 0  # Reset counter on action change
        
        # Update previous action for next step
        self.previous_action = action
        
        # Track shares traded for emergency reward calculation
        self.shares_traded_last_step = shares_traded_this_step

        # --- Log trade if executed ---
        if transaction_executed:
            # Track episode-level metrics
            self.episode_total_trades += 1
            self.episode_total_turnover += trade_value_executed
            self.episode_realized_pnl += realized_pnl_this_step
            
            # Track turnover for new penalty system
            self.total_traded_value += trade_value_executed
            
            # 🔧 ROLLING WINDOW: Add to rolling turnover window
            self.rolling_turnover_window.append(trade_value_executed)
            
            # 🔍 DIAGNOSTIC: Trade executed
            self.logger.log(self.TRADE_LOG_LEVEL, f"🔍 TRADE EXECUTED: Step {self.current_step}, Action {action} -> Position {desired_position_signal}, Shares: {shares_traded_this_step:.2f}, Value: ${trade_value_executed:.2f}")
            
            if self.log_trades_flag:
                trade_details = {
                    'step': self.current_step,
                    'timestamp': timestamp,
                    'action_signal': self._action_map[action], # The original agent action
                    'price': current_price,
                    'final_position': self.current_position,
                    'shares_traded': shares_traded_this_step,
                    'trade_value': trade_value_executed,
                    'realized_pnl_step': realized_pnl_this_step,
                    'portfolio_value_after_trade': self.portfolio_value,
                    'cash_capital_after_trade': self.current_capital,
                }
                self.trade_log.append(trade_details)
            
            # Update turnover tracking (hourly pruning already done at step start)
            self.trades_this_hour.append((timestamp, trade_value_executed, shares_traded_this_step))
            self.hourly_traded_value += trade_value_executed
        else:
            # 🔍 DIAGNOSTIC: No trade executed
            self.logger.debug(f"🔍 NO TRADE: Step {self.current_step}, Action {action} -> Position {desired_position_signal} (current: {self.current_position})")

        # --- Risk Management Checks ---
        terminated = False
        # Step 2: New soft/hard DD system with live penalties (no termination)
        # Use DD baseline for drawdown calculation if enabled
        if self.dd_baseline_reset_enabled and self.dd_baseline_value is not None:
            dd_reference_value = self.dd_baseline_value
            self.steps_since_dd_baseline += 1
        else:
            dd_reference_value = self.start_of_day_portfolio_value
        
        current_drawdown_pct = (dd_reference_value - self.portfolio_value) / dd_reference_value \
                               if dd_reference_value > 0 else 0
        
        # Use institutional safeguards DD limits if available
        if self.institutional_safeguards:
            soft_limit = self.institutional_safeguards.soft_dd_pct
            hard_limit = self.institutional_safeguards.hard_dd_pct
            terminate_on_hard = self.institutional_safeguards.terminate_on_hard
        else:
            # Fallback to legacy limits
            soft_limit = self.max_daily_drawdown_pct
            hard_limit = self.max_daily_drawdown_pct * 1.25
            terminate_on_hard = False
        
        # Note: Lambda decay now handled automatically in _get_current_penalty_lambda
        
        # Emit general drawdown telemetry (always recorded) - only if logger supports it
        if hasattr(self.logger, 'record'):
            self.logger.record('drawdown/current_pct', current_drawdown_pct * 100)
            self.logger.record('drawdown/soft_limit_pct', soft_limit * 100)
            self.logger.record('drawdown/hard_limit_pct', hard_limit * 100)
            self.logger.record('drawdown/consecutive_steps', self.consecutive_drawdown_steps)
            
            # Baseline reset telemetry (always recorded if enabled)
            if self.dd_baseline_reset_enabled and self.dd_baseline_value is not None:
                self.logger.record('baseline/current_value', self.dd_baseline_value)
                self.logger.record('baseline/steps_since_reset', self.steps_since_dd_baseline)
                self.logger.record('baseline/portfolio_vs_baseline', 
                                 (self.portfolio_value - self.dd_baseline_value) / self.dd_baseline_value * 100)
                
                # Calculate recovery threshold for monitoring
                purgatory_escape_buffer = self.dd_baseline_value * self.purgatory_escape_threshold_pct
                recovery_threshold = self.dd_baseline_value + purgatory_escape_buffer
                self.logger.record('baseline/recovery_threshold', recovery_threshold)
                self.logger.record('baseline/distance_to_escape', 
                                 (recovery_threshold - self.portfolio_value) / self.portfolio_value * 100)
        
        if current_drawdown_pct > soft_limit:
            self.consecutive_drawdown_steps += 1
            
            if current_drawdown_pct > hard_limit:
                # Step 2: Hard limit - severe penalty but NO termination for Phase 1
                self.logger.warning(f"Step {self.current_step}: HARD DD breached! Drawdown: {current_drawdown_pct*100:.2f}%, Hard Limit: {hard_limit*100:.2f}%. Portfolio Value: {self.portfolio_value:.2f} (consecutive: {self.consecutive_drawdown_steps})")
                
                # Step 2: Live reward penalty (no done=True)
                drawdown_excess = current_drawdown_pct - hard_limit
                # Get lambda with exponential scaling for this excess
                penalty_lambda = self._get_current_penalty_lambda(drawdown_excess)
                # Step 3: Quadratic penalty - mild at 2.1%, brutal past 3%
                # NOTE: Don't apply reward_scaling to penalties - they should bite regardless of reward scale
                hard_penalty = penalty_lambda * (drawdown_excess ** 2)
                reward -= hard_penalty
                
                # Emit telemetry for hard DD penalty
                if hasattr(self.logger, 'record'):
                    self.logger.record('penalties/hard_dd_penalty', hard_penalty)
                    self.logger.record('penalties/hard_dd_excess_pct', drawdown_excess * 100)
                self.logger.warning(f"Step {self.current_step}: Applied HARD DD penalty: {hard_penalty:.6f} "
                                  f"(λ={penalty_lambda:.1f}, excess={drawdown_excess*100:.2f}%, excess²={drawdown_excess**2:.6f})")
                
                # Only terminate if explicitly configured (Phase 1: False)
                if terminate_on_hard:
                    terminated = True
                    self.logger.error(f"🚨 HARD DD TERMINATION! (terminate_on_hard=True)")
                    
            else:
                # Soft limit - regular penalty
                self.logger.warning(f"Step {self.current_step}: Soft DD breached! Drawdown: {current_drawdown_pct*100:.2f}%, Soft Limit: {soft_limit*100:.2f}%. Portfolio Value: {self.portfolio_value:.2f} (consecutive: {self.consecutive_drawdown_steps})")
                
                drawdown_excess = current_drawdown_pct - soft_limit
                # Get lambda with exponential scaling for this excess
                penalty_lambda = self._get_current_penalty_lambda(drawdown_excess)
                # Step 3: Quadratic penalty with 0.5x multiplier for soft limit
                # NOTE: Don't apply reward_scaling to penalties - they should bite regardless of reward scale
                soft_penalty = penalty_lambda * (drawdown_excess ** 2) * 0.5
                reward -= soft_penalty
                
                # Emit telemetry for soft DD penalty
                if hasattr(self.logger, 'record'):
                    self.logger.record('penalties/soft_dd_penalty', soft_penalty)
                    self.logger.record('penalties/soft_dd_excess_pct', drawdown_excess * 100)
                self.logger.warning(f"Step {self.current_step}: Applied soft DD penalty: {soft_penalty:.6f} "
                                  f"(λ={penalty_lambda:.1f}, excess={drawdown_excess*100:.2f}%, excess²={drawdown_excess**2:.6f})")
        else:
            # Reset consecutive drawdown counter when not in drawdown
            self.consecutive_drawdown_steps = 0
        
        # Enhanced DD baseline reset logic: Escape DD purgatory
        if self.dd_baseline_reset_enabled and self.dd_baseline_value is not None:
            # Method 1: Reset when equity > baseline + purgatory_escape_threshold (escape purgatory)
            purgatory_escape_buffer = self.dd_baseline_value * self.purgatory_escape_threshold_pct
            recovery_threshold = self.dd_baseline_value + purgatory_escape_buffer
            recovery_met = self.portfolio_value > recovery_threshold
            
            # Method 2: Reset after N flat steps (prevent permanent purgatory)
            timeout_met = self.steps_since_dd_baseline >= BASELINE_RESET_FLAT_STEPS
            
            # Method 3: Legacy recovery threshold (keep for compatibility)
            equity_recovery = (self.portfolio_value - self.dd_baseline_value) / self.dd_baseline_value
            legacy_recovery_met = equity_recovery >= self.dd_recovery_threshold_pct
            
            # Reset baseline if any condition is met
            if recovery_met or timeout_met or legacy_recovery_met:
                old_baseline = self.dd_baseline_value
                old_step = self.dd_baseline_step
                
                # Determine reset reason
                if recovery_met:
                    reset_reason = "purgatory_escape"
                    reason_detail = f"equity ${self.portfolio_value:.2f} > threshold ${recovery_threshold:.2f}"
                elif timeout_met:
                    reset_reason = "flat_timeout"
                    reason_detail = f"{self.steps_since_dd_baseline} steps since baseline"
                else:
                    reset_reason = "legacy_recovery"
                    reason_detail = f"{equity_recovery:.2%} recovery"
                
                self.dd_baseline_value = self.portfolio_value
                self.dd_baseline_step = self.current_step
                self.steps_since_dd_baseline = 0
                
                # Emit telemetry for baseline resets
                if hasattr(self.logger, 'record'):
                    self.logger.record('baseline/reset_triggered', 1.0)
                    self.logger.record('baseline/reset_reason', 1.0 if reset_reason == "purgatory_escape" else 
                                     2.0 if reset_reason == "flat_timeout" else 3.0)
                    self.logger.record('baseline/old_baseline', old_baseline)
                    self.logger.record('baseline/new_baseline', self.dd_baseline_value)
                
                self.logger.info(f"🔄 DD baseline RESET ({reset_reason}): ${old_baseline:.2f} → ${self.dd_baseline_value:.2f} "
                               f"(step {old_step} → {self.dd_baseline_step}) - {reason_detail}")
        
        # Positive recovery bonus - reward when above baseline (excess DD < 0)
        if self.recovery_bonus_enabled and self.dd_baseline_reset_enabled and self.dd_baseline_value is not None:
            if current_drawdown_pct < 0:  # Portfolio value > baseline (recovery)
                recovery_bonus = self.recovery_bonus_amount
                reward += recovery_bonus
                
                # Log recovery bonus occasionally to avoid spam
                if self.current_step % 100 == 0:  # Log every 100 steps
                    recovery_pct = abs(current_drawdown_pct) * 100  # Convert to positive %
                    self.logger.info(f"💰 Recovery bonus: +{recovery_bonus:.2f} (above baseline by {recovery_pct:.2f}%)")
        
        # Bootstrap recovery bonus - additional reward when DD < 1% (pulls policy toward flat)
        if self.recovery_bonus_enabled and current_drawdown_pct < 0.01:  # Below 1% DD
            bootstrap_bonus = self.recovery_bonus_amount * 0.5  # Half the regular bonus
            reward += bootstrap_bonus
            
            # Log bootstrap bonus occasionally
            if self.current_step % 200 == 0:  # Log every 200 steps
                self.logger.info(f"🌱 Bootstrap bonus: +{bootstrap_bonus:.2f} (DD < 1%: {current_drawdown_pct*100:.2f}%)")
        
        # Early-warning logger - detect creeping risk
        if self.early_warning_enabled:
            # Calculate excess DD above soft limit (0.5% threshold)
            if self.institutional_safeguards:
                soft_limit = self.institutional_safeguards.soft_dd_pct
            else:
                soft_limit = self.max_daily_drawdown_pct
            
            excess_dd = current_drawdown_pct - soft_limit
            
            if excess_dd > self.early_warning_threshold_pct:
                # Above warning threshold - increment counter
                self.warning_excess_steps += 1
                
                # Issue warning if sustained for duration and not recently warned
                if (self.warning_excess_steps >= self.early_warning_duration_steps and 
                    self.current_step - self.last_warning_step > 200):  # Prevent spam (200 step cooldown)
                    
                    self.logger.warning(f"⚠️ EARLY WARNING: Excess DD {excess_dd*100:.2f}% sustained for {self.warning_excess_steps} steps "
                                      f"(threshold: {self.early_warning_threshold_pct*100:.1f}%, duration: {self.early_warning_duration_steps})")
                    self.last_warning_step = self.current_step
            else:
                # Below warning threshold - reset counter
                if self.warning_excess_steps > 0:
                    self.warning_excess_steps = 0

        # 2. Hourly Turnover Cap - BINDING ENFORCEMENT (More aggressive during emergency fix)
        if self.start_of_day_portfolio_value > 0:
            current_hourly_turnover_ratio = self.hourly_traded_value / self.start_of_day_portfolio_value
            
            # Use more aggressive turnover cap during emergency fix
            effective_turnover_cap = self.hourly_turnover_cap
            if self.use_emergency_reward_fix:
                effective_turnover_cap = min(self.hourly_turnover_cap, 1.0)  # Cap at 1.0x during emergency fix
            
            if current_hourly_turnover_ratio > effective_turnover_cap:
                emergency_marker = "🚨 EMERGENCY " if self.use_emergency_reward_fix else ""
                self.logger.warning(f"Step {self.current_step}: {emergency_marker}Hourly turnover cap breached! Ratio: {current_hourly_turnover_ratio:.2f}x, Effective Limit: {effective_turnover_cap:.2f}x.")
                
                # 🚨 STRONG NEGATIVE REWARD for turnover breach
                excess_turnover = current_hourly_turnover_ratio - effective_turnover_cap
                
                # FIRST-ORDER FIX: Dynamic Huber-shaped penalty (gentle near threshold, steeper for major violations)
                # Base penalty scales with portfolio value (dynamic scaling)
                base_penalty = excess_turnover * self.start_of_day_portfolio_value * self.turnover_penalty_factor
                
                # Huber-shaped penalty: quadratic for small violations, linear for large ones
                huber_threshold = 0.05  # Switch to linear after 5% excess
                if excess_turnover <= huber_threshold:
                    # Quadratic for small violations (gentle)
                    huber_penalty = (excess_turnover ** 2) * self.start_of_day_portfolio_value * self.turnover_exponential_penalty_factor
                else:
                    # Linear for large violations (prevents explosion)
                    quadratic_part = (huber_threshold ** 2) * self.start_of_day_portfolio_value * self.turnover_exponential_penalty_factor
                    linear_part = (excess_turnover - huber_threshold) * huber_threshold * self.start_of_day_portfolio_value * self.turnover_exponential_penalty_factor
                    huber_penalty = quadratic_part + linear_part
                
                total_turnover_penalty = base_penalty + huber_penalty
                
                reward -= total_turnover_penalty
                
                self.logger.warning(f"Step {self.current_step}: Applied turnover penalty: ${total_turnover_penalty:.2f} (base: ${base_penalty:.2f}, huber: ${huber_penalty:.2f})")
                
                # REMOVED: Turnover-based termination - let penalty term do the teaching
                # Keep episode alive and let agent learn from penalties instead of terminating
                if self.terminate_on_turnover_breach or self.use_emergency_reward_fix:
                    # Calculate what would have been termination threshold for logging
                    if self.use_emergency_reward_fix:
                        termination_threshold = effective_turnover_cap * 1.5
                    else:
                        termination_threshold = self.hourly_turnover_cap * self.turnover_termination_threshold_multiplier
                    
                    if current_hourly_turnover_ratio > termination_threshold:
                        self.logger.warning(f"Step {self.current_step}: Severe turnover breach (would have terminated)! Ratio: {current_hourly_turnover_ratio:.2f}x, Cap: {self.hourly_turnover_cap:.2f}x, Threshold: {termination_threshold:.2f}x")
                        # REMOVED: terminated = True  # Let penalty term teach instead
                        # Apply additional penalty but keep episode alive
                        severe_penalty = self.start_of_day_portfolio_value * self.turnover_termination_penalty_pct * 0.1  # Reduced penalty
                        reward -= severe_penalty
                        self.logger.warning(f"Step {self.current_step}: Applied severe turnover penalty: ${severe_penalty:.2f} (but episode continues)")
        
        # --- Advance Time & Log Portfolio ---
        self.current_step += 1
        self.portfolio_history.append(self.portfolio_value)

        # --- Check for Episode End ---
        truncated = False
        if self.current_step >= len(self.market_feature_data): # Ran out of market data
            self.logger.info(f"Episode ended: Reached end of data at step {self.current_step}.")
            truncated = True 
        if self._max_episode_steps is not None and self.current_step >= self._max_episode_steps: # Hit max configured steps
            self.logger.info(f"Episode truncated: Reached max episode steps {self._max_episode_steps}.")
            truncated = True 

        if not truncated and not terminated:
             self._handle_new_day()
             observation = self._get_observation()
        else: # Episode is ending
            if self.current_step >= len(self.market_feature_data) : 
                 # Create a zero observation if out of data bounds
                 if self.lookback_window > 1:
                     observation = np.zeros((self.lookback_window, self.num_market_features + 1), dtype=np.float32)
                 else:
                     observation = np.zeros((self.num_market_features + 1,), dtype=np.float32)
            else: 
                 observation = self._get_observation() # Get last valid observation
        
        # Store observation for monitoring (Q-value tracking)
        self.last_observation = observation

        info = self._get_info()
        info['portfolio_value'] = self.portfolio_value # Ensure latest is passed
        
        # 🎯 REWARD-P&L AUDIT: Expose detailed P&L breakdown for correlation analysis
        info['realized_pnl_step'] = realized_pnl_this_step
        info['unrealized_pnl_step'] = self._compute_unrealized_pnl()
        info['total_pnl_step'] = realized_pnl_this_step + info['unrealized_pnl_step']
        info['fees_step'] = self.total_fees_this_step
        info['net_pnl_step'] = self.net_pnl_this_step  # P&L after fees
        info['raw_reward'] = reward  # Before scaling
        info['scaled_reward'] = reward * self.reward_scaling  # After scaling
        info['cumulative_realized_pnl'] = self.daily_pnl  # Running total
        info['cumulative_portfolio_pnl'] = self.portfolio_value - self.initial_capital
        
        # 🚀 ENHANCED REWARD BREAKDOWN: Detailed reward component analysis
        if hasattr(self, '_last_reward_breakdown'):
            info['reward_breakdown'] = self._last_reward_breakdown
            
            # Enhanced reward statistics for monitoring
            if self.enhanced_reward_calculator:
                reward_stats = self.enhanced_reward_calculator.get_statistics()
                info['enhanced_reward_stats'] = reward_stats
        
        # MONITORING METRICS for TensorBoard custom scalars
        # Store metrics for monitoring callback access
        self.last_reward = reward * self.reward_scaling  # Store scaled reward
        self.last_vol_penalty = getattr(self, '_last_vol_penalty', 0.0)  # Volatility penalty from advanced reward shaping
        
        # Calculate current drawdown percentage
        if not hasattr(self, 'peak_portfolio_value'):
            self.peak_portfolio_value = self.initial_capital
        
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        current_drawdown_pct = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100
        self.current_drawdown_pct = max(0.0, current_drawdown_pct)  # Ensure non-negative
        
        # Store Lagrangian lambda if available
        if self.advanced_reward_shaper and hasattr(self.advanced_reward_shaper, 'lagrangian_manager'):
            self.lagrangian_lambda = getattr(self.advanced_reward_shaper.lagrangian_manager, 'lambda_value', 0.0)
        else:
            self.lagrangian_lambda = 0.0
        
        # Add monitoring info to step info
        info['monitoring'] = {
            'vol_penalty': self.last_vol_penalty,
            'drawdown_pct': self.current_drawdown_pct,
            'lagrangian_lambda': self.lagrangian_lambda,
            'reward_magnitude': abs(self.last_reward),
            'peak_portfolio_value': self.peak_portfolio_value
        }
        
        if transaction_executed:
            info['last_trade_details'] = self.trade_log[-1] if hasattr(self, 'trade_log') and self.trade_log else {}
        
        self.logger.debug(
            f"Step: {self.current_step-1}, Action: {action}(Signal:{self._action_map[action]}), "
            f"Price: {current_price:.2f}, Reward(scaled): {reward * self.reward_scaling:.4f}, PortfolioVal: {self.portfolio_value:.2f}, "
            f"Position: {self.current_position}({self.position_quantity:.2f} units), Term: {terminated}, Trunc: {truncated}"
        )
        
        # Log comprehensive episode summary when episode ends
        if terminated or truncated:
            episode_summary = self._get_episode_summary()
            info['episode_summary'] = episode_summary
            
            # Log key metrics for monitoring and drift detection
            self.logger.info("=" * 80)
            self.logger.info("📊 EPISODE SUMMARY - MONITORING & DRIFT DETECTION")
            self.logger.info("=" * 80)
            self.logger.info(f"🕐 Duration: {episode_summary['episode_start_time']} → {episode_summary['episode_end_time']}")
            self.logger.info(f"📈 Performance: {episode_summary['total_return_pct']:+.2f}% (${episode_summary['net_pnl_after_fees']:+,.2f})")
            self.logger.info(f"💰 P&L Breakdown: Realized=${episode_summary['realized_pnl']:+,.2f}, Unrealized=${episode_summary['unrealized_pnl']:+,.2f}")
            self.logger.info(f"💸 Total Fees: ${episode_summary['total_fees']:,.2f} ({episode_summary['fee_rate_pct']:.3f}% of turnover)")
            self.logger.info(f"🔄 Trading Activity: {episode_summary['total_trades']} trades, ${episode_summary['total_turnover']:,.0f} turnover ({episode_summary['turnover_ratio']:.2f}x)")
            self.logger.info(f"⚡ Trade Efficiency: ${episode_summary['avg_trade_size']:,.0f} avg size, {episode_summary['trades_per_hour']:.1f} trades/hour")
            self.logger.info(f"🎯 Actions: {episode_summary['action_histogram']}")
            self.logger.info(f"📍 Final Position: {episode_summary['final_position']} ({episode_summary['final_position_quantity']:.0f} shares)")
            self.logger.info(f"📊 Risk-Adjusted: Sharpe={episode_summary['sharpe_ratio']:.2f}, Sortino={episode_summary['sortino_ratio']:.2f}, Vol={episode_summary['volatility']:.2f}")
            self.logger.info("=" * 80)
            
            # Save episode summary to CSV for analysis and drift detection
            self._save_episode_summary_to_csv(episode_summary)
        
        # Apply PPO-friendly reward scaling to keep rewards in [-10, +10] range
        if self.ppo_reward_scaling:
            # Scale by portfolio value to normalize reward magnitude
            ppo_scale_divisor = self.start_of_day_portfolio_value / self.ppo_scale_factor
            scaled_reward = (reward * self.reward_scaling) / ppo_scale_divisor
        else:
            scaled_reward = reward * self.reward_scaling
        
        # Track episode total reward for curriculum
        if not hasattr(self, '_episode_total_reward'):
            self._episode_total_reward = 0.0
        self._episode_total_reward += scaled_reward
        
        # 🚨 FINAL SAFETY CHECK: Force termination if conditions are met
        # Legacy final safety termination - DISABLED for Phase 1
        # Phase 1 uses new soft/hard DD system with no termination
        if not terminated and not truncated and hasattr(self, 'institutional_safeguards_config') and self.institutional_safeguards_config and self.institutional_safeguards_config.get("phase") == "production":
            # Only apply final safety termination in production phase
            current_drawdown_pct = (self.start_of_day_portfolio_value - self.portfolio_value) / self.start_of_day_portfolio_value \
                                   if self.start_of_day_portfolio_value > 0 else 0
            severe_threshold = self.max_daily_drawdown_pct * 1.25
            
            if (current_drawdown_pct > severe_threshold or 
                self.consecutive_drawdown_steps >= self.max_consecutive_drawdown_steps):
                
                self.logger.error(f"🚨 FINAL SAFETY TERMINATION! Drawdown: {current_drawdown_pct*100:.2f}%, Consecutive: {self.consecutive_drawdown_steps}")
                terminated = True
                # Apply additional penalty
                safety_penalty = self.start_of_day_portfolio_value * 0.005  # 0.5% penalty
                scaled_reward -= safety_penalty
                self.logger.error(f"🚨 Applied final safety penalty: ${safety_penalty:.2f}")
        
        # Phase 1: Apply institutional safeguards validation
        if self.institutional_safeguards:
            observation, scaled_reward, terminated, info = self.institutional_safeguards.validate_step_output(
                observation, scaled_reward, terminated, info
            )
        
        return observation, scaled_reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            price_str = f"{self._get_current_price():.2f}" if self.current_step < len(self.price_data) else "N/A"
            print(f"Step: {self.current_step}, Price: {price_str}, "
                  f"Position: {self.current_position} ({self.position_quantity:.2f} units), "
                  f"Portfolio Value: {self.portfolio_value:.2f}, Cash: {self.current_capital:.2f}, "
                  f"Daily Realized P&L: {self.daily_pnl:.2f}")
        elif mode == 'logs':
            pass # Logging is done via self.logger
        else:
            super().render(mode=mode)

    def _get_info(self):
        # Ensure current_price is valid even at episode end
        current_price_safe = self._get_current_price() if self.current_step < len(self.price_data) else np.nan
        
        return {
            "current_step": self.current_step,
            "timestamp": self.dates[min(self.current_step, len(self.dates)-1)],
            "current_price": current_price_safe,
            "current_position_signal": self.current_position, # -1, 0, 1
            "current_position_quantity": self.position_quantity,
            "portfolio_value": self.portfolio_value,
            "cash_capital": self.current_capital,
            "entry_price": self.entry_price,
            "daily_realized_pnl": self.daily_pnl,
            "start_of_day_portfolio_value": self.start_of_day_portfolio_value,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "transaction_cost_pct": self.transaction_cost_pct,
            "hourly_turnover_cap": self.hourly_turnover_cap,
            "current_hourly_traded_value": self.hourly_traded_value,
            # 🔍 DIAGNOSTIC: Position tracking
            "portfolio_shares": self.position_quantity,  # Current shares held
            "position_signal_history": getattr(self, 'position_history', [])  # Track position changes
        }
    
    def _get_episode_summary(self):
        """Generate comprehensive episode summary for monitoring and drift detection."""
        self.episode_end_time = self.dates[min(self.current_step, len(self.dates)-1)]
        
        # Calculate episode duration
        episode_duration = None
        if self.episode_start_time and self.episode_end_time:
            episode_duration = (self.episode_end_time - self.episode_start_time).total_seconds() / 3600  # hours
        
        # Calculate performance metrics
        total_return_pct = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100
        turnover_ratio = self.episode_total_turnover / max(self.initial_capital, 1.0)
        fee_rate_pct = (self.episode_total_fees / max(self.episode_total_turnover, 1.0)) * 100
        
        # Calculate trade efficiency metrics
        avg_trade_size = self.episode_total_turnover / max(self.episode_total_trades, 1)
        trades_per_hour = self.episode_total_trades / max(episode_duration, 1.0) if episode_duration else 0
        
        # Calculate P&L breakdown
        unrealized_pnl = self.portfolio_value - self.initial_capital - self.episode_realized_pnl
        net_pnl_after_fees = self.portfolio_value - self.initial_capital
        
        # Calculate Sharpe and Sortino ratios for episode summary
        sharpe_ratio, sortino_ratio, volatility = self._calculate_risk_adjusted_returns()
        
        summary = {
            # Episode identification
            'episode_start_time': self.episode_start_time,
            'episode_end_time': self.episode_end_time,
            'episode_duration_hours': episode_duration,
            'total_steps': self.current_step,
            
            # Portfolio performance
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return_pct': total_return_pct,
            'net_pnl_after_fees': net_pnl_after_fees,
            
            # P&L breakdown
            'realized_pnl': self.episode_realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_fees': self.episode_total_fees,
            'fee_rate_pct': fee_rate_pct,
            
            # Trading activity
            'total_trades': self.episode_total_trades,
            'total_turnover': self.episode_total_turnover,
            'turnover_ratio': turnover_ratio,
            'avg_trade_size': avg_trade_size,
            'trades_per_hour': trades_per_hour,
            
            # Risk metrics
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'hourly_turnover_cap': self.hourly_turnover_cap,
            'final_position': self.current_position,
            'final_position_quantity': self.position_quantity,
            
            # Risk-adjusted performance metrics (for offline sweeps)
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            
            # Action distribution
            'action_histogram': dict(getattr(self, 'action_counter', {}))
        }
        
        return summary
    
    def _calculate_risk_adjusted_returns(self):
        """Calculate Sharpe and Sortino ratios for the episode."""
        if len(self.portfolio_history) < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate returns from portfolio history
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) == 0:
            return 0.0, 0.0, 0.0
        
        # Calculate basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0.0
        
        # Annualize metrics (assuming intraday data, ~390 steps per day)
        steps_per_day = 390
        annualization_factor = np.sqrt(252 * steps_per_day)  # 252 trading days
        
        sharpe_ratio_annualized = sharpe_ratio * annualization_factor
        sortino_ratio_annualized = sortino_ratio * annualization_factor
        volatility_annualized = volatility * annualization_factor
        
        return sharpe_ratio_annualized, sortino_ratio_annualized, volatility_annualized
    
    def _save_run_metadata(self, run_id: str = None, config_dict: Dict = None, additional_metadata: Dict = None):
        """Save run configuration and metadata for reproducibility."""
        if run_id is None:
            run_id = f"run_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        # Create logs directory if it doesn't exist
        logs_dir = "logs/run_metadata"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Collect environment configuration
        env_config = {
            'initial_capital': self.initial_capital,
            'transaction_cost_pct': self.transaction_cost_pct,
            'hourly_turnover_cap': self.hourly_turnover_cap,
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'reward_scaling': self.reward_scaling,
            'trade_cooldown_steps': self.trade_cooldown_steps,
            'equity_scaling_factor': self.equity_scaling_factor,
            'turnover_bonus_threshold': self.turnover_bonus_threshold,
            'turnover_bonus_factor': self.turnover_bonus_factor,
            'action_change_penalty_factor': self.action_change_penalty_factor,
            'enable_kyle_lambda_fills': self.enable_kyle_lambda_fills,
            'lookback_window': self.lookback_window,
            'max_episode_steps': self._max_episode_steps
        }
        
        # Get current random state for reproducibility
        numpy_state = np.random.get_state()
        
        metadata = {
            'run_id': run_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment_config': env_config,
            'data_info': {
                'market_features_shape': self.market_feature_data.shape,
                'price_data_length': len(self.price_data),
                'date_range': {
                    'start': str(self.dates[0]) if len(self.dates) > 0 else None,
                    'end': str(self.dates[-1]) if len(self.dates) > 0 else None
                }
            },
            'random_state': {
                'numpy_state_type': str(numpy_state[0]),
                'numpy_state_size': len(numpy_state[1]) if numpy_state[1] is not None else 0,
                'numpy_state_pos': int(numpy_state[2]),
                'numpy_state_has_gauss': int(numpy_state[3]),
                'numpy_state_cached_gaussian': float(numpy_state[4])
            },
            'system_info': {
                'python_version': os.sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            }
        }
        
        # Add external config if provided
        if config_dict:
            metadata['external_config'] = config_dict
            
        # Add additional metadata if provided
        if additional_metadata:
            metadata['additional'] = additional_metadata
        
        # Save to file
        filename = f"{logs_dir}/run_metadata_{run_id}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"💾 Run metadata saved: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save run metadata: {e}")
            return None
    
    def _load_run_metadata(self, metadata_file: str) -> Dict:
        """Load run metadata for reproducibility."""
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"📂 Run metadata loaded: {metadata_file}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load run metadata: {e}")
            return {}
    
    def set_reproducible_state(self, seed: int = None, metadata_file: str = None):
        """Set reproducible random state from seed or metadata file."""
        if metadata_file:
            metadata = self._load_run_metadata(metadata_file)
            if 'random_state' in metadata:
                # Note: Full numpy state restoration is complex, so we'll use seed if available
                self.logger.info("🔄 Loaded metadata - using seed for reproducibility")
        
        if seed is not None:
            np.random.seed(seed)
            self.logger.info(f"🎲 Random seed set: {seed}")
            return seed
        else:
            # Generate and set a random seed
            seed = np.random.randint(0, 2**31 - 1)
            np.random.seed(seed)
            self.logger.info(f"🎲 Random seed generated and set: {seed}")
            return seed
    
    def _save_episode_summary_to_csv(self, episode_summary: Dict, csv_file: str = "logs/episode_summaries.csv"):
        """Save episode summary to CSV for easy analysis and drift detection."""
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # Flatten the summary for CSV format
        flattened_summary = {}
        for key, value in episode_summary.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for subkey, subvalue in value.items():
                    flattened_summary[f"{key}_{subkey}"] = subvalue
            else:
                flattened_summary[key] = value
        
        # Convert to DataFrame
        df_row = pd.DataFrame([flattened_summary])
        
        # Append to CSV file
        try:
            if os.path.exists(csv_file):
                df_row.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                df_row.to_csv(csv_file, mode='w', header=True, index=False)
            self.logger.info(f"📊 Episode summary saved to: {csv_file}")
        except Exception as e:
            self.logger.error(f"Failed to save episode summary to CSV: {e}")

    def get_trade_log(self):
        if self.log_trades_flag:
            return pd.DataFrame(self.trade_log)
        return pd.DataFrame()

    def get_portfolio_history(self):
        """Returns the history of portfolio values at each step."""
        if not self.portfolio_history:
            return pd.Series(dtype=float)
        
        # Ensure lengths match to avoid index mismatch
        portfolio_len = len(self.portfolio_history)
        dates_len = len(self.dates)
        
        if portfolio_len <= dates_len:
            # Normal case: use dates up to portfolio length
            return pd.Series(self.portfolio_history, index=self.dates[:portfolio_len])
        else:
            # Edge case: portfolio history is longer than dates
            # This can happen if portfolio is updated after the last date
            # Truncate portfolio history to match dates length
            self.logger.warning(f"Portfolio history length ({portfolio_len}) > dates length ({dates_len}). Truncating.")
            return pd.Series(self.portfolio_history[:dates_len], index=self.dates)


    def update_risk_constraints(self, constraints: Dict[str, Any]):
        """
        Update risk constraints dynamically during training (for curriculum learning).
        
        Args:
            constraints: Dictionary with new constraint values
        """
        if 'drawdown_cap' in constraints:
            self.max_daily_drawdown_pct = constraints['drawdown_cap']
            self.logger.info(f"🎓 Updated drawdown cap: {self.max_daily_drawdown_pct:.1%}")
        
        if 'lambda_penalty' in constraints:
            # Update penalty lambda if using basic risk management
            if hasattr(self, 'penalty_lambda'):
                self.penalty_lambda = constraints['lambda_penalty']
                self.logger.info(f"🎓 Updated penalty lambda: {self.penalty_lambda}")
            
            # Update advanced reward shaping if enabled
            if self.advanced_reward_shaper:
                # Update Lagrangian constraint lambda
                if hasattr(self.advanced_reward_shaper.lagrangian_manager, 'lambda_value'):
                    # Scale the curriculum lambda for advanced reward shaping
                    new_lambda = constraints['lambda_penalty'] * 0.1  # Scale appropriately
                    self.advanced_reward_shaper.lagrangian_manager.lambda_value = new_lambda
                    self.logger.info(f"🎓 Updated advanced reward shaping lambda: {new_lambda:.3f}")
        
        self.logger.debug(f"🎓 Risk constraints updated: {constraints}")

    def close(self):
        self.logger.info("IntradayTradingEnv closed.")
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Create Dummy Data for Testing ---
    num_total_samples = 200 # Total number of minutes in the test data
    lookback = 5 # Lookback window for features
    num_market_features = 4 # e.g., RSI, EMA_diff, VWAP_dev, hour_sin # Market features ONLY

    # 1. Dummy Price Data (pd.Series with DatetimeIndex) for P&L
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_total_samples) * 0.1)
    price_dates = pd.to_datetime(pd.date_range(start='2023-01-01 09:30', periods=num_total_samples, freq='1min'))
    price_series = pd.Series(prices, index=price_dates, name='close')
    logger.info(f"Price data shape: {price_series.shape}")

    # 2. Dummy Market Feature Data (np.ndarray) - does NOT include position
    num_env_steps = 100 # Let the environment run for 100 steps
    
    if lookback > 1:
        # Shape: (num_env_steps, lookback_window, num_market_features)
        market_feature_array = np.random.rand(num_env_steps, lookback, num_market_features).astype(np.float32)
    else: # lookback is 1 or 0
        # Shape: (num_env_steps, num_market_features)
        market_feature_array = np.random.rand(num_env_steps, num_market_features).astype(np.float32)

    # Price data must align with the market feature data for the environment
    test_price_series_for_env = price_series.iloc[:num_env_steps]

    logger.info(f"Market Feature array shape for env: {market_feature_array.shape}")
    logger.info(f"Price series shape for env: {test_price_series_for_env.shape}")

    # --- Initialize Environment ---
    env_config = {
        'processed_feature_data': market_feature_array, # Pass market features only
        'price_data': test_price_series_for_env,
        'initial_capital': 50000.0,
        'lookback_window': lookback,
        'max_daily_drawdown_pct': 0.05, # 5%
        'hourly_turnover_cap': 2.0, # Example: 2x capital per hour
        'transaction_cost_pct': 0.0005, # 0.05%
        'reward_scaling': 1.0, # No scaling for test
        'max_episode_steps': 50, # Test truncation
        'log_trades': True,
        'turnover_penalty_factor': 0.05, # Penalize 5% of excess turnover value
        # Showcase new parameters (Step 8)
        'position_sizing_pct_capital': 0.30, # Use 30% of capital for sizing
        'trade_cooldown_steps': 5,           # Wait 5 steps after a trade (ANTI-PING-PONG)
        'terminate_on_turnover_breach': True,# Terminate if turnover is >> cap
        'turnover_termination_threshold_multiplier': 1.5, # Terminate if turnover > 1.5x cap
        # Enhanced turnover enforcement
        'turnover_exponential_penalty_factor': 0.2, # Strong quadratic penalty
        'turnover_termination_penalty_pct': 0.10    # 10% portfolio penalty on termination
    }
    env = IntradayTradingEnv(**env_config)

    # --- Test Environment Reset ---
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape} (Expected: {env.observation_space.shape})")
    logger.info(f"Initial info: {info}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch after reset."
    if lookback > 1:
        assert obs[0, -1] == 0.0, "Position feature in initial obs should be 0 (flat)"
    else:
        assert obs[-1] == 0.0, "Position feature in initial obs should be 0 (flat)"


    # --- Test Environment Step ---
    terminated = False
    truncated = False
    total_reward_unscaled = 0
    for i in range(env_config['max_episode_steps'] + 5): # Test going beyond max_episode_steps
        action = env.action_space.sample() # Random action
        # action = 2 # Force buy for testing
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.debug(
            f"Step {info['current_step']}: Action={action}, Price={info['current_price']:.2f}, "
            f"Reward={reward:.4f}, PortfolioVal={info['portfolio_value']:.2f}, PosSignal={info['current_position_signal']}, PosQty={info['current_position_quantity']:.2f}, "
            f"Term={terminated}, Trunc={truncated}"
        )
        if lookback > 1:
             assert obs[0, -1] == info['current_position_signal'], f"Position feature in obs ({obs[0,-1]}) != env state ({info['current_position_signal']})"
        else:
             assert obs[-1] == info['current_position_signal'], f"Position feature in obs ({obs[-1]}) != env state ({info['current_position_signal']})"

        total_reward_unscaled += reward / env_config['reward_scaling'] # Store unscaled for direct comparison
        
        if 'last_trade_details' in info:
            logger.info(f"Trade executed: {info['last_trade_details']}")

        if terminated or truncated:
            logger.info(f"Episode finished at step {i+1}. Reason: {'Terminated' if terminated else 'Truncated'}")
            logger.info(f"Final Portfolio Value: {info['portfolio_value']:.2f}, Total Unscaled Reward: {total_reward_unscaled:.3f}")
            final_pnl_pct = (info['portfolio_value'] - env_config['initial_capital']) / env_config['initial_capital'] * 100
            logger.info(f"Final P&L: {final_pnl_pct:.2f}%")
            break
            
    assert terminated or truncated, "Episode should have ended."

    # --- Test Trade Log and Portfolio History ---
    trade_log_df = env.get_trade_log()
    if not trade_log_df.empty:
        logger.info("\n--- Trade Log ---")
        print(trade_log_df.head())
    else:
        logger.info("\n--- No trades executed or logging disabled ---")
        
    portfolio_history_s = env.get_portfolio_history()
    if not portfolio_history_s.empty:
        logger.info("\n--- Portfolio History (first 5) ---")
        print(portfolio_history_s.head())
        assert len(portfolio_history_s) == (info['current_step'] + 1 if not truncated else info['current_step']), "Portfolio history length mismatch"

    env.close()
    logger.info("\nIntradayTradingEnv standalone test complete.")
