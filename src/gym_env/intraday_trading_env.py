# src/gym_env/intraday_trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

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
                 processed_feature_data: np.ndarray, # Market features only
                 price_data: pd.Series, # Unscaled close prices for P&L, aligned with feature_data
                 initial_capital: float = 100000.0,
                 lookback_window: int = 1,
                 max_daily_drawdown_pct: float = 0.02,
                 hourly_turnover_cap: float = 5.0,
                 transaction_cost_pct: float = 0.001,
                 reward_scaling: float = 1.0,
                 max_episode_steps: int = None,
                 log_trades: bool = True,
                 turnover_penalty_factor: float = 0.01, # Penalty per unit of excess turnover
                 position_sizing_pct_capital: float = 0.25, # Pct of capital to use for sizing
                 trade_cooldown_steps: int = 0, # Number of steps to wait after a trade
                 terminate_on_turnover_breach: bool = False, # Terminate if turnover cap breached significantly
                 turnover_termination_threshold_multiplier: float = 2.0 # e.g. 2x the cap
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
            position_sizing_pct_capital (float): Percentage of current capital to use for sizing new positions.
            trade_cooldown_steps (int): Minimum number of steps to wait before executing another trade.
            terminate_on_turnover_breach (bool): Whether to terminate the episode if hourly turnover cap is breached by a significant margin.
            turnover_termination_threshold_multiplier (float): Multiplier for `hourly_turnover_cap` to determine termination threshold.
        """
        super().__init__()
        self.logger = logging.getLogger(f"RLTradingPlatform.Env.IntradayTradingEnv")
        
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
        self.trade_cooldown_steps = trade_cooldown_steps
        self.terminate_on_turnover_breach = terminate_on_turnover_breach
        self.turnover_termination_threshold_multiplier = turnover_termination_threshold_multiplier

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

        self.hourly_traded_value = 0.0 # Sum of absolute value of trades in current hour (rolling window)
        self.trades_this_hour = [] # Stores (timestamp, trade_value, shares_traded)
        self.steps_since_last_trade = 0 

        if self.log_trades_flag:
            self.trade_log = []
        self.portfolio_history = [] # To log portfolio value at each step

        self.reset()


    def _get_observation(self):
        """Constructs the observation: market features + current position."""
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

    def _get_current_price(self) -> float:
        return self.price_data.iloc[self.current_step]

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
        
        # This definition of portfolio_value depends on how cash changes upon opening positions.
        # Let's define portfolio_value = self.current_capital (cash) + market_value_of_long_pos - market_value_of_covering_short_pos
        # If self.current_capital already reflects cash used/gained from trades:
        long_value = self.position_quantity * current_price if self.current_position == 1 else 0
        # For short, if entry_price is when short was opened:
        # cash increased by entry_price * qty. Obligation is current_price * qty.
        # Net value of short pos relative to initial cash state: (entry_price - current_price) * qty
        # Let's use a simpler P&L approach for reward and track current_capital as primary cash.
        # The reward will be based on change in portfolio value.
        # Previous capital before this step: self.portfolio_value
        
        # Portfolio value = current cash + value of (shares * current_price) if long
        # If short: portfolio value = current cash (which includes proceeds from shorting) - cost_to_cover (shares * current_price)
        # This needs a consistent definition. Let's use:
        # Portfolio Value = Cash + (current_price * quantity_long) - (current_price * quantity_short)
        # where quantity_short is positive if short.
        # For our simplified model (position is -1, 0, 1, and fixed quantity or full capital deployment):
        
        if self.current_position == 1: # Long
            self.portfolio_value = self.current_capital + (self.position_quantity * current_price)
        elif self.current_position == -1: # Short
            # current_capital already includes proceeds from short sale at self.entry_price
            # To mark-to-market, subtract the current cost to buy back
            self.portfolio_value = self.current_capital - (self.position_quantity * current_price)
        else: # Flat
            self.portfolio_value = self.current_capital
            
    def _handle_new_day(self):
        current_date = self.dates[self.current_step].date()
        if self.last_date is None or current_date != self.last_date:
            previous_day_realized_pnl = self.daily_pnl # Store before reset
            self.logger.info(f"New trading day: {current_date}. Previous day REALIZED P&L: {previous_day_realized_pnl:.2f}")
            self.daily_pnl = 0.0 # Reset daily *realized* P&L counter
            self.start_of_day_portfolio_value = self.portfolio_value # Portfolio value at start of new day
            self.last_date = current_date
            # Hourly turnover tracking resets implicitly via _update_hourly_trades if it spans days,
            # but should be explicitly cleared daily for cleaner accounting if desired.
            self.trades_this_hour.clear() 
            self.hourly_traded_value = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_position = 0
        self.current_capital = self.initial_capital # Cash
        self.portfolio_value = self.initial_capital # Total value
        self.entry_price = 0.0
        self.position_quantity = 0.0 # Number of shares/units

        self.daily_pnl = 0.0 # Realized P&L
        self.start_of_day_portfolio_value = self.initial_capital
        self.last_date = None
        
        self.hourly_traded_value = 0.0
        self.trades_this_hour = []
        self.steps_since_last_trade = self.trade_cooldown_steps # Initialize to allow immediate trade if cooldown is >0

        if self.log_trades_flag:
            self.trade_log = []
        self.portfolio_history = [self.initial_capital]


        self._handle_new_day() 
        
        observation = self._get_observation()
        info = self._get_info()
        
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
        assert isinstance(action, (int, np.integer)), f"Action must be int, got {type(action)}"
        desired_position_signal = self._action_map[action] # -1 (Sell), 0 (Hold), 1 (Buy)
        current_price = self._get_current_price()
        timestamp = self.dates[self.current_step]

        # Store portfolio value before action
        portfolio_value_before_action = self.portfolio_value
        
        self.steps_since_last_trade += 1 # Increment regardless of action
        realized_pnl_this_step = 0.0
        transaction_executed = False
        trade_value_executed = 0.0 # Absolute monetary value of stock traded
        shares_traded_this_step = 0.0

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

        # --- Cooldown Check ---
        if self.steps_since_last_trade < self.trade_cooldown_steps and desired_position_signal != self.current_position:
            self.logger.debug(f"Step {self.current_step}: Trade cooldown active. Steps since last trade: {self.steps_since_last_trade}/{self.trade_cooldown_steps}. Forcing HOLD.")
            desired_position_signal = self.current_position # Force hold
            # Action itself is not changed, only its effect for changing position.

        if desired_position_signal != self.current_position:
            transaction_executed = True
            self.steps_since_last_trade = 0 # Reset cooldown counter
            
            # 1. Exit current position if any (calculate realized P&L)
            if self.current_position == 1: # Was long, now closing or flipping to short
                realized_pnl_this_step = (current_price - self.entry_price) * self.position_quantity
                self.current_capital += (self.position_quantity * current_price) # Add sale proceeds to cash
                self.current_capital -= (self.transaction_cost_pct * self.position_quantity * current_price) # Exit cost
                trade_value_executed += self.position_quantity * current_price
                shares_traded_this_step += self.position_quantity
                self.logger.debug(f"Closed LONG: {self.position_quantity} @ {current_price:.2f}. Entry: {self.entry_price:.2f}. P&L: {realized_pnl_this_step:.2f}")
            elif self.current_position == -1: # Was short, now closing or flipping to long
                realized_pnl_this_step = (self.entry_price - current_price) * self.position_quantity
                self.current_capital -= (self.position_quantity * current_price) # Cost to buy back
                self.current_capital -= (self.transaction_cost_pct * self.position_quantity * current_price) # Exit cost
                trade_value_executed += self.position_quantity * current_price
                shares_traded_this_step += self.position_quantity
                self.logger.debug(f"Covered SHORT: {self.position_quantity} @ {current_price:.2f}. Entry: {self.entry_price:.2f}. P&L: {realized_pnl_this_step:.2f}")

            self.daily_pnl += realized_pnl_this_step # Accumulate daily *realized* P&L
            self.position_quantity = 0 # Flat after exiting

            # 2. Enter new position if not going flat
            if desired_position_signal == 1: # Buy to go Long
                # Determine quantity: for simplicity, assume we can buy 1 share for now.
                # A real system would use capital / price or a risk-managed quantity.
                # Let's assume, for this skeleton, we try to invest a significant portion of capital.
                # This is a placeholder for proper position sizing.
                if self.current_capital > current_price: # Can afford at least one share
                    # Use the new position_sizing_pct_capital parameter
                    capital_to_allocate = self.current_capital * self.position_sizing_pct_capital
                    self.position_quantity = np.floor(capital_to_allocate / current_price)
                    if self.position_quantity == 0 and self.current_capital > current_price : self.position_quantity = 1 # min 1 share if possible
                else:
                    self.position_quantity = 0 # Cannot afford

                if self.position_quantity > 0:
                    self.entry_price = current_price
                    self.current_capital -= (self.position_quantity * self.entry_price) # Cash used
                    self.current_capital -= (self.transaction_cost_pct * self.position_quantity * self.entry_price) # Entry cost
                    trade_value_executed += self.position_quantity * self.entry_price
                    shares_traded_this_step += self.position_quantity
                    self.logger.debug(f"Opened LONG: {self.position_quantity} @ {self.entry_price:.2f}")
                else: # Cannot afford to go long
                    desired_position_signal = 0 # Force flat

            elif desired_position_signal == -1: # Sell to go Short
                # Short selling quantity can also be based on a percentage of capital (notional value)
                if self.current_capital > 0 : # Simplified check for available margin/capital
                     capital_to_allocate_notional = self.current_capital * self.position_sizing_pct_capital
                     self.position_quantity = np.floor(capital_to_allocate_notional / current_price)
                     if self.position_quantity == 0 and self.current_capital > current_price : self.position_quantity = 1 # min 1 share if possible
                else:
                    self.position_quantity = 0

                if self.position_quantity > 0:
                    self.entry_price = current_price
                    self.current_capital += (self.position_quantity * self.entry_price) # Cash received from shorting (simplified)
                    self.current_capital -= (self.transaction_cost_pct * self.position_quantity * self.entry_price) # Entry cost
                    trade_value_executed += self.position_quantity * self.entry_price
                    shares_traded_this_step += self.position_quantity
                    self.logger.debug(f"Opened SHORT: {self.position_quantity} @ {self.entry_price:.2f}")
                else: # Cannot short (e.g. no capital for margin, though simplified here)
                    desired_position_signal = 0 # Force flat
            
            self.current_position = desired_position_signal
            if self.current_position == 0: # If ended up flat
                self.entry_price = 0.0
                self.position_quantity = 0.0
        
        # Update portfolio value after any transactions
        self._update_portfolio_value()
        reward = (self.portfolio_value - portfolio_value_before_action) # Reward is change in total portfolio value

        # --- Log trade if executed ---
        if transaction_executed:
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
            
            # Update turnover tracking
            self._update_hourly_trades(timestamp) # Prune old trades
            self.trades_this_hour.append((timestamp, trade_value_executed, shares_traded_this_step))
            self.hourly_traded_value += trade_value_executed


        # --- Risk Management Checks ---
        terminated = False
        # 1. Max Daily Drawdown (based on portfolio_value)
        current_drawdown_pct = (self.start_of_day_portfolio_value - self.portfolio_value) / self.start_of_day_portfolio_value \
                               if self.start_of_day_portfolio_value > 0 else 0
        
        if current_drawdown_pct > self.max_daily_drawdown_pct:
            self.logger.warning(f"Step {self.current_step}: Max daily drawdown breached! Drawdown: {current_drawdown_pct*100:.2f}%, Limit: {self.max_daily_drawdown_pct*100:.2f}%. Portfolio Value: {self.portfolio_value:.2f}")
            terminated = True
            # drawdown_penalty = self.initial_capital * 0.1 # Example fixed penalty
            # reward -= drawdown_penalty 
            # Reward already reflects the loss in portfolio value. Additional penalty can be added if desired.

        # 2. Hourly Turnover Cap
        if self.start_of_day_portfolio_value > 0:
            current_hourly_turnover_ratio = self.hourly_traded_value / self.start_of_day_portfolio_value
            if current_hourly_turnover_ratio > self.hourly_turnover_cap:
                self.logger.warning(f"Step {self.current_step}: Hourly turnover cap breached! Ratio: {current_hourly_turnover_ratio:.2f}x, Limit: {self.hourly_turnover_cap:.2f}x.")
                # Apply penalty based on excess turnover
                excess_turnover = current_hourly_turnover_ratio - self.hourly_turnover_cap
                turnover_penalty = excess_turnover * self.start_of_day_portfolio_value * self.turnover_penalty_factor
                reward -= turnover_penalty
                
                if self.terminate_on_turnover_breach:
                    termination_threshold = self.hourly_turnover_cap * self.turnover_termination_threshold_multiplier
                    if current_hourly_turnover_ratio > termination_threshold:
                        self.logger.warning(f"Step {self.current_step}: Terminating due to excessive hourly turnover! Ratio: {current_hourly_turnover_ratio:.2f}x, Cap: {self.hourly_turnover_cap:.2f}x, Termination Threshold: {termination_threshold:.2f}x")
                        terminated = True
        
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

        info = self._get_info()
        info['portfolio_value'] = self.portfolio_value # Ensure latest is passed
        info['realized_pnl_step'] = realized_pnl_this_step
        if transaction_executed:
            info['last_trade_details'] = self.trade_log[-1] if self.trade_log else {}
        
        self.logger.debug(
            f"Step: {self.current_step-1}, Action: {action}(Signal:{self._action_map[action]}), "
            f"Price: {current_price:.2f}, Reward(scaled): {reward * self.reward_scaling:.4f}, PortfolioVal: {self.portfolio_value:.2f}, "
            f"Position: {self.current_position}({self.position_quantity:.2f} units), Term: {terminated}, Trunc: {truncated}"
        )
        
        return observation, reward * self.reward_scaling, terminated, truncated, info

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
            "current_hourly_traded_value": self.hourly_traded_value
        }

    def get_trade_log(self):
        if self.log_trades_flag:
            return pd.DataFrame(self.trade_log)
        return pd.DataFrame()

    def get_portfolio_history(self):
        """Returns the history of portfolio values at each step."""
        return pd.Series(self.portfolio_history, index=self.dates[:len(self.portfolio_history)])


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
        'trade_cooldown_steps': 2,           # Wait 2 steps after a trade
        'terminate_on_turnover_breach': True,# Terminate if turnover is >> cap
        'turnover_termination_threshold_multiplier': 1.5 # Terminate if turnover > 1.5x cap
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
