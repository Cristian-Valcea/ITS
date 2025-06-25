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
    and current portfolio status (e.g., position held: -1 for short, 0 for flat, 1 for long).
    If `lookback_window > 1`, observations are sequences of shape (lookback_window, num_features).
    Otherwise, observations are flat arrays of shape (num_features,).

    **Action Space:**
    Discrete: 0 (Sell/Short), 1 (Hold/Stay Flat), 2 (Buy/Long).
    (Could be extended to include position sizing).

    **Reward Function:**
    Primarily based on realized or unrealized P&L from trades.
    Penalties for transaction costs, excessive turnover, or risk limit breaches.

    **Episode Termination:**
    - End of the trading data (e.g., end of a trading day or backtest period).
    - Max daily drawdown limit reached (e.g., 2% of initial capital).
    - Optional: other conditions like holding a position for too long, etc.
    """
    metadata = {'render_modes': ['human', 'logs'], 'render_fps': 1}

    def __init__(self,
                 processed_feature_data: np.ndarray,
                 price_data: pd.Series, # Unscaled close prices for P&L, aligned with feature_data
                 initial_capital: float = 100000.0,
                 lookback_window: int = 1, # Number of past timesteps in one observation
                 max_daily_drawdown_pct: float = 0.02, # 2% max daily drawdown
                 hourly_turnover_cap: float = 5.0, # Max 5x turnover of capital per hour (example)
                 transaction_cost_pct: float = 0.001, # 0.1% per trade (buy or sell)
                 reward_scaling: float = 1.0, # Scales the P&L reward
                 max_episode_steps: int = None, # If None, episode runs through all data
                 log_trades: bool = True
                 ):
        """
        Args:
            processed_feature_data (np.ndarray): The feature data for the environment.
                Shape: (num_samples, num_features) or (num_samples, lookback_window, num_features).
                Assumes features are already normalized/scaled as needed.
            price_data (pd.Series): Series of actual (unscaled) closing prices, aligned with
                                    the first dimension of `processed_feature_data`. Used for P&L calculation.
                                    Length must match `processed_feature_data.shape[0]`.
            initial_capital (float): Starting capital for trading.
            lookback_window (int): Number of past timesteps included in each observation.
                                   If > 1, `processed_feature_data` should be pre-sequenced.
            max_daily_drawdown_pct (float): Maximum percentage of capital that can be lost in a day.
            hourly_turnover_cap (float): Maximum turnover (total value of trades / capital) allowed per hour.
            transaction_cost_pct (float): Cost per transaction as a percentage of trade value.
            reward_scaling (float): Multiplier for the P&L reward.
            max_episode_steps (int): Max steps per episode. If None, runs until data ends or termination.
            log_trades (bool): Whether to maintain a log of trades.
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

        self.feature_data = processed_feature_data
        self.price_data = price_data # This should be the actual prices for P&L
        self.dates = price_data.index # Assuming price_data has a DatetimeIndex

        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.hourly_turnover_cap = hourly_turnover_cap # TODO: Implement hourly turnover tracking
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else len(self.feature_data)
        self.log_trades_flag = log_trades

        # --- Define Action Space ---
        # Action: 0 (Sell/Go Short), 1 (Hold/Stay Flat), 2 (Buy/Go Long)
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1, 1: 0, 2: 1} # Maps action to position: -1 (short), 0 (flat), 1 (long)

        # --- Define Observation Space ---
        # Features from data + 1 for current position encoding
        # If lookback_window > 1, features are (lookback_window, num_underlying_features)
        # We add portfolio state (current position) to the features.
        # For simplicity, we'll append the position to the last timestep's features if sequential,
        # or to the flat feature vector if not.
        
        if self.lookback_window > 1:
            if self.feature_data.ndim != 3:
                self.logger.error(f"Feature data for lookback_window > 1 must be 3D (samples, window, features), got {self.feature_data.ndim}D.")
                raise ValueError("Feature data shape mismatch for lookback_window.")
            self.num_underlying_features = self.feature_data.shape[2]
            # Observation: sequence of features + current position (scalar, appended to last step features or as separate channel)
            # For SB3, it's often easier if the observation space is Box and consistently shaped.
            # One way: append position to the features of each step in the window.
            # Another way: concatenate position as a new feature at the last step.
            # Let's make it simple: the observation will be the feature data as is,
            # and the current position will be part of the internal state, influencing reward and next state.
            # The model should learn to infer position if it's not explicitly in obs.
            # However, standard practice is to include it.
            # Let's assume the `processed_feature_data` ALREADY includes position if desired as a feature.
            # If not, we define the space to accommodate it.
            # For this version, let's assume `processed_feature_data` is purely market data.
            # The observation will be a Dict space: {'market': Box, 'position': Discrete}
            # Or, we can flatten and concatenate. Let's try flattening for now.
            # obs_shape = (self.lookback_window, self.num_underlying_features + 1) # +1 for position
            # Using a simpler approach for now: observation space is just the market features.
            # The agent needs to learn to map these to actions considering its implicit state.
            # This is common in many SB3 examples for simplicity, though adding position to obs is better.
            
            # To make it compatible with standard CNN/MLP policies in SB3, the observation space
            # for sequential data is often (num_features, lookback_window) if using Conv1D,
            # or flattened (lookback_window * num_features).
            # Or if using MlpLstmPolicy, it expects (num_features,).
            # Let's assume data is (samples, lookback_window, features) and policy handles it.
            # The observation space should match one item from self.feature_data.
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.lookback_window, self.num_underlying_features), # Matches one sample
                dtype=np.float32
            )
            # We will also add a separate state for current position for the agent to use.
            # This can be done by wrapping the environment or by using a Dict observation space.
            # For now, let's assume the policy can handle this implicitly or the Orchestrator adds it.
            # A common practice is to append position to the features.
            # If we do that, shape becomes (lookback_window, num_underlying_features + 1)
            # For this skeleton, we keep obs space as just market data.
            # TODO: Revisit observation space to include position explicitly for better agent learning.
            # For now, agent must infer position or environment must be wrapped.

        else: # lookback_window is 1 or 0 (current step only)
            if self.feature_data.ndim != 2:
                self.logger.error(f"Feature data for lookback_window <= 1 must be 2D (samples, features), got {self.feature_data.ndim}D.")
                raise ValueError("Feature data shape mismatch.")
            self.num_underlying_features = self.feature_data.shape[1]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_underlying_features,), # +1 if adding position here
                dtype=np.float32
            )
        
        self.logger.info(f"Observation space: {self.observation_space}")
        self.logger.info(f"Action space: {self.action_space}")

        # Internal state variables
        self.current_step = 0
        self.current_position = 0  # -1 short, 0 flat, 1 long
        self.current_capital = self.initial_capital
        self.entry_price = 0.0
        self.daily_pnl = 0.0
        self.start_of_day_capital = self.initial_capital # Resets each day
        self.last_date = None # To track day changes

        if self.log_trades_flag:
            self.trade_log = [] # List to store trade details

        self.reset()


    def _get_observation(self):
        """Constructs the observation for the current step."""
        # Observation is the market features at the current step.
        # If lookback_window > 1, self.feature_data is already (samples, window, features).
        # So, self.feature_data[self.current_step] is (window, features).
        # If lookback_window <= 1, self.feature_data is (samples, features).
        # So, self.feature_data[self.current_step] is (features,).
        obs = self.feature_data[self.current_step].astype(np.float32)
        
        # TODO: Explicitly include current_position in the observation.
        # This is crucial for the agent to make informed decisions.
        # Example if not using Dict space (flattening):
        # if self.lookback_window > 1:
        #     # Append position to each step in the lookback window
        #     position_feature = np.full((self.lookback_window, 1), self.current_position)
        #     obs = np.concatenate((obs, position_feature), axis=1).astype(np.float32)
        # else:
        #     obs = np.append(obs, self.current_position).astype(np.float32)
        # This would require changing observation_space shape definition too.
        # For now, this skeleton assumes the policy network implicitly handles position
        # or a wrapper adds it. This is a simplification.
        return obs

    def _get_current_price(self) -> float:
        """Returns the current market price (e.g., close price)."""
        return self.price_data.iloc[self.current_step]

    def _handle_new_day(self):
        """Resets daily tracking variables if a new day starts."""
        current_date = self.dates[self.current_step].date()
        if self.last_date is None or current_date != self.last_date:
            self.logger.info(f"New trading day: {current_date}. Previous day P&L: {self.daily_pnl:.2f}")
            self.daily_pnl = 0.0 # Reset daily P&L counter
            self.start_of_day_capital = self.current_capital # Capital at the start of this new day
            self.last_date = current_date
            # TODO: Reset hourly turnover tracking here if implemented.

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important for reproducibility with new Gymnasium versions

        self.current_step = 0 # Or a random start if desired, but usually 0 for backtests
        self.current_position = 0  # Start flat
        self.current_capital = self.initial_capital
        self.entry_price = 0.0
        self.daily_pnl = 0.0
        self.start_of_day_capital = self.initial_capital
        self.last_date = None
        
        if self.log_trades_flag:
            self.trade_log = []

        self._handle_new_day() # Initialize date tracking
        
        observation = self._get_observation()
        info = self._get_info()
        
        self.logger.debug(f"Environment reset. Initial capital: {self.initial_capital:.2f}")
        return observation, info

    def step(self, action: int):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action chosen by the agent (0: Sell, 1: Hold, 2: Buy).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        desired_position = self._action_map[action]
        current_price = self._get_current_price()
        timestamp = self.dates[self.current_step]
        
        reward = 0.0
        transaction_executed = False
        trade_details = {} # For logging

        # --- Calculate P&L from previous position (if any) ---
        # This is mark-to-market P&L for holding a position over the step
        if self.current_position == 1: # Was long
            reward += (current_price - self.entry_price) # P&L for this step if still long
        elif self.current_position == -1: # Was short
            reward += (self.entry_price - current_price) # P&L for this step if still short
        
        # --- Execute Trade based on desired_position vs current_position ---
        if desired_position != self.current_position:
            transaction_executed = True
            cost = 0.0
            
            # Calculate cost of exiting current position (if any)
            if self.current_position != 0:
                cost += self.transaction_cost_pct * self.entry_price # Cost on entry value
                self.logger.debug(f"Step {self.current_step}: Exiting position {self.current_position} at {current_price:.2f}. Entry: {self.entry_price:.2f}. P&L from trade: {reward:.2f}")
                # Log exited trade
                if self.log_trades_flag:
                    trade_details.update({
                        'exit_time': timestamp,
                        'exit_price': current_price,
                        'position_type': 'long' if self.current_position == 1 else 'short',
                        'profit': reward, # This reward is before this exit's cost
                        'cost_exit': cost
                    })


            # Update capital with P&L from closed trade (reward accumulated so far for this trade)
            self.current_capital += reward # Add P&L
            self.daily_pnl += reward # Add to daily P&L tracker

            # Reset reward for the new action's outcome for this step
            reward = 0 

            # Enter new position (if not going flat)
            if desired_position != 0:
                self.entry_price = current_price
                entry_cost = self.transaction_cost_pct * self.entry_price
                cost += entry_cost
                self.logger.debug(f"Step {self.current_step}: Entering position {desired_position} at {current_price:.2f}. Cost: {entry_cost:.2f}")
                if self.log_trades_flag:
                    trade_details.update({
                        'entry_time': timestamp,
                        'entry_price': current_price,
                        'new_position': 'long' if desired_position == 1 else 'short',
                        'cost_entry': entry_cost
                    })
            else: # Going flat
                self.entry_price = 0.0
                self.logger.debug(f"Step {self.current_step}: Moving to flat position.")
                if self.log_trades_flag and 'new_position' not in trade_details: # ensure log if only exiting
                     trade_details.update({'new_position': 'flat'})


            self.current_capital -= cost # Deduct total transaction costs for this change
            self.daily_pnl -= cost
            reward -= cost # Penalize reward by transaction costs

            self.current_position = desired_position
        else: # Holding position
            self.logger.debug(f"Step {self.current_step}: Holding position {self.current_position}. Price: {current_price:.2f}. Mark-to-market P&L for step: {reward:.2f}")
            # Capital and daily_pnl are updated with this step's P&L (reward) later
            pass


        # Update capital and daily P&L with the step's mark-to-market P&L if position held
        if not transaction_executed: # If just holding, add the calculated reward
            self.current_capital += reward
            self.daily_pnl += reward
        
        if self.log_trades_flag and transaction_executed:
            # Enrich trade_details
            trade_details['step'] = self.current_step
            trade_details['capital_after_trade'] = self.current_capital
            trade_details['daily_pnl_after_trade'] = self.daily_pnl
            self.trade_log.append(trade_details)

        # --- Risk Management Checks ---
        terminated = False
        # 1. Max Daily Drawdown
        current_drawdown = (self.start_of_day_capital - self.current_capital) / self.start_of_day_capital
        if current_drawdown > self.max_daily_drawdown_pct:
            self.logger.warning(f"Step {self.current_step}: Max daily drawdown breached! Drawdown: {current_drawdown*100:.2f}%, Limit: {self.max_daily_drawdown_pct*100:.2f}%. Capital: {self.current_capital:.2f}")
            terminated = True
            reward -= self.config.get('drawdown_penalty', self.initial_capital * 0.1) # Significant penalty

        # TODO: 2. Hourly Turnover Cap
        # This requires tracking trades within the current hour.
        # If breached, could prevent further trades for the hour or penalize.

        # --- Advance Time ---
        self.current_step += 1

        # --- Check for Episode End ---
        truncated = False
        if self.current_step >= len(self.feature_data):
            self.logger.info(f"Episode ended: Reached end of data at step {self.current_step}.")
            truncated = True # End of data
        if self._max_episode_steps is not None and self.current_step >= self._max_episode_steps:
            self.logger.info(f"Episode truncated: Reached max episode steps {self._max_episode_steps}.")
            truncated = True # Max steps reached

        if not truncated and not terminated: # If episode continues
             self._handle_new_day() # Check if a new day has started for next step
             observation = self._get_observation()
        else: # Episode is ending
            # If ending due to termination (e.g. drawdown), provide a final observation.
            # If truncated (end of data/steps), the last observation might not be valid or needed.
            # SB3 typically expects an observation even if terminated/truncated.
            # We can provide the current one, or a zero observation if it's truly the end.
            if self.current_step >= len(self.feature_data) : # Truly out of data
                 # Create a dummy observation if needed, e.g., zeros, or use last valid one
                 # For simplicity, let's use the last valid observation before incrementing step
                 # Or, if obs relies on current_step, it might error.
                 # A common way is to ensure get_observation can handle current_step being out of bounds.
                 # For now, let's assume the last call to _get_observation() in a valid step is sufficient.
                 # However, the current_step was already incremented.
                 # Let's provide a zero observation for safety if out of bounds.
                 if self.lookback_window > 1:
                     observation = np.zeros( (self.lookback_window, self.num_underlying_features), dtype=np.float32)
                 else:
                     observation = np.zeros( (self.num_underlying_features,), dtype=np.float32)
            else: # Terminated early, or truncated by max_steps but still have data
                 observation = self._get_observation()


        info = self._get_info()
        info['capital'] = self.current_capital
        info['current_position'] = self.current_position
        info['daily_pnl'] = self.daily_pnl
        if transaction_executed:
            info['trade_details'] = trade_details # Last trade details

        self.logger.debug(
            f"Step: {self.current_step-1}, Action: {action}({desired_position}), "
            f"Price: {current_price:.2f}, Reward: {reward * self.reward_scaling:.4f}, Capital: {self.current_capital:.2f}, "
            f"Position: {self.current_position}, Term: {terminated}, Trunc: {truncated}"
        )
        
        return observation, reward * self.reward_scaling, terminated, truncated, info

    def render(self, mode='human'):
        """Renders the environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Price: {self._get_current_price():.2f}, "
                  f"Position: {self.current_position}, Capital: {self.current_capital:.2f}, "
                  f"Daily P&L: {self.daily_pnl:.2f}")
        elif mode == 'logs':
            # Logging is done throughout the step method via self.logger
            pass
        else:
            super().render(mode=mode) # Raise error for unsupported modes

    def _get_info(self):
        """Returns a dictionary with auxiliary information about the current state."""
        return {
            "current_step": self.current_step,
            "timestamp": self.dates[min(self.current_step, len(self.dates)-1)], # Handle end of episode
            "current_price": self._get_current_price() if self.current_step < len(self.price_data) else np.nan,
            "current_position": self.current_position,
            "current_capital": self.current_capital,
            "entry_price": self.entry_price,
            "daily_pnl": self.daily_pnl,
            "start_of_day_capital": self.start_of_day_capital,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "transaction_cost_pct": self.transaction_cost_pct
        }

    def get_trade_log(self):
        if self.log_trades_flag:
            return pd.DataFrame(self.trade_log)
        return pd.DataFrame() # Empty if not logging

    def close(self):
        """Performs any necessary cleanup."""
        self.logger.info("IntradayTradingEnv closed.")
        # Usually, nothing complex needed for simple simulation envs unless using external resources.
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Create Dummy Data for Testing ---
    num_total_samples = 200 # Total number of minutes in the test data
    lookback = 5 # Lookback window for features
    num_market_features = 4 # e.g., RSI, EMA_diff, VWAP_dev, hour_sin

    # 1. Dummy Price Data (pd.Series with DatetimeIndex)
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_total_samples) * 0.1)
    price_dates = pd.to_datetime(pd.date_range(start='2023-01-01 09:30', periods=num_total_samples, freq='1min'))
    price_series = pd.Series(prices, index=price_dates, name='close')
    logger.info(f"Price data shape: {price_series.shape}")

    # 2. Dummy Feature Data (np.ndarray)
    # If lookback_window > 1, feature data should be (num_samples_env, lookback_window, num_market_features)
    # num_samples_env = num_total_samples - lookback + 1, if sequences are pre-generated this way.
    # Or, if FeatureAgent returns (num_total_samples, num_market_features) and env does sequencing,
    # this needs adjustment.
    # Let's assume FeatureAgent ALREADY creates sequences.
    # So, processed_feature_data.shape[0] == price_data.shape[0] IF price_data is also sliced.
    # This means FeatureAgent's output `feature_sequences` and `price_data_for_env` are used.
    
    # For this test, let's simulate that `feature_data` is (num_total_samples, lookback, num_market_features)
    # and `price_data` is (num_total_samples).
    # This implies that at step `t`, observation is features from `t-lookback+1` to `t`.
    # And price at `t` is `price_data[t]`.
    
    # Simpler: Assume FeatureAgent gives features that are already "ready" for each step.
    # If lookback = 5, then feature_data[i] is a sequence of 5 prior feature sets.
    # price_data[i] is the price at the END of that 5th feature set.
    # This means the length of feature_data and price_data should match.
    
    # Example: Feature data has shape (num_steps, lookback_window, num_features_per_step)
    # price_data has shape (num_steps,)
    # num_steps = num_total_samples - lookback + 1 (if features are from raw data of this length)
    # OR, if padding is used, num_steps = num_total_samples
    
    # Let's use the convention from FeatureAgent:
    # `feature_sequences` has shape (num_samples_after_seq, lookback_window, num_features)
    # `price_data_for_env` has shape (num_samples_after_seq,)
    # where num_samples_after_seq = total_raw_points - nan_drops - lookback_window + 1
    
    # For this standalone test:
    num_env_steps = 100 # Let the environment run for 100 steps
    
    if lookback > 1:
        feature_array = np.random.rand(num_env_steps, lookback, num_market_features).astype(np.float32)
    else: # lookback is 1 or 0
        feature_array = np.random.rand(num_env_steps, num_market_features).astype(np.float32)

    # Price data for these env_steps (must be same length as first dim of feature_array)
    test_price_series = price_series.iloc[:num_env_steps] # Take first num_env_steps prices

    logger.info(f"Feature array shape for env: {feature_array.shape}")
    logger.info(f"Price series shape for env: {test_price_series.shape}")

    # --- Initialize Environment ---
    env_config = {
        'processed_feature_data': feature_array,
        'price_data': test_price_series,
        'initial_capital': 50000.0,
        'lookback_window': lookback,
        'max_daily_drawdown_pct': 0.05, # 5%
        'transaction_cost_pct': 0.0005, # 0.05%
        'reward_scaling': 1.0, # No scaling for test
        'max_episode_steps': 50, # Test truncation
        'log_trades': True
    }
    env = IntradayTradingEnv(**env_config)

    # --- Test Environment Reset ---
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial info: {info}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch after reset."

    # --- Test Environment Step ---
    terminated = False
    truncated = False
    total_reward = 0
    for i in range(env_config['max_episode_steps'] + 5): # Test going beyond max_episode_steps
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(
            f"Step {info['current_step']}: Action={action}, Price={info['current_price']:.2f}, "
            f"Reward={reward:.3f}, Capital={info['current_capital']:.2f}, Pos={info['current_position']}, "
            f"Term={terminated}, Trunc={truncated}"
        )
        total_reward += reward
        if 'trade_details' in info:
            logger.info(f"Trade executed: {info['trade_details']}")

        if terminated or truncated:
            logger.info(f"Episode finished at step {i+1}. Reason: {'Terminated' if terminated else 'Truncated'}")
            logger.info(f"Final Capital: {info['current_capital']:.2f}, Total Reward: {total_reward:.3f}")
            break
            
    assert terminated or truncated, "Episode should have ended."

    # --- Test Trade Log ---
    trade_log_df = env.get_trade_log()
    if not trade_log_df.empty:
        logger.info("\n--- Trade Log ---")
        print(trade_log_df)
    else:
        logger.info("\n--- No trades executed or logging disabled ---")
        
    env.close()
    logger.info("\nIntradayTradingEnv standalone test complete.")
