import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import hashlib

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv

# Import gym for observation flattening wrapper
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

# Optional: Import SB3 env checker if available
try:
    from stable_baselines3.common.env_checker import check_env
    SB3_CHECK_AVAILABLE = True
except ImportError:
    SB3_CHECK_AVAILABLE = False

class EnvAgent(BaseAgent):
    """
    EnvAgent is responsible for:
    1. Defining and instantiating the `IntradayTradingEnv`.
    2. Preparing data (features, prices) in the format expected by the environment.
    3. Providing an interface to interact with the environment (reset, step).
    4. Validating the environment and input data for RL compatibility.
    5. (ENHANCED) Managing and caching multiple environments for multi-asset/multi-market support.
    """
    def __init__(self, config: dict, validate_env: bool = True):
        """
        Initializes the EnvAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                'env_config': A sub-dictionary with parameters for IntradayTradingEnv.
            validate_env (bool): Whether to run SB3 environment validation after creation.
        """
        super().__init__(agent_name="EnvAgent", config=config)
        self.env_params = self.config.get('env_config', {})
        if not isinstance(self.env_params, dict):
            self.logger.error("'env_config' must be a dictionary in EnvAgent's configuration.")
            self.env_params = {}

        if not self.env_params:
            self.logger.warning("'env_config' is empty or not found in EnvAgent's configuration. IntradayTradingEnv will use its defaults.")

        self.env_cache: Dict[str, IntradayTradingEnv] = {}
        self.active_env_id: Optional[str] = None
        self.validate_env = validate_env
        self.logger.info("EnvAgent initialized. Environment(s) will be created when data is provided.")

    def _diagnose_data(self, market_feature_data: np.ndarray, price_data_for_env: pd.Series) -> bool:
        """
        Checks for NaNs, infs, and index alignment in the input data.
        Returns True if data is valid, False otherwise.
        """
        valid = True
        if not isinstance(market_feature_data, np.ndarray):
            self.logger.error("market_feature_data must be a numpy ndarray.")
            return False
        if not isinstance(price_data_for_env, pd.Series):
            self.logger.error("price_data_for_env must be a pandas Series.")
            return False

        if np.isnan(market_feature_data).any():
            self.logger.error("market_feature_data contains NaN values.")
            valid = False
        if np.isinf(market_feature_data).any():
            self.logger.error("market_feature_data contains infinite values.")
            valid = False
        if price_data_for_env.isnull().any():
            self.logger.error("price_data_for_env contains NaN values.")
            valid = False
        if not np.isfinite(price_data_for_env.values).all():
            self.logger.error("price_data_for_env contains infinite values.")
            valid = False
        if len(market_feature_data) != len(price_data_for_env):
            self.logger.error(f"Data length mismatch: features {len(market_feature_data)}, prices {len(price_data_for_env)}")
            valid = False
        return valid

    def _validate_environment(self, env):
        """
        Runs SB3's check_env if available and enabled.
        """
        if self.validate_env and SB3_CHECK_AVAILABLE:
            try:
                check_env(env, warn=True)
                self.logger.info("Environment passed SB3 check_env validation.")
            except Exception as e:
                self.logger.error(f"SB3 check_env validation failed: {e}", exc_info=True)
        elif self.validate_env and not SB3_CHECK_AVAILABLE:
            self.logger.warning("SB3 check_env is not available. Skipping environment validation.")

    def _config_hash(self, config: dict, market_feature_data: np.ndarray, price_data_for_env: pd.Series) -> str:
        """
        Generates a unique hash for the environment configuration and data.
        """
        # Use config, data shapes, and price index hash for uniqueness
        config_str = str(sorted((k, str(v)) for k, v in config.items()))
        data_shape_str = str(market_feature_data.shape) + str(price_data_for_env.shape)
        price_index_hash = hashlib.md5(str(price_data_for_env.index.tolist()).encode()).hexdigest()
        return hashlib.md5((config_str + data_shape_str + price_index_hash).encode()).hexdigest()

    def get_or_create_env(
        self,
        market_feature_data: np.ndarray,
        price_data_for_env: pd.Series,
        env_params: Optional[dict] = None
    ) -> Optional[IntradayTradingEnv]:
        """
        Returns a cached environment if available, otherwise creates and caches a new one.
        """
        params = env_params if env_params is not None else self.env_params
        env_id = self._config_hash(params, market_feature_data, price_data_for_env)
        if env_id in self.env_cache:
            self.logger.info(f"Reusing cached environment with id: {env_id}")
            self.active_env_id = env_id
            return self.env_cache[env_id]

        if not self._diagnose_data(market_feature_data, price_data_for_env):
            self.logger.error("Data validation failed. Environment will not be created.")
            return None

        self.logger.info(f"Creating IntradayTradingEnv with {len(market_feature_data)} steps. (env_id={env_id})")
        self.logger.info(f"Market Feature data shape: {market_feature_data.shape}, Price data shape: {price_data_for_env.shape}")

        try:
            env_constructor_params = {
                **params,
                'processed_feature_data': market_feature_data,
                'price_data': price_data_for_env
            }
            env = IntradayTradingEnv(**env_constructor_params)
            self.logger.info("IntradayTradingEnv created successfully.")
            
            # Apply FlattenObservation wrapper to fix SB3 observation shape warnings
            if GYM_AVAILABLE:
                env = gym.wrappers.FlattenObservation(env)
                self.logger.info("Applied FlattenObservation wrapper to fix SB3 compatibility.")
            else:
                self.logger.warning("Gym not available - observation shape warnings may occur with SB3.")
            
            self._validate_environment(env)
            self.env_cache[env_id] = env
            self.active_env_id = env_id
            return env
        except Exception as e:
            self.logger.error(f"Failed to create IntradayTradingEnv: {e}", exc_info=True)
            return None

    def switch_env(
        self,
        market_feature_data: np.ndarray,
        price_data_for_env: pd.Series,
        env_params: Optional[dict] = None
    ) -> Optional[IntradayTradingEnv]:
        """
        Switches to a different environment (creates or retrieves from cache).
        """
        return self.get_or_create_env(market_feature_data, price_data_for_env, env_params)

    def get_env(self) -> Optional[IntradayTradingEnv]:
        """
        Returns the currently active environment.
        """
        if self.active_env_id is None or self.active_env_id not in self.env_cache:
            self.logger.warning("No active environment. Use get_or_create_env() or switch_env() first.")
            return None
        return self.env_cache[self.active_env_id]

    def remove_env(
        self,
        market_feature_data: np.ndarray,
        price_data_for_env: pd.Series,
        env_params: Optional[dict] = None
    ):
        """
        Removes an environment from the cache and closes it.
        """
        params = env_params if env_params is not None else self.env_params
        env_id = self._config_hash(params, market_feature_data, price_data_for_env)
        if env_id in self.env_cache:
            self.env_cache[env_id].close()
            del self.env_cache[env_id]
            if self.active_env_id == env_id:
                self.active_env_id = None
            self.logger.info(f"Environment with id {env_id} removed from cache.")

    def list_envs(self) -> Dict[str, IntradayTradingEnv]:
        """
        Returns a copy of the environment cache dictionary.
        """
        return self.env_cache.copy()

    def run(
        self,
        processed_feature_data: np.ndarray,
        price_data_for_env: pd.Series
    ) -> Optional[IntradayTradingEnv]:
        """
        Creates or retrieves an environment and sets it as active.
        """
        self.logger.info("EnvAgent run: Creating or retrieving trading environment.")
        return self.get_or_create_env(processed_feature_data, price_data_for_env)

if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock FeatureAgent output ---
    num_env_steps = 150
    lookback = 7
    num_market_features = 6

    if lookback > 1:
        mock_market_feature_data = np.random.rand(num_env_steps, lookback, num_market_features).astype(np.float32)
    else:
        mock_market_feature_data = np.random.rand(num_env_steps, num_market_features).astype(np.float32)

    mock_prices = 100 + np.cumsum(np.random.randn(num_env_steps) * 0.2)
    mock_price_dates = pd.to_datetime(pd.date_range(start='2023-02-01 10:00', periods=num_env_steps, freq='1min'))
    mock_price_series = pd.Series(mock_prices, index=mock_price_dates, name='close')

    print(f"Mock market feature data shape: {mock_market_feature_data.shape}")
    print(f"Mock price series shape: {mock_price_series.shape}")

    env_agent_config = {
        'env_config': {
            'initial_capital': 75000.0,
            'max_daily_drawdown_pct': 0.03,
            'hourly_turnover_cap': 3.0,
            'turnover_penalty_factor': 0.02,
            'transaction_cost_pct': 0.0007,
            'reward_scaling': 1.5,
            'max_episode_steps': 100,
            'log_trades': True,
            'lookback_window': lookback
        }
    }

    env_agent = EnvAgent(config=env_agent_config)
    trading_env_instance = env_agent.run(
        processed_feature_data=mock_market_feature_data,
        price_data_for_env=mock_price_series
    )

    if trading_env_instance:
        print(f"\nIntradayTradingEnv instance created successfully by EnvAgent: {trading_env_instance}")

        print("\nTesting the created environment with a few random steps...")
        obs, info = trading_env_instance.reset()
        print(f"Initial observation shape from env: {obs.shape}")
        assert obs.shape == trading_env_instance.observation_space.shape

        total_reward_test = 0
        for i in range(5):
            action = trading_env_instance.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env_instance.step(action)
            total_reward_test += reward / env_agent_config['env_config']['reward_scaling']
            print(f"  Step {i+1}: Action={action}, Reward={reward:.3f}, Term={terminated}, Trunc={truncated}, PortfolioVal={info['portfolio_value']:.2f}")
            if terminated or truncated:
                break

        print(f"Test completed. Total reward from random actions: {total_reward_test:.3f}")

        final_trade_log = trading_env_instance.get_trade_log()
        if not final_trade_log.empty:
            print("\nTrade log from the environment instance:")
            print(final_trade_log.head())
        else:
            print("\nNo trades were logged by the environment instance.")

        portfolio_hist = trading_env_instance.get_portfolio_history()
        if not portfolio_hist.empty:
            print("\nPortfolio history (first 5 from env):")
            print(portfolio_hist.head())

        trading_env_instance.close()
    else:
        print("\nFailed to create IntradayTradingEnv instance via EnvAgent.")

    print("\nEnvAgent example run complete.")