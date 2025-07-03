# src/agents/env_agent.py
import logging
import numpy as np
import pandas as pd

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv # Adjusted import path

class EnvAgent(BaseAgent):
    """
    EnvAgent is responsible for:
    1. Defining and instantiating the `IntradayTradingEnv`.
    2. Preparing data (features, prices) in the format expected by the environment.
    3. Providing an interface to interact with the environment (reset, step).
    """
    def __init__(self, config: dict):
        """
        Initializes the EnvAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'env_config': A sub-dictionary with parameters for IntradayTradingEnv.
                                         This should include all parameters IntradayTradingEnv expects,
                                         such as 'initial_capital', 'max_daily_drawdown_pct',
                                         'hourly_turnover_cap', 'turnover_penalty_factor', etc.
                           'lookback_window': The lookback window size, which is also passed to
                                              IntradayTradingEnv and should match FeatureAgent's config.
        """
        super().__init__(agent_name="EnvAgent", config=config)
        
        # self.env_params now directly holds the full sub-dictionary for IntradayTradingEnv
        # This is expected to be populated by OrchestratorAgent with all necessary keys
        # including the new ones: position_sizing_pct_capital, trade_cooldown_steps, etc.
        self.env_params = self.config.get('env_config', {})
        
        if not self.env_params:
            self.logger.warning("'env_config' is empty or not found in EnvAgent's configuration. IntradayTradingEnv will use its defaults.")

        # Example check for new parameters (optional, for better debugging)
        # These checks help confirm OrchestratorAgent is passing them. IntradayTradingEnv has defaults.
        expected_keys = [
            'lookback_window', 'position_sizing_pct_capital', 'trade_cooldown_steps',
            'terminate_on_turnover_breach', 'turnover_termination_threshold_multiplier'
        ]
        for key in expected_keys:
            if key not in self.env_params:
                self.logger.info(f"'{key}' not explicitly found in env_config for EnvAgent. IntradayTradingEnv will use its default if applicable.")

        self.env = None # Will be initialized with data
        self.logger.info("EnvAgent initialized. Environment will be created when data is provided.")

    def create_env(self, 
                   market_feature_data: np.ndarray, 
                   price_data_for_env: pd.Series) -> IntradayTradingEnv | None:
        """
        Creates an instance of the IntradayTradingEnv.

        Args:
            market_feature_data (np.ndarray): The market feature data (excluding position info)
                for the environment. Expected shape depends on lookback_window.
                This comes from FeatureAgent's `feature_sequences` output (which should be just market data).
            price_data_for_env (pd.Series): Series of actual (unscaled) closing prices, aligned with
                                           `market_feature_data`. Comes from FeatureAgent.

        Returns:
            IntradayTradingEnv or None: The initialized trading environment, or None if error.
        """
        if market_feature_data is None or price_data_for_env is None:
            self.logger.error("Cannot create environment: market feature data or price data is None.")
            return None
        if len(market_feature_data) == 0 or len(price_data_for_env) == 0:
            self.logger.error("Cannot create environment: market feature data or price data is empty.")
            return None
        if len(market_feature_data) != len(price_data_for_env):
             self.logger.error(f"Data length mismatch for env: market features {len(market_feature_data)}, prices {len(price_data_for_env)}")
             return None

        self.logger.info(f"Creating IntradayTradingEnv with {len(market_feature_data)} steps.")
        self.logger.info(f"Market Feature data shape: {market_feature_data.shape}, Price data shape: {price_data_for_env.shape}")
        
        try:
            # Combine static env_params from config with dynamic data
            # Combine static env_params from config with dynamic data
            # self.env_params is expected to contain ALL config keys for IntradayTradingEnv,
            # including the new ones, as set up by OrchestratorAgent.
            env_constructor_params = {
                **self.env_params,  # This will spread all keys from env_config
                'processed_feature_data': market_feature_data, 
                'price_data': price_data_for_env
            }
            
            # Log the parameters being passed for easier debugging
            # self.logger.debug(f"Parameters being passed to IntradayTradingEnv constructor: {env_constructor_params}") # Can be very verbose
            self.logger.info(f"Attempting to create IntradayTradingEnv with {len(env_constructor_params)} total constructor parameters.")


            self.env = IntradayTradingEnv(**env_constructor_params)
            self.logger.info(f"IntradayTradingEnv created successfully.") # Params logged by Env itself or above if debugged
            
            # Optional: Validate the created environment using SB3's check_env
            # from stable_baselines3.common.env_checker import check_env
            # try:
            #    check_env(self.env, warn=True) # warn=True to not raise exception on minor issues
            #    self.logger.info("IntradayTradingEnv passed SB3 check_env.")
            # except Exception as ce:
            #    self.logger.error(f"IntradayTradingEnv failed SB3 check_env: {ce}", exc_info=True)
            #    # Depending on severity, might nullify self.env or raise error
            
            return self.env
        except Exception as e:
            self.logger.error(f"Failed to create IntradayTradingEnv: {e}", exc_info=True)
            self.env = None
            return None

    def get_env(self) -> IntradayTradingEnv | None:
        """Returns the current environment instance."""
        if self.env is None:
            self.logger.warning("Environment accessed before creation. Call create_env() first with data.")
        return self.env

    def run(self, processed_feature_data: np.ndarray, price_data_for_env: pd.Series):
        """
        Primary method for EnvAgent, which is essentially to create the environment.
        The actual "running" (stepping through) of the environment is handled by TrainerAgent or EvaluatorAgent.
        
        Args:
            processed_feature_data (np.ndarray): Feature data from FeatureAgent.
            price_data_for_env (pd.Series): Price data from FeatureAgent.

        Returns:
            IntradayTradingEnv or None: The created environment instance.
        """
        self.logger.info("EnvAgent run: Creating trading environment.")
        return self.create_env(market_feature_data=processed_feature_data, price_data_for_env=price_data_for_env)


if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock FeatureAgent output ---
    # This data needs to be consistent with what FeatureAgent would produce.
    # `feature_sequences` and `price_data_for_env`
    
    num_env_steps = 150
    lookback = 7 # Must match env_agent_config and feature_agent_config for consistency
    num_market_features = 6 # Number of MARKET features (excluding position)

    # 1. Mock `market_feature_data` (np.ndarray)
    # Shape: (num_env_steps, lookback_window, num_market_features) if lookback > 1
    # Shape: (num_env_steps, num_market_features) if lookback <= 1
    if lookback > 1:
        mock_market_feature_data = np.random.rand(num_env_steps, lookback, num_market_features).astype(np.float32)
    else:
        mock_market_feature_data = np.random.rand(num_env_steps, num_market_features).astype(np.float32)

    # 2. Mock `price_data_for_env` (pd.Series)
    # Length must be num_env_steps, with a DatetimeIndex
    mock_prices = 100 + np.cumsum(np.random.randn(num_env_steps) * 0.2)
    mock_price_dates = pd.to_datetime(pd.date_range(start='2023-02-01 10:00', periods=num_env_steps, freq='1min'))
    mock_price_series = pd.Series(mock_prices, index=mock_price_dates, name='close')

    print(f"Mock market feature data shape: {mock_market_feature_data.shape}")
    print(f"Mock price series shape: {mock_price_series.shape}")

    # --- EnvAgent Configuration ---
    # This config would typically come from a YAML file loaded by OrchestratorAgent
    env_agent_config = {
        'env_config': { # Parameters passed directly to IntradayTradingEnv constructor
            'initial_capital': 75000.0,
            'max_daily_drawdown_pct': 0.03, # 3%
            'hourly_turnover_cap': 3.0, # Example
            'turnover_penalty_factor': 0.02, # Example
            'transaction_cost_pct': 0.0007, # 0.07%
            'reward_scaling': 1.5,
            'max_episode_steps': 100, # Env can run for max 100 steps per episode
            'log_trades': True,
            'lookback_window': lookback # This must be consistent with data generation for market_feature_data
        }
        # Note: 'lookback_window' is now part of 'env_config' as IntradayTradingEnv needs it.
        # EnvAgent itself doesn't strictly need a separate 'lookback_window' config key if it's all in 'env_config'.
    }

    # --- Initialize and Run EnvAgent ---
    env_agent = EnvAgent(config=env_agent_config)
    
    # The "run" method of EnvAgent is to create the environment instance
    trading_env_instance = env_agent.run(
        processed_feature_data=mock_market_feature_data, # Pass market features
        price_data_for_env=mock_price_series
    )

    if trading_env_instance:
        print(f"\nIntradayTradingEnv instance created successfully by EnvAgent: {trading_env_instance}")
        
        # --- Test the created environment (optional, basic interaction) ---
        print("\nTesting the created environment with a few random steps...")
        obs, info = trading_env_instance.reset()
        print(f"Initial observation shape from env: {obs.shape}")
        assert obs.shape == trading_env_instance.observation_space.shape

        total_reward_test = 0
        for i in range(5): # Test a few steps
            action = trading_env_instance.action_space.sample() # Random action
            obs, reward, terminated, truncated, info = trading_env_instance.step(action)
            total_reward_test += reward / env_agent_config['env_config']['reward_scaling'] # unscaled
            print(f"  Step {i+1}: Action={action}, Reward={reward:.3f}, Term={terminated}, Trunc={truncated}, PortfolioVal={info['portfolio_value']:.2f}")
            if terminated or truncated:
                break
        
        print(f"Test completed. Total reward from random actions: {total_reward_test:.3f}")
        
        # Example: Accessing trade log from the environment instance
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
