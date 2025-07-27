# src/gym_env/__init__.py

from .intraday_trading_env import IntradayTradingEnv
from .kyle_lambda_fill_simulator import (
    KyleLambdaFillSimulator, 
    SimpleFillSimulator, 
    FillPriceSimulatorFactory
)

__all__ = [
    "IntradayTradingEnv",
    "KyleLambdaFillSimulator",
    "SimpleFillSimulator", 
    "FillPriceSimulatorFactory"
]

# This allows `from src.gym_env import IntradayTradingEnv`
# Register the environment with Gymnasium if you want to use `gym.make()`
# This is typically done by importing the environment where gym.make() will be called,
# or in a centralized registration file.

# from gymnasium.envs.registration import register

# register(
#     id='IntradayTradingEnv-v0', # You can version your environment
#     entry_point='src.gym_env:IntradayTradingEnv',
#     # Optional:
#     # max_episode_steps=1000, # Max steps per episode
#     # reward_threshold=0.8,   # Optional: a threshold reward to consider the task solved
# )
# print("Custom Gym environment 'IntradayTradingEnv-v0' registered.")
