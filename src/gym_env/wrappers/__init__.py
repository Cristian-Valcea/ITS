"""
Trading Environment Wrappers

Modular Gym wrappers for trading rules and functionality.
Lightweight adapters that transform observations, rewards, and actions
without touching the database or maintaining complex state.

These wrappers interface with the stateful risk engines in src/risk/controls/
to provide a clean separation between domain logic and RL environment concerns.
"""

from .risk_wrapper import RiskObsWrapper, VolatilityPenaltyReward
from .cooldown_wrapper import CooldownWrapper
from .size_limit_wrapper import SizeLimitWrapper  
from .action_penalty_wrapper import ActionPenaltyWrapper
from .streaming_trade_log_wrapper import StreamingTradeLogWrapper
from .wrapper_factory import TradingWrapperFactory, create_trading_env, migrate_from_monolithic

__all__ = [
    'RiskObsWrapper',
    'VolatilityPenaltyReward',
    'CooldownWrapper',
    'SizeLimitWrapper', 
    'ActionPenaltyWrapper',
    'StreamingTradeLogWrapper',
    'TradingWrapperFactory',
    'create_trading_env',
    'migrate_from_monolithic'
]