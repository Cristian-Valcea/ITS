"""
Gym Environment Wrappers

Lightweight adapters that transform observations, rewards, and actions
without touching the database or maintaining complex state.

These wrappers interface with the stateful risk engines in src/risk/controls/
to provide a clean separation between domain logic and RL environment concerns.
"""

from .risk_wrapper import RiskObsWrapper, VolatilityPenaltyReward

__all__ = [
    'RiskObsWrapper',
    'VolatilityPenaltyReward',
]