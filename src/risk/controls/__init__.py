"""
Risk Controls Module

This module contains plug-and-play risk control components for the trading environment:
- RiskManager: Central coordinator for all risk controls
- VolatilityPenalty: Online variance tracking with Welford algorithm
- CurriculumScheduler: Episode-based progression system

All components are designed to be stateful domain logic that can be wrapped
by lightweight Gym adapters.
"""

from .risk_manager import RiskManager
from .volatility_penalty import VolatilityPenalty
from .curriculum_scheduler import CurriculumScheduler

__all__ = [
    'RiskManager',
    'VolatilityPenalty',
    'CurriculumScheduler',
]