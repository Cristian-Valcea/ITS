# src/gym_env/components/__init__.py
"""
Environment Components Package

This package contains modular components for the trading environment,
following separation of concerns principles.
"""

from .turnover_penalty import TurnoverPenaltyCalculator

__all__ = [
    'TurnoverPenaltyCalculator',
]