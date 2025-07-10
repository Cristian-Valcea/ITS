# src/agents/trainer_agent.py
"""
LEGACY SHIM: This module provides backward compatibility.

This file re-exports TrainerAgent from the new location for backward compatibility.
New code should import directly from src.training.trainer_agent.

DEPRECATED - Use src.training.trainer_agent instead.
"""

# Legacy shim - re-export from new location
from src.training.trainer_agent import TrainerAgent, create_trainer_agent  # pragma: no cover

# Maintain backward compatibility
__all__ = ['TrainerAgent', 'create_trainer_agent']