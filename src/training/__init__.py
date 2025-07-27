# src/training/__init__.py
"""
Training module for RL model development and risk-aware training.

This module contains components for training RL models with:
- Risk-aware callbacks and reward shaping
- Policy bundle generation (TorchScript + metadata)
- SB3 integration with production-grade error handling
- GPU optimization for training workloads
"""

from .trainer_agent import create_trainer_agent, TrainerAgent

__all__ = [
    'create_trainer_agent',
    'TrainerAgent',
]
