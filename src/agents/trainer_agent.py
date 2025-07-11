# src/agents/trainer_agent.py
"""
LEGACY SHIM: This module provides backward compatibility.

This file re-exports TrainerAgent from the new location for backward compatibility.
New code should import directly from src.training.trainer_agent.

DEPRECATED - Use src.training.trainer_agent instead.
This shim will be removed in v2.0.0.
"""

import warnings
import sys
from pathlib import Path

# Add project root to path for absolute imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.deprecation import deprecation_warning, check_legacy_imports_enabled

# Check if legacy imports are still enabled
check_legacy_imports_enabled()

# Issue deprecation warning for TrainerAgent
deprecation_warning(
    old_path="src.agents.trainer_agent.TrainerAgent",
    new_path="src.training.trainer_agent.TrainerAgent",
    removal_version="v2.0.0",
    additional_info=(
        "The TrainerAgent has been moved to the training module with enhanced "
        "capabilities. Please update your imports:\n"
        "  OLD: from src.agents.trainer_agent import TrainerAgent\n"
        "  NEW: from src.training.trainer_agent import TrainerAgent"
    )
)

# Issue deprecation warning for create_trainer_agent
deprecation_warning(
    old_path="src.agents.trainer_agent.create_trainer_agent",
    new_path="src.training.trainer_agent.create_trainer_agent",
    removal_version="v2.0.0",
    additional_info=(
        "The create_trainer_agent factory function has been moved to the training module. "
        "Please update your imports:\n"
        "  OLD: from src.agents.trainer_agent import create_trainer_agent\n"
        "  NEW: from src.training.trainer_agent import create_trainer_agent"
    )
)

# Legacy shim - re-export from new location using absolute import
from src.training.trainer_agent import TrainerAgent, create_trainer_agent

# Maintain backward compatibility
__all__ = ['TrainerAgent', 'create_trainer_agent']