# src/agents/orchestrator_agent.py
"""
LEGACY SHIM: This module provides backward compatibility.

This file re-exports OrchestratorAgent from the new location for backward compatibility.
New code should import directly from src.execution.orchestrator_agent.

DEPRECATED - Use src.execution.orchestrator_agent instead.
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

# Issue deprecation warning
deprecation_warning(
    old_path="src.agents.orchestrator_agent.OrchestratorAgent",
    new_path="src.execution.orchestrator_agent.OrchestratorAgent",
    removal_version="v2.0.0",
    additional_info=(
        "The OrchestratorAgent has been moved to the execution module as part of "
        "the architectural refactoring. Please update your imports:\n"
        "  OLD: from src.agents.orchestrator_agent import OrchestratorAgent\n"
        "  NEW: from src.execution.orchestrator_agent import OrchestratorAgent"
    )
)

# Legacy shim - re-export from new location using absolute import
from src.execution.orchestrator_agent import OrchestratorAgent

# Maintain backward compatibility
__all__ = ['OrchestratorAgent']