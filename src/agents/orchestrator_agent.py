# src/agents/orchestrator_agent.py
"""
LEGACY SHIM: This module provides backward compatibility.

This file re-exports OrchestratorAgent from the new location for backward compatibility.
New code should import directly from src.execution.orchestrator_agent.

DEPRECATED - Use src.execution.orchestrator_agent instead.
"""

# Legacy shim - re-export from new location
from src.execution.orchestrator_agent import OrchestratorAgent  # pragma: no cover

# Maintain backward compatibility
__all__ = ['OrchestratorAgent']