# src/execution/__init__.py
"""
Execution context for production trading.

This module contains components optimized for low-latency production execution:
- ExecutionAgentStub: Policy bundle loading and <100µs prediction SLO
- OrchestratorAgent: Production entry point for live trading coordination
- Contract testing and SLO validation
- Minimal dependencies (PyTorch CPU only)
"""

from .execution_agent_stub import ExecutionAgentStub, create_execution_agent_stub
from .orchestrator_agent import OrchestratorAgent

__all__ = [
    'ExecutionAgentStub',
    'create_execution_agent_stub', 
    'OrchestratorAgent',
]
