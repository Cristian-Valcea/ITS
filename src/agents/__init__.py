# src/agents/__init__.py

# This file makes the 'agents' directory a Python package.
# You can optionally import specific agent classes here for easier access, e.g.:
# from .data_agent import DataAgent
# from .feature_agent import FeatureAgent
# ... and so on for all agents

# Import legacy shims for backward compatibility
from .trainer_agent import TrainerAgent, create_trainer_agent
# Note: OrchestratorAgent import is lazy to avoid circular imports

def __getattr__(name):
    """Lazy import for OrchestratorAgent to avoid circular imports."""
    if name == "OrchestratorAgent":
        from .orchestrator_agent import OrchestratorAgent
        return OrchestratorAgent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Or, more generally, allow selective imports using __all__
__all__ = [
    "DataAgent",
    "FeatureAgent", 
    "EnvAgent",
    "RiskAgent",
    "EvaluatorAgent",
    # Legacy shims (DEPRECATED - use new locations)
    "TrainerAgent",        # Use src.training.trainer_agent instead
    "create_trainer_agent", # Use src.training.trainer_agent instead
    "OrchestratorAgent",   # Use src.execution.orchestrator_agent instead
    # "BaseAgent" # if you create a BaseAgent
]

# It's also a good place for any package-level initializations
# or configurations related to the agents, if necessary.

print("Agents package initialized.")
