# src/agents/__init__.py

# This file makes the 'agents' directory a Python package.
# You can optionally import specific agent classes here for easier access, e.g.:
# from .data_agent import DataAgent
# from .feature_agent import FeatureAgent
# ... and so on for all agents

# Or, more generally, allow selective imports using __all__
__all__ = [
    "DataAgent",
    "FeatureAgent",
    "EnvAgent",
    "TrainerAgent",
    "RiskAgent",
    "EvaluatorAgent",
    "OrchestratorAgent",
    # "BaseAgent" # if you create a BaseAgent
]

# It's also a good place for any package-level initializations
# or configurations related to the agents, if necessary.

print("Agents package initialized.")
