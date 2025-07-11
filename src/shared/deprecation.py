# src/shared/deprecation.py
"""
Deprecation warning system for IntradayJules refactoring.

This module provides utilities for managing the transition from legacy
import paths to the new modular architecture.
"""

import warnings
import functools
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta

# Configure deprecation logger
deprecation_logger = logging.getLogger("IntradayJules.Deprecation")

# Track deprecation warnings to avoid spam
_warning_cache: Dict[str, datetime] = {}
_warning_cooldown = timedelta(hours=1)  # Only warn once per hour per import


class DeprecationConfig:
    """Configuration for deprecation warnings."""
    
    # Phase 7.1: Warnings enabled, legacy imports work
    WARNINGS_ENABLED = True
    LEGACY_IMPORTS_ENABLED = True
    
    # Phase 7.2: After validation period, disable legacy imports
    # WARNINGS_ENABLED = True
    # LEGACY_IMPORTS_ENABLED = False


def deprecation_warning(
    old_path: str,
    new_path: str,
    removal_version: str = "v2.0.0",
    additional_info: Optional[str] = None
):
    """
    Issue a deprecation warning for legacy import paths.
    
    Args:
        old_path: The deprecated import path
        new_path: The new recommended import path
        removal_version: Version when the legacy path will be removed
        additional_info: Additional migration information
    """
    if not DeprecationConfig.WARNINGS_ENABLED:
        return
    
    # Check if we've already warned about this path recently
    now = datetime.now()
    if old_path in _warning_cache:
        if now - _warning_cache[old_path] < _warning_cooldown:
            return
    
    _warning_cache[old_path] = now
    
    # Construct warning message
    message = (
        f"DEPRECATION WARNING: Import path '{old_path}' is deprecated and will be "
        f"removed in {removal_version}. Please update your imports to use "
        f"'{new_path}' instead."
    )
    
    if additional_info:
        message += f"\n{additional_info}"
    
    # Issue warning
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    deprecation_logger.warning(message)


def deprecated_import(
    old_path: str,
    new_path: str,
    removal_version: str = "v2.0.0",
    additional_info: Optional[str] = None
):
    """
    Decorator for deprecated import functions.
    
    Args:
        old_path: The deprecated import path
        new_path: The new recommended import path
        removal_version: Version when the legacy path will be removed
        additional_info: Additional migration information
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            deprecation_warning(old_path, new_path, removal_version, additional_info)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_legacy_import_shim(
    legacy_module: str,
    new_module: str,
    class_name: str,
    removal_version: str = "v2.0.0"
):
    """
    Create a shim for legacy imports that provides backward compatibility.
    
    Args:
        legacy_module: The old module path (e.g., "src.agents.orchestrator_agent")
        new_module: The new module path (e.g., "src.execution.orchestrator_agent")
        class_name: The class being imported
        removal_version: Version when legacy support will be removed
    
    Returns:
        The class from the new module with deprecation warning
    """
    deprecation_warning(
        old_path=f"{legacy_module}.{class_name}",
        new_path=f"{new_module}.{class_name}",
        removal_version=removal_version,
        additional_info=f"Migration Guide: https://github.com/IntradayJules/docs/migration-guide.md"
    )
    
    # Import from new location
    try:
        import importlib
        module = importlib.import_module(new_module)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import {class_name} from new location {new_module}. "
            f"This indicates a problem with the refactoring. Error: {e}"
        )


# Migration mapping for all moved classes
MIGRATION_MAP = {
    # Orchestrator Agent
    "src.agents.orchestrator_agent.OrchestratorAgent": {
        "new_path": "src.execution.orchestrator_agent.OrchestratorAgent",
        "info": "OrchestratorAgent moved to execution module for better organization"
    },
    
    # Trainer Agent
    "src.agents.trainer_agent.TrainerAgent": {
        "new_path": "src.training.trainer_agent.TrainerAgent",
        "info": "TrainerAgent moved to training module with enhanced capabilities"
    },
    "src.agents.trainer_agent.create_trainer_agent": {
        "new_path": "src.training.trainer_agent.create_trainer_agent",
        "info": "Factory function moved to training module"
    },
    
    # Core execution components (new in refactoring)
    "src.agents.execution_loop": {
        "new_path": "src.execution.core.execution_loop",
        "info": "Execution logic extracted to dedicated core module"
    },
    "src.agents.order_router": {
        "new_path": "src.execution.core.order_router",
        "info": "Order routing logic extracted to dedicated core module"
    },
    "src.agents.pnl_tracker": {
        "new_path": "src.execution.core.pnl_tracker",
        "info": "P&L tracking logic extracted to dedicated core module"
    },
    "src.agents.live_data_loader": {
        "new_path": "src.execution.core.live_data_loader",
        "info": "Live data handling extracted to dedicated core module"
    },
    
    # Core training components (new in refactoring)
    "src.agents.trainer_core": {
        "new_path": "src.training.core.trainer_core",
        "info": "Core training logic extracted to dedicated module"
    },
    "src.agents.env_builder": {
        "new_path": "src.training.core.env_builder",
        "info": "Environment building logic extracted to dedicated module"
    },
    "src.agents.policy_export": {
        "new_path": "src.training.core.policy_export",
        "info": "Policy export utilities extracted to dedicated module"
    },
    "src.agents.hyperparam_search": {
        "new_path": "src.training.core.hyperparam_search",
        "info": "Hyperparameter search extracted to dedicated module"
    }
}


def get_migration_info(old_path: str) -> Optional[Dict[str, str]]:
    """Get migration information for a deprecated import path."""
    return MIGRATION_MAP.get(old_path)


def print_migration_summary():
    """Print a summary of all available migrations."""
    print("\n" + "="*80)
    print("INTRADAYJULES MIGRATION GUIDE")
    print("="*80)
    print("\nThe following import paths have been deprecated:")
    print("\nFormat: OLD_PATH -> NEW_PATH")
    print("-" * 80)
    
    for old_path, info in MIGRATION_MAP.items():
        print(f"\n• {old_path}")
        print(f"  -> {info['new_path']}")
        print(f"  Info: {info['info']}")
    
    print("\n" + "="*80)
    print("For detailed migration instructions, see:")
    print("https://github.com/IntradayJules/docs/migration-guide.md")
    print("="*80 + "\n")


# Utility function for checking if legacy imports are still enabled
def check_legacy_imports_enabled():
    """Check if legacy imports are still enabled."""
    if not DeprecationConfig.LEGACY_IMPORTS_ENABLED:
        raise ImportError(
            "Legacy import paths have been disabled. Please update your imports "
            "to use the new module structure. Run 'python -c \"from src.shared.deprecation "
            "import print_migration_summary; print_migration_summary()\"' for guidance."
        )