# src/shared/version.py
"""
Version management for IntradayJules refactoring phases.

This module tracks the current version and deployment phase of the system.
"""

from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path

# Version Information
VERSION = "1.9.0-legacy"  # Phase 7.1: Production with legacy support
VERSION_MAJOR = 1
VERSION_MINOR = 9
VERSION_PATCH = 0
VERSION_SUFFIX = "legacy"  # Indicates legacy compatibility mode

# Phase Information
CURRENT_PHASE = "7.1"
PHASE_NAME = "DEPLOYMENT_WITH_SHIMS"
PHASE_DESCRIPTION = "Production deployment with backward compatibility shims"

# Target versions for different phases
PHASE_VERSIONS = {
    "7.1": {
        "version": "1.9.0-legacy",
        "name": "DEPLOYMENT_WITH_SHIMS",
        "description": "Production deployment with backward compatibility",
        "legacy_support": True,
        "deprecation_warnings": True,
        "cleanup_ready": False
    },
    "7.2": {
        "version": "2.0.0-clean",
        "name": "CLEANUP_COMPLETE",
        "description": "Clean architecture without legacy compatibility",
        "legacy_support": False,
        "deprecation_warnings": False,
        "cleanup_ready": True
    }
}

# Build Information
BUILD_DATE = datetime.now().isoformat()
BUILD_COMMIT = "phase7-deployment"  # Would be populated by CI/CD

# Architecture Information
ARCHITECTURE_VERSION = "2.0"
REFACTORING_COMPLETE = True
BACKWARD_COMPATIBLE = True  # Phase 7.1 maintains compatibility

def get_version_info() -> Dict[str, Any]:
    """Get comprehensive version information."""
    return {
        "version": VERSION,
        "version_parts": {
            "major": VERSION_MAJOR,
            "minor": VERSION_MINOR,
            "patch": VERSION_PATCH,
            "suffix": VERSION_SUFFIX
        },
        "phase": {
            "current": CURRENT_PHASE,
            "name": PHASE_NAME,
            "description": PHASE_DESCRIPTION
        },
        "build": {
            "date": BUILD_DATE,
            "commit": BUILD_COMMIT
        },
        "architecture": {
            "version": ARCHITECTURE_VERSION,
            "refactoring_complete": REFACTORING_COMPLETE,
            "backward_compatible": BACKWARD_COMPATIBLE
        },
        "features": {
            "legacy_support": PHASE_VERSIONS[CURRENT_PHASE]["legacy_support"],
            "deprecation_warnings": PHASE_VERSIONS[CURRENT_PHASE]["deprecation_warnings"],
            "cleanup_ready": PHASE_VERSIONS[CURRENT_PHASE]["cleanup_ready"]
        }
    }

def get_phase_info(phase: str = None) -> Dict[str, Any]:
    """Get information about a specific phase."""
    if phase is None:
        phase = CURRENT_PHASE
    
    return PHASE_VERSIONS.get(phase, {})

def is_legacy_supported() -> bool:
    """Check if legacy imports are currently supported."""
    return PHASE_VERSIONS[CURRENT_PHASE]["legacy_support"]

def should_show_deprecation_warnings() -> bool:
    """Check if deprecation warnings should be shown."""
    return PHASE_VERSIONS[CURRENT_PHASE]["deprecation_warnings"]

def is_cleanup_ready() -> bool:
    """Check if the system is ready for cleanup phase."""
    return PHASE_VERSIONS[CURRENT_PHASE]["cleanup_ready"]

def print_version_info():
    """Print comprehensive version information."""
    info = get_version_info()
    
    print("=" * 60)
    print("INTRADAYJULES VERSION INFORMATION")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Phase: {info['phase']['current']} - {info['phase']['name']}")
    print(f"Description: {info['phase']['description']}")
    print(f"Architecture Version: {info['architecture']['version']}")
    print(f"Build Date: {info['build']['date']}")
    print()
    print("FEATURES:")
    print(f"  Legacy Support: {'✅' if info['features']['legacy_support'] else '❌'}")
    print(f"  Deprecation Warnings: {'✅' if info['features']['deprecation_warnings'] else '❌'}")
    print(f"  Cleanup Ready: {'✅' if info['features']['cleanup_ready'] else '❌'}")
    print(f"  Backward Compatible: {'✅' if info['architecture']['backward_compatible'] else '❌'}")
    print(f"  Refactoring Complete: {'✅' if info['architecture']['refactoring_complete'] else '❌'}")
    print("=" * 60)

def save_version_info(filepath: str = None):
    """Save version information to a JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "version_info.json"
    
    info = get_version_info()
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Version information saved to: {filepath}")

def load_version_info(filepath: str = None) -> Dict[str, Any]:
    """Load version information from a JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "version_info.json"
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Version file not found at {filepath}, using current version info")
        return get_version_info()

# Compatibility check functions
def check_import_compatibility(import_path: str) -> Dict[str, Any]:
    """Check if an import path is compatible with the current version."""
    legacy_paths = [
        "src.agents.orchestrator_agent",
        "src.agents.trainer_agent"
    ]
    
    new_paths = [
        "src.execution.orchestrator_agent",
        "src.training.trainer_agent"
    ]
    
    result = {
        "path": import_path,
        "is_legacy": import_path in legacy_paths,
        "is_new": import_path in new_paths,
        "supported": True,
        "deprecated": False,
        "alternative": None
    }
    
    if import_path in legacy_paths:
        result["deprecated"] = True
        result["supported"] = is_legacy_supported()
        
        # Find alternative
        if "orchestrator_agent" in import_path:
            result["alternative"] = "src.execution.orchestrator_agent"
        elif "trainer_agent" in import_path:
            result["alternative"] = "src.training.trainer_agent"
    
    return result

# Phase transition utilities
def prepare_for_phase_transition(target_phase: str):
    """Prepare system for transition to target phase."""
    current_info = get_phase_info(CURRENT_PHASE)
    target_info = get_phase_info(target_phase)
    
    if not target_info:
        raise ValueError(f"Unknown target phase: {target_phase}")
    
    print(f"Preparing transition from Phase {CURRENT_PHASE} to Phase {target_phase}")
    print(f"Current: {current_info['name']}")
    print(f"Target: {target_info['name']}")
    
    # Phase 7.1 -> 7.2 transition
    if CURRENT_PHASE == "7.1" and target_phase == "7.2":
        print("\nPhase 7.2 Transition Checklist:")
        print("- [ ] Validate all users have migrated from legacy imports")
        print("- [ ] Run comprehensive test suite")
        print("- [ ] Backup current system state")
        print("- [ ] Update version configuration")
        print("- [ ] Remove legacy shim files")
        print("- [ ] Update documentation")
        
        return {
            "ready": True,
            "actions_required": [
                "validate_migration_complete",
                "run_tests",
                "backup_system",
                "update_version",
                "remove_shims",
                "update_docs"
            ]
        }
    
    return {"ready": False, "reason": f"Transition from {CURRENT_PHASE} to {target_phase} not defined"}

if __name__ == "__main__":
    print_version_info()
    save_version_info()