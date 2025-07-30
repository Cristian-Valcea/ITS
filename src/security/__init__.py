# src/security/__init__.py
"""
Enhanced Security Module for IntradayJules Trading System

This module provides enterprise-grade secrets management with:
- Backend-agnostic architecture (local, cloud, HashiCorp Vault)
- Argon2id key derivation for memory-hard security
- Atomic operations with concurrency safety
- TPM hardware binding support
- Audit-grade logging and metrics
- CLI tools for operations teams
"""

from .protocols import VaultBackend, SecretData, SecretMetadata
from .secrets_manager import SecretsManager, AdvancedSecretsManager
from .encryption import HardenedEncryption
from .backends.local_vault import LocalVaultBackend

# Import ITS Integration Helper
from .its_integration import (
    ITSSecretsHelper,
    get_its_secret,
    get_database_config,
    get_alert_config,
    get_broker_config
)

__version__ = "2.0.0"
__all__ = [
    "VaultBackend",
    "SecretData", 
    "SecretMetadata",
    "SecretsManager",
    "AdvancedSecretsManager",
    "HardenedEncryption",
    "LocalVaultBackend",
    "ITSSecretsHelper",
    "get_its_secret",
    "get_database_config", 
    "get_alert_config",
    "get_broker_config"
]
