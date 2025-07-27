# src/security/backends/__init__.py
"""
Vault backend implementations.
"""

from .local_vault import LocalVaultBackend

# Optional cloud backends (require additional dependencies)
available_backends = ["LocalVaultBackend"]

try:
    from .aws_secrets_manager import AWSSecretsBackend
    available_backends.append("AWSSecretsBackend")
except ImportError:
    # boto3 not installed
    pass

try:
    from .azure_keyvault import AzureKeyVaultBackend
    available_backends.append("AzureKeyVaultBackend")
except ImportError:
    # azure-keyvault-secrets not installed
    pass

try:
    from .hashicorp_vault import HashiCorpVaultBackend
    available_backends.append("HashiCorpVaultBackend")
except ImportError:
    # hvac not installed
    pass

__all__ = available_backends
