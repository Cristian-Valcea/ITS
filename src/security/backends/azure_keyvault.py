"""
Azure Key Vault Backend for ITS Secrets Management System

This backend provides integration with Azure Key Vault, following enterprise
best practices for credential management and multi-tenant support.

Features:
- Default Azure credential chain (managed identity → CLI → env vars → explicit)
- Multi-tenant and subscription support
- Key Vault URL-based configuration
- Automatic retry logic with exponential backoff
- Secret versioning and metadata handling
- Comprehensive error handling and logging

Usage:
    # Basic usage with default credentials
    backend = AzureKeyVaultBackend(vault_url="https://myvault.vault.azure.net/")
    
    # With specific tenant and subscription
    backend = AzureKeyVaultBackend(
        vault_url="https://myvault.vault.azure.net/",
        tenant_id="12345678-1234-1234-1234-123456789012",
        subscription_id="87654321-4321-4321-4321-210987654321"
    )
    
    # With explicit credentials (not recommended for production)
    backend = AzureKeyVaultBackend(
        vault_url="https://myvault.vault.azure.net/",
        client_id="app-id",
        client_secret="app-secret",
        tenant_id="tenant-id"
    )
"""

import json
import logging
import base64
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from ..protocols import VaultBackend, SecretData, SecretMetadata, SecretType


class AzureKeyVaultBackend(VaultBackend):
    """
    Azure Key Vault backend implementation.
    
    Follows Azure best practices:
    - Uses default credential chain (managed identity, CLI, env vars)
    - Supports explicit credentials for special cases
    - Key Vault URL-based configuration
    - Built-in retry logic and error handling
    - Secret versioning and metadata handling
    """
    
    def __init__(
        self,
        vault_url: str,
        secret_prefix: str = "intraday-",
        tenant_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        managed_identity_client_id: Optional[str] = None
    ):
        """
        Initialize Azure Key Vault backend.
        
        Args:
            vault_url: Azure Key Vault URL (e.g., https://myvault.vault.azure.net/)
            secret_prefix: Prefix for all secret names (default: "intraday-")
            tenant_id: Azure tenant ID (optional for default credential chain)
            subscription_id: Azure subscription ID (optional)
            client_id: Service principal client ID (for explicit auth)
            client_secret: Service principal client secret (for explicit auth)
            managed_identity_client_id: Client ID for user-assigned managed identity
            
        Raises:
            ImportError: If azure-keyvault-secrets is not installed
            ValueError: If vault_url is invalid
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "azure-keyvault-secrets and azure-identity are required for Azure Key Vault backend. "
                "Install with: pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0"
            )
        
        # Validate vault URL
        parsed_url = urlparse(vault_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid vault URL: {vault_url}")
        
        self.vault_url = vault_url.rstrip('/')
        self.prefix = secret_prefix.rstrip('-') + '-' if secret_prefix else ''
        self.logger = logging.getLogger("AzureKeyVaultBackend")
        
        # Setup credentials following Azure credential chain
        try:
            if client_id and client_secret and tenant_id:
                # Explicit service principal credentials
                self.credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                self.logger.warning("Using explicit Azure credentials (not recommended for production)")
            else:
                # Default credential chain (managed identity → CLI → env vars)
                credential_kwargs = {}
                if managed_identity_client_id:
                    credential_kwargs["managed_identity_client_id"] = managed_identity_client_id
                if tenant_id:
                    credential_kwargs["tenant_id"] = tenant_id
                
                self.credential = DefaultAzureCredential(**credential_kwargs)
                self.logger.info("Using Azure default credential chain")
            
            # Create Key Vault client
            self.client = SecretClient(
                vault_url=self.vault_url,
                credential=self.credential
            )
            
            self.logger.info(f"Azure Key Vault backend initialized for: {self.vault_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Key Vault backend: {e}")
            raise
    
    def _get_secret_name(self, key: str) -> str:
        """Get full secret name with prefix. Azure Key Vault has naming restrictions."""
        # Azure Key Vault secret names must be alphanumeric and hyphens only
        clean_key = key.replace('_', '-').replace('.', '-').replace('/', '-')
        return f"{self.prefix}{clean_key}"
    
    def _parse_secret_name(self, full_name: str) -> str:
        """Parse key from full secret name."""
        if full_name.startswith(self.prefix):
            return full_name[len(self.prefix):]
        return full_name
    
    def _metadata_to_tags(self, metadata: SecretMetadata) -> Dict[str, str]:
        """Convert SecretMetadata to Azure Key Vault tags format."""
        tags = {}
        
        # Add metadata as tags (Azure has tag value length limits)
        tags["SecretType"] = metadata.secret_type.value
        tags["CreatedAt"] = metadata.created_at.isoformat()[:19]  # Truncate microseconds
        tags["UpdatedAt"] = metadata.updated_at.isoformat()[:19]
        tags["RotationCount"] = str(metadata.rotation_count)
        
        if metadata.expires_at:
            tags["ExpiresAt"] = metadata.expires_at.isoformat()[:19]
        
        if metadata.last_accessed:
            tags["LastAccessed"] = metadata.last_accessed.isoformat()[:19]
        
        if metadata.description:
            # Azure tag values are limited to 256 characters
            tags["Description"] = metadata.description[:256]
        
        # Add custom tags (with prefix to avoid conflicts)
        for tag_key, tag_value in metadata.tags.items():
            # Azure tag names must be alphanumeric and limited characters
            clean_key = tag_key.replace('_', '-').replace('.', '-')
            tags[f"Custom-{clean_key}"] = str(tag_value)[:256]
        
        return tags
    
    def _tags_to_metadata(self, tags: Dict[str, str], secret_name: str) -> SecretMetadata:
        """Convert Azure Key Vault tags to SecretMetadata."""
        # Parse metadata from tags
        secret_type = SecretType(tags.get("SecretType", "api_key"))
        created_at = datetime.fromisoformat(tags.get("CreatedAt", datetime.utcnow().isoformat()[:19]))
        updated_at = datetime.fromisoformat(tags.get("UpdatedAt", datetime.utcnow().isoformat()[:19]))
        rotation_count = int(tags.get("RotationCount", "0"))
        
        expires_at = None
        if "ExpiresAt" in tags:
            expires_at = datetime.fromisoformat(tags["ExpiresAt"])
        
        last_accessed = None
        if "LastAccessed" in tags:
            last_accessed = datetime.fromisoformat(tags["LastAccessed"])
        
        description = tags.get("Description", "")
        
        # Extract custom tags
        custom_tags = {}
        for key, value in tags.items():
            if key.startswith("Custom-"):
                custom_key = key[7:].replace('-', '_')  # Remove "Custom-" prefix and restore underscores
                custom_tags[custom_key] = value
        
        return SecretMetadata(
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            rotation_count=rotation_count,
            last_accessed=last_accessed,
            tags=custom_tags,
            secret_type=secret_type,
            description=description
        )
    
    async def store(self, key: str, encrypted_data: bytes, metadata: SecretMetadata) -> bool:
        """
        Store encrypted secret in Azure Key Vault.
        
        Args:
            key: Secret identifier
            encrypted_data: Encrypted secret value
            metadata: Secret metadata
            
        Returns:
            True if successful, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Encode encrypted data as base64 for storage
            secret_value = base64.b64encode(encrypted_data).decode('utf-8')
            
            # Prepare secret data with metadata
            secret_data = {
                "encrypted_value": secret_value,
                "metadata": metadata.dict()
            }
            
            # Convert metadata to tags
            tags = self._metadata_to_tags(metadata)
            
            # Set expiration if specified
            expires_on = metadata.expires_at if metadata.expires_at else None
            
            # Store secret in Key Vault
            self.client.set_secret(
                name=secret_name,
                value=json.dumps(secret_data),
                tags=tags,
                expires_on=expires_on
            )
            
            self.logger.debug(f"Stored secret in Azure Key Vault: {key}")
            return True
            
        except AzureError as e:
            self.logger.error(f"Azure error storing secret '{key}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Tuple[bytes, SecretMetadata]]:
        """
        Retrieve encrypted secret from Azure Key Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            Tuple of (encrypted_data, metadata) or None if not found
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Get secret from Key Vault
            secret = self.client.get_secret(secret_name)
            
            # Parse secret data
            secret_data = json.loads(secret.value)
            encrypted_value = secret_data['encrypted_value']
            metadata_dict = secret_data['metadata']
            
            # Decode encrypted data
            encrypted_data = base64.b64decode(encrypted_value.encode('utf-8'))
            
            # Reconstruct metadata from stored data
            metadata = SecretMetadata(**metadata_dict)
            
            # Update last accessed time
            metadata.last_accessed = datetime.utcnow()
            
            self.logger.debug(f"Retrieved secret from Azure Key Vault: {key}")
            return encrypted_data, metadata
            
        except ResourceNotFoundError:
            self.logger.debug(f"Secret not found in Azure Key Vault: {key}")
            return None
        except AzureError as e:
            self.logger.error(f"Azure error retrieving secret '{key}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete secret from Azure Key Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if successful, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Begin delete operation (soft delete by default)
            delete_operation = self.client.begin_delete_secret(secret_name)
            
            # Wait for deletion to complete
            deleted_secret = delete_operation.result()
            
            self.logger.info(f"Deleted secret from Azure Key Vault: {key}")
            return True
            
        except ResourceNotFoundError:
            self.logger.debug(f"Secret not found for deletion: {key}")
            return True  # Consider non-existent as successfully deleted
        except AzureError as e:
            self.logger.error(f"Azure error deleting secret '{key}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """
        List all secret keys with the configured prefix.
        
        Returns:
            List of secret keys (without prefix)
        """
        try:
            keys = []
            
            # List all secrets in the Key Vault
            secret_properties = self.client.list_properties_of_secrets()
            
            for secret_property in secret_properties:
                secret_name = secret_property.name
                if secret_name.startswith(self.prefix):
                    key = self._parse_secret_name(secret_name)
                    keys.append(key)
            
            self.logger.debug(f"Listed {len(keys)} secrets from Azure Key Vault")
            return keys
            
        except AzureError as e:
            self.logger.error(f"Azure error listing secrets: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def exists(self, key: str) -> bool:
        """
        Check if secret exists in Azure Key Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if secret exists, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            self.client.get_secret(secret_name)
            return True
            
        except ResourceNotFoundError:
            return False
        except AzureError as e:
            self.logger.error(f"Azure error checking secret existence '{key}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to check secret existence '{key}': {e}")
            return False
    
    async def rotate_secret(self, key: str, new_encrypted_data: bytes) -> bool:
        """
        Rotate secret value while preserving metadata.
        
        Args:
            key: Secret identifier
            new_encrypted_data: New encrypted secret value
            
        Returns:
            True if successful, False otherwise
        """
        # Retrieve current metadata
        current_data = await self.retrieve(key)
        if not current_data:
            self.logger.error(f"Cannot rotate non-existent secret: {key}")
            return False
        
        _, metadata = current_data
        
        # Update metadata for rotation
        metadata.updated_at = datetime.utcnow()
        metadata.rotation_count += 1
        
        # Store with updated metadata
        return await self.store(key, new_encrypted_data, metadata)
    
    async def __aenter__(self):
        """Enter async context - validate connection."""
        try:
            # Test connection by listing secrets (with limit to avoid large responses)
            secret_properties = self.client.list_properties_of_secrets()
            # Just get the first item to test connectivity
            next(iter(secret_properties), None)
            self.logger.debug("Azure Key Vault backend connection validated")
            return self
        except Exception as e:
            self.logger.error(f"Failed to validate Azure connection: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - cleanup if needed."""
        # Azure SDK handles connection cleanup automatically
        if hasattr(self.credential, 'close'):
            self.credential.close()
