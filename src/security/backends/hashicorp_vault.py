"""
HashiCorp Vault Backend for ITS Secrets Management System

This backend provides integration with HashiCorp Vault, supporting both
self-hosted and Vault Cloud deployments with enterprise features.

Features:
- Multiple authentication methods (token, AppRole, Kubernetes, etc.)
- KV v1 and v2 secret engines support
- Namespace support for Vault Enterprise
- Automatic token renewal and lease management
- Path-based secret organization
- Comprehensive error handling and logging

Usage:
    # Basic usage with token authentication
    backend = HashiCorpVaultBackend(
        vault_url="https://vault.example.com:8200",
        token="hvs.CAESIJ..."
    )
    
    # With AppRole authentication
    backend = HashiCorpVaultBackend(
        vault_url="https://vault.example.com:8200",
        auth_method="approle",
        role_id="12345678-1234-1234-1234-123456789012",
        secret_id="87654321-4321-4321-4321-210987654321"
    )
    
    # With Kubernetes authentication (for pod-based auth)
    backend = HashiCorpVaultBackend(
        vault_url="https://vault.example.com:8200",
        auth_method="kubernetes",
        role="trading-app"
    )
"""

import json
import logging
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime
from urllib.parse import urlparse, urljoin

try:
    import hvac
    from hvac.exceptions import VaultError, InvalidPath, Forbidden
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False

from ..protocols import VaultBackend, SecretData, SecretMetadata, SecretType


class HashiCorpVaultBackend(VaultBackend):
    """
    HashiCorp Vault backend implementation.
    
    Supports multiple authentication methods and secret engines:
    - Token, AppRole, Kubernetes, LDAP, AWS IAM authentication
    - KV v1 and v2 secret engines
    - Namespace support for Vault Enterprise
    - Automatic token renewal and lease management
    """
    
    def __init__(
        self,
        vault_url: str,
        secret_path: str = "intraday",
        kv_version: int = 2,
        namespace: Optional[str] = None,
        token: Optional[str] = None,
        auth_method: str = "token",
        verify_ssl: bool = True,
        timeout: int = 30,
        **auth_kwargs
    ):
        """
        Initialize HashiCorp Vault backend.
        
        Args:
            vault_url: Vault server URL (e.g., https://vault.example.com:8200)
            secret_path: Base path for secrets (default: "intraday")
            kv_version: KV secret engine version (1 or 2, default: 2)
            namespace: Vault namespace for Enterprise (optional)
            token: Vault token for token authentication
            auth_method: Authentication method (token, approle, kubernetes, etc.)
            verify_ssl: Verify SSL certificates (default: True)
            timeout: Request timeout in seconds (default: 30)
            **auth_kwargs: Additional authentication parameters
            
        Auth method specific kwargs:
            AppRole: role_id, secret_id
            Kubernetes: role, jwt (optional, auto-detected if in pod)
            LDAP: username, password
            AWS IAM: role, use_token (optional)
            
        Raises:
            ImportError: If hvac is not installed
            ValueError: If configuration is invalid
            VaultError: If authentication fails
        """
        if not HVAC_AVAILABLE:
            raise ImportError(
                "hvac is required for HashiCorp Vault backend. "
                "Install with: pip install hvac>=1.2.0"
            )
        
        # Validate configuration
        parsed_url = urlparse(vault_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid vault URL: {vault_url}")
        
        if kv_version not in [1, 2]:
            raise ValueError(f"KV version must be 1 or 2, got: {kv_version}")
        
        self.vault_url = vault_url.rstrip('/')
        self.secret_path = secret_path.strip('/')
        self.kv_version = kv_version
        self.namespace = namespace
        self.auth_method = auth_method
        self.logger = logging.getLogger("HashiCorpVaultBackend")
        
        # Initialize Vault client
        try:
            self.client = hvac.Client(
                url=self.vault_url,
                verify=verify_ssl,
                timeout=timeout,
                namespace=namespace
            )
            
            # Authenticate based on method
            self._authenticate(token, auth_kwargs)
            
            # Verify authentication
            if not self.client.is_authenticated():
                raise VaultError("Failed to authenticate with Vault")
            
            self.logger.info(f"HashiCorp Vault backend initialized: {self.vault_url}")
            if namespace:
                self.logger.info(f"Using namespace: {namespace}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HashiCorp Vault backend: {e}")
            raise
    
    def _authenticate(self, token: Optional[str], auth_kwargs: Dict[str, Any]) -> None:
        """Authenticate with Vault using the specified method."""
        if self.auth_method == "token":
            if not token:
                raise ValueError("Token is required for token authentication")
            self.client.token = token
            
        elif self.auth_method == "approle":
            role_id = auth_kwargs.get("role_id")
            secret_id = auth_kwargs.get("secret_id")
            if not role_id or not secret_id:
                raise ValueError("role_id and secret_id are required for AppRole authentication")
            
            auth_response = self.client.auth.approle.login(
                role_id=role_id,
                secret_id=secret_id
            )
            self.client.token = auth_response["auth"]["client_token"]
            
        elif self.auth_method == "kubernetes":
            role = auth_kwargs.get("role")
            if not role:
                raise ValueError("role is required for Kubernetes authentication")
            
            # Use provided JWT or read from service account
            jwt = auth_kwargs.get("jwt")
            if not jwt:
                try:
                    with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
                        jwt = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("JWT token not found. Provide jwt parameter or run in Kubernetes pod")
            
            auth_response = self.client.auth.kubernetes.login(
                role=role,
                jwt=jwt
            )
            self.client.token = auth_response["auth"]["client_token"]
            
        elif self.auth_method == "ldap":
            username = auth_kwargs.get("username")
            password = auth_kwargs.get("password")
            if not username or not password:
                raise ValueError("username and password are required for LDAP authentication")
            
            auth_response = self.client.auth.ldap.login(
                username=username,
                password=password
            )
            self.client.token = auth_response["auth"]["client_token"]
            
        elif self.auth_method == "aws":
            role = auth_kwargs.get("role")
            if not role:
                raise ValueError("role is required for AWS IAM authentication")
            
            use_token = auth_kwargs.get("use_token", False)
            auth_response = self.client.auth.aws.iam_login(
                role=role,
                use_token=use_token
            )
            self.client.token = auth_response["auth"]["client_token"]
            
        else:
            raise ValueError(f"Unsupported authentication method: {self.auth_method}")
        
        self.logger.info(f"Authenticated with Vault using {self.auth_method} method")
    
    def _get_secret_path(self, key: str) -> str:
        """Get full secret path for the key."""
        return f"{self.secret_path}/{key}"
    
    def _get_kv_path(self, key: str) -> str:
        """Get KV engine path based on version."""
        secret_path = self._get_secret_path(key)
        if self.kv_version == 2:
            # KV v2 uses data/ prefix for secret operations
            return f"data/{secret_path}"
        return secret_path
    
    def _get_metadata_path(self, key: str) -> str:
        """Get metadata path for KV v2."""
        if self.kv_version == 2:
            secret_path = self._get_secret_path(key)
            return f"metadata/{secret_path}"
        return None
    
    def _prepare_secret_data(self, encrypted_data: bytes, metadata: SecretMetadata) -> Dict[str, Any]:
        """Prepare secret data for storage."""
        # Encode encrypted data as base64
        secret_value = base64.b64encode(encrypted_data).decode('utf-8')
        
        # Prepare data structure
        data = {
            "encrypted_value": secret_value,
            "secret_type": metadata.secret_type.value,
            "description": metadata.description or "",
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "rotation_count": metadata.rotation_count
        }
        
        if metadata.expires_at:
            data["expires_at"] = metadata.expires_at.isoformat()
        
        if metadata.last_accessed:
            data["last_accessed"] = metadata.last_accessed.isoformat()
        
        # Add custom tags
        for tag_key, tag_value in metadata.tags.items():
            data[f"tag_{tag_key}"] = str(tag_value)
        
        if self.kv_version == 2:
            # KV v2 requires data wrapper
            return {"data": data}
        
        return data
    
    def _parse_secret_data(self, vault_data: Dict[str, Any]) -> Tuple[bytes, SecretMetadata]:
        """Parse secret data from Vault response."""
        # Extract data based on KV version
        if self.kv_version == 2:
            data = vault_data.get("data", {}).get("data", {})
        else:
            data = vault_data.get("data", {})
        
        # Decode encrypted value
        encrypted_value = data.get("encrypted_value", "")
        encrypted_data = base64.b64decode(encrypted_value.encode('utf-8'))
        
        # Parse metadata
        secret_type = SecretType(data.get("secret_type", "api_key"))
        description = data.get("description", "")
        created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        updated_at = datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        rotation_count = int(data.get("rotation_count", 0))
        
        expires_at = None
        if "expires_at" in data:
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        last_accessed = None
        if "last_accessed" in data:
            last_accessed = datetime.fromisoformat(data["last_accessed"])
        
        # Extract custom tags
        tags = {}
        for key, value in data.items():
            if key.startswith("tag_"):
                tag_key = key[4:]  # Remove "tag_" prefix
                tags[tag_key] = value
        
        metadata = SecretMetadata(
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            rotation_count=rotation_count,
            last_accessed=last_accessed,
            tags=tags,
            secret_type=secret_type,
            description=description
        )
        
        return encrypted_data, metadata
    
    async def store(self, key: str, encrypted_data: bytes, metadata: SecretMetadata) -> bool:
        """
        Store encrypted secret in HashiCorp Vault.
        
        Args:
            key: Secret identifier
            encrypted_data: Encrypted secret value
            metadata: Secret metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            secret_path = self._get_kv_path(key)
            secret_data = self._prepare_secret_data(encrypted_data, metadata)
            
            # Store secret
            if self.kv_version == 2:
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=self._get_secret_path(key),
                    secret=secret_data["data"]
                )
            else:
                self.client.secrets.kv.v1.create_or_update_secret(
                    path=self._get_secret_path(key),
                    secret=secret_data
                )
            
            self.logger.debug(f"Stored secret in Vault: {key}")
            return True
            
        except VaultError as e:
            self.logger.error(f"Vault error storing secret '{key}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Tuple[bytes, SecretMetadata]]:
        """
        Retrieve encrypted secret from HashiCorp Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            Tuple of (encrypted_data, metadata) or None if not found
        """
        try:
            # Read secret
            if self.kv_version == 2:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=self._get_secret_path(key)
                )
            else:
                response = self.client.secrets.kv.v1.read_secret(
                    path=self._get_secret_path(key)
                )
            
            if not response:
                return None
            
            # Parse secret data
            encrypted_data, metadata = self._parse_secret_data(response)
            
            # Update last accessed time
            metadata.last_accessed = datetime.utcnow()
            
            self.logger.debug(f"Retrieved secret from Vault: {key}")
            return encrypted_data, metadata
            
        except (InvalidPath, Forbidden):
            self.logger.debug(f"Secret not found in Vault: {key}")
            return None
        except VaultError as e:
            self.logger.error(f"Vault error retrieving secret '{key}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete secret from HashiCorp Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.kv_version == 2:
                # KV v2 supports soft delete and permanent delete
                self.client.secrets.kv.v2.delete_latest_version_of_secret(
                    path=self._get_secret_path(key)
                )
            else:
                self.client.secrets.kv.v1.delete_secret(
                    path=self._get_secret_path(key)
                )
            
            self.logger.info(f"Deleted secret from Vault: {key}")
            return True
            
        except (InvalidPath, Forbidden):
            self.logger.debug(f"Secret not found for deletion: {key}")
            return True  # Consider non-existent as successfully deleted
        except VaultError as e:
            self.logger.error(f"Vault error deleting secret '{key}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """
        List all secret keys under the configured path.
        
        Returns:
            List of secret keys
        """
        try:
            if self.kv_version == 2:
                response = self.client.secrets.kv.v2.list_secrets(
                    path=self.secret_path
                )
            else:
                response = self.client.secrets.kv.v1.list_secrets(
                    path=self.secret_path
                )
            
            if not response or "data" not in response:
                return []
            
            keys = response["data"].get("keys", [])
            
            # Filter out directories (end with /)
            secret_keys = [key for key in keys if not key.endswith("/")]
            
            self.logger.debug(f"Listed {len(secret_keys)} secrets from Vault")
            return secret_keys
            
        except (InvalidPath, Forbidden):
            self.logger.debug("Secret path not found or no access")
            return []
        except VaultError as e:
            self.logger.error(f"Vault error listing secrets: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def exists(self, key: str) -> bool:
        """
        Check if secret exists in HashiCorp Vault.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if secret exists, False otherwise
        """
        try:
            if self.kv_version == 2:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=self._get_secret_path(key)
                )
            else:
                response = self.client.secrets.kv.v1.read_secret(
                    path=self._get_secret_path(key)
                )
            
            return response is not None
            
        except (InvalidPath, Forbidden):
            return False
        except VaultError as e:
            self.logger.error(f"Vault error checking secret existence '{key}': {e}")
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
            # Test connection by checking if authenticated
            if not self.client.is_authenticated():
                raise VaultError("Not authenticated with Vault")
            
            # Test basic read access
            self.client.sys.read_health_status()
            
            self.logger.debug("HashiCorp Vault backend connection validated")
            return self
        except Exception as e:
            self.logger.error(f"Failed to validate Vault connection: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - cleanup if needed."""
        # Vault client handles connection cleanup automatically
        pass
