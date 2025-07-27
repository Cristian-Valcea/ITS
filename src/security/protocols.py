# src/security/protocols.py
"""
Protocol definitions for the enhanced secrets management system.
Defines interfaces and data structures for backend-agnostic operations.
"""

from typing import Protocol, Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class SecretType(Enum):
    """Types of secrets supported by the system."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    OAUTH_TOKEN = "oauth_token"
    ENCRYPTION_KEY = "encryption_key"


class SecretMetadata(BaseModel):
    """Metadata associated with a secret."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotation_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    secret_type: SecretType = SecretType.API_KEY
    description: str = ""
    
    def dict(self, **kwargs):
        """Override dict method to handle datetime and enum serialization."""
        data = super().model_dump(**kwargs)
        # Convert datetime objects to ISO format strings and enums to values
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
            elif value is None:
                data[key] = None
        return data


class SecretData(BaseModel):
    """Complete secret data structure."""
    key: str
    value: str
    metadata: SecretMetadata
    
    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if self.metadata.expires_at is None:
            return False
        return datetime.utcnow() > self.metadata.expires_at
    
    def days_until_expiry(self) -> Optional[int]:
        """Get days until expiry, None if no expiry set."""
        if self.metadata.expires_at is None:
            return None
        delta = self.metadata.expires_at - datetime.utcnow()
        return max(0, delta.days)


class VaultBackend(Protocol):
    """
    Protocol defining the interface for vault backends.
    
    This allows for different storage implementations:
    - Local encrypted files
    - HashiCorp Vault
    - AWS Secrets Manager
    - Azure Key Vault
    - Google Secret Manager
    """
    
    async def store(self, key: str, encrypted_data: bytes, metadata: SecretMetadata) -> bool:
        """
        Store encrypted secret data with metadata.
        
        Args:
            key: Secret identifier
            encrypted_data: Already encrypted secret data
            metadata: Secret metadata
            
        Returns:
            True if successful, False otherwise
        """
        ...
    
    async def retrieve(self, key: str) -> SecretData:
        """
        Retrieve secret data by key.
        
        Args:
            key: Secret identifier
            
        Returns:
            SecretData object
            
        Raises:
            KeyError: If secret not found
        """
        ...
    
    async def delete(self, key: str) -> bool:
        """
        Delete a secret.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if successful, False if not found
        """
        ...
    
    async def list_keys(self) -> List[str]:
        """
        List all secret keys.
        
        Returns:
            List of secret keys
        """
        ...
    
    async def exists(self, key: str) -> bool:
        """
        Check if a secret exists.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if exists, False otherwise
        """
        ...
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
