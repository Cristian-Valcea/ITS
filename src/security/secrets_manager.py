# src/security/secrets_manager.py
"""
Secrets Management System for IntradayJules Trading System.

Replaces environment variables with secure secret management using:
- AWS Secrets Manager for cloud deployments
- HashiCorp Vault for on-premise deployments
- Local encrypted storage for development

Handles:
- S3 credentials
- Broker API keys
- Database passwords
- API tokens
- Encryption keys
"""

import os
import json
import logging
import base64
import hashlib
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretProvider(Enum):
    """Secret provider types."""
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    LOCAL_ENCRYPTED = "local_encrypted"
    ENVIRONMENT = "environment"  # Fallback


class SecretType(Enum):
    """Types of secrets."""
    DATABASE_PASSWORD = "database_password"
    API_KEY = "api_key"
    S3_CREDENTIALS = "s3_credentials"
    BROKER_CREDENTIALS = "broker_credentials"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    WEBHOOK_TOKEN = "webhook_token"


@dataclass
class SecretMetadata:
    """Secret metadata."""
    secret_id: str
    secret_type: SecretType
    provider: SecretProvider
    created_at: float
    last_accessed: float
    expires_at: Optional[float]
    rotation_enabled: bool
    tags: Dict[str, str]


@dataclass
class Secret:
    """Secret data structure."""
    secret_id: str
    value: Union[str, Dict[str, Any]]
    metadata: SecretMetadata
    
    def get_string_value(self) -> str:
        """Get secret as string."""
        if isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, dict):
            return json.dumps(self.value)
        else:
            return str(self.value)
    
    def get_dict_value(self) -> Dict[str, Any]:
        """Get secret as dictionary."""
        if isinstance(self.value, dict):
            return self.value
        elif isinstance(self.value, str):
            try:
                return json.loads(self.value)
            except json.JSONDecodeError:
                return {"value": self.value}
        else:
            return {"value": str(self.value)}


class AWSSecretsManagerProvider:
    """AWS Secrets Manager provider."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.logger = logging.getLogger("AWSSecretsManagerProvider")
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of AWS client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region_name)
            except ImportError:
                raise RuntimeError("boto3 not installed. Install with: pip install boto3")
        return self._client
    
    async def get_secret(self, secret_id: str) -> Optional[Secret]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            
            # Parse secret value
            if 'SecretString' in response:
                secret_value = response['SecretString']
                try:
                    # Try to parse as JSON
                    secret_value = json.loads(secret_value)
                except json.JSONDecodeError:
                    # Keep as string if not JSON
                    pass
            else:
                # Binary secret
                secret_value = base64.b64decode(response['SecretBinary']).decode('utf-8')
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=SecretType.API_KEY,  # Default, should be tagged
                provider=SecretProvider.AWS_SECRETS_MANAGER,
                created_at=response.get('CreatedDate', time.time()).timestamp(),
                last_accessed=time.time(),
                expires_at=None,
                rotation_enabled=response.get('RotationEnabled', False),
                tags=response.get('Tags', {})
            )
            
            return Secret(
                secret_id=secret_id,
                value=secret_value,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get secret {secret_id} from AWS: {e}")
            return None
    
    async def put_secret(self, secret_id: str, secret_value: Union[str, Dict[str, Any]], 
                        secret_type: SecretType, description: str = "") -> bool:
        """Store secret in AWS Secrets Manager."""
        try:
            # Prepare secret value
            if isinstance(secret_value, dict):
                secret_string = json.dumps(secret_value)
            else:
                secret_string = str(secret_value)
            
            # Check if secret exists
            try:
                self.client.describe_secret(SecretId=secret_id)
                # Update existing secret
                self.client.update_secret(
                    SecretId=secret_id,
                    SecretString=secret_string,
                    Description=description
                )
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                self.client.create_secret(
                    Name=secret_id,
                    SecretString=secret_string,
                    Description=description,
                    Tags=[
                        {'Key': 'SecretType', 'Value': secret_type.value},
                        {'Key': 'Service', 'Value': 'IntradayJules'},
                        {'Key': 'Environment', 'Value': os.getenv('ENVIRONMENT', 'development')}
                    ]
                )
            
            self.logger.info(f"Secret {secret_id} stored in AWS Secrets Manager")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {secret_id} in AWS: {e}")
            return False


class HashiCorpVaultProvider:
    """HashiCorp Vault provider."""
    
    def __init__(self, vault_url: str, vault_token: Optional[str] = None, 
                 mount_point: str = "secret"):
        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token or os.getenv('VAULT_TOKEN')
        self.mount_point = mount_point
        self.logger = logging.getLogger("HashiCorpVaultProvider")
        
        if not self.vault_token:
            raise ValueError("Vault token required. Set VAULT_TOKEN environment variable.")
    
    async def get_secret(self, secret_id: str) -> Optional[Secret]:
        """Get secret from HashiCorp Vault."""
        try:
            import aiohttp
            
            url = f"{self.vault_url}/v1/{self.mount_point}/data/{secret_id}"
            headers = {
                'X-Vault-Token': self.vault_token,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Vault KV v2 format
                        secret_data = data.get('data', {}).get('data', {})
                        secret_metadata = data.get('data', {}).get('metadata', {})
                        
                        # Create metadata
                        metadata = SecretMetadata(
                            secret_id=secret_id,
                            secret_type=SecretType.API_KEY,  # Default
                            provider=SecretProvider.HASHICORP_VAULT,
                            created_at=time.time(),
                            last_accessed=time.time(),
                            expires_at=None,
                            rotation_enabled=False,
                            tags={}
                        )
                        
                        return Secret(
                            secret_id=secret_id,
                            value=secret_data,
                            metadata=metadata
                        )
                    elif response.status == 404:
                        self.logger.warning(f"Secret {secret_id} not found in Vault")
                        return None
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Vault error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Failed to get secret {secret_id} from Vault: {e}")
            return None
    
    async def put_secret(self, secret_id: str, secret_value: Union[str, Dict[str, Any]], 
                        secret_type: SecretType, description: str = "") -> bool:
        """Store secret in HashiCorp Vault."""
        try:
            import aiohttp
            
            url = f"{self.vault_url}/v1/{self.mount_point}/data/{secret_id}"
            headers = {
                'X-Vault-Token': self.vault_token,
                'Content-Type': 'application/json'
            }
            
            # Prepare secret data
            if isinstance(secret_value, str):
                secret_data = {"value": secret_value}
            else:
                secret_data = secret_value
            
            # Add metadata
            secret_data["_metadata"] = {
                "secret_type": secret_type.value,
                "description": description,
                "created_at": time.time()
            }
            
            payload = {
                "data": secret_data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in [200, 204]:
                        self.logger.info(f"Secret {secret_id} stored in Vault")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Vault error {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to store secret {secret_id} in Vault: {e}")
            return False


class LocalEncryptedProvider:
    """Local encrypted storage provider for development."""
    
    def __init__(self, storage_path: Path, master_password: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.secrets_file = self.storage_path / "secrets.enc"
        self.logger = logging.getLogger("LocalEncryptedProvider")
        
        # Initialize encryption
        self.master_password = master_password or os.getenv('SECRETS_MASTER_PASSWORD', 'default-dev-password')
        self.fernet = self._create_fernet_key(self.master_password)
        
        # Load existing secrets
        self.secrets_cache: Dict[str, Secret] = {}
        self._load_secrets()
    
    def _create_fernet_key(self, password: str) -> Fernet:
        """Create Fernet encryption key from password."""
        password_bytes = password.encode()
        salt = b'intradayjules_salt'  # Fixed salt for development
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def _load_secrets(self) -> None:
        """Load secrets from encrypted file."""
        if not self.secrets_file.exists():
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            secrets_data = json.loads(decrypted_data.decode())
            
            for secret_id, secret_info in secrets_data.items():
                metadata = SecretMetadata(
                    secret_id=secret_id,
                    secret_type=SecretType(secret_info['metadata']['secret_type']),
                    provider=SecretProvider.LOCAL_ENCRYPTED,
                    created_at=secret_info['metadata']['created_at'],
                    last_accessed=secret_info['metadata']['last_accessed'],
                    expires_at=secret_info['metadata'].get('expires_at'),
                    rotation_enabled=secret_info['metadata'].get('rotation_enabled', False),
                    tags=secret_info['metadata'].get('tags', {})
                )
                
                self.secrets_cache[secret_id] = Secret(
                    secret_id=secret_id,
                    value=secret_info['value'],
                    metadata=metadata
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
    
    def _save_secrets(self) -> None:
        """Save secrets to encrypted file."""
        try:
            secrets_data = {}
            for secret_id, secret in self.secrets_cache.items():
                secrets_data[secret_id] = {
                    'value': secret.value,
                    'metadata': {
                        'secret_type': secret.metadata.secret_type.value,
                        'created_at': secret.metadata.created_at,
                        'last_accessed': secret.metadata.last_accessed,
                        'expires_at': secret.metadata.expires_at,
                        'rotation_enabled': secret.metadata.rotation_enabled,
                        'tags': secret.metadata.tags
                    }
                }
            
            json_data = json.dumps(secrets_data, indent=2)
            encrypted_data = self.fernet.encrypt(json_data.encode())
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
    
    async def get_secret(self, secret_id: str) -> Optional[Secret]:
        """Get secret from local encrypted storage."""
        secret = self.secrets_cache.get(secret_id)
        if secret:
            # Update last accessed time
            secret.metadata.last_accessed = time.time()
            self._save_secrets()
        return secret
    
    async def put_secret(self, secret_id: str, secret_value: Union[str, Dict[str, Any]], 
                        secret_type: SecretType, description: str = "") -> bool:
        """Store secret in local encrypted storage."""
        try:
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                provider=SecretProvider.LOCAL_ENCRYPTED,
                created_at=time.time(),
                last_accessed=time.time(),
                expires_at=None,
                rotation_enabled=False,
                tags={"description": description}
            )
            
            self.secrets_cache[secret_id] = Secret(
                secret_id=secret_id,
                value=secret_value,
                metadata=metadata
            )
            
            self._save_secrets()
            self.logger.info(f"Secret {secret_id} stored locally")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {secret_id} locally: {e}")
            return False


class SecretsManager:
    """
    Main Secrets Manager for IntradayJules Trading System.
    
    Provides unified interface to multiple secret providers with fallback chain.
    """
    
    def __init__(self, providers: List[Union[AWSSecretsManagerProvider, HashiCorpVaultProvider, LocalEncryptedProvider]]):
        self.providers = providers
        self.logger = logging.getLogger("SecretsManager")
        
        # Secret cache with TTL
        self.cache: Dict[str, Secret] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
        
        self.logger.info(f"SecretsManager initialized with {len(providers)} providers")
    
    async def get_secret(self, secret_id: str, use_cache: bool = True) -> Optional[Secret]:
        """
        Get secret from providers with fallback chain.
        
        Args:
            secret_id: Secret identifier
            use_cache: Whether to use cached value
            
        Returns:
            Secret if found, None otherwise
        """
        # Check cache first
        if use_cache and secret_id in self.cache:
            cache_age = time.time() - self.cache_timestamps.get(secret_id, 0)
            if cache_age < self.cache_ttl:
                self.logger.debug(f"Returning cached secret: {secret_id}")
                return self.cache[secret_id]
        
        # Try each provider in order
        for provider in self.providers:
            try:
                secret = await provider.get_secret(secret_id)
                if secret:
                    # Cache the secret
                    self.cache[secret_id] = secret
                    self.cache_timestamps[secret_id] = time.time()
                    
                    self.logger.info(f"Secret {secret_id} retrieved from {provider.__class__.__name__}")
                    return secret
                    
            except Exception as e:
                self.logger.warning(f"Provider {provider.__class__.__name__} failed for {secret_id}: {e}")
                continue
        
        self.logger.error(f"Secret {secret_id} not found in any provider")
        return None
    
    async def put_secret(self, secret_id: str, secret_value: Union[str, Dict[str, Any]], 
                        secret_type: SecretType, description: str = "") -> bool:
        """Store secret in the first available provider."""
        for provider in self.providers:
            try:
                success = await provider.put_secret(secret_id, secret_value, secret_type, description)
                if success:
                    # Invalidate cache
                    if secret_id in self.cache:
                        del self.cache[secret_id]
                        del self.cache_timestamps[secret_id]
                    
                    self.logger.info(f"Secret {secret_id} stored in {provider.__class__.__name__}")
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Provider {provider.__class__.__name__} failed to store {secret_id}: {e}")
                continue
        
        self.logger.error(f"Failed to store secret {secret_id} in any provider")
        return False
    
    async def get_database_credentials(self, database_name: str) -> Optional[Dict[str, str]]:
        """Get database credentials."""
        secret = await self.get_secret(f"database/{database_name}")
        if secret:
            return secret.get_dict_value()
        return None
    
    async def get_s3_credentials(self) -> Optional[Dict[str, str]]:
        """Get S3 credentials."""
        secret = await self.get_secret("aws/s3_credentials")
        if secret:
            return secret.get_dict_value()
        return None
    
    async def get_broker_credentials(self, broker_name: str) -> Optional[Dict[str, str]]:
        """Get broker API credentials."""
        secret = await self.get_secret(f"broker/{broker_name}")
        if secret:
            return secret.get_dict_value()
        return None
    
    async def get_api_key(self, service_name: str) -> Optional[str]:
        """Get API key for service."""
        secret = await self.get_secret(f"api_key/{service_name}")
        if secret:
            return secret.get_string_value()
        return None
    
    def clear_cache(self) -> None:
        """Clear secret cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Secret cache cleared")


def create_secrets_manager() -> SecretsManager:
    """Factory function to create SecretsManager with environment-based configuration."""
    providers = []
    
    # Determine environment and configure providers
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production':
        # Production: AWS Secrets Manager primary, Vault secondary
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        providers.append(AWSSecretsManagerProvider(aws_region))
        
        vault_url = os.getenv('VAULT_URL')
        if vault_url:
            providers.append(HashiCorpVaultProvider(vault_url))
    
    elif environment == 'staging':
        # Staging: Vault primary, AWS secondary
        vault_url = os.getenv('VAULT_URL')
        if vault_url:
            providers.append(HashiCorpVaultProvider(vault_url))
        
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        providers.append(AWSSecretsManagerProvider(aws_region))
    
    else:
        # Development: Local encrypted storage
        storage_path = Path(os.getenv('SECRETS_STORAGE_PATH', './secrets'))
        providers.append(LocalEncryptedProvider(storage_path))
    
    if not providers:
        # Fallback to local encrypted storage
        storage_path = Path('./secrets')
        providers.append(LocalEncryptedProvider(storage_path))
        logging.warning("No secret providers configured, using local encrypted storage")
    
    return SecretsManager(providers)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_secrets_manager():
        """Test secrets manager functionality."""
        # Create secrets manager
        secrets_manager = create_secrets_manager()
        
        # Test storing secrets
        print("Testing secret storage...")
        
        # Store database credentials
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "username": "trading_user",
            "password": "secure_password_123",
            "database": "intradayjules"
        }
        
        success = await secrets_manager.put_secret(
            "database/main",
            db_creds,
            SecretType.DATABASE_PASSWORD,
            "Main trading database credentials"
        )
        print(f"Database credentials stored: {success}")
        
        # Store S3 credentials
        s3_creds = {
            "access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "region": "us-east-1"
        }
        
        success = await secrets_manager.put_secret(
            "aws/s3_credentials",
            s3_creds,
            SecretType.S3_CREDENTIALS,
            "S3 bucket access credentials"
        )
        print(f"S3 credentials stored: {success}")
        
        # Store broker credentials
        broker_creds = {
            "api_key": "your_broker_api_key",
            "api_secret": "your_broker_api_secret",
            "account_id": "your_account_id",
            "base_url": "https://api.broker.com"
        }
        
        success = await secrets_manager.put_secret(
            "broker/interactive_brokers",
            broker_creds,
            SecretType.BROKER_CREDENTIALS,
            "Interactive Brokers API credentials"
        )
        print(f"Broker credentials stored: {success}")
        
        # Test retrieving secrets
        print("\nTesting secret retrieval...")
        
        # Get database credentials
        db_creds_retrieved = await secrets_manager.get_database_credentials("main")
        print(f"Database credentials: {db_creds_retrieved}")
        
        # Get S3 credentials
        s3_creds_retrieved = await secrets_manager.get_s3_credentials()
        print(f"S3 credentials: {s3_creds_retrieved}")
        
        # Get broker credentials
        broker_creds_retrieved = await secrets_manager.get_broker_credentials("interactive_brokers")
        print(f"Broker credentials: {broker_creds_retrieved}")
        
        # Test API key
        await secrets_manager.put_secret(
            "api_key/pagerduty",
            "pd_api_key_12345",
            SecretType.API_KEY,
            "PagerDuty integration key"
        )
        
        api_key = await secrets_manager.get_api_key("pagerduty")
        print(f"PagerDuty API key: {api_key}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_secrets_manager())