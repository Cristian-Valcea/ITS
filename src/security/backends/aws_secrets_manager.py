"""
AWS Secrets Manager Backend for ITS Secrets Management System

This backend provides integration with AWS Secrets Manager, following enterprise
best practices for credential management and region handling.

Features:
- Default AWS credential chain (roles → profiles → env vars → explicit keys)
- Configurable region support with SDK defaults
- Automatic STS handling for temporary credentials
- Secret prefixing for namespace isolation
- Retry logic with exponential backoff
- Comprehensive error handling and logging

Usage:
    # Basic usage with default credentials
    backend = AWSSecretsBackend()
    
    # With specific region and profile
    backend = AWSSecretsBackend(
        region="us-east-1",
        boto_profile="trading-prod"
    )
    
    # With explicit credentials (not recommended for production)
    backend = AWSSecretsBackend(
        explicit_access_key="AKIA...",
        explicit_secret_key="...",
        region="eu-west-1"
    )
"""

import json
import logging
import base64
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ..protocols import VaultBackend, SecretData, SecretMetadata, SecretType


class AWSSecretsBackend(VaultBackend):
    """
    AWS Secrets Manager backend implementation.
    
    Follows AWS best practices:
    - Uses default credential chain (IAM roles, profiles, env vars)
    - Supports explicit credentials for special cases
    - Automatic region resolution with override capability
    - Built-in retry logic and error handling
    - Secret prefixing for namespace isolation
    """
    
    def __init__(
        self,
        secret_prefix: str = "intraday/",
        region: Optional[str] = None,
        boto_profile: Optional[str] = None,
        explicit_access_key: Optional[str] = None,
        explicit_secret_key: Optional[str] = None,
        session_token: Optional[str] = None,
        max_retries: int = 10
    ):
        """
        Initialize AWS Secrets Manager backend.
        
        Args:
            secret_prefix: Prefix for all secret names (default: "intraday/")
            region: AWS region (None = use SDK default resolution)
            boto_profile: AWS profile name for credential resolution
            explicit_access_key: Explicit AWS access key (not recommended for prod)
            explicit_secret_key: Explicit AWS secret key (not recommended for prod)
            session_token: AWS session token for temporary credentials
            max_retries: Maximum retry attempts for AWS API calls
            
        Raises:
            ImportError: If boto3 is not installed
            NoCredentialsError: If no valid AWS credentials found
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Secrets Manager backend. "
                "Install with: pip install boto3>=1.34.0"
            )
        
        self.prefix = secret_prefix.rstrip('/') + '/' if secret_prefix else ''
        self.logger = logging.getLogger("AWSSecretsBackend")
        
        # Build session kwargs following credential precedence
        session_kwargs = {}
        if boto_profile:
            session_kwargs["profile_name"] = boto_profile
            self.logger.info(f"Using AWS profile: {boto_profile}")
        
        if explicit_access_key:
            session_kwargs.update(
                aws_access_key_id=explicit_access_key,
                aws_secret_access_key=explicit_secret_key,
                aws_session_token=session_token,
            )
            self.logger.warning("Using explicit AWS credentials (not recommended for production)")
        
        try:
            # Create session with credential chain
            self.session = boto3.Session(**session_kwargs)
            
            # Create Secrets Manager client with retry configuration
            self.client = self.session.client(
                "secretsmanager",
                region_name=region,
                config=Config(
                    retries={"max_attempts": max_retries, "mode": "adaptive"},
                    max_pool_connections=50
                )
            )
            
            # Log effective region
            effective_region = self.client.meta.region_name
            self.logger.info(f"AWS Secrets Manager backend initialized in region: {effective_region}")
            
        except NoCredentialsError as e:
            self.logger.error("No valid AWS credentials found")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Secrets Manager backend: {e}")
            raise
    
    def _get_secret_name(self, key: str) -> str:
        """Get full secret name with prefix."""
        return f"{self.prefix}{key}"
    
    def _parse_secret_name(self, full_name: str) -> str:
        """Parse key from full secret name."""
        if full_name.startswith(self.prefix):
            return full_name[len(self.prefix):]
        return full_name
    
    def _metadata_to_tags(self, metadata: SecretMetadata) -> List[Dict[str, str]]:
        """Convert SecretMetadata to AWS tags format."""
        tags = []
        
        # Add metadata as tags
        tags.append({"Key": "SecretType", "Value": metadata.secret_type.value})
        tags.append({"Key": "CreatedAt", "Value": metadata.created_at.isoformat()})
        tags.append({"Key": "UpdatedAt", "Value": metadata.updated_at.isoformat()})
        tags.append({"Key": "RotationCount", "Value": str(metadata.rotation_count)})
        
        if metadata.expires_at:
            tags.append({"Key": "ExpiresAt", "Value": metadata.expires_at.isoformat()})
        
        if metadata.last_accessed:
            tags.append({"Key": "LastAccessed", "Value": metadata.last_accessed.isoformat()})
        
        if metadata.description:
            tags.append({"Key": "Description", "Value": metadata.description})
        
        # Add custom tags
        for tag_key, tag_value in metadata.tags.items():
            tags.append({"Key": f"Custom_{tag_key}", "Value": tag_value})
        
        return tags
    
    def _tags_to_metadata(self, tags: List[Dict[str, str]], secret_name: str) -> SecretMetadata:
        """Convert AWS tags to SecretMetadata."""
        tag_dict = {tag["Key"]: tag["Value"] for tag in tags}
        
        # Parse metadata from tags
        secret_type = SecretType(tag_dict.get("SecretType", "api_key"))
        created_at = datetime.fromisoformat(tag_dict.get("CreatedAt", datetime.utcnow().isoformat()))
        updated_at = datetime.fromisoformat(tag_dict.get("UpdatedAt", datetime.utcnow().isoformat()))
        rotation_count = int(tag_dict.get("RotationCount", "0"))
        
        expires_at = None
        if "ExpiresAt" in tag_dict:
            expires_at = datetime.fromisoformat(tag_dict["ExpiresAt"])
        
        last_accessed = None
        if "LastAccessed" in tag_dict:
            last_accessed = datetime.fromisoformat(tag_dict["LastAccessed"])
        
        description = tag_dict.get("Description", "")
        
        # Extract custom tags
        custom_tags = {}
        for key, value in tag_dict.items():
            if key.startswith("Custom_"):
                custom_key = key[7:]  # Remove "Custom_" prefix
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
        Store encrypted secret in AWS Secrets Manager.
        
        Args:
            key: Secret identifier
            encrypted_data: Encrypted secret value
            metadata: Secret metadata
            
        Returns:
            True if successful, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Encode encrypted data as base64 for JSON storage
            secret_value = base64.b64encode(encrypted_data).decode('utf-8')
            
            # Prepare secret data
            secret_data = {
                "encrypted_value": secret_value,
                "metadata": metadata.dict()
            }
            
            # Convert metadata to tags
            tags = self._metadata_to_tags(metadata)
            
            try:
                # Try to update existing secret
                self.client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(secret_data)
                )
                
                # Update tags
                self.client.tag_resource(
                    SecretId=secret_name,
                    Tags=tags
                )
                
                self.logger.debug(f"Updated existing secret: {key}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new secret
                    self.client.create_secret(
                        Name=secret_name,
                        SecretString=json.dumps(secret_data),
                        Description=f"ITS Secret: {metadata.description or key}",
                        Tags=tags
                    )
                    
                    self.logger.debug(f"Created new secret: {key}")
                else:
                    raise
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            self.logger.error(f"AWS error storing secret '{key}': {error_code} - {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Tuple[bytes, SecretMetadata]]:
        """
        Retrieve encrypted secret from AWS Secrets Manager.
        
        Args:
            key: Secret identifier
            
        Returns:
            Tuple of (encrypted_data, metadata) or None if not found
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Get secret value
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_string = response['SecretString']
            
            # Parse secret data
            secret_data = json.loads(secret_string)
            encrypted_value = secret_data['encrypted_value']
            metadata_dict = secret_data['metadata']
            
            # Decode encrypted data
            encrypted_data = base64.b64decode(encrypted_value.encode('utf-8'))
            
            # Reconstruct metadata from stored data
            metadata = SecretMetadata(**metadata_dict)
            
            # Update last accessed time
            metadata.last_accessed = datetime.utcnow()
            
            self.logger.debug(f"Retrieved secret: {key}")
            return encrypted_data, metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                self.logger.debug(f"Secret not found: {key}")
                return None
            else:
                self.logger.error(f"AWS error retrieving secret '{key}': {error_code} - {e}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete secret from AWS Secrets Manager.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if successful, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            # Schedule secret for deletion (7-day recovery window by default)
            self.client.delete_secret(
                SecretId=secret_name,
                RecoveryWindowInDays=7
            )
            
            self.logger.info(f"Scheduled secret for deletion: {key} (7-day recovery window)")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                self.logger.debug(f"Secret not found for deletion: {key}")
                return True  # Consider non-existent as successfully deleted
            else:
                self.logger.error(f"AWS error deleting secret '{key}': {error_code} - {e}")
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
            paginator = self.client.get_paginator('list_secrets')
            
            # List secrets with prefix filter
            page_iterator = paginator.paginate(
                Filters=[
                    {
                        'Key': 'name',
                        'Values': [f'{self.prefix}*']
                    }
                ]
            )
            
            for page in page_iterator:
                for secret in page['SecretList']:
                    secret_name = secret['Name']
                    key = self._parse_secret_name(secret_name)
                    keys.append(key)
            
            self.logger.debug(f"Listed {len(keys)} secrets")
            return keys
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            self.logger.error(f"AWS error listing secrets: {error_code} - {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def exists(self, key: str) -> bool:
        """
        Check if secret exists in AWS Secrets Manager.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if secret exists, False otherwise
        """
        secret_name = self._get_secret_name(key)
        
        try:
            self.client.describe_secret(SecretId=secret_name)
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                return False
            else:
                self.logger.error(f"AWS error checking secret existence '{key}': {error_code} - {e}")
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
            self.client.list_secrets(MaxResults=1)
            self.logger.debug("AWS Secrets Manager backend connection validated")
            return self
        except Exception as e:
            self.logger.error(f"Failed to validate AWS connection: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - cleanup if needed."""
        # AWS SDK handles connection cleanup automatically
        pass
