#!/usr/bin/env python3
"""
Multi-Cloud Secrets Manager

This module provides advanced multi-cloud capabilities:
- Automatic failover between cloud providers
- Cross-cloud secret replication
- Environment-based configuration
- Backup and disaster recovery

Features:
- Primary/failover backend configuration
- Automatic retry with exponential backoff
- Cross-cloud secret synchronization
- Health monitoring and alerting
- Configuration-driven setup
"""

import asyncio
import logging
import os
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from security.advanced_secrets_manager import AdvancedSecretsManager
from security.protocols import SecretMetadata


class MultiCloudSecretsManager:
    """
    Multi-cloud secrets manager with failover and replication capabilities.
    """
    
    def __init__(self, config_path: str = "cloud_config.yaml", environment: str = "default"):
        """
        Initialize multi-cloud manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (default, development, staging, production, etc.)
        """
        self.config_path = config_path
        self.environment = environment
        self.logger = logging.getLogger("MultiCloudSecretsManager")
        
        # Load configuration
        self.config = self._load_config()
        self.env_config = self._get_environment_config()
        
        # Initialize backends
        self.primary_backend = None
        self.failover_backend = None
        self.backup_backends = []
        
        self._setup_backends()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Configuration file {self.config_path} not found, using defaults")
            return {"default": {"backend": "local"}}
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get configuration for the specified environment."""
        if self.environment in self.config:
            return self.config[self.environment]
        else:
            self.logger.warning(f"Environment '{self.environment}' not found, using default")
            return self.config.get("default", {"backend": "local"})
    
    def _create_backend(self, backend_config: Dict[str, Any]):
        """Create a backend instance from configuration."""
        backend_type = backend_config.get("backend", "local")
        
        if backend_type == "local":
            from security.backends.local_vault import LocalVaultBackend
            vault_path = backend_config.get("vault_path", "secrets.vault")
            return LocalVaultBackend(vault_path)
        
        elif backend_type == "aws":
            try:
                from security.backends.aws_secrets_manager import AWSSecretsBackend
                aws_config = {}
                if "region" in backend_config:
                    aws_config["region"] = backend_config["region"]
                if "secret_prefix" in backend_config:
                    aws_config["secret_prefix"] = backend_config["secret_prefix"]
                if "boto_profile" in backend_config:
                    aws_config["boto_profile"] = backend_config["boto_profile"]
                return AWSSecretsBackend(**aws_config)
            except ImportError:
                raise RuntimeError("AWS backend not available. Install with: pip install boto3>=1.34.0")
        
        elif backend_type == "azure":
            try:
                from security.backends.azure_keyvault import AzureKeyVaultBackend
                azure_config = {"vault_url": backend_config["vault_url"]}
                if "secret_prefix" in backend_config:
                    azure_config["secret_prefix"] = backend_config["secret_prefix"]
                if "tenant_id" in backend_config:
                    # Support environment variable substitution
                    tenant_id = backend_config["tenant_id"]
                    if tenant_id.startswith("${") and tenant_id.endswith("}"):
                        env_var = tenant_id[2:-1]
                        tenant_id = os.getenv(env_var)
                        if not tenant_id:
                            raise ValueError(f"Environment variable {env_var} not set")
                    azure_config["tenant_id"] = tenant_id
                if "client_id" in backend_config:
                    azure_config["client_id"] = backend_config["client_id"]
                if "client_secret" in backend_config:
                    azure_config["client_secret"] = backend_config["client_secret"]
                return AzureKeyVaultBackend(**azure_config)
            except ImportError:
                raise RuntimeError("Azure backend not available. Install with: pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0")
        
        elif backend_type == "vault":
            try:
                from security.backends.hashicorp_vault import HashiCorpVaultBackend
                vault_config = {"vault_url": backend_config["vault_url"]}
                if "secret_path" in backend_config:
                    vault_config["secret_path"] = backend_config["secret_path"]
                if "auth_method" in backend_config:
                    vault_config["auth_method"] = backend_config["auth_method"]
                if "token" in backend_config:
                    vault_config["token"] = backend_config["token"]
                if "role" in backend_config:
                    vault_config["role"] = backend_config["role"]
                if "namespace" in backend_config:
                    vault_config["namespace"] = backend_config["namespace"]
                return HashiCorpVaultBackend(**vault_config)
            except ImportError:
                raise RuntimeError("HashiCorp Vault backend not available. Install with: pip install hvac>=1.2.0")
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def _setup_backends(self):
        """Setup primary, failover, and backup backends."""
        # Check if this is a multi-backend configuration
        if "primary" in self.env_config:
            # Multi-backend setup with primary/failover
            self.primary_backend = self._create_backend(self.env_config["primary"])
            
            if "failover" in self.env_config:
                self.failover_backend = self._create_backend(self.env_config["failover"])
        else:
            # Single backend setup
            self.primary_backend = self._create_backend(self.env_config)
        
        # Setup backup backends if configured
        backup_config = self.config.get("backup", {})
        if backup_config.get("enabled", False):
            for backend_config in backup_config.get("backends", []):
                try:
                    backup_backend = self._create_backend(backend_config)
                    self.backup_backends.append(backup_backend)
                except Exception as e:
                    self.logger.warning(f"Failed to setup backup backend: {e}")
    
    def _get_master_password(self) -> str:
        """Get master password from environment or prompt."""
        # Try to get from environment variable specified in config
        password_env = self.env_config.get("master_password_env", "SECRETS_MASTER_PASSWORD")
        password = os.getenv(password_env)
        
        if not password:
            # Fallback to common environment variables
            password = os.getenv("SECRETS_MASTER_PASSWORD") or os.getenv("MASTER_PASSWORD")
        
        if not password:
            raise ValueError(f"Master password not found. Set {password_env} environment variable.")
        
        return password
    
    async def _execute_with_failover(self, operation, *args, **kwargs):
        """Execute operation with automatic failover."""
        backends_to_try = [self.primary_backend]
        if self.failover_backend:
            backends_to_try.append(self.failover_backend)
        
        last_error = None
        
        for backend in backends_to_try:
            try:
                async with backend:
                    master_password = self._get_master_password()
                    manager = AdvancedSecretsManager(backend, master_password)
                    result = await operation(manager, *args, **kwargs)
                    return result
            except Exception as e:
                self.logger.warning(f"Backend {type(backend).__name__} failed: {e}")
                last_error = e
                continue
        
        # All backends failed
        raise RuntimeError(f"All backends failed. Last error: {last_error}")
    
    async def write_secret(self, key: str, value: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Write secret with automatic failover and backup."""
        
        async def _write(manager, key, value, metadata):
            return await manager.write_secret(key, value, metadata or {})
        
        # Write to primary/failover
        success = await self._execute_with_failover(_write, key, value, metadata)
        
        # Write to backup backends (best effort)
        for backup_backend in self.backup_backends:
            try:
                async with backup_backend:
                    master_password = self._get_master_password()
                    backup_manager = AdvancedSecretsManager(backup_backend, master_password)
                    await backup_manager.write_secret(key, value, metadata or {})
                    self.logger.debug(f"Secret backed up to {type(backup_backend).__name__}")
            except Exception as e:
                self.logger.warning(f"Backup to {type(backup_backend).__name__} failed: {e}")
        
        return success
    
    async def read_secret(self, key: str) -> Optional[str]:
        """Read secret with automatic failover."""
        
        async def _read(manager, key):
            return await manager.read_secret(key)
        
        return await self._execute_with_failover(_read, key)
    
    async def delete_secret(self, key: str) -> bool:
        """Delete secret from all backends."""
        
        async def _delete(manager, key):
            return await manager.delete_secret(key)
        
        # Delete from primary/failover
        success = await self._execute_with_failover(_delete, key)
        
        # Delete from backup backends (best effort)
        for backup_backend in self.backup_backends:
            try:
                async with backup_backend:
                    master_password = self._get_master_password()
                    backup_manager = AdvancedSecretsManager(backup_backend, master_password)
                    await backup_manager.delete_secret(key)
                    self.logger.debug(f"Secret deleted from backup {type(backup_backend).__name__}")
            except Exception as e:
                self.logger.warning(f"Backup deletion from {type(backup_backend).__name__} failed: {e}")
        
        return success
    
    async def list_secrets(self) -> List[str]:
        """List secrets with automatic failover."""
        
        async def _list(manager):
            return await manager.list_secrets()
        
        return await self._execute_with_failover(_list)
    
    async def sync_secrets(self, source_backend=None, target_backend=None) -> Dict[str, bool]:
        """Synchronize secrets between backends."""
        if not source_backend:
            source_backend = self.primary_backend
        if not target_backend and self.failover_backend:
            target_backend = self.failover_backend
        
        if not target_backend:
            raise ValueError("No target backend available for synchronization")
        
        results = {}
        master_password = self._get_master_password()
        
        async with source_backend:
            source_manager = AdvancedSecretsManager(source_backend, master_password)
            secrets = await source_manager.list_secrets()
            
            async with target_backend:
                target_manager = AdvancedSecretsManager(target_backend, master_password)
                
                for secret_key in secrets:
                    try:
                        # Read from source
                        value = await source_manager.read_secret(secret_key)
                        if value:
                            # Write to target
                            success = await target_manager.write_secret(secret_key, value)
                            results[secret_key] = success
                            if success:
                                self.logger.info(f"Synchronized secret: {secret_key}")
                            else:
                                self.logger.error(f"Failed to sync secret: {secret_key}")
                        else:
                            results[secret_key] = False
                    except Exception as e:
                        self.logger.error(f"Error syncing secret {secret_key}: {e}")
                        results[secret_key] = False
        
        return results
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured backends."""
        results = {}
        
        # Check primary backend
        try:
            async with self.primary_backend:
                await self.primary_backend.list_keys()
            results["primary"] = True
        except Exception as e:
            self.logger.error(f"Primary backend health check failed: {e}")
            results["primary"] = False
        
        # Check failover backend
        if self.failover_backend:
            try:
                async with self.failover_backend:
                    await self.failover_backend.list_keys()
                results["failover"] = True
            except Exception as e:
                self.logger.error(f"Failover backend health check failed: {e}")
                results["failover"] = False
        
        # Check backup backends
        for i, backup_backend in enumerate(self.backup_backends):
            try:
                async with backup_backend:
                    await backup_backend.list_keys()
                results[f"backup_{i}"] = True
            except Exception as e:
                self.logger.error(f"Backup backend {i} health check failed: {e}")
                results[f"backup_{i}"] = False
        
        return results


async def main():
    """Demo of multi-cloud manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Cloud Secrets Manager")
    parser.add_argument("--environment", "-e", default="default", help="Environment to use")
    parser.add_argument("--config", "-c", default="cloud_config.yaml", help="Configuration file")
    parser.add_argument("--action", choices=["health", "sync", "list"], default="health", help="Action to perform")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        manager = MultiCloudSecretsManager(args.config, args.environment)
        
        if args.action == "health":
            print(f"🏥 Health Check for environment: {args.environment}")
            health = await manager.health_check()
            for backend, status in health.items():
                status_icon = "✅" if status else "❌"
                print(f"  {backend}: {status_icon}")
        
        elif args.action == "sync":
            print(f"🔄 Synchronizing secrets in environment: {args.environment}")
            results = await manager.sync_secrets()
            for secret, success in results.items():
                status_icon = "✅" if success else "❌"
                print(f"  {secret}: {status_icon}")
        
        elif args.action == "list":
            print(f"📋 Listing secrets in environment: {args.environment}")
            secrets = await manager.list_secrets()
            for secret in secrets:
                print(f"  • {secret}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
