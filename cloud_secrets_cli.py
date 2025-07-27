#!/usr/bin/env python3
"""
Cloud Secrets CLI

Enhanced CLI for managing secrets across multiple cloud backends:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Local file-based vault

Usage:
    # AWS Secrets Manager
    python cloud_secrets_cli.py --backend aws --region us-east-1 set api-key "sk-1234567890"
    
    # Azure Key Vault
    python cloud_secrets_cli.py --backend azure --vault-url https://myvault.vault.azure.net/ set db-password "secure-pass"
    
    # HashiCorp Vault
    python cloud_secrets_cli.py --backend vault --vault-url https://vault.example.com:8200 --token hvs.ABC set jwt-secret "signing-key"
    
    # Local vault (default)
    python cloud_secrets_cli.py set local-secret "test-value"
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from security.advanced_secrets_manager import AdvancedSecretsManager
from security.backends import available_backends

# Setup logging
logging.basicConfig(level=logging.WARNING)

class CloudSecretsConfig:
    """Configuration for cloud secrets CLI."""
    
    def __init__(self):
        self.backend_type = "local"
        self.master_password = None
        self.backend_config = {}
    
    def get_backend(self):
        """Create and return the appropriate backend."""
        if self.backend_type == "local":
            from security.backends.local_vault import LocalVaultBackend
            vault_path = self.backend_config.get("vault_path", "secrets.vault")
            return LocalVaultBackend(vault_path)
        
        elif self.backend_type == "aws":
            try:
                from security.backends.aws_secrets_manager import AWSSecretsBackend
                return AWSSecretsBackend(**self.backend_config)
            except ImportError:
                raise click.ClickException("AWS backend not available. Install with: pip install boto3>=1.34.0")
        
        elif self.backend_type == "azure":
            try:
                from security.backends.azure_keyvault import AzureKeyVaultBackend
                return AzureKeyVaultBackend(**self.backend_config)
            except ImportError:
                raise click.ClickException("Azure backend not available. Install with: pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0")
        
        elif self.backend_type == "vault":
            try:
                from security.backends.hashicorp_vault import HashiCorpVaultBackend
                return HashiCorpVaultBackend(**self.backend_config)
            except ImportError:
                raise click.ClickException("HashiCorp Vault backend not available. Install with: pip install hvac>=1.2.0")
        
        else:
            raise click.ClickException(f"Unknown backend type: {self.backend_type}")

# Global config
config = CloudSecretsConfig()

@click.group()
@click.option('--backend', type=click.Choice(['local', 'aws', 'azure', 'vault']), default='local',
              help='Backend type to use')
@click.option('--master-password', envvar='SECRETS_MASTER_PASSWORD',
              help='Master password for encryption (can use SECRETS_MASTER_PASSWORD env var)')
@click.option('--vault-path', default='secrets.vault',
              help='Path to local vault file (local backend only)')
@click.option('--region', help='AWS region (AWS backend only)')
@click.option('--profile', help='AWS profile (AWS backend only)')
@click.option('--vault-url', help='Vault URL (Azure/HashiCorp backends)')
@click.option('--tenant-id', help='Azure tenant ID (Azure backend only)')
@click.option('--client-id', help='Azure client ID (Azure backend only)')
@click.option('--client-secret', help='Azure client secret (Azure backend only)')
@click.option('--token', help='Vault token (HashiCorp Vault only)')
@click.option('--auth-method', default='token', help='Vault auth method (HashiCorp Vault only)')
@click.option('--role-id', help='AppRole role ID (HashiCorp Vault only)')
@click.option('--secret-id', help='AppRole secret ID (HashiCorp Vault only)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(backend, master_password, vault_path, region, profile, vault_url, tenant_id, 
        client_id, client_secret, token, auth_method, role_id, secret_id, verbose):
    """Cloud Secrets Management CLI"""
    
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    config.backend_type = backend
    config.master_password = master_password
    
    # Configure backend-specific options
    if backend == "local":
        config.backend_config = {"vault_path": vault_path}
    
    elif backend == "aws":
        aws_config = {}
        if region:
            aws_config["region"] = region
        if profile:
            aws_config["boto_profile"] = profile
        config.backend_config = aws_config
    
    elif backend == "azure":
        if not vault_url:
            raise click.ClickException("--vault-url is required for Azure backend")
        
        azure_config = {"vault_url": vault_url}
        if tenant_id:
            azure_config["tenant_id"] = tenant_id
        if client_id and client_secret:
            azure_config["client_id"] = client_id
            azure_config["client_secret"] = client_secret
        config.backend_config = azure_config
    
    elif backend == "vault":
        if not vault_url:
            raise click.ClickException("--vault-url is required for HashiCorp Vault backend")
        
        vault_config = {"vault_url": vault_url, "auth_method": auth_method}
        if token:
            vault_config["token"] = token
        if role_id and secret_id:
            vault_config["role_id"] = role_id
            vault_config["secret_id"] = secret_id
        config.backend_config = vault_config

@cli.command()
@click.argument('key')
@click.argument('value')
@click.option('--description', help='Secret description')
@click.option('--tags', help='Tags in key=value,key2=value2 format')
def set(key, value, description, tags):
    """Set a secret value"""
    
    async def _set():
        if not config.master_password:
            config.master_password = click.prompt("Master password", hide_input=True)
        
        backend = config.get_backend()
        
        async with backend:
            manager = AdvancedSecretsManager(backend, config.master_password)
            
            # Parse tags
            tag_dict = {}
            if tags:
                for tag_pair in tags.split(','):
                    if '=' in tag_pair:
                        k, v = tag_pair.split('=', 1)
                        tag_dict[k.strip()] = v.strip()
            
            # Add description to tags if provided
            if description:
                tag_dict['description'] = description
            
            success = await manager.write_secret(key, value, tag_dict)
            
            if success:
                click.echo(f"✅ Secret '{key}' stored successfully in {config.backend_type} backend")
            else:
                click.echo(f"❌ Failed to store secret '{key}'")
                sys.exit(1)
    
    asyncio.run(_set())

@cli.command()
@click.argument('key')
def get(key):
    """Get a secret value"""
    
    async def _get():
        if not config.master_password:
            config.master_password = click.prompt("Master password", hide_input=True)
        
        backend = config.get_backend()
        
        async with backend:
            manager = AdvancedSecretsManager(backend, config.master_password)
            
            value = await manager.read_secret(key)
            
            if value is not None:
                click.echo(value)
            else:
                click.echo(f"❌ Secret '{key}' not found")
                sys.exit(1)
    
    asyncio.run(_get())

@cli.command()
@click.argument('key')
def delete(key):
    """Delete a secret"""
    
    async def _delete():
        if not config.master_password:
            config.master_password = click.prompt("Master password", hide_input=True)
        
        backend = config.get_backend()
        
        async with backend:
            manager = AdvancedSecretsManager(backend, config.master_password)
            
            success = await manager.delete_secret(key)
            
            if success:
                click.echo(f"✅ Secret '{key}' deleted successfully")
            else:
                click.echo(f"❌ Failed to delete secret '{key}'")
                sys.exit(1)
    
    asyncio.run(_delete())

@cli.command()
def list():
    """List all secrets"""
    
    async def _list():
        if not config.master_password:
            config.master_password = click.prompt("Master password", hide_input=True)
        
        backend = config.get_backend()
        
        async with backend:
            manager = AdvancedSecretsManager(backend, config.master_password)
            
            secrets = await manager.list_secrets()
            
            if secrets:
                click.echo(f"📋 Secrets in {config.backend_type} backend:")
                for secret_key in sorted(secrets):
                    click.echo(f"  • {secret_key}")
            else:
                click.echo("📭 No secrets found")
    
    asyncio.run(_list())

@cli.command()
@click.argument('key')
def info(key):
    """Get secret metadata"""
    
    async def _info():
        backend = config.get_backend()
        
        async with backend:
            # Get raw metadata without decryption
            result = await backend.retrieve(key)
            
            if result:
                encrypted_data, metadata = result
                click.echo(f"📊 Secret '{key}' metadata:")
                click.echo(f"  Type: {metadata.secret_type.value}")
                click.echo(f"  Created: {metadata.created_at}")
                click.echo(f"  Updated: {metadata.updated_at}")
                click.echo(f"  Rotations: {metadata.rotation_count}")
                if metadata.expires_at:
                    click.echo(f"  Expires: {metadata.expires_at}")
                if metadata.last_accessed:
                    click.echo(f"  Last accessed: {metadata.last_accessed}")
                if metadata.description:
                    click.echo(f"  Description: {metadata.description}")
                if metadata.tags:
                    click.echo(f"  Tags: {metadata.tags}")
            else:
                click.echo(f"❌ Secret '{key}' not found")
                sys.exit(1)
    
    asyncio.run(_info())

@cli.command()
def backends():
    """List available backends"""
    click.echo("🌐 Available backends:")
    
    # Local backend (always available)
    click.echo("  ✅ local - File-based vault")
    
    # Test cloud backends
    try:
        from security.backends.aws_secrets_manager import AWSSecretsBackend
        click.echo("  ✅ aws - AWS Secrets Manager")
    except ImportError:
        click.echo("  ❌ aws - AWS Secrets Manager (pip install boto3>=1.34.0)")
    
    try:
        from security.backends.azure_keyvault import AzureKeyVaultBackend
        click.echo("  ✅ azure - Azure Key Vault")
    except ImportError:
        click.echo("  ❌ azure - Azure Key Vault (pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0)")
    
    try:
        from security.backends.hashicorp_vault import HashiCorpVaultBackend
        click.echo("  ✅ vault - HashiCorp Vault")
    except ImportError:
        click.echo("  ❌ vault - HashiCorp Vault (pip install hvac>=1.2.0)")

@cli.command()
def examples():
    """Show usage examples"""
    click.echo("📖 Usage Examples:")
    click.echo()
    
    click.echo("🏠 Local Backend (default):")
    click.echo("  python cloud_secrets_cli.py set api-key 'sk-1234567890'")
    click.echo("  python cloud_secrets_cli.py get api-key")
    click.echo("  python cloud_secrets_cli.py list")
    click.echo()
    
    click.echo("☁️ AWS Secrets Manager:")
    click.echo("  python cloud_secrets_cli.py --backend aws --region us-east-1 set api-key 'sk-1234567890'")
    click.echo("  python cloud_secrets_cli.py --backend aws --profile trading-prod get api-key")
    click.echo()
    
    click.echo("🔷 Azure Key Vault:")
    click.echo("  python cloud_secrets_cli.py --backend azure --vault-url https://myvault.vault.azure.net/ set db-password 'secure-pass'")
    click.echo("  python cloud_secrets_cli.py --backend azure --vault-url https://myvault.vault.azure.net/ --tenant-id 12345 get db-password")
    click.echo()
    
    click.echo("🏛️ HashiCorp Vault:")
    click.echo("  python cloud_secrets_cli.py --backend vault --vault-url https://vault.example.com:8200 --token hvs.ABC set jwt-secret 'signing-key'")
    click.echo("  python cloud_secrets_cli.py --backend vault --vault-url https://vault.example.com:8200 --auth-method approle --role-id 123 --secret-id 456 get jwt-secret")
    click.echo()
    
    click.echo("🏷️ With metadata:")
    click.echo("  python cloud_secrets_cli.py set api-key 'sk-1234567890' --description 'OpenAI API key' --tags 'env=prod,service=trading'")

# Direct function API for programmatic use (expected by tests)
def set_secret(key: str, value: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> bool:
    """Programmatic API to set a secret."""
    async def _set():
        # Use default local backend for tests
        from security.backends.local_vault import LocalVaultBackend
        vault_path = os.environ.get('ITS_VAULT_PATH', 'secrets.vault')
        master_password = os.environ.get('ITS_MASTER_PASSWORD', 'default_test_password')
        
        backend = LocalVaultBackend(vault_path, master_password)
        async with backend:
            manager = AdvancedSecretsManager(backend, master_password)
            
            # Prepare metadata
            metadata_dict = {}
            if description:
                metadata_dict['description'] = description
            if tags:
                metadata_dict['tags'] = tags
            
            return await manager.write_secret(key, value, metadata_dict=metadata_dict)
    
    return asyncio.run(_set())

def get_secret(key: str) -> Optional[str]:
    """Programmatic API to get a secret."""
    async def _get():
        # Use default local backend for tests
        from security.backends.local_vault import LocalVaultBackend
        vault_path = os.environ.get('ITS_VAULT_PATH', 'secrets.vault')
        master_password = os.environ.get('ITS_MASTER_PASSWORD', 'default_test_password')
        
        backend = LocalVaultBackend(vault_path, master_password)
        async with backend:
            manager = AdvancedSecretsManager(backend, master_password)
            
            try:
                secret_data = await manager.read_secret(key)
                return secret_data['value']
            except (KeyError, Exception):
                return None
    
    return asyncio.run(_get())

def list_secrets() -> list:
    """Programmatic API to list secrets."""
    async def _list():
        # Use default local backend for tests
        from security.backends.local_vault import LocalVaultBackend
        vault_path = os.environ.get('ITS_VAULT_PATH', 'secrets.vault')
        master_password = os.environ.get('ITS_MASTER_PASSWORD', 'default_test_password')
        
        backend = LocalVaultBackend(vault_path, master_password)
        async with backend:
            manager = AdvancedSecretsManager(backend, master_password)
            return await manager.list_secrets()
    
    return asyncio.run(_list())

def delete_secret(key: str) -> bool:
    """Programmatic API to delete a secret."""
    async def _delete():
        # Use default local backend for tests
        from security.backends.local_vault import LocalVaultBackend
        vault_path = os.environ.get('ITS_VAULT_PATH', 'secrets.vault')
        master_password = os.environ.get('ITS_MASTER_PASSWORD', 'default_test_password')
        
        backend = LocalVaultBackend(vault_path, master_password)
        async with backend:
            manager = AdvancedSecretsManager(backend, master_password)
            return await manager.delete_secret(key)
    
    return asyncio.run(_delete())

if __name__ == '__main__':
    cli()
