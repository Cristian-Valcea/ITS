# src/security/cli.py
"""
CLI wrapper for secrets management - ops team friendly.
Usage: python -m src.security.cli <command> [args]
"""

import click
import asyncio
import os
from pathlib import Path
from .secrets_manager import AdvancedSecretsManager
from .backends.local_vault import LocalVaultBackend


@click.group()
@click.option('--vault-path', default='./vault', help='Path to vault directory')
@click.option('--password', envvar='SECRETS_MASTER_PASSWORD', help='Master password')
@click.pass_context
def cli(ctx, vault_path, password):
    """Secrets management CLI for IntradayJules Trading System."""
    if not password:
        password = click.prompt('Master password', hide_input=True)
    
    ctx.ensure_object(dict)
    ctx.obj['vault_path'] = Path(vault_path)
    ctx.obj['password'] = password


@cli.command()
@click.argument('key')
@click.argument('value')
@click.option('--description', default='', help='Secret description')
@click.pass_context
def set(ctx, key, value, description):
    """Set a secret value."""
    async def _set():
        backend = LocalVaultBackend(ctx.obj['vault_path'], ctx.obj['password'])
        async with backend:
            manager = AdvancedSecretsManager(backend)
            success = await manager.write_secret(key, value, {'description': description})
            if success:
                click.echo(f"✅ Secret '{key}' set successfully")
            else:
                click.echo(f"❌ Failed to set secret '{key}'")
                exit(1)
    
    asyncio.run(_set())


@cli.command()
@click.argument('key')
@click.pass_context
def get(ctx, key):
    """Get a secret value."""
    async def _get():
        backend = LocalVaultBackend(ctx.obj['vault_path'], ctx.obj['password'])
        async with backend:
            manager = AdvancedSecretsManager(backend)
            try:
                secret_data = await manager.read_secret(key)
                click.echo(secret_data['value'])
            except KeyError:
                click.echo(f"❌ Secret '{key}' not found")
                exit(1)
    
    asyncio.run(_get())


@cli.command()
@click.pass_context
def list(ctx):
    """List all secret keys."""
    async def _list():
        backend = LocalVaultBackend(ctx.obj['vault_path'], ctx.obj['password'])
        async with backend:
            keys = await backend.list_keys()
            if keys:
                click.echo("📋 Available secrets:")
                for key in sorted(keys):
                    click.echo(f"  • {key}")
            else:
                click.echo("📭 No secrets found")
    
    asyncio.run(_list())


@cli.command()
@click.argument('key')
@click.pass_context
def rotate(ctx, key):
    """Rotate a secret (generate new value)."""
    async def _rotate():
        backend = LocalVaultBackend(ctx.obj['vault_path'], ctx.obj['password'])
        async with backend:
            manager = AdvancedSecretsManager(backend)
            new_value = click.prompt(f'New value for {key}', hide_input=True)
            success = await manager.rotate_secret(key, new_value)
            if success:
                click.echo(f"🔄 Secret '{key}' rotated successfully")
            else:
                click.echo(f"❌ Failed to rotate secret '{key}'")
                exit(1)
    
    asyncio.run(_rotate())


if __name__ == '__main__':
    cli()
