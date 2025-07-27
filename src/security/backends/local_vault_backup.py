import base64
# src/security/backends/local_vault.py
"""
Local file-based vault backend with atomic operations and concurrency safety.
Uses cross-platform file locking for Windows/Linux compatibility.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

try:
    import portalocker
    PORTALOCKER_AVAILABLE = True
except ImportError:
    PORTALOCKER_AVAILABLE = False
    import fcntl  # Fallback for Linux

import aiofiles
from ..protocols import VaultBackend, SecretData, SecretMetadata
from ..encryption import HardenedEncryption


class VaultCorruptionError(Exception):
    """Raised when vault file is corrupted - fail closed."""
    pass


class LocalVaultBackend:
    """
    Local encrypted file-based vault with atomic operations.
    Implements the VaultBackend protocol.
    """
    
    def __init__(self, vault_path: Path, master_password: str):
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.vault_file = self.vault_path / "secrets.vault"
        self.lock_file = self.vault_path / "secrets.lock"
        
        self.master_password = master_password
        self.encryption = HardenedEncryption()
        self.logger = logging.getLogger("LocalVaultBackend")
        
        # In-memory cache for performance
        self._cache: Dict[str, SecretData] = {}
        self._cache_loaded = False
    
    async def __aenter__(self) -> "LocalVaultBackend":
        """Enter async context - load vault if needed."""
        if not self._cache_loaded:
            await self._load_vault()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - cleanup."""
        # Could implement cache cleanup here if needed
        pass
    
    def _acquire_lock(self):
        """Acquire cross-platform file lock."""
        if PORTALOCKER_AVAILABLE:
            # Use portalocker for cross-platform compatibility
            lock_handle = open(self.lock_file, 'w')
            portalocker.lock(lock_handle, portalocker.LOCK_EX)
            return lock_handle
        else:
            # Fallback to fcntl on Linux
            lock_handle = open(self.lock_file, 'w')
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            return lock_handle
    
    def _release_lock(self, lock_handle):
        """Release file lock."""
        if PORTALOCKER_AVAILABLE:
            portalocker.unlock(lock_handle)
        else:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    
    async def _load_vault(self) -> None:
        """Load and decrypt vault file."""
        if not self.vault_file.exists():
            self._cache = {}
            self._cache_loaded = True
            return
        
        try:
            async with aiofiles.open(self.vault_file, 'rb') as f:
                vault_data = await f.read()
            
            if len(vault_data) < 32:  # Minimum size check
                raise VaultCorruptionError("Vault file too small - possibly corrupted")
            
            # Extract salt and encrypted data
            salt = vault_data[:32]
            encrypted_data = vault_data[32:]
            
            # Decrypt vault
            try:
                decrypted_json = self.encryption.decrypt(encrypted_data, salt, self.master_password)
                vault_contents = json.loads(decrypted_json)
            except Exception as e:
                raise VaultCorruptionError(f"Failed to decrypt vault: {e}")
            
            # Load secrets into cache
            self._cache = {}
            for key, secret_dict in vault_contents.items():
                try:
                    metadata = SecretMetadata(**secret_dict['metadata'])
                    self._cache[key] = SecretData(
                        key=key,
                        value=secret_dict['value'],
                        metadata=metadata
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load secret {key}: {e}")
                    # Continue loading other secrets
            
            self._cache_loaded = True
            self.logger.info(f"Loaded {len(self._cache)} secrets from vault")
            
        except VaultCorruptionError:
            raise  # Re-raise corruption errors
        except Exception as e:
            self.logger.error(f"Unexpected error loading vault: {e}")
            raise VaultCorruptionError(f"Vault loading failed: {e}")
    
    async def _save_vault(self) -> None:
        """Save and encrypt vault file atomically."""
        # Prepare data for encryption
        vault_contents = {}
        for key, secret_data in self._cache.items():
            vault_contents[key] = {
                'value': secret_data.value,
                'metadata': secret_data.metadata.dict()
            }
        
        # Encrypt data
        vault_json = json.dumps(vault_contents, indent=2)
        encrypted_data, salt = self.encryption.encrypt(vault_json, self.master_password)
        
        # Combine salt + encrypted data
        vault_bytes = salt + encrypted_data
        
        # Atomic write with file locking
        temp_file = self.vault_file.with_suffix('.tmp')
        lock_handle = await asyncio.get_event_loop().run_in_executor(
            None, self._acquire_lock
        )
        
        try:
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(vault_bytes)
                await f.fsync()  # Force write to disk
            
            # Atomic replace
            temp_file.replace(self.vault_file)
            self.logger.debug("Vault saved atomically")
            
        finally:
            await asyncio.get_event_loop().run_in_executor(
                None, self._release_lock, lock_handle
            )
    
    async def store(self, key: str, encrypted_data: bytes, metadata: SecretMetadata) -> bool:
        """Store encrypted secret data with metadata."""
        try:
            # Note: encrypted_data is already encrypted by AdvancedSecretsManager
            # We just store it as-is with metadata
            secret_data = SecretData(
                key=key,
                value=base64.b64encode(encrypted_data).decode('utf-8'),  # Store as base64 string
                metadata=metadata
            )
            
            self._cache[key] = secret_data
            await self._save_vault()
            
            self.logger.info(f"Secret '{key}' stored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def retrieve(self, key: str) -> SecretData:
        """Retrieve secret data."""
        if not self._cache_loaded:
            await self._load_vault()
        
        if key not in self._cache:
            raise KeyError(f"Secret '{key}' not found")
        
        secret_data = self._cache[key]
        
        # Update last accessed time
        secret_data.metadata.updated_at = datetime.utcnow()
        await self._save_vault()  # Persist access time update
        
        self.logger.debug(f"Secret '{key}' retrieved")
        return secret_data
    
    async def delete(self, key: str) -> bool:
        """Delete secret by key."""
        try:
            if key in self._cache:
                del self._cache[key]
                await self._save_vault()
                self.logger.info(f"Secret '{key}' deleted")
                return True
            else:
                self.logger.warning(f"Secret '{key}' not found for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """List all secret keys."""
        if not self._cache_loaded:
            await self._load_vault()
        
        return list(self._cache.keys())
    
    async def exists(self, key: str) -> bool:
        """Check if secret exists."""
        if not self._cache_loaded:
            await self._load_vault()
        
        return key in self._cache
