import base64
# src/security/backends/local_vault.py
"""
Local file-based vault backend with atomic operations and concurrency safety.
Uses cross-platform file locking for Windows/Linux compatibility.
"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List
import logging

# Cross-platform file locking
try:
    import portalocker
    PORTALOCKER_AVAILABLE = True
except ImportError:
    import fcntl
    PORTALOCKER_AVAILABLE = False

from ..protocols import VaultBackend, SecretData, SecretMetadata


class VaultCorruptionError(Exception):
    """Raised when vault file is corrupted or unreadable."""
    pass


class LocalVaultBackend:
    """
    Local file-based vault backend with atomic operations.
    
    Features:
    - Atomic file operations
    - Cross-platform file locking
    - In-memory caching for performance
    - JSON-based storage with metadata
    """
    
    def __init__(self, vault_path: str = "secrets.vault", password: str = None):
        vault_path_obj = Path(vault_path)
        
        # If vault_path is a directory, create a vault file inside it
        if vault_path_obj.is_dir():
            self.vault_file = vault_path_obj / "secrets.vault"
        else:
            self.vault_file = vault_path_obj
            
        self.lock_file = Path(f"{self.vault_file}.lock")
        self.logger = logging.getLogger("LocalVaultBackend")
        
        # In-memory cache for performance
        self._cache: Dict[str, SecretData] = {}
        self._cache_loaded = False
        
        # Ensure directory exists
        self.vault_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Enter async context - load vault."""
        await self._load_vault()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
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
    
    async def _load_vault(self):
        """Load vault from disk with file locking."""
        if self._cache_loaded:
            return
        
        if not self.vault_file.exists():
            self.logger.info("Vault file doesn't exist, starting with empty vault")
            self._cache = {}
            self._cache_loaded = True
            return
        
        # Acquire lock for reading
        lock_handle = await asyncio.get_event_loop().run_in_executor(
            None, self._acquire_lock
        )
        
        try:
            async with aiofiles.open(self.vault_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            if content.strip():
                vault_data = json.loads(content)
                
                # Convert dict back to SecretData objects
                for key, data in vault_data.items():
                    # Reconstruct metadata
                    metadata_dict = data['metadata']
                    metadata = SecretMetadata(**metadata_dict)
                    
                    # Create SecretData object
                    secret_data = SecretData(
                        key=key,
                        value=data['value'],
                        metadata=metadata
                    )
                    
                    self._cache[key] = secret_data
                
                self.logger.info(f"Loaded {len(self._cache)} secrets from vault")
            else:
                self._cache = {}
                self.logger.info("Empty vault file, starting with empty cache")
                
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error loading vault: {e}")
            self._cache = {}
        finally:
            await asyncio.get_event_loop().run_in_executor(
                None, self._release_lock, lock_handle
            )
        
        self._cache_loaded = True
    
    async def _save_vault(self):
        """Save vault to disk atomically with file locking."""
        # Convert cache to serializable format
        vault_data = {}
        for key, secret_data in self._cache.items():
            vault_data[key] = {
                'value': secret_data.value,
                'metadata': secret_data.metadata.dict()
            }
        
        # Combine salt + encrypted data
        vault_bytes = json.dumps(vault_data, indent=2).encode('utf-8')
        
        # Atomic write with file locking
        temp_file = self.vault_file.with_suffix('.tmp')
        lock_handle = await asyncio.get_event_loop().run_in_executor(
            None, self._acquire_lock
        )
        
        try:
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(vault_bytes)
                # fsync not available in aiofiles
            
            # Atomic replace
            temp_file.replace(self.vault_file)
            self.logger.debug("Vault saved atomically")
            
        finally:
            await asyncio.get_event_loop().run_in_executor(
                None, self._release_lock, lock_handle
            )
    
    async def store(self, key: str, encrypted_data: bytes, metadata: SecretMetadata) -> bool:
        """Store encrypted secret data."""
        try:
            await self._load_vault()
            
            # Note: encrypted_data is already encrypted by AdvancedSecretsManager
            # We just store it as-is with metadata
            secret_data = SecretData(
                key=key,
                value=base64.b64encode(encrypted_data).decode('utf-8'),  # Store as base64 string
                metadata=metadata
            )
            
            self._cache[key] = secret_data
            await self._save_vault()
            
            self.logger.debug(f"Stored secret '{key}' successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def retrieve(self, key: str) -> SecretData:
        """Retrieve secret data by key."""
        await self._load_vault()
        
        if key not in self._cache:
            raise KeyError(f"Secret '{key}' not found")
        
        secret_data = self._cache[key]
        
        # Update last accessed time
        from datetime import datetime
        secret_data.metadata.last_accessed = datetime.utcnow()
        await self._save_vault()  # Persist the access time update
        
        self.logger.debug(f"Retrieved secret '{key}' successfully")
        return secret_data
    
    async def delete(self, key: str) -> bool:
        """Delete a secret."""
        try:
            await self._load_vault()
            
            if key not in self._cache:
                self.logger.warning(f"Secret '{key}' not found for deletion")
                return False
            
            del self._cache[key]
            await self._save_vault()
            
            self.logger.debug(f"Deleted secret '{key}' successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """List all secret keys."""
        await self._load_vault()
        keys = list(self._cache.keys())
        self.logger.debug(f"Listed {len(keys)} secret keys")
        return keys
    
    async def exists(self, key: str) -> bool:
        """Check if a secret exists."""
        await self._load_vault()
        exists = key in self._cache
        self.logger.debug(f"Secret '{key}' exists: {exists}")
        return exists
