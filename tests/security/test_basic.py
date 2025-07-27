# tests/security/test_basic.py
"""
Basic functionality tests for secrets vault.
Target: Green test for write → read → rotate
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from src.security.backends.local_vault import LocalVaultBackend
from src.security.protocols import SecretMetadata, SecretType


@pytest.mark.asyncio
async def test_write_read_rotate():
    """Test basic write → read → rotate workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        # Test write
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            metadata = SecretMetadata(
                secret_type=SecretType.API_KEY,
                description="Test secret"
            )
            
            # Store secret
            success = await backend.store(
                "test-key", 
                b"test-value-encrypted", 
                metadata
            )
            assert success, "Failed to store secret"
            
            # Read secret
            secret_data = await backend.retrieve("test-key")
            assert secret_data.key == "test-key"
            # Value is stored as base64 encoded
            import base64
            assert base64.b64decode(secret_data.value) == b"test-value-encrypted"
            
            # Rotate secret
            new_metadata = SecretMetadata(
                secret_type=SecretType.API_KEY,
                description="Rotated test secret",
                rotation_count=1
            )
            
            success = await backend.store(
                "test-key",
                b"new-test-value-encrypted",
                new_metadata
            )
            assert success, "Failed to rotate secret"
            
            # Verify rotation
            rotated_secret = await backend.retrieve("test-key")
            # Value is stored as base64 encoded
            assert base64.b64decode(rotated_secret.value) == b"new-test-value-encrypted"
            assert rotated_secret.metadata.rotation_count == 1


@pytest.mark.asyncio
async def test_list_and_exists():
    """Test listing and existence checks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            # Initially empty
            keys = await backend.list_keys()
            assert keys == []
            
            # Add secrets
            metadata = SecretMetadata(secret_type=SecretType.API_KEY)
            await backend.store("key1", b"value1", metadata)
            await backend.store("key2", b"value2", metadata)
            
            # Check existence
            assert await backend.exists("key1")
            assert await backend.exists("key2")
            assert not await backend.exists("nonexistent")
            
            # List keys
            keys = await backend.list_keys()
            assert set(keys) == {"key1", "key2"}


if __name__ == "__main__":
    pytest.main([__file__])
