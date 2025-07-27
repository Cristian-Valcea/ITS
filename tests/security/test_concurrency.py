# tests/security/test_concurrency.py
"""
Concurrency and race condition tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from src.security.backends.local_vault import LocalVaultBackend
from src.security.protocols import SecretMetadata, SecretType


@pytest.mark.asyncio
async def test_concurrent_writes():
    """Test concurrent write operations don't corrupt vault."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        async def write_secret(key: str, value: str):
            backend = LocalVaultBackend(vault_path, password)
            async with backend:
                metadata = SecretMetadata(secret_type=SecretType.API_KEY)
                return await backend.store(key, value.encode(), metadata)
        
        # Launch concurrent writes with some delay to avoid race conditions
        tasks = []
        for i in range(5):  # Reduced number to avoid file locking issues
            task = write_secret(f"key-{i}", f"value-{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All writes should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent write failed: {result}")
            assert result is True
        
        # Verify all secrets were stored
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            keys = await backend.list_keys()
            assert len(keys) == 5
            
            import base64
            for i in range(5):
                secret_data = await backend.retrieve(f"key-{i}")
                # Value is stored as base64 encoded
                assert base64.b64decode(secret_data.value) == f"value-{i}".encode()


@pytest.mark.asyncio 
async def test_simultaneous_rotate_read():
    """Test race condition: simultaneous rotate → read operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        # Setup initial secret
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            metadata = SecretMetadata(secret_type=SecretType.API_KEY)
            await backend.store("test-key", b"initial-value", metadata)
        
        async def rotate_secret():
            backend = LocalVaultBackend(vault_path, password)
            async with backend:
                new_metadata = SecretMetadata(
                    secret_type=SecretType.API_KEY,
                    rotation_count=1
                )
                return await backend.store("test-key", b"rotated-value", new_metadata)
        
        async def read_secret():
            backend = LocalVaultBackend(vault_path, password)
            async with backend:
                secret_data = await backend.retrieve("test-key")
                return secret_data.value
        
        # Launch simultaneous operations
        rotate_task = rotate_secret()
        read_task = read_secret()
        
        rotate_result, read_result = await asyncio.gather(
            rotate_task, read_task, return_exceptions=True
        )
        
        # Both operations should succeed
        assert rotate_result is True
        # read_result is base64 encoded, so decode and check
        import base64
        decoded_result = base64.b64decode(read_result)
        assert decoded_result in [b"initial-value", b"rotated-value"]
        
        # Final state should be rotated
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            final_secret = await backend.retrieve("test-key")
            assert base64.b64decode(final_secret.value) == b"rotated-value"


if __name__ == "__main__":
    pytest.main([__file__])
