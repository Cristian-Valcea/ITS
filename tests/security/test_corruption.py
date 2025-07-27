# tests/security/test_corruption.py
"""
Vault corruption and recovery tests.
"""

import pytest
import tempfile
from pathlib import Path
from src.security.backends.local_vault import LocalVaultBackend, VaultCorruptionError
from src.security.protocols import SecretMetadata, SecretType


@pytest.mark.asyncio
async def test_corrupted_vault_detection():
    """Test detection of corrupted vault files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        vault_file = vault_path / "secrets.vault"
        password = "test-password-123"
        
        # Create corrupted vault file (too small)
        vault_file.write_bytes(b"corrupted")
        
        backend = LocalVaultBackend(vault_path, password)
        
        with pytest.raises(VaultCorruptionError):
            async with backend:
                await backend.list_keys()


@pytest.mark.asyncio
async def test_invalid_password_detection():
    """Test detection of wrong master password."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        correct_password = "correct-password"
        wrong_password = "wrong-password"
        
        # Create vault with correct password
        backend = LocalVaultBackend(vault_path, correct_password)
        async with backend:
            metadata = SecretMetadata(secret_type=SecretType.API_KEY)
            await backend.store("test-key", b"test-value", metadata)
        
        # Try to access with wrong password
        backend_wrong = LocalVaultBackend(vault_path, wrong_password)
        
        with pytest.raises(VaultCorruptionError):
            async with backend_wrong:
                await backend_wrong.list_keys()


@pytest.mark.asyncio
async def test_empty_vault_handling():
    """Test handling of empty/new vault."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            # Should handle empty vault gracefully
            keys = await backend.list_keys()
            assert keys == []
            
            # Should be able to add first secret
            metadata = SecretMetadata(secret_type=SecretType.API_KEY)
            success = await backend.store("first-key", b"first-value", metadata)
            assert success
            
            # Verify it was stored
            keys = await backend.list_keys()
            assert keys == ["first-key"]


if __name__ == "__main__":
    pytest.main([__file__])
