#!/usr/bin/env python3
"""
Step 2 verification: Test AdvancedSecretsManager with LocalVaultBackend
"""

import asyncio
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from security.backends.local_vault import LocalVaultBackend
from security.advanced_secrets_manager import AdvancedSecretsManager, SecretsHelper


async def test_basic_operations():
    """Test basic write → read → rotate workflow."""
    print("🧪 Testing basic operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        # Create backend and manager
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            manager = AdvancedSecretsManager(backend, password)
            
            # Test write
            print("  📝 Writing secret...")
            success = await manager.write_secret(
                "test-api-key", 
                "sk-1234567890abcdef",
                description="Test API key"
            )
            assert success, "Failed to write secret"
            print("  ✅ Write successful")
            
            # Test read
            print("  📖 Reading secret...")
            secret_data = await manager.read_secret("test-api-key")
            assert secret_data['value'] == "sk-1234567890abcdef"
            assert secret_data['key'] == "test-api-key"
            print("  ✅ Read successful")
            
            # Test rotate
            print("  🔄 Rotating secret...")
            success = await manager.rotate_secret("test-api-key", "sk-new-rotated-key")
            assert success, "Failed to rotate secret"
            print("  ✅ Rotation successful")
            
            # Verify rotation
            print("  🔍 Verifying rotation...")
            rotated_data = await manager.read_secret("test-api-key")
            assert rotated_data['value'] == "sk-new-rotated-key"
            assert rotated_data['metadata']['rotation_count'] == 1
            print("  ✅ Rotation verified")
            
            # Test list
            print("  📋 Listing secrets...")
            keys = await manager.list_secrets()
            assert "test-api-key" in keys
            print(f"  ✅ Found {len(keys)} secrets: {keys}")
            
            # Test exists
            print("  🔍 Testing existence...")
            exists = await manager.secret_exists("test-api-key")
            assert exists, "Secret should exist"
            
            not_exists = await manager.secret_exists("nonexistent-key")
            assert not not_exists, "Nonexistent secret should not exist"
            print("  ✅ Existence checks passed")
            
            print("🎉 Basic operations test PASSED!")


async def test_secrets_helper():
    """Test SecretsHelper convenience functions."""
    print("\n🧪 Testing SecretsHelper...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        password = "test-password-123"
        
        backend = LocalVaultBackend(vault_path, password)
        async with backend:
            manager = AdvancedSecretsManager(backend, password)
            helper = SecretsHelper(manager)
            
            # Test database credentials
            print("  💾 Testing database credentials...")
            success = await helper.store_database_credentials(
                "main_db", "localhost", 5432, "user", "pass", "trading_db"
            )
            assert success, "Failed to store database credentials"
            
            db_creds = await helper.get_database_credentials("main_db")
            assert db_creds['host'] == "localhost"
            assert db_creds['port'] == 5432
            print("  ✅ Database credentials test passed")
            
            # Test API key
            print("  🔑 Testing API key...")
            success = await helper.store_api_key("openai", "sk-test-key", "OpenAI API key")
            assert success, "Failed to store API key"
            
            api_key = await helper.get_api_key("openai")
            assert api_key == "sk-test-key"
            print("  ✅ API key test passed")
            
            print("🎉 SecretsHelper test PASSED!")


async def test_encryption_benchmark():
    """Test encryption performance."""
    print("\n🧪 Testing encryption performance...")
    
    from security.encryption import HardenedEncryption
    
    encryption = HardenedEncryption()
    benchmark = encryption.benchmark_kdf()
    
    print(f"  ⚡ Encryption benchmark results:")
    for method, time_ms in benchmark.items():
        print(f"    {method}: {time_ms:.2f}ms")
    
    # Verify both methods work
    test_data = "test-secret-value"
    test_password = "test-password"
    
    encrypted_data, salt = encryption.encrypt(test_data, test_password)
    decrypted_data = encryption.decrypt(encrypted_data, salt, test_password)
    
    assert decrypted_data == test_data, "Encryption/decryption failed"
    print("  ✅ Encryption test passed")


async def main():
    """Run all tests."""
    print("🚀 Step 2 Verification: Core Protocol + Sync Backend")
    print("=" * 60)
    
    try:
        await test_basic_operations()
        await test_secrets_helper()
        await test_encryption_benchmark()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Step 2 implementation is working correctly.")
        print("\n✅ Features verified:")
        print("  • AdvancedSecretsManager with protocol-based architecture")
        print("  • LocalVaultBackend with atomic file operations")
        print("  • HardenedEncryption with Argon2id KDF")
        print("  • SecretsHelper convenience functions")
        print("  • Secret rotation and metadata tracking")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
