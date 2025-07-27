#!/usr/bin/env python3
"""
Phase 4 Comprehensive Test Suite
Tests all Phase 4 security infrastructure components as mentioned in the implementation document.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_imports():
    """Test that all security components can be imported."""
    print("1. Testing imports...")
    
    try:
        # Test main security imports
        from src.security import get_database_config, get_its_secret, ITSSecretsHelper
        from src.security.advanced_secrets_manager import AdvancedSecretsManager
        from src.security.encryption import HardenedEncryption
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

async def test_its_secrets_helper():
    """Test ITSSecretsHelper functionality."""
    print("2. Testing ITSSecretsHelper...")
    
    try:
        from src.security import get_database_config, get_alert_config
        
        # Test database config
        db_config = get_database_config()
        required_keys = ['host', 'port', 'database', 'user', 'password']
        
        if all(key in db_config for key in required_keys):
            print(f"   ✅ Database config: {list(db_config.keys())}")
        else:
            print(f"   ❌ Missing database config keys")
            return False
            
        # Test alert config
        alert_config = get_alert_config()
        alert_keys = ['pagerduty_key', 'slack_webhook', 'slack_channel']
        
        if all(key in alert_config for key in alert_keys):
            print(f"   ✅ Alert config: {list(alert_config.keys())}")
        else:
            print(f"   ❌ Missing alert config keys")
            return False
            
        return True
    except Exception as e:
        print(f"   ❌ ITSSecretsHelper test failed: {e}")
        return False

async def test_convenience_functions():
    """Test convenience functions."""
    print("3. Testing convenience functions...")
    
    try:
        from src.security import get_database_config, get_alert_config
        
        db_config = get_database_config()
        alert_config = get_alert_config()
        
        print(f"   ✅ Convenience functions work: db={len(db_config)} keys, alerts={len(alert_config)} keys")
        return True
    except Exception as e:
        print(f"   ❌ Convenience functions failed: {e}")
        return False

async def test_advanced_secrets_manager():
    """Test AdvancedSecretsManager."""
    print("4. Testing AdvancedSecretsManager...")
    
    try:
        import tempfile
        from pathlib import Path
        from src.security.backends.local_vault import LocalVaultBackend
        from src.security.advanced_secrets_manager import AdvancedSecretsManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)
            password = "test-password-123"
            
            backend = LocalVaultBackend(vault_path, password)
            async with backend:
                manager = AdvancedSecretsManager(backend, password)
                
                # Test basic functionality
                secrets = await manager.list_secrets()
                print(f"   ✅ Manager works: {len(secrets)} secrets found")
                return True
    except Exception as e:
        print(f"   ❌ AdvancedSecretsManager test failed: {e}")
        return False

async def test_encryption():
    """Test encryption functionality."""
    print("5. Testing encryption...")
    
    try:
        from src.security.encryption import HardenedEncryption
        
        encryption = HardenedEncryption()
        password = "test-password-123"
        plaintext = "Hello, World!"
        
        # Test encryption/decryption
        encrypted_data, salt = encryption.encrypt(plaintext, password)
        decrypted_text = encryption.decrypt(encrypted_data, salt, password)
        
        if decrypted_text == plaintext:
            print(f'   ✅ Encryption works: "{plaintext}" -> encrypted -> "{decrypted_text}"')
            return True
        else:
            print(f"   ❌ Encryption failed: expected '{plaintext}', got '{decrypted_text}'")
            return False
    except Exception as e:
        print(f"   ❌ Encryption test failed: {e}")
        return False

async def main():
    """Run all Phase 4 tests."""
    print("=== FINAL COMPREHENSIVE TEST ===")
    print()
    
    tests = [
        test_imports,
        test_its_secrets_helper,
        test_convenience_functions,
        test_advanced_secrets_manager,
        test_encryption
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉🎉🎉 ALL TESTS PASSED! 🎉🎉🎉")
        return True
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)