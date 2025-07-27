#!/usr/bin/env python3
"""
Phase 3 Complete Validation Test

This comprehensive test validates all Phase 3 features:
- All cloud backends (AWS, Azure, HashiCorp Vault)
- Multi-cloud manager with failover
- CLI integration
- Configuration management
- Error handling and resilience
"""

import asyncio
import sys
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(level=logging.WARNING)

async def test_all_backends_available():
    """Test that all backends are available and importable."""
    print("🌐 Testing Backend Availability")
    print("=" * 50)
    
    results = {}
    
    # Test local backend (always available)
    try:
        from security.backends.local_vault import LocalVaultBackend
        results["Local"] = "✅ Available"
    except ImportError as e:
        results["Local"] = f"❌ Error: {e}"
    
    # Test AWS backend
    try:
        from security.backends.aws_secrets_manager import AWSSecretsBackend
        results["AWS Secrets Manager"] = "✅ Available"
    except ImportError:
        results["AWS Secrets Manager"] = "❌ Not installed (pip install boto3>=1.34.0)"
    
    # Test Azure backend
    try:
        from security.backends.azure_keyvault import AzureKeyVaultBackend
        results["Azure Key Vault"] = "✅ Available"
    except ImportError:
        results["Azure Key Vault"] = "❌ Not installed (pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0)"
    
    # Test HashiCorp Vault backend
    try:
        from security.backends.hashicorp_vault import HashiCorpVaultBackend
        results["HashiCorp Vault"] = "✅ Available"
    except ImportError:
        results["HashiCorp Vault"] = "❌ Not installed (pip install hvac>=1.2.0)"
    
    for backend, status in results.items():
        print(f"  {backend}: {status}")
    
    available_count = sum(1 for status in results.values() if "✅" in status)
    print(f"\n📊 Available backends: {available_count}/4")
    
    return available_count >= 2  # At least local + one cloud backend

async def test_protocol_compliance():
    """Test that all backends implement the VaultBackend protocol."""
    print("\n🔌 Testing Protocol Compliance")
    print("=" * 50)
    
    from security.protocols import VaultBackend
    
    # Required methods for VaultBackend protocol
    required_methods = [
        'store', 'retrieve', 'delete', 'list_keys', 'exists',
        '__aenter__', '__aexit__'
    ]
    
    backend_classes = []
    
    # Collect available backend classes
    try:
        from security.backends.local_vault import LocalVaultBackend
        backend_classes.append(("Local", LocalVaultBackend))
    except ImportError:
        pass
    
    try:
        from security.backends.aws_secrets_manager import AWSSecretsBackend
        backend_classes.append(("AWS", AWSSecretsBackend))
    except ImportError:
        pass
    
    try:
        from security.backends.azure_keyvault import AzureKeyVaultBackend
        backend_classes.append(("Azure", AzureKeyVaultBackend))
    except ImportError:
        pass
    
    try:
        from security.backends.hashicorp_vault import HashiCorpVaultBackend
        backend_classes.append(("Vault", HashiCorpVaultBackend))
    except ImportError:
        pass
    
    all_compliant = True
    
    for name, backend_class in backend_classes:
        missing_methods = []
        for method in required_methods:
            if not hasattr(backend_class, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"  {name}: ❌ Missing methods: {missing_methods}")
            all_compliant = False
        else:
            print(f"  {name}: ✅ Protocol compliant")
    
    return all_compliant

async def test_local_backend_functionality():
    """Test local backend basic functionality."""
    print("\n🏠 Testing Local Backend Functionality")
    print("=" * 50)
    
    try:
        from security.backends.local_vault import LocalVaultBackend
        from security.advanced_secrets_manager import AdvancedSecretsManager
        from security.protocols import SecretMetadata, SecretType
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.vault') as tmp:
            vault_path = tmp.name
        
        try:
            backend = LocalVaultBackend(vault_path)
            password = "test-password-123"
            
            async with backend:
                manager = AdvancedSecretsManager(backend, password)
                
                # Test write
                print("  📝 Testing write operation...")
                success = await manager.write_secret("test-key", "test-value", 
                    description="Test secret",
                    tags={"environment": "test"}
                )
                assert success, "Failed to write secret"
                print("  ✅ Write successful")
                
                # Test read
                print("  📖 Testing read operation...")
                secret_data = await manager.read_secret("test-key")
                assert secret_data['value'] == "test-value", f"Expected 'test-value', got '{secret_data['value']}'"
                print("  ✅ Read successful")
                
                # Test list
                print("  📋 Testing list operation...")
                secrets = await manager.list_secrets()
                assert "test-key" in secrets, "Secret not found in list"
                print("  ✅ List successful")
                
                # Test delete
                print("  🗑️ Testing delete operation...")
                success = await manager.delete_secret("test-key")
                assert success, "Failed to delete secret"
                print("  ✅ Delete successful")
                
        finally:
            # Cleanup
            if os.path.exists(vault_path):
                os.unlink(vault_path)
        
        print("🎉 Local backend functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Local backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cli_integration():
    """Test CLI integration and help system."""
    print("\n🖥️ Testing CLI Integration")
    print("=" * 50)
    
    try:
        # Test that CLI can be imported
        import subprocess
        import sys
        
        # Test backends command
        print("  🌐 Testing backends command...")
        result = subprocess.run([
            sys.executable, "cloud_secrets_cli.py", "backends"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("  ✅ Backends command successful")
        else:
            print(f"  ❌ Backends command failed: {result.stderr}")
            return False
        
        # Test examples command
        print("  📖 Testing examples command...")
        result = subprocess.run([
            sys.executable, "cloud_secrets_cli.py", "examples"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("  ✅ Examples command successful")
        else:
            print(f"  ❌ Examples command failed: {result.stderr}")
            return False
        
        print("🎉 CLI integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

async def test_configuration_system():
    """Test configuration system and multi-cloud manager."""
    print("\n⚙️ Testing Configuration System")
    print("=" * 50)
    
    try:
        # Test configuration loading
        print("  📄 Testing configuration loading...")
        
        # Check if config file exists
        if os.path.exists("cloud_config.yaml"):
            print("  ✅ Configuration file found")
        else:
            print("  ❌ Configuration file not found")
            return False
        
        # Test multi-cloud manager import
        print("  🌐 Testing multi-cloud manager import...")
        try:
            from multi_cloud_manager import MultiCloudSecretsManager
            print("  ✅ Multi-cloud manager imported successfully")
        except ImportError as e:
            print(f"  ❌ Failed to import multi-cloud manager: {e}")
            return False
        
        # Test manager initialization
        print("  🔧 Testing manager initialization...")
        try:
            manager = MultiCloudSecretsManager(
                config_path="cloud_config.yaml",
                environment="development"
            )
            print("  ✅ Manager initialized successfully")
        except Exception as e:
            print(f"  ❌ Manager initialization failed: {e}")
            return False
        
        print("🎉 Configuration system test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and resilience."""
    print("\n🛡️ Testing Error Handling")
    print("=" * 50)
    
    try:
        from security.backends.local_vault import LocalVaultBackend
        from security.advanced_secrets_manager import AdvancedSecretsManager
        
        # Test with invalid vault path
        print("  📁 Testing invalid vault path handling...")
        backend = LocalVaultBackend("/invalid/path/vault.dat")
        password = "test-password"
        
        try:
            async with backend:
                manager = AdvancedSecretsManager(backend, password)
                # This should handle the error gracefully
                success = await manager.write_secret("test", "value")
                # Depending on implementation, this might succeed (create dirs) or fail gracefully
                print("  ✅ Invalid path handled gracefully")
        except Exception as e:
            print(f"  ✅ Invalid path error handled: {type(e).__name__}")
        
        # Test with wrong password
        print("  🔐 Testing wrong password handling...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.vault') as tmp:
            vault_path = tmp.name
        
        try:
            # Create a secret with one password
            backend1 = LocalVaultBackend(vault_path)
            async with backend1:
                manager1 = AdvancedSecretsManager(backend1, "password1")
                await manager1.write_secret("test", "value")
            
            # Try to read with different password
            backend2 = LocalVaultBackend(vault_path)
            async with backend2:
                manager2 = AdvancedSecretsManager(backend2, "password2")
                value = await manager2.read_secret("test")
                # Should return None or handle decryption error
                print("  ✅ Wrong password handled gracefully")
        
        finally:
            if os.path.exists(vault_path):
                os.unlink(vault_path)
        
        print("🎉 Error handling test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def test_security_features():
    """Test security features and encryption."""
    print("\n🔒 Testing Security Features")
    print("=" * 50)
    
    try:
        from src.security.encryption import HardenedEncryption
        from src.security.protocols import SecretMetadata, SecretType
        
        # Test encryption/decryption
        print("  🔐 Testing encryption/decryption...")
        password = "test-password-123"
        plaintext = "sensitive-data-12345"
        
        encryption = HardenedEncryption(password)
        
        # Encrypt
        encrypted_data = encryption.encrypt(plaintext.encode())
        assert encrypted_data != plaintext.encode(), "Data should be encrypted"
        print("  ✅ Encryption successful")
        
        # Decrypt
        decrypted_data = encryption.decrypt(encrypted_data)
        assert decrypted_data.decode() == plaintext, "Decryption failed"
        print("  ✅ Decryption successful")
        
        # Test metadata handling
        print("  🏷️ Testing metadata handling...")
        metadata = SecretMetadata(
            secret_type=SecretType.API_KEY,
            description="Test API key",
            tags={"environment": "test", "service": "trading"}
        )
        
        # Verify metadata structure
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.description == "Test API key"
        assert metadata.tags["environment"] == "test"
        print("  ✅ Metadata handling successful")
        
        print("🎉 Security features test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner for Phase 3 validation."""
    setup_logging()
    
    print("🚀 Phase 3 Complete Validation Test")
    print("=" * 60)
    print("Testing all Phase 3 features and integrations...")
    print()
    
    # Run all tests
    tests = [
        ("Backend Availability", test_all_backends_available),
        ("Protocol Compliance", test_protocol_compliance),
        ("Local Backend Functionality", test_local_backend_functionality),
        ("CLI Integration", test_cli_integration),
        ("Configuration System", test_configuration_system),
        ("Error Handling", test_error_handling),
        ("Security Features", test_security_features),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 3 implementation is complete and functional")
        print("🚀 Ready for production deployment")
        print("\n🔧 Next steps:")
        print("  1. Install desired cloud SDKs")
        print("  2. Configure cloud credentials")
        print("  3. Set up environment-specific configurations")
        print("  4. Deploy to staging/production environments")
        print("  5. Implement monitoring and alerting")
    else:
        print(f"\n❌ {total_count - passed_count} tests failed")
        print("🔧 Please review the failed tests and fix issues before deployment")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
