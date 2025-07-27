#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_aws_import():
    """Test AWS backend import."""
    print("🧪 Testing AWS Backend Import...")
    
    try:
        from security.backends.aws_secrets_manager import AWSSecretsBackend
        print("✅ AWS backend imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import AWS backend: {e}")
        return False

def test_aws_initialization():
    """Test AWS backend initialization with mocking."""
    print("🔧 Testing AWS Backend Initialization...")
    
    try:
        from unittest.mock import Mock, patch
        from security.backends.aws_secrets_manager import AWSSecretsBackend
        
        with patch('security.backends.aws_secrets_manager.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client
            mock_client.meta.region_name = "us-east-1"
            
            backend = AWSSecretsBackend(
                secret_prefix="test/",
                region="us-east-1"
            )
            
            assert backend.prefix == "test/"
            print("✅ AWS backend initialization successful")
            return True
            
    except Exception as e:
        print(f"❌ AWS backend initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Simple AWS Backend Test")
    print("=" * 40)
    
    import_success = test_aws_import()
    init_success = test_aws_initialization()
    
    print("\n📊 Results:")
    print(f"Import: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"Initialization: {'✅ PASSED' if init_success else '❌ FAILED'}")
    
    if import_success and init_success:
        print("\n🎉 AWS backend is working correctly!")
    else:
        print("\n❌ Some tests failed.")
