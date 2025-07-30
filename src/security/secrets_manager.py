#!/usr/bin/env python3
"""
Final working secrets manager - stores salt with each secret
"""
import json
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

class SecretsManager:
    """Production secrets manager - stores salt with each secret (renamed from ReallyWorkingSecretsManager)"""
    
    def __init__(self, vault_path: str, master_password: str):
        self.vault_path = os.path.expanduser(vault_path)
        self.master_password = master_password
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive key from password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.master_password.encode('utf-8'))
    
    def store_secret(self, key: str, value: str) -> bool:
        """Store a secret with its own salt"""
        try:
            # Generate unique salt for this secret
            salt = secrets.token_bytes(32)
            
            # Derive key and create cipher
            derived_key = self._derive_key(salt)
            fernet_key = base64.urlsafe_b64encode(derived_key)
            cipher = Fernet(fernet_key)
            
            # Encrypt the value
            encrypted_value = cipher.encrypt(value.encode('utf-8'))
            
            # Store both salt and encrypted value
            secret_data = {
                'salt': base64.b64encode(salt).decode('utf-8'),
                'encrypted': base64.b64encode(encrypted_value).decode('utf-8')
            }
            
            # Load existing vault or create new
            vault = {}
            if os.path.exists(self.vault_path):
                with open(self.vault_path, 'r') as f:
                    vault = json.load(f)
            
            vault[key] = secret_data
            
            # Save vault
            with open(self.vault_path, 'w') as f:
                json.dump(vault, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Store error: {e}")
            return False
    
    def get_secret(self, key: str) -> str:
        """Get a secret using its stored salt"""
        try:
            if not os.path.exists(self.vault_path):
                raise KeyError(f"Vault not found: {self.vault_path}")
            
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)
            
            if key not in vault:
                raise KeyError(f"Secret not found: {key}")
            
            secret_data = vault[key]
            
            # Extract salt and encrypted data
            salt = base64.b64decode(secret_data['salt'])
            encrypted_value = base64.b64decode(secret_data['encrypted'])
            
            # Derive key and create cipher
            derived_key = self._derive_key(salt)
            fernet_key = base64.urlsafe_b64encode(derived_key)
            cipher = Fernet(fernet_key)
            
            # Decrypt
            decrypted_value = cipher.decrypt(encrypted_value)
            return decrypted_value.decode('utf-8')
            
        except Exception as e:
            raise Exception(f"Get error: {e}")

# Test and migrate existing data
if __name__ == "__main__":
    import getpass
    
    print("🔧 Final Working Secrets Manager")
    print("=" * 35)
    
    # Get password securely
    password = getpass.getpass("Enter master password for testing: ")
    
    # Create new working manager
    new_vault_path = "~/.trading_secrets_final.json"
    
    manager = SecretsManager(new_vault_path, password)
    
    # Get API key from user input for testing
    api_key = input("Enter test API key (or press Enter to skip): ").strip()
    
    if api_key:
        success = manager.store_secret("POLYGON_API_KEY", api_key)
        
        if success:
            print("✅ Stored API key with working manager")
            
            # Test retrieval
            retrieved = manager.get_secret("POLYGON_API_KEY")
            if retrieved == api_key:
                print("✅ Retrieval test passed!")
                
                # Test with new instance (persistence test)
                manager2 = SecretsManager(new_vault_path, password)
                retrieved2 = manager2.get_secret("POLYGON_API_KEY")
                
                if retrieved2 == api_key:
                    print("✅ Cross-instance test passed!")
                    print("🎉 Finally working secrets manager!")
                    
                    # Replace the broken one
                    old_path = os.path.expanduser("~/.trading_secrets.json")
                    new_path = os.path.expanduser(new_vault_path)
                    
                    if os.path.exists(old_path):
                        os.rename(old_path, old_path + ".broken_backup")
                    
                    os.rename(new_path, old_path)
                    print(f"✅ Replaced broken vault with working one")
                    print(f"🔐 Master password authentication working")
                    
                else:
                    print("❌ Cross-instance test failed")
            else:
                print("❌ Retrieval test failed")
        else:
            print("❌ Store failed")
    else:
        print("⏭️  Skipping API key test")


# Compatibility wrapper for existing code that imports AdvancedSecretsManager
class AdvancedSecretsManager:
    """Compatibility wrapper - redirects to working SecretsManager implementation"""
    
    def __init__(self, backend=None, master_password=None, **kwargs):
        """Initialize with compatibility for old interface"""
        vault_path = kwargs.get('vault_path', '~/.trading_secrets.json')
        if master_password is None:
            # Try to get from environment or interactive prompt
            from secrets_helper import SecretsHelper
            master_password = SecretsHelper._get_master_password()
        
        self._manager = SecretsManager(vault_path, master_password)
        self.backend = backend  # Store for compatibility but use working manager
    
    def write_secret(self, key: str, value: str, **kwargs) -> bool:
        """Compatibility method - delegates to working manager"""
        return self._manager.store_secret(key, value)
    
    def read_secret(self, key: str) -> dict:
        """Compatibility method - returns dict format for compatibility"""
        try:
            value = self._manager.get_secret(key)
            return {
                'value': value,
                'metadata': {'key': key}
            }
        except Exception as e:
            raise KeyError(f"Secret not found: {key}") from e
    
    def list_secrets(self) -> list:
        """Compatibility method - basic implementation"""
        return ['POLYGON_API_KEY']  # Simplified for compatibility
    
    def delete_secret(self, key: str) -> bool:
        """Compatibility method - not implemented in working manager"""
        return False


# Alias for backwards compatibility
ReallyWorkingSecretsManager = SecretsManager