#!/usr/bin/env python3
"""
Simple, working secrets helper that actually works
"""
import json
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SimpleSecretsManager:
    """Actually working secrets manager"""
    
    def __init__(self, vault_path: str, master_password: str):
        self.vault_path = vault_path
        self.master_password = master_password
        
        # Use a fixed salt derived from the vault path for consistency
        self.salt = hashes.Hash(hashes.SHA256())
        self.salt.update(vault_path.encode('utf-8'))
        self.salt = self.salt.finalize()[:32]  # Use first 32 bytes as salt
        
        # Derive key consistently
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = kdf.derive(master_password.encode('utf-8'))
        self.cipher = Fernet(base64.urlsafe_b64encode(key))
    
    def store_secret(self, key: str, value: str) -> bool:
        """Store a secret"""
        try:
            # Load existing vault or create new
            vault = {}
            if os.path.exists(self.vault_path):
                with open(self.vault_path, 'r') as f:
                    vault = json.load(f)
            
            # Encrypt and store
            encrypted_value = self.cipher.encrypt(value.encode('utf-8'))
            vault[key] = base64.b64encode(encrypted_value).decode('utf-8')
            
            # Save vault
            with open(self.vault_path, 'w') as f:
                json.dump(vault, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Store error: {e}")
            return False
    
    def get_secret(self, key: str) -> str:
        """Get a secret"""
        try:
            if not os.path.exists(self.vault_path):
                raise KeyError(f"Vault not found: {self.vault_path}")
            
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)
            
            if key not in vault:
                raise KeyError(f"Secret not found: {key}")
            
            encrypted_value = base64.b64decode(vault[key])
            decrypted_value = self.cipher.decrypt(encrypted_value)
            return decrypted_value.decode('utf-8')
            
        except Exception as e:
            print(f"Get error: {e}")
            raise

# Quick usage example
if __name__ == "__main__":
    import os
    vault_path = os.path.expanduser("~/.simple_secrets.json")
    manager = SimpleSecretsManager(vault_path, "test123")
    
    # Store secret
    success = manager.store_secret("POLYGON_API_KEY", "2xNzHuDZIJ_fdQSctq159_YDJYsfoFQ_")
    print(f"Store success: {success}")
    
    # Retrieve secret
    retrieved = manager.get_secret("POLYGON_API_KEY")
    print(f"Retrieved: {retrieved}")
