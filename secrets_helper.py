#!/usr/bin/env python3
"""
Simple helper to access secrets in your scripts
"""
import os
import sys
import getpass
sys.path.append('.')

from final_working_secrets import ReallyWorkingSecretsManager

class SecretsHelper:
    """Easy access to secrets for your trading scripts"""
    
    @staticmethod
    def _get_master_password():
        """Get master password securely"""
        # Try environment variable first
        password = os.getenv('TRADING_VAULT_PASSWORD')
        if password:
            return password
        
        # Fall back to interactive prompt
        return getpass.getpass("Enter vault master password: ")
    
    @staticmethod
    def get_polygon_api_key():
        """Get Polygon API key"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = ReallyWorkingSecretsManager(vault_path, password)
        return manager.get_secret("POLYGON_API_KEY")
    
    @staticmethod
    def store_secret(key: str, value: str) -> bool:
        """Store a new secret"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = ReallyWorkingSecretsManager(vault_path, password)
        return manager.store_secret(key, value)

# Quick test
if __name__ == "__main__":
    print("🔑 Testing Secrets Helper")
    api_key = SecretsHelper.get_polygon_api_key()
    print(f"✅ Polygon API Key: {api_key[:8]}...{api_key[-8:]}")