#!/usr/bin/env python3
"""
Enhanced secrets helper with secure password management
Supports: System Keyring, Environment File (.env), Environment Variable, Interactive Prompt
"""
import os
import sys
import getpass
from pathlib import Path
sys.path.append('.')

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    print("⚠️  keyring not available - install with: pip install keyring")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not available - install with: pip install python-dotenv")

from src.security.secrets_manager import SecretsManager

class SecretsHelper:
    """Enhanced secrets helper with secure password management"""
    
    @staticmethod
    def _get_master_password():
        """Get master password with secure fallback hierarchy"""
        
        # 1. Try environment variable (for automation/CI)
        password = os.getenv('TRADING_VAULT_PASSWORD')
        if password:
            print("🔑 Using password from environment variable")
            return password
        
        # 2. Try system keyring (for workstation security)
        if KEYRING_AVAILABLE:
            try:
                password = keyring.get_password("trading_vault", "master")
                if password:
                    print("🔑 Using password from system keyring")
                    return password
            except Exception as e:
                print(f"⚠️  Keyring access failed: {e}")
        
        # 3. Try .env file (for development)
        env_file = Path(".env")
        if env_file.exists() and DOTENV_AVAILABLE:
            try:
                load_dotenv()
                password = os.getenv('TRADING_VAULT_PASSWORD')
                if password:
                    print("🔑 Using password from .env file")
                    return password
            except Exception as e:
                print(f"⚠️  .env file read failed: {e}")
        
        # 4. Interactive prompt (fallback)
        print("🔑 Using interactive password prompt")
        return getpass.getpass("Enter vault master password: ")
    
    @staticmethod
    def setup_master_password():
        """One-time setup to store password securely"""
        print("🔐 Master Password Setup")
        print("=" * 30)
        print("Choose secure storage method:")
        print("1. System Keyring (Recommended - most secure)")
        print("2. Environment File (.env - good for development)")
        print("3. Environment Variable (manual setup)")
        print("4. Test current password retrieval")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "4":
            # Test current setup
            try:
                password = SecretsHelper._get_master_password()
                print("✅ Password retrieval successful!")
                return True
            except Exception as e:
                print(f"❌ Password retrieval failed: {e}")
                return False
        
        password = getpass.getpass("\nEnter master password to store: ")
        confirm = getpass.getpass("Confirm master password: ")
        
        if password != confirm:
            print("❌ Passwords don't match!")
            return False
        
        success = False
        
        if choice == "1":
            if KEYRING_AVAILABLE:
                try:
                    keyring.set_password("trading_vault", "master", password)
                    print("✅ Password stored in system keyring")
                    success = True
                except Exception as e:
                    print(f"❌ Keyring storage failed: {e}")
            else:
                print("❌ Keyring not available - install with: pip install keyring")
        
        elif choice == "2":
            try:
                with open(".env", "w") as f:
                    f.write(f"TRADING_VAULT_PASSWORD={password}\n")
                print("✅ Password stored in .env file")
                print("⚠️  Important: Add .env to .gitignore!")
                
                # Check if .gitignore exists and update it
                gitignore_path = Path(".gitignore")
                if gitignore_path.exists():
                    with open(gitignore_path, "r") as f:
                        content = f.read()
                    if ".env" not in content:
                        with open(gitignore_path, "a") as f:
                            f.write("\n# Environment variables\n.env\n")
                        print("✅ Added .env to .gitignore")
                else:
                    with open(gitignore_path, "w") as f:
                        f.write("# Environment variables\n.env\n")
                    print("✅ Created .gitignore with .env")
                
                success = True
            except Exception as e:
                print(f"❌ .env file creation failed: {e}")
        
        elif choice == "3":
            print(f"\n📋 Set this environment variable in your shell:")
            print(f"export TRADING_VAULT_PASSWORD='{password}'")
            print(f"\nOr add to your ~/.bashrc or ~/.zshrc:")
            print(f"echo 'export TRADING_VAULT_PASSWORD=\"{password}\"' >> ~/.bashrc")
            success = True
        
        # Test the setup
        if success:
            print("\n🧪 Testing password retrieval...")
            try:
                test_password = SecretsHelper._get_master_password()
                if test_password == password:
                    print("✅ Setup successful! Password retrieval working.")
                    return True
                else:
                    print("❌ Setup verification failed - retrieved password doesn't match")
                    return False
            except Exception as e:
                print(f"❌ Setup verification failed: {e}")
                return False
        
        return success
    
    @staticmethod
    def get_polygon_api_key():
        """Get Polygon API key"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = SecretsManager(vault_path, password)
        return manager.get_secret("POLYGON_API_KEY")
    
    @staticmethod
    def store_secret(key: str, value: str) -> bool:
        """Store a new secret"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = SecretsManager(vault_path, password)
        return manager.store_secret(key, value)
    
    @staticmethod
    def get_timescaledb_password():
        """Get TimescaleDB password from secure vault"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = SecretsManager(vault_path, password)
        return manager.get_secret("TIMESCALEDB_PASSWORD")
    
    @staticmethod
    def get_postgres_password():
        """Get PostgreSQL password from secure vault"""
        vault_path = "~/.trading_secrets.json"
        password = SecretsHelper._get_master_password()
        
        manager = SecretsManager(vault_path, password)
        return manager.get_secret("POSTGRES_PASSWORD")
    
    @staticmethod
    def get_database_url(host="localhost", port="5432", database="trading_data", user="postgres"):
        """Get complete database URL with secure password"""
        password = SecretsHelper.get_timescaledb_password()
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Quick test
if __name__ == "__main__":
    print("🔑 Testing Secrets Helper")
    api_key = SecretsHelper.get_polygon_api_key()
    print(f"✅ Polygon API Key: {api_key[:8]}...{api_key[-8:]}")