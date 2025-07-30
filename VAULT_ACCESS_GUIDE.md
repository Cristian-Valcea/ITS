# 🔐 **VAULT ACCESS GUIDE**

**Date**: July 30, 2025  
**Purpose**: Guide to vault contents and access methods (NO SENSITIVE DATA)

---

## 📊 **VAULT CONTENTS INVENTORY**

### **🔑 Secrets Currently Stored in Vault**

| **Secret Key** | **Type** | **Description** | **Status** |
|----------------|----------|-----------------|------------|
| `POLYGON_API_KEY` | API Key | Polygon.io market data API key | ✅ Active |
| `TIMESCALEDB_PASSWORD` | Database | TimescaleDB/PostgreSQL password | ✅ Active |
| `POSTGRES_PASSWORD` | Database | PostgreSQL password | ✅ Active |

### **🎯 Total Secrets**: 3 active secrets stored securely

---

## 🗂️ **VAULT LOCATION AND STRUCTURE**

### **Physical Vault Location**
```
📁 Vault File: ~/.trading_secrets.json
📍 Full Path: /home/cristian/.trading_secrets.json
🔐 Format: Encrypted JSON with salt-per-secret encryption
🛡️ Protection: Argon2id key derivation + AES-256-GCM encryption
```

---

## 🔑 **MASTER PASSWORD STORAGE LOCATIONS**

### **🏆 Primary Storage: Environment Variable**
```bash
# Environment variable name
export TRADING_VAULT_PASSWORD="[YOUR_SECURE_PASSWORD]"

# Used by: All Python scripts and applications
# Security: Process environment (not persistent)
# Access: os.getenv('TRADING_VAULT_PASSWORD')
```

### **💾 Secondary Storage: .env File** 
```bash
# File: /home/cristian/IntradayTrading/ITS/.env
TRADING_VAULT_PASSWORD=[YOUR_SECURE_PASSWORD]

# Security: File-based storage (git-protected)
# Protection: .gitignore prevents version control exposure
```

### **🔧 Tertiary Storage: System Keyring** (Not Currently Active)
```bash
# Status: Available but not configured due to backend unavailability
# Command to set: keyring set trading_vault master
# Error: "No recommended backend was available"
# Solution: Install keyring backend: pip install keyrings.cryptfile
```

### **💬 Fallback: Interactive Prompt**
```bash
# Used when: All other methods fail
# Behavior: Prompts user for password securely
# Security: No storage, memory-only during session
```

---

## 🛠️ **ACCESS METHODS**

### **1. Direct SecretsHelper Access (Recommended)**
```python
from secrets_helper import SecretsHelper

# Access specific secrets
polygon_key = SecretsHelper.get_polygon_api_key()
timescale_pass = SecretsHelper.get_timescaledb_password()
postgres_pass = SecretsHelper.get_postgres_password()
database_url = SecretsHelper.get_database_url()

# Store new secrets
success = SecretsHelper.store_secret("NEW_API_KEY", "secret_value")
```

### **2. SecretsManager Direct Access**
```python
from src.security.secrets_manager import SecretsManager

# Initialize with vault path and master password
vault_path = "~/.trading_secrets.json"
password = "[SECURE_PASSWORD_FROM_ENV]"
manager = SecretsManager(vault_path, password)

# Access any secret
secret_value = manager.get_secret("SECRET_KEY")
```

### **3. Environment Variable Access (For Docker)**
```bash
# Load vault passwords to environment
eval "$(python3 scripts/setup_secure_docker_env.py --shell)"

# Access via environment
echo $TIMESCALEDB_PASSWORD
echo $POSTGRES_PASSWORD
```

---

## 🔐 **SECURITY HIERARCHY**

### **Master Password Retrieval Order**
```
1️⃣ Environment Variable: TRADING_VAULT_PASSWORD
2️⃣ System Keyring: keyring.get_password("trading_vault", "master")  
3️⃣ .env File: Load from .env file in project root
4️⃣ Interactive Prompt: getpass.getpass() secure input
```

---

## 📚 **USAGE EXAMPLES**

### **Example 1: Access Polygon API Key**
```python
from secrets_helper import SecretsHelper

try:
    api_key = SecretsHelper.get_polygon_api_key()
    print(f"API Key: {api_key[:8]}...{api_key[-8:]}")
except Exception as e:
    print(f"Failed to get API key: {e}")
```

### **Example 2: Database Connection**
```python
from secrets_helper import SecretsHelper

# Get secure database URL
db_url = SecretsHelper.get_database_url()

# Use with psycopg2 or similar
import psycopg2
conn = psycopg2.connect(db_url)
```

### **Example 3: Docker Environment Setup**
```bash
# Load vault passwords for Docker
source scripts/secure_docker_setup.sh

# Then run Docker with secure passwords
docker compose -f docker-compose.timescale.yml up -d primary
```

---

## 🚨 **SECURITY WARNINGS & BEST PRACTICES**

### **⚠️ Master Password Security**
- **Never commit master password to git**: Protected by .gitignore
- **Never document actual password**: Keep passwords out of documentation
- **Environment variable exposure**: Set in shell, not in scripts
- **Recommendation**: Use system keyring when backend available

### **🔒 Vault File Protection**
```bash
# Ensure proper permissions
chmod 600 ~/.trading_secrets.json
```

---

## 🔧 **MAINTENANCE OPERATIONS**

### **Backup Vault**
```bash
# Create encrypted backup
cp ~/.trading_secrets.json ~/.trading_secrets.backup.$(date +%Y%m%d)
```

### **Add New Secret**
```python
from secrets_helper import SecretsHelper

# Store new secret
success = SecretsHelper.store_secret("NEW_API_KEY", "secret_value")
```

---

## 📋 **TROUBLESHOOTING**

### **Common Issues & Solutions**

| **Problem** | **Solution** |
|-------------|--------------|
| Master password not found | Set TRADING_VAULT_PASSWORD environment variable |
| Vault file not found | Initialize vault with first secret storage |
| Permission denied | `chmod 600 ~/.trading_secrets.json` |
| Keyring backend error | `pip install keyrings.cryptfile` |

### **Diagnostic Commands**
```bash
# Check vault file exists and permissions
ls -la ~/.trading_secrets.json

# Test vault access (shows only first 8 chars)
python3 -c "from secrets_helper import SecretsHelper; print(SecretsHelper.get_polygon_api_key()[:8])"

# Run comprehensive security test
python3 test_secure_database_passwords.py
```

---

## 🎯 **SUMMARY**

Your vault is fully operational and secure. All secrets are properly encrypted and access is controlled through multiple security layers. **No sensitive passwords are documented anywhere** - they exist only in the secure vault and environment variables.

---

**🔐 SECURE ACCESS ESTABLISHED - NO SENSITIVE DATA IN DOCUMENTATION**