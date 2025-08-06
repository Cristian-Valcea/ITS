# 🔐 Secrets Vault Guide - IntradayJules Trading System

## Overview

The IntradayJules trading system uses a secure, encrypted vault to store all sensitive credentials and API keys. This document explains how to use the secrets management system without exposing any actual passwords.

## Architecture

- **Vault Location**: `~/.trading_secrets.json` (encrypted file)
- **Encryption**: AES-256-GCM with Argon2id key derivation
- **Master Password**: Stored securely via multiple fallback methods
- **Per-Secret Salting**: Each secret has its own unique salt for maximum security

## Stored Secrets

The vault currently contains these credential types:

| Secret Key | Purpose | Usage |
|------------|---------|-------|
| `POLYGON_API_KEY` | Market data API access | Historical and real-time price data |
| `TIMESCALEDB_PASSWORD` | Database authentication | TimescaleDB/PostgreSQL connection |
| `POSTGRES_PASSWORD` | Database access | Alternative database connection |

## Usage Examples

### 1. Basic Secret Retrieval

```python
from secrets_helper import SecretsHelper

# Get API key for market data
api_key = SecretsHelper.get_polygon_api_key()

# Get database password
db_password = SecretsHelper.get_timescaledb_password()

# Get PostgreSQL password
postgres_password = SecretsHelper.get_postgres_password()
```

### 2. Database Connection with Vault

```python
import psycopg2
from secrets_helper import SecretsHelper

# Secure database connection
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='trading_data',
    user='postgres',
    password=SecretsHelper.get_timescaledb_password()
)
```

### 3. API Client with Vault

```python
import requests
from secrets_helper import SecretsHelper

# Secure API request
api_key = SecretsHelper.get_polygon_api_key()
response = requests.get(
    f"https://api.polygon.io/v2/aggs/ticker/NVDA/range/1/day/2023-01-01/2023-12-31",
    params={'apikey': api_key}
)
```

### 4. Docker Environment with Vault

```python
import subprocess
from secrets_helper import SecretsHelper

# Start container with secure password from vault
secure_password = SecretsHelper.get_timescaledb_password()
cmd = [
    'docker', 'run', '-d', '--name', 'timescaledb',
    '-e', f'POSTGRES_PASSWORD={secure_password}',
    'timescale/timescaledb:latest-pg14'
]
subprocess.run(cmd)
```

## Master Password Management

The vault master password is retrieved through a secure hierarchy:

### 1. Environment Variable (Highest Priority)
```bash
export TRADING_VAULT_PASSWORD="your_master_password"
```

### 2. System Keyring (Recommended for Workstations)
```python
# Automatically used if available
import keyring
# Password stored in OS keyring service
```

### 3. Environment File (Development)
```bash
# In .env file
TRADING_VAULT_PASSWORD=your_master_password
```

### 4. Interactive Prompt (Fallback)
```python
# Automatically prompts if other methods fail
# Enter vault master password: [hidden input]
```

## Administrative Operations

### Adding New Secrets

```python
from secrets_helper import SecretsHelper
from src.security.secrets_manager import SecretsManager

# Get master password (uses secure hierarchy)
master_password = SecretsHelper._get_master_password()

# Initialize secrets manager
vault_path = '~/.trading_secrets.json'
secrets_manager = SecretsManager(vault_path, master_password)

# Store new secret
success = secrets_manager.store_secret('NEW_SECRET_KEY', 'secret_value')
if success:
    print("✅ Secret stored successfully")
```

### Updating Existing Secrets

```python
# Same process as adding - will overwrite existing
secrets_manager.store_secret('EXISTING_KEY', 'new_value')
```

### Setup Master Password (One-time)

```python
from secrets_helper import SecretsHelper

# Interactive setup wizard
SecretsHelper.setup_master_password()
```

## Security Features

### 1. **Per-Secret Encryption**
- Each secret has its unique salt
- No single point of failure
- Rainbow table resistant

### 2. **Key Derivation**
- PBKDF2-HMAC with SHA-256
- 100,000 iterations
- 32-byte salt per secret

### 3. **Multiple Storage Options**
- System keyring integration
- Environment variable support
- Secure file fallback

### 4. **No Hardcoded Passwords**
- All passwords retrieved from vault
- No plaintext storage in code
- Automatic secure fallback hierarchy

## Best Practices

### ✅ Do's
- Always use `SecretsHelper` for credential retrieval
- Store master password in system keyring for workstations
- Use environment variables for CI/CD pipelines
- Keep vault file (`~/.trading_secrets.json`) backed up securely

### ❌ Don'ts
- Never hardcode passwords in source code
- Don't commit the vault file to version control
- Don't share the master password via insecure channels
- Don't bypass the vault system for "convenience"

## Troubleshooting

### Common Issues

**"Password authentication failed"**
```python
# Verify vault access
try:
    password = SecretsHelper.get_timescaledb_password()
    print("✅ Vault access successful")
except Exception as e:
    print(f"❌ Vault error: {e}")
```

**"Keyring not available"**
```bash
# Install keyring support
pip install keyring
```

**"Master password prompt every time"**
```python
# Set up persistent storage
SecretsHelper.setup_master_password()
```

## Environment Integration

### Development Setup
```bash
# Create .env file (not committed to git)
echo "TRADING_VAULT_PASSWORD=your_master_password" > .env
```

### Production Setup
```bash
# Use environment variable
export TRADING_VAULT_PASSWORD="secure_master_password"
```

### CI/CD Pipeline
```yaml
# GitHub Actions example
env:
  TRADING_VAULT_PASSWORD: ${{ secrets.VAULT_MASTER_PASSWORD }}
```

## Security Audit Trail

The vault system logs access attempts:
- ✅ Successful retrievals
- ⚠️ Fallback method usage
- ❌ Failed access attempts

Monitor logs for:
```
🔑 Using password from environment variable
🔑 Using password from system keyring  
🔑 Using password from .env file
⚠️  Keyring access failed
```

## Backup and Recovery

### Backup Vault
```bash
# Secure backup of encrypted vault
cp ~/.trading_secrets.json ~/.trading_secrets.backup.json
```

### Recovery Process
1. Restore vault file: `~/.trading_secrets.json`
2. Ensure master password is available via one of the secure methods
3. Test access: `python -c "from secrets_helper import SecretsHelper; print(SecretsHelper.get_polygon_api_key())"`

## Integration Examples

### Training Scripts
```python
#!/usr/bin/env python3
from secrets_helper import SecretsHelper

# All database connections automatically secure
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'trading_data',
    'user': 'postgres',
    'password': SecretsHelper.get_timescaledb_password()
}
```

### Data Fetchers
```python
class DataFetcher:
    def __init__(self):
        # Secure API key retrieval
        self.api_key = SecretsHelper.get_polygon_api_key()
        
        # Secure database config
        self.db_config = {
            'password': SecretsHelper.get_timescaledb_password()
        }
```

---

**Note**: This guide contains no actual passwords or sensitive information. All examples show the secure usage patterns for the vault system. The actual credentials are safely encrypted in the vault and retrieved only when needed.