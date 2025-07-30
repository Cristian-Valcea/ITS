#!/bin/bash
# Secure Environment Setup Script
# This script sets environment variables from the secure vault

echo "🔐 Loading secure environment variables from vault..."

# Set master password from environment or .env file
if [ -z "$TRADING_VAULT_PASSWORD" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

# Load database passwords from secure vault
eval "$(python3 scripts/setup_secure_docker_env.py --shell 2>/dev/null | grep '^export')"

if [ $? -eq 0 ]; then
    echo "✅ Secure environment variables loaded from vault"
    echo "🔐 TRADING_VAULT_PASSWORD: [HIDDEN]"
    echo "🔐 TIMESCALEDB_PASSWORD: [HIDDEN from vault]" 
    echo "🔐 POSTGRES_PASSWORD: [HIDDEN from vault]"
    echo ""
    echo "Usage: source secure_env.sh"
    echo "All passwords are now loaded from secure vault"
else
    echo "❌ Failed to load passwords from vault"
    echo "⚠️  Falling back to .env file if available"
fi