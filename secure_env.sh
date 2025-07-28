#!/bin/bash
# Secure Environment Setup Script
# This script sets environment variables without exposing them in command history

# Set secure environment variables
export TRADING_VAULT_PASSWORD="PetrosaniPacii16."
export TIMESCALEDB_PASSWORD="secure_trading_password_2025"

echo "✅ Secure environment variables set"
echo "🔐 TRADING_VAULT_PASSWORD: [HIDDEN]"
echo "🔐 TIMESCALEDB_PASSWORD: [HIDDEN]"
echo ""
echo "Usage: source secure_env.sh"
echo "Then run your commands without exposing passwords"