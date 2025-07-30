#!/bin/bash
# Secure Docker Setup Script
# This script sets up environment variables from the secure vault for Docker Compose

echo "🔐 Setting up secure Docker environment..."

# Export environment variables from vault
eval "$(python3 scripts/setup_secure_docker_env.py --shell)"

if [ $? -eq 0 ]; then
    echo "✅ Secure environment variables loaded from vault"
    echo "🐳 Ready to run Docker Compose commands"
    echo ""
    echo "Example commands:"
    echo "  docker compose -f docker-compose.timescale.yml up -d primary"
    echo "  docker compose -f docker-compose.timescale.yml up -d replica --profile replica"
    echo "  docker compose -f docker-compose.timescale.yml up -d pgadmin --profile tools"
else
    echo "❌ Failed to load secure environment variables"
    exit 1
fi