#!/bin/bash
# Quick dependency installer for IntradayJules
# Installs commonly missing packages

echo "🔧 Installing missing dependencies..."

cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# Install psycopg2 for TimescaleDB
pip install psycopg2-binary

echo "✅ Dependencies installed successfully!"
echo "🚀 Ready to resume training!"