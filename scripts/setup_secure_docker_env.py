#!/usr/bin/env python3
"""
Setup secure environment variables for Docker from vault
This script extracts passwords from the secure vault and exports them as environment variables
that Docker Compose can use.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from secrets_helper import SecretsHelper

def setup_docker_environment():
    """Export database passwords from vault to environment variables"""
    print("🔐 Setting up secure Docker environment from vault...")
    
    try:
        # Get TimescaleDB password from vault
        timescale_password = SecretsHelper.get_timescaledb_password()
        if timescale_password:
            os.environ['TIMESCALE_PASSWORD'] = timescale_password
            print("✅ TIMESCALE_PASSWORD set from vault")
        else:
            print("❌ Could not get TIMESCALE_PASSWORD from vault")
            return False
        
        # Get PostgreSQL password from vault
        postgres_password = SecretsHelper.get_postgres_password()
        if postgres_password:
            os.environ['POSTGRES_PASSWORD'] = postgres_password
            print("✅ POSTGRES_PASSWORD set from vault")
        else:
            print("❌ Could not get POSTGRES_PASSWORD from vault")
            return False
        
        # Also set TimescaleDB password for consistency
        os.environ['TIMESCALEDB_PASSWORD'] = timescale_password
        print("✅ TIMESCALEDB_PASSWORD set from vault")
        
        print("🎯 All Docker environment variables configured securely")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Docker environment: {e}")
        return False

def export_to_shell():
    """Export environment variables in shell format"""
    print("🐚 Generating shell export commands...")
    
    try:
        timescale_password = SecretsHelper.get_timescaledb_password()
        postgres_password = SecretsHelper.get_postgres_password()
        
        print(f"export TIMESCALE_PASSWORD='{timescale_password}'")
        print(f"export POSTGRES_PASSWORD='{postgres_password}'") 
        print(f"export TIMESCALEDB_PASSWORD='{timescale_password}'")
        
        return True
    except Exception as e:
        print(f"❌ Failed to export shell variables: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup secure Docker environment from vault")
    parser.add_argument("--shell", action="store_true", help="Output shell export commands")
    args = parser.parse_args()
    
    if args.shell:
        success = export_to_shell()
    else:
        success = setup_docker_environment()
    
    sys.exit(0 if success else 1)