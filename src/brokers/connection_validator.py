#!/usr/bin/env python3
"""
🔒 IBKR Connection Validator
Hard-fails on missing/invalid credentials - no silent simulation fallback
"""

import os
import logging
from typing import Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ConnectionMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"
    EXPLICIT_SIM = "explicit_simulation"  # User explicitly requested simulation

class ConnectionConfig:
    """Validated IBKR connection configuration"""
    
    def __init__(self, mode: ConnectionMode, host: str, port: int, client_id: int, 
                 username: Optional[str] = None, password: Optional[str] = None):
        self.mode = mode
        self.host = host
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password

class IBKRConnectionValidator:
    """
    🔒 Validates IBKR connection parameters with hard failure on credential issues
    
    FIXES SECURITY ISSUE: No silent fallback to simulation mode
    """
    
    @staticmethod
    def validate_connection_config(force_simulation: bool = False) -> ConnectionConfig:
        """
        🚨 SECURE: Validate connection config with hard failure on missing credentials
        
        Args:
            force_simulation: Explicitly request simulation mode
            
        Returns:
            ConnectionConfig with validated parameters
            
        Raises:
            ValueError: On missing/invalid credentials for live mode
            RuntimeError: On ib_insync availability issues
        """
        
        # Check ib_insync availability
        try:
            import ib_insync
            ib_available = True
        except ImportError:
            ib_available = False
            logger.error("ib_insync not available - cannot connect to IBKR")
        
        # Get connection parameters
        host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
        port = int(os.getenv('IBKR_PORT', '7497'))
        client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
        username = os.getenv('IB_USERNAME')
        password = os.getenv('IB_PASSWORD')
        
        # Determine connection mode
        if force_simulation:
            if not ib_available:
                raise RuntimeError("ib_insync not available - cannot run even simulation mode")
            
            logger.info("🎭 EXPLICIT SIMULATION MODE requested")
            return ConnectionConfig(
                mode=ConnectionMode.EXPLICIT_SIM,
                host=host, port=port, client_id=client_id,
                username=username, password=password
            )
        
        # For live mode, validate everything strictly
        if not ib_available:
            raise RuntimeError(
                "ib_insync not available - cannot connect to IBKR. "
                "Install with: pip install ib_insync"
            )
        
        # 🚨 HARD VALIDATION: No silent fallback to simulation
        if not username or not password:
            raise ValueError(
                "🚨 MISSING IBKR CREDENTIALS for live mode!\n"
                "Required environment variables:\n"
                "  - IB_USERNAME: Your IBKR username\n"
                "  - IB_PASSWORD: Your IBKR password\n"
                "If you want simulation mode, call with force_simulation=True"
            )
        
        if not host or port <= 0:
            raise ValueError(
                f"🚨 INVALID CONNECTION PARAMETERS!\n"
                f"  - IBKR_HOST_IP: {host}\n"
                f"  - IBKR_PORT: {port}\n"
                f"Check your .env configuration"
            )
        
        logger.info(f"✅ LIVE MODE validated: {host}:{port} (client_id={client_id})")
        return ConnectionConfig(
            mode=ConnectionMode.LIVE,
            host=host, port=port, client_id=client_id,
            username=username, password=password
        )
    
    @staticmethod
    def validate_paper_trading_config() -> ConnectionConfig:
        """
        🎯 Validate paper trading configuration (less strict than live)
        """
        try:
            import ib_insync
        except ImportError:
            raise RuntimeError("ib_insync not available for paper trading")
        
        host = os.getenv('IBKR_HOST_IP', '172.24.32.1')
        port = int(os.getenv('IBKR_PORT', '7497'))  # Paper trading port
        client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
        
        # Paper trading is more forgiving on credentials
        username = os.getenv('IB_USERNAME')
        password = os.getenv('IB_PASSWORD')
        
        if port not in [7497, 4002]:  # Paper trading ports
            logger.warning(f"⚠️  Port {port} may not be paper trading port (expected 7497 or 4002)")
        
        logger.info(f"✅ PAPER TRADING validated: {host}:{port}")
        return ConnectionConfig(
            mode=ConnectionMode.SIMULATION,
            host=host, port=port, client_id=client_id,
            username=username, password=password
        )