# src/security/its_integration.py
"""
ITS-specific integration helpers for seamless secrets management.
Provides convenience functions and configuration helpers for the IntradayJules Trading System.
"""

import os
import logging
from typing import Dict, Any, Optional
from .advanced_secrets_manager import AdvancedSecretsManager
from .backends.local_vault import LocalVaultBackend

logger = logging.getLogger(__name__)


class ITSSecretsHelper:
    """
    Helper class for ITS-specific secret management.
    Provides convenient methods for common ITS configurations.
    """
    
    def __init__(self, master_password: Optional[str] = None):
        """Initialize ITS secrets helper."""
        self.master_password = master_password or os.getenv('ITS_MASTER_PASSWORD', 'default_dev_password')
        self._manager = None
    
    @property
    def manager(self) -> AdvancedSecretsManager:
        """Lazy-load the secrets manager."""
        if self._manager is None:
            vault_path = os.getenv('ITS_VAULT_PATH', 'its_secrets.vault')
            backend = LocalVaultBackend(vault_path=vault_path, password=self.master_password)
            self._manager = AdvancedSecretsManager(backend=backend, master_password=self.master_password)
        return self._manager
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by key with fallback to environment variables.
        
        Args:
            key: Secret key to retrieve
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                secret_data = loop.run_until_complete(self.manager.read_secret(key))
                if secret_data:
                    return secret_data.value
            except Exception as e:
                logger.debug(f"Failed to get secret {key}: {e}")
            finally:
                loop.close()
        except Exception as e:
            logger.debug(f"Failed to get secret {key}: {e}")
        
        # Fallback to environment variable
        env_value = os.getenv(key.upper())
        if env_value:
            return env_value
            
        return default
    
    def get_database_config(self) -> Dict[str, str]:
        """
        Get database configuration with secure defaults.
        
        Returns:
            Dictionary with database connection parameters
        """
        return {
            'host': self.get_secret('pg_host') or os.getenv('PG_HOST', 'localhost'),
            'port': self.get_secret('pg_port') or os.getenv('PG_PORT', '5432'),
            'database': self.get_secret('pg_database') or os.getenv('PG_DATABASE', 'featurestore_manifest'),
            'user': self.get_secret('pg_user') or os.getenv('PG_USER', 'postgres'),
            'password': self.get_secret('pg_password') or os.getenv('PG_PASSWORD', 'secure_postgres_password')
        }
    
    def get_alert_config(self) -> Dict[str, str]:
        """
        Get alert system configuration.
        
        Returns:
            Dictionary with alert system parameters
        """
        return {
            'pagerduty_key': self.get_secret('pagerduty_key') or os.getenv('PAGERDUTY_KEY', 'pd_integration_key_12345'),
            'slack_webhook': self.get_secret('slack_webhook') or os.getenv('SLACK_WEBHOOK', 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'),
            'slack_channel': self.get_secret('slack_channel') or os.getenv('SLACK_CHANNEL', '#trading-alerts')
        }
    
    def get_broker_config(self) -> Dict[str, Optional[str]]:
        """
        Get broker API configuration.
        
        Returns:
            Dictionary with broker API parameters
        """
        return {
            'api_key': self.get_secret('broker_api_key'),
            'api_secret': self.get_secret('broker_api_secret'),
            'base_url': self.get_secret('broker_base_url') or os.getenv('BROKER_BASE_URL', 'https://api.broker.com'),
            'environment': self.get_secret('broker_environment') or os.getenv('BROKER_ENVIRONMENT', 'sandbox')
        }


# Global helper instance
_its_helper = None

def get_its_helper() -> ITSSecretsHelper:
    """Get global ITS secrets helper instance."""
    global _its_helper
    if _its_helper is None:
        _its_helper = ITSSecretsHelper()
    return _its_helper


# Convenience functions
def get_its_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret.
    
    Args:
        key: Secret key to retrieve
        default: Default value if secret not found
        
    Returns:
        Secret value or default
    """
    return get_its_helper().get_secret(key, default)


def get_database_config() -> Dict[str, str]:
    """
    Convenience function to get database configuration.
    
    Returns:
        Dictionary with database connection parameters
    """
    return get_its_helper().get_database_config()


def get_alert_config() -> Dict[str, str]:
    """
    Convenience function to get alert configuration.
    
    Returns:
        Dictionary with alert system parameters
    """
    return get_its_helper().get_alert_config()


def get_broker_config() -> Dict[str, Optional[str]]:
    """
    Convenience function to get broker configuration.
    
    Returns:
        Dictionary with broker API parameters
    """
    return get_its_helper().get_broker_config()
