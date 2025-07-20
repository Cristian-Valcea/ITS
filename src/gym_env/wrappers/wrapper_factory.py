"""
Wrapper Factory for Trading Environment

Provides easy configuration and chaining of trading rule wrappers
to create modular, testable trading environments.
"""

import gymnasium as gym
from typing import Dict, Any, Optional, List, Type
import logging
from pathlib import Path

# Import all wrapper classes
from .cooldown_wrapper import CooldownWrapper
from .size_limit_wrapper import SizeLimitWrapper
from .action_penalty_wrapper import ActionPenaltyWrapper
from .streaming_trade_log_wrapper import StreamingTradeLogWrapper
from .risk_wrapper import RiskObsWrapper


class TradingWrapperFactory:
    """
    Factory for creating trading environments with configurable wrapper chains.
    
    This factory allows you to:
    - Configure wrapper chains declaratively
    - Enable/disable wrappers based on config
    - Maintain consistent wrapper ordering
    - Test individual wrappers in isolation
    - Migrate gradually from monolithic to modular architecture
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TradingWrapperFactory")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Registry of available wrappers
        self.wrapper_registry = {
            'cooldown': CooldownWrapper,
            'size_limit': SizeLimitWrapper,
            'action_penalty': ActionPenaltyWrapper,
            'streaming_log': StreamingTradeLogWrapper,
            'risk_obs': RiskObsWrapper
        }
        
        # Default wrapper order (applied bottom to top)
        self.default_order = [
            'cooldown',      # Apply cooldown restrictions first
            'size_limit',    # Then check size limits
            'action_penalty', # Apply penalties for bad behavior
            'streaming_log', # Log trades (near the top to catch actual trades)
            'risk_obs'       # Add risk features to observations (outermost)
        ]
    
    def create_wrapped_env(self, 
                          base_env: gym.Env,
                          wrapper_config: Dict[str, Any],
                          wrapper_order: Optional[List[str]] = None) -> gym.Env:
        """
        Create a wrapped trading environment with specified configuration.
        
        Args:
            base_env: Base trading environment (should be simple/clean)
            wrapper_config: Configuration for each wrapper
            wrapper_order: Custom wrapper order (uses default if None)
        
        Returns:
            Wrapped environment with applied trading rules
        
        Example wrapper_config:
        {
            'cooldown': {
                'enabled': True,
                'cooldown_steps': 5
            },
            'size_limit': {
                'enabled': True,
                'max_position_pct': 0.25,
                'max_portfolio_risk_pct': 0.05
            },
            'action_penalty': {
                'enabled': True,
                'action_change_penalty': 0.001,
                'ping_pong_penalty': 0.005
            },
            'streaming_log': {
                'enabled': True,
                'log_dir': 'logs/trade_logs',
                'batch_size': 100
            },
            'risk_obs': {
                'enabled': False  # Disable risk observations
            }
        }
        """
        env = base_env
        order = wrapper_order or self.default_order
        
        self.logger.info(f"Creating wrapped environment with order: {order}")
        
        # Apply wrappers in specified order
        for wrapper_name in order:
            if wrapper_name not in self.wrapper_registry:
                self.logger.warning(f"Unknown wrapper: {wrapper_name}")
                continue
            
            wrapper_conf = wrapper_config.get(wrapper_name, {})
            
            # Skip if disabled
            if not wrapper_conf.get('enabled', True):
                self.logger.info(f"Skipping disabled wrapper: {wrapper_name}")
                continue
            
            # Get wrapper class
            wrapper_class = self.wrapper_registry[wrapper_name]
            
            # Extract wrapper parameters (exclude 'enabled' flag)
            wrapper_params = {k: v for k, v in wrapper_conf.items() if k != 'enabled'}
            
            # Apply wrapper
            try:
                env = wrapper_class(env, **wrapper_params)
                self.logger.info(f"Applied wrapper: {wrapper_name} with params: {wrapper_params}")
            except Exception as e:
                self.logger.error(f"Failed to apply wrapper {wrapper_name}: {e}")
                raise
        
        return env
    
    def create_minimal_env(self, base_env: gym.Env) -> gym.Env:
        """Create environment with minimal wrappers for testing."""
        config = {
            'streaming_log': {'enabled': True, 'batch_size': 10}
        }
        return self.create_wrapped_env(base_env, config, ['streaming_log'])
    
    def create_training_env(self, base_env: gym.Env) -> gym.Env:
        """Create environment optimized for training."""
        config = {
            'cooldown': {
                'enabled': True,
                'cooldown_steps': 5
            },
            'size_limit': {
                'enabled': True,
                'max_position_pct': 0.25,
                'max_portfolio_risk_pct': 0.05,
                'min_cash_reserve_pct': 0.10
            },
            'action_penalty': {
                'enabled': True,
                'action_change_penalty': 0.001,
                'ping_pong_penalty': 0.005,
                'rapid_flip_penalty': 0.01
            },
            'streaming_log': {
                'enabled': True,
                'log_dir': 'logs/training_trade_logs',
                'batch_size': 50,
                'write_frequency': 10
            }
        }
        return self.create_wrapped_env(base_env, config)
    
    def create_production_env(self, base_env: gym.Env) -> gym.Env:
        """Create environment optimized for production trading."""
        config = {
            'cooldown': {
                'enabled': True,
                'cooldown_steps': 3  # Shorter cooldown for production
            },
            'size_limit': {
                'enabled': True,
                'max_position_pct': 0.20,  # More conservative
                'max_portfolio_risk_pct': 0.03,
                'min_cash_reserve_pct': 0.15
            },
            'action_penalty': {
                'enabled': True,
                'action_change_penalty': 0.002,  # Higher penalties
                'ping_pong_penalty': 0.01,
                'rapid_flip_penalty': 0.02,
                'enable_timing_penalties': True
            },
            'streaming_log': {
                'enabled': True,
                'log_dir': 'logs/production_trade_logs',
                'batch_size': 1,  # Immediate logging for production
                'write_frequency': 1,
                'compression': 'snappy'
            },
            'risk_obs': {
                'enabled': True,
                'preserve_sequence': True
            }
        }
        return self.create_wrapped_env(base_env, config)
    
    def create_debug_env(self, base_env: gym.Env) -> gym.Env:
        """Create environment for debugging with all wrappers enabled."""
        config = {
            'cooldown': {
                'enabled': True,
                'cooldown_steps': 2
            },
            'size_limit': {
                'enabled': True,
                'max_position_pct': 0.30,
                'max_portfolio_risk_pct': 0.10
            },
            'action_penalty': {
                'enabled': True,
                'action_change_penalty': 0.0001,  # Low penalties for debugging
                'ping_pong_penalty': 0.001
            },
            'streaming_log': {
                'enabled': True,
                'log_dir': 'logs/debug_trade_logs',
                'batch_size': 5,
                'write_frequency': 5
            }
        }
        return self.create_wrapped_env(base_env, config)
    
    def register_wrapper(self, name: str, wrapper_class: Type[gym.Wrapper]):
        """Register a custom wrapper."""
        self.wrapper_registry[name] = wrapper_class
        self.logger.info(f"Registered custom wrapper: {name}")
    
    def get_wrapper_info(self, env: gym.Env) -> List[str]:
        """Get list of wrappers applied to an environment."""
        wrappers = []
        current_env = env
        
        while hasattr(current_env, 'env'):
            wrapper_name = current_env.__class__.__name__
            wrappers.append(wrapper_name)
            current_env = current_env.env
        
        # Add base environment
        wrappers.append(current_env.__class__.__name__)
        
        return wrappers


# Convenience function for backward compatibility
def create_trading_env(base_env: gym.Env, 
                      config_type: str = 'training',
                      custom_config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Convenience function to create wrapped trading environments.
    
    Args:
        base_env: Base trading environment
        config_type: 'minimal', 'training', 'production', 'debug'
        custom_config: Custom wrapper configuration (overrides preset)
    
    Returns:
        Wrapped trading environment
    """
    factory = TradingWrapperFactory()
    
    if custom_config:
        return factory.create_wrapped_env(base_env, custom_config)
    elif config_type == 'minimal':
        return factory.create_minimal_env(base_env)
    elif config_type == 'training':
        return factory.create_training_env(base_env)
    elif config_type == 'production':
        return factory.create_production_env(base_env)
    elif config_type == 'debug':
        return factory.create_debug_env(base_env)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


# Migration helper function
def migrate_from_monolithic(current_env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert monolithic environment config to wrapper-based config.
    
    Args:
        current_env_config: Current IntradayTradingEnv configuration
    
    Returns:
        Wrapper-based configuration
    """
    wrapper_config = {}
    
    # Map cooldown settings
    if current_env_config.get('trade_cooldown_steps', 0) > 0:
        wrapper_config['cooldown'] = {
            'enabled': True,
            'cooldown_steps': current_env_config['trade_cooldown_steps']
        }
    
    # Map size limit settings
    wrapper_config['size_limit'] = {
        'enabled': True,
        'max_position_pct': current_env_config.get('position_sizing_pct_capital', 0.25),
        'max_portfolio_risk_pct': current_env_config.get('max_daily_drawdown_pct', 0.05),
        'min_cash_reserve_pct': 0.10  # Default value
    }
    
    # Map action penalty settings
    if current_env_config.get('action_change_penalty_factor', 0) > 0:
        wrapper_config['action_penalty'] = {
            'enabled': True,
            'action_change_penalty': current_env_config['action_change_penalty_factor'],
            'ping_pong_penalty': 0.005,  # Default value
            'rapid_flip_penalty': 0.01   # Default value
        }
    
    # Map trade logging settings
    if current_env_config.get('log_trades', True):
        wrapper_config['streaming_log'] = {
            'enabled': True,
            'log_dir': 'logs/migrated_trade_logs',
            'batch_size': 100,
            'write_frequency': 10
        }
    
    return wrapper_config