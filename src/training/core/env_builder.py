"""
Environment Builder Core Module

Contains environment creation and configuration logic for training.
This module handles:
- Environment creation and setup
- Observation space configuration
- Action space configuration
- Environment parameter validation

This is an internal module - use src.training.TrainerAgent for public API.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Fallback for testing without gym
        gym = None
        spaces = None

# Internal imports
try:
    from ...gym_env.intraday_trading_env import IntradayTradingEnv
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gym_env.intraday_trading_env import IntradayTradingEnv


def make_env(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Optional[IntradayTradingEnv]:
    """
    Create a trading environment based on configuration.
    
    Args:
        config: Environment configuration
        logger: Optional logger instance
        
    Returns:
        Configured trading environment or None if creation fails
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        # Validate configuration
        if not validate_environment_config(config, logger):
            return None
        
        logger.info("Creating trading environment...")
        
        # Extract environment parameters
        env_params = {
            'observation_feature_cols': config.get('observation_feature_cols', []),
            'initial_balance': config.get('initial_balance', 100000),
            'transaction_cost': config.get('transaction_cost', 0.001),
            'lookback_window': config.get('lookback_window', 1),
            'max_steps': config.get('max_steps', 1000),
            'reward_type': config.get('reward_type', 'pnl'),
            'action_type': config.get('action_type', 'discrete')
        }
        
        # Configure reward parameters
        reward_config = configure_environment_rewards(config, logger)
        env_params.update(reward_config)
        
        # Create environment
        env = IntradayTradingEnv(**env_params)
        
        # Validate spaces
        obs_space = build_observation_space(
            config.get('observation_feature_cols', []), 
            config, 
            logger
        )
        action_space = build_action_space(config, logger)
        
        if obs_space is None or action_space is None:
            logger.error("Failed to build environment spaces")
            return None
        
        # Override spaces if needed
        env.observation_space = obs_space
        env.action_space = action_space
        
        logger.info(f"Trading environment created successfully")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        return env
        
    except Exception as e:
        logger.error(f"Failed to create trading environment: {e}")
        return None


def build_observation_space(
    feature_columns: list,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Optional[Any]:
    """
    Build the observation space for the environment.
    
    Args:
        feature_columns: List of feature column names
        config: Environment configuration
        logger: Optional logger instance
        
    Returns:
        Configured observation space or None if build fails
    """
    logger = logger or logging.getLogger(__name__)
    
    if spaces is None:
        logger.warning("Gym/Gymnasium not available, creating mock observation space")
        return {"shape": (len(feature_columns),), "type": "Box"}
    
    try:
        num_features = len(feature_columns)
        
        # Default bounds for normalized features
        low = np.full(num_features, -np.inf, dtype=np.float32)
        high = np.full(num_features, np.inf, dtype=np.float32)
        
        # Apply custom bounds if specified
        if 'observation_bounds' in config:
            bounds = config['observation_bounds']
            if 'low' in bounds:
                low = np.array(bounds['low'], dtype=np.float32)
            if 'high' in bounds:
                high = np.array(bounds['high'], dtype=np.float32)
                
        observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_features,),
            dtype=np.float32
        )
        
        logger.info(f"Built observation space with {num_features} features")
        return observation_space
        
    except Exception as e:
        logger.error(f"Failed to build observation space: {e}")
        return None


def build_action_space(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Optional[Any]:
    """
    Build the action space for the environment.
    
    Args:
        config: Environment configuration
        logger: Optional logger instance
        
    Returns:
        Configured action space or None if build fails
    """
    logger = logger or logging.getLogger(__name__)
    
    if spaces is None:
        logger.warning("Gym/Gymnasium not available, creating mock action space")
        action_type = config.get('action_type', 'discrete')
        if action_type == 'discrete':
            return {"n": config.get('num_actions', 3), "type": "Discrete"}
        else:
            return {"shape": (1,), "type": "Box", "low": -1.0, "high": 1.0}
    
    try:
        action_type = config.get('action_type', 'discrete')
        
        if action_type == 'discrete':
            # Discrete actions: 0=Hold, 1=Buy, 2=Sell
            num_actions = config.get('num_actions', 3)
            action_space = spaces.Discrete(num_actions)
            logger.info(f"Built discrete action space with {num_actions} actions")
            
        elif action_type == 'continuous':
            # Continuous actions: [-1, 1] for position sizing
            action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
            logger.info("Built continuous action space [-1, 1]")
            
        else:
            logger.error(f"Unsupported action type: {action_type}")
            return None
            
        return action_space
        
    except Exception as e:
        logger.error(f"Failed to build action space: {e}")
        return None


def validate_environment_config(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate environment configuration parameters.
    
    Args:
        config: Environment configuration to validate
        logger: Optional logger instance
        
    Returns:
        True if configuration is valid, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    required_keys = [
        'observation_feature_cols',
        'action_type',
        'initial_balance',
        'transaction_cost'
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required environment config key: {key}")
            return False
            
    # Validate observation features
    obs_features = config.get('observation_feature_cols', [])
    if not obs_features or len(obs_features) == 0:
        logger.error("No observation features specified")
        return False
        
    # Validate action type
    action_type = config.get('action_type')
    if action_type not in ['discrete', 'continuous']:
        logger.error(f"Invalid action type: {action_type}")
        return False
        
    # Validate initial balance
    initial_balance = config.get('initial_balance', 0)
    if initial_balance <= 0:
        logger.error(f"Invalid initial balance: {initial_balance}")
        return False
        
    # Validate transaction cost
    transaction_cost = config.get('transaction_cost', 0)
    if transaction_cost < 0 or transaction_cost > 1:
        logger.error(f"Invalid transaction cost: {transaction_cost}")
        return False
        
    logger.info("Environment configuration validation passed")
    return True


def configure_environment_rewards(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Configure reward parameters for the environment.
    
    Args:
        config: Environment configuration
        logger: Optional logger instance
        
    Returns:
        Dictionary of reward configuration parameters
    """
    logger = logger or logging.getLogger(__name__)
    
    reward_config = {
        'reward_type': config.get('reward_type', 'pnl'),
        'reward_scaling': config.get('reward_scaling', 1.0),
        'risk_penalty': config.get('risk_penalty', 0.0),
        'transaction_penalty': config.get('transaction_penalty', 0.0),
        'holding_penalty': config.get('holding_penalty', 0.0)
    }
    
    # Validate reward parameters
    if reward_config['reward_scaling'] <= 0:
        logger.warning("Invalid reward scaling, using default 1.0")
        reward_config['reward_scaling'] = 1.0
        
    if reward_config['risk_penalty'] < 0:
        logger.warning("Negative risk penalty, setting to 0")
        reward_config['risk_penalty'] = 0.0
        
    logger.info(f"Configured environment rewards: {reward_config}")
    return reward_config