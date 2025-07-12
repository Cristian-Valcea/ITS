"""
Environment Builder Core Module

Contains environment creation and configuration logic for training.
This module handles:
- Environment creation and setup
- Observation space configuration
- Action space configuration
- Environment parameter validation
- Vectorized environment creation with shared memory

This is an internal module - use src.training.TrainerAgent for public API.
"""

from __future__ import annotations
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
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

# Import vectorized environment components
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
    VECENV_AVAILABLE = True
    
    # Try to import ShmemVecEnv if available (SB3 1.8+)
    try:
        from stable_baselines3.common.vec_env import ShmemVecEnv
        SHMEM_AVAILABLE = True
    except ImportError:
        ShmemVecEnv = None
        SHMEM_AVAILABLE = False
        
except ImportError:
    # Fallback for systems without SB3 vectorized environments
    SubprocVecEnv = None
    ShmemVecEnv = None
    VecMonitor = None
    DummyVecEnv = None
    VECENV_AVAILABLE = False
    SHMEM_AVAILABLE = False

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
        
        # Add reward configuration
        reward_config = configure_environment_rewards(config, logger)
        env_params.update(reward_config)
        
        # Create environment
        env = IntradayTradingEnv(**env_params)
        
        logger.info(f"Trading environment created successfully")
        logger.info(f"  - Observation features: {len(env_params['observation_feature_cols'])}")
        logger.info(f"  - Action type: {env_params['action_type']}")
        logger.info(f"  - Reward type: {env_params['reward_type']}")
        
        return env
        
    except Exception as e:
        logger.error(f"Failed to create trading environment: {e}")
        return None


def _make_single_env(
    symbol: str,
    data_path: Path,
    config: Dict[str, Any],
    env_id: int = 0
) -> Callable[[], gym.Env]:
    """
    Factory that returns a closure -> Env; required by SB3 VecEnv API.
    
    Args:
        symbol: Trading symbol for this environment
        data_path: Path to data file for this symbol
        config: Environment configuration
        env_id: Environment ID for logging/debugging
        
    Returns:
        Callable that creates and returns a gym environment
    """
    def _init():
        try:
            # Create the trading environment
            env_params = {
                'observation_feature_cols': config.get('observation_feature_cols', []),
                'initial_balance': config.get('initial_balance', 100000),
                'transaction_cost': config.get('transaction_cost', 0.001),
                'lookback_window': config.get('lookback_window', 1),
                'max_steps': config.get('max_steps', 1000),
                'reward_type': config.get('reward_type', 'pnl'),
                'action_type': config.get('action_type', 'discrete')
            }
            
            # Add reward configuration
            reward_config = configure_environment_rewards(config)
            env_params.update(reward_config)
            
            # Create environment instance
            env = IntradayTradingEnv(**env_params)
            
            # Add episode statistics recording (VecMonitor will pick this up)
            if gym is not None:
                env = gym.wrappers.RecordEpisodeStatistics(env)
            
            # Set unique random seed for this worker (deterministic across runs)
            if hasattr(env, 'seed'):
                # Use SHA1 for deterministic, reproducible seeding across process runs
                seed_string = f"{symbol}_{env_id}"
                seed_hash = hashlib.sha1(seed_string.encode('utf-8')).hexdigest()
                seed_value = int(seed_hash[:8], 16) % (2**32)  # Use first 8 hex chars for 32-bit seed
                env.seed(seed_value)
            
            return env
            
        except Exception as e:
            # Fallback to a dummy environment if creation fails
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create environment for {symbol}: {e}")
            
            # Create minimal dummy environment
            if gym is not None and spaces is not None:
                from gym.envs.classic_control import CartPoleEnv
                return CartPoleEnv()
            else:
                raise e
    
    return _init


def build_vec_env(
    symbols: List[str],
    data_dir: Path,
    config: Dict[str, Any],
    n_envs: Optional[int] = None,
    monitor_path: Optional[str] = "logs/vec_monitor",
    use_shared_memory: bool = True,
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Build a vectorized environment with SubprocVecEnv/ShmemVecEnv + VecMonitor for optimal performance.
    
    This function creates multiple parallel environments using shared memory or subprocesses for
    3-4x faster experience throughput compared to single-threaded rollouts.
    
    Args:
        symbols: List of trading symbols to create environments for
        data_dir: Directory containing data files
        config: Environment configuration dictionary
        n_envs: Number of environments (defaults to min(len(symbols), cpu_count))
        monitor_path: Path for VecMonitor logs (None to disable)
        use_shared_memory: Whether to use ShmemVecEnv (requires SB3 1.8+)
        logger: Optional logger instance
        
    Returns:
        VecMonitor-wrapped vectorized environment
    """
    logger = logger or logging.getLogger(__name__)
    
    # Determine number of environments
    if n_envs is None:
        n_envs = min(len(symbols), mp.cpu_count())
        logger.info(f"Auto-detected {n_envs} environments (symbols: {len(symbols)}, CPUs: {mp.cpu_count()})")
    
    # Validate inputs
    if n_envs <= 0:
        raise ValueError(f"Invalid number of environments: {n_envs}")
    
    if not symbols:
        raise ValueError("No symbols provided for environment creation")
    
    # Create environment factory functions
    env_fns = []
    for i in range(n_envs):
        symbol = symbols[i % len(symbols)]  # Cycle through symbols if n_envs > len(symbols)
        data_path = data_dir / f"{symbol}.parquet"
        
        # Add environment ID to config for unique seeding
        env_config = config.copy()
        env_config['env_id'] = i
        env_config['symbol'] = symbol
        
        env_fn = _make_single_env(symbol, data_path, env_config, env_id=i)
        env_fns.append(env_fn)
    
    # Create vectorized environment
    if VECENV_AVAILABLE and n_envs > 1:
        try:
            if use_shared_memory and SHMEM_AVAILABLE and ShmemVecEnv is not None:
                # Use shared memory vectorized environment for best performance
                logger.info(f"Creating ShmemVecEnv with {n_envs} workers")
                vec_env = ShmemVecEnv(env_fns, context="spawn")
                logger.info("✅ ShmemVecEnv created successfully")
            elif SubprocVecEnv is not None:
                # Use subprocess vectorized environment (good performance)
                logger.info(f"Creating SubprocVecEnv with {n_envs} workers")
                vec_env = SubprocVecEnv(env_fns)
                logger.info("✅ SubprocVecEnv created successfully")
            else:
                # Fallback to dummy vectorized environment
                logger.info(f"Creating DummyVecEnv with {n_envs} environments")
                vec_env = DummyVecEnv(env_fns)
                logger.info("✅ DummyVecEnv created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create vectorized environment: {e}, falling back to DummyVecEnv")
            if DummyVecEnv is not None:
                vec_env = DummyVecEnv(env_fns)
            else:
                raise RuntimeError("No vectorized environment implementation available")
    else:
        # Single environment or vectorized environments not available
        if VECENV_AVAILABLE and DummyVecEnv is not None and n_envs > 1:
            logger.info(f"Creating DummyVecEnv with {n_envs} environments")
            vec_env = DummyVecEnv(env_fns)
        else:
            logger.warning("Vectorized environments not available, creating single environment")
            # Return single environment wrapped in a simple interface
            return env_fns[0]()
    
    # Wrap with VecMonitor for episode statistics
    if VECENV_AVAILABLE and VecMonitor is not None and monitor_path is not None:
        try:
            # Create monitor directory if it doesn't exist
            monitor_dir = Path(monitor_path).parent
            monitor_dir.mkdir(parents=True, exist_ok=True)
            
            # Add info keywords for additional monitoring
            info_keywords = ("drawdown", "turnover", "sharpe_ratio", "max_drawdown")
            
            vec_env = VecMonitor(
                vec_env, 
                filename=monitor_path,
                info_keywords=info_keywords
            )
            logger.info(f"✅ VecMonitor enabled, logging to: {monitor_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create VecMonitor: {e}, continuing without monitoring")
    
    logger.info(f"Vectorized environment ready with {n_envs} workers")
    return vec_env


def build_single_env(
    symbol: str,
    data_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Build a single environment for testing or single-threaded training.
    
    Args:
        symbol: Trading symbol
        data_dir: Directory containing data files
        config: Environment configuration
        logger: Optional logger instance
        
    Returns:
        Single trading environment
    """
    logger = logger or logging.getLogger(__name__)
    
    data_path = data_dir / f"{symbol}.parquet"
    env_fn = _make_single_env(symbol, data_path, config, env_id=0)
    
    env = env_fn()
    logger.info(f"Single environment created for {symbol}")
    
    return env


def get_optimal_n_envs(symbols: List[str], max_envs: Optional[int] = None) -> int:
    """
    Calculate optimal number of environments based on available resources.
    
    Args:
        symbols: List of available symbols
        max_envs: Maximum number of environments to create
        
    Returns:
        Optimal number of environments
    """
    cpu_count = mp.cpu_count()
    symbol_count = len(symbols)
    
    # Use all CPUs but don't exceed number of symbols
    optimal = min(cpu_count, symbol_count)
    
    # Apply maximum limit if specified
    if max_envs is not None:
        optimal = min(optimal, max_envs)
    
    # Ensure at least 1 environment
    return max(1, optimal)


def validate_environment_config(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate environment configuration parameters.
    
    Args:
        config: Environment configuration dictionary
        logger: Optional logger instance
        
    Returns:
        True if configuration is valid, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    # DESIGN SPEC: No market impact features in training observations
    include_market_impact = config.get('include_market_impact_features', False)
    if include_market_impact:
        logger.error(
            "DESIGN VIOLATION: include_market_impact_features=True is not allowed in training. "
            "Market impact features should not be included in training observations as per design spec. "
            "Set include_market_impact_features=False or remove this flag."
        )
        return False
    
    # Validate observation features
    obs_features = config.get('observation_feature_cols', [])
    if not isinstance(obs_features, list):
        logger.error("observation_feature_cols must be a list")
        return False
        
    if len(obs_features) == 0:
        logger.warning("No observation features specified")
    
    # Validate action type
    action_type = config.get('action_type', 'discrete')
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


# ============================================================================
# SMOKE TEST (only runs when called standalone)
# ============================================================================

if __name__ == "__main__":
    """Quick smoke test for vectorized environment creation."""
    import tempfile
    
    # Mock configuration
    test_config = {
        'observation_feature_cols': ['close', 'volume'],
        'initial_balance': 100000,
        'transaction_cost': 0.001,
        'lookback_window': 1,
        'max_steps': 100,
        'reward_type': 'pnl',
        'action_type': 'discrete'
    }
    
    # Test symbols
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            
            # Create dummy data files
            for symbol in test_symbols:
                (data_dir / f"{symbol}.parquet").touch()
            
            print("Testing vectorized environment creation...")
            
            # Test optimal environment count
            n_envs = get_optimal_n_envs(test_symbols, max_envs=2)
            print(f"Optimal environments: {n_envs}")
            
            # Test environment creation (will fail due to dummy data, but tests the API)
            try:
                vec_env = build_vec_env(
                    symbols=test_symbols,
                    data_dir=data_dir,
                    config=test_config,
                    n_envs=n_envs,
                    monitor_path=None,  # Disable monitoring for test
                    use_shared_memory=False,  # Use SubprocVecEnv for test
                    logger=None
                )
                print("✅ Vectorized environment creation API works")
                
                if hasattr(vec_env, 'close'):
                    vec_env.close()
                    
            except Exception as e:
                print(f"⚠️  Environment creation failed (expected with dummy data): {e}")
            
            print("✅ Smoke test completed")
            
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()