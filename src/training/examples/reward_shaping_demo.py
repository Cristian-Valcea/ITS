# src/training/examples/reward_shaping_demo.py
"""
Demonstration of functional reward shaping with risk penalties.

This script shows how to properly integrate risk-based reward shaping
that actually modifies the learning signal (unlike the broken callback).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ...gym_env.intraday_trading_env import IntradayTradingEnv
    from ...gym_env.reward_shaping_wrapper import (
        RewardShapingWrapper,
        RiskPenaltyRewardShaper,
        FunctionalRiskPenaltyCallback
    )
    from ..core.reward_shaping_integration import integrate_reward_shaping_with_training
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from gym_env.intraday_trading_env import IntradayTradingEnv
    from gym_env.reward_shaping_wrapper import (
        RewardShapingWrapper,
        RiskPenaltyRewardShaper,
        FunctionalRiskPenaltyCallback
    )
    from training.core.reward_shaping_integration import integrate_reward_shaping_with_training


class MockRiskAdvisor:
    """Mock risk advisor for demonstration purposes."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.step_count = 0
    
    def evaluate(self, obs_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Mock risk evaluation that returns higher penalties for extreme positions.
        
        Args:
            obs_dict: Observation dictionary with market_features and position
            
        Returns:
            Dictionary with risk metrics
        """
        self.step_count += 1
        
        # Get position from observation
        position = obs_dict.get('position', 0.0)
        
        # Create mock drawdown velocity based on position extremes
        # Higher penalty for extreme positions (simulate risk)
        drawdown_vel = abs(position) * 0.1  # 10% penalty per unit position
        
        # Add some randomness to simulate market conditions
        if self.step_count % 50 == 0:  # Periodic "risk events"
            drawdown_vel += 0.2
        
        return {
            'drawdown_vel': drawdown_vel,
            'position_risk': abs(position),
            'step_count': self.step_count
        }


def create_mock_trading_data(num_steps: int = 1000) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Create mock trading data for demonstration."""
    
    # Generate mock price data (random walk)
    np.random.seed(42)
    price_changes = np.random.normal(0, 1, num_steps)
    prices = 100 + np.cumsum(price_changes * 0.1)  # Start at $100
    
    # Generate mock feature data (normalized technical indicators)
    features = np.random.randn(num_steps, 5)  # 5 mock features
    
    # Create pandas series with timestamps
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=1),
        periods=num_steps,
        freq='1min'
    )
    
    price_series = pd.Series(prices, index=timestamps)
    volume_series = pd.Series(np.random.randint(1000, 10000, num_steps), index=timestamps)
    
    return features, price_series, volume_series


def demonstrate_broken_callback():
    """Demonstrate the broken callback that only logs penalties."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING BROKEN CALLBACK (LOGGING ONLY)")
    logger.info("=" * 60)
    
    # Create mock environment
    features, prices, volumes = create_mock_trading_data(100)
    
    env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        volume_data=volumes,
        initial_capital=100000,
        reward_scaling=1.0
    )
    
    # Create mock risk advisor
    risk_advisor = MockRiskAdvisor()
    
    # Simulate the broken callback approach
    from ..core.risk_callbacks import RiskPenaltyCallback
    
    # This callback only logs penalties - doesn't modify rewards!
    broken_callback = RiskPenaltyCallback(risk_advisor, lam=0.1, verbose=1)
    
    # Run a few steps to show the problem
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Simulate what the broken callback does
        obs_dict = {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }
        
        risk = risk_advisor.evaluate(obs_dict)
        penalty = 0.1 * risk.get('drawdown_velocity', 0)
        
        total_reward += reward
        
        logger.info(f"Step {step}: Reward={reward:.6f}, Calculated Penalty={penalty:.6f} (NOT APPLIED)")
        
        if terminated or truncated:
            break
    
    logger.info(f"Total reward (broken approach): {total_reward:.6f}")
    logger.info("❌ Penalties were calculated but NOT applied to learning signal!")


def demonstrate_fixed_reward_shaping():
    """Demonstrate the fixed reward shaping that actually modifies rewards."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING FIXED REWARD SHAPING (FUNCTIONAL)")
    logger.info("=" * 60)
    
    # Create mock environment
    features, prices, volumes = create_mock_trading_data(100)
    
    base_env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        volume_data=volumes,
        initial_capital=100000,
        reward_scaling=1.0
    )
    
    # Create mock risk advisor
    risk_advisor = MockRiskAdvisor()
    
    # Wrap environment with functional reward shaping
    wrapped_env, risk_callback = integrate_reward_shaping_with_training(
        base_env, risk_advisor, penalty_weight=0.1
    )
    
    # Run a few steps to show the fix
    obs, _ = wrapped_env.reset()
    total_base_reward = 0.0
    total_shaped_reward = 0.0
    total_penalties = 0.0
    
    for step in range(10):
        action = wrapped_env.action_space.sample()
        obs, shaped_reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Extract reward shaping info
        shaping_info = info.get('reward_shaping', {})
        base_reward = shaping_info.get('base_reward', shaped_reward)
        penalty = shaping_info.get('total_penalty', 0.0)
        
        total_base_reward += base_reward
        total_shaped_reward += shaped_reward
        total_penalties += penalty
        
        logger.info(
            f"Step {step}: Base={base_reward:.6f}, Penalty={penalty:.6f}, "
            f"Shaped={shaped_reward:.6f} ✅ APPLIED"
        )
        
        if terminated or truncated:
            break
    
    logger.info(f"Total base reward: {total_base_reward:.6f}")
    logger.info(f"Total penalties: {total_penalties:.6f}")
    logger.info(f"Total shaped reward: {total_shaped_reward:.6f}")
    logger.info("✅ Penalties were actually applied to learning signal!")
    
    # Show wrapper statistics
    stats = wrapped_env.get_penalty_stats()
    logger.info(f"Wrapper stats: {stats}")


def demonstrate_callback_integration():
    """Demonstrate callback-based penalty injection."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING CALLBACK-BASED PENALTY INJECTION")
    logger.info("=" * 60)
    
    # Create mock environment
    features, prices, volumes = create_mock_trading_data(100)
    
    base_env = IntradayTradingEnv(
        processed_feature_data=features,
        price_data=prices,
        volume_data=volumes,
        initial_capital=100000,
        reward_scaling=1.0
    )
    
    # Wrap with reward shaping
    wrapped_env = RewardShapingWrapper(base_env)
    
    # Create risk advisor and functional callback
    risk_advisor = MockRiskAdvisor()
    risk_callback = FunctionalRiskPenaltyCallback(
        wrapped_env, risk_advisor, penalty_weight=0.1
    )
    
    # Run simulation with callback
    obs, _ = wrapped_env.reset()
    total_reward = 0.0
    
    for step in range(10):
        # Call callback to inject penalties (simulates SB3 callback system)
        risk_callback.on_step(obs)
        
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        total_reward += reward
        
        # Show penalty injection
        shaping_info = info.get('reward_shaping', {})
        penalty = shaping_info.get('total_penalty', 0.0)
        
        logger.info(f"Step {step}: Reward={reward:.6f}, Injected Penalty={penalty:.6f}")
        
        if terminated or truncated:
            break
    
    logger.info(f"Total reward with callback injection: {total_reward:.6f}")
    
    # Show callback statistics
    callback_stats = risk_callback.get_stats()
    logger.info(f"Callback stats: {callback_stats}")


def main():
    """Run all demonstrations."""
    logger.info("🔧 REWARD SHAPING FIX DEMONSTRATION")
    logger.info("This shows the difference between broken and functional reward shaping")
    
    # Show the broken approach
    demonstrate_broken_callback()
    
    print("\n")
    
    # Show the fixed approach
    demonstrate_fixed_reward_shaping()
    
    print("\n")
    
    # Show callback integration
    demonstrate_callback_integration()
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("❌ Broken: RiskPenaltyCallback only logs penalties")
    logger.info("✅ Fixed: RewardShapingWrapper actually modifies rewards")
    logger.info("✅ Integration: Use integrate_reward_shaping_with_training()")
    logger.info("✅ Callback: FunctionalRiskPenaltyCallback injects penalties")


if __name__ == "__main__":
    main()