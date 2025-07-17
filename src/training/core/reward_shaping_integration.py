# src/training/core/reward_shaping_integration.py
"""
Integration utilities for reward shaping in the training pipeline.

This module provides utilities to integrate RewardShapingWrapper and
functional risk penalties into the existing training infrastructure.
"""

import gymnasium as gym
from typing import Dict, Any, Optional
import logging

try:
    from ...gym_env.reward_shaping_wrapper import (
        RewardShapingWrapper, 
        RiskPenaltyRewardShaper,
        FunctionalRiskPenaltyCallback
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gym_env.reward_shaping_wrapper import (
        RewardShapingWrapper, 
        RiskPenaltyRewardShaper,
        FunctionalRiskPenaltyCallback
    )


def wrap_environment_with_risk_shaping(
    env: gym.Env, 
    risk_advisor, 
    penalty_weight: float = 0.1,
    enable_logging: bool = True
) -> RewardShapingWrapper:
    """
    Wrap an environment with risk-based reward shaping.
    
    Args:
        env: Base trading environment
        risk_advisor: Risk advisor for penalty calculation
        penalty_weight: Weight for risk penalties
        enable_logging: Whether to enable detailed logging
        
    Returns:
        Wrapped environment with functional risk penalties
    """
    logger = logging.getLogger(__name__)
    
    # Wrap environment with reward shaping capability
    wrapped_env = RewardShapingWrapper(env)
    
    # Add risk penalty modifier
    risk_shaper = RiskPenaltyRewardShaper(risk_advisor, penalty_weight)
    wrapped_env.add_reward_modifier(risk_shaper, "risk_penalty")
    
    if enable_logging:
        logger.info(f"Environment wrapped with risk-based reward shaping (weight: {penalty_weight})")
    
    return wrapped_env


def create_functional_risk_callback(
    wrapped_env: RewardShapingWrapper,
    risk_advisor,
    penalty_weight: float = 0.1
) -> FunctionalRiskPenaltyCallback:
    """
    Create a functional risk callback that injects penalties into the environment.
    
    Args:
        wrapped_env: RewardShapingWrapper instance
        risk_advisor: Risk advisor for penalty calculation
        penalty_weight: Weight for risk penalties
        
    Returns:
        Functional callback for use during training
    """
    return FunctionalRiskPenaltyCallback(wrapped_env, risk_advisor, penalty_weight)


class RewardShapingCallbackAdapter:
    """
    Adapter to integrate FunctionalRiskPenaltyCallback with SB3 callback system.
    
    This bridges the gap between our functional callback and SB3's callback interface.
    """
    
    def __init__(self, functional_callback: FunctionalRiskPenaltyCallback):
        self.functional_callback = functional_callback
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def on_step(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> bool:
        """
        Called by SB3 callback system.
        
        Args:
            locals_dict: Local variables from training step
            globals_dict: Global variables from training step
            
        Returns:
            True to continue training
        """
        try:
            # Extract observation from locals
            obs = None
            
            # Try different ways to get observation
            if 'obs' in locals_dict:
                obs = locals_dict['obs']
            elif 'observations' in locals_dict:
                obs = locals_dict['observations']
            elif hasattr(locals_dict.get('self'), '_last_obs'):
                obs = locals_dict['self']._last_obs
            
            if obs is not None:
                # Handle VecEnv case (take first environment)
                if hasattr(obs, '__len__') and len(obs) > 0 and hasattr(obs[0], 'shape'):
                    obs = obs[0]
                
                # Call functional callback
                self.functional_callback.on_step(obs)
            
        except Exception as e:
            self.logger.error(f"Reward shaping callback adapter failed: {e}")
        
        return True


def integrate_reward_shaping_with_training(
    env: gym.Env,
    risk_advisor,
    penalty_weight: float = 0.1,
    use_callback_approach: bool = True
) -> tuple[gym.Env, Optional[FunctionalRiskPenaltyCallback]]:
    """
    Integrate reward shaping into the training pipeline.
    
    Args:
        env: Base trading environment
        risk_advisor: Risk advisor for penalty calculation
        penalty_weight: Weight for risk penalties
        use_callback_approach: Whether to use callback-based approach
        
    Returns:
        Tuple of (wrapped_environment, optional_callback)
    """
    logger = logging.getLogger(__name__)
    
    # Wrap environment with reward shaping
    wrapped_env = wrap_environment_with_risk_shaping(
        env, risk_advisor, penalty_weight
    )
    
    callback = None
    if use_callback_approach:
        # Create functional callback for additional control
        callback = create_functional_risk_callback(
            wrapped_env, risk_advisor, penalty_weight
        )
        logger.info("Created functional risk callback for additional penalty injection")
    
    logger.info(f"Reward shaping integration complete (penalty_weight: {penalty_weight})")
    
    return wrapped_env, callback


# Example usage for trainer integration
def example_trainer_integration():
    """
    Example of how to integrate reward shaping into the trainer.
    
    This shows the pattern for modifying the trainer to use reward shaping.
    """
    # This would be called in the trainer setup
    
    # 1. Create base environment (existing code)
    # base_env = create_trading_environment(...)
    
    # 2. Create risk advisor (existing code)
    # risk_advisor = create_risk_advisor(...)
    
    # 3. Wrap environment with reward shaping
    # wrapped_env, risk_callback = integrate_reward_shaping_with_training(
    #     base_env, risk_advisor, penalty_weight=0.1
    # )
    
    # 4. Use wrapped environment for training
    # model = PPO("MlpPolicy", wrapped_env, ...)
    
    # 5. Add callback to training (if using callback approach)
    # callbacks = [existing_callbacks...]
    # if risk_callback:
    #     adapter = RewardShapingCallbackAdapter(risk_callback)
    #     callbacks.append(adapter)
    
    # 6. Train model
    # model.learn(total_timesteps=..., callback=callbacks)
    
    pass


if __name__ == "__main__":
    # Test the integration
    print("Reward shaping integration utilities loaded successfully")
    print("Use integrate_reward_shaping_with_training() to add functional risk penalties")