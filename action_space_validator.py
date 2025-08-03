#!/usr/bin/env python3
"""
🛡️ ACTION SPACE VALIDATOR
Pre-flight validation to prevent action space mismatches
"""

def assert_action_compat(model, env):
    """
    Validate model and environment action space compatibility.
    
    Args:
        model: Stable-baselines3 model
        env: Gym environment
        
    Raises:
        AssertionError: If action spaces don't match
    """
    
    # Get model action space
    if hasattr(model.policy, 'action_net'):
        if isinstance(model.policy.action_net, list):
            model_actions = model.policy.action_net[-1].out_features
        else:
            model_actions = model.policy.action_net.out_features
    else:
        # Try alternative access patterns
        try:
            model_actions = model.policy.action_dist.action_dim
        except:
            raise ValueError("Cannot determine model action space")
    
    # Get environment action space
    env_actions = env.action_space.n
    
    # Validate compatibility
    assert model_actions == env_actions, \
        f"Model/Env action-space mismatch! Model: {model_actions}, Env: {env_actions}"
    
    print(f"✅ Action space validation passed: {model_actions} actions")
    
    return True

def validate_action_space_integrity(model_path, env):
    """
    Comprehensive action space validation.
    
    Args:
        model_path: Path to model file
        env: Environment instance
        
    Returns:
        bool: True if validation passes
    """
    
    print("🛡️ ACTION SPACE VALIDATION")
    print("=" * 30)
    
    try:
        from stable_baselines3 import PPO
        
        # Load model
        model = PPO.load(model_path, device="cpu")
        print(f"✅ Model loaded: {model_path}")
        
        # Validate compatibility
        assert_action_compat(model, env)
        
        # Additional checks
        env_actions = env.action_space.n
        print(f"✅ Environment actions: {env_actions}")
        
        if env_actions == 5:
            print("✅ Using 5-action system (fixed)")
            expected_actions = ["Buy A", "Sell A", "Buy B", "Sell B", "Hold Both"]
            for i, action_name in enumerate(expected_actions):
                print(f"   {i}: {action_name}")
        elif env_actions == 9:
            print("⚠️ Using 9-action system (legacy)")
        else:
            raise ValueError(f"Unexpected action space size: {env_actions}")
        
        print("✅ All validation checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🛡️ Action Space Validator - Standalone Test")
    print("This module provides validation functions for training scripts.")