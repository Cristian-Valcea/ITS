#!/usr/bin/env python3
"""
🔧 CREATE 5-ACTION MODEL
Create a new PPO model with 5-action architecture for the fixed environment
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from stable_baselines3 import PPO
    from gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
    from action_space_validator import validate_action_space_integrity
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def create_fresh_5action_model():
    """Create a fresh PPO model with 5-action architecture."""
    
    print("🔧 CREATING FRESH 5-ACTION MODEL")
    print("=" * 40)
    
    # Create 5-action environment
    try:
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        
        feature_data = np.random.randn(5000, 26).astype(np.float32)
        price_data = np.random.randn(5000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(5000)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=400,
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.45,
            hold_bonus_weight=0.020,  # Increased parameter
            verbose=False
        )
        
        print(f"✅ Created 5-action environment")
        print(f"   Action space: {env.action_space.n} actions")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Create fresh PPO model
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=None,
            policy_kwargs=None,
            verbose=1,
            seed=None,
            device="auto",
            _init_setup_model=True
        )
        
        print(f"✅ Created fresh PPO model")
        print(f"   Policy architecture: MlpPolicy")
        print(f"   Action space: {model.action_space.n} actions")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Manual validation since we don't have a saved model yet
    try:
        model_actions = model.policy.action_net.out_features
        env_actions = env.action_space.n
        
        if model_actions != env_actions:
            print(f"❌ Action space mismatch: Model {model_actions}, Env {env_actions}")
            return False
        
        print(f"✅ Action space validation passed: {model_actions} actions")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Save the fresh model as Cycle 7 base
    output_dir = Path("train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model_fresh_5action_base.zip"
    
    try:
        model.save(str(model_path))
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"💾 Fresh 5-action model saved: {model_path.name} ({model_size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False
    
    # Test the model with a few steps
    print("\n🧪 TESTING FRESH MODEL:")
    
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        actions = []
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions.append(action)
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            if done:
                break
        
        print(f"   Test actions: {actions}")
        
        # Check action validity
        valid_actions = [a for a in actions if 0 <= a <= 4]
        invalid_actions = [a for a in actions if a < 0 or a > 4]
        
        if invalid_actions:
            print(f"   ❌ Invalid actions found: {invalid_actions}")
            return False
        else:
            print(f"   ✅ All actions valid (0-4)")
        
        # Check hold rate
        hold_count = sum(1 for a in actions if a == 4)
        hold_rate = hold_count / len(actions) if actions else 0
        print(f"   Hold rate: {hold_rate:.1%} (action 4 count: {hold_count})")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False
    
    print("\n✅ FRESH 5-ACTION MODEL READY")
    print("🚀 Ready for Cycle 7 training")
    
    return str(model_path)

def main():
    """Main function."""
    
    model_path = create_fresh_5action_model()
    
    print("\n" + "=" * 40)
    print("📊 5-ACTION MODEL CREATION:")
    
    if model_path:
        print("✅ SUCCESS: Fresh 5-action model created")
        print(f"📁 Model path: {model_path}")
        print("🚀 Ready to launch Cycle 7 with fixed architecture")
        print("\nNext steps:")
        print("1. Use this fresh model for Cycle 7 training")
        print("2. Run full 6,000 step training")
        print("3. Expect 10-15% hold rate recovery")
    else:
        print("❌ FAILED: Could not create 5-action model")
        print("🔧 Manual intervention required")
    
    return bool(model_path)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)