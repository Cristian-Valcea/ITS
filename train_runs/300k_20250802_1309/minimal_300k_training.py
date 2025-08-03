#!/usr/bin/env python3
"""
Minimal 300K Training - Simplest possible approach
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cristian/IntradayTrading/ITS')

def main():
    try:
        print("🚀 STARTING MINIMAL 300K MODEL TRAINING")
        print("=" * 60)
        
        # Minimal imports
        from sb3_contrib import RecurrentPPO
        import numpy as np
        
        # Configuration - very conservative
        config = {
            'base_model_path': 'deploy_models/dual_ticker_prod_20250731_step201k_stable.zip',
            'training_steps': 10000,  # Very small chunk
            'save_dir': 'train_runs/300k_20250802_1309/models'
        }
        
        print(f"📦 Base model: {config['base_model_path']}")
        print(f"🎯 Training steps: {config['training_steps']:,} (minimal chunk)")
        
        # Load model WITHOUT environment first
        print('🤖 Loading base model without environment...')
        model = RecurrentPPO.load(config['base_model_path'])
        print('✅ Base model loaded')
        
        # Test model prediction
        print('🧪 Testing model prediction...')
        dummy_obs = np.zeros((1, 26), dtype=np.float32)
        action, state = model.predict(dummy_obs, deterministic=True)
        print(f'✅ Model prediction works: action={action}')
        
        # Create a DUMMY environment for training
        print('🎭 Creating dummy environment...')
        import gymnasium as gym
        from gymnasium import spaces
        
        class DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
                self.action_space = spaces.Discrete(9)
                self.step_count = 0
                
            def reset(self, seed=None, options=None):
                self.step_count = 0
                obs = np.random.randn(26).astype(np.float32)
                return obs, {}
                
            def step(self, action):
                self.step_count += 1
                obs = np.random.randn(26).astype(np.float32)
                reward = np.random.randn() * 0.01  # Small random reward
                done = self.step_count >= 100  # Short episodes
                truncated = False
                return obs, reward, done, truncated, {}
        
        dummy_env = DummyEnv()
        print('✅ Dummy environment created')
        
        # Load model WITH dummy environment
        print('🔄 Reloading model with dummy environment...')
        model = RecurrentPPO.load(config['base_model_path'], env=dummy_env)
        print('✅ Model loaded with dummy environment')
        
        # Minimal callback
        class MinimalCallback:
            def __init__(self):
                self.step_count = 0
                
            def on_step(self):
                self.step_count += 1
                if self.step_count % 1000 == 0:
                    print(f"📈 Minimal progress: {self.step_count:,} steps")
                return True
        
        callback = MinimalCallback()
        
        print('📈 Starting minimal training...')
        print(f"⏰ Start time: {datetime.now()}")
        start_time = time.time()
        
        # Train with minimal setup
        model.learn(
            total_timesteps=config['training_steps'],
            log_interval=50,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        print(f'✅ Minimal training completed in {training_time:.1f} seconds')
        
        # Save model
        os.makedirs(config['save_dir'], exist_ok=True)
        final_model_path = f"{config['save_dir']}/minimal_300k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.save(final_model_path)
        print(f'💾 Minimal model saved: {final_model_path}')
        
        print('🎉 MINIMAL 300K TRAINING COMPLETED!')
        print('💡 If this works, the issue is with the environment/data setup')
        
    except Exception as e:
        print(f"❌ Minimal training failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)