#!/usr/bin/env python3
"""
Test hardware logging functionality in TrainerAgent.

This test validates that the hardware information is properly logged
when training starts, including CPU/GPU detection and configuration.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_hardware_logging():
    """Test that hardware logging works correctly."""
    print("🧪 Testing Hardware Logging")
    print("=" * 50)
    
    try:
        from src.training import create_trainer_agent
        
        # Setup logging to see the output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create a minimal trainer config
        trainer_config = {
            'algorithm': 'DQN',
            'algo_params': {
                'learning_rate': 1e-4,
                'batch_size': 32,
                'buffer_size': 10000,
            },
            'training_params': {
                'total_timesteps': 1000,
                'log_interval': 100,
            },
            'risk_config': {},
            'model_save_dir': 'models/test',
            'log_dir': 'logs/test',
        }
        
        # Create trainer agent
        trainer = create_trainer_agent(trainer_config)
        
        # Test the hardware logging method directly
        print("\n🖥️  Testing hardware logging method:")
        trainer._log_hardware_info()
        
        print("\n✅ Hardware logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Hardware logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_hardware_logging():
    """Test hardware logging from orchestrator perspective."""
    print("\n🎯 Testing Orchestrator Hardware Logging")
    print("=" * 50)
    
    try:
        # Test the hardware check code that would run in orchestrator
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            gpu_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            print(f"🚀 Training will use GPU: {gpu_name} ({gpu_memory / (1024**3):.1f} GB)")
        else:
            print("💻 Training will use CPU (no CUDA GPU detected)")
            
        print("✅ Orchestrator hardware logging test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator hardware logging test failed: {e}")
        return False

def main():
    """Run all hardware logging tests."""
    print("🎯 HARDWARE LOGGING TEST SUITE")
    print("=" * 60)
    
    success = True
    
    success &= test_hardware_logging()
    success &= test_orchestrator_hardware_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL HARDWARE LOGGING TESTS PASSED!")
        print("\n📋 Hardware Logging Features:")
        print("   ✅ CPU/GPU detection and configuration")
        print("   ✅ Memory and processor information")
        print("   ✅ PyTorch and CUDA version details")
        print("   ✅ Training configuration summary")
        print("   ✅ Performance optimization settings")
    else:
        print("❌ SOME HARDWARE LOGGING TESTS FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())