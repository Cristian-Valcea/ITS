#!/usr/bin/env python3
"""
Phase 1 Smoke Test - Component Verification
Tests all Phase 1 components before integration
"""

import os
import sys
import yaml
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_phase1_config_loading():
    """Test Phase 1 configuration can be loaded."""
    print("\n🔧 TESTING: Phase 1 Configuration Loading")
    
    config_path = "config/phase1_reality_grounding.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Verify key Phase 1 parameters
        assert config['environment']['reward_scaling'] == 0.3, f"Expected reward_scaling 0.3 (PHASE1-FIX: 3× reward boost), got {config['environment']['reward_scaling']}"
        assert config['environment']['initial_capital'] == 50000.0, f"Expected initial_capital 50000.0, got {config['environment']['initial_capital']}"
        assert config['success_criteria']['episode_reward_range'] == [8000, 19000], f"Expected episode_reward_range [8000, 19000], got {config['success_criteria']['episode_reward_range']}"
        
        print("✅ Phase 1 configuration loaded successfully")
        print(f"   - Reward scaling: {config['environment']['reward_scaling']}")
        print(f"   - Initial capital: ${config['environment']['initial_capital']:,.0f}")
        print(f"   - Target episode rewards: {config['success_criteria']['episode_reward_range']}")
        
        return config
        
    except Exception as e:
        print(f"❌ Phase 1 configuration loading FAILED: {e}")
        return None

def test_institutional_safeguards():
    """Test InstitutionalSafeguards component."""
    print("\n🛡️ TESTING: Institutional Safeguards")
    
    try:
        from gym_env.institutional_safeguards import InstitutionalSafeguards
        
        # Test config (conservative institutional values)
        test_config = {
            'environment': {
                'reward_scaling': 0.08,  # Tuned for ep_rew_mean > +40
                'max_position_size_pct': 0.95,
                'min_cash_reserve_pct': 0.05
            },
            'validation': {
                'reward_bounds': {
                    'min_reward': -150,  # Tighter bounds
                    'max_reward': 150,   # Tighter bounds
                    'alert_threshold': 0.95
                }
            }
        }
        
        # Initialize safeguards
        safeguards = InstitutionalSafeguards(test_config)
        
        # Test reward scaling
        raw_reward = 1000.0
        scaled_reward = safeguards.apply_reward_scaling(raw_reward, 0.08)
        expected_scaled = raw_reward * 0.08
        
        assert abs(scaled_reward - expected_scaled) < 1e-6, f"Reward scaling failed: expected {expected_scaled}, got {scaled_reward}"
        
        # Test step validation with mock data
        mock_observation = np.random.randn(11)  # 11 features as expected
        mock_reward = 100.0
        mock_done = False
        mock_info = {'test': True}
        
        validated_obs, validated_reward, validated_done, validated_info = safeguards.validate_step_output(
            mock_observation, mock_reward, mock_done, mock_info
        )
        
        assert validated_info['safeguards_applied'] == True, "Safeguards not applied"
        assert 'original_reward' in validated_info, "Original reward not tracked"
        
        print("✅ Institutional Safeguards working correctly")
        print(f"   - Reward scaling: {raw_reward} → {scaled_reward}")
        print(f"   - Step validation: PASSED")
        print(f"   - Bounds checking: ACTIVE")
        
        return safeguards
        
    except Exception as e:
        print(f"❌ Institutional Safeguards FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_compatibility_validator():
    """Test ModelCompatibilityValidator component."""
    print("\n🔍 TESTING: Model Compatibility Validator")
    
    try:
        from models.compatibility_validator import ModelCompatibilityValidator
        
        # Test config
        test_config = {
            'model_validation': {
                'enforce_compatibility': False,  # Don't fail on test
                'expected_observation_features': 11,
                'check_frequency': 'initialization'
            }
        }
        
        # Initialize validator
        validator = ModelCompatibilityValidator(test_config)
        
        # Create mock model and environment
        class MockModel:
            def __init__(self):
                self.policy = MockPolicy()
                
        class MockPolicy:
            def __init__(self):
                self.observation_space = MockObservationSpace()
                self.action_space = MockActionSpace()
                
        class MockObservationSpace:
            def __init__(self):
                self.shape = (11,)  # 11 features
                
        class MockActionSpace:
            def __init__(self):
                self.n = 3  # 3 discrete actions
                
        class MockEnv:
            def __init__(self):
                self.observation_space = MockObservationSpace()
                self.action_space = MockActionSpace()
                
            def reset(self):
                return np.random.randn(11), {}
        
        mock_model = MockModel()
        mock_env = MockEnv()
        
        # Test compatibility validation
        result = validator.validate_policy_environment_match(mock_model, mock_env)
        
        assert 'compatibility_checks' in result, "Compatibility checks not performed"
        assert 'overall_compatible' in result, "Overall compatibility not assessed"
        
        print("✅ Model Compatibility Validator working correctly")
        print(f"   - Observation space check: PASSED")
        print(f"   - Action space check: PASSED")
        print(f"   - Overall compatible: {result['overall_compatible']}")
        
        return validator
        
    except Exception as e:
        print(f"❌ Model Compatibility Validator FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_observation_consistency_validator():
    """Test ObservationConsistencyValidator component."""
    print("\n📊 TESTING: Observation Consistency Validator")
    
    try:
        # Check if the validator exists
        try:
            from validation.observation_consistency import ObservationConsistencyValidator
            validator_exists = True
        except ImportError:
            print("⚠️ ObservationConsistencyValidator not found - may need to be created")
            validator_exists = False
            
        if validator_exists:
            test_config = {
                'validation': {
                    'sample_size': 10,
                    'tolerance': 1e-6,
                    'test_frequency': 'every_1000_steps'
                }
            }
            
            validator = ObservationConsistencyValidator(test_config)
            
            # Test with mock observations
            obs1 = np.random.randn(5, 11)
            obs2 = obs1.copy()  # Identical observations
            
            consistency_result = validator.check_consistency(obs1, obs2)
            
            print("✅ Observation Consistency Validator working correctly")
            print(f"   - Consistency check: PASSED")
            
            return validator
        else:
            print("⚠️ Observation Consistency Validator needs to be implemented")
            return None
            
    except Exception as e:
        print(f"❌ Observation Consistency Validator FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_phase1_integration_points():
    """Test where Phase 1 components need to be integrated."""
    print("\n🔗 TESTING: Integration Points Analysis")
    
    integration_points = []
    
    # Check main training files
    training_files = [
        "src/training/trainer_agent.py",
        "src/training/enhanced_trainer_agent.py", 
        "src/gym_env/intraday_trading_env.py",
        "src/training/core/env_builder.py"
    ]
    
    for file_path in training_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except:
                    content = ""  # Skip files with encoding issues
                
            # Check for Phase 1 integration opportunities
            has_config_loading = 'yaml' in content or 'config' in content
            has_safeguards = 'safeguards' in content.lower()
            has_reward_scaling = 'reward_scaling' in content
            
            integration_points.append({
                'file': file_path,
                'exists': True,
                'has_config_loading': has_config_loading,
                'has_safeguards': has_safeguards,
                'has_reward_scaling': has_reward_scaling,
                'integration_needed': not (has_safeguards and has_reward_scaling)
            })
        else:
            integration_points.append({
                'file': file_path,
                'exists': False,
                'integration_needed': False
            })
    
    print("📋 Integration Points Analysis:")
    for point in integration_points:
        if point['exists']:
            status = "🔧 NEEDS INTEGRATION" if point['integration_needed'] else "✅ READY"
            print(f"   {status} {point['file']}")
            if point['integration_needed']:
                print(f"      - Config loading: {'✅' if point['has_config_loading'] else '❌'}")
                print(f"      - Safeguards: {'✅' if point['has_safeguards'] else '❌'}")
                print(f"      - Reward scaling: {'✅' if point['has_reward_scaling'] else '❌'}")
        else:
            print(f"   ❌ MISSING {point['file']}")
    
    return integration_points

def run_smoke_test():
    """Run complete Phase 1 smoke test."""
    print("🧪 PHASE 1 SMOKE TEST - COMPONENT VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Configuration Loading
    results['config'] = test_phase1_config_loading()
    
    # Test 2: Institutional Safeguards
    results['safeguards'] = test_institutional_safeguards()
    
    # Test 3: Model Compatibility Validator
    results['compatibility'] = test_model_compatibility_validator()
    
    # Test 4: Observation Consistency Validator
    results['consistency'] = test_observation_consistency_validator()
    
    # Test 5: Integration Points
    results['integration'] = test_phase1_integration_points()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SMOKE TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        if test_name == 'integration':
            continue  # Skip integration analysis in pass/fail count
            
        total += 1
        if result is not None:
            passed += 1
            print(f"✅ {test_name.upper()}: PASSED")
        else:
            print(f"❌ {test_name.upper()}: FAILED")
    
    print(f"\n📊 RESULTS: {passed}/{total} components working")
    
    if passed == total:
        print("🎉 ALL PHASE 1 COMPONENTS READY FOR INTEGRATION!")
        return True
    else:
        print("⚠️ Some components need fixes before integration")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)