"""
Observation Consistency Validator for Phase 1
Ensures perfect consistency between training and evaluation environments
"""

import numpy as np
import logging
import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ObservationConsistencyValidator:
    """Batch sanity testing for train/eval consistency (addressing team feedback)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('validation', {})
        self.tolerance = self.config.get('tolerance', 1e-6)
        self.sample_size = self.config.get('sample_size', 128)
        self.test_frequency = self.config.get('test_frequency', 'every_1000_steps')
        
        # Tracking
        self.test_history = []
        self.last_test_step = 0
        
        logger.info(f"ObservationConsistencyValidator initialized:")
        logger.info(f"  - Sample size: {self.sample_size}")
        logger.info(f"  - Tolerance: {self.tolerance}")
        logger.info(f"  - Test frequency: {self.test_frequency}")
        
    def should_run_test(self, current_step: int) -> bool:
        """Determine if consistency test should run at current step."""
        
        if self.test_frequency == "every_1000_steps":
            return (current_step - self.last_test_step) >= 1000
        elif self.test_frequency == "initialization":
            return self.last_test_step == 0
        else:
            return False
            
    def run_batch_consistency_test(self, train_env, eval_env, sample_size: int = None) -> Dict[str, Any]:
        """Sample N random env resets, verify identical outputs."""
        
        if sample_size is None:
            sample_size = self.sample_size
            
        logger.info(f"Running batch consistency test with {sample_size} samples...")
        
        consistency_results = []
        
        try:
            for i in range(sample_size):
                # Set identical random seed for numpy
                seed = np.random.randint(0, 1000000)
                np.random.seed(seed)
                
                # Reset both environments
                train_obs = train_env.reset()
                
                # Reset numpy seed again for eval env
                np.random.seed(seed)
                eval_obs = eval_env.reset()
                
                # Handle different observation formats
                if isinstance(train_obs, tuple):
                    train_obs = train_obs[0]  # Extract observation from (obs, info) tuple
                if isinstance(eval_obs, tuple):
                    eval_obs = eval_obs[0]
                    
                # Check shape consistency
                shape_match = train_obs.shape == eval_obs.shape
                
                # Check dtype consistency  
                dtype_match = train_obs.dtype == eval_obs.dtype
                
                # Check value consistency (within tolerance)
                if shape_match and dtype_match:
                    value_diff = np.abs(train_obs - eval_obs)
                    max_diff = np.max(value_diff)
                    value_match = max_diff < self.tolerance
                else:
                    value_match = False
                    max_diff = float('inf')
                    
                consistency_results.append({
                    'sample_id': i,
                    'seed': seed,
                    'shape_match': shape_match,
                    'dtype_match': dtype_match, 
                    'value_match': value_match,
                    'max_difference': max_diff,
                    'train_shape': train_obs.shape,
                    'eval_shape': eval_obs.shape,
                    'train_dtype': str(train_obs.dtype),
                    'eval_dtype': str(eval_obs.dtype)
                })
                
        except Exception as e:
            logger.error(f"Error during consistency test: {e}")
            return {
                'test_passed': False,
                'error': str(e),
                'timestamp': datetime.datetime.utcnow()
            }
            
        # Generate summary report
        total_samples = len(consistency_results)
        passed_samples = sum(1 for r in consistency_results if 
                           r['shape_match'] and r['dtype_match'] and r['value_match'])
        consistency_rate = passed_samples / total_samples if total_samples > 0 else 0
        
        # Detailed failure analysis
        shape_failures = sum(1 for r in consistency_results if not r['shape_match'])
        dtype_failures = sum(1 for r in consistency_results if not r['dtype_match'])
        value_failures = sum(1 for r in consistency_results if not r['value_match'])
        
        test_result = {
            'consistency_rate': consistency_rate,
            'total_samples': total_samples,
            'passed_samples': passed_samples,
            'failed_samples': total_samples - passed_samples,
            'failure_analysis': {
                'shape_failures': shape_failures,
                'dtype_failures': dtype_failures,
                'value_failures': value_failures
            },
            'test_passed': consistency_rate >= 0.99,  # 99% consistency required
            'detailed_results': consistency_results,
            'timestamp': datetime.datetime.utcnow(),
            'tolerance_used': self.tolerance
        }
        
        # Log results
        if test_result['test_passed']:
            logger.info(f"✅ Consistency test PASSED: {consistency_rate:.3%} success rate")
        else:
            logger.error(f"❌ Consistency test FAILED: {consistency_rate:.3%} success rate")
            logger.error(f"   Shape failures: {shape_failures}")
            logger.error(f"   Dtype failures: {dtype_failures}")
            logger.error(f"   Value failures: {value_failures}")
            
        # Store in history
        self.test_history.append(test_result)
        
        return test_result
        
    def validate_observation_features(self, observation: np.ndarray, expected_features: int) -> Dict[str, Any]:
        """Validate observation has expected number of features."""
        
        if len(observation.shape) < 2:
            return {
                'valid': False,
                'error': f"Expected 2D observation, got shape {observation.shape}"
            }
            
        actual_features = observation.shape[-1]
        
        return {
            'valid': actual_features == expected_features,
            'expected_features': expected_features,
            'actual_features': actual_features,
            'observation_shape': observation.shape,
            'error': None if actual_features == expected_features else 
                    f"Feature mismatch: expected {expected_features}, got {actual_features}"
        }
        
    def check_for_nan_inf(self, observation: np.ndarray) -> Dict[str, Any]:
        """Check observation for NaN or Inf values."""
        
        has_nan = np.any(np.isnan(observation))
        has_inf = np.any(np.isinf(observation))
        
        result = {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'clean': not (has_nan or has_inf)
        }
        
        if has_nan:
            nan_count = np.sum(np.isnan(observation))
            result['nan_count'] = nan_count
            result['nan_positions'] = np.where(np.isnan(observation))
            
        if has_inf:
            inf_count = np.sum(np.isinf(observation))
            result['inf_count'] = inf_count
            result['inf_positions'] = np.where(np.isinf(observation))
            
        return result
        
    def check_consistency(self, obs1: np.ndarray, obs2: np.ndarray) -> Dict[str, Any]:
        """Simple consistency check between two observations."""
        
        # Shape consistency
        shape_match = obs1.shape == obs2.shape
        
        # Dtype consistency  
        dtype_match = obs1.dtype == obs2.dtype
        
        # Value consistency (within tolerance)
        if shape_match and dtype_match:
            value_diff = np.abs(obs1 - obs2)
            max_diff = np.max(value_diff)
            value_match = max_diff < self.tolerance
        else:
            value_match = False
            max_diff = float('inf')
            
        return {
            'consistent': shape_match and dtype_match and value_match,
            'shape_match': shape_match,
            'dtype_match': dtype_match,
            'value_match': value_match,
            'max_difference': max_diff,
            'tolerance': self.tolerance
        }
        
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate comprehensive consistency report from test history."""
        
        if not self.test_history:
            return {'error': 'No test history available'}
            
        # Overall statistics
        total_tests = len(self.test_history)
        passed_tests = sum(1 for test in self.test_history if test['test_passed'])
        overall_pass_rate = passed_tests / total_tests
        
        # Recent performance
        recent_tests = self.test_history[-5:] if len(self.test_history) >= 5 else self.test_history
        recent_pass_rate = sum(1 for test in recent_tests if test['test_passed']) / len(recent_tests)
        
        # Average consistency rates
        avg_consistency_rate = np.mean([test['consistency_rate'] for test in self.test_history])
        
        report = {
            'summary': {
                'total_tests_run': total_tests,
                'tests_passed': passed_tests,
                'overall_pass_rate': overall_pass_rate,
                'recent_pass_rate': recent_pass_rate,
                'average_consistency_rate': avg_consistency_rate
            },
            'latest_test': self.test_history[-1] if self.test_history else None,
            'institutional_compliance': {
                'meets_99_percent_threshold': avg_consistency_rate >= 0.99,
                'consistent_passing': recent_pass_rate >= 0.8,
                'ready_for_production': overall_pass_rate >= 0.95 and avg_consistency_rate >= 0.99
            },
            'generated_at': datetime.datetime.utcnow()
        }
        
        return report


def create_validator_from_config(config_path: str) -> ObservationConsistencyValidator:
    """Factory function to create validator from config file."""
    
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return ObservationConsistencyValidator(config)


if __name__ == "__main__":
    # Simple test of the validator
    logging.basicConfig(level=logging.INFO)
    
    # Mock config for testing
    test_config = {
        'validation': {
            'sample_size': 10,
            'tolerance': 1e-6,
            'test_frequency': 'every_1000_steps'
        }
    }
    
    validator = ObservationConsistencyValidator(test_config)
    print("✅ ObservationConsistencyValidator test initialization successful")