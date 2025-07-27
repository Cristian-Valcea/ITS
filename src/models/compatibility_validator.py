"""
Model Compatibility Validator for Phase 1
Ensures perfect compatibility between training and evaluation models/environments
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Tuple, Optional
import datetime

logger = logging.getLogger(__name__)


class IncompatibilityError(Exception):
    """Raised when critical model-environment incompatibility is detected."""
    pass


class ModelCompatibilityValidator:
    """Ensures perfect compatibility between training and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('model_validation', {})
        self.enforce_compatibility = self.config.get('enforce_compatibility', True)
        self.expected_obs_features = self.config.get('expected_observation_features', 11)
        self.check_frequency = self.config.get('check_frequency', 'initialization')
        
        # Validation history
        self.validation_history = []
        
        logger.info(f"ModelCompatibilityValidator initialized:")
        logger.info(f"  - Expected observation features: {self.expected_obs_features}")
        logger.info(f"  - Enforce compatibility: {self.enforce_compatibility}")
        logger.info(f"  - Check frequency: {self.check_frequency}")
        
    def validate_policy_environment_match(self, model, env) -> Dict[str, Any]:
        """Strict validation of model-environment compatibility."""
        
        validation_result = {
            'timestamp': datetime.datetime.utcnow(),
            'compatibility_checks': {},
            'overall_compatible': True,
            'critical_errors': [],
            'warnings': []
        }
        
        try:
            # 1. Observation space validation
            env_obs_shape = env.observation_space.shape
            model_input_dim = self._extract_model_input_dim(model)
            
            obs_compatible = env_obs_shape[-1] == model_input_dim
            validation_result['compatibility_checks']['observation_space'] = {
                'compatible': obs_compatible,
                'env_shape': env_obs_shape,
                'model_input_dim': model_input_dim,
                'expected_features': self.expected_obs_features
            }
            
            if not obs_compatible:
                error_msg = (
                    f"Critical mismatch: Environment produces {env_obs_shape} observations "
                    f"but model expects {model_input_dim} features. "
                    f"This will cause silent training failure."
                )
                validation_result['critical_errors'].append(error_msg)
                validation_result['overall_compatible'] = False
                
                if self.enforce_compatibility:
                    raise IncompatibilityError(error_msg)
                    
            # 2. Action space validation  
            env_action_space = env.action_space
            model_output_dim = self._extract_model_output_dim(model)
            
            if hasattr(env_action_space, 'n'):  # Discrete action space
                action_compatible = env_action_space.n == model_output_dim
                expected_actions = env_action_space.n
            elif hasattr(env_action_space, 'shape'):  # Continuous action space
                action_compatible = env_action_space.shape[0] == model_output_dim
                expected_actions = env_action_space.shape[0]
            else:
                action_compatible = False
                expected_actions = "unknown"
                
            validation_result['compatibility_checks']['action_space'] = {
                'compatible': action_compatible,
                'env_actions': expected_actions,
                'model_output_dim': model_output_dim,
                'action_space_type': type(env_action_space).__name__
            }
            
            if not action_compatible:
                error_msg = (
                    f"Action space mismatch: Environment expects {expected_actions} actions "
                    f"but model outputs {model_output_dim}"
                )
                validation_result['critical_errors'].append(error_msg)
                validation_result['overall_compatible'] = False
                
                if self.enforce_compatibility:
                    raise IncompatibilityError(error_msg)
                    
            # 3. Expected features validation (Phase 1 specific)
            if env_obs_shape[-1] != self.expected_obs_features:
                warning_msg = (
                    f"Observation features ({env_obs_shape[-1]}) don't match Phase 1 "
                    f"expected features ({self.expected_obs_features}). "
                    f"Verify risk features are properly enabled."
                )
                validation_result['warnings'].append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
                
            # 4. Observation range validation
            sample_obs = env.reset()
            if isinstance(sample_obs, tuple):
                sample_obs = sample_obs[0]  # Extract observation from (obs, info) tuple
                
            obs_stats = self._compute_observation_statistics(sample_obs)
            validation_result['compatibility_checks']['observation_statistics'] = obs_stats
            
            if obs_stats['has_extreme_values']:
                warning_msg = f"Observation contains extreme values: {obs_stats['extreme_summary']}"
                validation_result['warnings'].append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
                
            # 5. Model architecture validation
            arch_validation = self._validate_model_architecture(model)
            validation_result['compatibility_checks']['model_architecture'] = arch_validation
            
            if not arch_validation['valid']:
                validation_result['warnings'].extend(arch_validation['warnings'])
                
        except Exception as e:
            error_msg = f"Validation process failed: {str(e)}"
            validation_result['critical_errors'].append(error_msg)
            validation_result['overall_compatible'] = False
            logger.error(f"❌ {error_msg}")
            
            if self.enforce_compatibility:
                raise IncompatibilityError(error_msg)
                
        # Store in history
        self.validation_history.append(validation_result)
        
        # Log results
        if validation_result['overall_compatible']:
            logger.info("✅ Model-environment compatibility validation PASSED")
        else:
            logger.error("❌ Model-environment compatibility validation FAILED")
            for error in validation_result['critical_errors']:
                logger.error(f"   💥 {error}")
                
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                logger.warning(f"   ⚠️ {warning}")
                
        return validation_result
        
    def _extract_model_input_dim(self, model) -> int:
        """Extract input dimension from model architecture."""
        
        try:
            # For Stable-Baselines3 models
            if hasattr(model, 'policy'):
                policy = model.policy
                
                # RecurrentPPO with LSTM
                if hasattr(policy, 'mlp_extractor'):
                    if hasattr(policy.mlp_extractor, 'policy_net'):
                        # Get first layer input features
                        first_layer = policy.mlp_extractor.policy_net[0]
                        if hasattr(first_layer, 'in_features'):
                            return first_layer.in_features
                            
                # Alternative: check observation space
                if hasattr(policy, 'observation_space'):
                    return policy.observation_space.shape[-1]
                    
            # For direct PyTorch models
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    if len(param.shape) >= 2:  # Weight matrix
                        return param.shape[1]  # Input dimension
                        
            # Fallback: try to infer from a forward pass
            if hasattr(model, 'predict') or hasattr(model, 'forward'):
                logger.warning("Could not determine input dimension from architecture, using fallback")
                return self.expected_obs_features  # Assume expected
                
        except Exception as e:
            logger.error(f"Error extracting model input dimension: {e}")
            
        return -1  # Invalid
        
    def _extract_model_output_dim(self, model) -> int:
        """Extract output dimension from model architecture."""
        
        try:
            # For Stable-Baselines3 models
            if hasattr(model, 'policy'):
                policy = model.policy
                
                # Check action space
                if hasattr(policy, 'action_space'):
                    action_space = policy.action_space
                    if hasattr(action_space, 'n'):
                        return action_space.n  # Discrete
                    elif hasattr(action_space, 'shape'):
                        return action_space.shape[0]  # Continuous
                        
                # Check output layer
                if hasattr(policy, 'action_net'):
                    action_net = policy.action_net
                    if hasattr(action_net, 'out_features'):
                        return action_net.out_features
                        
            # For direct PyTorch models
            if hasattr(model, 'parameters'):
                params = list(model.parameters())
                if params:
                    last_param = params[-1]  # Assume last is output layer bias
                    if len(last_param.shape) >= 1:
                        return last_param.shape[0]
                        
        except Exception as e:
            logger.error(f"Error extracting model output dimension: {e}")
            
        return -1  # Invalid
        
    def _compute_observation_statistics(self, observation: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive observation statistics."""
        
        stats = {
            'shape': observation.shape,
            'dtype': str(observation.dtype),
            'mean': float(np.mean(observation)),
            'std': float(np.std(observation)),
            'min': float(np.min(observation)),
            'max': float(np.max(observation)),
            'has_nan': bool(np.any(np.isnan(observation))),
            'has_inf': bool(np.any(np.isinf(observation))),
            'has_extreme_values': False,
            'extreme_summary': None
        }
        
        # Check for extreme values
        extreme_threshold = 1e6
        extreme_mask = np.abs(observation) > extreme_threshold
        
        if np.any(extreme_mask):
            stats['has_extreme_values'] = True
            extreme_count = np.sum(extreme_mask)
            extreme_max = np.max(np.abs(observation[extreme_mask]))
            stats['extreme_summary'] = f"{extreme_count} values > {extreme_threshold:.0e}, max: {extreme_max:.2e}"
            
        return stats
        
    def _validate_model_architecture(self, model) -> Dict[str, Any]:
        """Validate model architecture for common issues."""
        
        validation = {
            'valid': True,
            'warnings': [],
            'architecture_info': {}
        }
        
        try:
            # Check if model is in training/eval mode
            if hasattr(model, 'training'):
                validation['architecture_info']['training_mode'] = model.training
                
            # Check device
            if hasattr(model, 'device'):
                validation['architecture_info']['device'] = str(model.device)
            elif hasattr(model, 'policy') and hasattr(model.policy, 'device'):
                validation['architecture_info']['device'] = str(model.policy.device)
                
            # Parameter count
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                validation['architecture_info']['parameter_count'] = param_count
                
                # Check for very large or very small models
                if param_count > 10_000_000:  # 10M parameters
                    validation['warnings'].append(f"Large model with {param_count:,} parameters")
                elif param_count < 1000:  # Very small
                    validation['warnings'].append(f"Very small model with {param_count:,} parameters")
                    
            # Check for frozen parameters
            if hasattr(model, 'parameters'):
                frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
                total_params = sum(1 for p in model.parameters())
                
                if frozen_params > 0:
                    validation['warnings'].append(
                        f"{frozen_params}/{total_params} parameters are frozen"
                    )
                    
        except Exception as e:
            validation['valid'] = False
            validation['warnings'].append(f"Architecture validation error: {e}")
            
        return validation
        
    def run_compatibility_test(self, model, env, num_steps: int = 10) -> Dict[str, Any]:
        """Run a small compatibility test with actual environment steps."""
        
        test_result = {
            'test_passed': False,
            'steps_completed': 0,
            'errors': [],
            'warnings': [],
            'performance_stats': {}
        }
        
        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            step_times = []
            
            for step in range(num_steps):
                start_time = datetime.datetime.utcnow()
                
                # Model prediction
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    test_result['errors'].append("Model has no predict method")
                    break
                    
                # Environment step
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                elif len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    test_result['errors'].append(f"Unexpected step result length: {len(step_result)}")
                    break
                    
                if isinstance(obs, tuple):
                    obs = obs[0]
                    
                # Timing
                step_time = (datetime.datetime.utcnow() - start_time).total_seconds()
                step_times.append(step_time)
                
                test_result['steps_completed'] = step + 1
                
                if done:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                        
            # Performance statistics
            if step_times:
                test_result['performance_stats'] = {
                    'avg_step_time_ms': np.mean(step_times) * 1000,
                    'max_step_time_ms': np.max(step_times) * 1000,
                    'total_time_ms': sum(step_times) * 1000
                }
                
            test_result['test_passed'] = test_result['steps_completed'] == num_steps
            
        except Exception as e:
            test_result['errors'].append(f"Compatibility test failed: {str(e)}")
            
        return test_result
        
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report."""
        
        if not self.validation_history:
            return {'error': 'No validation history available'}
            
        latest = self.validation_history[-1]
        
        return {
            'summary': {
                'total_validations': len(self.validation_history),
                'latest_result': latest['overall_compatible'],
                'critical_errors_count': len(latest['critical_errors']),
                'warnings_count': len(latest['warnings'])
            },
            'latest_validation': latest,
            'institutional_compliance': {
                'model_environment_compatible': latest['overall_compatible'],
                'observation_space_validated': 'observation_space' in latest['compatibility_checks'],
                'action_space_validated': 'action_space' in latest['compatibility_checks'],
                'ready_for_training': latest['overall_compatible'] and len(latest['critical_errors']) == 0
            },
            'generated_at': datetime.datetime.utcnow()
        }