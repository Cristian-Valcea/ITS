"""
Policy Export Core Module

Contains model export and serialization logic extracted from TrainerAgent.
This module handles:
- Model bundle saving
- TorchScript export for production deployment
- Metadata generation and validation
- Model versioning and packaging

This is an internal module - use src.training.TrainerAgent for public API.
"""

import logging
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import platform

# TODO: Import statements will be added during extraction phase


def export_torchscript_bundle(
    model: Any,
    run_dir: Path,
    run_name: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Export model as TorchScript bundle for production deployment.
    
    Args:
        model: Trained model to export
        run_dir: Directory to save the bundle
        run_name: Name of the training run
        logger: Optional logger instance
        
    Returns:
        Path to exported bundle or None if export fails
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        logger.info(f"Exporting TorchScript bundle for {run_name}")
        
        # Create a wrapper class for the policy
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
                
            def forward(self, obs):
                # Convert observation to the format expected by the policy
                device = next(self.policy.parameters()).device  # Get policy device
                
                # Handle different observation types
                if isinstance(obs, dict):
                    # Handle Dict observations - convert each value to proper device
                    obs_tensor = {
                        key: value.to(device) if isinstance(value, torch.Tensor) 
                        else torch.tensor(value, dtype=torch.float32, device=device)
                        for key, value in obs.items()
                    }
                elif isinstance(obs, torch.Tensor):
                    obs_tensor = obs.to(device)
                else:
                    obs_tensor = torch.FloatTensor(obs).to(device)
                
                # Get action from policy
                with torch.no_grad():
                    action, _ = self.policy.predict(obs_tensor, deterministic=True)
                    
                # Handle different action space types
                if hasattr(self.policy.action_space, 'dtype'):
                    if self.policy.action_space.dtype == np.float32:
                        return torch.tensor(action, dtype=torch.float32, device=device)
                    else:
                        return torch.tensor(action, dtype=torch.long, device=device)
                else:
                    # Default to long for discrete actions
                    return torch.tensor(action, dtype=torch.long, device=device)
        
        # Wrap the policy
        policy_wrapper = PolicyWrapper(model.policy)
        
        # Create example input for tracing
        obs_space = model.observation_space
        device = next(policy_wrapper.policy.parameters()).device  # Get policy device
        
        # Handle different observation space types
        try:
            if hasattr(obs_space, 'sample'):
                # Use observation_space.sample() for robust handling of Dict/Box/Discrete spaces
                example_obs_np = obs_space.sample()
                
                # Convert to tensor based on observation type
                if isinstance(example_obs_np, dict):
                    # Handle Dict observation spaces
                    example_obs = {
                        key: torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
                        for key, value in example_obs_np.items()
                    }
                elif isinstance(example_obs_np, np.ndarray):
                    # Handle Box observation spaces
                    example_obs = torch.tensor(example_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    # Handle discrete or other spaces
                    example_obs = torch.tensor([example_obs_np], dtype=torch.float32, device=device)
                    
            elif hasattr(obs_space, 'shape'):
                # Fallback: use shape for Box spaces
                example_obs = torch.randn(1, *obs_space.shape, device=device)
            else:
                # Final fallback for unknown spaces
                logger.warning("Unknown observation space type, using default size")
                example_obs = torch.randn(1, 10, device=device)
                
        except Exception as obs_error:
            logger.warning(f"Failed to create example observation: {obs_error}, using fallback")
            # Safe fallback
            example_obs = torch.randn(1, 10, device=device)
        
        # Trace the model
        try:
            traced_policy = torch.jit.trace(policy_wrapper, example_obs)
            
            # Save TorchScript model
            torchscript_path = run_dir / f"{run_name}_torchscript.pt"
            traced_policy.save(str(torchscript_path))
            
            logger.info(f"TorchScript model saved to {torchscript_path}")
            
            # Test the traced model
            test_output = traced_policy(example_obs)
            logger.info(f"TorchScript model test successful, output shape: {test_output.shape}")
            
            return torchscript_path
            
        except Exception as trace_error:
            logger.warning(f"TorchScript tracing failed: {trace_error}")
            logger.info("Attempting to save policy state dict instead")
            
            # Fallback: save policy state dict
            policy_path = run_dir / f"{run_name}_policy.pt"
            torch.save(model.policy.state_dict(), str(policy_path))
            
            logger.info(f"Policy state dict saved to {policy_path}")
            return policy_path
        
    except Exception as e:
        logger.error(f"Failed to export TorchScript bundle: {e}")
        return None


def write_model_metadata(
    bundle_dir: Path,
    model: Any,
    config: Dict[str, Any],
    training_stats: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Write model metadata to the bundle directory.
    
    Args:
        bundle_dir: Directory containing the model bundle
        model: Trained model
        config: Training configuration
        training_stats: Optional training statistics
        logger: Optional logger instance
        
    Returns:
        True if metadata written successfully, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'model_type': config.get('algorithm', 'unknown'),
            'framework': 'stable-baselines3',
            'pytorch_version': torch.__version__,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'training_config': config,
            'training_stats': training_stats or {}
        }
        
        # Add model-specific metadata if available
        if hasattr(model, 'get_parameters'):
            try:
                metadata['model_parameters'] = model.get_parameters()
            except Exception as e:
                logger.warning(f"Could not extract model parameters: {e}")
                
        metadata_path = bundle_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Model metadata written to {metadata_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write model metadata: {e}")
        return False


def save_model_bundle(
    model: Any,
    run_dir: Path,
    run_name: str,
    config: Dict[str, Any],
    training_stats: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Save complete model bundle with all necessary files.
    
    This function will be populated with the actual saving logic
    during the extraction phase.
    
    Args:
        model: Trained model to save
        run_dir: Directory to save the bundle
        run_name: Name of the training run
        config: Training configuration
        training_stats: Optional training statistics
        logger: Optional logger instance
        
    Returns:
        Path to saved bundle or None if save fails
    """
    logger = logger or logging.getLogger(__name__)
    
    # TODO: Extract from _save_model_bundle in trainer_agent.py
    
    try:
        logger.info(f"Saving model bundle for {run_name}")
        
        # Create bundle directory
        bundle_dir = run_dir / "model_bundle"
        bundle_dir.mkdir(exist_ok=True)
        
        # Save the model (placeholder)
        model_path = bundle_dir / "model.zip"
        # TODO: Extract actual model saving logic
        
        # Write metadata
        if not write_model_metadata(bundle_dir, model, config, training_stats, logger):
            logger.warning("Failed to write metadata, but continuing with bundle save")
            
        logger.info(f"Model bundle saved to {bundle_dir}")
        return bundle_dir
        
    except Exception as e:
        logger.error(f"Failed to save model bundle: {e}")
        return None


def validate_model_bundle(
    bundle_path: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that a model bundle contains all required files.
    
    Args:
        bundle_path: Path to the model bundle directory
        logger: Optional logger instance
        
    Returns:
        True if bundle is valid, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    if not bundle_path.exists() or not bundle_path.is_dir():
        logger.error(f"Bundle path does not exist or is not a directory: {bundle_path}")
        return False
        
    required_files = ["metadata.json"]
    optional_files = ["model.zip", "policy.pt", "scaler.pkl"]
    
    # Check required files
    for file_name in required_files:
        file_path = bundle_path / file_name
        if not file_path.exists():
            logger.error(f"Required file missing from bundle: {file_name}")
            return False
            
    # Check for at least one model file
    model_files_present = any(
        (bundle_path / file_name).exists() for file_name in optional_files
    )
    
    if not model_files_present:
        logger.error("No model files found in bundle")
        return False
        
    # Validate metadata
    try:
        metadata_path = bundle_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        required_metadata_keys = ['export_timestamp', 'model_type', 'framework']
        for key in required_metadata_keys:
            if key not in metadata:
                logger.error(f"Required metadata key missing: {key}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to validate metadata: {e}")
        return False
        
    logger.info(f"Model bundle validation passed: {bundle_path}")
    return True


def load_model_from_bundle(
    bundle_path: Path,
    logger: Optional[logging.Logger] = None
) -> Optional[Any]:
    """
    Load a model from a saved bundle.
    
    Args:
        bundle_path: Path to the model bundle directory
        logger: Optional logger instance
        
    Returns:
        Loaded model or None if loading fails
    """
    logger = logger or logging.getLogger(__name__)
    
    if not validate_model_bundle(bundle_path, logger):
        return None
        
    try:
        # Try to load different model formats
        model_zip_path = bundle_path / "model.zip"
        policy_pt_path = bundle_path / "policy.pt"
        
        if model_zip_path.exists():
            # TODO: Load stable-baselines3 model
            logger.info(f"Loading SB3 model from {model_zip_path}")
            return None  # Placeholder
            
        elif policy_pt_path.exists():
            # TODO: Load TorchScript model
            logger.info(f"Loading TorchScript model from {policy_pt_path}")
            return None  # Placeholder
            
        else:
            logger.error("No supported model format found in bundle")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load model from bundle: {e}")
        return None