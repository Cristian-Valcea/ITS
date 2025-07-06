# src/training/interfaces/rl_policy.py
"""
Abstract interface for RL policies.
Ensures clean separation between training and execution environments.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging

from ...shared.constants import POLICY_BUNDLE_VERSION, MAX_PREDICTION_LATENCY_US


class RLPolicy(ABC):
    """
    Abstract base class for RL policies.
    
    Provides a clean interface that can be implemented by different RL frameworks
    (SB3, Ray RLlib, custom implementations) while maintaining consistent
    prediction and serialization APIs.
    """
    
    def __init__(self, policy_id: str, version: str = POLICY_BUNDLE_VERSION):
        self.policy_id = policy_id
        self.version = version
        self.logger = logging.getLogger(f"RLPolicy.{policy_id}")
    
    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Predict action given observation.
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, info_dict)
            
        Performance requirement: Must complete in <100µs for production use.
        """
        pass
    
    @abstractmethod
    def save_bundle(self, bundle_path: Path) -> None:
        """
        Save policy as a production-ready bundle.
        
        Bundle structure:
        bundle_path/
        ├── policy.pt      # TorchScript model
        └── metadata.json  # Model metadata and specs
        
        Args:
            bundle_path: Directory to save the bundle
        """
        pass
    
    @classmethod
    @abstractmethod
    def load_bundle(cls, bundle_path: Path) -> 'RLPolicy':
        """
        Load policy from a production bundle.
        
        Args:
            bundle_path: Directory containing the policy bundle
            
        Returns:
            Loaded RLPolicy instance
        """
        pass
    
    def validate_prediction_latency(self, obs: np.ndarray, num_trials: int = 100) -> Dict[str, float]:
        """
        Validate that prediction meets latency SLO.
        
        Args:
            obs: Sample observation for testing
            num_trials: Number of prediction trials to run
            
        Returns:
            Dictionary with latency statistics
        """
        import time
        
        latencies = []
        for _ in range(num_trials):
            start_time = time.perf_counter()
            self.predict(obs, deterministic=True)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
        
        stats = {
            'mean_latency_us': np.mean(latencies),
            'p95_latency_us': np.percentile(latencies, 95),
            'p99_latency_us': np.percentile(latencies, 99),
            'max_latency_us': np.max(latencies),
            'slo_violations': sum(1 for lat in latencies if lat > MAX_PREDICTION_LATENCY_US),
            'slo_violation_rate': sum(1 for lat in latencies if lat > MAX_PREDICTION_LATENCY_US) / len(latencies)
        }
        
        if stats['slo_violation_rate'] > 0.01:  # More than 1% violations
            self.logger.warning(
                f"Policy {self.policy_id} violates latency SLO: "
                f"{stats['slo_violation_rate']:.2%} predictions > {MAX_PREDICTION_LATENCY_US}µs"
            )
        
        return stats
    
    def _create_metadata(self, obs_space_info: Dict[str, Any], 
                        action_space_info: Dict[str, Any],
                        model_hash: str,
                        additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create metadata dictionary for policy bundle."""
        metadata = {
            'version': self.version,
            'policy_id': self.policy_id,
            'bundle_format_version': POLICY_BUNDLE_VERSION,
            'obs_space': obs_space_info,
            'action_space': action_space_info,
            'model_hash': model_hash,
            'created_at': None,  # Will be set by implementation
            'framework': None,   # Will be set by implementation
        }
        
        if additional_info:
            metadata.update(additional_info)
            
        return metadata
    
    def _save_metadata(self, bundle_path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata to bundle directory."""
        metadata_path = bundle_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved policy metadata to {metadata_path}")
    
    def _load_metadata(self, bundle_path: Path) -> Dict[str, Any]:
        """Load metadata from bundle directory."""
        metadata_path = bundle_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


__all__ = ["RLPolicy"]