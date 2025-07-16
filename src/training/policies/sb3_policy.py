# src/training/policies/sb3_policy.py
"""
Stable-Baselines3 policy implementation.
Wraps SB3 models with the RLPolicy interface for clean abstraction.
"""

import torch
import torch.jit
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

try:
    from ..interfaces.rl_policy import RLPolicy
    from ...shared.constants import MODEL_VERSION_FORMAT
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from training.interfaces.rl_policy import RLPolicy
    from shared.constants import MODEL_VERSION_FORMAT

# Import SB3 components
try:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3 import DQN, PPO, A2C, SAC

    SB3_AVAILABLE = True
    SB3_ALGORITHMS = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }
    
    # Import advanced algorithms from sb3-contrib
    try:
        from sb3_contrib import QRDQN
        SB3_ALGORITHMS["QR-DQN"] = QRDQN
        SB3_CONTRIB_AVAILABLE = True
    except ImportError:
        SB3_CONTRIB_AVAILABLE = False
        
except ImportError:
    SB3_AVAILABLE = False
    SB3_CONTRIB_AVAILABLE = False
    BaseAlgorithm = None
    SB3_ALGORITHMS = {}


class SB3Policy(RLPolicy):
    """
    Stable-Baselines3 policy wrapper implementing the RLPolicy interface.

    Provides clean abstraction over SB3 models with TorchScript export
    for production deployment.
    """

    def __init__(self, model: BaseAlgorithm, policy_id: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for SB3Policy")

        if policy_id is None:
            policy_id = f"sb3_{model.__class__.__name__.lower()}_{datetime.now().strftime(MODEL_VERSION_FORMAT)}"

        super().__init__(policy_id)
        self.model = model
        self.algorithm_name = model.__class__.__name__

        # Extract model components for TorchScript export
        self._extract_model_components()

        self.logger.info(f"SB3Policy initialized: {self.algorithm_name} ({policy_id})")

    def _extract_model_components(self):
        """Extract PyTorch components from SB3 model for TorchScript export."""
        try:
            # Get the policy network from the SB3 model
            if hasattr(self.model, "policy"):
                self.policy_net = self.model.policy
            elif hasattr(self.model, "actor"):
                self.policy_net = self.model.actor
            else:
                raise AttributeError("Cannot find policy network in SB3 model")

            # Get observation and action space info
            self.obs_space = self.model.observation_space
            self.action_space = self.model.action_space

        except Exception as e:
            self.logger.error(f"Failed to extract model components: {e}")
            raise

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Predict action using SB3 model.

        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, info_dict)
        """
        try:
            action, _states = self.model.predict(obs, deterministic=deterministic)

            # Convert to int if discrete action space
            if hasattr(self.action_space, "n"):  # Discrete space
                action = int(action)

            info = {
                "algorithm": self.algorithm_name,
                "deterministic": deterministic,
                "policy_id": self.policy_id,
            }

            return action, info

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return safe default action
            default_action = 1 if hasattr(self.action_space, "n") and self.action_space.n > 1 else 0
            return default_action, {"error": str(e)}

    def save_bundle(self, bundle_path: Path) -> None:
        """
        Save policy bundle with both SB3 and TorchScript models.

        Args:
            bundle_path: Directory to save the bundle
        """
        bundle_path = Path(bundle_path)
        bundle_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save original SB3 model for evaluation compatibility
            sb3_model_path = bundle_path / "policy.pt"
            self.model.save(str(sb3_model_path))
            self.logger.info(f"SB3 model saved to {sb3_model_path}")
            
            # Save TorchScript model for production deployment
            torchscript_path = bundle_path / "policy_torchscript.pt"
            self._export_torchscript(torchscript_path)
            self.logger.info(f"TorchScript model saved to {torchscript_path}")
            
            # Use SB3 model path for metadata hash calculation
            policy_path = sb3_model_path

            # Create metadata
            model_hash = self._compute_file_hash(policy_path)
            metadata = self._create_metadata(
                obs_space_info=self._serialize_space(self.obs_space),
                action_space_info=self._serialize_space(self.action_space),
                model_hash=model_hash,
                additional_info={
                    "algorithm": self.algorithm_name,
                    "framework": "stable-baselines3",
                    "created_at": datetime.now().isoformat(),
                    "sb3_version": self._get_sb3_version(),
                    "torch_version": torch.__version__,
                },
            )

            # Save metadata
            self._save_metadata(bundle_path, metadata)

            self.logger.info(f"Policy bundle saved to {bundle_path}")

        except Exception as e:
            self.logger.error(f"Failed to save policy bundle: {e}")
            raise

    def _export_torchscript(self, policy_path: Path) -> None:
        """Export policy network as TorchScript."""
        try:
            # Create a wrapper for the policy network that handles SB3 specifics
            wrapper = SB3PolicyWrapper(self.policy_net, self.obs_space, self.action_space)

            # Create example input for tracing
            if hasattr(self.obs_space, "shape"):
                example_input = torch.randn(1, *self.obs_space.shape)
            else:
                # Handle Box spaces or other types
                example_input = torch.randn(1, self.obs_space.shape[0])

            # Trace the model
            traced_model = torch.jit.trace(wrapper, example_input)

            # Save TorchScript model
            traced_model.save(str(policy_path))

            self.logger.info(f"TorchScript model exported to {policy_path}")

        except Exception as e:
            self.logger.error(f"TorchScript export failed: {e}")
            # Fallback: save the entire SB3 model (less optimal for production)
            self.model.save(str(policy_path.with_suffix(".zip")))
            self.logger.warning(f"Fallback: Saved SB3 model to {policy_path.with_suffix('.zip')}")
            raise

    @classmethod
    def load_bundle(cls, bundle_path: Path) -> "SB3Policy":
        """
        Load policy from bundle.

        Note: This creates a TorchScriptPolicy for production use,
        not a full SB3Policy (which requires the original SB3 model).
        """
        bundle_path = Path(bundle_path)

        # Load metadata
        metadata_path = bundle_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # For production deployment, return a TorchScript-based policy
        return TorchScriptPolicy.load_bundle(bundle_path)

    def _serialize_space(self, space) -> Dict[str, Any]:
        """Serialize gym space to JSON-compatible format."""
        if hasattr(space, "shape"):
            return {
                "type": space.__class__.__name__,
                "shape": list(space.shape),
                "dtype": str(space.dtype),
            }
        elif hasattr(space, "n"):  # Discrete space
            return {
                "type": "Discrete",
                "n": space.n,
            }
        else:
            return {
                "type": space.__class__.__name__,
                "info": str(space),
            }

    def _get_sb3_version(self) -> str:
        """Get Stable-Baselines3 version."""
        try:
            import stable_baselines3

            return stable_baselines3.__version__
        except ImportError:
            return "unknown"


class SB3PolicyWrapper(torch.nn.Module):
    """
    PyTorch wrapper for SB3 policy networks to enable TorchScript export.
    """

    def __init__(self, policy_net, obs_space, action_space):
        super().__init__()
        self.policy_net = policy_net
        self.obs_space = obs_space
        self.action_space = action_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        # Handle different SB3 policy types
        if hasattr(self.policy_net, "forward"):
            # Direct forward pass
            with torch.no_grad():
                actions = self.policy_net.forward(obs, deterministic=True)
                if isinstance(actions, tuple):
                    actions = actions[0]  # Take first element if tuple
                return actions
        else:
            # Fallback for complex policies
            raise NotImplementedError("Policy network type not supported for TorchScript export")


class TorchScriptPolicy(RLPolicy):
    """
    Production policy that loads and runs TorchScript models.
    Optimized for low-latency inference without SB3 dependencies.
    """

    def __init__(self, torchscript_model: torch.jit.ScriptModule, metadata: Dict[str, Any]):
        super().__init__(metadata["policy_id"], metadata["version"])
        self.model = torchscript_model
        self.metadata = metadata

        # Set model to evaluation mode
        self.model.eval()

        self.logger.info(f"TorchScriptPolicy loaded: {self.policy_id}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """Predict using TorchScript model with soft-deadline latency check."""
        import time
        
        try:
            # Convert to tensor
            obs_tensor = torch.from_numpy(obs).float()
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

            # High-precision timing for policy forward pass
            start = time.perf_counter_ns()
            
            # Predict
            with torch.no_grad():
                action_tensor = self.model(obs_tensor)
            
            # Soft-deadline assertion - fail fast on SLA violation
            lat_us = (time.perf_counter_ns() - start) / 1_000
            assert lat_us < 100, f"Inference {lat_us:.1f}µs exceeds SLA"

            # Convert to action
            if action_tensor.dim() > 1:
                action_tensor = action_tensor.squeeze(0)

            # Handle discrete vs continuous actions
            if "Discrete" in self.metadata.get("action_space", {}).get("type", ""):
                action = int(torch.argmax(action_tensor).item())
            else:
                action = action_tensor.numpy()

            info = {
                "policy_id": self.policy_id,
                "deterministic": deterministic,
                "framework": "torchscript",
                "inference_latency_us": lat_us,
            }

            return action, info

        except Exception as e:
            self.logger.error(f"TorchScript prediction failed: {e}")
            # Return safe default
            return 1, {"error": str(e)}

    def save_bundle(self, bundle_path: Path) -> None:
        """TorchScript policies are already in bundle format."""
        raise NotImplementedError("TorchScript policies are loaded from bundles, not saved")

    @classmethod
    def load_bundle(cls, bundle_path: Path) -> "TorchScriptPolicy":
        """Load TorchScript policy from bundle."""
        bundle_path = Path(bundle_path)

        # Load metadata
        metadata_path = bundle_path / "metadata.json"
        with open(metadata_path, "r") as f:
            import json

            metadata = json.load(f)

        # Load TorchScript model
        policy_path = bundle_path / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        model = torch.jit.load(str(policy_path))

        return cls(model, metadata)


__all__ = ["SB3Policy", "TorchScriptPolicy", "SB3_AVAILABLE", "SB3_ALGORITHMS"]
