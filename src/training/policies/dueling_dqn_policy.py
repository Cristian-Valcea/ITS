# src/training/policies/dueling_dqn_policy.py
"""
Custom Dueling DQN Policy for Stable-Baselines3.

Implements the Dueling Network Architecture that separates state value
and action advantage estimation for improved Q-learning performance.

Reference: "Dueling Network Architectures for Deep Reinforcement Learning"
https://arxiv.org/abs/1511.06581
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Union, Any
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import Schedule
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    DQNPolicy = None
    QNetwork = None
    BaseFeaturesExtractor = None


class DuelingQNetwork(QNetwork):
    """
    Dueling Q-Network that separates state value and action advantage streams.
    
    The network architecture:
    1. Shared feature extraction layers
    2. Split into two streams:
       - Value stream: V(s) - estimates state value
       - Advantage stream: A(s,a) - estimates action advantages
    3. Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: List[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        dueling: bool = True,
    ):
        # Initialize parent without creating q_net (we'll create our own)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch or [256, 256],
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        
        self.dueling = dueling
        self.action_dim = int(action_space.n)
        
        if self.dueling:
            # Create dueling architecture
            self._create_dueling_network()
        else:
            # Fall back to standard DQN architecture
            self._create_standard_network()
    
    def _create_dueling_network(self) -> None:
        """Create the dueling network architecture with separate value and advantage streams."""
        
        # Shared layers (feature extraction is handled by features_extractor)
        shared_layers = []
        input_dim = self.features_dim
        
        # Build shared layers
        for layer_size in self.net_arch[:-1]:  # All but last layer
            shared_layers.extend([
                nn.Linear(input_dim, layer_size),
                self.activation_fn(),
            ])
            input_dim = layer_size
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Value stream: V(s) - single output
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, self.net_arch[-1]),
            self.activation_fn(),
            nn.Linear(self.net_arch[-1], 1)  # Single value output
        )
        
        # Advantage stream: A(s,a) - one output per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, self.net_arch[-1]),
            self.activation_fn(),
            nn.Linear(self.net_arch[-1], self.action_dim)  # One output per action
        )
        
        # Remove the original q_net since we're using dueling architecture
        if hasattr(self, 'q_net'):
            delattr(self, 'q_net')
    
    def _create_standard_network(self) -> None:
        """Create standard DQN network as fallback."""
        # Use the parent's q_net (already created in super().__init__)
        pass
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-values for each action
        """
        if not self.dueling:
            # Standard DQN forward pass
            return self.q_net(self.extract_features(obs, self.features_extractor))
        
        # Dueling network forward pass
        features = self.extract_features(obs, self.features_extractor)
        shared_features = self.shared_net(features)
        
        # Compute value and advantage
        value = self.value_stream(shared_features)  # V(s): [batch_size, 1]
        advantage = self.advantage_stream(shared_features)  # A(s,a): [batch_size, action_dim]
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Subtract mean advantage to ensure identifiability
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        
        return q_values


class DuelingDQNPolicy(DQNPolicy):
    """
    Custom DQN Policy with Dueling Network Architecture.
    
    This policy implements the dueling architecture that separates
    state value estimation from action advantage estimation.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Dict[str, Any] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None,
        dueling: bool = True,
    ):
        self.dueling = dueling
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch or [256, 256],
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        # Store dueling parameter for make_q_net
        self._dueling = dueling
    
    def make_q_net(self) -> DuelingQNetwork:
        """Create the dueling Q-network."""
        net_arch = self._get_constructor_parameters()["net_arch"]
        
        return DuelingQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            dueling=self._dueling,
        )


# Policy registry for easy access
DUELING_POLICIES = {
    "DuelingMultiInputPolicy": DuelingDQNPolicy,
}


def create_dueling_policy_kwargs(dueling: bool = True, net_arch: List[int] = None) -> Dict[str, Any]:
    """
    Create policy_kwargs for dueling DQN.
    
    Args:
        dueling: Whether to enable dueling architecture
        net_arch: Network architecture (default: [256, 256])
        
    Returns:
        Dictionary of policy kwargs
    """
    return {
        "dueling": dueling,
        "net_arch": net_arch or [256, 256],
    }


__all__ = [
    "DuelingQNetwork", 
    "DuelingDQNPolicy", 
    "DUELING_POLICIES",
    "create_dueling_policy_kwargs"
]