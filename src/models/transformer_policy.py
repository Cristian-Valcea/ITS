#!/usr/bin/env python3
"""
🧠 TRANSFORMER POLICY WITH CROSS-TICKER ATTENTION
Shared MLP → Transformer encoder (cross-ticker attention) → LSTM head
Designed for dual-ticker trading with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

logger = logging.getLogger(__name__)

class CrossTickerAttention(nn.Module):
    """Multi-head attention for cross-ticker feature interaction"""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Multi-head attention layers
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, nvda_features: torch.Tensor, msft_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-ticker attention
        
        Args:
            nvda_features: [batch_size, d_model] NVDA features
            msft_features: [batch_size, d_model] MSFT features
            
        Returns:
            (attended_nvda, attended_msft): Enhanced features with cross-attention
        """
        batch_size = nvda_features.size(0)
        
        # Stack features for attention: [batch_size, 2, d_model]
        combined_features = torch.stack([nvda_features, msft_features], dim=1)
        
        # Compute Q, K, V
        Q = self.query(combined_features)  # [batch_size, 2, d_model]
        K = self.key(combined_features)
        V = self.value(combined_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 2, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, 2, head_dim]
        K = K.view(batch_size, 2, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 2, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [batch_size, n_heads, 2, 2]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)  # [batch_size, n_heads, 2, head_dim]
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 2, self.d_model)
        attended = self.out_proj(attended)  # [batch_size, 2, d_model]
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + combined_features)
        
        # Split back to individual assets
        attended_nvda = attended[:, 0, :]  # [batch_size, d_model]
        attended_msft = attended[:, 1, :]  # [batch_size, d_model]
        
        return attended_nvda, attended_msft

class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple attention layers"""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        feedforward_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            CrossTickerAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Feedforward networks
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, feedforward_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
    def forward(self, nvda_features: torch.Tensor, msft_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformer encoding with cross-ticker attention
        
        Args:
            nvda_features: [batch_size, d_model] NVDA features
            msft_features: [batch_size, d_model] MSFT features
            
        Returns:
            (encoded_nvda, encoded_msft): Transformer-encoded features
        """
        
        for i in range(self.n_layers):
            # Cross-ticker attention
            attended_nvda, attended_msft = self.attention_layers[i](nvda_features, msft_features)
            
            # Feedforward with residual connection (apply to each asset separately)
            ff_nvda = self.feedforward[i](attended_nvda)
            ff_msft = self.feedforward[i](attended_msft)
            
            nvda_features = self.layer_norms[i](ff_nvda + attended_nvda)
            msft_features = self.layer_norms[i](ff_msft + attended_msft)
        
        return nvda_features, msft_features

class DualTickerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for dual-ticker observations
    Shared MLP → Transformer encoder → Feature combination
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        shared_mlp_dim: int = 128,
        transformer_layers: int = 3,
        attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        # Observation space analysis
        # Expected: [12 NVDA features, 1 NVDA position, 12 MSFT features, 1 MSFT position] = 26 dim
        obs_dim = observation_space.shape[0]
        
        if obs_dim != 26:
            logger.warning(f"Expected 26-dim observation space, got {obs_dim}")
        
        # Feature dimensions per asset (12 microstructural features)
        self.nvda_feature_dim = 12
        self.msft_feature_dim = 12
        self.position_dim = 1
        
        # Shared MLP for individual asset features
        self.nvda_mlp = nn.Sequential(
            nn.Linear(self.nvda_feature_dim, shared_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_mlp_dim, shared_mlp_dim),
            nn.ReLU()
        )
        
        self.msft_mlp = nn.Sequential(
            nn.Linear(self.msft_feature_dim, shared_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_mlp_dim, shared_mlp_dim),
            nn.ReLU()
        )
        
        # Position encoders
        self.nvda_position_encoder = nn.Linear(self.position_dim, shared_mlp_dim // 4)
        self.msft_position_encoder = nn.Linear(self.position_dim, shared_mlp_dim // 4)
        
        # Transformer encoder for cross-ticker attention
        self.transformer = TransformerEncoder(
            d_model=shared_mlp_dim,
            n_heads=attention_heads,
            n_layers=transformer_layers,
            feedforward_dim=shared_mlp_dim * 2,
            dropout=dropout
        )
        
        # Final combination layer
        combined_dim = shared_mlp_dim * 2 + (shared_mlp_dim // 4) * 2  # NVDA + MSFT features + positions
        self.final_projection = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )
        
        logger.info(f"🧠 DualTickerFeatureExtractor initialized:")
        logger.info(f"   📊 Observation dim: {obs_dim}")
        logger.info(f"   🔄 Shared MLP dim: {shared_mlp_dim}")
        logger.info(f"   🎯 Transformer layers: {transformer_layers}")
        logger.info(f"   🎯 Attention heads: {attention_heads}")
        logger.info(f"   📈 Output features: {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from dual-ticker observations
        
        Args:
            observations: [batch_size, 26] observations
            
        Returns:
            features: [batch_size, features_dim] extracted features
        """
        batch_size = observations.size(0)
        
        # Split observations into components
        # [12 NVDA features, 1 NVDA position, 12 MSFT features, 1 MSFT position]
        nvda_features = observations[:, :self.nvda_feature_dim]  # [batch_size, 12]
        nvda_position = observations[:, self.nvda_feature_dim:self.nvda_feature_dim + 1]  # [batch_size, 1]
        
        msft_start = self.nvda_feature_dim + self.position_dim
        msft_features = observations[:, msft_start:msft_start + self.msft_feature_dim]  # [batch_size, 12]
        msft_position = observations[:, msft_start + self.msft_feature_dim:]  # [batch_size, 1]
        
        # Process through shared MLPs
        nvda_mlp_out = self.nvda_mlp(nvda_features)  # [batch_size, shared_mlp_dim]
        msft_mlp_out = self.msft_mlp(msft_features)  # [batch_size, shared_mlp_dim]
        
        # Encode positions
        nvda_pos_encoded = self.nvda_position_encoder(nvda_position)  # [batch_size, shared_mlp_dim//4]
        msft_pos_encoded = self.msft_position_encoder(msft_position)  # [batch_size, shared_mlp_dim//4]
        
        # Apply transformer with cross-ticker attention
        nvda_attended, msft_attended = self.transformer(nvda_mlp_out, msft_mlp_out)
        
        # Combine all features
        combined_features = torch.cat([
            nvda_attended, msft_attended,  # Transformer outputs
            nvda_pos_encoded, msft_pos_encoded  # Position encodings
        ], dim=1)  # [batch_size, shared_mlp_dim * 2 + shared_mlp_dim//4 * 2]
        
        # Final projection
        final_features = self.final_projection(combined_features)  # [batch_size, features_dim]
        
        return final_features

class TransformerDualTickerPolicy(ActorCriticPolicy):
    """
    Custom policy with Transformer feature extraction + LSTM memory
    Architecture: Shared MLP → Transformer encoder → LSTM head → Actor/Critic
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # Transformer parameters
        features_dim: int = 256,
        shared_mlp_dim: int = 128,
        transformer_layers: int = 3,
        attention_heads: int = 8,
        # LSTM parameters
        lstm_hidden_size: int = 128,
        # Standard parameters
        net_arch: Optional[Dict[str, Any]] = None,
        activation_fn = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class = DualTickerFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        # Set up features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        features_extractor_kwargs.update({
            'features_dim': features_dim,
            'shared_mlp_dim': shared_mlp_dim,
            'transformer_layers': transformer_layers,
            'attention_heads': attention_heads
        })
        
        # Set up network architecture for LSTM
        if net_arch is None:
            net_arch = {
                'pi': [lstm_hidden_size],  # LSTM for policy
                'vf': [lstm_hidden_size]   # LSTM for value function
            }
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        
        logger.info(f"🧠 TransformerDualTickerPolicy initialized:")
        logger.info(f"   🎯 Action space: {action_space}")
        logger.info(f"   📊 Features dim: {features_dim}")
        logger.info(f"   🧠 LSTM hidden: {lstm_hidden_size}")

def create_transformer_policy(
    observation_space,
    action_space,
    lr_schedule,
    config: Dict[str, Any]
) -> TransformerDualTickerPolicy:
    """
    Factory function to create transformer policy from config
    
    Args:
        observation_space: Gym observation space
        action_space: Gym action space  
        lr_schedule: Learning rate schedule function
        config: Configuration dictionary
        
    Returns:
        TransformerDualTickerPolicy: Configured policy
    """
    
    return TransformerDualTickerPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        features_dim=config.get('features_dim', 256),
        shared_mlp_dim=config.get('shared_mlp_dim', 128),
        transformer_layers=config.get('transformer_layers', 3),
        attention_heads=config.get('attention_heads', 8),
        lstm_hidden_size=config.get('lstm_hidden_size', 128),
        features_extractor_kwargs=config.get('features_extractor_kwargs', {})
    )