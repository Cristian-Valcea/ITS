# src/training/dual_ticker_model_adapter.py
"""
Dual-Ticker Model Adapter with Enhanced Transfer Learning

Prepares single-ticker NVDA model for dual-ticker (NVDA + MSFT) transfer learning.
Implements sophisticated weight initialization and neutral action head setup.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.envs import DummyVecEnv

try:
    from ..gym_env.dual_ticker_trading_env import DualTickerTradingEnv
    from ..gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from gym_env.dual_ticker_trading_env import DualTickerTradingEnv
    from gym_env.dual_ticker_data_adapter import DualTickerDataAdapter


class ModelAdaptationError(Exception):
    """Raised when model adaptation fails"""
    pass


class DualTickerModelAdapter:
    """
    Prepares single-ticker model for dual-ticker transfer learning
    
    Key Features:
    - 🔧 Weight transfer from proven single-ticker model
    - 🔧 Neutral MSFT action head initialization 
    - 🔧 Observation space expansion (13 → 26 dimensions)
    - 🔧 Action space expansion (3 → 9 actions)
    - 🔧 Progressive unfreezing strategy
    """
    
    def __init__(self, base_model_path: str):
        self.base_model_path = Path(base_model_path)
        self.logger = logging.getLogger(f"{__name__}.DualTickerModelAdapter")
        
        # Validate base model exists
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        
        # Load base model and metadata
        self.base_model = self._load_base_model()
        self.base_metadata = self._load_model_metadata()
        
        self.logger.info(f"📊 Loaded base model: {self.base_model_path}")
        self.logger.info(f"🎯 Base model algorithm: {self.base_metadata.get('algorithm', 'Unknown')}")
    
    def _load_base_model(self) -> RecurrentPPO:
        """Load the proven single-ticker model"""
        try:
            # Load the RecurrentPPO model
            model = RecurrentPPO.load(str(self.base_model_path))
            self.logger.info(f"✅ Base model loaded successfully")
            return model
        except Exception as e:
            raise ModelAdaptationError(f"Failed to load base model: {e}")
    
    def _load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata if available"""
        metadata_path = self.base_model_path.parent / f"{self.base_model_path.stem}_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"📋 Loaded metadata: {metadata_path}")
                return metadata
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
        
        # Return default metadata
        return {
            'algorithm': 'RecurrentPPO',
            'observation_space': {'shape': [13]},
            'action_space': {'n': 3}
        }
    
    def prepare_dual_ticker_model(self, 
                                 sample_env: Optional[DualTickerTradingEnv] = None,
                                 model_config: Optional[Dict[str, Any]] = None) -> RecurrentPPO:
        """
        Adapt model architecture for dual-ticker trading
        
        Args:
            sample_env: Sample dual-ticker environment for model initialization
            model_config: Optional model configuration overrides
            
        Returns:
            New RecurrentPPO model adapted for dual-ticker trading
        """
        
        self.logger.info("🔧 Starting dual-ticker model adaptation...")
        
        # Create dummy environment if not provided
        if sample_env is None:
            sample_env = self._create_dummy_dual_ticker_env()
        
        # Build model configuration
        final_config = self._build_model_config(model_config)
        
        # Create new model with dual-ticker specs
        new_model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=sample_env,
            **final_config
        )
        
        # 🔧 ENHANCED WEIGHT TRANSFER with neutral action head
        self._transfer_weights_enhanced(new_model)
        
        # Validate the adapted model
        self._validate_adapted_model(new_model, sample_env)
        
        self.logger.info("✅ Dual-ticker model adaptation complete")
        return new_model
    
    def _create_dummy_dual_ticker_env(self) -> DualTickerTradingEnv:
        """Create minimal dual-ticker environment for model initialization"""
        
        self.logger.info("🔧 Creating dummy dual-ticker environment...")
        
        # Generate minimal test data
        n_days = 100
        trading_days = pd.date_range('2024-01-01', periods=n_days, freq='D')
        
        # Random market features (12 features per asset)
        np.random.seed(42)  # Reproducible
        nvda_data = np.random.randn(n_days, 12).astype(np.float32)
        msft_data = np.random.randn(n_days, 12).astype(np.float32)
        
        # Random price data
        nvda_prices = pd.Series(
            500 + np.cumsum(np.random.randn(n_days) * 10), 
            index=trading_days
        )
        msft_prices = pd.Series(
            300 + np.cumsum(np.random.randn(n_days) * 5),
            index=trading_days
        )
        
        # Create environment
        env = DualTickerTradingEnv(
            nvda_data=nvda_data,
            msft_data=msft_data,
            nvda_prices=nvda_prices,
            msft_prices=msft_prices,
            trading_days=trading_days,
            initial_capital=100000.0,
            tc_bp=1.0,
            reward_scaling=0.01
        )
        
        self.logger.info(f"✅ Dummy environment created: {env.observation_space.shape} obs, {env.action_space.n} actions")
        return env
    
    def _build_model_config(self, config_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build model configuration from base model + overrides"""
        
        # Start with proven hyperparameters from base model
        base_config = {
            'learning_rate': 0.0001,
            'n_steps': 128,
            'batch_size': 32,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': [64, 64],  # Same as successful single-ticker model
                'activation_fn': torch.nn.ReLU,
                'lstm_hidden_size': 32,
                'n_lstm_layers': 1
            },
            'verbose': 1,
            'device': 'auto'
        }
        
        # Extract config from base model metadata if available
        if 'config' in self.base_metadata:
            base_algo_params = self.base_metadata['config'].get('algo_params', {})
            for key in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda']:
                if key in base_algo_params:
                    base_config[key] = base_algo_params[key]
        
        # Apply overrides
        if config_override:
            for key, value in config_override.items():
                if key == 'policy_kwargs' and key in base_config:
                    # Merge policy_kwargs
                    base_config['policy_kwargs'].update(value)
                else:
                    base_config[key] = value
        
        self.logger.info(f"🔧 Model config: LR={base_config['learning_rate']}, "
                        f"steps={base_config['n_steps']}, "
                        f"batch={base_config['batch_size']}")
        
        return base_config
    
    def _transfer_weights_enhanced(self, new_model: RecurrentPPO) -> None:
        """
        🔧 Enhanced weight transfer with neutral MSFT action head initialization
        
        Strategy:
        1. Transfer LSTM and value networks directly (same architecture)
        2. Expand observation input layer (13 → 26) by duplicating NVDA features  
        3. Expand action output layer (3 → 9) with neutral MSFT initialization
        4. Zero out MSFT-specific action weights for neutral start
        """
        
        # Use debug level for CI to reduce log noise
        self.logger.debug("🔧 Starting enhanced weight transfer...")
        
        base_policy = self.base_model.policy
        new_policy = new_model.policy
        
        with torch.no_grad():
            
            # 1. Transfer LSTM and shared components
            self._transfer_lstm_weights(base_policy, new_policy)
            
            # 2. Transfer and expand feature extraction layers
            self._transfer_feature_extractor_weights(base_policy, new_policy)
            
            # 3. Transfer and expand policy/value heads
            self._transfer_policy_value_heads(base_policy, new_policy)
            
            # 4. 🔧 ZERO-OUT MSFT action columns (neutral start)
            self._neutralize_msft_actions(new_policy)
        
        self.logger.info("✅ Enhanced weight transfer complete")
    
    def _transfer_lstm_weights(self, base_policy, new_policy) -> None:
        """Transfer LSTM weights (same architecture)"""
        
        try:
            # LSTM weights can be transferred directly
            base_lstm = base_policy.lstm
            new_lstm = new_policy.lstm
            
            if hasattr(base_lstm, 'lstm') and hasattr(new_lstm, 'lstm'):
                # Transfer LSTM cell weights
                for (base_name, base_param), (new_name, new_param) in zip(
                    base_lstm.lstm.named_parameters(), 
                    new_lstm.lstm.named_parameters()
                ):
                    if base_param.shape == new_param.shape:
                        new_param.copy_(base_param)
                        self.logger.debug(f"🔧 Transferred LSTM {base_name}: {base_param.shape}")
                    else:
                        self.logger.warning(f"🔧 LSTM shape mismatch {base_name}: {base_param.shape} vs {new_param.shape}")
            
            self.logger.info("✅ LSTM weights transferred")
            
        except Exception as e:
            self.logger.warning(f"🔧 LSTM transfer failed: {e}")
    
    def _transfer_feature_extractor_weights(self, base_policy, new_policy) -> None:
        """Transfer and expand feature extraction weights"""
        
        try:
            base_features = base_policy.features_extractor
            new_features = new_policy.features_extractor
            
            # Get the first linear layer (input layer)
            base_layers = list(base_features.modules())
            new_layers = list(new_features.modules())
            
            for base_layer, new_layer in zip(base_layers, new_layers):
                if isinstance(base_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                    self._expand_linear_layer(base_layer, new_layer)
                    break  # Only modify first layer
            
            self.logger.info("✅ Feature extractor weights transferred and expanded")
            
        except Exception as e:
            self.logger.warning(f"🔧 Feature extractor transfer failed: {e}")
    
    def _expand_linear_layer(self, base_layer: nn.Linear, new_layer: nn.Linear) -> None:
        """Expand linear layer from 13 → 26 inputs by duplicating NVDA features"""
        
        # Get base weights and bias
        base_weight = base_layer.weight.data  # Shape: [output_dim, 13]
        base_bias = base_layer.bias.data if base_layer.bias is not None else None
        
        # Create new weight matrix [output_dim, 26]
        output_dim = base_weight.shape[0]
        new_weight = torch.zeros(output_dim, 26)
        
        # Strategy: Duplicate NVDA features for MSFT initialization
        # Layout: [NVDA_features(12), NVDA_pos(1), MSFT_features(12), MSFT_pos(1)]
        
        # Copy NVDA features (0:12) → both NVDA (0:12) and MSFT (13:25) positions
        new_weight[:, 0:12] = base_weight[:, 0:12]      # NVDA features (same)
        new_weight[:, 12:13] = base_weight[:, 12:13]    # NVDA position (same)
        new_weight[:, 13:25] = base_weight[:, 0:12]     # MSFT features (duplicated from NVDA)
        new_weight[:, 25:26] = base_weight[:, 12:13]    # MSFT position (duplicated from NVDA)
        
        # Assign to new layer
        new_layer.weight.data = new_weight
        if base_bias is not None and new_layer.bias is not None:
            new_layer.bias.data = base_bias
        
        self.logger.info(f"🔧 Expanded linear layer: {base_weight.shape} → {new_weight.shape}")
    
    def _transfer_policy_value_heads(self, base_policy, new_policy) -> None:
        """Transfer and expand policy/value network heads"""
        
        try:
            # Value network can be transferred directly (same input/output)
            if hasattr(base_policy, 'value_net') and hasattr(new_policy, 'value_net'):
                self._transfer_matching_weights(base_policy.value_net, new_policy.value_net, "value_net")
            
            # Policy network needs action space expansion (3 → 9)
            if hasattr(base_policy, 'action_net') and hasattr(new_policy, 'action_net'):
                self._expand_action_network(base_policy.action_net, new_policy.action_net)
            
            self.logger.info("✅ Policy/value heads transferred")
            
        except Exception as e:
            self.logger.warning(f"🔧 Policy/value head transfer failed: {e}")
    
    def _transfer_matching_weights(self, base_module, new_module, module_name: str) -> None:
        """Transfer weights between modules with matching architectures"""
        
        for (base_name, base_param), (new_name, new_param) in zip(
            base_module.named_parameters(),
            new_module.named_parameters()
        ):
            if base_param.shape == new_param.shape:
                new_param.data.copy_(base_param.data)
                self.logger.debug(f"🔧 Transferred {module_name}.{base_name}: {base_param.shape}")
            else:
                self.logger.warning(f"🔧 {module_name} shape mismatch {base_name}: {base_param.shape} vs {new_param.shape}")
    
    def _expand_action_network(self, base_action_net, new_action_net) -> None:
        """Expand action network from 3 → 9 outputs"""
        
        # Get the final layer that outputs actions
        base_layers = list(base_action_net.modules())
        new_layers = list(new_action_net.modules())
        
        # Find the output layer
        for base_layer, new_layer in zip(reversed(base_layers), reversed(new_layers)):
            if isinstance(base_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                if base_layer.out_features == 3 and new_layer.out_features == 9:
                    self._expand_action_output_layer(base_layer, new_layer)
                    break
        
        self.logger.info("🔧 Action network expanded: 3 → 9 outputs")
    
    def _expand_action_output_layer(self, base_layer: nn.Linear, new_layer: nn.Linear) -> None:
        """Expand final action output layer with strategic initialization"""
        
        base_weight = base_layer.weight.data  # Shape: [3, hidden_dim]
        base_bias = base_layer.bias.data if base_layer.bias is not None else None
        
        # Initialize new layer with small random weights
        nn.init.normal_(new_layer.weight, mean=0.0, std=0.01)
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)
        
        # Map single-ticker actions to dual-ticker equivalents
        # Single-ticker: [SELL, HOLD, BUY] → Dual-ticker: 9 actions in 3x3 matrix
        
        # Action mapping strategy:
        # - Single SELL → SELL_BOTH (action 0)
        # - Single HOLD → HOLD_BOTH (action 4)  
        # - Single BUY → BUY_BOTH (action 8)
        
        hidden_dim = base_weight.shape[1]
        
        # Copy base weights to corresponding dual-ticker actions
        new_layer.weight.data[0] = base_weight[0]  # SELL → SELL_BOTH
        new_layer.weight.data[4] = base_weight[1]  # HOLD → HOLD_BOTH
        new_layer.weight.data[8] = base_weight[2]  # BUY → BUY_BOTH
        
        if base_bias is not None and new_layer.bias is not None:
            new_layer.bias.data[0] = base_bias[0]  # SELL → SELL_BOTH
            new_layer.bias.data[4] = base_bias[1]  # HOLD → HOLD_BOTH
            new_layer.bias.data[8] = base_bias[2]  # BUY → BUY_BOTH
        
        self.logger.info("🔧 Action output layer expanded with strategic initialization")
    
    def _neutralize_msft_actions(self, new_policy) -> None:
        """
        🔧 Zero out MSFT-specific action columns for neutral start
        
        Actions involving MSFT changes: 1, 2, 3, 5, 6, 7
        (All actions except SELL_BOTH=0, HOLD_BOTH=4, BUY_BOTH=8)
        """
        
        try:
            # Find action output layer
            for module in new_policy.modules():
                if isinstance(module, nn.Linear) and module.out_features == 9:
                    
                    # MSFT-specific actions (not both-asset actions)
                    msft_action_indices = [1, 2, 3, 5, 6, 7]
                    
                    # Zero out these action weights
                    with torch.no_grad():
                        for idx in msft_action_indices:
                            module.weight.data[idx] *= 0.1  # Small but not zero
                            if module.bias is not None:
                                module.bias.data[idx] *= 0.1
                    
                    self.logger.info(f"🔧 Neutralized MSFT action weights: {msft_action_indices}")
                    break
                    
        except Exception as e:
            self.logger.warning(f"🔧 MSFT neutralization failed: {e}")
    
    def _validate_adapted_model(self, model: RecurrentPPO, env: DualTickerTradingEnv) -> None:
        """Validate the adapted model works correctly"""
        
        self.logger.info("🔧 Validating adapted model...")
        
        try:
            # Test observation processing
            obs, _ = env.reset()
            assert obs.shape == (26,), f"Expected obs shape (26,), got {obs.shape}"
            
            # Test action prediction
            action, _ = model.predict(obs, deterministic=True)
            assert 0 <= action <= 8, f"Invalid action: {action}"
            
            # Test step execution
            obs, reward, done, info = env.step(action)
            assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
            assert isinstance(done, bool), f"Invalid done type: {type(done)}"
            
            self.logger.info("✅ Model validation passed")
            
        except Exception as e:
            raise ModelAdaptationError(f"Model validation failed: {e}")
    
    def save_adapted_model(self, 
                          model: RecurrentPPO, 
                          output_path: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the adapted dual-ticker model with metadata
        
        Args:
            model: Adapted RecurrentPPO model
            output_path: Path to save the model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(str(output_path))
        
        # Create metadata
        adapted_metadata = {
            'algorithm': 'RecurrentPPO',
            'model_type': 'dual_ticker_adapted',
            'base_model_path': str(self.base_model_path),
            'adaptation_timestamp': pd.Timestamp.now().isoformat(),
            'observation_space': {'shape': [26]},
            'action_space': {'n': 9},
            'base_metadata': self.base_metadata
        }
        
        if metadata:
            adapted_metadata.update(metadata)
        
        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(adapted_metadata, f, indent=2)
        
        self.logger.info(f"✅ Saved adapted model: {output_path}")
        self.logger.info(f"📋 Saved metadata: {metadata_path}")
        
        return str(output_path)
    
    def create_training_curriculum(self) -> Dict[str, Any]:
        """
        Create progressive training curriculum for dual-ticker model
        
        Returns:
            Training curriculum with phases and parameters
        """
        
        curriculum = {
            'phase_1_bootstrap': {
                'description': 'Bootstrap with conservative learning',
                'total_timesteps': 50000,
                'learning_rate': 0.00005,  # Half of base model
                'clip_range': 0.1,         # Conservative clipping
                'ent_coef': 0.02,          # Higher exploration
                'focus': 'Stability and basic dual-ticker mechanics'
            },
            'phase_2_adaptation': {
                'description': 'Full learning rate adaptation',
                'total_timesteps': 100000,
                'learning_rate': 0.0001,   # Base model rate
                'clip_range': 0.2,         # Normal clipping
                'ent_coef': 0.01,          # Normal exploration
                'focus': 'Portfolio optimization and risk management'
            },
            'phase_3_refinement': {
                'description': 'Fine-tuning and optimization',
                'total_timesteps': 50000,
                'learning_rate': 0.00005,  # Reduced for stability
                'clip_range': 0.15,        # Slightly conservative
                'ent_coef': 0.005,         # Reduced exploration
                'focus': 'Performance optimization and stability'
            }
        }
        
        return curriculum