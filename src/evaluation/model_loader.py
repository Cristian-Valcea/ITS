"""
ModelLoader module for loading and validating trading models.

This module handles:
- Loading Stable-Baselines3 models
- Loading dummy models for testing
- Model validation and error handling
- Model type detection and compatibility checks
"""

import os
import logging
from typing import Optional, Any, Dict

try:
    from stable_baselines3 import DQN, PPO, A2C, SAC
    SB3_MODEL_CLASSES = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}
    SB3_AVAILABLE = True
    
    # Try to import advanced algorithms from sb3-contrib
    try:
        from sb3_contrib import QRDQN, RecurrentPPO
        SB3_MODEL_CLASSES['QR-DQN'] = QRDQN
        SB3_MODEL_CLASSES['QRDQN'] = QRDQN
        SB3_MODEL_CLASSES['RecurrentPPO'] = RecurrentPPO
        SB3_MODEL_CLASSES['RECURRENTPPO'] = RecurrentPPO  # Add uppercase version for compatibility
        SB3_CONTRIB_AVAILABLE = True
    except ImportError:
        SB3_CONTRIB_AVAILABLE = False
        
except ImportError:
    SB3_AVAILABLE = False
    SB3_CONTRIB_AVAILABLE = False
    SB3_MODEL_CLASSES = {}


class DummyEvalModel:
    """
    Dummy model for testing and evaluation when SB3 is not available.
    """
    
    def __init__(self, path_loaded_from: str):
        """
        Initialize dummy model.
        
        Args:
            path_loaded_from: Path the model was supposedly loaded from
        """
        self.path = path_loaded_from
        self.logger = logging.getLogger("DummyEvalModel")
        self.env = None
        
    def predict(self, obs, deterministic=True):
        """
        Make a prediction (random action for dummy model).
        
        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        # Use the environment's action_space if available
        if hasattr(self, 'env') and hasattr(self.env, 'action_space'):
            return self.env.action_space.sample(), None
        return 0, None
    
    @classmethod
    def load(cls, path: str, env=None):
        """
        Load dummy model from path.
        
        Args:
            path: Path to load model from
            env: Environment (optional)
            
        Returns:
            DummyEvalModel instance
            
        Raises:
            FileNotFoundError: If dummy model file not found
        """
        logger = logging.getLogger("DummyEvalModel.Load")
        if os.path.exists(path + ".dummy"):
            logger.info(f"DummyEvalModel loaded (simulated) from {path}")
            instance = cls(path)
            if env is not None:
                instance.env = env.evaluation_env if hasattr(env, 'evaluation_env') else env
            return instance
        logger.error(f"Dummy model file {path}.dummy not found.")
        raise FileNotFoundError(f"No dummy model found at {path}.dummy")


class ModelLoader:
    """
    Handles loading and validation of trading models for evaluation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the ModelLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.default_model_load_path = config.get('model_load_path', None)
        self.loaded_model = None
        
        # Log available algorithms for debugging
        if SB3_AVAILABLE:
            available_algorithms = list(SB3_MODEL_CLASSES.keys())
            self.logger.info(f"Available algorithms: {available_algorithms}")
            if SB3_CONTRIB_AVAILABLE:
                self.logger.info("sb3-contrib algorithms are available (RecurrentPPO, QRDQN)")
            else:
                self.logger.warning("sb3-contrib is not available - RecurrentPPO and QRDQN will not work")
        else:
            self.logger.error("Stable-Baselines3 is not available")
        
    def load_model(self, model_path: str, algorithm_name: str = "DQN", env_context=None) -> bool:
        """
        Load a model for evaluation.
        
        Args:
            model_path: Path to the model file
            algorithm_name: Name of the algorithm (e.g., "DQN")
            env_context: Environment context for dummy models
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        algo_name_upper = algorithm_name.upper()
        
        if not isinstance(model_path, str) or not model_path:
            self.logger.error("Invalid model path provided.")
            self.loaded_model = None
            return False

        # Check for SB3 model files (.zip) or dummy files (.dummy)
        model_zip_exists = os.path.exists(model_path + ".zip")
        model_dummy_exists = os.path.exists(model_path + ".dummy")
        
        if not model_zip_exists and not model_dummy_exists:
            self.logger.error(f"Model file not found at {model_path}.zip or {model_path}.dummy")
            self.loaded_model = None
            return False

        self.logger.info(f"Loading model for evaluation from: {model_path} (Algorithm: {algo_name_upper})")
        
        # Try to load SB3 model first
        if self._try_load_sb3_model(model_path, algo_name_upper):
            return True
        
        # Fall back to dummy model
        if self._try_load_dummy_model(model_path, env_context):
            return True
            
        # Both methods failed
        available_algorithms = list(SB3_MODEL_CLASSES.keys()) if SB3_AVAILABLE else []
        self.logger.error(f"Cannot load model: Algorithm '{algo_name_upper}' not available and no dummy file found at {model_path}.dummy")
        self.logger.error(f"Available algorithms: {available_algorithms}")
        if not SB3_AVAILABLE:
            self.logger.error("Stable-Baselines3 is not available")
        if not SB3_CONTRIB_AVAILABLE:
            self.logger.error("sb3-contrib is not available (required for RecurrentPPO, QRDQN)")
        self.loaded_model = None
        return False
    
    def _try_load_sb3_model(self, model_path: str, algo_name_upper: str) -> bool:
        """
        Try to load a Stable-Baselines3 model.
        
        Args:
            model_path: Path to the model file
            algo_name_upper: Algorithm name in uppercase
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not SB3_AVAILABLE:
            self.logger.debug("SB3 not available, skipping SB3 model loading")
            return False
            
        if algo_name_upper not in SB3_MODEL_CLASSES:
            self.logger.debug(f"Algorithm '{algo_name_upper}' not in available SB3 model classes")
            return False
            
        ModelClass = SB3_MODEL_CLASSES[algo_name_upper]
        try:
            self.logger.info(f"Loading {algo_name_upper} model from {model_path}")
            self.loaded_model = ModelClass.load(model_path, env=None)
            self.logger.info(f"Successfully loaded SB3 model {algo_name_upper} from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading SB3 model {algo_name_upper} from {model_path}: {e}", exc_info=True)
            return False
    
    def _try_load_dummy_model(self, model_path: str, env_context=None) -> bool:
        """
        Try to load a dummy model for testing.
        
        Args:
            model_path: Path to the model file
            env_context: Environment context
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if os.path.exists(model_path + ".dummy"):
            self.logger.warning(f"SB3 model not available. Attempting to load as dummy from {model_path}.dummy")
            try:
                self.loaded_model = DummyEvalModel.load(model_path, env=env_context)
                self.logger.info("Dummy model loaded successfully for evaluation.")
                return True
            except Exception as e:
                self.logger.error(f"Error loading dummy model from {model_path}.dummy: {e}", exc_info=True)
                return False
        return False
    
    def get_loaded_model(self) -> Optional[Any]:
        """
        Get the currently loaded model.
        
        Returns:
            The loaded model or None if no model is loaded
        """
        return self.loaded_model
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            True if a model is loaded, False otherwise
        """
        return self.loaded_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_model_loaded():
            return {"loaded": False, "type": None, "path": None}
        
        model_type = "SB3" if SB3_AVAILABLE and not isinstance(self.loaded_model, DummyEvalModel) else "Dummy"
        model_path = getattr(self.loaded_model, 'path', 'Unknown')
        
        return {
            "loaded": True,
            "type": model_type,
            "path": model_path,
            "class": self.loaded_model.__class__.__name__
        }
    
    def clear_model(self) -> None:
        """Clear the currently loaded model."""
        self.loaded_model = None
        self.logger.info("Model cleared from memory.")