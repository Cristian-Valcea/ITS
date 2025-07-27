# src/features/config_validator.py
import logging
from typing import Dict, List, Any, Optional, Tuple
from .feature_registry import get_global_registry


class ConfigValidator:
    """
    Validates feature engineering configurations.
    Ensures configurations are complete, consistent, and compatible.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.registry = get_global_registry()
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a complete feature engineering configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required top-level keys
        if 'features' not in config:
            errors.append("Missing required 'features' key")
            return False, errors
        
        features = config['features']
        if not isinstance(features, list):
            errors.append("'features' must be a list")
            return False, errors
        
        # Validate each feature type
        for feature_type in features:
            feature_errors = self._validate_feature_type(feature_type, config)
            errors.extend(feature_errors)
        
        # Validate data processing configuration
        processing_errors = self._validate_processing_config(config)
        errors.extend(processing_errors)
        
        # Check for configuration conflicts
        conflict_errors = self._check_conflicts(config)
        errors.extend(conflict_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_feature_type(self, feature_type: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for a specific feature type."""
        errors = []
        
        # Check if feature type is registered
        if not self.registry.get_calculator_class(feature_type):
            errors.append(f"Unknown feature type: {feature_type}")
            return errors
        
        # Check if feature-specific config exists
        feature_config_key = feature_type.lower()
        if feature_config_key not in config:
            errors.append(f"Missing configuration for feature type: {feature_type}")
            return errors
        
        # Validate feature-specific configuration
        feature_config = config[feature_config_key]
        if not self.registry.validate_config(feature_type, feature_config):
            errors.append(f"Invalid configuration for feature type: {feature_type}")
        
        return errors
    
    def _validate_processing_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate data processing configuration."""
        errors = []
        
        # Validate lookback window
        lookback_window = config.get('lookback_window', 1)
        if not isinstance(lookback_window, int) or lookback_window < 1:
            errors.append("'lookback_window' must be a positive integer")
        
        # Validate feature columns to scale
        feature_cols_to_scale = config.get('feature_cols_to_scale', [])
        if not isinstance(feature_cols_to_scale, list):
            errors.append("'feature_cols_to_scale' must be a list")
        
        # Validate observation feature columns
        obs_feature_cols = config.get('observation_feature_cols', [])
        if not isinstance(obs_feature_cols, list):
            errors.append("'observation_feature_cols' must be a list")
        
        return errors
    
    def _check_conflicts(self, config: Dict[str, Any]) -> List[str]:
        """Check for configuration conflicts."""
        errors = []
        
        # Check if observation columns are subset of scalable columns + unscaled columns
        feature_cols_to_scale = set(config.get('feature_cols_to_scale', []))
        obs_feature_cols = set(config.get('observation_feature_cols', []))
        
        # Get all possible feature names from calculators
        all_possible_features = set()
        for feature_type in config.get('features', []):
            calculator_class = self.registry.get_calculator_class(feature_type)
            if calculator_class:
                try:
                    feature_config = config.get(feature_type.lower(), {})
                    dummy_calc = calculator_class(config=feature_config)
                    all_possible_features.update(dummy_calc.get_feature_names())
                except:
                    pass  # Skip if can't create dummy instance
        
        # Add common unscaled columns
        unscaled_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        all_possible_features.update(unscaled_cols)
        
        # Check if observation columns are valid
        invalid_obs_cols = obs_feature_cols - all_possible_features
        if invalid_obs_cols:
            errors.append(f"Invalid observation feature columns: {list(invalid_obs_cols)}")
        
        return errors
    
    def suggest_improvements(self, config: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements for the configuration.
        
        Args:
            config: Configuration to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check if lookback window is reasonable
        lookback_window = config.get('lookback_window', 1)
        if lookback_window > 50:
            suggestions.append(f"Large lookback window ({lookback_window}) may impact performance")
        
        # Check if too many features are being scaled
        feature_cols_to_scale = config.get('feature_cols_to_scale', [])
        if len(feature_cols_to_scale) > 20:
            suggestions.append("Consider reducing number of features to scale for better performance")
        
        # Check for missing commonly useful features
        features = config.get('features', [])
        if 'Time' not in features:
            suggestions.append("Consider adding 'Time' features for better temporal modeling")
        
        if 'RSI' not in features and 'EMA' not in features:
            suggestions.append("Consider adding momentum indicators like RSI or EMA")
        
        # Check for potential data leakage
        obs_feature_cols = config.get('observation_feature_cols', [])
        if 'Close' in obs_feature_cols:
            suggestions.append("Including 'Close' price in observations may cause data leakage")
        
        return suggestions
    
    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the configuration.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            'feature_types': config.get('features', []),
            'lookback_window': config.get('lookback_window', 1),
            'num_scalable_features': len(config.get('feature_cols_to_scale', [])),
            'num_observation_features': len(config.get('observation_feature_cols', [])),
            'estimated_max_lookback': 0
        }
        
        # Calculate estimated max lookback
        max_lookback = 0
        for feature_type in config.get('features', []):
            calculator_class = self.registry.get_calculator_class(feature_type)
            if calculator_class:
                try:
                    feature_config = config.get(feature_type.lower(), {})
                    dummy_calc = calculator_class(config=feature_config)
                    max_lookback = max(max_lookback, dummy_calc.get_max_lookback())
                except:
                    pass
        
        summary['estimated_max_lookback'] = max_lookback
        
        return summary