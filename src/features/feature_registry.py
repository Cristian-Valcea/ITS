# src/features/feature_registry.py
import logging
from typing import Dict, Type, List, Optional, Any
from .base_calculator import BaseFeatureCalculator


class FeatureRegistry:
    """
    Centralized registry for managing feature calculators.
    Provides discovery, validation, and metadata management for feature calculators.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._calculators: Dict[str, Type[BaseFeatureCalculator]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, calculator_class: Type[BaseFeatureCalculator], 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Register a feature calculator.
        
        Args:
            name: Unique name for the calculator
            calculator_class: Calculator class to register
            metadata: Optional metadata about the calculator
        """
        if not issubclass(calculator_class, BaseFeatureCalculator):
            raise ValueError(f"Calculator class must inherit from BaseFeatureCalculator")
            
        if name in self._calculators:
            self.logger.warning(f"Overwriting existing calculator: {name}")
            
        self._calculators[name] = calculator_class
        self._metadata[name] = metadata or {}
        
        self.logger.info(f"Registered calculator: {name}")
    
    def unregister(self, name: str):
        """Unregister a feature calculator."""
        if name in self._calculators:
            del self._calculators[name]
            del self._metadata[name]
            self.logger.info(f"Unregistered calculator: {name}")
        else:
            self.logger.warning(f"Calculator not found for unregistration: {name}")
    
    def get_calculator_class(self, name: str) -> Optional[Type[BaseFeatureCalculator]]:
        """Get calculator class by name."""
        return self._calculators.get(name)
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a calculator."""
        return self._metadata.get(name, {})
    
    def list_calculators(self) -> List[str]:
        """List all registered calculator names."""
        return list(self._calculators.keys())
    
    def create_calculator(self, name: str, config: Dict[str, Any], 
                         logger: Optional[logging.Logger] = None) -> Optional[BaseFeatureCalculator]:
        """
        Create an instance of a registered calculator.
        
        Args:
            name: Calculator name
            config: Configuration for the calculator
            logger: Logger instance
            
        Returns:
            Calculator instance or None if not found
        """
        calculator_class = self.get_calculator_class(name)
        if calculator_class is None:
            self.logger.error(f"Calculator not found: {name}")
            return None
            
        try:
            return calculator_class(config=config, logger=logger)
        except Exception as e:
            self.logger.error(f"Failed to create calculator {name}: {e}")
            return None
    
    def validate_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a calculator.
        
        Args:
            name: Calculator name
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        calculator_class = self.get_calculator_class(name)
        if calculator_class is None:
            return False
            
        # Try to create instance to validate config
        try:
            calculator_class(config=config)
            return True
        except Exception as e:
            self.logger.error(f"Invalid config for {name}: {e}")
            return False
    
    def get_calculator_info(self, name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a calculator.
        
        Args:
            name: Calculator name
            
        Returns:
            Dictionary with calculator information
        """
        calculator_class = self.get_calculator_class(name)
        if calculator_class is None:
            return {}
            
        info = {
            'name': name,
            'class': calculator_class.__name__,
            'module': calculator_class.__module__,
            'docstring': calculator_class.__doc__,
            'metadata': self.get_metadata(name)
        }
        
        # Try to get additional info from a dummy instance
        try:
            dummy_instance = calculator_class(config={})
            info['max_lookback'] = dummy_instance.get_max_lookback()
            info['feature_names'] = dummy_instance.get_feature_names()
        except:
            info['max_lookback'] = 'Unknown'
            info['feature_names'] = 'Unknown'
            
        return info
    
    def discover_calculators(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover and return information about all registered calculators.
        
        Returns:
            Dictionary mapping calculator names to their information
        """
        return {name: self.get_calculator_info(name) for name in self.list_calculators()}


# Global registry instance
_global_registry = FeatureRegistry()

def get_global_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    return _global_registry

def register_calculator(name: str, calculator_class: Type[BaseFeatureCalculator], 
                       metadata: Optional[Dict[str, Any]] = None):
    """Register a calculator in the global registry."""
    _global_registry.register(name, calculator_class, metadata)

def get_calculator_class(name: str) -> Optional[Type[BaseFeatureCalculator]]:
    """Get calculator class from global registry."""
    return _global_registry.get_calculator_class(name)