"""
Configurable leak validation thresholds
Supports both strict and loose modes for backwards compatibility
"""

import os
from typing import Dict, Any

# Get test mode from environment
LEAK_TEST_MODE = os.getenv('LEAK_TEST_MODE', 'strict').lower()

class LeakValidationConfig:
    """Configuration for leak validation with mode-specific thresholds"""
    
    def __init__(self, mode: str = None):
        self.mode = mode or LEAK_TEST_MODE
        self.config = self._get_config_for_mode(self.mode)
    
    def _get_config_for_mode(self, mode: str) -> Dict[str, Any]:
        """Get configuration parameters for specified mode"""
        
        if mode == 'strict':
            return {
                'correlation_threshold': 0.01,
                'p_value_threshold': 0.01,
                'causality_threshold': 0.05,
                'temporal_lag_min': 1,
                'max_future_correlation': 0.02,
                'min_historical_correlation': 0.1,
                'leak_detection_sensitivity': 'high'
            }
        elif mode == 'loose':
            return {
                'correlation_threshold': 0.05,
                'p_value_threshold': 0.05,
                'causality_threshold': 0.1,
                'temporal_lag_min': 1,
                'max_future_correlation': 0.1,
                'min_historical_correlation': 0.05,
                'leak_detection_sensitivity': 'medium'
            }
        else:
            # Default to strict mode
            return self._get_config_for_mode('strict')
    
    def get_threshold(self, threshold_name: str) -> float:
        """Get specific threshold value"""
        return self.config.get(threshold_name, 0.05)  # Default fallback
    
    def is_strict_mode(self) -> bool:
        """Check if running in strict mode"""
        return self.mode == 'strict'
    
    def is_loose_mode(self) -> bool:
        """Check if running in loose mode"""
        return self.mode == 'loose'

# Global configuration instance
leak_config = LeakValidationConfig()

def get_leak_threshold(threshold_type: str = 'correlation_threshold') -> float:
    """Get leak detection threshold for specified type"""
    return leak_config.get_threshold(threshold_type)

def is_strict_leak_mode() -> bool:
    """Check if leak validation is in strict mode"""
    return leak_config.is_strict_mode()

def configure_leak_validation(mode: str):
    """Configure leak validation mode"""
    global leak_config
    leak_config = LeakValidationConfig(mode)
    print(f"🔧 Leak validation configured for {mode} mode")
    return leak_config

if __name__ == "__main__":
    print("🔧 Leak Validator Configuration")
    print("=" * 40)
    print(f"Current mode: {leak_config.mode}")
    print(f"Configuration: {leak_config.config}")
    print()
    print("Available modes:")
    print("  - strict: High sensitivity leak detection")
    print("  - loose: Backwards compatible thresholds")
    print()
    print("Environment variable: LEAK_TEST_MODE=strict|loose")