# src/features/__init__.py
"""
Feature engineering module for technical analysis and market data processing.
"""

from .base_calculator import BaseFeatureCalculator
from .rsi_calculator import RSICalculator
from .ema_calculator import EMACalculator
from .vwap_calculator import VWAPCalculator
from .time_calculator import TimeFeatureCalculator
from .market_impact_calculator import MarketImpactCalculator
from .feature_manager import FeatureManager
from .data_processor import DataProcessor
from .feature_registry import FeatureRegistry, get_global_registry, register_calculator
from .config_validator import ConfigValidator
from .performance_tracker import PerformanceTracker, TimingContext

__all__ = [
    'BaseFeatureCalculator',
    'RSICalculator',
    'EMACalculator', 
    'VWAPCalculator',
    'TimeFeatureCalculator',
    'MarketImpactCalculator',
    'FeatureManager',
    'DataProcessor',
    'FeatureRegistry',
    'get_global_registry',
    'register_calculator',
    'ConfigValidator',
    'PerformanceTracker',
    'TimingContext'
]