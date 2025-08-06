"""
Risk Governor Stress Testing Platform

A comprehensive testing framework for validating the production readiness
of IntradayJules' three-layer risk management system.

Version: 1.0
Date: August 5, 2025
Status: Implementation Ready
"""

__version__ = "1.0.0"
__author__ = "IntradayJules Team"
__description__ = "Risk Governor Stress Testing Platform"

# Core components
from .core.config import StressTestConfig
from .core.metrics import StressTestMetrics
from .core.governor_wrapper import InstrumentedGovernor

# Simulators
from .simulators.flash_crash_simulator import FlashCrashSimulator
from .simulators.decision_flood_generator import DecisionFloodGenerator
from .simulators.price_feed_interface import PriceFeedInterface

# Validators
from .validators.portfolio_integrity_validator import PortfolioIntegrityValidator
from .validators.latency_validator import LatencyValidator
from .validators.risk_limit_validator import RiskLimitValidator

# Results
from .results.results_analyzer import ResultsAnalyzer
from .results.html_reporter import HTMLReporter

__all__ = [
    'StressTestConfig',
    'StressTestMetrics', 
    'InstrumentedGovernor',
    'FlashCrashSimulator',
    'DecisionFloodGenerator',
    'PriceFeedInterface',
    'PortfolioIntegrityValidator',
    'LatencyValidator',
    'RiskLimitValidator',
    'ResultsAnalyzer',
    'HTMLReporter'
]