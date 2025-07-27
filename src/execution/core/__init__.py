"""
Execution Core Module

This module contains the core components for the execution system:
- ExecutionLoop: Main trading loop logic
- OrderRouter: Order placement and routing
- PnLTracker: Portfolio and P&L tracking
- LiveDataLoader: Real-time data loading and caching
- RiskCallbacks: Risk management callbacks

These modules are internal implementation details and should not be
imported directly. Use the public API through src.execution instead.
"""

# Internal core components - not part of public API
__all__ = []

# Version info for internal tracking
__version__ = "1.0.0"
__status__ = "Development"