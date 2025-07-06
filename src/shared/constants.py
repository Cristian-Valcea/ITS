# src/shared/constants.py
"""
Shared constants for the IntradayJules trading system.
Centralized to avoid magic strings and ensure consistency across modules.
"""

# Price data column names
CLOSE = "close"
OPEN = "open"
HIGH = "high"
LOW = "low"
VOLUME = "volume"
VWAP = "vwap"

# Feature names
RSI = "rsi"
EMA_FAST = "ema_fast"
EMA_SLOW = "ema_slow"
BOLLINGER_UPPER = "bb_upper"
BOLLINGER_LOWER = "bb_lower"
VWAP_DEVIATION = "vwap_dev"

# Trading actions
ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY = 2

# Position states
POSITION_SHORT = -1
POSITION_FLAT = 0
POSITION_LONG = 1

# Risk thresholds
DEFAULT_MAX_DRAWDOWN_PCT = 0.02
DEFAULT_TRANSACTION_COST_PCT = 0.001
DEFAULT_POSITION_SIZE_PCT = 0.25

# Model versioning
MODEL_VERSION_FORMAT = "%Y-%m-%d_%H-%M-%S"
POLICY_BUNDLE_VERSION = "1.0"

# Performance SLOs
MAX_PREDICTION_LATENCY_US = 100.0
MAX_RISK_EVALUATION_LATENCY_US = 50.0

__all__ = [
    # Price columns
    "CLOSE", "OPEN", "HIGH", "LOW", "VOLUME", "VWAP",
    # Features
    "RSI", "EMA_FAST", "EMA_SLOW", "BOLLINGER_UPPER", "BOLLINGER_LOWER", "VWAP_DEVIATION",
    # Actions and positions
    "ACTION_SELL", "ACTION_HOLD", "ACTION_BUY",
    "POSITION_SHORT", "POSITION_FLAT", "POSITION_LONG",
    # Risk defaults
    "DEFAULT_MAX_DRAWDOWN_PCT", "DEFAULT_TRANSACTION_COST_PCT", "DEFAULT_POSITION_SIZE_PCT",
    # Versioning
    "MODEL_VERSION_FORMAT", "POLICY_BUNDLE_VERSION",
    # Performance
    "MAX_PREDICTION_LATENCY_US", "MAX_RISK_EVALUATION_LATENCY_US",
]