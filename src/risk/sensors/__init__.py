# src/risk/sensors/__init__.py
"""
Advanced Risk Sensors System - Blind-Spot Detection Framework

This module implements a sensor-based approach to risk management, treating risk as
real-time sensors that detect hidden failure modes rather than just static metrics.

Architecture:
- BaseSensor: Abstract sensor interface with latency budgets
- SensorRegistry: Fast sensor lookup and execution
- SensorPipeline: Multi-lane processing (fast/slow paths)
- FailureModeDetector: Pattern recognition for failure modes

Sensor Categories:
1. Path-Fragility Sensors (Ulcer Index, TTR, Drawdown Velocity)
2. Tail & Regime-Shift Sensors (CVaR, Vol-of-Vol, Regime Detection)
3. Liquidity & Execution Sensors (ADV Participation, Kyle Lambda)
4. Funding & Margin Sensors (Time-to-Margin, LaR)
5. Counterparty Sensors (PFE, Herstatt Window)
6. Operational & Tech Sensors (Latency Drift, Feed Staleness)
"""

from .base_sensor import BaseSensor, SensorResult, SensorPriority, FailureMode, SensorAction
from .sensor_registry import SensorRegistry
from .sensor_pipeline import SensorPipeline, FastLane, SlowLane
from .failure_mode_detector import FailureModeDetector

# Path-Fragility Sensors
from .path_fragility import (
    UlcerIndexSensor,
    DrawdownVelocitySensor,
    TimeToRecoverySensor,
    DrawdownAdjustedLeverageSensor
)

# Tail & Regime-Shift Sensors
from .tail_regime import (
    ExpectedShortfallSensor,
    VolOfVolSensor,
    RegimeSwitchSensor
)

# Liquidity & Execution Sensors
from .liquidity_execution import (
    ADVParticipationSensor,
    DepthAtPriceShockSensor,
    KyleLambdaSensor
)

# Funding & Margin Sensors
from .funding_margin import (
    TimeToMarginExhaustionSensor,
    LiquidityAtRiskSensor,
    HaircutSensitivitySensor
)

# Counterparty Sensors
from .counterparty import (
    CorrelationAdjustedPFESensor,
    HerstattWindowSensor
)

# Operational & Tech Sensors
from .operational_tech import (
    LatencyDriftSensor,
    FeedStalenessSensor,
    ExceptionRateSensor
)

__all__ = [
    # Core Framework
    'BaseSensor', 'SensorResult', 'SensorPriority', 'FailureMode', 'SensorAction',
    'SensorRegistry', 'SensorPipeline', 'FastLane', 'SlowLane',
    'FailureModeDetector',
    
    # Path-Fragility Sensors
    'UlcerIndexSensor', 'DrawdownVelocitySensor', 'TimeToRecoverySensor',
    'DrawdownAdjustedLeverageSensor',
    
    # Tail & Regime-Shift Sensors
    'ExpectedShortfallSensor', 'VolOfVolSensor', 'RegimeSwitchSensor',
    
    # Liquidity & Execution Sensors
    'ADVParticipationSensor', 'DepthAtPriceShockSensor', 'KyleLambdaSensor',
    
    # Funding & Margin Sensors
    'TimeToMarginExhaustionSensor', 'LiquidityAtRiskSensor', 'HaircutSensitivitySensor',
    
    # Counterparty Sensors
    'CorrelationAdjustedPFESensor', 'HerstattWindowSensor',
    
    # Operational & Tech Sensors
    'LatencyDriftSensor', 'FeedStalenessSensor', 'ExceptionRateSensor'
]