# src/risk/calculators/__init__.py
"""
High-performance risk calculators with microsecond-level latency.

All calculators are designed to be:
- Stateless for thread safety
- Vectorized for performance  
- GPU/SIMD friendly
- Single responsibility
- Easily unit testable
"""

from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType, VectorizedCalculator
from .drawdown_calculator import DrawdownCalculator
from .turnover_calculator import TurnoverCalculator
from .var_calculator import VaRCalculator
from .greeks_calculator import GreeksCalculator
from .volatility_calculator import VolatilityCalculator
from .concentration_calculator import ConcentrationCalculator

# New sensor-based calculators
from .feed_staleness_calculator import FeedStalenessCalculator
from .drawdown_velocity_calculator import DrawdownVelocityCalculator
from .ulcer_index_calculator import UlcerIndexCalculator
from .expected_shortfall_calculator import ExpectedShortfallCalculator
from .kyle_lambda_calculator import KyleLambdaCalculator
from .depth_shock_calculator import DepthShockCalculator
from .latency_drift_calculator import LatencyDriftCalculator
from .adv_participation_calculator import ADVParticipationCalculator

__all__ = [
    # Base classes
    'BaseRiskCalculator',
    'VectorizedCalculator',
    'RiskCalculationResult',
    'RiskMetricType',
    
    # Original calculators
    'DrawdownCalculator',
    'TurnoverCalculator',
    'VaRCalculator',
    'GreeksCalculator',
    'VolatilityCalculator',
    'ConcentrationCalculator',
    
    # New sensor calculators
    'FeedStalenessCalculator',
    'DrawdownVelocityCalculator', 
    'UlcerIndexCalculator',
    'ExpectedShortfallCalculator',
    'KyleLambdaCalculator',
    'DepthShockCalculator',
    'LatencyDriftCalculator',
    'ADVParticipationCalculator'
]