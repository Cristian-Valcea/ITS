# src/execution/order_throttling.py
"""
Order Throttling System - Dynamic size reduction and trade skipping.

Hooks into THROTTLE signals from risk sensors to actually reduce order sizes
or skip trades instead of just logging warnings.

Features:
- Dynamic position size reduction based on risk levels
- Trade skipping for high-risk conditions
- Configurable throttling strategies
- Integration with Kyle Lambda impact modeling
- Performance tracking and monitoring
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import numpy as np

try:
    from ..risk.sensors.base_sensor import SensorAction, SensorResult
    from ..risk.rules_engine import RuleAction, RuleResult
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from risk.sensors.base_sensor import SensorAction, SensorResult
    from risk.rules_engine import RuleAction, RuleResult


class ThrottleAction(Enum):
    """Actions that can be taken when throttling is triggered."""
    ALLOW = "allow"                    # Allow full order
    REDUCE_25 = "reduce_25"           # Reduce size by 25%
    REDUCE_50 = "reduce_50"           # Reduce size by 50%
    REDUCE_75 = "reduce_75"           # Reduce size by 75%
    REDUCE_90 = "reduce_90"           # Reduce size by 90%
    SKIP = "skip"                     # Skip trade entirely
    DELAY = "delay"                   # Delay trade execution


class ThrottleReason(Enum):
    """Reasons for throttling orders."""
    KYLE_LAMBDA_HIGH = "kyle_lambda_high"
    TURNOVER_LIMIT = "turnover_limit"
    DRAWDOWN_RISK = "drawdown_risk"
    MARKET_IMPACT = "market_impact"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_LOW = "liquidity_low"
    RISK_SENSOR = "risk_sensor"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class OrderRequest:
    """Represents an order request before throttling."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = "MKT"  # Market order default
    price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThrottleResult:
    """Result of order throttling evaluation."""
    action: ThrottleAction
    original_quantity: float
    final_quantity: float
    reason: ThrottleReason
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_us: float = 0.0
    
    @property
    def size_reduction_pct(self) -> float:
        """Calculate percentage size reduction."""
        if self.original_quantity == 0:
            return 0.0
        return (self.original_quantity - self.final_quantity) / self.original_quantity
    
    @property
    def is_blocked(self) -> bool:
        """Check if order is completely blocked."""
        return self.action == ThrottleAction.SKIP or self.final_quantity == 0


class BaseThrottleStrategy(ABC):
    """Base class for order throttling strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate_order(self, order: OrderRequest, risk_signals: List[SensorResult]) -> ThrottleResult:
        """Evaluate order and determine throttling action."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name for logging."""
        pass


class KyleLambdaThrottleStrategy(BaseThrottleStrategy):
    """
    Throttling strategy based on Kyle Lambda market impact.
    
    Reduces order size when market impact would be excessive.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.impact_thresholds = {
            'low': config.get('low_impact_bps', 10.0),      # 10 bps
            'medium': config.get('medium_impact_bps', 25.0), # 25 bps
            'high': config.get('high_impact_bps', 50.0),     # 50 bps
            'extreme': config.get('extreme_impact_bps', 100.0) # 100 bps
        }
        
        self.size_reductions = {
            'low': 0.0,      # No reduction
            'medium': 0.25,  # 25% reduction
            'high': 0.50,    # 50% reduction
            'extreme': 0.90  # 90% reduction
        }
        
        self.skip_threshold_bps = config.get('skip_threshold_bps', 150.0)
    
    def get_strategy_name(self) -> str:
        return "KyleLambdaThrottle"
    
    def evaluate_order(self, order: OrderRequest, risk_signals: List[SensorResult]) -> ThrottleResult:
        """Evaluate order based on Kyle Lambda market impact."""
        start_time = time.perf_counter()
        
        # Find Kyle Lambda sensor results
        kyle_lambda_results = [
            signal for signal in risk_signals 
            if 'kyle_lambda' in signal.sensor_id.lower()
        ]
        
        if not kyle_lambda_results:
            # No Kyle Lambda data, allow order
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.KYLE_LAMBDA_HIGH,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'no_kyle_lambda_data': True}
            )
        
        # Get the latest Kyle Lambda result
        latest_result = kyle_lambda_results[-1]
        kyle_lambda = latest_result.metadata.get('kyle_lambda', latest_result.value)
        
        # Estimate market impact for this order
        notional_value = order.quantity * order.metadata.get('price', 100.0)
        estimated_impact_bps = kyle_lambda * notional_value  # Kyle lambda already in bps equivalent
        
        # Determine throttling action based on impact
        action = ThrottleAction.ALLOW
        reduction_pct = 0.0
        confidence = 0.8
        
        if estimated_impact_bps >= self.skip_threshold_bps:
            action = ThrottleAction.SKIP
            reduction_pct = 1.0
            confidence = 0.95
        elif estimated_impact_bps >= self.impact_thresholds['extreme']:
            action = ThrottleAction.REDUCE_90
            reduction_pct = self.size_reductions['extreme']
            confidence = 0.9
        elif estimated_impact_bps >= self.impact_thresholds['high']:
            action = ThrottleAction.REDUCE_50
            reduction_pct = self.size_reductions['high']
            confidence = 0.85
        elif estimated_impact_bps >= self.impact_thresholds['medium']:
            action = ThrottleAction.REDUCE_25
            reduction_pct = self.size_reductions['medium']
            confidence = 0.75
        
        final_quantity = order.quantity * (1.0 - reduction_pct)
        processing_time = (time.perf_counter() - start_time) * 1_000_000
        
        return ThrottleResult(
            action=action,
            original_quantity=order.quantity,
            final_quantity=final_quantity,
            reason=ThrottleReason.KYLE_LAMBDA_HIGH,
            confidence=confidence,
            processing_time_us=processing_time,
            metadata={
                'kyle_lambda': kyle_lambda,
                'estimated_impact_bps': estimated_impact_bps,
                'notional_value': notional_value,
                'reduction_pct': reduction_pct
            }
        )


class TurnoverThrottleStrategy(BaseThrottleStrategy):
    """
    Throttling strategy based on turnover limits.
    
    Reduces order size when turnover limits would be breached.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hourly_limit = config.get('hourly_turnover_limit', 5.0)
        self.daily_limit = config.get('daily_turnover_limit', 20.0)
        
        # Throttling levels based on turnover ratio
        self.throttle_levels = {
            0.8: 0.0,   # 80% of limit: no reduction
            0.9: 0.25,  # 90% of limit: 25% reduction
            0.95: 0.50, # 95% of limit: 50% reduction
            1.0: 0.75,  # 100% of limit: 75% reduction
            1.1: 1.0    # 110% of limit: skip trade
        }
    
    def get_strategy_name(self) -> str:
        return "TurnoverThrottle"
    
    def evaluate_order(self, order: OrderRequest, risk_signals: List[SensorResult]) -> ThrottleResult:
        """Evaluate order based on turnover limits."""
        start_time = time.perf_counter()
        
        # Find turnover sensor results
        turnover_results = [
            signal for signal in risk_signals 
            if 'turnover' in signal.sensor_id.lower()
        ]
        
        if not turnover_results:
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.TURNOVER_LIMIT,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'no_turnover_data': True}
            )
        
        # Get current turnover ratios
        latest_result = turnover_results[-1]
        hourly_ratio = latest_result.metadata.get('hourly_turnover_ratio', latest_result.value)
        daily_ratio = latest_result.metadata.get('daily_turnover_ratio', latest_result.value * 0.5)
        
        # Use the higher ratio for throttling decision
        max_ratio = max(hourly_ratio / self.hourly_limit, daily_ratio / self.daily_limit)
        
        # Determine throttling action
        action = ThrottleAction.ALLOW
        reduction_pct = 0.0
        
        for threshold, reduction in sorted(self.throttle_levels.items()):
            if max_ratio >= threshold:
                reduction_pct = reduction
                if reduction >= 1.0:
                    action = ThrottleAction.SKIP
                elif reduction >= 0.75:
                    action = ThrottleAction.REDUCE_75
                elif reduction >= 0.50:
                    action = ThrottleAction.REDUCE_50
                elif reduction >= 0.25:
                    action = ThrottleAction.REDUCE_25
        
        final_quantity = order.quantity * (1.0 - reduction_pct)
        processing_time = (time.perf_counter() - start_time) * 1_000_000
        
        return ThrottleResult(
            action=action,
            original_quantity=order.quantity,
            final_quantity=final_quantity,
            reason=ThrottleReason.TURNOVER_LIMIT,
            confidence=0.9,
            processing_time_us=processing_time,
            metadata={
                'hourly_ratio': hourly_ratio,
                'daily_ratio': daily_ratio,
                'max_ratio': max_ratio,
                'reduction_pct': reduction_pct
            }
        )


class CompositeThrottleStrategy(BaseThrottleStrategy):
    """
    Composite strategy that combines multiple throttling strategies.
    
    Takes the most conservative action across all strategies.
    """
    
    def __init__(self, config: Dict[str, Any], strategies: List[BaseThrottleStrategy]):
        super().__init__(config)
        self.strategies = strategies
        self.combination_method = config.get('combination_method', 'most_conservative')
    
    def get_strategy_name(self) -> str:
        return "CompositeThrottle"
    
    def evaluate_order(self, order: OrderRequest, risk_signals: List[SensorResult]) -> ThrottleResult:
        """Evaluate order using all strategies and combine results."""
        start_time = time.perf_counter()
        
        if not self.strategies:
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.MANUAL_OVERRIDE,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'no_strategies': True}
            )
        
        # Evaluate all strategies
        results = []
        for strategy in self.strategies:
            try:
                result = strategy.evaluate_order(order, risk_signals)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Strategy {strategy.get_strategy_name()} failed: {e}")
                continue
        
        if not results:
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.MANUAL_OVERRIDE,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'all_strategies_failed': True}
            )
        
        # Combine results using most conservative approach
        if self.combination_method == 'most_conservative':
            # Find the result with the smallest final quantity
            most_conservative = min(results, key=lambda r: r.final_quantity)
            
            # Combine metadata from all strategies
            combined_metadata = {
                'strategy_results': [
                    {
                        'strategy': self.strategies[i].get_strategy_name(),
                        'action': results[i].action.value,
                        'final_quantity': results[i].final_quantity,
                        'confidence': results[i].confidence
                    }
                    for i in range(len(results))
                ],
                'combination_method': self.combination_method
            }
            combined_metadata.update(most_conservative.metadata)
            
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            
            return ThrottleResult(
                action=most_conservative.action,
                original_quantity=order.quantity,
                final_quantity=most_conservative.final_quantity,
                reason=most_conservative.reason,
                confidence=np.mean([r.confidence for r in results]),
                processing_time_us=processing_time,
                metadata=combined_metadata
            )
        
        # Default fallback
        processing_time = (time.perf_counter() - start_time) * 1_000_000
        return results[0]


class OrderThrottler:
    """
    Main order throttling system that intercepts orders and applies throttling.
    
    Integrates with risk sensors and applies dynamic size reduction or trade skipping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order throttler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Performance tracking
        self.total_orders = 0
        self.throttled_orders = 0
        self.skipped_orders = 0
        self.total_size_reduction = 0.0
        self.total_processing_time_us = 0.0
        
        # Enable/disable throttling
        self.enabled = config.get('enabled', True)
        
        self.logger.info(f"OrderThrottler initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> List[BaseThrottleStrategy]:
        """Initialize throttling strategies from configuration."""
        strategies = []
        
        strategy_configs = self.config.get('strategies', {})
        
        # Kyle Lambda strategy
        if strategy_configs.get('kyle_lambda', {}).get('enabled', True):
            kyle_config = strategy_configs.get('kyle_lambda', {})
            strategies.append(KyleLambdaThrottleStrategy(kyle_config))
        
        # Turnover strategy
        if strategy_configs.get('turnover', {}).get('enabled', True):
            turnover_config = strategy_configs.get('turnover', {})
            strategies.append(TurnoverThrottleStrategy(turnover_config))
        
        # Wrap in composite strategy if multiple strategies
        if len(strategies) > 1:
            composite_config = self.config.get('composite', {})
            return [CompositeThrottleStrategy(composite_config, strategies)]
        
        return strategies
    
    def throttle_order(self, order: OrderRequest, risk_signals: List[SensorResult]) -> ThrottleResult:
        """
        Main throttling method - evaluates order and applies throttling.
        
        Args:
            order: Order request to evaluate
            risk_signals: Current risk sensor signals
        
        Returns:
            ThrottleResult with final order parameters
        """
        start_time = time.perf_counter()
        self.total_orders += 1
        
        # Check if throttling is disabled
        if not self.enabled:
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.MANUAL_OVERRIDE,
                confidence=1.0,
                processing_time_us=processing_time,
                metadata={'throttling_disabled': True}
            )
        
        # No strategies configured
        if not self.strategies:
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.MANUAL_OVERRIDE,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'no_strategies_configured': True}
            )
        
        # Evaluate using primary strategy (first in list)
        try:
            result = self.strategies[0].evaluate_order(order, risk_signals)
            
            # Update performance tracking
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            self.total_processing_time_us += processing_time
            
            if result.action != ThrottleAction.ALLOW:
                self.throttled_orders += 1
                self.total_size_reduction += result.size_reduction_pct
                
                if result.is_blocked:
                    self.skipped_orders += 1
            
            # Log throttling action
            if result.action != ThrottleAction.ALLOW:
                self.logger.info(
                    f"Order throttled: {order.symbol} {order.side} "
                    f"{order.quantity:.0f} -> {result.final_quantity:.0f} "
                    f"({result.action.value}, {result.reason.value})"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order throttling failed: {e}")
            processing_time = (time.perf_counter() - start_time) * 1_000_000
            
            # Return safe default (allow order)
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                reason=ThrottleReason.MANUAL_OVERRIDE,
                confidence=0.0,
                processing_time_us=processing_time,
                metadata={'error': str(e)}
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get throttling performance statistics."""
        if self.total_orders == 0:
            return {
                'total_orders': 0,
                'throttled_orders': 0,
                'skipped_orders': 0,
                'throttle_rate': 0.0,
                'skip_rate': 0.0,
                'avg_size_reduction': 0.0,
                'avg_processing_time_us': 0.0
            }
        
        return {
            'total_orders': self.total_orders,
            'throttled_orders': self.throttled_orders,
            'skipped_orders': self.skipped_orders,
            'throttle_rate': self.throttled_orders / self.total_orders,
            'skip_rate': self.skipped_orders / self.total_orders,
            'avg_size_reduction': self.total_size_reduction / max(self.throttled_orders, 1),
            'avg_processing_time_us': self.total_processing_time_us / self.total_orders,
            'enabled': self.enabled,
            'strategies_count': len(self.strategies)
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_orders = 0
        self.throttled_orders = 0
        self.skipped_orders = 0
        self.total_size_reduction = 0.0
        self.total_processing_time_us = 0.0
    
    def enable(self) -> None:
        """Enable order throttling."""
        self.enabled = True
        self.logger.info("Order throttling enabled")
    
    def disable(self) -> None:
        """Disable order throttling."""
        self.enabled = False
        self.logger.info("Order throttling disabled")


def create_order_throttler(config: Dict[str, Any] = None) -> OrderThrottler:
    """
    Factory function to create OrderThrottler with default configuration.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured OrderThrottler instance
    """
    default_config = {
        'enabled': True,
        'strategies': {
            'kyle_lambda': {
                'enabled': True,
                'low_impact_bps': 10.0,
                'medium_impact_bps': 25.0,
                'high_impact_bps': 50.0,
                'extreme_impact_bps': 100.0,
                'skip_threshold_bps': 150.0
            },
            'turnover': {
                'enabled': True,
                'hourly_turnover_limit': 5.0,
                'daily_turnover_limit': 20.0
            }
        },
        'composite': {
            'combination_method': 'most_conservative'
        }
    }
    
    if config:
        # Deep merge configurations
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        final_config = deep_merge(default_config, config)
    else:
        final_config = default_config
    
    return OrderThrottler(final_config)


__all__ = [
    'OrderThrottler',
    'OrderRequest', 
    'ThrottleResult',
    'ThrottleAction',
    'ThrottleReason',
    'BaseThrottleStrategy',
    'KyleLambdaThrottleStrategy',
    'TurnoverThrottleStrategy',
    'CompositeThrottleStrategy',
    'create_order_throttler'
]