# src/execution/throttled_execution_agent.py
"""
Throttled Execution Agent - Integrates order throttling with execution.

Wraps the existing execution system to intercept orders and apply
dynamic size reduction or trade skipping based on risk signals.

Replaces simple logging with actual order modification.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from .order_throttling import (
        OrderThrottler, OrderRequest, ThrottleResult, 
        ThrottleAction, create_order_throttler
    )
    from ..risk.sensors.base_sensor import SensorResult
    from ..risk.risk_agent_v2 import RiskAgentV2
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from execution.order_throttling import (
        OrderThrottler, OrderRequest, ThrottleResult, 
        ThrottleAction, create_order_throttler
    )
    from risk.sensors.base_sensor import SensorResult
    from risk.risk_agent_v2 import RiskAgentV2


@dataclass
class ExecutionDecision:
    """Result of execution decision including throttling."""
    execute: bool
    original_quantity: float
    final_quantity: float
    throttle_result: Optional[ThrottleResult]
    execution_reason: str
    metadata: Dict[str, Any]


class ThrottledExecutionAgent:
    """
    Execution agent that applies order throttling before sending orders.
    
    Integrates with risk sensors to dynamically adjust order sizes or skip trades
    instead of just logging THROTTLE warnings.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 risk_agent: Optional[RiskAgentV2] = None,
                 throttler: Optional[OrderThrottler] = None):
        """
        Initialize throttled execution agent.
        
        Args:
            config: Configuration dictionary
            risk_agent: Risk agent for getting current risk signals
            throttler: Order throttler (created if None)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.risk_agent = risk_agent
        self.throttler = throttler or create_order_throttler(
            config.get('throttling', {})
        )
        
        # Execution tracking
        self.total_execution_requests = 0
        self.executed_orders = 0
        self.throttled_orders = 0
        self.skipped_orders = 0
        self.total_size_reduction = 0.0
        
        # Configuration
        self.min_order_size = config.get('min_order_size', 1.0)
        self.enable_throttling = config.get('enable_throttling', True)
        self.log_all_decisions = config.get('log_all_decisions', True)
        
        self.logger.info("ThrottledExecutionAgent initialized")
    
    def evaluate_execution(self, 
                          symbol: str,
                          side: str,
                          quantity: float,
                          price: Optional[float] = None,
                          order_type: str = "MKT",
                          metadata: Dict[str, Any] = None) -> ExecutionDecision:
        """
        Evaluate whether to execute an order and with what size.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Requested order quantity
            price: Order price (for limit orders)
            order_type: Order type (MKT, LMT, etc.)
            metadata: Additional order metadata
        
        Returns:
            ExecutionDecision with final execution parameters
        """
        start_time = time.perf_counter()
        self.total_execution_requests += 1
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            metadata=metadata or {}
        )
        
        # Get current risk signals
        risk_signals = self._get_current_risk_signals()
        
        # Apply throttling if enabled
        if self.enable_throttling:
            throttle_result = self.throttler.throttle_order(order_request, risk_signals)
        else:
            # No throttling - allow full order
            throttle_result = ThrottleResult(
                action=ThrottleAction.ALLOW,
                original_quantity=quantity,
                final_quantity=quantity,
                reason=None,
                confidence=1.0,
                metadata={'throttling_disabled': True}
            )
        
        # Make execution decision
        execute = True
        final_quantity = throttle_result.final_quantity
        execution_reason = "normal_execution"
        
        # Check if order should be skipped
        if throttle_result.is_blocked:
            execute = False
            final_quantity = 0.0
            execution_reason = f"blocked_by_throttling_{throttle_result.reason.value}"
            self.skipped_orders += 1
        
        # Check minimum order size
        elif final_quantity < self.min_order_size:
            execute = False
            final_quantity = 0.0
            execution_reason = f"below_min_size_{self.min_order_size}"
            self.skipped_orders += 1
        
        # Order will be executed (possibly with reduced size)
        else:
            if throttle_result.action != ThrottleAction.ALLOW:
                self.throttled_orders += 1
                self.total_size_reduction += throttle_result.size_reduction_pct
                execution_reason = f"throttled_{throttle_result.action.value}"
            
            self.executed_orders += 1
        
        # Create execution decision
        decision = ExecutionDecision(
            execute=execute,
            original_quantity=quantity,
            final_quantity=final_quantity,
            throttle_result=throttle_result,
            execution_reason=execution_reason,
            metadata={
                'processing_time_us': (time.perf_counter() - start_time) * 1_000_000,
                'risk_signals_count': len(risk_signals),
                'throttling_enabled': self.enable_throttling
            }
        )
        
        # Log decision
        if self.log_all_decisions or not execute or throttle_result.action != ThrottleAction.ALLOW:
            self._log_execution_decision(decision)
        
        return decision
    
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     quantity: float,
                     price: Optional[float] = None,
                     order_type: str = "MKT",
                     metadata: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute order with throttling applied.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Requested order quantity
            price: Order price (for limit orders)
            order_type: Order type
            metadata: Additional metadata
        
        Returns:
            Tuple of (executed, execution_info)
        """
        # Evaluate execution
        decision = self.evaluate_execution(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            metadata=metadata
        )
        
        # Prepare execution info
        execution_info = {
            'original_quantity': decision.original_quantity,
            'final_quantity': decision.final_quantity,
            'execution_reason': decision.execution_reason,
            'throttle_action': decision.throttle_result.action.value if decision.throttle_result else None,
            'throttle_confidence': decision.throttle_result.confidence if decision.throttle_result else None,
            'size_reduction_pct': decision.throttle_result.size_reduction_pct if decision.throttle_result else 0.0,
            'metadata': decision.metadata
        }
        
        if decision.execute:
            # Execute the order with final quantity
            # This is where you would integrate with actual order execution system
            # For now, we'll simulate successful execution
            execution_info.update({
                'executed': True,
                'order_id': f"ORDER_{int(time.time() * 1000)}",
                'execution_time': time.time(),
                'status': 'submitted'
            })
            
            self.logger.info(
                f"Order executed: {symbol} {side} {decision.final_quantity:.0f} "
                f"(original: {decision.original_quantity:.0f})"
            )
            
            return True, execution_info
        else:
            # Order was blocked/skipped
            execution_info.update({
                'executed': False,
                'block_reason': decision.execution_reason,
                'status': 'blocked'
            })
            
            self.logger.warning(
                f"Order blocked: {symbol} {side} {decision.original_quantity:.0f} "
                f"- {decision.execution_reason}"
            )
            
            return False, execution_info
    
    def _get_current_risk_signals(self) -> List[SensorResult]:
        """Get current risk signals from risk agent."""
        if not self.risk_agent:
            return []
        
        try:
            # Get latest risk assessment
            # This would need to be adapted based on actual RiskAgentV2 interface
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Failed to get risk signals: {e}")
            return []
    
    def _log_execution_decision(self, decision: ExecutionDecision) -> None:
        """Log execution decision details."""
        if decision.execute:
            if decision.throttle_result and decision.throttle_result.action != ThrottleAction.ALLOW:
                self.logger.info(
                    f"THROTTLED EXECUTION: {decision.original_quantity:.0f} -> "
                    f"{decision.final_quantity:.0f} "
                    f"({decision.throttle_result.action.value}, "
                    f"{decision.throttle_result.reason.value if decision.throttle_result.reason else 'unknown'})"
                )
            else:
                self.logger.debug(f"Normal execution: {decision.final_quantity:.0f}")
        else:
            self.logger.warning(
                f"EXECUTION BLOCKED: {decision.original_quantity:.0f} - "
                f"{decision.execution_reason}"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        throttler_stats = self.throttler.get_performance_stats()
        
        if self.total_execution_requests == 0:
            execution_rate = 0.0
            avg_size_reduction = 0.0
        else:
            execution_rate = self.executed_orders / self.total_execution_requests
            avg_size_reduction = self.total_size_reduction / max(self.throttled_orders, 1)
        
        return {
            'execution_stats': {
                'total_requests': self.total_execution_requests,
                'executed_orders': self.executed_orders,
                'throttled_orders': self.throttled_orders,
                'skipped_orders': self.skipped_orders,
                'execution_rate': execution_rate,
                'throttle_rate': self.throttled_orders / max(self.total_execution_requests, 1),
                'skip_rate': self.skipped_orders / max(self.total_execution_requests, 1),
                'avg_size_reduction': avg_size_reduction
            },
            'throttler_stats': throttler_stats,
            'config': {
                'throttling_enabled': self.enable_throttling,
                'min_order_size': self.min_order_size,
                'log_all_decisions': self.log_all_decisions
            }
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_execution_requests = 0
        self.executed_orders = 0
        self.throttled_orders = 0
        self.skipped_orders = 0
        self.total_size_reduction = 0.0
        
        if self.throttler:
            self.throttler.reset_stats()
    
    def enable_throttling(self) -> None:
        """Enable order throttling."""
        self.enable_throttling = True
        if self.throttler:
            self.throttler.enable()
        self.logger.info("Order throttling enabled")
    
    def disable_throttling(self) -> None:
        """Disable order throttling."""
        self.enable_throttling = False
        if self.throttler:
            self.throttler.disable()
        self.logger.info("Order throttling disabled")
    
    def update_throttling_config(self, config: Dict[str, Any]) -> None:
        """Update throttling configuration."""
        # This would recreate the throttler with new config
        # For now, just log the update
        self.logger.info(f"Throttling config update requested: {config}")
        # TODO: Implement config update logic


def create_throttled_execution_agent(config: Dict[str, Any] = None,
                                    risk_agent: Optional[RiskAgentV2] = None) -> ThrottledExecutionAgent:
    """
    Factory function to create ThrottledExecutionAgent.
    
    Args:
        config: Configuration dictionary
        risk_agent: Optional risk agent for risk signals
    
    Returns:
        Configured ThrottledExecutionAgent
    """
    default_config = {
        'enable_throttling': True,
        'min_order_size': 1.0,
        'log_all_decisions': False,
        'throttling': {
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
            }
        }
    }
    
    if config:
        # Merge with defaults
        final_config = {**default_config, **config}
        if 'throttling' in config:
            final_config['throttling'] = {**default_config['throttling'], **config['throttling']}
    else:
        final_config = default_config
    
    return ThrottledExecutionAgent(final_config, risk_agent)


__all__ = [
    'ThrottledExecutionAgent',
    'ExecutionDecision',
    'create_throttled_execution_agent'
]