"""
Instrumented Risk Governor Wrapper

Provides comprehensive instrumentation for the Risk Governor with
timing, error tracking, and decision monitoring for stress testing.
"""

import time
import logging
import copy
from typing import Any, Dict, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from .metrics import get_metrics, StressTestMetrics
from .config import get_config


class InstrumentedGovernor:
    """
    Wrapper around the Risk Governor that adds comprehensive instrumentation
    for stress testing without modifying the original governor code.
    """
    
    def __init__(self, base_governor: Any, model: Any, metrics: Optional[StressTestMetrics] = None):
        """
        Initialize instrumented governor.
        
        Args:
            base_governor: The original Risk Governor instance
            model: The ML model for predictions
            metrics: Metrics collector (uses global if None)
        """
        self.governor = base_governor
        self.model = model
        self.metrics = metrics or get_metrics()
        self.config = get_config()
        
        # State tracking
        self._decision_count = 0
        self._error_count = 0
        self._last_decision_time = None
        self._positions = {}
        self._cash = 0.0
        
        # Performance tracking
        self._latency_history = []
        self._error_history = []
        
        logging.info("InstrumentedGovernor initialized")
    
    def make_decision(self, observation: Any) -> Any:
        """
        Make a trading decision with full instrumentation.
        
        This is the main entry point that measures the complete decision pipeline:
        1. Model inference
        2. Risk governor filtering
        3. Metrics collection
        4. Error handling
        """
        decision_start = time.perf_counter_ns()
        
        try:
            # Model inference timing
            model_start = time.perf_counter_ns()
            raw_action = self.model.predict(observation)
            model_time = time.perf_counter_ns() - model_start
            
            # Risk governor filtering timing
            governor_start = time.perf_counter_ns()
            filtered_action = self.governor.filter(raw_action)
            governor_time = time.perf_counter_ns() - governor_start
            
            # Total decision time
            total_time = time.perf_counter_ns() - decision_start
            
            # Record metrics
            self.metrics.timing('decision_ns', total_time)
            self.metrics.counter('decisions_total').inc()
            
            # Update internal state
            self._decision_count += 1
            self._last_decision_time = time.time()
            self._latency_history.append(total_time / 1_000_000)  # Convert to ms
            
            # Log detailed timing for analysis
            if self._decision_count % 1000 == 0:
                logging.info(
                    f"Decision #{self._decision_count}: "
                    f"Total={total_time/1_000_000:.2f}ms, "
                    f"Model={model_time/1_000_000:.2f}ms, "
                    f"Governor={governor_time/1_000_000:.2f}ms"
                )
            
            return filtered_action
            
        except Exception as e:
            # Record error metrics
            self.metrics.counter('decision_errors').inc()
            self._error_count += 1
            self._error_history.append({
                'timestamp': time.time(),
                'error': str(e),
                'decision_count': self._decision_count
            })
            
            logging.error(f"Decision error #{self._error_count}: {e}")
            raise
    
    def update_position(self, symbol: str, position: float, price: float):
        """Update position tracking for integrity validation."""
        old_position = self._positions.get(symbol, 0.0)
        self._positions[symbol] = position
        
        # Update cash based on position change
        position_change = position - old_position
        self._cash -= position_change * price
        
        logging.debug(f"Position update: {symbol} {old_position} -> {position} @ ${price}")
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for validation."""
        return {
            'positions': self._positions.copy(),
            'cash': self._cash,
            'timestamp': time.time(),
            'decision_count': self._decision_count
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the instrumented governor."""
        if not self._latency_history:
            return {'error': 'No decisions recorded'}
        
        latencies = self._latency_history[-10000:]  # Last 10k decisions
        
        return {
            'decision_count': self._decision_count,
            'error_count': self._error_count,
            'error_rate_pct': (self._error_count / max(self._decision_count, 1)) * 100,
            'latency_stats': {
                'mean_ms': sum(latencies) / len(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'p95_ms': sorted(latencies)[int(0.95 * len(latencies))],
                'p99_ms': sorted(latencies)[int(0.99 * len(latencies))]
            },
            'last_decision_time': self._last_decision_time
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._decision_count = 0
        self._error_count = 0
        self._latency_history.clear()
        self._error_history.clear()
        self._last_decision_time = None
        logging.info("InstrumentedGovernor stats reset")
    
    @contextmanager
    def shadow_mode(self):
        """
        Context manager for shadow testing mode.
        
        Creates a deep copy of the governor for isolated testing
        without affecting the original state.
        """
        # Save original state
        original_positions = self._positions.copy()
        original_cash = self._cash
        original_governor = self.governor
        
        try:
            # Create shadow copy
            self.governor = copy.deepcopy(original_governor)
            self._positions = {}
            self._cash = 0.0
            
            logging.info("Entered shadow mode - isolated testing environment")
            yield self
            
        finally:
            # Restore original state
            self.governor = original_governor
            self._positions = original_positions
            self._cash = original_cash
            logging.info("Exited shadow mode - restored original state")
    
    def validate_integrity(self, expected_positions: Dict[str, float], expected_cash: float) -> Dict[str, Any]:
        """
        Validate portfolio integrity against expected values.
        
        Returns validation results with pass/fail status and deltas.
        """
        position_deltas = {}
        max_position_delta = 0.0
        
        # Check position deltas
        all_symbols = set(self._positions.keys()) | set(expected_positions.keys())
        for symbol in all_symbols:
            actual = self._positions.get(symbol, 0.0)
            expected = expected_positions.get(symbol, 0.0)
            delta = abs(actual - expected)
            position_deltas[symbol] = delta
            max_position_delta = max(max_position_delta, delta)
        
        # Check cash delta
        cash_delta = abs(self._cash - expected_cash)
        
        # Determine pass/fail status
        position_pass = max_position_delta <= self.config.position_tolerance_usd
        cash_pass = cash_delta <= self.config.cash_tolerance_usd
        overall_pass = position_pass and cash_pass
        
        # Update metrics
        self.metrics.gauge('position_delta_usd').set(max_position_delta)
        
        result = {
            'overall_pass': overall_pass,
            'position_validation': {
                'pass': position_pass,
                'max_delta_usd': max_position_delta,
                'tolerance_usd': self.config.position_tolerance_usd,
                'deltas_by_symbol': position_deltas
            },
            'cash_validation': {
                'pass': cash_pass,
                'delta_usd': cash_delta,
                'tolerance_usd': self.config.cash_tolerance_usd,
                'actual': self._cash,
                'expected': expected_cash
            },
            'timestamp': time.time()
        }
        
        logging.info(f"Integrity validation: {'PASS' if overall_pass else 'FAIL'}")
        return result


def instrument_governor_function(governor_func: Callable, metrics: Optional[StressTestMetrics] = None) -> Callable:
    """
    Decorator to instrument any governor function with timing and error tracking.
    
    Usage:
        @instrument_governor_function
        def my_governor_decision(obs):
            return governor.filter(model.predict(obs))
    """
    metrics = metrics or get_metrics()
    
    @wraps(governor_func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            result = governor_func(*args, **kwargs)
            metrics.timing('decision_ns', time.perf_counter_ns() - t0)
            metrics.counter('decisions_total').inc()
            return result
        except Exception as e:
            metrics.counter('decision_errors').inc()
            raise
    
    return wrapper


class ShadowGovernor:
    """
    Lightweight shadow governor for load testing.
    
    Provides the same interface as InstrumentedGovernor but with
    minimal overhead for high-frequency testing scenarios.
    """
    
    def __init__(self, base_governor: Any, model: Any):
        self.governor = copy.deepcopy(base_governor)
        self.model = model
        self.metrics = get_metrics()
        self._decision_count = 0
    
    def make_decision(self, observation: Any) -> Any:
        """Lightweight decision making for load testing."""
        t0 = time.perf_counter_ns()
        
        try:
            raw_action = self.model.predict(observation)
            filtered_action = self.governor.filter(raw_action)
            
            # Minimal metrics collection
            self.metrics.timing('decision_ns', time.perf_counter_ns() - t0)
            self.metrics.counter('decisions_total').inc()
            
            self._decision_count += 1
            return filtered_action
            
        except Exception:
            self.metrics.counter('decision_errors').inc()
            raise
    
    def get_decision_count(self) -> int:
        """Get total number of decisions made."""
        return self._decision_count