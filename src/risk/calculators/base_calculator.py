# src/risk/calculators/base_calculator.py
"""
Base class for all risk calculators with performance optimization features.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging


class RiskMetricType(Enum):
    """Types of risk metrics."""
    DRAWDOWN = "drawdown"
    TURNOVER = "turnover"
    VAR = "var"
    GREEKS = "greeks"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"


@dataclass
class RiskCalculationResult:
    """
    Result of a risk calculation with performance metadata.
    Designed for high-frequency, low-latency processing.
    """
    metric_type: RiskMetricType
    values: Dict[str, Union[float, np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    calculation_time_ns: int = field(default_factory=lambda: time.time_ns())
    is_valid: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Add performance tracking metadata."""
        self.metadata.setdefault('calculator_version', '1.0.0')
        self.metadata.setdefault('input_size', 0)
        self.metadata.setdefault('vectorized', False)
    
    def get_calculation_time_us(self) -> float:
        """Get calculation time in microseconds."""
        return (time.time_ns() - self.calculation_time_ns) / 1000.0
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Safely get a value from the results."""
        return self.values.get(key, default)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result."""
        self.metadata[key] = value


class BaseRiskCalculator(ABC):
    """
    Abstract base class for all risk calculators.
    
    Design principles:
    - Stateless for thread safety
    - Vectorized operations for performance
    - Microsecond-level latency targets
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the risk calculator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.metric_type = self._get_metric_type()
        
        # Performance tracking
        self._calculation_count = 0
        self._total_calculation_time_ns = 0
        self._error_count = 0
        
        # Validation
        self._validate_config()
        
        self.logger.info(f"Initialized {self.__class__.__name__} calculator")
    
    @abstractmethod
    def _get_metric_type(self) -> RiskMetricType:
        """Return the metric type this calculator computes."""
        pass
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate risk metrics from input data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Risk calculation result
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        pass
    
    @abstractmethod
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        pass
    
    def validate_inputs(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_inputs = self.get_required_inputs()
        
        for key in required_inputs:
            if key not in data:
                self.logger.error(f"Missing required input: {key}")
                return False
            
            value = data[key]
            if value is None:
                self.logger.error(f"Required input {key} is None")
                return False
            
            # Check for numpy arrays
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    self.logger.error(f"Required input {key} is empty array")
                    return False
                if np.any(np.isnan(value)):
                    self.logger.warning(f"Input {key} contains NaN values")
        
        return True
    
    def calculate_safe(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Safe wrapper around calculate with error handling and performance tracking.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Risk calculation result (may contain error information)
        """
        start_time = time.time_ns()
        
        try:
            # Validate inputs
            if not self.validate_inputs(data):
                return RiskCalculationResult(
                    metric_type=self.metric_type,
                    values={},
                    is_valid=False,
                    error_message="Input validation failed"
                )
            
            # Perform calculation
            result = self.calculate(data)
            
            # Update performance metrics
            calculation_time = time.time_ns() - start_time
            self._calculation_count += 1
            self._total_calculation_time_ns += calculation_time
            
            # Add performance metadata
            result.add_metadata('calculation_time_ns', calculation_time)
            result.add_metadata('calculation_time_us', calculation_time / 1000.0)
            result.add_metadata('input_size', self._estimate_input_size(data))
            
            return result
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Calculation failed: {e}", exc_info=True)
            
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={},
                is_valid=False,
                error_message=str(e)
            )
    
    def _estimate_input_size(self, data: Dict[str, Any]) -> int:
        """Estimate the size of input data for performance tracking."""
        total_size = 0
        
        for value in data.values():
            if isinstance(value, np.ndarray):
                total_size += value.size
            elif isinstance(value, (list, tuple)):
                total_size += len(value)
            else:
                total_size += 1
        
        return total_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this calculator."""
        if self._calculation_count == 0:
            return {
                'calculation_count': 0,
                'error_count': self._error_count,
                'avg_calculation_time_us': 0.0,
                'total_calculation_time_ms': 0.0,
                'error_rate': 0.0
            }
        
        avg_time_ns = self._total_calculation_time_ns / self._calculation_count
        
        return {
            'calculation_count': self._calculation_count,
            'error_count': self._error_count,
            'avg_calculation_time_us': avg_time_ns / 1000.0,
            'total_calculation_time_ms': self._total_calculation_time_ns / 1_000_000.0,
            'error_rate': self._error_count / self._calculation_count
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._calculation_count = 0
        self._total_calculation_time_ns = 0
        self._error_count = 0
        self.logger.info(f"Reset performance stats for {self.__class__.__name__}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration validation.
        Override in subclasses to provide specific schema.
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def benchmark(self, data: Dict[str, Any], iterations: int = 1000) -> Dict[str, Any]:
        """
        Benchmark the calculator performance.
        
        Args:
            data: Test data
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark with {iterations} iterations")
        
        # Reset stats for clean benchmark
        original_stats = self.get_performance_stats()
        self.reset_performance_stats()
        
        start_time = time.time_ns()
        
        # Run benchmark
        successful_runs = 0
        for i in range(iterations):
            result = self.calculate_safe(data)
            if result.is_valid:
                successful_runs += 1
        
        end_time = time.time_ns()
        total_time_ms = (end_time - start_time) / 1_000_000.0
        
        # Get final stats
        final_stats = self.get_performance_stats()
        
        # Restore original stats
        self._calculation_count = original_stats['calculation_count']
        self._total_calculation_time_ns = int(original_stats['total_calculation_time_ms'] * 1_000_000)
        self._error_count = original_stats['error_count']
        
        benchmark_results = {
            'iterations': iterations,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / iterations,
            'total_time_ms': total_time_ms,
            'avg_time_per_iteration_us': (total_time_ms * 1000) / iterations,
            'throughput_per_second': iterations / (total_time_ms / 1000),
            'avg_calculation_time_us': final_stats['avg_calculation_time_us']
        }
        
        self.logger.info(f"Benchmark completed: {benchmark_results}")
        return benchmark_results


class VectorizedCalculator(BaseRiskCalculator):
    """
    Base class for vectorized calculators that operate on numpy arrays.
    Provides additional utilities for high-performance array operations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
        # Vectorization settings
        self.use_vectorization = config.get('use_vectorization', True)
        self.chunk_size = config.get('chunk_size', 10000)  # For large arrays
        
    def _ensure_array(self, data: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Ensure data is a numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=np.float64)
        else:
            return np.array([data], dtype=np.float64)
    
    def _process_in_chunks(self, data: np.ndarray, 
                          func: callable, 
                          chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Process large arrays in chunks to manage memory usage.
        
        Args:
            data: Input array
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Processed array
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if len(data) <= chunk_size:
            return func(data)
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            result = func(chunk)
            results.append(result)
        
        return np.concatenate(results)
    
    def _validate_array_inputs(self, data: Dict[str, Any]) -> bool:
        """Validate that array inputs have compatible shapes."""
        arrays = {}
        
        # Collect all array inputs
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
        
        if not arrays:
            return True  # No arrays to validate
        
        # Check that all arrays have compatible shapes
        shapes = [arr.shape for arr in arrays.values()]
        first_shape = shapes[0]
        
        for i, shape in enumerate(shapes[1:], 1):
            if shape != first_shape:
                array_names = list(arrays.keys())
                self.logger.error(
                    f"Array shape mismatch: {array_names[0]} has shape {first_shape}, "
                    f"{array_names[i]} has shape {shape}"
                )
                return False
        
        return True