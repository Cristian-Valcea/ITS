# tests/test_risk_calculators.py
"""
Comprehensive tests for risk calculators with latency benchmarks and golden-file tests.
"""

import pytest
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk.calculators import (
    DrawdownCalculator, TurnoverCalculator, BaseRiskCalculator
)
from src.risk.calculators.base_calculator import RiskMetricType


class TestDrawdownCalculator:
    """Test DrawdownCalculator with golden-file tests and latency benchmarks."""
    
    @pytest.fixture
    def calculator(self):
        """Create DrawdownCalculator instance."""
        config = {
            'lookback_periods': [1, 5, 20],
            'min_periods': 1,
            'annualization_factor': 252
        }
        return DrawdownCalculator(config)
    
    @pytest.fixture
    def golden_data(self):
        """Load golden test data."""
        return {
            'simple_decline': {
                'portfolio_values': np.array([100000, 98000, 96000, 94000, 92000]),
                'expected_daily_drawdown': -0.08,  # -8%
                'expected_max_drawdown': -0.08,
                'expected_recovery_time': None  # Still in drawdown
            },
            'recovery_scenario': {
                'portfolio_values': np.array([100000, 95000, 90000, 95000, 102000]),
                'expected_daily_drawdown': 0.02,  # +2% (recovered)
                'expected_max_drawdown': -0.10,  # -10% was the max
                'expected_recovery_time': 2  # Recovered after 2 periods
            },
            'volatile_market': {
                'portfolio_values': np.array([100000, 105000, 98000, 103000, 97000, 101000]),
                'expected_daily_drawdown': 0.01,  # +1%
                'expected_max_drawdown': -0.07,  # -7% max decline
                'expected_recovery_time': 1  # Quick recovery
            }
        }
    
    def test_golden_file_simple_decline(self, calculator, golden_data):
        """Test simple decline scenario against golden values."""
        data = golden_data['simple_decline']
        
        result = calculator.calculate_safe({
            'portfolio_values': data['portfolio_values'],
            'start_of_day_value': data['portfolio_values'][0]
        })
        
        assert result.is_valid, f"Calculation failed: {result.error_message}"
        
        # Check daily drawdown
        daily_dd = result.get_value('daily_drawdown')
        assert abs(daily_dd - data['expected_daily_drawdown']) < 0.001, \
            f"Daily drawdown mismatch: {daily_dd} vs {data['expected_daily_drawdown']}"
        
        # Check max drawdown
        max_dd = result.get_value('max_drawdown')
        assert abs(max_dd - data['expected_max_drawdown']) < 0.001, \
            f"Max drawdown mismatch: {max_dd} vs {data['expected_max_drawdown']}"
    
    def test_golden_file_recovery_scenario(self, calculator, golden_data):
        """Test recovery scenario against golden values."""
        data = golden_data['recovery_scenario']
        
        result = calculator.calculate_safe({
            'portfolio_values': data['portfolio_values'],
            'start_of_day_value': data['portfolio_values'][0]
        })
        
        assert result.is_valid
        
        # Check recovery
        daily_dd = result.get_value('daily_drawdown')
        assert abs(daily_dd - data['expected_daily_drawdown']) < 0.001
        
        max_dd = result.get_value('max_drawdown')
        assert abs(max_dd - data['expected_max_drawdown']) < 0.001
    
    def test_golden_file_volatile_market(self, calculator, golden_data):
        """Test volatile market scenario against golden values."""
        data = golden_data['volatile_market']
        
        result = calculator.calculate_safe({
            'portfolio_values': data['portfolio_values'],
            'start_of_day_value': data['portfolio_values'][0]
        })
        
        assert result.is_valid
        
        daily_dd = result.get_value('daily_drawdown')
        assert abs(daily_dd - data['expected_daily_drawdown']) < 0.001
        
        max_dd = result.get_value('max_drawdown')
        assert abs(max_dd - data['expected_max_drawdown']) < 0.001
    
    def test_latency_benchmark_small_dataset(self, calculator):
        """Benchmark latency with small dataset (target: <100µs)."""
        portfolio_values = np.random.normal(100000, 5000, 50)  # 50 data points
        data = {
            'portfolio_values': portfolio_values,
            'start_of_day_value': portfolio_values[0]
        }
        
        # Warm up
        for _ in range(10):
            calculator.calculate_safe(data)
        
        # Benchmark
        latencies = []
        for _ in range(100):
            start_time = time.time_ns()
            result = calculator.calculate_safe(data)
            end_time = time.time_ns()
            
            assert result.is_valid
            latencies.append((end_time - start_time) / 1000.0)  # Convert to µs
        
        # Statistics
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"\nDrawdownCalculator Latency (50 points):")
        print(f"  P50: {p50:.2f}µs")
        print(f"  P95: {p95:.2f}µs")
        print(f"  P99: {p99:.2f}µs")
        
        # Assert performance targets
        assert p50 < 150.0, f"P50 latency {p50:.2f}µs exceeds 150µs target"
        assert p95 < 300.0, f"P95 latency {p95:.2f}µs exceeds 300µs target"
    
    def test_latency_benchmark_large_dataset(self, calculator):
        """Benchmark latency with large dataset (target: <500µs)."""
        portfolio_values = np.random.normal(100000, 5000, 1000)  # 1000 data points
        data = {
            'portfolio_values': portfolio_values,
            'start_of_day_value': portfolio_values[0]
        }
        
        # Warm up
        for _ in range(5):
            calculator.calculate_safe(data)
        
        # Benchmark
        latencies = []
        for _ in range(50):
            start_time = time.time_ns()
            result = calculator.calculate_safe(data)
            end_time = time.time_ns()
            
            assert result.is_valid
            latencies.append((end_time - start_time) / 1000.0)
        
        # Statistics
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        print(f"\nDrawdownCalculator Latency (1000 points):")
        print(f"  P50: {p50:.2f}µs")
        print(f"  P95: {p95:.2f}µs")
        
        # Assert performance targets for larger datasets
        assert p50 < 500.0, f"P50 latency {p50:.2f}µs exceeds 500µs target"
        assert p95 < 1000.0, f"P95 latency {p95:.2f}µs exceeds 1000µs target"
    
    def test_edge_cases(self, calculator):
        """Test edge cases and error handling."""
        # Empty data
        result = calculator.calculate_safe({'portfolio_values': np.array([])})
        assert not result.is_valid
        assert 'empty' in result.error_message.lower()
        
        # Single data point
        result = calculator.calculate_safe({
            'portfolio_values': np.array([100000]),
            'start_of_day_value': 100000
        })
        assert result.is_valid
        assert result.get_value('daily_drawdown') == 0.0
        
        # NaN values
        result = calculator.calculate_safe({
            'portfolio_values': np.array([100000, np.nan, 98000]),
            'start_of_day_value': 100000
        })
        assert not result.is_valid
        assert 'nan' in result.error_message.lower()


class TestTurnoverCalculator:
    """Test TurnoverCalculator with golden-file tests and latency benchmarks."""
    
    @pytest.fixture
    def calculator(self):
        """Create TurnoverCalculator instance."""
        config = {
            'hourly_window_minutes': 60,
            'daily_window_hours': 24,
            'use_absolute_values': True
        }
        return TurnoverCalculator(config)
    
    @pytest.fixture
    def golden_data(self):
        """Load golden test data."""
        base_time = datetime.now()
        return {
            'steady_trading': {
                'trade_values': [10000, 15000, 12000, 8000, 20000],
                'trade_timestamps': [base_time + timedelta(minutes=i*10) for i in range(5)],
                'capital_base': 1000000,
                'expected_total_turnover': 65000,
                'expected_turnover_ratio': 0.065  # 6.5%
            },
            'burst_trading': {
                'trade_values': [100000, 150000, 200000],
                'trade_timestamps': [base_time + timedelta(minutes=i*5) for i in range(3)],
                'capital_base': 1000000,
                'expected_total_turnover': 450000,
                'expected_turnover_ratio': 0.45  # 45%
            },
            'no_trading': {
                'trade_values': [],
                'trade_timestamps': [],
                'capital_base': 1000000,
                'expected_total_turnover': 0,
                'expected_turnover_ratio': 0.0
            }
        }
    
    def test_golden_file_steady_trading(self, calculator, golden_data):
        """Test steady trading scenario against golden values."""
        data = golden_data['steady_trading']
        
        result = calculator.calculate_safe({
            'trade_values': data['trade_values'],
            'trade_timestamps': data['trade_timestamps'],
            'capital_base': data['capital_base']
        })
        
        assert result.is_valid, f"Calculation failed: {result.error_message}"
        
        # Check total turnover
        total_turnover = result.get_value('total_turnover')
        assert abs(total_turnover - data['expected_total_turnover']) < 0.01, \
            f"Total turnover mismatch: {total_turnover} vs {data['expected_total_turnover']}"
        
        # Check turnover ratio
        turnover_ratio = result.get_value('total_turnover_ratio')
        assert abs(turnover_ratio - data['expected_turnover_ratio']) < 0.001, \
            f"Turnover ratio mismatch: {turnover_ratio} vs {data['expected_turnover_ratio']}"
    
    def test_golden_file_burst_trading(self, calculator, golden_data):
        """Test burst trading scenario against golden values."""
        data = golden_data['burst_trading']
        
        result = calculator.calculate_safe({
            'trade_values': data['trade_values'],
            'trade_timestamps': data['trade_timestamps'],
            'capital_base': data['capital_base']
        })
        
        assert result.is_valid
        
        total_turnover = result.get_value('total_turnover')
        assert abs(total_turnover - data['expected_total_turnover']) < 0.01
        
        turnover_ratio = result.get_value('total_turnover_ratio')
        assert abs(turnover_ratio - data['expected_turnover_ratio']) < 0.001
    
    def test_golden_file_no_trading(self, calculator, golden_data):
        """Test no trading scenario against golden values."""
        data = golden_data['no_trading']
        
        result = calculator.calculate_safe({
            'trade_values': data['trade_values'],
            'trade_timestamps': data['trade_timestamps'],
            'capital_base': data['capital_base']
        })
        
        assert result.is_valid
        
        total_turnover = result.get_value('total_turnover')
        assert total_turnover == data['expected_total_turnover']
        
        turnover_ratio = result.get_value('total_turnover_ratio')
        assert turnover_ratio == data['expected_turnover_ratio']
    
    def test_latency_benchmark_small_dataset(self, calculator):
        """Benchmark latency with small dataset (target: <50µs)."""
        base_time = datetime.now()
        trade_values = [np.random.uniform(1000, 50000) for _ in range(20)]
        trade_timestamps = [base_time + timedelta(minutes=i) for i in range(20)]
        
        data = {
            'trade_values': trade_values,
            'trade_timestamps': trade_timestamps,
            'capital_base': 1000000
        }
        
        # Warm up
        for _ in range(10):
            calculator.calculate_safe(data)
        
        # Benchmark
        latencies = []
        for _ in range(100):
            start_time = time.time_ns()
            result = calculator.calculate_safe(data)
            end_time = time.time_ns()
            
            assert result.is_valid
            latencies.append((end_time - start_time) / 1000.0)
        
        # Statistics
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"\nTurnoverCalculator Latency (20 trades):")
        print(f"  P50: {p50:.2f}µs")
        print(f"  P95: {p95:.2f}µs")
        print(f"  P99: {p99:.2f}µs")
        
        # Assert performance targets
        assert p50 < 100.0, f"P50 latency {p50:.2f}µs exceeds 100µs target"
        assert p95 < 200.0, f"P95 latency {p95:.2f}µs exceeds 200µs target"
    
    def test_latency_benchmark_large_dataset(self, calculator):
        """Benchmark latency with large dataset (target: <200µs)."""
        base_time = datetime.now()
        trade_values = [np.random.uniform(1000, 50000) for _ in range(500)]
        trade_timestamps = [base_time + timedelta(seconds=i*10) for i in range(500)]
        
        data = {
            'trade_values': trade_values,
            'trade_timestamps': trade_timestamps,
            'capital_base': 1000000
        }
        
        # Warm up
        for _ in range(5):
            calculator.calculate_safe(data)
        
        # Benchmark
        latencies = []
        for _ in range(50):
            start_time = time.time_ns()
            result = calculator.calculate_safe(data)
            end_time = time.time_ns()
            
            assert result.is_valid
            latencies.append((end_time - start_time) / 1000.0)
        
        # Statistics
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        print(f"\nTurnoverCalculator Latency (500 trades):")
        print(f"  P50: {p50:.2f}µs")
        print(f"  P95: {p95:.2f}µs")
        
        # Assert performance targets for larger datasets
        assert p50 < 300.0, f"P50 latency {p50:.2f}µs exceeds 300µs target"
        assert p95 < 600.0, f"P95 latency {p95:.2f}µs exceeds 600µs target"
    
    def test_edge_cases(self, calculator):
        """Test edge cases and error handling."""
        # Missing capital base
        result = calculator.calculate_safe({
            'trade_values': [10000, 20000],
            'trade_timestamps': [datetime.now(), datetime.now()]
        })
        assert not result.is_valid
        assert 'capital_base' in result.error_message.lower()
        
        # Mismatched arrays
        result = calculator.calculate_safe({
            'trade_values': [10000, 20000],
            'trade_timestamps': [datetime.now()],  # Only one timestamp
            'capital_base': 1000000
        })
        assert not result.is_valid
        assert 'mismatch' in result.error_message.lower()
        
        # Zero capital base
        result = calculator.calculate_safe({
            'trade_values': [10000],
            'trade_timestamps': [datetime.now()],
            'capital_base': 0
        })
        assert not result.is_valid
        assert 'zero' in result.error_message.lower()


class TestCalculatorPerformance:
    """Performance tests across all calculators."""
    
    def test_calculator_initialization_latency(self):
        """Test calculator initialization latency."""
        configs = [
            {'lookback_periods': [1, 5, 20]},
            {'hourly_window_minutes': 60}
        ]
        
        calculators = [DrawdownCalculator, TurnoverCalculator]
        
        for calc_class, config in zip(calculators, configs):
            latencies = []
            
            for _ in range(50):
                start_time = time.time_ns()
                calc = calc_class(config)
                end_time = time.time_ns()
                
                latencies.append((end_time - start_time) / 1000.0)
            
            p50 = np.percentile(latencies, 50)
            print(f"\n{calc_class.__name__} Initialization Latency: {p50:.2f}µs")
            
            # Initialization should be fast
            assert p50 < 1000.0, f"{calc_class.__name__} initialization too slow: {p50:.2f}µs"
    
    def test_memory_usage(self):
        """Test memory usage of calculators."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple calculators
        calculators = []
        for _ in range(100):
            calc1 = DrawdownCalculator({'lookback_periods': [1, 5, 20]})
            calc2 = TurnoverCalculator({'hourly_window_minutes': 60})
            calculators.extend([calc1, calc2])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"\nMemory usage for 200 calculators: {memory_increase:.2f} MB")
        
        # Should not use excessive memory
        assert memory_increase < 50.0, f"Excessive memory usage: {memory_increase:.2f} MB"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])