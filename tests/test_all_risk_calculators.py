#!/usr/bin/env python3
"""
Comprehensive test suite for all risk calculators.
Tests VaR, Greeks, Volatility, and Concentration calculators.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.risk.calculators import (
    VaRCalculator, GreeksCalculator, VolatilityCalculator, ConcentrationCalculator
)


class TestRunner:
    """Simple test runner for risk calculators."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name):
        """Decorator for test functions."""
        def decorator(func):
            self.tests.append((name, func))
            return func
        return decorator
    
    def run_all(self):
        """Run all tests."""
        print("🧮 Risk Calculators Test Suite")
        print("=" * 60)
        
        for name, test_func in self.tests:
            print(f"\n🧪 {name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                test_func()
                duration = time.time() - start_time
                print(f"✅ PASSED ({duration:.3f}s)")
                self.passed += 1
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ FAILED ({duration:.3f}s): {e}")
                import traceback
                traceback.print_exc()
                self.failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Total:  {self.passed + self.failed}")
        
        return self.failed == 0


# Create test runner
runner = TestRunner()


@runner.test("VaR Calculator - Parametric Method")
def test_var_parametric():
    """Test VaR calculator with parametric method."""
    calculator = VaRCalculator({
        'confidence_levels': [0.95, 0.99],
        'method': 'parametric',
        'distribution': 'normal',
        'window_days': 100
    })
    
    # Generate sample returns (daily)
    np.random.seed(42)
    returns = np.random.normal(-0.001, 0.02, 150)  # Slight negative drift, 2% daily vol
    
    result = calculator.calculate_safe({
        'returns': returns,
        'portfolio_value': 1000000
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    var_95 = result.get_value('var_95')
    var_99 = result.get_value('var_99')
    
    assert var_95 > 0, f"VaR 95% should be positive: {var_95}"
    assert var_99 > var_95, f"VaR 99% should be higher than VaR 95%: {var_99} vs {var_95}"
    
    print(f"✓ VaR 95%: ${var_95:,.0f}")
    print(f"✓ VaR 99%: ${var_99:,.0f}")
    print(f"✓ Volatility: {result.get_value('volatility'):.3f}")


@runner.test("VaR Calculator - Historical Method")
def test_var_historical():
    """Test VaR calculator with historical method."""
    calculator = VaRCalculator({
        'confidence_levels': [0.95, 0.99],
        'method': 'historical',
        'window_days': 100
    })
    
    # Generate sample returns with some extreme events
    np.random.seed(42)
    returns = np.random.normal(-0.001, 0.02, 150)
    returns[50] = -0.08  # Add a crash day
    returns[100] = -0.05  # Add another bad day
    
    result = calculator.calculate_safe({
        'returns': returns,
        'portfolio_value': 1000000
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    var_95 = result.get_value('var_95')
    var_99 = result.get_value('var_99')
    
    assert var_95 > 0, f"Historical VaR 95% should be positive: {var_95}"
    assert var_99 > var_95, f"Historical VaR 99% should be higher than VaR 95%"
    
    print(f"✓ Historical VaR 95%: ${var_95:,.0f}")
    print(f"✓ Historical VaR 99%: ${var_99:,.0f}")


@runner.test("Greeks Calculator - Single Call Option")
def test_greeks_single_call():
    """Test Greeks calculator for a single call option."""
    calculator = GreeksCalculator({
        'risk_free_rate': 0.05,
        'dividend_yield': 0.02
    })
    
    result = calculator.calculate_safe({
        'spot_price': 100.0,
        'strike_price': 105.0,
        'time_to_expiry': 30,  # 30 days
        'volatility': 0.25,    # 25% vol
        'option_type': 'call',
        'position_size': 100   # 100 contracts
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    delta = result.get_value('delta')
    gamma = result.get_value('gamma')
    vega = result.get_value('vega')
    theta = result.get_value('theta')
    option_value = result.get_value('option_value')
    
    assert 0 < delta < 100, f"Call delta should be between 0 and 100: {delta}"
    assert gamma > 0, f"Gamma should be positive: {gamma}"
    assert vega > 0, f"Vega should be positive: {vega}"
    assert theta < 0, f"Theta should be negative (time decay): {theta}"
    assert option_value > 0, f"Call option value should be positive: {option_value}"
    
    print(f"✓ Delta: {delta:.2f}")
    print(f"✓ Gamma: {gamma:.4f}")
    print(f"✓ Vega: {vega:.2f}")
    print(f"✓ Theta: {theta:.2f}")
    print(f"✓ Option Value: ${option_value:,.2f}")


@runner.test("Greeks Calculator - Portfolio of Options")
def test_greeks_portfolio():
    """Test Greeks calculator for a portfolio of options."""
    calculator = GreeksCalculator({
        'risk_free_rate': 0.05,
        'dividend_yield': 0.0
    })
    
    result = calculator.calculate_safe({
        'spot_price': [100.0, 100.0, 100.0],
        'strike_price': [95.0, 100.0, 105.0],
        'time_to_expiry': [30, 30, 30],
        'volatility': [0.25, 0.25, 0.25],
        'option_type': ['call', 'call', 'put'],
        'position_size': [100, -50, 75]  # Long 100 calls, short 50 calls, long 75 puts
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    portfolio_delta = result.get_value('portfolio_delta')
    portfolio_gamma = result.get_value('portfolio_gamma')
    portfolio_vega = result.get_value('portfolio_vega')
    portfolio_value = result.get_value('portfolio_value')
    
    assert portfolio_delta is not None, "Portfolio delta should be calculated"
    assert portfolio_gamma is not None, "Portfolio gamma should be calculated"
    assert portfolio_vega is not None, "Portfolio vega should be calculated"
    
    print(f"✓ Portfolio Delta: {portfolio_delta:.2f}")
    print(f"✓ Portfolio Gamma: {portfolio_gamma:.4f}")
    print(f"✓ Portfolio Vega: {portfolio_vega:.2f}")
    print(f"✓ Portfolio Value: ${portfolio_value:,.2f}")
    print(f"✓ Options Count: {result.get_value('options_count')}")


@runner.test("Volatility Calculator - Realized Volatility")
def test_volatility_realized():
    """Test volatility calculator with realized volatility method."""
    calculator = VolatilityCalculator({
        'method': 'realized',
        'window_days': 60,
        'annualization_factor': 252
    })
    
    # Generate sample price series
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 100)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Convert to price series
    
    result = calculator.calculate_safe({
        'prices': prices
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    realized_vol = result.get_value('realized_volatility')
    mean_return = result.get_value('mean_return')
    
    assert realized_vol > 0, f"Realized volatility should be positive: {realized_vol}"
    assert 0.1 < realized_vol < 1.0, f"Realized volatility should be reasonable: {realized_vol}"
    
    print(f"✓ Realized Volatility: {realized_vol:.3f} ({realized_vol*100:.1f}%)")
    print(f"✓ Mean Return: {mean_return:.6f}")
    print(f"✓ Observations: {result.get_value('observations')}")


@runner.test("Volatility Calculator - EWMA Method")
def test_volatility_ewma():
    """Test volatility calculator with EWMA method."""
    calculator = VolatilityCalculator({
        'method': 'ewma',
        'ewma_lambda': 0.94,
        'window_days': 100
    })
    
    # Generate sample returns with volatility clustering
    np.random.seed(42)
    returns = []
    vol = 0.02
    for i in range(150):
        vol = 0.95 * vol + 0.05 * 0.02  # Mean reversion
        returns.append(np.random.normal(0, vol))
    
    returns = np.array(returns)
    
    result = calculator.calculate_safe({
        'returns': returns
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    ewma_vol = result.get_value('ewma_volatility')
    ewma_var = result.get_value('ewma_variance')
    
    assert ewma_vol > 0, f"EWMA volatility should be positive: {ewma_vol}"
    assert ewma_var > 0, f"EWMA variance should be positive: {ewma_var}"
    
    print(f"✓ EWMA Volatility: {ewma_vol:.3f} ({ewma_vol*100:.1f}%)")
    print(f"✓ EWMA Variance: {ewma_var:.6f}")


@runner.test("Concentration Calculator - Single Name")
def test_concentration_single_name():
    """Test concentration calculator for single-name concentration."""
    calculator = ConcentrationCalculator({
        'top_n_positions': [5, 10],
        'concentration_thresholds': [0.1, 0.2, 0.3]
    })
    
    # Create sample portfolio with concentration
    positions = {
        'AAPL': 500000,   # 50% concentration
        'MSFT': 200000,   # 20%
        'GOOGL': 150000,  # 15%
        'TSLA': 100000,   # 10%
        'NVDA': 50000,    # 5%
    }
    
    result = calculator.calculate_safe({
        'positions': positions
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    max_single_pct = result.get_value('max_single_name_pct')
    max_symbol = result.get_value('max_single_name_symbol')
    hhi = result.get_value('hhi')
    risk_level = result.get_value('concentration_risk_level')
    
    assert max_single_pct == 0.5, f"Max single name should be 50%: {max_single_pct}"
    assert max_symbol == 'AAPL', f"Max symbol should be AAPL: {max_symbol}"
    assert hhi > 0, f"HHI should be positive: {hhi}"
    assert risk_level in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'], f"Invalid risk level: {risk_level}"
    
    print(f"✓ Max Single Name: {max_single_pct:.1%} ({max_symbol})")
    print(f"✓ HHI: {hhi:.3f}")
    print(f"✓ Risk Level: {risk_level}")
    print(f"✓ Position Count: {result.get_value('position_count')}")


@runner.test("Concentration Calculator - Sector Analysis")
def test_concentration_sector():
    """Test concentration calculator with sector analysis."""
    calculator = ConcentrationCalculator({
        'include_sector_analysis': True,
        'concentration_thresholds': [0.3, 0.5, 0.7]
    })
    
    positions = {
        'AAPL': 300000,
        'MSFT': 200000,
        'GOOGL': 150000,
        'JPM': 200000,
        'BAC': 100000,
        'XOM': 50000
    }
    
    sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'JPM': 'Financials',
        'BAC': 'Financials',
        'XOM': 'Energy'
    }
    
    result = calculator.calculate_safe({
        'positions': positions,
        'sectors': sectors
    })
    
    assert result.is_valid, f"Calculation failed: {result.error_message}"
    
    max_sector_pct = result.get_value('max_sector_concentration_pct')
    max_sector_name = result.get_value('max_sector_name')
    sector_count = result.get_value('sector_count')
    sector_hhi = result.get_value('sector_hhi')
    
    assert max_sector_pct > 0, f"Max sector concentration should be positive: {max_sector_pct}"
    assert max_sector_name is not None, f"Max sector name should be provided"
    assert sector_count == 3, f"Should have 3 sectors: {sector_count}"
    assert sector_hhi > 0, f"Sector HHI should be positive: {sector_hhi}"
    
    print(f"✓ Max Sector: {max_sector_pct:.1%} ({max_sector_name})")
    print(f"✓ Sector Count: {sector_count}")
    print(f"✓ Sector HHI: {sector_hhi:.3f}")


@runner.test("All Calculators - Performance Benchmark")
def test_all_calculators_performance():
    """Benchmark performance of all calculators."""
    print("Performance benchmarking...")
    
    # VaR Calculator
    var_calc = VaRCalculator({'method': 'parametric'})
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 100)
    
    start_time = time.time_ns()
    for _ in range(100):
        var_calc.calculate_safe({'returns': returns, 'portfolio_value': 1000000})
    var_time = (time.time_ns() - start_time) / 100 / 1000.0  # Average time in µs
    
    # Greeks Calculator
    greeks_calc = GreeksCalculator({})
    
    start_time = time.time_ns()
    for _ in range(100):
        greeks_calc.calculate_safe({
            'spot_price': 100.0,
            'strike_price': 105.0,
            'time_to_expiry': 30,
            'volatility': 0.25,
            'option_type': 'call',
            'position_size': 100
        })
    greeks_time = (time.time_ns() - start_time) / 100 / 1000.0
    
    # Volatility Calculator
    vol_calc = VolatilityCalculator({'method': 'realized'})
    prices = 100 * np.exp(np.cumsum(returns))
    
    start_time = time.time_ns()
    for _ in range(100):
        vol_calc.calculate_safe({'prices': prices})
    vol_time = (time.time_ns() - start_time) / 100 / 1000.0
    
    # Concentration Calculator
    conc_calc = ConcentrationCalculator({})
    positions = {'A': 100000, 'B': 80000, 'C': 60000, 'D': 40000, 'E': 20000}
    
    start_time = time.time_ns()
    for _ in range(100):
        conc_calc.calculate_safe({'positions': positions})
    conc_time = (time.time_ns() - start_time) / 100 / 1000.0
    
    print(f"✓ VaR Calculator: {var_time:.2f}µs")
    print(f"✓ Greeks Calculator: {greeks_time:.2f}µs")
    print(f"✓ Volatility Calculator: {vol_time:.2f}µs")
    print(f"✓ Concentration Calculator: {conc_time:.2f}µs")
    
    # Assert performance targets (generous for complex calculations)
    assert var_time < 1000.0, f"VaR calculation too slow: {var_time:.2f}µs"
    assert greeks_time < 1000.0, f"Greeks calculation too slow: {greeks_time:.2f}µs"
    assert vol_time < 1000.0, f"Volatility calculation too slow: {vol_time:.2f}µs"
    assert conc_time < 1000.0, f"Concentration calculation too slow: {conc_time:.2f}µs"


if __name__ == "__main__":
    success = runner.run_all()
    
    if success:
        print(f"\n🎉 ALL RISK CALCULATORS TESTS PASSED!")
        print("Risk calculators are ready for production use.")
    else:
        print(f"\n❌ Some tests failed. Please review and fix issues.")
        sys.exit(1)