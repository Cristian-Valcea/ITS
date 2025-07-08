#!/usr/bin/env python3
"""
Final integration test demonstrating all risk calculators working together
in a realistic trading scenario.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.risk.calculators import (
    VaRCalculator, GreeksCalculator, VolatilityCalculator, ConcentrationCalculator
)

def main():
    print("🎯 Final Risk Calculators Integration Test")
    print("=" * 60)
    
    # Simulate a realistic trading portfolio
    print("\n📊 Simulating Trading Portfolio...")
    
    # Generate realistic market data
    np.random.seed(42)
    
    # Stock portfolio with concentration
    portfolio_positions = {
        'AAPL': 500000,   # Large tech position
        'MSFT': 300000,   # Another tech position  
        'GOOGL': 200000,  # Third tech position
        'JPM': 150000,    # Financial sector
        'JNJ': 100000,    # Healthcare
        'XOM': 75000,     # Energy
        'TSLA': 50000,    # Growth stock
    }
    
    # Sector mapping
    sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'JPM': 'Financials', 'JNJ': 'Healthcare', 'XOM': 'Energy', 'TSLA': 'Technology'
    }
    
    # Generate price history (252 trading days)
    returns = np.random.normal(0.0005, 0.018, 252)  # Slight positive drift, 1.8% daily vol
    returns[50] = -0.05   # Market correction
    returns[150] = -0.03  # Another down day
    prices = 100 * np.exp(np.cumsum(returns))
    
    total_portfolio_value = sum(portfolio_positions.values())
    
    print(f"✓ Portfolio Value: ${total_portfolio_value:,.0f}")
    print(f"✓ Positions: {len(portfolio_positions)}")
    print(f"✓ Price History: {len(prices)} days")
    
    # 1. CONCENTRATION RISK ANALYSIS
    print(f"\n🎯 1. CONCENTRATION RISK ANALYSIS")
    print("-" * 40)
    
    conc_calc = ConcentrationCalculator({
        'top_n_positions': [3, 5],
        'concentration_thresholds': [0.15, 0.25, 0.35],
        'include_sector_analysis': True
    })
    
    conc_result = conc_calc.calculate_safe({
        'positions': portfolio_positions,
        'sectors': sectors
    })
    
    if conc_result.is_valid:
        print(f"✓ Max Single Position: {conc_result.get_value('max_single_name_pct'):.1%} ({conc_result.get_value('max_single_name_symbol')})")
        print(f"✓ Top 3 Concentration: {conc_result.get_value('top_3_concentration_pct'):.1%}")
        print(f"✓ HHI Index: {conc_result.get_value('hhi'):.3f}")
        print(f"✓ Max Sector: {conc_result.get_value('max_sector_concentration_pct'):.1%} ({conc_result.get_value('max_sector_name')})")
        print(f"✓ Risk Level: {conc_result.get_value('concentration_risk_level')}")
        print(f"✓ Calculation Time: {conc_result.get_calculation_time_us():.1f}µs")
    else:
        print(f"❌ Concentration calculation failed: {conc_result.error_message}")
        return False
    
    # 2. VOLATILITY ANALYSIS
    print(f"\n🎯 2. VOLATILITY ANALYSIS")
    print("-" * 40)
    
    vol_calc = VolatilityCalculator({
        'method': 'all',  # Calculate all methods
        'window_days': 60,
        'ewma_lambda': 0.94
    })
    
    vol_result = vol_calc.calculate_safe({
        'prices': prices
    })
    
    if vol_result.is_valid:
        print(f"✓ Realized Volatility: {vol_result.get_value('realized_volatility'):.1%}")
        print(f"✓ EWMA Volatility: {vol_result.get_value('ewma_volatility'):.1%}")
        print(f"✓ GARCH Volatility: {vol_result.get_value('garch_volatility'):.1%}")
        print(f"✓ Current Return: {vol_result.get_value('current_return'):.3%}")
        print(f"✓ Return Skewness: {vol_result.get_value('return_skewness'):.3f}")
        print(f"✓ Calculation Time: {vol_result.get_calculation_time_us():.1f}µs")
    else:
        print(f"❌ Volatility calculation failed: {vol_result.error_message}")
        return False
    
    # 3. VAR ANALYSIS
    print(f"\n🎯 3. VALUE AT RISK ANALYSIS")
    print("-" * 40)
    
    var_calc = VaRCalculator({
        'confidence_levels': [0.95, 0.99, 0.995],
        'method': 'parametric',
        'distribution': 'normal'
    })
    
    var_result = var_calc.calculate_safe({
        'returns': returns,
        'portfolio_value': total_portfolio_value
    })
    
    if var_result.is_valid:
        print(f"✓ VaR 95%: ${var_result.get_value('var_95'):,.0f} ({var_result.get_value('var_95')/total_portfolio_value:.2%})")
        print(f"✓ VaR 99%: ${var_result.get_value('var_99'):,.0f} ({var_result.get_value('var_99')/total_portfolio_value:.2%})")
        var_995 = var_result.get_value('var_995')
        if var_995 is not None:
            print(f"✓ VaR 99.5%: ${var_995:,.0f} ({var_995/total_portfolio_value:.2%})")
        else:
            print(f"✓ VaR 99.5%: Not calculated")
        print(f"✓ Portfolio Volatility: {var_result.get_value('volatility'):.1%}")
        print(f"✓ Calculation Time: {var_result.get_calculation_time_us():.1f}µs")
    else:
        print(f"❌ VaR calculation failed: {var_result.error_message}")
        return False
    
    # 4. OPTIONS GREEKS ANALYSIS (Simulated options portfolio)
    print(f"\n🎯 4. OPTIONS GREEKS ANALYSIS")
    print("-" * 40)
    
    greeks_calc = GreeksCalculator({
        'risk_free_rate': 0.05,
        'dividend_yield': 0.02
    })
    
    # Simulate options portfolio
    current_price = prices[-1]  # Current stock price
    
    greeks_result = greeks_calc.calculate_safe({
        'spot_price': [current_price, current_price, current_price],
        'strike_price': [current_price * 0.95, current_price, current_price * 1.05],  # ITM, ATM, OTM
        'time_to_expiry': [30, 30, 30],  # 30 days to expiry
        'volatility': [vol_result.get_value('realized_volatility')] * 3,
        'option_type': ['call', 'call', 'put'],
        'position_size': [100, -50, 75]  # Long 100 calls, short 50 calls, long 75 puts
    })
    
    if greeks_result.is_valid:
        print(f"✓ Portfolio Delta: {greeks_result.get_value('portfolio_delta'):.2f}")
        print(f"✓ Portfolio Gamma: {greeks_result.get_value('portfolio_gamma'):.4f}")
        print(f"✓ Portfolio Vega: {greeks_result.get_value('portfolio_vega'):.2f}")
        print(f"✓ Portfolio Theta: {greeks_result.get_value('portfolio_theta'):.2f}")
        print(f"✓ Portfolio Value: ${greeks_result.get_value('portfolio_value'):,.2f}")
        print(f"✓ Delta Neutral: {greeks_result.get_value('delta_neutral')}")
        print(f"✓ Calculation Time: {greeks_result.get_calculation_time_us():.1f}µs")
    else:
        print(f"❌ Greeks calculation failed: {greeks_result.error_message}")
        return False
    
    # 5. INTEGRATED RISK SUMMARY
    print(f"\n🎯 5. INTEGRATED RISK SUMMARY")
    print("-" * 40)
    
    # Calculate total risk metrics
    stock_var_99 = var_result.get_value('var_99')
    options_value = greeks_result.get_value('portfolio_value')
    total_exposure = total_portfolio_value + abs(options_value)
    
    concentration_risk = conc_result.get_value('concentration_risk_level')
    volatility_level = vol_result.get_value('realized_volatility')
    
    # Risk scoring
    risk_score = 0
    risk_factors = []
    
    if concentration_risk in ['HIGH', 'MEDIUM']:
        risk_score += 2 if concentration_risk == 'HIGH' else 1
        risk_factors.append(f"Concentration Risk: {concentration_risk}")
    
    if volatility_level > 0.25:  # 25% annual vol threshold
        risk_score += 2
        risk_factors.append(f"High Volatility: {volatility_level:.1%}")
    elif volatility_level > 0.20:
        risk_score += 1
        risk_factors.append(f"Elevated Volatility: {volatility_level:.1%}")
    
    if stock_var_99 / total_portfolio_value > 0.05:  # 5% VaR threshold
        risk_score += 2
        risk_factors.append(f"High VaR: {stock_var_99/total_portfolio_value:.1%}")
    
    if not greeks_result.get_value('delta_neutral'):
        delta_exposure = abs(greeks_result.get_value('portfolio_delta'))
        if delta_exposure > 50:
            risk_score += 1
            risk_factors.append(f"Delta Exposure: {delta_exposure:.0f}")
    
    # Overall risk assessment
    if risk_score >= 5:
        overall_risk = "HIGH"
    elif risk_score >= 3:
        overall_risk = "MEDIUM"
    elif risk_score >= 1:
        overall_risk = "LOW"
    else:
        overall_risk = "MINIMAL"
    
    print(f"✓ Total Portfolio Exposure: ${total_exposure:,.0f}")
    print(f"✓ Stock Portfolio VaR (99%): ${stock_var_99:,.0f} ({stock_var_99/total_portfolio_value:.2%})")
    print(f"✓ Options Portfolio Value: ${options_value:,.2f}")
    print(f"✓ Overall Risk Level: {overall_risk}")
    print(f"✓ Risk Score: {risk_score}/8")
    
    if risk_factors:
        print(f"✓ Risk Factors:")
        for factor in risk_factors:
            print(f"  - {factor}")
    else:
        print(f"✓ No significant risk factors identified")
    
    # 6. PERFORMANCE SUMMARY
    print(f"\n🎯 6. PERFORMANCE SUMMARY")
    print("-" * 40)
    
    total_calc_time = (
        conc_result.get_calculation_time_us() +
        vol_result.get_calculation_time_us() +
        var_result.get_calculation_time_us() +
        greeks_result.get_calculation_time_us()
    )
    
    print(f"✓ Total Calculation Time: {total_calc_time:.1f}µs")
    print(f"✓ Concentration: {conc_result.get_calculation_time_us():.1f}µs")
    print(f"✓ Volatility: {vol_result.get_calculation_time_us():.1f}µs")
    print(f"✓ VaR: {var_result.get_calculation_time_us():.1f}µs")
    print(f"✓ Greeks: {greeks_result.get_calculation_time_us():.1f}µs")
    
    # Performance validation
    if total_calc_time < 2000:  # 2ms total
        print(f"✅ PERFORMANCE TARGET MET: {total_calc_time:.1f}µs < 2000µs")
    else:
        print(f"⚠️  Performance target missed: {total_calc_time:.1f}µs > 2000µs")
    
    print(f"\n🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print(f"All risk calculators are working together seamlessly.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)