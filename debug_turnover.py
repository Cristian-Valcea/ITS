#!/usr/bin/env python3
"""
Debug script to test turnover penalty calculation
"""

import sys
import os
sys.path.append('src')

from src.gym_env.components.turnover_penalty import TurnoverPenaltyCalculator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_turnover_calculation():
    """Test the turnover penalty calculation with realistic values"""
    
    print("🔍 DEBUGGING TURNOVER PENALTY CALCULATION")
    print("=" * 60)
    
    # Test parameters from the log
    episode_length = 367
    initial_portfolio_value = 50000.0
    target_range = 0.02  # 2%
    weight_factor = 0.001
    
    # Create calculator
    calculator = TurnoverPenaltyCalculator(
        episode_length=episode_length,
        portfolio_value=initial_portfolio_value,
        target_range=target_range,
        adaptive_weight_factor=weight_factor,
        smoothness=10.0,
        curve='softplus',
        logger=logger
    )
    
    print(f"📊 Calculator initialized:")
    print(f"   Portfolio Value: ${calculator.portfolio_value:,.0f}")
    print(f"   Episode Length: {calculator.episode_length}")
    print(f"   Target Range: {calculator.target_range:.3f}")
    print(f"   Adaptive Weight: {calculator.adaptive_weight:.2f}")
    print()
    
    # Test with different turnover values
    test_turnovers = [0, 1000, 5000, 10000, 50000, 96196]  # Last one from actual log
    
    print("🧪 TESTING DIFFERENT TURNOVER VALUES:")
    print("-" * 60)
    
    for turnover in test_turnovers:
        # Calculate normalized turnover manually
        manual_normalized = turnover / (calculator.portfolio_value * calculator.episode_length)
        
        # Calculate using component
        penalty = calculator.compute_penalty(turnover)
        
        print(f"Turnover: ${turnover:>8,.0f} | "
              f"Manual Normalized: {manual_normalized:.6f} | "
              f"Penalty: {penalty:>8.2f}")
        
        # Debug log
        calculator.log_debug(turnover)
        print()
    
    print("🔍 DETAILED CALCULATION BREAKDOWN:")
    print("-" * 60)
    
    # Test with the actual value from the log
    actual_turnover = 96196
    portfolio_value = 50000
    episode_length = 367
    
    denominator = portfolio_value * episode_length
    normalized = actual_turnover / denominator
    
    print(f"Actual Turnover: ${actual_turnover:,.0f}")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Episode Length: {episode_length}")
    print(f"Denominator (PV * EL): {denominator:,.0f}")
    print(f"Normalized Turnover: {actual_turnover} / {denominator} = {normalized:.6f}")
    print(f"Expected: {normalized:.6f} (should be ~0.005238)")
    
    # Test if portfolio value is being updated during episode
    print("\n🔄 TESTING PORTFOLIO VALUE UPDATES:")
    print("-" * 60)
    
    # Simulate portfolio value changes during episode
    portfolio_values = [50000, 49500, 49000, 48500, 48000]
    
    for i, pv in enumerate(portfolio_values):
        calculator.update_portfolio_value(pv)
        penalty = calculator.compute_penalty(actual_turnover)
        normalized = actual_turnover / (pv * episode_length)
        
        print(f"Step {i}: PV=${pv:,.0f}, Normalized={normalized:.6f}, Penalty={penalty:.2f}")

if __name__ == "__main__":
    test_turnover_calculation()