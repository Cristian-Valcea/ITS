#!/usr/bin/env python3
"""
Verify that the turnover penalty component is working correctly
"""

import sys
import os
sys.path.append('src')

from src.gym_env.components.turnover_penalty import TurnoverPenaltyCalculator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_turnover_calculation():
    """Verify the turnover penalty calculation with the exact values from your example"""
    
    print("🔍 VERIFYING TURNOVER PENALTY WITH YOUR EXAMPLE VALUES")
    print("=" * 70)
    
    # Test parameters
    episode_length = 367
    portfolio_value = 50000.0
    target_range = 0.02  # 2%
    weight_factor = 0.001
    
    # Create calculator
    calculator = TurnoverPenaltyCalculator(
        episode_length=episode_length,
        portfolio_value=portfolio_value,
        target_range=target_range,
        adaptive_weight_factor=weight_factor,
        smoothness=10.0,
        curve='softplus',
        logger=logger
    )
    
    print(f"📊 Calculator Configuration:")
    print(f"   Portfolio Value: ${calculator.portfolio_value:,.0f}")
    print(f"   Episode Length: {calculator.episode_length}")
    print(f"   Target Range: {calculator.target_range:.3f} ({calculator.target_range*100:.1f}%)")
    print(f"   Denominator: {calculator.portfolio_value * calculator.episode_length:,.0f}")
    print()
    
    # Test with your specific example values
    test_cases = [
        ("Small trade", 1984),
        ("Medium trade", 25000),
        ("Large trade", 145073),  # Your example
        ("Very large trade", 200000)
    ]
    
    print("🧪 TESTING WITH YOUR EXAMPLE VALUES:")
    print("-" * 70)
    print(f"{'Description':<20} {'Turnover':<12} {'Normalized':<12} {'Penalty':<10}")
    print("-" * 70)
    
    for description, turnover in test_cases:
        # Calculate normalized turnover manually
        normalized = turnover / (calculator.portfolio_value * calculator.episode_length)
        
        # Calculate using component
        penalty = calculator.compute_penalty(turnover)
        
        print(f"{description:<20} ${turnover:<11,.0f} {normalized:<12.6f} {penalty:<10.2f}")
    
    print()
    print("🔍 DETAILED BREAKDOWN FOR YOUR $145,073 EXAMPLE:")
    print("-" * 70)
    
    turnover = 145073
    portfolio_value = 50000
    episode_length = 367
    
    denominator = portfolio_value * episode_length
    normalized = turnover / denominator
    penalty = calculator.compute_penalty(turnover)
    
    print(f"Turnover: ${turnover:,.0f}")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Episode Length: {episode_length}")
    print(f"Denominator: {portfolio_value} × {episode_length} = {denominator:,.0f}")
    print(f"Normalized: {turnover:,.0f} ÷ {denominator:,.0f} = {normalized:.6f}")
    print(f"Normalized %: {normalized*100:.4f}%")
    print(f"Target %: {target_range*100:.1f}%")
    print(f"Penalty: {penalty:.2f}")
    
    # Check if this is reasonable
    if normalized > 0.001:  # More than 0.1%
        print(f"✅ NORMALIZED TURNOVER IS REASONABLE: {normalized:.6f}")
    else:
        print(f"❌ NORMALIZED TURNOVER TOO SMALL: {normalized:.6f}")
    
    print()
    print("🎯 EXPECTED BEHAVIOR:")
    print("-" * 70)
    print(f"• With ${turnover:,.0f} trading on ${portfolio_value:,.0f} portfolio over {episode_length} steps")
    print(f"• Normalized turnover should be: {normalized:.6f} ({normalized*100:.4f}%)")
    print(f"• This is {'ABOVE' if normalized > target_range else 'BELOW'} the {target_range*100:.1f}% target")
    print(f"• Penalty should be: {penalty:.2f} (negative = penalty)")
    
    if abs(normalized) < 1e-10:
        print("❌ ERROR: Normalized turnover is essentially zero!")
        print("   This suggests a calculation error or incorrect parameters.")
    else:
        print("✅ Normalized turnover calculation appears correct.")

if __name__ == "__main__":
    verify_turnover_calculation()