#!/usr/bin/env python3
"""
Export turnover penalty data to CSV for external plotting
"""

import sys
import os
sys.path.append('src')

import numpy as np
import csv
from src.gym_env.components.turnover_penalty import TurnoverPenaltyCalculator

def export_penalty_data(target_turnover=0.02, adaptive_weight=100.0, slope=50, max_turnover=0.05, num_points=100):
    """Export penalty curve data to CSV"""
    
    print("📊 EXPORTING TURNOVER PENALTY DATA")
    print("=" * 40)
    print(f"Target: {target_turnover:.1%}, Weight: {adaptive_weight}, Slope: {slope}")
    
    # Setup calculator
    portfolio_value = 50000.0
    episode_length = 100
    
    calc = TurnoverPenaltyCalculator(
        episode_length=episode_length,
        portfolio_value_getter=lambda: portfolio_value,
        target_range=target_turnover,
        adaptive_weight_factor=adaptive_weight / portfolio_value,
        smoothness=slope,
        curve='sigmoid'
    )
    
    # Generate data
    turnovers = np.linspace(0, max_turnover, num_points)
    penalties = []
    
    for turnover in turnovers:
        absolute_turnover = turnover * portfolio_value * episode_length
        penalty = calc.compute_penalty(absolute_turnover)
        penalties.append(penalty)
    
    # Export to CSV
    filename = f"penalty_data_t{target_turnover*100:.0f}_w{adaptive_weight:.0f}_s{slope:.0f}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Normalized_Turnover', 'Penalty'])
        
        for turnover, penalty in zip(turnovers, penalties):
            writer.writerow([f"{turnover:.6f}", f"{penalty:.4f}"])
    
    print(f"✅ Data exported to: {filename}")
    print(f"📈 {num_points} data points from 0% to {max_turnover:.1%}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Min penalty: {min(penalties):.2f}")
    print(f"   Max penalty: {max(penalties):.2f}")
    print(f"   Penalty at target: {penalties[np.argmin(np.abs(turnovers - target_turnover))]:.2f}")

if __name__ == "__main__":
    # Export a few different configurations
    configs = [
        {"target_turnover": 0.02, "adaptive_weight": 50, "slope": 10},
        {"target_turnover": 0.02, "adaptive_weight": 100, "slope": 20},
        {"target_turnover": 0.01, "adaptive_weight": 75, "slope": 15},
        {"target_turnover": 0.05, "adaptive_weight": 100, "slope": 25},
    ]
    
    for config in configs:
        export_penalty_data(**config)
        print()