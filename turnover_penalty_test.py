#!/usr/bin/env python3
"""
CLI Test Tool for Turnover Penalty Curve
Simulates different turnover levels and visualizes penalty behavior
"""

import sys
import os
sys.path.append('src')

import numpy as np
import argparse
from src.gym_env.components.turnover_penalty import TurnoverPenaltyCalculator
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)

def calculate_turnover_penalty(normalized_turnover, target_turnover=0.02, adaptive_weight=100.0, slope=50):
    """
    Wrapper function that matches the expected interface
    Uses our TurnoverPenaltyCalculator component internally
    """
    # Create a temporary calculator
    portfolio_value = 50000.0
    episode_length = 100
    
    calc = TurnoverPenaltyCalculator(
        episode_length=episode_length,
        portfolio_value_getter=lambda: portfolio_value,
        target_range=target_turnover,
        adaptive_weight_factor=adaptive_weight / portfolio_value,  # Convert to factor
        smoothness=slope,
        curve='sigmoid'
    )
    
    # Convert normalized turnover back to absolute turnover
    absolute_turnover = normalized_turnover * portfolio_value * episode_length
    
    # Calculate penalty
    penalty = calc.compute_penalty(absolute_turnover)
    
    return penalty

def test_penalty_curve(target_turnover=0.02, adaptive_weight=100.0, slope=50, max_turnover=0.05, num_points=100, save_plot=True, show_plot=False):
    """Test and visualize the penalty curve"""
    
    print("🎯 TURNOVER PENALTY CURVE TEST TOOL")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   Target Turnover: {target_turnover:.1%}")
    print(f"   Adaptive Weight: {adaptive_weight:.1f}")
    print(f"   Slope (Smoothness): {slope:.1f}")
    print(f"   Max Turnover: {max_turnover:.1%}")
    print(f"   Test Points: {num_points}")
    print()
    
    # Generate test range
    turnovers = np.linspace(0, max_turnover, num_points)
    penalties = []
    
    print("🔄 Computing penalty curve...")
    
    # Simulate penalties
    for turnover in turnovers:
        penalty = calculate_turnover_penalty(
            normalized_turnover=turnover,
            target_turnover=target_turnover,
            adaptive_weight=adaptive_weight,
            slope=slope
        )
        penalties.append(penalty)
    
    # Print key values
    print("\n📈 KEY PENALTY VALUES:")
    print("-" * 40)
    print("Turnover | Penalty")
    print("---------|--------")
    
    key_points = [0.0, target_turnover/2, target_turnover, target_turnover*1.5, target_turnover*2, target_turnover*3, max_turnover]
    
    for point in key_points:
        if point <= max_turnover:
            # Find closest index
            idx = np.argmin(np.abs(turnovers - point))
            penalty = penalties[idx]
            print(f"{point:6.1%} | {penalty:6.2f}")
    
    # Analysis
    print(f"\n🔍 CURVE ANALYSIS:")
    print("-" * 40)
    
    # Find penalty at target
    target_idx = np.argmin(np.abs(turnovers - target_turnover))
    target_penalty = penalties[target_idx]
    
    # Find penalty at zero
    zero_penalty = penalties[0]
    
    # Find penalty at max
    max_penalty = penalties[-1]
    
    # Find penalty range
    penalty_range = max_penalty - zero_penalty
    
    print(f"Penalty at 0%: {zero_penalty:.2f}")
    print(f"Penalty at target ({target_turnover:.1%}): {target_penalty:.2f}")
    print(f"Penalty at max ({max_turnover:.1%}): {max_penalty:.2f}")
    print(f"Penalty range: {penalty_range:.2f}")
    
    # Check curve properties
    print(f"\n✅ CURVE PROPERTIES:")
    print("-" * 40)
    
    # Check if curve is smooth (no big jumps)
    penalty_diffs = np.diff(penalties)
    max_jump = np.max(np.abs(penalty_diffs))
    print(f"Max penalty jump: {max_jump:.4f} (should be small)")
    
    # Check if penalty increases with turnover
    is_monotonic = np.all(penalty_diffs <= 0.001)  # Allow small numerical errors
    print(f"Monotonic decrease: {'✅' if is_monotonic else '❌'}")
    
    # Check sigmoid behavior
    sigmoid_midpoint = target_penalty
    expected_midpoint = adaptive_weight / 2
    midpoint_close = abs(sigmoid_midpoint + expected_midpoint) < expected_midpoint * 0.1
    print(f"Sigmoid midpoint correct: {'✅' if midpoint_close else '❌'}")
    
    # Try to create plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(turnovers * 100, penalties, 'b-', linewidth=2, label="Turnover Penalty")
        plt.axvline(target_turnover * 100, color="green", linestyle="--", linewidth=2, label=f"Target Turnover ({target_turnover:.1%})")
        plt.axhline(target_penalty, color="red", linestyle=":", alpha=0.7, label=f"Target Penalty ({target_penalty:.1f})")
        
        plt.title(f"Penalty Curve (Weight={adaptive_weight}, Slope={slope})")
        plt.xlabel("Normalized Turnover (%)")
        plt.ylabel("Penalty")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_plot:
            filename = f"penalty_curve_w{adaptive_weight}_s{slope}_t{target_turnover*100:.0f}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\n💾 Plot saved: {filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except ImportError:
        print("\n⚠️  Matplotlib not available - skipping plot generation")
    except Exception as e:
        print(f"\n❌ Plot generation failed: {e}")
    
    print(f"\n✅ PENALTY CURVE TEST COMPLETE!")
    
    return turnovers, penalties

def compare_configurations():
    """Compare different penalty configurations"""
    
    print("\n" + "="*60)
    print("🔬 CONFIGURATION COMPARISON")
    print("="*60)
    
    configs = [
        {"name": "Conservative", "weight": 50, "slope": 10, "target": 0.02},
        {"name": "Moderate", "weight": 100, "slope": 20, "target": 0.02},
        {"name": "Aggressive", "weight": 200, "slope": 50, "target": 0.02},
        {"name": "High Target", "weight": 100, "slope": 20, "target": 0.05},
    ]
    
    test_turnovers = [0.01, 0.02, 0.03, 0.05, 0.08]
    
    print("Config      | 1.0%  | 2.0%  | 3.0%  | 5.0%  | 8.0%")
    print("------------|-------|-------|-------|-------|-------")
    
    for config in configs:
        penalties = []
        for turnover in test_turnovers:
            penalty = calculate_turnover_penalty(
                normalized_turnover=turnover,
                target_turnover=config["target"],
                adaptive_weight=config["weight"],
                slope=config["slope"]
            )
            penalties.append(penalty)
        
        penalty_str = " | ".join([f"{p:5.1f}" for p in penalties])
        print(f"{config['name']:11s} | {penalty_str}")
    
    print("\n💡 Use this comparison to choose optimal parameters!")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Test and visualize turnover penalty curves")
    parser.add_argument("--target", type=float, default=0.02, help="Target turnover ratio (default: 0.02)")
    parser.add_argument("--weight", type=float, default=100.0, help="Adaptive weight (default: 100.0)")
    parser.add_argument("--slope", type=float, default=50.0, help="Sigmoid slope/smoothness (default: 50.0)")
    parser.add_argument("--max-turnover", type=float, default=0.05, help="Maximum turnover for testing (default: 0.05)")
    parser.add_argument("--points", type=int, default=100, help="Number of test points (default: 100)")
    parser.add_argument("--no-save", action="store_true", help="Don't save plot to file")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--compare", action="store_true", help="Run configuration comparison")
    
    args = parser.parse_args()
    
    # Run main test
    test_penalty_curve(
        target_turnover=args.target,
        adaptive_weight=args.weight,
        slope=args.slope,
        max_turnover=args.max_turnover,
        num_points=args.points,
        save_plot=not args.no_save,
        show_plot=args.show
    )
    
    # Run comparison if requested
    if args.compare:
        compare_configurations()

if __name__ == "__main__":
    main()