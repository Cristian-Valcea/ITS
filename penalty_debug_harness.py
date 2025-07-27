import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_turnover_penalty(total_turnover, portfolio_value, target_ratio=0.02, weight=50, smoothness=10):
    epsilon = 1e-6  # prevent division by zero
    normalized = total_turnover / (portfolio_value + epsilon)
    penalty = -weight * sigmoid(smoothness * (normalized - target_ratio) / target_ratio)
    return normalized, penalty

# Simulate different trading scenarios
portfolio_value = 50_000
turnovers = np.linspace(0, 100_000, 200)  # simulate up to 2x turnover

normalized_list = []
penalty_list = []

for t in turnovers:
    norm, pen = compute_turnover_penalty(
        total_turnover=t,
        portfolio_value=portfolio_value,
        target_ratio=0.02,  # 2% daily
        weight=50,          # tunable
        smoothness=10       # controls curve sharpness
    )
    normalized_list.append(norm)
    penalty_list.append(pen)

# Print results instead of plotting (to avoid display issues)
print("🎯 TURNOVER PENALTY DEBUG HARNESS")
print("=" * 50)
print(f"Portfolio Value: ${portfolio_value:,}")
print(f"Target Ratio: {0.02:.1%}")
print(f"Weight: {50}")
print(f"Smoothness: {10}")
print()

# Show key points
key_points = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
print("Turnover  | Normalized | Penalty")
print("----------|------------|--------")
for turnover in key_points:
    norm, pen = compute_turnover_penalty(turnover, portfolio_value, 0.02, 50, 10)
    print(f"${turnover:>8,} | {norm:8.4f} | {pen:6.2f}")

print()
print("🔍 KEY OBSERVATIONS:")
print("-" * 30)

# Find penalty at target
target_turnover = 0.02 * portfolio_value
norm_target, pen_target = compute_turnover_penalty(target_turnover, portfolio_value, 0.02, 50, 10)
print(f"At target (2%): Turnover=${target_turnover:,.0f}, Penalty={pen_target:.2f}")

# Find penalty at 2x target
double_target = 2 * target_turnover
norm_double, pen_double = compute_turnover_penalty(double_target, portfolio_value, 0.02, 50, 10)
print(f"At 2x target: Turnover=${double_target:,.0f}, Penalty={pen_double:.2f}")

# Find penalty at 5x target
five_target = 5 * target_turnover
norm_five, pen_five = compute_turnover_penalty(five_target, portfolio_value, 0.02, 50, 10)
print(f"At 5x target: Turnover=${five_target:,.0f}, Penalty={pen_five:.2f}")

print()
print("✅ Expected behavior:")
print("- Penalty ≈ -25 near target (sigmoid midpoint)")
print("- Penalty approaches -50 as turnover increases")
print("- Penalty approaches 0 as turnover decreases")
print("- Smooth curve without cliffs")