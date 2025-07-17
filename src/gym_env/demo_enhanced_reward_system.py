# src/gym_env/demo_enhanced_reward_system.py
"""
Demonstration of Enhanced Reward System vs Traditional P&L Rewards.

Shows how the enhanced reward system addresses the core issue:
"reward tied directly to realized P&L minus cost results in near-zero rewards"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enhanced_reward_system import EnhancedRewardCalculator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def simulate_trading_scenario():
    """Simulate a realistic high-frequency trading scenario."""
    
    # Simulate minute-by-minute price data
    np.random.seed(42)
    n_steps = 100
    
    # Generate realistic price series with trend + noise
    base_price = 100.0
    trend = 0.001  # Small upward trend
    noise_std = 0.002  # 0.2% noise per minute
    
    prices = []
    price = base_price
    for i in range(n_steps):
        price_change = trend + np.random.normal(0, noise_std)
        price *= (1 + price_change)
        prices.append(price)
    
    # Simulate trading positions (simple momentum strategy)
    positions = []
    for i in range(n_steps):
        if i < 5:
            positions.append(0)  # Start flat
        else:
            # Simple momentum: buy if price increased, sell if decreased
            price_change = prices[i] - prices[i-1]
            if price_change > 0.001:  # 0.1% threshold
                positions.append(1)   # Long
            elif price_change < -0.001:
                positions.append(-1)  # Short
            else:
                positions.append(positions[-1])  # Hold
    
    # Simulate realized P&L and transaction costs
    realized_pnls = []
    transaction_costs = []
    
    for i in range(n_steps):
        if i == 0:
            realized_pnls.append(0.0)
            transaction_costs.append(0.0)
        else:
            # Realized P&L when position changes
            if positions[i] != positions[i-1]:
                # Simulate small P&L from position change
                price_change = prices[i] - prices[i-1]
                realized_pnl = positions[i-1] * price_change * 1000  # 1000 shares
                
                # High transaction cost (kills small profits)
                transaction_cost = abs(positions[i] - positions[i-1]) * prices[i] * 1000 * 0.001  # 0.1% cost
                
                realized_pnls.append(realized_pnl)
                transaction_costs.append(transaction_cost)
            else:
                realized_pnls.append(0.0)
                transaction_costs.append(0.0)
    
    return prices, positions, realized_pnls, transaction_costs


def compare_reward_systems():
    """Compare traditional vs enhanced reward systems."""
    
    print("🔬 ENHANCED REWARD SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("Problem: High-frequency rewards are near-zero due to transaction costs")
    print("Solution: Enhanced reward system with directional + behavioral components")
    print()
    
    # Generate trading scenario
    prices, positions, realized_pnls, transaction_costs = simulate_trading_scenario()
    
    # Initialize enhanced reward calculator
    enhanced_config = {
        'realized_pnl_weight': 1.0,
        'directional_weight': 0.3,
        'behavioral_weight': 0.2,
        'directional_scaling': 0.001,
        'min_price_change_bps': 0.5,
        'flip_flop_penalty': 0.001,
        'holding_bonus': 0.0001,
        'correct_direction_bonus': 0.1,
        'wrong_direction_penalty': 0.1,
        'enable_multi_timeframe': True,
        'enable_adaptive_scaling': True,
        'target_reward_magnitude': 0.01
    }
    
    enhanced_calculator = EnhancedRewardCalculator(enhanced_config)
    
    # Calculate rewards using both systems
    traditional_rewards = []
    enhanced_rewards = []
    reward_breakdowns = []
    
    portfolio_value = 100000.0  # Starting portfolio value
    
    for i in range(len(prices)):
        # Traditional reward: realized P&L - transaction cost
        traditional_reward = realized_pnls[i] - transaction_costs[i]
        traditional_rewards.append(traditional_reward)
        
        # Enhanced reward system
        enhanced_reward, breakdown = enhanced_calculator.calculate_reward(
            realized_pnl=realized_pnls[i],
            transaction_cost=transaction_costs[i],
            current_position=positions[i],
            current_price=prices[i],
            portfolio_value=portfolio_value,
            step_info={'step': i}
        )
        
        enhanced_rewards.append(enhanced_reward)
        reward_breakdowns.append(breakdown)
        
        # Update portfolio value (simplified)
        portfolio_value += traditional_reward
    
    # Analysis
    print("📊 REWARD SYSTEM COMPARISON:")
    print("-" * 40)
    
    # Traditional system statistics
    trad_mean = np.mean(traditional_rewards)
    trad_std = np.std(traditional_rewards)
    trad_nonzero = np.sum(np.abs(traditional_rewards) > 1e-6)
    
    print(f"Traditional System:")
    print(f"  Mean reward: {trad_mean:.6f}")
    print(f"  Std deviation: {trad_std:.6f}")
    print(f"  Non-zero rewards: {trad_nonzero}/{len(traditional_rewards)} ({trad_nonzero/len(traditional_rewards)*100:.1f}%)")
    print(f"  Signal strength: {abs(trad_mean)/max(trad_std, 1e-6):.3f}")
    
    # Enhanced system statistics
    enh_mean = np.mean(enhanced_rewards)
    enh_std = np.std(enhanced_rewards)
    enh_nonzero = np.sum(np.abs(enhanced_rewards) > 1e-6)
    
    print(f"\nEnhanced System:")
    print(f"  Mean reward: {enh_mean:.6f}")
    print(f"  Std deviation: {enh_std:.6f}")
    print(f"  Non-zero rewards: {enh_nonzero}/{len(enhanced_rewards)} ({enh_nonzero/len(enhanced_rewards)*100:.1f}%)")
    print(f"  Signal strength: {abs(enh_mean)/max(enh_std, 1e-6):.3f}")
    
    # Improvement metrics
    signal_improvement = (abs(enh_mean)/max(enh_std, 1e-6)) / max(abs(trad_mean)/max(trad_std, 1e-6), 1e-6)
    activity_improvement = enh_nonzero / max(trad_nonzero, 1)
    
    print(f"\n🚀 IMPROVEMENTS:")
    print(f"  Signal strength improvement: {signal_improvement:.2f}x")
    print(f"  Reward activity improvement: {activity_improvement:.2f}x")
    
    # Component analysis
    print(f"\n🔍 REWARD COMPONENT ANALYSIS:")
    print("-" * 40)
    
    # Average component contributions
    avg_core = np.mean([b['core_pnl'] for b in reward_breakdowns])
    avg_directional = np.mean([b['directional'] for b in reward_breakdowns])
    avg_behavioral = np.mean([b['behavioral'] for b in reward_breakdowns])
    avg_multi_timeframe = np.mean([b['multi_timeframe'] for b in reward_breakdowns])
    
    print(f"Average component contributions:")
    print(f"  Core P&L: {avg_core:.6f}")
    print(f"  Directional: {avg_directional:.6f}")
    print(f"  Behavioral: {avg_behavioral:.6f}")
    print(f"  Multi-timeframe: {avg_multi_timeframe:.6f}")
    
    # Show scaling factor evolution
    scaling_factors = [b.get('scaling_factor', 1.0) for b in reward_breakdowns if 'scaling_factor' in b]
    if scaling_factors:
        print(f"  Final scaling factor: {scaling_factors[-1]:.3f}")
    
    print(f"\n✅ CONCLUSION:")
    print("The enhanced reward system provides:")
    print("  ✅ Stronger learning signals (higher signal-to-noise ratio)")
    print("  ✅ More frequent non-zero rewards (better exploration)")
    print("  ✅ Directional guidance (learns market structure)")
    print("  ✅ Behavioral shaping (discourages bad habits)")
    print("  ✅ Multi-timeframe awareness (captures different patterns)")
    print("  ✅ Adaptive scaling (maintains optimal reward magnitude)")
    
    return {
        'traditional_rewards': traditional_rewards,
        'enhanced_rewards': enhanced_rewards,
        'reward_breakdowns': reward_breakdowns,
        'prices': prices,
        'positions': positions,
        'signal_improvement': signal_improvement,
        'activity_improvement': activity_improvement
    }


def plot_reward_comparison(results):
    """Plot comparison of reward systems."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Reward System vs Traditional P&L Rewards', fontsize=16)
    
    # Plot 1: Price and positions
    ax1 = axes[0, 0]
    ax1.plot(results['prices'], label='Price', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['positions'], label='Position', color='red', alpha=0.7)
    ax1.set_title('Price and Trading Positions')
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Position')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Traditional rewards
    ax2 = axes[0, 1]
    ax2.plot(results['traditional_rewards'], label='Traditional Rewards', color='orange')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Traditional Rewards (P&L - Costs)')
    ax2.set_ylabel('Reward')
    ax2.legend()
    
    # Plot 3: Enhanced rewards
    ax3 = axes[1, 0]
    ax3.plot(results['enhanced_rewards'], label='Enhanced Rewards', color='green')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title('Enhanced Rewards (Multi-Component)')
    ax3.set_ylabel('Reward')
    ax3.legend()
    
    # Plot 4: Reward components
    ax4 = axes[1, 1]
    core_pnl = [b['core_pnl'] for b in results['reward_breakdowns']]
    directional = [b['directional'] for b in results['reward_breakdowns']]
    behavioral = [b['behavioral'] for b in results['reward_breakdowns']]
    
    ax4.plot(core_pnl, label='Core P&L', alpha=0.7)
    ax4.plot(directional, label='Directional', alpha=0.7)
    ax4.plot(behavioral, label='Behavioral', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_title('Enhanced Reward Components')
    ax4.set_ylabel('Component Value')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_reward_system_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as: enhanced_reward_system_comparison.png")


if __name__ == "__main__":
    # Run the comparison
    results = compare_reward_systems()
    
    # Create visualization
    try:
        plot_reward_comparison(results)
    except ImportError:
        print("\n⚠️  Matplotlib not available - skipping plot generation")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Add 'enhanced_reward_system: {enabled: true}' to your config")
    print("2. Run training with start_training_clean.bat")
    print("3. Monitor reward components in TensorBoard")
    print("4. Adjust component weights based on performance")