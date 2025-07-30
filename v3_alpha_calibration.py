#!/usr/bin/env python3
"""
🎯 V3 + ALPHA CALIBRATION TEST
Find the right balance between V3 safety and alpha signal extraction
Test different impact strengths to see when alpha signals become profitable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data
from src.gym_env.dual_reward_v3 import DualTickerRewardV3

def test_alpha_vs_impact_balance():
    """Test if alpha signals can overcome different impact model strengths"""
    
    logger.info("🎯 V3 + ALPHA CALIBRATION TEST")
    logger.info("Finding balance between safety and signal extraction")
    
    # Create strong alpha signal data
    enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
        n_periods=1000, 
        seed=42, 
        alpha_strength=0.2  # Strong alpha for testing
    )
    
    logger.info(f"Alpha data: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish signals")
    
    # Test different impact model strengths
    impact_strengths = [20, 40, 60, 80, 100]  # Different bp levels
    
    logger.info(f"\n{'Impact BP':>10} {'Alpha Win':>10} {'Impact Win':>11} {'Net Reward':>11} {'Trade Worth?':>12}")
    logger.info("-" * 65)
    
    for impact_bp in impact_strengths:
        # Create V3 reward calculator with this impact strength
        reward_calc = DualTickerRewardV3(
            base_impact_bp=float(impact_bp),
            verbose=False
        )
        
        # Simulate a typical trade with strong alpha signal
        typical_trade_value = 42500.0  # $42.5K trade (250 shares * $170)
        strong_alpha_pnl = 500.0       # $500 P&L from alpha signal
        
        reward, components = reward_calc.calculate_reward(
            prev_portfolio_value=100000.0,
            curr_portfolio_value=100500.0,  # +$500 from alpha
            nvda_trade_value=typical_trade_value,
            msft_trade_value=0.0,
            nvda_position=250.0,
            msft_position=0.0,
            nvda_price=170.0,
            msft_price=510.0,
            step=1
        )
        
        # Analysis
        alpha_signal_value = components.risk_free_nav_change  # Risk-adjusted P&L
        impact_cost = components.embedded_impact
        net_reward = reward
        
        trade_worthwhile = reward > 0
        status = "✅ TRADE" if trade_worthwhile else "❌ HOLD"
        
        logger.info(f"{impact_bp:8d}bp {alpha_signal_value:9.2f} {impact_cost:10.2f} {net_reward:10.2f} {status:>12}")
        
        reward_calc.reset()
    
    logger.info("\n🎯 CALIBRATION INSIGHTS:")
    logger.info("   - Impact BP where alpha signals become profitable")
    logger.info("   - Lower impact = more trading, higher risk")
    logger.info("   - Higher impact = less trading, lower risk") 
    logger.info("   - Sweet spot: Alpha wins but not by too much")

def find_optimal_impact_strength():
    """Find the impact strength where strong alpha barely wins"""
    
    logger.info("\n🔬 FINDING OPTIMAL IMPACT STRENGTH")
    
    # Create alpha data
    enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
        n_periods=1000, seed=42, alpha_strength=0.15
    )
    
    # Binary search for optimal impact level
    low_impact = 10.0
    high_impact = 150.0
    target_reward = 10.0  # Small positive reward (alpha barely wins)
    
    for iteration in range(10):  # Max 10 iterations
        mid_impact = (low_impact + high_impact) / 2
        
        reward_calc = DualTickerRewardV3(base_impact_bp=mid_impact, verbose=False)
        
        # Test with typical alpha trade
        reward, components = reward_calc.calculate_reward(
            prev_portfolio_value=100000.0,
            curr_portfolio_value=100300.0,  # $300 alpha gain
            nvda_trade_value=42500.0,
            msft_trade_value=0.0,
            nvda_position=250.0,
            msft_position=0.0,
            nvda_price=170.0,
            msft_price=510.0,
            step=1
        )
        
        logger.info(f"  Iteration {iteration+1}: {mid_impact:.1f}bp → Reward {reward:.2f}")
        
        if abs(reward - target_reward) < 1.0:  # Close enough
            logger.info(f"🎯 OPTIMAL IMPACT FOUND: {mid_impact:.1f}bp")
            logger.info(f"   Alpha P&L: ${components.risk_free_nav_change:.2f}")
            logger.info(f"   Impact cost: ${components.embedded_impact:.2f}")
            logger.info(f"   Net reward: {reward:.2f}")
            return mid_impact
        elif reward > target_reward:
            low_impact = mid_impact  # Need higher impact
        else:
            high_impact = mid_impact  # Need lower impact
        
        reward_calc.reset()
    
    return mid_impact

if __name__ == "__main__":
    test_alpha_vs_impact_balance()
    optimal_impact = find_optimal_impact_strength()
    
    print(f"\n🎯 CALIBRATION RECOMMENDATION:")
    print(f"   Optimal V3 impact strength: {optimal_impact:.1f}bp")
    print(f"   This allows strong alpha signals to be profitable")
    print(f"   while still preventing cost-blind trading")