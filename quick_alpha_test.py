#!/usr/bin/env python3
"""
🎯 QUICK ALPHA TEST - Calibrated 68bp Impact
Test if calibrated V3 allows profitable alpha trading
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

def test_calibrated_impact():
    """Test if 68bp impact allows alpha signals to be profitable"""
    
    logger.info("🎯 QUICK ALPHA TEST - Calibrated 68bp Impact")
    
    # Create V3 reward calculator with calibrated parameters
    reward_calc = DualTickerRewardV3(
        base_impact_bp=68.0,    # Calibrated value
        verbose=True
    )
    
    logger.info("\n🧪 Testing different trade scenarios:")
    logger.info(f"{'Scenario':>20} {'Alpha P&L':>10} {'Impact':>8} {'Net':>8} {'Trade?':>8}")
    logger.info("-" * 60)
    
    test_scenarios = [
        ("No Trade", 0, 0),
        ("Small Alpha", 100, 10625),     # $100 alpha, $10.6K trade
        ("Medium Alpha", 300, 31875),    # $300 alpha, $31.9K trade  
        ("Large Alpha", 500, 42500),     # $500 alpha, $42.5K trade
    ]
    
    for scenario_name, alpha_pnl, trade_value in test_scenarios:
        reward, components = reward_calc.calculate_reward(
            prev_portfolio_value=100000.0,
            curr_portfolio_value=100000.0 + alpha_pnl,
            nvda_trade_value=float(trade_value),
            msft_trade_value=0.0,
            nvda_position=trade_value / 170.0 if trade_value > 0 else 0.0,
            msft_position=0.0,
            nvda_price=170.0,
            msft_price=510.0,
            step=1
        )
        
        alpha_signal = components.risk_free_nav_change
        impact_cost = components.embedded_impact
        net_reward = reward
        trade_worthwhile = reward > 0
        
        status = "✅" if trade_worthwhile else "❌"
        
        logger.info(f"{scenario_name:>20} {alpha_signal:>9.0f} {impact_cost:>7.0f} {net_reward:>7.0f} {status:>8}")
        
        reward_calc.reset()
    
    logger.info("\n🎯 CALIBRATION VALIDATION:")
    logger.info("   68bp impact should allow medium-large alpha to be profitable")
    logger.info("   while preventing cost-blind small trades")
    
    return True

if __name__ == "__main__":
    test_calibrated_impact()