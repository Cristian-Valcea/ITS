#!/usr/bin/env python3
"""
🔍 PHASE A: V3 OFFLINE REWARD AUDIT
Run V3 reward formula over random-trading replay to verify expected penalties

Expected behavior:
- Random trading should consistently lose money due to embedded impact
- Do-nothing strategy should have small negative reward (risk-free drag)
- High turnover should be heavily penalized
- No algorithmic edge = negative cumulative reward
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gym_env.dual_reward_v3 import DualTickerRewardV3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_random_trading_replay(n_steps=2000, seed=0):
    """Generate random trading replay for audit"""
    np.random.seed(seed)
    
    # Portfolio starts at $100K
    initial_portfolio = 100000.0
    portfolio_values = [initial_portfolio]
    
    # Generate random price movements (realistic market dynamics)
    base_nvda_price = 170.0
    base_msft_price = 510.0
    
    nvda_returns = np.random.normal(0.0001, 0.015, n_steps)  # Realistic intraday returns
    msft_returns = np.random.normal(0.0001, 0.012, n_steps)  # Slightly less volatile
    
    nvda_prices = base_nvda_price * np.exp(np.cumsum(nvda_returns))
    msft_prices = base_msft_price * np.exp(np.cumsum(msft_returns))
    
    # Generate random trading actions
    # Mix of: do-nothing (40%), small trades (40%), large trades (20%)
    trade_types = np.random.choice(['nothing', 'small', 'large'], 
                                   size=n_steps, 
                                   p=[0.4, 0.4, 0.2])
    
    trades = []
    positions = {'nvda': 0.0, 'msft': 0.0}
    
    for step in range(n_steps):
        if trade_types[step] == 'nothing':
            nvda_trade = 0.0
            msft_trade = 0.0
        elif trade_types[step] == 'small':
            # Small trades (1-5% of portfolio)
            nvda_trade = np.random.uniform(-0.05, 0.05) * portfolio_values[-1]
            msft_trade = np.random.uniform(-0.05, 0.05) * portfolio_values[-1]
        else:  # large
            # Large trades (10-30% of portfolio)
            nvda_trade = np.random.uniform(-0.3, 0.3) * portfolio_values[-1]
            msft_trade = np.random.uniform(-0.3, 0.3) * portfolio_values[-1]
        
        # Update positions
        positions['nvda'] += nvda_trade
        positions['msft'] += msft_trade
        
        # Calculate portfolio value more realistically
        prev_portfolio = portfolio_values[-1]
        
        # For do-nothing steps, portfolio stays exactly the same (true do-nothing)
        if trade_types[step] == 'nothing':
            market_pnl = 0.0
        else:
            # For trading steps, add some market P&L (but much smaller)
            market_pnl = np.random.normal(0, prev_portfolio * 0.0005)  # 0.05% std dev per step
        
        # Apply trading costs (simplified)
        total_trade_value = abs(nvda_trade) + abs(msft_trade)
        trading_costs = total_trade_value * 0.0001  # 1bp trading cost
        
        new_portfolio = prev_portfolio + market_pnl - trading_costs
        
        portfolio_values.append(max(new_portfolio, 1000))  # Prevent negative
        
        trades.append({
            'step': step,
            'nvda_trade': nvda_trade,
            'msft_trade': msft_trade,
            'nvda_position': positions['nvda'],
            'msft_position': positions['msft'],
            'nvda_price': nvda_prices[step],
            'msft_price': msft_prices[step],
            'portfolio_before': portfolio_values[-2],
            'portfolio_after': portfolio_values[-1],
            'trade_type': trade_types[step]
        })
    
    return trades, portfolio_values

def audit_v3_rewards(trades, portfolio_values):
    """Audit V3 reward system with random trading replay"""
    
    logger.info("🔍 Starting V3 reward audit with random trading replay...")
    
    # Initialize V3 reward calculator
    reward_calculator = DualTickerRewardV3(
        risk_free_rate_annual=0.05,
        base_impact_bp=20.0,
        impact_exponent=0.5,
        verbose=False
    )
    
    # Track results
    rewards = []
    components_list = []
    cumulative_reward = 0.0
    
    # Statistics tracking
    do_nothing_rewards = []
    small_trade_rewards = []
    large_trade_rewards = []
    
    for i, trade in enumerate(trades):
        reward, components = reward_calculator.calculate_reward(
            prev_portfolio_value=trade['portfolio_before'],
            curr_portfolio_value=trade['portfolio_after'],
            nvda_trade_value=trade['nvda_trade'],
            msft_trade_value=trade['msft_trade'],
            nvda_position=trade['nvda_position'],
            msft_position=trade['msft_position'],
            nvda_price=trade['nvda_price'],
            msft_price=trade['msft_price'],
            step=i
        )
        
        rewards.append(reward)
        components_list.append(components)
        cumulative_reward += reward
        
        # Categorize by trade type
        if trade['trade_type'] == 'nothing':
            do_nothing_rewards.append(reward)
        elif trade['trade_type'] == 'small':
            small_trade_rewards.append(reward)
        else:
            large_trade_rewards.append(reward)
        
        # Log progress
        if i % 500 == 0:
            logger.info(f"Step {i:4d}: Reward {reward:8.2f}, Cumulative {cumulative_reward:10.2f}, Portfolio ${trade['portfolio_after']:8,.0f}")
    
    return {
        'rewards': rewards,
        'components': components_list,
        'cumulative_reward': cumulative_reward,
        'do_nothing_rewards': do_nothing_rewards,
        'small_trade_rewards': small_trade_rewards,
        'large_trade_rewards': large_trade_rewards,
        'final_portfolio': portfolio_values[-1],
        'initial_portfolio': portfolio_values[0]
    }

def analyze_audit_results(results):
    """Analyze V3 audit results and validate expected behavior"""
    
    logger.info("📊 V3 REWARD AUDIT ANALYSIS")
    logger.info("=" * 50)
    
    # Basic statistics
    rewards = np.array(results['rewards'])
    total_steps = len(rewards)
    
    logger.info(f"📈 OVERALL PERFORMANCE:")
    logger.info(f"   Total steps: {total_steps:,}")
    logger.info(f"   Cumulative reward: {results['cumulative_reward']:,.2f}")
    logger.info(f"   Mean reward per step: {np.mean(rewards):.4f}")
    logger.info(f"   Final portfolio: ${results['final_portfolio']:,.2f}")
    logger.info(f"   Portfolio return: {(results['final_portfolio']/results['initial_portfolio']-1)*100:.2f}%")
    
    # By trade type analysis
    logger.info(f"\n🎯 REWARD BY TRADE TYPE:")
    
    if results['do_nothing_rewards']:
        do_nothing_mean = np.mean(results['do_nothing_rewards'])
        logger.info(f"   Do-nothing (N={len(results['do_nothing_rewards'])}): {do_nothing_mean:.4f} avg")
    
    if results['small_trade_rewards']:
        small_trade_mean = np.mean(results['small_trade_rewards'])
        logger.info(f"   Small trades (N={len(results['small_trade_rewards'])}): {small_trade_mean:.4f} avg")
    
    if results['large_trade_rewards']:
        large_trade_mean = np.mean(results['large_trade_rewards'])
        logger.info(f"   Large trades (N={len(results['large_trade_rewards'])}): {large_trade_mean:.4f} avg")
    
    # Component analysis
    logger.info(f"\n💰 REWARD COMPONENT BREAKDOWN:")
    total_nav = sum(c.risk_free_nav_change for c in results['components'])
    total_impact = sum(c.embedded_impact for c in results['components'])
    total_downside = sum(c.downside_semi_variance for c in results['components'])
    total_kelly = sum(c.kelly_bonus for c in results['components'])
    
    logger.info(f"   Risk-free NAV change: {total_nav:8.2f}")
    logger.info(f"   Embedded impact:      {-total_impact:8.2f} (penalty)")
    logger.info(f"   Downside penalty:     {-total_downside:8.2f} (penalty)")
    logger.info(f"   Kelly bonus:          {total_kelly:8.2f}")
    logger.info(f"   Total:                {total_nav - total_impact - total_downside + total_kelly:8.2f}")
    
    # V3 Design Validation
    logger.info(f"\n✅ V3 DESIGN VALIDATION:")
    
    # Test 1: Random trading should lose money
    random_trading_negative = results['cumulative_reward'] < 0
    logger.info(f"   Random trading loses money: {'✅ PASS' if random_trading_negative else '❌ FAIL'} ({results['cumulative_reward']:.2f})")
    
    # Test 2: Do-nothing should be small negative (risk-free drag)
    if results['do_nothing_rewards']:
        expected_rf_drag = -0.051  # Approx risk-free drag per step
        do_nothing_close_to_rf = abs(do_nothing_mean - expected_rf_drag) < 0.01
        logger.info(f"   Do-nothing ≈ risk-free drag: {'✅ PASS' if do_nothing_close_to_rf else '❌ FAIL'} ({do_nothing_mean:.4f} vs {expected_rf_drag:.4f})")
    
    # Test 3: Large trades more negative than small trades
    if results['small_trade_rewards'] and results['large_trade_rewards']:
        large_more_negative = large_trade_mean < small_trade_mean
        logger.info(f"   Large trades < small trades: {'✅ PASS' if large_more_negative else '❌ FAIL'} ({large_trade_mean:.4f} vs {small_trade_mean:.4f})")
    
    # Test 4: Impact dominates the penalty structure
    impact_significant = total_impact > abs(total_nav) * 0.1  # Impact should be at least 10% of NAV movements
    logger.info(f"   Embedded impact significant: {'✅ PASS' if impact_significant else '❌ FAIL'} ({total_impact:.2f})")
    
    # Overall audit assessment
    tests_passed = sum([
        random_trading_negative,
        do_nothing_close_to_rf if results['do_nothing_rewards'] else True,
        large_more_negative if (results['small_trade_rewards'] and results['large_trade_rewards']) else True,
        impact_significant
    ])
    
    total_tests = 4
    audit_pass = tests_passed == total_tests
    
    logger.info(f"\n🎯 PHASE A AUDIT RESULT:")
    logger.info(f"   Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"   V3 reward structure: {'✅ VALIDATED' if audit_pass else '❌ NEEDS ADJUSTMENT'}")
    
    if audit_pass:
        logger.info(f"   ✅ V3 successfully prevents cost-blind trading")
        logger.info(f"   ✅ Ready for Phase B integration")
    else:
        logger.info(f"   ❌ V3 parameters need adjustment before integration")
    
    return audit_pass

def main():
    """Run Phase A: V3 offline reward audit"""
    
    logger.info("🔍 PHASE A: V3 OFFLINE REWARD AUDIT")
    logger.info("🎯 Testing V3 reward formula with random trading patterns")
    logger.info("📋 Expected: Random trading loses money, do-nothing ≈ risk-free drag")
    
    start_time = datetime.now()
    
    # Generate random trading replay
    logger.info("📊 Generating random trading replay (2000 steps)...")
    trades, portfolio_values = generate_random_trading_replay(n_steps=2000, seed=0)
    
    # Audit V3 rewards
    results = audit_v3_rewards(trades, portfolio_values)
    
    # Analyze results
    audit_pass = analyze_audit_results(results)
    
    # Final assessment
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n⏱️ Phase A completed in {elapsed_time}")
    
    if audit_pass:
        logger.info("🎉 ✅ PHASE A COMPLETE: V3 reward structure validated")
        logger.info("📋 Next: Phase B - Integrate V3 into intraday_trading_env")
        return True
    else:
        logger.info("⚠️ ❌ PHASE A INCOMPLETE: V3 needs parameter adjustment")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Phase A audit passed - V3 ready for integration")
    else:
        print("❌ Phase A audit failed - V3 needs adjustment")