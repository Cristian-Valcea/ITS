#!/usr/bin/env python3
"""
Fix excessive trading by increasing action-change penalty and trade cooldown
"""

import sys
sys.path.append('src')
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_current_penalties():
    """Test current penalty levels"""
    
    print('🧪 TESTING CURRENT PENALTY LEVELS')
    print('=' * 60)
    
    # Simulate action changes
    actions = [0, 1, 0, 1, 0, 1, 0]  # Ping-ponging between BUY/SELL
    
    # Current penalty factor
    current_factor = 0.001
    reward_scaling = 1.0
    
    print(f'📊 Current action_change_penalty_factor: {current_factor}')
    print(f'📊 Reward scaling: {reward_scaling}')
    print()
    
    print('Action Sequence Analysis:')
    print('Step | Prev→Curr | Change² | Penalty   | Cumulative')
    print('-' * 50)
    
    total_penalty = 0.0
    prev_action = 0
    
    for step, action in enumerate(actions[1:], 1):
        change_squared = (action - prev_action) ** 2
        penalty = current_factor * change_squared * reward_scaling
        total_penalty += penalty
        
        print(f'{step:4d} | {prev_action:4d}→{action:4d} | {change_squared:7.1f} | ${penalty:8.6f} | ${total_penalty:8.6f}')
        prev_action = action
    
    print(f'\n💰 Total penalty for 6 ping-pong actions: ${total_penalty:.6f}')
    print('❌ This is WAY too small to deter ping-ponging!')
    
    return total_penalty

def calculate_effective_penalties():
    """Calculate what penalty levels would be effective"""
    
    print('\n🎯 CALCULATING EFFECTIVE PENALTY LEVELS')
    print('=' * 60)
    
    # Typical reward magnitudes in your system
    typical_pnl_reward = 10.0  # $10 P&L reward per step
    typical_turnover_penalty = 100.0  # $100 turnover penalty
    
    print(f'📊 Typical P&L reward: ${typical_pnl_reward:.2f}')
    print(f'📊 Typical turnover penalty: ${typical_turnover_penalty:.2f}')
    print()
    
    # We want action change penalty to be significant enough to deter ping-ponging
    # Target: Each action change should cost 10-20% of typical reward
    target_penalty_per_change = typical_pnl_reward * 0.15  # 15% of typical reward
    
    print(f'🎯 Target penalty per action change: ${target_penalty_per_change:.2f}')
    
    # Calculate required factor
    # penalty = factor * (action_change)² * reward_scaling
    # For BUY→SELL or SELL→BUY: action_change = ±1, so (action_change)² = 1
    required_factor = target_penalty_per_change / (1.0 * 1.0)  # change²=1, scaling=1
    
    print(f'🔧 Required action_change_penalty_factor: {required_factor:.3f}')
    print(f'   (This is {required_factor/0.001:.0f}x larger than current {0.001})')
    
    return required_factor

def test_proposed_penalties():
    """Test the proposed penalty levels"""
    
    print('\n🧪 TESTING PROPOSED PENALTY LEVELS')
    print('=' * 60)
    
    # Proposed values
    proposed_factor = 1.5  # Much higher than current 0.001
    proposed_cooldown = 5   # Higher than current 2
    
    print(f'📊 Proposed action_change_penalty_factor: {proposed_factor}')
    print(f'📊 Proposed trade_cooldown_steps: {proposed_cooldown}')
    print()
    
    # Test ping-ponging scenario
    actions = [0, 1, 0, 1, 0, 1, 0]  # Agent tries to ping-pong
    
    print('Ping-Pong Scenario with Proposed Penalties:')
    print('Step | Action | Change² | Penalty  | Cooldown | Can Trade?')
    print('-' * 60)
    
    total_penalty = 0.0
    prev_action = 0
    steps_since_trade = proposed_cooldown  # Start ready to trade
    
    for step, action in enumerate(actions[1:], 1):
        change_squared = (action - prev_action) ** 2
        penalty = proposed_factor * change_squared * 1.0  # reward_scaling = 1
        
        # Check if trade is allowed (cooldown)
        can_trade = steps_since_trade >= proposed_cooldown
        trade_status = "✅ YES" if can_trade else f"❌ NO ({steps_since_trade}/{proposed_cooldown})"
        
        if can_trade and action != prev_action:
            # Trade executed, reset cooldown
            steps_since_trade = 0
            total_penalty += penalty
        else:
            # No trade (either cooldown or no change)
            penalty = 0.0
        
        print(f'{step:4d} | {action:6d} | {change_squared:7.1f} | ${penalty:7.2f} | {steps_since_trade:8d} | {trade_status}')
        
        steps_since_trade += 1
        if can_trade and action != prev_action:
            prev_action = action
    
    print(f'\n💰 Total penalty with proposed settings: ${total_penalty:.2f}')
    print(f'🎯 Trades executed: {sum(1 for p in [penalty for penalty in [proposed_factor * ((actions[i] - actions[i-1]) ** 2) for i in range(1, len(actions))] if p > 0])}')
    
    return proposed_factor, proposed_cooldown

def generate_config_update():
    """Generate the configuration update"""
    
    print('\n🔧 GENERATING CONFIGURATION UPDATE')
    print('=' * 60)
    
    # Recommended values based on analysis
    recommended_action_penalty = 1.5  # 1500x increase from 0.001
    recommended_cooldown = 5          # 2.5x increase from 2
    
    config_update = f"""
# 🔧 EMERGENCY FIX: Stop Excessive Trading (Ping-Ponging)
# 
# Problem: Agent making ~40 trades/hour by oscillating BUY/SELL
# Solution: Increase action-change penalty and trade cooldown

UPDATED_CONFIG = {{
    # Action Change Penalty (was 0.001, now {recommended_action_penalty})
    'action_change_penalty_factor': {recommended_action_penalty},
    
    # Trade Cooldown (was 2, now {recommended_cooldown})  
    'trade_cooldown_steps': {recommended_cooldown},
    
    # Keep other settings
    'turnover_penalty_factor': 0.15,  # Your normalized penalty
    'transaction_cost_pct': 0.0005,   # 0.05% transaction cost
    'reward_scaling': 1.0,
}}

# Expected Impact:
# - Each action change now costs ${recommended_action_penalty:.1f} (was $0.001)
# - Must wait {recommended_cooldown} steps between trades (was 2)
# - Should reduce trading frequency from ~40/hour to ~8/hour
# - Eliminates ping-ponging incentive
"""
    
    print(config_update)
    
    return recommended_action_penalty, recommended_cooldown

if __name__ == '__main__':
    print('🎯 FIXING EXCESSIVE TRADING (PING-PONGING)')
    print('=' * 80)
    
    # Step 1: Analyze current penalties
    current_penalty = test_current_penalties()
    
    # Step 2: Calculate effective penalties
    required_factor = calculate_effective_penalties()
    
    # Step 3: Test proposed penalties
    proposed_factor, proposed_cooldown = test_proposed_penalties()
    
    # Step 4: Generate config update
    final_factor, final_cooldown = generate_config_update()
    
    print('\n🎯 SUMMARY:')
    print('=' * 50)
    print(f'❌ Current action penalty: ${current_penalty:.6f} (ineffective)')
    print(f'✅ Proposed action penalty: ${proposed_factor:.1f} per change')
    print(f'✅ Proposed cooldown: {proposed_cooldown} steps between trades')
    print()
    print('🚀 This should reduce trading from ~40/hour to ~8/hour')
    print('   and eliminate ping-ponging behavior!')