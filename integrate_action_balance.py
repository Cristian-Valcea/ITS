#!/usr/bin/env python3
"""
Quick integration script to add action balance rewards to IntradayTradingEnv.

Fixes the >800 SELL vs <40 BUY problem by modifying the reward function.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from action_balance_rewards import ActionBalanceRewardModifier

def integrate_action_balance_rewards(env_class, severity: str = 'moderate'):
    """
    Modify IntradayTradingEnv to include action balance rewards.
    
    Args:
        env_class: IntradayTradingEnv class
        severity: 'mild', 'moderate', or 'severe'
    
    Returns:
        Modified environment class
    """
    
    class BalancedIntradayTradingEnv(env_class):
        """IntradayTradingEnv with action balance rewards."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize action balance modifier
            severity_configs = {
                'mild': {
                    'diversity_bonus_weight': 0.05,
                    'balance_penalty_weight': 1.0,
                    'per_trade_fee_penalty': 2.0
                },
                'moderate': {
                    'diversity_bonus_weight': 0.1,
                    'balance_penalty_weight': 2.0,
                    'per_trade_fee_penalty': 5.0
                },
                'severe': {
                    'diversity_bonus_weight': 0.2,
                    'balance_penalty_weight': 5.0,
                    'per_trade_fee_penalty': 10.0
                }
            }
            
            config = severity_configs.get(severity, severity_configs['moderate'])
            self.action_balance_modifier = ActionBalanceRewardModifier(**config)
            
            self.logger.info(f"🎯 Action balance rewards enabled (severity: {severity})")
        
        def reset(self, **kwargs):
            """Reset environment and action balance tracker."""
            obs = super().reset(**kwargs)
            self.action_balance_modifier.reset_episode()
            return obs
        
        def step(self, action):
            """Step with action balance reward modifications."""
            # Call original step
            obs, reward, done, info = super().step(action)
            
            # Apply action balance modifications
            modified_reward, modifications = self.action_balance_modifier.calculate_reward_modification(
                action, reward
            )
            
            # Add modification info to episode summary
            info['action_balance_modifications'] = modifications
            
            # Log significant modifications
            if abs(modifications['total_modification']) > 1.0:
                self.logger.log(self.TRADE_LOG_LEVEL, 
                    f"🎯 ACTION BALANCE: {modifications['total_modification']:+.3f} "
                    f"(Diversity: {modifications['diversity_bonus']:+.3f}, "
                    f"Balance: {modifications['balance_penalty']:+.3f}, "
                    f"Fee: {modifications['enhanced_fee_penalty']:+.3f})")
            
            return obs, modified_reward, done, info
        
        def _get_episode_summary(self):
            """Enhanced episode summary with action balance info."""
            summary = super()._get_episode_summary()
            
            # Add action balance summary
            balance_summary = self.action_balance_modifier.get_episode_summary()
            summary.update({
                'action_balance_status': balance_summary.get('balance_status', 'UNKNOWN'),
                'action_imbalance_score': balance_summary.get('imbalance_score', 0.0),
                'action_counts_detailed': balance_summary.get('action_counts', {}),
                'trade_balance_ratios': balance_summary.get('trade_balance', {})
            })
            
            return summary
    
    return BalancedIntradayTradingEnv

def create_integration_patch():
    """
    Create a patch that can be applied to existing IntradayTradingEnv.
    
    This shows the exact code changes needed.
    """
    patch_code = '''
# Add to IntradayTradingEnv.__init__():
from action_balance_rewards import ActionBalanceRewardModifier

# In __init__ method, after self.TRADE_LOG_LEVEL = logging.DEBUG:
self.action_balance_modifier = ActionBalanceRewardModifier(
    diversity_bonus_weight=0.1,    # Adjust based on severity
    balance_penalty_weight=2.0,    # Increase for severe imbalance
    per_trade_fee_penalty=5.0      # Higher fees to discourage overtrading
)

# Modify reset() method:
def reset(self, **kwargs):
    obs = super().reset(**kwargs)  # If inheriting, otherwise existing reset code
    self.action_balance_modifier.reset_episode()
    return obs

# Modify step() method - add after calculating base reward:
def step(self, action):
    # ... existing step code ...
    # After calculating base reward but before returning:
    
    modified_reward, modifications = self.action_balance_modifier.calculate_reward_modification(
        action, reward
    )
    
    # Optional: Log significant modifications
    if abs(modifications['total_modification']) > 1.0:
        self.logger.log(self.TRADE_LOG_LEVEL, 
            f"🎯 ACTION BALANCE: {modifications['total_modification']:+.3f}")
    
    # Use modified_reward instead of reward in return
    return obs, modified_reward, done, info

# Modify _get_episode_summary() method - add at the end:
def _get_episode_summary(self):
    summary = # ... existing summary code ...
    
    # Add action balance info
    balance_summary = self.action_balance_modifier.get_episode_summary()
    summary.update({
        'action_balance_status': balance_summary.get('balance_status', 'UNKNOWN'),
        'action_imbalance_score': balance_summary.get('imbalance_score', 0.0)
    })
    
    return summary
'''
    
    return patch_code

def main():
    """Show integration examples and patches."""
    print("🎯 ACTION BALANCE INTEGRATION FOR INTRADAYJULES")
    print("=" * 60)
    
    print("""
🔍 PROBLEM IDENTIFIED:
   >800 SELL vs <40 BUY actions = 95% action imbalance!
   
   Causes:
   • Exploration skew (agent stuck in local optimum)
   • High trading fees (discourages BUY actions)
   • Reward function bias (favors short positions)

✅ SOLUTION IMPLEMENTED:
   Action balance reward modifications that:
   • Add diversity bonus (entropy-based)
   • Penalize extreme imbalances
   • Increase per-trade fees to discourage overtrading
""")
    
    print(f"\n🔧 INTEGRATION OPTIONS:")
    print(f"1. Use the wrapper class (recommended for testing)")
    print(f"2. Apply the patch directly to IntradayTradingEnv")
    print(f"3. Use the ActionBalanceRewardModifier manually")
    
    print(f"\n📊 SEVERITY LEVELS:")
    print(f"• MILD: Small adjustments for minor imbalances")
    print(f"• MODERATE: Standard fix for typical problems")
    print(f"• SEVERE: Strong corrections for extreme cases (like >800 vs <40)")
    
    print(f"\n💡 QUICK START:")
    print(f"""
# Option 1: Wrapper (easiest)
from integrate_action_balance import integrate_action_balance_rewards
from src.gym_env.intraday_trading_env import IntradayTradingEnv

BalancedEnv = integrate_action_balance_rewards(IntradayTradingEnv, severity='severe')
env = BalancedEnv(processed_feature_data, price_data, ...)

# Option 2: Direct integration
from action_balance_rewards import ActionBalanceRewardModifier
# Add to your existing IntradayTradingEnv - see patch below
""")
    
    print(f"\n🔧 DIRECT INTEGRATION PATCH:")
    print("=" * 40)
    patch = create_integration_patch()
    print(patch)
    
    print(f"\n📈 EXPECTED RESULTS:")
    print(f"• Action distribution: 40-60% SELL/BUY split (instead of 95/5)")
    print(f"• Better exploration of both long and short strategies")
    print(f"• Reduced overtrading through enhanced fee penalties")
    print(f"• More stable training with balanced action space")
    
    print(f"\n🎯 MONITORING:")
    print(f"• Check logs/episode_summaries.csv for action_balance_status")
    print(f"• Run action_imbalance_diagnostic.py to verify improvements")
    print(f"• Monitor action_imbalance_score < 0.6 for healthy balance")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())