#!/usr/bin/env python3
"""
Action balance reward components for fixing action imbalance in RL trading.

Provides reward modifications to encourage balanced action exploration.
"""

import numpy as np
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple
import logging

class ActionBalanceRewardModifier:
    """
    Reward modifier to encourage balanced action exploration in trading.
    
    Addresses the >800 SELL vs <40 BUY problem through reward engineering.
    """
    
    def __init__(self,
                 diversity_bonus_weight: float = 0.1,
                 balance_penalty_weight: float = 2.0,
                 per_trade_fee_penalty: float = 5.0,
                 lookback_window: int = 50,
                 target_balance_ratio: float = 0.4,  # Target: 40-60% split
                 enable_diversity_bonus: bool = True,
                 enable_balance_penalty: bool = True,
                 enable_enhanced_fees: bool = True):
        """
        Initialize action balance reward modifier.
        
        Args:
            diversity_bonus_weight: Weight for action diversity bonus
            balance_penalty_weight: Weight for imbalance penalty
            per_trade_fee_penalty: Additional fee penalty per trade
            lookback_window: Window for tracking recent actions
            target_balance_ratio: Target ratio for balanced actions (0.4 = 40-60% split)
            enable_diversity_bonus: Enable entropy-based diversity bonus
            enable_balance_penalty: Enable imbalance penalty
            enable_enhanced_fees: Enable higher per-trade fees
        """
        self.diversity_bonus_weight = diversity_bonus_weight
        self.balance_penalty_weight = balance_penalty_weight
        self.per_trade_fee_penalty = per_trade_fee_penalty
        self.lookback_window = lookback_window
        self.target_balance_ratio = target_balance_ratio
        self.enable_diversity_bonus = enable_diversity_bonus
        self.enable_balance_penalty = enable_balance_penalty
        self.enable_enhanced_fees = enable_enhanced_fees
        
        # Action tracking
        self.recent_actions = deque(maxlen=lookback_window)
        self.episode_actions = []
        self.action_names = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"🎯 ActionBalanceRewardModifier initialized:")
        self.logger.info(f"   Diversity bonus: {diversity_bonus_weight}")
        self.logger.info(f"   Balance penalty: {balance_penalty_weight}")
        self.logger.info(f"   Enhanced fees: {per_trade_fee_penalty}")
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_actions = []
    
    def calculate_reward_modification(self, action: int, base_reward: float) -> Tuple[float, Dict]:
        """
        Calculate reward modifications for action balance.
        
        Args:
            action: Action taken (0=SELL, 1=HOLD, 2=BUY)
            base_reward: Original reward from environment
            
        Returns:
            Tuple of (modified_reward, modification_info)
        """
        # Track actions
        self.recent_actions.append(action)
        self.episode_actions.append(action)
        
        modifications = {
            'diversity_bonus': 0.0,
            'balance_penalty': 0.0,
            'enhanced_fee_penalty': 0.0,
            'total_modification': 0.0
        }
        
        # 1. Action diversity bonus (entropy-based)
        if self.enable_diversity_bonus and len(self.recent_actions) >= 10:
            diversity_bonus = self._calculate_diversity_bonus()
            modifications['diversity_bonus'] = diversity_bonus
        
        # 2. Balance penalty for extreme imbalances
        if self.enable_balance_penalty and len(self.recent_actions) >= 20:
            balance_penalty = self._calculate_balance_penalty()
            modifications['balance_penalty'] = balance_penalty
        
        # 3. Enhanced per-trade fee penalty
        if self.enable_enhanced_fees and action != 1:  # Not HOLD
            fee_penalty = -self.per_trade_fee_penalty
            modifications['enhanced_fee_penalty'] = fee_penalty
        
        # Calculate total modification
        total_modification = sum(modifications.values())
        modifications['total_modification'] = total_modification
        
        modified_reward = base_reward + total_modification
        
        return modified_reward, modifications
    
    def _calculate_diversity_bonus(self) -> float:
        """Calculate entropy-based diversity bonus."""
        if len(self.recent_actions) < 10:
            return 0.0
        
        # Count actions in recent window
        action_counts = np.bincount(list(self.recent_actions), minlength=3)
        action_probs = action_counts / len(self.recent_actions)
        
        # Calculate entropy (higher = more diverse)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        max_entropy = np.log(3)  # Maximum entropy for 3 actions
        
        # Normalize and scale
        normalized_entropy = entropy / max_entropy
        diversity_bonus = normalized_entropy * self.diversity_bonus_weight
        
        return diversity_bonus
    
    def _calculate_balance_penalty(self) -> float:
        """Calculate penalty for action imbalance."""
        if len(self.recent_actions) < 20:
            return 0.0
        
        # Count trading actions only (SELL and BUY)
        action_counts = np.bincount(list(self.recent_actions), minlength=3)
        sell_count = action_counts[0]
        buy_count = action_counts[2]
        total_trades = sell_count + buy_count
        
        if total_trades == 0:
            return 0.0
        
        # Calculate imbalance
        sell_ratio = sell_count / total_trades
        buy_ratio = buy_count / total_trades
        
        # Penalty for deviation from target balance
        target_min = self.target_balance_ratio
        target_max = 1.0 - self.target_balance_ratio
        
        penalty = 0.0
        if sell_ratio < target_min or sell_ratio > target_max:
            # Distance from acceptable range
            if sell_ratio < target_min:
                penalty = (target_min - sell_ratio) * self.balance_penalty_weight
            else:
                penalty = (sell_ratio - target_max) * self.balance_penalty_weight
        
        return -penalty  # Negative because it's a penalty
    
    def get_episode_summary(self) -> Dict:
        """Get summary of episode action balance."""
        if not self.episode_actions:
            return {'status': 'NO_ACTIONS'}
        
        action_counts = np.bincount(self.episode_actions, minlength=3)
        total_actions = len(self.episode_actions)
        total_trades = action_counts[0] + action_counts[2]  # SELL + BUY
        
        summary = {
            'total_actions': total_actions,
            'action_counts': {
                'SELL': int(action_counts[0]),
                'HOLD': int(action_counts[1]),
                'BUY': int(action_counts[2])
            },
            'action_ratios': {
                'SELL': action_counts[0] / total_actions,
                'HOLD': action_counts[1] / total_actions,
                'BUY': action_counts[2] / total_actions
            },
            'total_trades': int(total_trades),
            'trade_balance': {
                'sell_ratio': action_counts[0] / max(total_trades, 1),
                'buy_ratio': action_counts[2] / max(total_trades, 1)
            }
        }
        
        # Calculate imbalance score
        if total_trades > 0:
            sell_ratio = action_counts[0] / total_trades
            buy_ratio = action_counts[2] / total_trades
            imbalance_score = abs(sell_ratio - buy_ratio)
            summary['imbalance_score'] = imbalance_score
            
            if imbalance_score >= 0.8:
                summary['balance_status'] = 'SEVERE_IMBALANCE'
            elif imbalance_score >= 0.6:
                summary['balance_status'] = 'MODERATE_IMBALANCE'
            else:
                summary['balance_status'] = 'BALANCED'
        else:
            summary['imbalance_score'] = 0.0
            summary['balance_status'] = 'NO_TRADES'
        
        return summary

class ActionBalanceIntegration:
    """
    Integration helper for adding action balance rewards to existing environments.
    """
    
    @staticmethod
    def create_balanced_reward_modifier(severity: str = 'moderate') -> ActionBalanceRewardModifier:
        """
        Create pre-configured reward modifier based on imbalance severity.
        
        Args:
            severity: 'mild', 'moderate', or 'severe'
        """
        configs = {
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
        
        config = configs.get(severity, configs['moderate'])
        return ActionBalanceRewardModifier(**config)
    
    @staticmethod
    def integrate_with_environment(env, severity: str = 'moderate'):
        """
        Example integration with trading environment.
        
        This shows how to modify the step() method to include balance rewards.
        """
        modifier = ActionBalanceIntegration.create_balanced_reward_modifier(severity)
        
        # Store original step method
        original_step = env.step
        
        def balanced_step(action):
            # Call original step
            obs, reward, done, info = original_step(action)
            
            # Apply balance modifications
            modified_reward, modifications = modifier.calculate_reward_modification(action, reward)
            
            # Add modification info to info dict
            info['action_balance'] = modifications
            
            return obs, modified_reward, done, info
        
        # Replace step method
        env.step = balanced_step
        env.action_balance_modifier = modifier
        
        return env

# Example usage and testing
def example_usage():
    """Example of how to use ActionBalanceRewardModifier."""
    print("🎯 ACTION BALANCE REWARD MODIFIER EXAMPLE")
    print("=" * 50)
    
    # Create modifier for severe imbalance
    modifier = ActionBalanceRewardModifier(
        diversity_bonus_weight=0.2,
        balance_penalty_weight=5.0,
        per_trade_fee_penalty=10.0
    )
    
    # Simulate episode with severe SELL bias
    print("\n📊 Simulating episode with SELL bias...")
    modifier.reset_episode()
    
    total_reward = 0.0
    total_modifications = 0.0
    
    # Simulate 100 steps with heavy SELL bias
    actions = [0] * 80 + [1] * 15 + [2] * 5  # 80 SELL, 15 HOLD, 5 BUY
    np.random.shuffle(actions)
    
    for step, action in enumerate(actions):
        base_reward = np.random.normal(0.1, 0.5)  # Random base reward
        
        modified_reward, modifications = modifier.calculate_reward_modification(action, base_reward)
        
        total_reward += modified_reward
        total_modifications += modifications['total_modification']
        
        if step % 20 == 0:
            print(f"Step {step}: Action={modifier.action_names[action]}, "
                  f"Base={base_reward:.3f}, Modified={modified_reward:.3f}, "
                  f"Modification={modifications['total_modification']:+.3f}")
    
    # Episode summary
    summary = modifier.get_episode_summary()
    print(f"\n📈 EPISODE SUMMARY:")
    print(f"Balance Status: {summary['balance_status']}")
    print(f"Action Counts: {summary['action_counts']}")
    print(f"Imbalance Score: {summary['imbalance_score']:.3f}")
    print(f"Total Reward Modification: {total_modifications:+.3f}")
    
    print(f"\n💡 EFFECT:")
    print(f"The reward modifications penalize the severe SELL bias,")
    print(f"encouraging the agent to explore more balanced actions.")

if __name__ == "__main__":
    example_usage()