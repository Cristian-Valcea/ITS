# src/gym_env/portfolio_action_space.py
"""
Portfolio Action Space for Dual-Ticker Trading

Handles 9-action portfolio combinations with validation and utilities.
Provides clean interface for action encoding/decoding and descriptions.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from enum import IntEnum


class SingleAssetAction(IntEnum):
    """Individual asset actions"""
    SELL = -1
    HOLD = 0
    BUY = 1


class PortfolioActionSpace:
    """
    Handles 9-action portfolio combinations with validation
    
    Action Matrix (3x3):
    ```
    NVDA →    SELL  HOLD  BUY
    MSFT ↓
    SELL      0     1     2
    HOLD      3     4     5
    BUY       6     7     8
    ```
    
    Action IDs 0-8 map to (NVDA_action, MSFT_action) combinations
    """
    
    def __init__(self):
        self.action_matrix = self._build_action_matrix()
        self.action_descriptions = self._build_descriptions()
        self.reverse_mapping = self._build_reverse_mapping()
    
    def _build_action_matrix(self) -> Dict[int, Tuple[int, int]]:
        """Build 3x3 matrix of (NVDA_action, MSFT_action) combinations"""
        actions = {}
        action_id = 0
        
        # Iterate through all combinations
        for msft_action in [SingleAssetAction.SELL, SingleAssetAction.HOLD, SingleAssetAction.BUY]:
            for nvda_action in [SingleAssetAction.SELL, SingleAssetAction.HOLD, SingleAssetAction.BUY]:
                actions[action_id] = (int(nvda_action), int(msft_action))
                action_id += 1
                
        return actions
    
    def _build_descriptions(self) -> Dict[int, str]:
        """Build human-readable action descriptions"""
        action_names = {
            SingleAssetAction.SELL: "SELL",
            SingleAssetAction.HOLD: "HOLD", 
            SingleAssetAction.BUY: "BUY"
        }
        
        descriptions = {}
        for action_id, (nvda_action, msft_action) in self.action_matrix.items():
            nvda_name = action_names[nvda_action]
            msft_name = action_names[msft_action]
            
            if nvda_action == msft_action:
                # Both assets same action
                descriptions[action_id] = f"{nvda_name}_BOTH"
            else:
                # Different actions
                descriptions[action_id] = f"{nvda_name}_NVDA_{msft_name}_MSFT"
                
        return descriptions
    
    def _build_reverse_mapping(self) -> Dict[Tuple[int, int], int]:
        """Build reverse mapping from (nvda_action, msft_action) to action_id"""
        return {actions: action_id for action_id, actions in self.action_matrix.items()}
    
    def decode_action(self, action_id: int) -> Tuple[int, int]:
        """
        Convert action ID to (nvda_action, msft_action)
        
        Args:
            action_id: Integer 0-8
            
        Returns:
            Tuple of (nvda_action, msft_action) where each is -1, 0, or 1
            
        Raises:
            ValueError: If action_id is invalid
        """
        if action_id not in self.action_matrix:
            raise ValueError(f"Invalid action ID: {action_id}. Must be 0-8.")
        return self.action_matrix[action_id]
    
    def encode_action(self, nvda_action: int, msft_action: int) -> int:
        """
        Convert (nvda_action, msft_action) to action ID
        
        Args:
            nvda_action: -1 (sell), 0 (hold), or 1 (buy)
            msft_action: -1 (sell), 0 (hold), or 1 (buy)
            
        Returns:
            Action ID (0-8)
            
        Raises:
            ValueError: If action combination is invalid
        """
        action_tuple = (nvda_action, msft_action)
        
        if action_tuple not in self.reverse_mapping:
            raise ValueError(f"Invalid action combination: {action_tuple}. "
                           f"Each action must be -1, 0, or 1.")
        
        return self.reverse_mapping[action_tuple]
    
    def get_action_description(self, action_id: int) -> str:
        """Get human-readable action description"""
        if action_id not in self.action_descriptions:
            raise ValueError(f"Invalid action ID: {action_id}")
        return self.action_descriptions[action_id]
    
    def get_all_actions(self) -> Dict[int, Dict[str, any]]:
        """Get complete action information"""
        return {
            action_id: {
                'nvda_action': nvda_action,
                'msft_action': msft_action,
                'description': self.action_descriptions[action_id],
                'action_tuple': (nvda_action, msft_action)
            }
            for action_id, (nvda_action, msft_action) in self.action_matrix.items()
        }
    
    def filter_actions_by_constraints(self, 
                                    current_nvda_position: int,
                                    current_msft_position: int,
                                    allow_position_increase: bool = True,
                                    max_position_magnitude: int = 1) -> List[int]:
        """
        Filter valid actions based on current positions and constraints
        
        Args:
            current_nvda_position: Current NVDA position (-1, 0, 1)
            current_msft_position: Current MSFT position (-1, 0, 1)
            allow_position_increase: Whether to allow increasing position magnitude
            max_position_magnitude: Maximum allowed position magnitude
            
        Returns:
            List of valid action IDs
        """
        valid_actions = []
        
        for action_id, (nvda_action, msft_action) in self.action_matrix.items():
            
            # Check position magnitude constraints
            if abs(nvda_action) > max_position_magnitude or abs(msft_action) > max_position_magnitude:
                continue
            
            # Check position increase constraints
            if not allow_position_increase:
                nvda_increase = abs(nvda_action) > abs(current_nvda_position)
                msft_increase = abs(msft_action) > abs(current_msft_position)
                
                if nvda_increase or msft_increase:
                    continue
            
            valid_actions.append(action_id)
        
        return valid_actions
    
    def get_neutral_action(self) -> int:
        """Get the neutral (HOLD_BOTH) action ID"""
        return self.encode_action(SingleAssetAction.HOLD, SingleAssetAction.HOLD)
    
    def get_aggressive_actions(self) -> List[int]:
        """Get actions that involve buying or selling both assets"""
        aggressive = []
        
        for action_id, (nvda_action, msft_action) in self.action_matrix.items():
            # Both actions are non-zero (not holding)
            if nvda_action != 0 and msft_action != 0:
                aggressive.append(action_id)
        
        return aggressive
    
    def get_conservative_actions(self) -> List[int]:
        """Get actions that involve holding at least one asset"""
        conservative = []
        
        for action_id, (nvda_action, msft_action) in self.action_matrix.items():
            # At least one action is hold
            if nvda_action == 0 or msft_action == 0:
                conservative.append(action_id)
        
        return conservative
    
    def calculate_portfolio_turnover(self, 
                                   action_id: int,
                                   current_nvda_position: int,
                                   current_msft_position: int) -> float:
        """
        Calculate portfolio turnover for a given action
        
        Args:
            action_id: Proposed action
            current_nvda_position: Current NVDA position
            current_msft_position: Current MSFT position
            
        Returns:
            Portfolio turnover (0.0 to 2.0, where 2.0 = both assets flip positions)
        """
        nvda_action, msft_action = self.decode_action(action_id)
        
        # Calculate position changes
        nvda_change = abs(nvda_action - current_nvda_position)
        msft_change = abs(msft_action - current_msft_position)
        
        # Normalize by maximum possible change (2 = full flip from -1 to +1)
        nvda_turnover = nvda_change / 2.0
        msft_turnover = msft_change / 2.0
        
        # Portfolio turnover is average of asset turnovers
        portfolio_turnover = (nvda_turnover + msft_turnover) / 2.0
        
        return portfolio_turnover
    
    def analyze_action_distribution(self, action_history: List[int]) -> Dict[str, any]:
        """
        Analyze the distribution of actions taken
        
        Args:
            action_history: List of action IDs taken during trading
            
        Returns:
            Dictionary with action distribution statistics
        """
        if not action_history:
            return {'error': 'No action history provided'}
        
        # Count action frequencies
        action_counts = {}
        for action_id in action_history:
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
        
        total_actions = len(action_history)
        
        # Calculate statistics
        action_stats = {}
        for action_id, count in action_counts.items():
            action_stats[action_id] = {
                'count': count,
                'frequency': count / total_actions,
                'description': self.get_action_description(action_id)
            }
        
        # Analyze action types
        neutral_count = action_counts.get(self.get_neutral_action(), 0)
        aggressive_actions = self.get_aggressive_actions()
        aggressive_count = sum(action_counts.get(aid, 0) for aid in aggressive_actions)
        
        return {
            'total_actions': total_actions,
            'unique_actions': len(action_counts),
            'action_distribution': action_stats,
            'neutral_frequency': neutral_count / total_actions,
            'aggressive_frequency': aggressive_count / total_actions,
            'most_common_action': max(action_counts, key=action_counts.get),
            'action_diversity': len(action_counts) / 9  # Out of 9 possible actions
        }
    
    def validate_action_sequence(self, action_sequence: List[int]) -> Dict[str, any]:
        """
        Validate a sequence of actions for common issues
        
        Args:
            action_sequence: List of action IDs
            
        Returns:
            Validation results with any issues found
        """
        issues = []
        warnings = []
        
        # Check for invalid actions
        for i, action_id in enumerate(action_sequence):
            if action_id not in self.action_matrix:
                issues.append(f"Step {i}: Invalid action ID {action_id}")
        
        # Check for excessive neutral actions
        neutral_action = self.get_neutral_action()
        neutral_count = action_sequence.count(neutral_action)
        neutral_pct = neutral_count / len(action_sequence) if action_sequence else 0
        
        if neutral_pct > 0.8:
            warnings.append(f"High neutral action rate: {neutral_pct:.1%}")
        
        # Check for action diversity
        unique_actions = len(set(action_sequence))
        if unique_actions < 3:
            warnings.append(f"Low action diversity: {unique_actions}/9 actions used")
        
        # Check for rapid position changes (potential ping-ponging)
        if len(action_sequence) > 1:
            rapid_changes = 0
            for i in range(1, len(action_sequence)):
                prev_nvda, prev_msft = self.decode_action(action_sequence[i-1])
                curr_nvda, curr_msft = self.decode_action(action_sequence[i])
                
                # Check if positions flipped completely
                if (prev_nvda * curr_nvda < 0) or (prev_msft * curr_msft < 0):
                    rapid_changes += 1
            
            if rapid_changes > len(action_sequence) * 0.3:
                warnings.append(f"Potential ping-ponging: {rapid_changes} rapid position changes")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'sequence_length': len(action_sequence),
            'neutral_rate': neutral_pct
        }
    
    def __str__(self) -> str:
        """String representation of action space"""
        lines = ["Portfolio Action Space (9 actions):"]
        lines.append("ID | NVDA | MSFT | Description")
        lines.append("---|------|------|------------")
        
        for action_id in range(9):
            nvda_action, msft_action = self.decode_action(action_id)
            description = self.get_action_description(action_id)
            
            nvda_str = {-1: "SELL", 0: "HOLD", 1: "BUY"}[nvda_action]
            msft_str = {-1: "SELL", 0: "HOLD", 1: "BUY"}[msft_action]
            
            lines.append(f"{action_id:2} | {nvda_str:4} | {msft_str:4} | {description}")
        
        return "\n".join(lines)