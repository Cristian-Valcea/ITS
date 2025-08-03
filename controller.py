"""
Dual-Lane Proportional Controller for V3 Trading Frequency Optimization

This module implements the reviewer-validated dual-lane controller that enhances
V3's trading behavior through industrial control theory while preserving its
proven risk management characteristics.

Key Features:
- Dual-lane control: Fast response (every step) + Slow drift control (every 25 steps)
- Integral wind-up protection for oscillating market regimes
- Scalar float return validation (reviewer requirement)
- Mathematical stability guarantees with bounded output

Author: Stairways to Heaven v3.0 Implementation
Created: August 3, 2025
"""

import numpy as np
import math
from typing import Union


class DualLaneController:
    """
    Industrial-grade dual-lane proportional controller for trading frequency optimization.
    
    This controller implements a two-speed control system:
    - Fast lane: Immediate response to market regime changes (every step)
    - Slow lane: Long-term drift control and stability (every 25 steps)
    
    The design prevents integral wind-up during oscillating market conditions
    while maintaining responsive adaptation to genuine regime shifts.
    
    Attributes:
        kp_fast (float): Fast lane proportional gain (0.25 - reviewer approved)
        kp_slow (float): Slow lane proportional gain (0.05 - reviewer approved)  
        base_bonus (float): Base hold bonus from V3 original (typically 0.01)
        slow_adj (float): Persistent slow lane adjustment
        step (int): Step counter for slow lane sample-and-hold
    """
    
    def __init__(self, base_hold_bonus: float):
        """
        Initialize the dual-lane controller.
        
        Args:
            base_hold_bonus (float): Base hold bonus from V3 environment (typically 0.01)
        """
        # Reviewer-approved control gains
        self.kp_fast = 0.25          # Fast lane gain (high responsiveness)
        self.kp_slow = 0.05          # Slow lane gain (drift control)
        
        # Controller state
        self.slow_adj = 0.0          # Persistent slow adjustment
        self.base_bonus = base_hold_bonus
        self.step = 0                # Step counter for slow lane timing
        
        # Validation
        if base_hold_bonus <= 0:
            raise ValueError(f"base_hold_bonus must be positive, got {base_hold_bonus}")
    
    def compute_bonus(self, hold_error: float, regime_score: float) -> float:
        """
        Compute adaptive hold bonus based on current trading behavior and market regime.
        
        This is the core control logic that combines fast market response with 
        slow drift correction to optimize trading frequency while preventing
        integral wind-up during oscillating conditions.
        
        Args:
            hold_error (float): Current holding rate error [-1, 1]
                               Positive = holding too much, Negative = trading too much
            regime_score (float): Market regime score [-3, 3], pre-clamped
                                 Positive = high opportunity, Negative = low opportunity
        
        Returns:
            float: Adaptive hold bonus, clipped to 2 × base_hold_bonus
                  (currently [0, 0.02] with base_hold_bonus=0.01)
        
        Raises:
            TypeError: If inputs are not numeric
            
        Notes:
            - REVIEWER CRITICAL: Always returns scalar float (never array)
            - Fast lane uses market multiplier for immediate regime response
            - Slow lane ignores market multiplier for long-term stability
            - Output is hard-clipped to prevent controller wind-up
        """
        # Input validation
        if not isinstance(hold_error, (int, float)):
            raise TypeError(f"hold_error must be numeric, got {type(hold_error)}")
        if not isinstance(regime_score, (int, float)):
            raise TypeError(f"regime_score must be numeric, got {type(regime_score)}")
        
        # Convert to float to ensure scalar return
        hold_error = float(hold_error)
        regime_score = float(regime_score)
        
        # NaN handling - replace NaN with neutral values
        if math.isnan(hold_error):
            hold_error = 0.0
        if math.isnan(regime_score):
            regime_score = 0.0
        
        # Infinity handling - clamp to reasonable bounds
        if math.isinf(hold_error):
            hold_error = 1.0 if hold_error > 0 else -1.0
        if math.isinf(regime_score):
            regime_score = 3.0 if regime_score > 0 else -3.0
        
        # Market multiplier transformation (30% adjustment range)
        # Allows ±30% modification based on market opportunity
        market_mult = 1.0 + regime_score * 0.3
        
        # Fast lane: Immediate market regime response
        # Uses market multiplier to respond quickly to regime changes
        fast = self.kp_fast * hold_error * market_mult
        
        # Slow lane: IIR accumulator every 25 steps (NO market multiplier)
        # Provides long-term drift correction without market noise
        # R3 FIX: Accumulate instead of replace for IIR smoother behavior
        if self.step % 25 == 0:
            self.slow_adj += self.kp_slow * hold_error
            # Clamp slow adjustment to prevent runaway accumulation
            self.slow_adj = np.clip(self.slow_adj, -0.5, 0.5)
        self.step += 1
        
        # Combined adjustment calculation
        # Base bonus is modified by both fast and slow components
        bonus = self.base_bonus * (1 + fast + self.slow_adj)
        
        # REVIEWER CRITICAL: Hard clipping prevents integral wind-up
        # Bounds: [0, 2 × base_hold_bonus] - currently [0, 0.02] with base=0.01
        clipped_bonus = np.clip(bonus, 0.0, 2.0 * self.base_bonus)
        
        # REVIEWER REQUIREMENT: Return scalar float (not array)
        return float(clipped_bonus)
    
    def reset_state(self):
        """
        Reset controller internal state.
        
        Useful for episode boundaries or when switching between different
        market conditions. Preserves configuration but clears accumulated state.
        """
        self.slow_adj = 0.0
        self.step = 0
    
    def get_controller_health(self) -> dict:
        """
        Get current controller health metrics for monitoring.
        
        Returns:
            dict: Controller health information including:
                - fast_gain: Current fast lane gain
                - slow_gain: Current slow lane gain  
                - slow_adjustment: Current slow lane accumulated adjustment
                - step_count: Current step counter
                - base_bonus: Base bonus being used
        """
        return {
            "fast_gain": self.kp_fast,
            "slow_gain": self.kp_slow,
            "slow_adjustment": self.slow_adj,
            "step_count": self.step,
            "base_bonus": self.base_bonus,
            "controller_type": "dual_lane_proportional"
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"DualLaneController(kp_fast={self.kp_fast}, kp_slow={self.kp_slow}, "
                f"base_bonus={self.base_bonus}, step={self.step})")


def test_controller_wind_up_protection():
    """
    REVIEWER REQUIREMENT: Test integral wind-up protection.
    
    This test validates that the controller remains bounded even under
    extreme oscillating conditions that would normally cause wind-up
    in simpler control systems.
    """
    controller = DualLaneController(0.01)
    
    print("Testing controller wind-up protection...")
    
    # Simulate oscillating hold_error ±0.6 for 100 steps (reviewer spec)
    max_bonus = 0.0
    min_bonus = float('inf')
    
    for i in range(100):
        hold_error = 0.6 * (-1) ** i  # Oscillating ±0.6
        regime_score = 2.0 * math.sin(i * 0.1)  # Oscillating regime
        bonus = controller.compute_bonus(hold_error, regime_score)
        
        max_bonus = max(max_bonus, bonus)
        min_bonus = min(min_bonus, bonus)
        
        # Verify bounded output despite oscillations
        assert 0.0 <= bonus <= 0.02, f"Bonus {bonus} outside bounds [0, 0.02] at step {i}"
        assert isinstance(bonus, float), f"Must return float, got {type(bonus)} at step {i}"
    
    print(f"✅ Wind-up protection test passed")
    print(f"   Bonus range: [{min_bonus:.6f}, {max_bonus:.6f}]")
    print(f"   All values within bounds [0.0, 0.02]")
    return True


def test_controller_return_type():
    """
    REVIEWER CRITICAL: Verify controller returns scalar float.
    
    Ensures the controller always returns a Python float, never a numpy array
    or other numeric type that could cause downstream issues.
    """
    controller = DualLaneController(0.01)
    
    test_cases = [
        (0.0, 0.0),    # Neutral case
        (0.5, 1.0),    # Positive case
        (-0.3, -2.0),  # Negative case
        (1.0, 3.0),    # Extreme positive (should be clipped)
        (-1.0, -3.0)   # Extreme negative (should be clipped)
    ]
    
    print("Testing controller return type...")
    
    for hold_error, regime_score in test_cases:
        result = controller.compute_bonus(hold_error, regime_score)
        
        # REVIEWER REQUIREMENT: Must return Python float
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert not isinstance(result, np.ndarray), f"Must not return array, got {type(result)}"
        
        # Verify bounds
        assert 0.0 <= result <= 0.02, f"Result {result} outside bounds [0, 0.02]"
    
    print("✅ Return type test passed - all results are scalar floats")
    return True


if __name__ == "__main__":
    """
    Quick validation of controller implementation.
    Run this module directly to perform basic controller tests.
    """
    print("🎛️ Dual-Lane Controller Validation")
    print("=" * 40)
    
    # Test 1: Basic functionality
    controller = DualLaneController(0.01)
    print(f"Controller initialized: {controller}")
    
    # Test 2: Basic computation
    bonus = controller.compute_bonus(0.5, 1.0)
    print(f"Sample computation: compute_bonus(0.5, 1.0) = {bonus}")
    
    # Test 3: Wind-up protection (reviewer requirement)
    test_controller_wind_up_protection()
    
    # Test 4: Return type validation (reviewer critical)
    test_controller_return_type()
    
    # Test 5: Health monitoring
    health = controller.get_controller_health()
    print(f"Controller health: {health}")
    
    print("\n✅ All controller validations passed!")
    print("Dual-lane controller is ready for integration.")