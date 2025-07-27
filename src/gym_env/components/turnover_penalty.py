# src/gym_env/components/turnover_penalty.py
"""
🎯 Turnover Penalty Calculator Component

This module provides a sophisticated turnover penalty system that addresses
the overtrading problem in reinforcement learning trading agents.

Key Features:
- Normalized turnover calculation (portfolio-value and episode-length adjusted)
- Smooth penalty functions (sigmoid/softplus) for LSTM-friendly gradients
- Adaptive weighting based on portfolio size
- Professional logging and debugging capabilities
- Clean separation of concerns from main environment

Author: IntradayJules Trading System
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, Union, Callable
import numpy as np


class TurnoverPenaltyCalculator:
    """
    🎯 Advanced Turnover Penalty Calculator
    
    Implements a normalized, smooth penalty system for controlling trading frequency
    in reinforcement learning environments. Designed to replace cliff-effect penalties
    with smooth, differentiable functions that provide proper learning gradients.
    
    The penalty is calculated using different curve types:
    
    normalized_turnover = total_turnover / (episode_length * portfolio_value)  # Proper normalization
    
    Curve Types:
    - sigmoid: penalty = -weight / (1 + exp(-k * (ratio - target)))  # Smooth S-curve
    - quadratic: penalty = -weight * (ratio / target)^2  # Quadratic growth for overtrading
    - softplus: penalty = -weight * softplus(k * (ratio - target))  # Smooth exponential
    - steep_softplus: penalty = -weight * softplus(10k * (ratio - target))  # Very steep
    
    Where weight scales with portfolio NAV (e.g., 15% of NAV) and k controls curve sharpness.
    This creates economically meaningful penalties that scale with portfolio size.
    """
    
    def __init__(self, 
                 portfolio_value_getter: Union[float, Callable[[], float]],
                 episode_length_getter: Union[int, Callable[[], int]] = None,
                 target_ratio: float = 0.02,
                 weight_factor: float = 0.02,
                 curve_sharpness: float = 25.0,
                 curve: str = "sigmoid",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Turnover Penalty Calculator.
        
        Args:
            portfolio_value_getter: Either a float (static) or callable returning current portfolio value
            episode_length_getter: Either an int (static) or callable returning current episode length
            target_ratio (float): Target turnover ratio (default: 2% = 0.02)
            weight_factor (float): Penalty weight as fraction of NAV (default: 2% = 0.02)
            curve_sharpness (float): Curve sharpness parameter k (default: 25.0)
            curve (str): Penalty curve type ('sigmoid', 'softplus', 'quadratic', 'steep_softplus')
            logger (Optional[logging.Logger]): Logger instance for debugging
        """
        self.portfolio_value_getter = portfolio_value_getter
        self.episode_length_getter = episode_length_getter or (lambda: 390)  # Default: 1 trading day
        self.target_ratio = target_ratio
        self.weight_factor = weight_factor
        self.curve_sharpness = curve_sharpness
        self.curve = curve.lower()
        
        # No need to pre-calculate weight - it will scale dynamically with portfolio value
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        if not logger:  # Only set propagate if we created our own logger
            self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Validation
        self._validate_parameters()
        
        # Performance tracking
        self.penalty_history = []
        self.normalized_turnover_history = []
        
        # 🎯 DYNAMIC CURRICULUM LEARNING
        self.base_weight_factor = weight_factor  # Store original weight
        self.turnover_history_1d = []  # Track recent turnover for curriculum
        self.curriculum_window = 20  # Episodes to average for curriculum decision
        
        self.logger.debug(
            f"🎯 TurnoverPenaltyCalculator initialized: "
            f"target_ratio={self.target_ratio:.3f}, weight_factor={self.weight_factor:.3f}, "
            f"curve={self.curve}, sharpness={self.curve_sharpness}"
        )
    
    def _get_current_portfolio_value(self) -> float:
        """Get the current portfolio value, either from static value or callable."""
        if callable(self.portfolio_value_getter):
            return self.portfolio_value_getter()
        else:
            return self.portfolio_value_getter
    
    def _get_current_episode_length(self) -> int:
        """Get the current episode length, either from static value or callable."""
        if callable(self.episode_length_getter):
            return self.episode_length_getter()
        else:
            return self.episode_length_getter
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        current_portfolio_value = self._get_current_portfolio_value()
        if current_portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {current_portfolio_value}")
        
        if self.target_ratio < 0:
            raise ValueError(f"Target ratio must be non-negative, got {self.target_ratio}")
        
        if self.weight_factor <= 0:
            raise ValueError(f"Weight factor must be positive, got {self.weight_factor}")
        
        if self.curve not in ['sigmoid', 'softplus', 'quadratic', 'steep_softplus']:
            raise ValueError(f"Unsupported curve type: {self.curve}. Use 'sigmoid', 'softplus', 'quadratic', or 'steep_softplus'")
        
        if self.curve_sharpness <= 0:
            raise ValueError(f"Curve sharpness must be positive, got {self.curve_sharpness}")
    
    def _update_dynamic_curriculum(self, turnover_ratio: float) -> float:
        """
        🎯 DYNAMIC CURRICULUM LEARNING
        
        Adjusts penalty weight based on recent overtrading behavior:
        - Start mild (α = base_weight_factor)
        - Escalate if avg_turnover > 2x target (α = 3x base)
        - Severe if avg_turnover > 10x target (α = 7x base)
        
        Args:
            turnover_ratio (float): Current episode's turnover ratio
            
        Returns:
            float: Adjusted weight factor for this episode
        """
        # Track recent turnover ratios
        self.turnover_history_1d.append(turnover_ratio)
        
        # Keep only recent history for curriculum decision
        if len(self.turnover_history_1d) > self.curriculum_window:
            self.turnover_history_1d.pop(0)
        
        # Calculate average recent turnover
        if len(self.turnover_history_1d) < 5:  # Need some history
            return self.base_weight_factor  # Start mild
        
        avg_turnover = sum(self.turnover_history_1d) / len(self.turnover_history_1d)
        
        # 🎯 DYNAMIC CURRICULUM ESCALATION
        if avg_turnover > 10 * self.target_ratio:
            # Severe overtrading: 7x penalty weight
            adjusted_weight = self.base_weight_factor * 7.0
            curriculum_level = "SEVERE"
        elif avg_turnover > 2 * self.target_ratio:
            # Moderate overtrading: 3x penalty weight  
            adjusted_weight = self.base_weight_factor * 3.0
            curriculum_level = "MODERATE"
        else:
            # Good behavior: base penalty weight
            adjusted_weight = self.base_weight_factor
            curriculum_level = "MILD"
        
        # Log curriculum adjustments
        if len(self.turnover_history_1d) % 10 == 0:  # Log every 10 episodes
            self.logger.info(
                f"🎯 CURRICULUM: {curriculum_level} | "
                f"Avg Turnover: {avg_turnover:.3f} | Target: {self.target_ratio:.3f} | "
                f"Weight: {adjusted_weight:.3f} ({adjusted_weight/self.base_weight_factor:.1f}x base)"
            )
        
        return adjusted_weight
    
    def compute_penalty(self, total_turnover: float, step: Optional[int] = None) -> float:
        """
        Compute the turnover penalty for the given turnover.
        
        Uses the CORRECTED normalized formula:
        normalized_turnover = total_turnover / (episode_length * portfolio_value)  # Proper normalization
        weight = penalty_weight_pct * portfolio_value  # Scales with NAV
        penalty = -weight * softplus(k * (normalized_turnover - target_ratio))  # Smooth penalty
        
        Args:
            total_turnover (float): Total traded value for the current day
            
        Returns:
            float: Penalty value (negative for penalty, 0 for no penalty)
        """
        if total_turnover < 0:
            raise ValueError(f"Total turnover must be non-negative, got {total_turnover}")
        
        # Step 1: Get current portfolio value
        current_portfolio_value = self._get_current_portfolio_value()
        if current_portfolio_value <= 0:
            return 0.0
        
        # Step 2: Calculate NORMALIZED turnover ratio (your correct approach)
        # normalized_turnover = turnover / (episode_length * portfolio_value)
        current_episode_length = self._get_current_episode_length()
        normalized_turnover = total_turnover / (current_episode_length * current_portfolio_value + 1e-6)
        
        # For comparison, calculate the old (incorrect) ratio
        old_ratio = total_turnover / (current_portfolio_value + 1e-6)
        
        # Use the normalized version
        turnover_ratio = normalized_turnover
        
        # Log the difference for debugging
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 1
            
        if self._log_counter % 100 == 0:  # Log every 100 calls
            self.logger.info(
                f"🎯 NORMALIZATION COMPARISON - "
                f"Old ratio: {old_ratio:.6f}, "
                f"New normalized: {normalized_turnover:.6f}, "
                f"Episode length: {current_episode_length}, "
                f"Improvement: {old_ratio/normalized_turnover:.1f}x reduction" if normalized_turnover > 0 else f"Improvement: No trading (infinite reduction)"
            )
        
        # Step 3: 🎯 DYNAMIC CURRICULUM - Adjust penalty weight based on behavior
        dynamic_weight_factor = self._update_dynamic_curriculum(turnover_ratio)
        # NOTE: dynamic_weight_factor is already a percentage (0.15, 0.45, 1.05)
        # We'll apply it to NAV inside each penalty calculation to avoid double multiplication
        
        # Step 4: Calculate penalty using different curve types
        if self.curve == "sigmoid":
            # 🎯 FIXED Sigmoid: Zero turnover = zero penalty, proper neutral zone
            if turnover_ratio == 0.0:
                # Zero turnover: neutral (no penalty, no reward)
                penalty = 0.0
            else:
                # Sigmoid centered at target ratio
                # When ratio < target: sigmoid → 0, penalty → positive (reward)
                # When ratio > target: sigmoid → 1, penalty → negative (penalty)
                excess_input = self.curve_sharpness * (turnover_ratio - self.target_ratio)
                clipped_input = np.clip(excess_input, -50.0, 50.0)  # Prevent overflow
                sigmoid_factor = 1.0 / (1.0 + np.exp(-clipped_input))  # Numerically safe sigmoid
                # Transform: 0.5 = neutral, <0.5 = reward, >0.5 = penalty
                penalty = dynamic_weight_factor * current_portfolio_value * (0.5 - sigmoid_factor) * 2.0
            
        elif self.curve == "softplus":
            # 🎯 FIXED Softplus: Zero turnover = zero penalty, proper neutral zone
            if turnover_ratio == 0.0:
                # Zero turnover: neutral (no penalty, no reward)
                penalty = 0.0
            elif turnover_ratio <= self.target_ratio:
                # Below target but > 0: small reward proportional to how close to target
                # Scale reward so it's zero at target and small positive below target
                distance_from_target = self.target_ratio - turnover_ratio
                max_distance = self.target_ratio  # Maximum possible distance (from 0 to target)
                efficiency_ratio = distance_from_target / max_distance  # 0 to 1
                penalty = dynamic_weight_factor * current_portfolio_value * 0.05 * efficiency_ratio  # Reduced from 0.1 to 0.05
            else:
                # Above target: softplus penalty with clipping to prevent explosion
                excess_input = self.curve_sharpness * (turnover_ratio - self.target_ratio)
                clipped_input = np.clip(excess_input, -50.0, 50.0)  # Prevent FP64 overflow
                penalty_factor = F.softplus(torch.tensor(clipped_input))
                # Additional clipping on penalty factor itself
                penalty_factor_clipped = min(penalty_factor.item(), 100.0)  # Cap at 100x
                penalty = -dynamic_weight_factor * current_portfolio_value * penalty_factor_clipped
            
        elif self.curve == "quadratic":
            # 🎯 FIXED: Zero turnover = zero penalty, proper neutral zone
            if turnover_ratio == 0.0:
                # Zero turnover: neutral (no penalty, no reward)
                penalty = 0.0
            elif turnover_ratio <= self.target_ratio:
                # Below target but > 0: small reward proportional to distance from target
                distance_from_target = self.target_ratio - turnover_ratio
                max_distance = self.target_ratio
                efficiency_ratio = distance_from_target / max_distance
                penalty = dynamic_weight_factor * current_portfolio_value * 0.05 * efficiency_ratio  # POSITIVE reward
            else:
                # Above target: GENTLE QUADRATIC penalty (negative value)
                excess_over_target = max(0, turnover_ratio - self.target_ratio)
                # Gentle quadratic penalty: penalty grows smoothly with excess
                normalized_excess = excess_over_target / (self.target_ratio + 1e-6)
                # Use a gentler quadratic curve: 0.5 * x^2 instead of x^2
                quadratic_factor = 0.5 * (normalized_excess ** 2)
                # Cap the quadratic factor to prevent explosion
                capped_quadratic = min(quadratic_factor, 2.0)  # Cap at 2x for very high turnover
                penalty = -dynamic_weight_factor * current_portfolio_value * capped_quadratic  # NEGATIVE penalty
                
        elif self.curve == "steep_softplus":
            # 🎯 FIXED Steep Softplus: Zero turnover = zero penalty, proper neutral zone
            if turnover_ratio == 0.0:
                # Zero turnover: neutral (no penalty, no reward)
                penalty = 0.0
            elif turnover_ratio <= self.target_ratio:
                # Below target but > 0: small reward proportional to distance from target
                distance_from_target = self.target_ratio - turnover_ratio
                max_distance = self.target_ratio
                efficiency_ratio = distance_from_target / max_distance
                penalty = dynamic_weight_factor * current_portfolio_value * 0.05 * efficiency_ratio
            else:
                # Above target: steep softplus penalty with clipping
                steep_beta = self.curve_sharpness * 10.0  # Make it extra steep
                excess_input = steep_beta * (turnover_ratio - self.target_ratio)
                clipped_input = np.clip(excess_input, -50.0, 50.0)  # Prevent FP64 overflow
                penalty_factor = F.softplus(torch.tensor(clipped_input))
                # Additional clipping on penalty factor itself
                penalty_factor_clipped = min(penalty_factor.item(), 100.0)  # Cap at 100x
                penalty = -dynamic_weight_factor * current_portfolio_value * penalty_factor_clipped
            
        else:
            # 🎯 FIXED Fallback sigmoid with proper neutral zone
            if turnover_ratio == 0.0:
                # Zero turnover: neutral (no penalty, no reward)
                penalty = 0.0
            else:
                # Sigmoid centered at target ratio
                excess_input = self.curve_sharpness * (turnover_ratio - self.target_ratio)
                clipped_input = np.clip(excess_input, -50.0, 50.0)  # Prevent overflow
                sigmoid_factor = 1.0 / (1.0 + np.exp(-clipped_input))  # Numerically safe sigmoid
                # Adjust so that at target ratio, penalty = 0
                penalty = dynamic_weight_factor * current_portfolio_value * (0.5 - sigmoid_factor) * 2.0
        
        # Track for analysis
        self.penalty_history.append(penalty)
        self.normalized_turnover_history.append(turnover_ratio)  # Now stores actual ratio
        
        # 🎯 DETAILED TURNOVER PENALTY LOGGING (Your requested format)
        if step is not None and (step % 50 == 0 or abs(penalty) > 100):  # Log every 50 steps or significant penalties
            # Calculate weight used in penalty calculation
            weight_used = dynamic_weight_factor * current_portfolio_value
            
            # Calculate old ratio for comparison (without episode length normalization)
            old_ratio = total_turnover / (current_portfolio_value + 1e-6)
            
            self.logger.info(
                f"🎯 [TurnoverPenalty] Step {step}, Ratio: {old_ratio:.5f} (Norm: {turnover_ratio:.5f}) | "
                f"Target: {self.target_ratio:.3f} | Penalty: ${penalty:.2f} | "
                f"Weight: ${weight_used:.2f} | Total: ${total_turnover:.0f} | "
                f"Portfolio: ${current_portfolio_value:.2f}"
            )
        
        return penalty
    
    def compute_penalty_gradient(self, total_turnover: float) -> float:
        """
        Compute the gradient of the penalty with respect to turnover.
        Useful for understanding the penalty landscape.
        
        Args:
            total_turnover (float): Total traded value
            
        Returns:
            float: Penalty gradient
        """
        current_portfolio_value = self._get_current_portfolio_value()
        if current_portfolio_value <= 0:
            return 0.0
        
        # Calculate turnover ratio
        turnover_ratio = total_turnover / (current_portfolio_value + 1e-6)
        penalty_weight = self.weight_factor * current_portfolio_value
        
        if self.curve == "sigmoid":
            # Gradient of sigmoid penalty: d/dx[-w/(1+exp(-k*(x-t)))] = -w*k*exp(-k*(x-t))/(1+exp(-k*(x-t)))^2
            exp_term = np.exp(-self.curve_sharpness * (turnover_ratio - self.target_ratio))
            gradient = -penalty_weight * self.curve_sharpness * exp_term / ((1 + exp_term) ** 2)
            # Chain rule: multiply by d(ratio)/d(turnover) = 1/portfolio_value
            gradient = gradient / current_portfolio_value
        elif self.curve == "softplus":
            # Gradient of softplus penalty
            sigmoid_val = torch.sigmoid(torch.tensor(self.curve_sharpness * (turnover_ratio - self.target_ratio)))
            gradient = -penalty_weight * self.curve_sharpness * sigmoid_val.item()
            # Chain rule: multiply by d(ratio)/d(turnover) = 1/portfolio_value
            gradient = gradient / current_portfolio_value
        else:
            # Fallback to sigmoid
            exp_term = np.exp(-self.curve_sharpness * (turnover_ratio - self.target_ratio))
            gradient = -penalty_weight * self.curve_sharpness * exp_term / ((1 + exp_term) ** 2)
            gradient = gradient / current_portfolio_value
        
        return gradient
    

    
    def get_penalty_curve_preview(self, turnover_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate a preview of the penalty curve for visualization.
        
        Args:
            turnover_range (np.ndarray): Range of turnover values to evaluate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with turnover values and corresponding penalties
        """
        penalties = []
        turnover_ratios = []
        current_portfolio_value = self._get_current_portfolio_value()
        
        for turnover in turnover_range:
            penalty = self.compute_penalty(turnover, step=None)  # No step for preview
            ratio = turnover / (current_portfolio_value + 1e-6)
            
            penalties.append(penalty)
            turnover_ratios.append(ratio)
        
        return {
            'turnover_values': turnover_range,
            'turnover_ratios': np.array(turnover_ratios),
            'penalties': np.array(penalties),
            'target_ratio': self.target_ratio,
            'curve_type': self.curve,
            'portfolio_value': current_portfolio_value
        }
    
    def log_debug(self, total_turnover: float, step: Optional[int] = None, penalty: Optional[float] = None) -> None:
        """
        Log detailed debug information about the penalty calculation.
        
        Args:
            total_turnover (float): Total traded value
            step (Optional[int]): Current step number for context
            penalty (Optional[float]): Pre-calculated penalty to avoid duplicate computation
        """
        current_portfolio_value = self._get_current_portfolio_value()
        if current_portfolio_value <= 0:
            self.logger.warning(
                f"🎯 [TurnoverPenalty] INVALID STATE: "
                f"portfolio_value={current_portfolio_value}"
            )
            return
        
        turnover_ratio = total_turnover / (current_portfolio_value + 1e-6)
        
        # Use provided penalty or calculate it (but don't duplicate)
        if penalty is None:
            penalty = self.compute_penalty(total_turnover, step=step)
        
        penalty_weight = self.weight_factor * current_portfolio_value
        step_info = f"Step {step}, " if step is not None else ""
        
        # 🔧 PERFORMANCE: Only log if debug level is enabled
        if self.logger.isEnabledFor(logging.DEBUG):
            # Raw input logging for debugging (only once per call)
            print(f"[TurnoverPenalty] Raw: turnover={total_turnover:.2f}, portfolio={current_portfolio_value:.2f}")
        
        # Enhanced debug logging with all calculation details (gated for performance)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
            f"🎯 [TurnoverPenalty] {step_info}"
            f"Ratio: {turnover_ratio:.5f} | "
            f"Target: {self.target_ratio:.3f} | "
            f"Penalty: ${penalty:.2f} | "
            f"Weight: ${penalty_weight:.2f} | "
            f"Total: ${total_turnover:.0f} | "
            f"Portfolio: ${current_portfolio_value:.2f}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about penalty calculations.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        if not self.penalty_history:
            return {
                'num_calculations': 0,
                'avg_penalty': 0.0,
                'avg_normalized_turnover': 0.0,
                'penalty_std': 0.0,
                'turnover_std': 0.0
            }
        
        return {
            'num_calculations': len(self.penalty_history),
            'avg_penalty': np.mean(self.penalty_history),
            'avg_turnover_ratio': np.mean(self.normalized_turnover_history),
            'penalty_std': np.std(self.penalty_history),
            'turnover_ratio_std': np.std(self.normalized_turnover_history),
            'min_penalty': np.min(self.penalty_history),
            'max_penalty': np.max(self.penalty_history),
            'target_ratio': self.target_ratio,
            'curve_type': self.curve,
            'weight_factor': self.weight_factor,
            'current_portfolio_value': self._get_current_portfolio_value()
        }
    
    def reset_history(self) -> None:
        """Reset penalty and turnover history."""
        self.penalty_history.clear()
        self.normalized_turnover_history.clear()
        self.logger.debug("🎯 TurnoverPenalty history reset")
    
    def __repr__(self) -> str:
        """String representation of the calculator."""
        current_portfolio_value = self._get_current_portfolio_value()
        return (
            f"TurnoverPenaltyCalculator("
            f"target_ratio={self.target_ratio:.3f}, "
            f"weight_factor={self.weight_factor:.3f}, "
            f"curve={self.curve}, "
            f"portfolio_value=${current_portfolio_value:.2f})"
        )


class TurnoverPenaltyFactory:
    """
    🏭 Factory class for creating TurnoverPenaltyCalculator instances
    with common configurations.
    """
    
    @staticmethod
    def create_conservative(portfolio_value: float, 
                          logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create a conservative penalty calculator (low turnover target)."""
        return TurnoverPenaltyCalculator(
            portfolio_value_getter=portfolio_value,
            target_ratio=0.01,  # 1% target
            weight_factor=0.03,  # 3% of NAV penalty weight
            curve_sharpness=30.0,  # Steeper curve
            curve="sigmoid",
            logger=logger
        )
    
    @staticmethod
    def create_moderate(portfolio_value: float,
                       logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create a moderate penalty calculator (balanced approach)."""
        return TurnoverPenaltyCalculator(
            portfolio_value_getter=portfolio_value,
            target_ratio=0.02,  # 2% target
            weight_factor=0.02,  # 2% of NAV penalty weight
            curve_sharpness=25.0,  # Moderate curve
            curve="sigmoid",
            logger=logger
        )
    
    @staticmethod
    def create_aggressive(portfolio_value: float,
                         logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create an aggressive penalty calculator (higher turnover allowed)."""
        return TurnoverPenaltyCalculator(
            portfolio_value_getter=portfolio_value,
            target_ratio=0.05,  # 5% target
            weight_factor=0.01,  # 1% of NAV penalty weight
            curve_sharpness=15.0,  # Gentler curve
            curve="sigmoid",
            logger=logger
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Example: Create a moderate penalty calculator
    calculator = TurnoverPenaltyFactory.create_moderate(
        portfolio_value=50000.0,
        logger=logger
    )
    
    # Test penalty calculation
    test_turnovers = [0, 1000, 5000, 10000, 25000, 50000]
    
    print("\n🎯 Turnover Penalty Calculator Test")
    print("=" * 50)
    print(f"Portfolio: $50,000, Target: 2%, Weight: 2% of NAV")
    print("=" * 50)
    
    for turnover in test_turnovers:
        penalty = calculator.compute_penalty(turnover, step=None)  # No step for test
        ratio = turnover / 50000.0
        print(f"Turnover: ${turnover:>6.0f} | Ratio: {ratio:>6.1%} | Penalty: ${penalty:>8.2f}")
    
    # Show statistics
    stats = calculator.get_statistics()
    print(f"\n📊 Statistics: {stats}")
    
    # Preview penalty curve
    turnover_range = np.linspace(0, 100000, 100)
    curve_data = calculator.get_penalty_curve_preview(turnover_range)
    
    print(f"\n📈 Penalty curve generated with {len(curve_data['penalties'])} points")
    print(f"Target ratio: {curve_data['target_ratio']:.1%}")
    print(f"Curve type: {curve_data['curve_type']}")
    print(f"Portfolio value: ${curve_data['portfolio_value']:,.2f}")