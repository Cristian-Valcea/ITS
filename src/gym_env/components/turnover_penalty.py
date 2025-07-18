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
    
    The penalty is calculated as:
    normalized_turnover = total_turnover / (portfolio_value * episode_length)
    relative_excess = (normalized_turnover - target) / target  # Adaptive scaling
    penalty = -adaptive_weight * smooth_function(smoothness * relative_excess)
    
    Where smooth_function can be sigmoid or softplus. The adaptive scaling ensures
    consistent penalty sharpness across different target ranges.
    """
    
    def __init__(self, 
                 episode_length: int,
                 portfolio_value_getter: Union[float, Callable[[], float]],
                 target_range: float = 0.02,
                 adaptive_weight_factor: float = 0.001,
                 smoothness: float = 10.0,
                 curve: str = "softplus",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Turnover Penalty Calculator.
        
        Args:
            episode_length (int): Total number of steps in the episode
            portfolio_value_getter: Either a float (static) or callable returning current portfolio value
            target_range (float): Target normalized turnover (default: 2%)
            adaptive_weight_factor (float): Base weight factor (scales with portfolio)
            smoothness (float): Smoothness parameter for penalty curves
            curve (str): Penalty curve type ('sigmoid' or 'softplus')
            logger (Optional[logging.Logger]): Logger instance for debugging
        """
        self.episode_length = episode_length
        self.portfolio_value_getter = portfolio_value_getter
        self.target_range = target_range
        self.adaptive_weight_factor = adaptive_weight_factor
        self.smoothness = smoothness
        self.curve = curve.lower()
        
        # Get initial portfolio value for weight calculation
        initial_portfolio_value = self._get_current_portfolio_value()
        
        # Calculate adaptive weight (scales with portfolio size)
        self.adaptive_weight = adaptive_weight_factor * initial_portfolio_value
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation
        self._validate_parameters()
        
        # Performance tracking
        self.penalty_history = []
        self.normalized_turnover_history = []
        
        self.logger.debug(
            f"🎯 TurnoverPenaltyCalculator initialized: "
            f"target={target_range:.3f}, weight={self.adaptive_weight:.2f}, "
            f"curve={curve}, smoothness={smoothness}"
        )
    
    def _get_current_portfolio_value(self) -> float:
        """Get the current portfolio value, either from static value or callable."""
        if callable(self.portfolio_value_getter):
            return self.portfolio_value_getter()
        else:
            return self.portfolio_value_getter
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.episode_length <= 0:
            raise ValueError(f"Episode length must be positive, got {self.episode_length}")
        
        current_portfolio_value = self._get_current_portfolio_value()
        if current_portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {current_portfolio_value}")
        
        if self.target_range < 0:
            raise ValueError(f"Target range must be non-negative, got {self.target_range}")
        
        if self.curve not in ['sigmoid', 'softplus']:
            raise ValueError(f"Unsupported curve type: {self.curve}. Use 'sigmoid' or 'softplus'")
        
        if self.smoothness <= 0:
            raise ValueError(f"Smoothness must be positive, got {self.smoothness}")
    
    def compute_penalty(self, total_turnover: float) -> float:
        """
        Compute the turnover penalty for the given total turnover.
        
        Args:
            total_turnover (float): Total traded value in the episode
            
        Returns:
            float: Penalty value (negative for penalty, 0 for no penalty)
        """
        if total_turnover < 0:
            raise ValueError(f"Total turnover must be non-negative, got {total_turnover}")
        
        # Step 1: Calculate normalized turnover using current portfolio value
        current_portfolio_value = self._get_current_portfolio_value()
        if self.episode_length == 0 or current_portfolio_value == 0:
            return 0.0
        
        normalized_turnover = total_turnover / (current_portfolio_value * self.episode_length)
        
        # Step 2: Calculate penalty using adaptive scaling
        # Normalize excess turnover by target range for consistent sharpness
        excess_turnover = normalized_turnover - self.target_range
        
        # Adaptive scaling: divide by target_range to get relative deviation
        if self.target_range > 0:
            relative_excess = excess_turnover / self.target_range
        else:
            relative_excess = excess_turnover  # Fallback if target is zero
        
        if self.curve == "sigmoid":
            # Sigmoid with adaptive sharpness based on target range
            penalty_factor = torch.sigmoid(torch.tensor(self.smoothness * relative_excess))
        elif self.curve == "softplus":
            # Softplus with adaptive sharpness based on target range
            penalty_factor = F.softplus(torch.tensor(relative_excess * self.smoothness))
        else:
            # Fallback (should not reach here due to validation)
            penalty_factor = F.softplus(torch.tensor(relative_excess * self.smoothness))
        
        # Step 3: Apply adaptive weight and return penalty
        penalty = -self.adaptive_weight * penalty_factor.item()
        
        # Track for analysis
        self.penalty_history.append(penalty)
        self.normalized_turnover_history.append(normalized_turnover)
        
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
        if self.episode_length == 0 or current_portfolio_value == 0:
            return 0.0
        
        normalized_turnover = total_turnover / (current_portfolio_value * self.episode_length)
        excess_turnover = normalized_turnover - self.target_range
        
        # Adaptive scaling: divide by target_range for consistent gradient behavior
        if self.target_range > 0:
            relative_excess = excess_turnover / self.target_range
            scaling_factor = 1.0 / self.target_range  # For gradient chain rule
        else:
            relative_excess = excess_turnover
            scaling_factor = 1.0
        
        if self.curve == "sigmoid":
            # Gradient of sigmoid with adaptive scaling
            sigmoid_val = torch.sigmoid(torch.tensor(self.smoothness * relative_excess))
            gradient = self.smoothness * sigmoid_val * (1 - sigmoid_val) * scaling_factor
        elif self.curve == "softplus":
            # Gradient of softplus with adaptive scaling
            gradient = torch.sigmoid(torch.tensor(self.smoothness * relative_excess)) * self.smoothness * scaling_factor
        else:
            gradient = torch.sigmoid(torch.tensor(self.smoothness * relative_excess)) * self.smoothness * scaling_factor
        
        # Scale by adaptive weight and normalization factor
        current_portfolio_value = self._get_current_portfolio_value()
        scaled_gradient = -self.adaptive_weight * gradient.item() / (current_portfolio_value * self.episode_length)
        
        return scaled_gradient
    

    
    def get_penalty_curve_preview(self, turnover_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate a preview of the penalty curve for visualization.
        
        Args:
            turnover_range (np.ndarray): Range of turnover values to evaluate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with turnover values and corresponding penalties
        """
        penalties = []
        normalized_turnovers = []
        
        for turnover in turnover_range:
            penalty = self.compute_penalty(turnover)
            normalized = turnover / (self.portfolio_value * self.episode_length)
            
            penalties.append(penalty)
            normalized_turnovers.append(normalized)
        
        return {
            'turnover_values': turnover_range,
            'normalized_turnover': np.array(normalized_turnovers),
            'penalties': np.array(penalties),
            'target_range': self.target_range,
            'curve_type': self.curve
        }
    
    def log_debug(self, total_turnover: float, step: Optional[int] = None) -> None:
        """
        Log detailed debug information about the penalty calculation.
        
        Args:
            total_turnover (float): Total traded value
            step (Optional[int]): Current step number for context
        """
        current_portfolio_value = self._get_current_portfolio_value()
        if self.episode_length == 0 or current_portfolio_value == 0:
            self.logger.warning(
                f"🎯 [TurnoverPenalty] INVALID STATE: "
                f"episode_length={self.episode_length}, portfolio_value={current_portfolio_value}"
            )
            return
        
        normalized_turnover = total_turnover / (current_portfolio_value * self.episode_length)
        penalty = self.compute_penalty(total_turnover)
        gradient = self.compute_penalty_gradient(total_turnover)
        
        step_info = f"Step {step}, " if step is not None else ""
        
        # Raw input logging for debugging
        print(f"[TurnoverPenalty] Raw: turnover={total_turnover:.2f}, portfolio={current_portfolio_value:.2f}, steps={self.episode_length}")
        
        # Enhanced debug logging with all calculation details
        self.logger.info(
            f"🎯 [TurnoverPenalty] {step_info}"
            f"Normalized: {normalized_turnover:.5f} | "
            f"Target: {self.target_range:.3f} | "
            f"Penalty: {penalty:.2f} | "
            f"Total: ${total_turnover:.0f} | "
            f"Portfolio: ${current_portfolio_value:.2f} | "
            f"EpisodeLen: {self.episode_length} | "
            f"Denominator: {current_portfolio_value * self.episode_length:.0f}"
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
            'avg_normalized_turnover': np.mean(self.normalized_turnover_history),
            'penalty_std': np.std(self.penalty_history),
            'turnover_std': np.std(self.normalized_turnover_history),
            'min_penalty': np.min(self.penalty_history),
            'max_penalty': np.max(self.penalty_history),
            'target_range': self.target_range,
            'curve_type': self.curve,
            'adaptive_weight': self.adaptive_weight
        }
    
    def reset_history(self) -> None:
        """Reset penalty and turnover history."""
        self.penalty_history.clear()
        self.normalized_turnover_history.clear()
        self.logger.debug("🎯 TurnoverPenalty history reset")
    
    def __repr__(self) -> str:
        """String representation of the calculator."""
        return (
            f"TurnoverPenaltyCalculator("
            f"target={self.target_range:.3f}, "
            f"weight={self.adaptive_weight:.2f}, "
            f"curve={self.curve}, "
            f"episode_length={self.episode_length})"
        )


class TurnoverPenaltyFactory:
    """
    🏭 Factory class for creating TurnoverPenaltyCalculator instances
    with common configurations.
    """
    
    @staticmethod
    def create_conservative(episode_length: int, portfolio_value: float, 
                          logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create a conservative penalty calculator (low turnover target)."""
        return TurnoverPenaltyCalculator(
            episode_length=episode_length,
            portfolio_value=portfolio_value,
            target_range=0.01,  # 1% target
            adaptive_weight_factor=0.002,  # Higher penalty
            smoothness=15.0,  # Steeper curve
            curve="sigmoid",
            logger=logger
        )
    
    @staticmethod
    def create_moderate(episode_length: int, portfolio_value: float,
                       logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create a moderate penalty calculator (balanced approach)."""
        return TurnoverPenaltyCalculator(
            episode_length=episode_length,
            portfolio_value=portfolio_value,
            target_range=0.02,  # 2% target
            adaptive_weight_factor=0.001,  # Standard penalty
            smoothness=10.0,  # Moderate curve
            curve="softplus",
            logger=logger
        )
    
    @staticmethod
    def create_aggressive(episode_length: int, portfolio_value: float,
                         logger: Optional[logging.Logger] = None) -> TurnoverPenaltyCalculator:
        """Create an aggressive penalty calculator (higher turnover allowed)."""
        return TurnoverPenaltyCalculator(
            episode_length=episode_length,
            portfolio_value=portfolio_value,
            target_range=0.05,  # 5% target
            adaptive_weight_factor=0.0005,  # Lower penalty
            smoothness=5.0,  # Gentler curve
            curve="softplus",
            logger=logger
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Example: Create a moderate penalty calculator
    calculator = TurnoverPenaltyFactory.create_moderate(
        episode_length=5000,
        portfolio_value=50000.0,
        logger=logger
    )
    
    # Test penalty calculation
    test_turnovers = [0, 1000, 5000, 10000, 25000, 50000]
    
    print("\n🎯 Turnover Penalty Calculator Test")
    print("=" * 50)
    
    for turnover in test_turnovers:
        penalty = calculator.compute_penalty(turnover)
        calculator.log_debug(turnover)
    
    # Show statistics
    stats = calculator.get_statistics()
    print(f"\n📊 Statistics: {stats}")
    
    # Preview penalty curve
    turnover_range = np.linspace(0, 100000, 100)
    curve_data = calculator.get_penalty_curve_preview(turnover_range)
    
    print(f"\n📈 Penalty curve generated with {len(curve_data['penalties'])} points")
    print(f"Target normalized turnover: {curve_data['target_range']:.3f}")
    print(f"Curve type: {curve_data['curve_type']}")