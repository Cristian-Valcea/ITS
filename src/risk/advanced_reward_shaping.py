"""
Advanced Reward & Risk Shaping Module

Implements cutting-edge reward shaping techniques for risk-aware reinforcement learning:
1. Lagrangian Constraint: Learnable multiplier λ for volatility-based drawdown punishment
2. Sharpe-Adjusted Reward: Normalize PnL by rolling volatility for Sharpe optimization  
3. CVaR-RL: Conditional Value at Risk policy gradient for tail risk control

Based on:
- Lagrangian methods for constrained RL (Achiam et al., 2017)
- Sharpe ratio optimization in RL (Moody & Saffell, 2001)
- CVaR-RL for tail risk control (Bae et al., 2022)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import logging
from dataclasses import dataclass


@dataclass
class AdvancedRewardConfig:
    """Configuration for advanced reward shaping."""
    # Lagrangian constraint
    lagrangian_enabled: bool = True
    initial_lambda: float = 0.1
    lambda_lr: float = 0.001
    target_volatility: float = 0.02
    vol_window: int = 60
    constraint_tolerance: float = 0.001
    lambda_min: float = 0.01
    lambda_max: float = 10.0
    update_frequency: int = 100
    
    # Sharpe-adjusted reward
    sharpe_enabled: bool = True
    rolling_window: int = 60
    min_periods: int = 30
    sharpe_scaling: float = 1.0
    volatility_floor: float = 0.001
    annualization_factor: int = 252
    
    # CVaR-RL
    cvar_enabled: bool = True
    confidence_level: float = 0.05
    cvar_window: int = 120
    cvar_weight: float = 0.3
    tail_penalty_factor: float = 2.0
    quantile_smoothing: float = 0.1
    min_samples_cvar: int = 50


class LagrangianConstraintManager:
    """
    Manages learnable Lagrangian multiplier for volatility-based constraints.
    
    Implements adaptive λ that increases when volatility exceeds target,
    decreases when volatility is below target, optimizing the trade-off
    between returns and risk automatically.
    """
    
    def __init__(self, config: AdvancedRewardConfig):
        self.config = config
        self.lambda_value = config.initial_lambda
        self.step_count = 0
        self.volatility_history = deque(maxlen=config.vol_window)
        self.constraint_violations = deque(maxlen=1000)  # Track violations
        
        self.logger = logging.getLogger(__name__)
        
    def update_lambda(self, current_volatility: float) -> float:
        """
        Update the Lagrangian multiplier based on constraint violation.
        
        Args:
            current_volatility: Current realized volatility
            
        Returns:
            Updated lambda value
        """
        self.step_count += 1
        
        if self.step_count % self.config.update_frequency == 0:
            # Calculate constraint violation
            violation = current_volatility - self.config.target_volatility
            self.constraint_violations.append(violation)
            
            # Update lambda using gradient ascent on dual problem
            if abs(violation) > self.config.constraint_tolerance:
                # Increase λ if volatility too high, decrease if too low
                lambda_update = self.config.lambda_lr * violation
                self.lambda_value += lambda_update
                
                # Clip lambda to valid range
                self.lambda_value = np.clip(
                    self.lambda_value, 
                    self.config.lambda_min, 
                    self.config.lambda_max
                )
                
                self.logger.debug(f"Updated λ: {self.lambda_value:.4f} (violation: {violation:.4f})")
        
        return self.lambda_value
    
    def get_constraint_penalty(self, current_volatility: float, drawdown: float) -> float:
        """
        Calculate constraint penalty for current state.
        
        Args:
            current_volatility: Current realized volatility
            drawdown: Current drawdown
            
        Returns:
            Constraint penalty to subtract from reward
        """
        if not self.config.lagrangian_enabled:
            return 0.0
            
        # Update lambda based on current volatility
        current_lambda = self.update_lambda(current_volatility)
        
        # Calculate penalty: λ * max(0, volatility - target) * drawdown_severity
        volatility_excess = max(0, current_volatility - self.config.target_volatility)
        drawdown_severity = max(0, drawdown)  # Only penalize positive drawdowns
        
        penalty = current_lambda * volatility_excess * drawdown_severity
        
        return penalty
    
    def get_stats(self) -> Dict[str, float]:
        """Get current Lagrangian constraint statistics."""
        return {
            'lambda_value': self.lambda_value,
            'avg_violation': np.mean(self.constraint_violations) if self.constraint_violations else 0.0,
            'violation_std': np.std(self.constraint_violations) if len(self.constraint_violations) > 1 else 0.0,
            'step_count': self.step_count
        }


class SharpeAdjustedRewardCalculator:
    """
    Calculates Sharpe-adjusted rewards by normalizing PnL by rolling volatility.
    
    This encourages the agent to optimize risk-adjusted returns rather than
    raw returns, leading to more stable and robust trading strategies.
    """
    
    def __init__(self, config: AdvancedRewardConfig):
        self.config = config
        self.returns_history = deque(maxlen=config.rolling_window)
        self.pnl_history = deque(maxlen=config.rolling_window)
        
    def calculate_sharpe_reward(self, pnl: float, base_reward: float) -> float:
        """
        Calculate Sharpe-adjusted reward.
        
        Args:
            pnl: Current period PnL
            base_reward: Original reward before Sharpe adjustment
            
        Returns:
            Sharpe-adjusted reward
        """
        if not self.config.sharpe_enabled:
            return base_reward
            
        self.pnl_history.append(pnl)
        
        if len(self.pnl_history) < self.config.min_periods:
            return base_reward
        
        # Calculate rolling statistics
        pnl_array = np.array(self.pnl_history)
        mean_return = np.mean(pnl_array)
        volatility = np.std(pnl_array)
        
        # Apply volatility floor to avoid division by zero
        volatility = max(volatility, self.config.volatility_floor)
        
        # Calculate Sharpe ratio proxy
        sharpe_proxy = mean_return / volatility
        
        # Annualize if needed (optional)
        if self.config.annualization_factor > 1:
            sharpe_proxy *= np.sqrt(self.config.annualization_factor)
        
        # Scale and combine with base reward
        sharpe_reward = self.config.sharpe_scaling * sharpe_proxy
        
        return sharpe_reward
    
    def get_current_sharpe(self) -> float:
        """Get current rolling Sharpe ratio."""
        if len(self.pnl_history) < self.config.min_periods:
            return 0.0
            
        pnl_array = np.array(self.pnl_history)
        mean_return = np.mean(pnl_array)
        volatility = max(np.std(pnl_array), self.config.volatility_floor)
        
        return mean_return / volatility


class CVaRRLCalculator:
    """
    Implements CVaR-RL (Conditional Value at Risk Reinforcement Learning).
    
    Based on Bae et al. (2022), this directly minimizes extreme losses by
    incorporating CVaR (Expected Shortfall) into the reward function,
    providing superior tail risk control.
    """
    
    def __init__(self, config: AdvancedRewardConfig):
        self.config = config
        self.returns_history = deque(maxlen=config.cvar_window)
        self.losses_history = deque(maxlen=config.cvar_window)
        
    def calculate_cvar_penalty(self, current_return: float) -> float:
        """
        Calculate CVaR-based penalty for tail risk control.
        
        Args:
            current_return: Current period return
            
        Returns:
            CVaR penalty to subtract from reward
        """
        if not self.config.cvar_enabled:
            return 0.0
            
        self.returns_history.append(current_return)
        
        # Only calculate CVaR if we have enough samples
        if len(self.returns_history) < self.config.min_samples_cvar:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        
        # Calculate Value at Risk (VaR) at confidence level
        var_quantile = np.quantile(returns_array, self.config.confidence_level)
        
        # Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR
        tail_losses = returns_array[returns_array <= var_quantile]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = np.mean(tail_losses)  # This will be negative for losses
        
        # Convert to penalty (positive value to subtract from reward)
        cvar_penalty = -cvar * self.config.cvar_weight
        
        # Add extra penalty for extreme losses
        if current_return <= var_quantile:
            extreme_penalty = abs(current_return - var_quantile) * self.config.tail_penalty_factor
            cvar_penalty += extreme_penalty
        
        return cvar_penalty
    
    def get_cvar_stats(self) -> Dict[str, float]:
        """Get current CVaR statistics."""
        if len(self.returns_history) < self.config.min_samples_cvar:
            return {'cvar': 0.0, 'var': 0.0, 'tail_probability': 0.0}
        
        returns_array = np.array(self.returns_history)
        var_quantile = np.quantile(returns_array, self.config.confidence_level)
        tail_losses = returns_array[returns_array <= var_quantile]
        
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
        tail_probability = len(tail_losses) / len(returns_array)
        
        return {
            'cvar': cvar,
            'var': var_quantile,
            'tail_probability': tail_probability,
            'samples': len(self.returns_history)
        }


class AdvancedRewardShaper:
    """
    Main class that orchestrates all advanced reward shaping techniques.
    
    Combines Lagrangian constraints, Sharpe-adjusted rewards, and CVaR-RL
    to create a sophisticated risk-aware reward signal for the RL agent.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        # Parse configuration
        advanced_config = config_dict.get('advanced_reward_shaping', {})
        
        self.config = AdvancedRewardConfig(
            # Lagrangian constraint
            lagrangian_enabled=advanced_config.get('lagrangian_constraint', {}).get('enabled', True),
            initial_lambda=advanced_config.get('lagrangian_constraint', {}).get('initial_lambda', 0.1),
            lambda_lr=advanced_config.get('lagrangian_constraint', {}).get('lambda_lr', 0.001),
            target_volatility=advanced_config.get('lagrangian_constraint', {}).get('target_volatility', 0.02),
            vol_window=advanced_config.get('lagrangian_constraint', {}).get('vol_window', 60),
            constraint_tolerance=advanced_config.get('lagrangian_constraint', {}).get('constraint_tolerance', 0.001),
            lambda_min=advanced_config.get('lagrangian_constraint', {}).get('lambda_min', 0.01),
            lambda_max=advanced_config.get('lagrangian_constraint', {}).get('lambda_max', 10.0),
            update_frequency=advanced_config.get('lagrangian_constraint', {}).get('update_frequency', 100),
            
            # Sharpe-adjusted reward
            sharpe_enabled=advanced_config.get('sharpe_adjusted_reward', {}).get('enabled', True),
            rolling_window=advanced_config.get('sharpe_adjusted_reward', {}).get('rolling_window', 60),
            min_periods=advanced_config.get('sharpe_adjusted_reward', {}).get('min_periods', 30),
            sharpe_scaling=advanced_config.get('sharpe_adjusted_reward', {}).get('sharpe_scaling', 1.0),
            volatility_floor=advanced_config.get('sharpe_adjusted_reward', {}).get('volatility_floor', 0.001),
            annualization_factor=advanced_config.get('sharpe_adjusted_reward', {}).get('annualization_factor', 252),
            
            # CVaR-RL
            cvar_enabled=advanced_config.get('cvar_rl', {}).get('enabled', True),
            confidence_level=advanced_config.get('cvar_rl', {}).get('confidence_level', 0.05),
            cvar_window=advanced_config.get('cvar_rl', {}).get('cvar_window', 120),
            cvar_weight=advanced_config.get('cvar_rl', {}).get('cvar_weight', 0.3),
            tail_penalty_factor=advanced_config.get('cvar_rl', {}).get('tail_penalty_factor', 2.0),
            quantile_smoothing=advanced_config.get('cvar_rl', {}).get('quantile_smoothing', 0.1),
            min_samples_cvar=advanced_config.get('cvar_rl', {}).get('min_samples_cvar', 50)
        )
        
        # Initialize components
        self.lagrangian_manager = LagrangianConstraintManager(self.config)
        self.sharpe_calculator = SharpeAdjustedRewardCalculator(self.config)
        self.cvar_calculator = CVaRRLCalculator(self.config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Advanced Reward Shaping initialized:")
        self.logger.info(f"   - Lagrangian Constraint: {'✅' if self.config.lagrangian_enabled else '❌'}")
        self.logger.info(f"   - Sharpe-Adjusted Reward: {'✅' if self.config.sharpe_enabled else '❌'}")
        self.logger.info(f"   - CVaR-RL: {'✅' if self.config.cvar_enabled else '❌'}")
        
    def shape_reward(self, 
                    base_reward: float,
                    pnl: float,
                    current_return: float,
                    volatility: float,
                    drawdown: float) -> Tuple[float, Dict[str, float]]:
        """
        Apply advanced reward shaping to base reward.
        
        Args:
            base_reward: Original reward from environment
            pnl: Current period PnL
            current_return: Current period return
            volatility: Current realized volatility
            drawdown: Current drawdown
            
        Returns:
            Tuple of (shaped_reward, shaping_info)
        """
        shaped_reward = base_reward
        shaping_info = {}
        
        # 1. Apply Lagrangian constraint penalty
        if self.config.lagrangian_enabled:
            lagrangian_penalty = self.lagrangian_manager.get_constraint_penalty(volatility, drawdown)
            shaped_reward -= lagrangian_penalty
            shaping_info['lagrangian_penalty'] = lagrangian_penalty
            shaping_info.update(self.lagrangian_manager.get_stats())
        
        # 2. Apply Sharpe-adjusted reward
        if self.config.sharpe_enabled:
            sharpe_reward = self.sharpe_calculator.calculate_sharpe_reward(pnl, shaped_reward)
            shaped_reward = sharpe_reward  # Replace with Sharpe-adjusted version
            shaping_info['sharpe_reward'] = sharpe_reward
            shaping_info['current_sharpe'] = self.sharpe_calculator.get_current_sharpe()
        
        # 3. Apply CVaR penalty
        if self.config.cvar_enabled:
            cvar_penalty = self.cvar_calculator.calculate_cvar_penalty(current_return)
            shaped_reward -= cvar_penalty
            shaping_info['cvar_penalty'] = cvar_penalty
            shaping_info.update(self.cvar_calculator.get_cvar_stats())
        
        # Calculate total shaping effect
        shaping_info['base_reward'] = base_reward
        shaping_info['shaped_reward'] = shaped_reward
        shaping_info['total_shaping'] = shaped_reward - base_reward
        
        return shaped_reward, shaping_info
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all shaping components."""
        stats = {
            'config': {
                'lagrangian_enabled': self.config.lagrangian_enabled,
                'sharpe_enabled': self.config.sharpe_enabled,
                'cvar_enabled': self.config.cvar_enabled
            }
        }
        
        if self.config.lagrangian_enabled:
            stats['lagrangian'] = self.lagrangian_manager.get_stats()
            
        if self.config.sharpe_enabled:
            stats['sharpe'] = {'current_sharpe': self.sharpe_calculator.get_current_sharpe()}
            
        if self.config.cvar_enabled:
            stats['cvar'] = self.cvar_calculator.get_cvar_stats()
            
        return stats
    
    def reset(self):
        """Reset all components for new episode."""
        # Note: We don't reset the Lagrangian lambda as it should persist across episodes
        # Only reset episode-specific histories if needed
        pass