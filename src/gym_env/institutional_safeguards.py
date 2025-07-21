"""
Institutional Safeguards for Trading Environment
Implements enterprise-grade environment validation and error handling
"""

import numpy as np
import logging
import datetime
from typing import Dict, Any, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class InstitutionalSafeguards:
    """Institutional-grade environment safeguards and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward bounds from config
        reward_bounds = config.get('validation', {}).get('reward_bounds', {})
        self.reward_bounds = (
            reward_bounds.get('min_reward', -5000),
            reward_bounds.get('max_reward', 10000)
        )
        self.reward_alert_threshold = reward_bounds.get('alert_threshold', 0.95)
        
        # Position limits
        self.max_position_size = config.get('environment', {}).get('max_position_size_pct', 0.95)
        self.min_cash_reserve = config.get('environment', {}).get('min_cash_reserve_pct', 0.05)
        
        # Reward scaling
        self.reward_scaling = config.get('environment', {}).get('reward_scaling', 1.0)
        
        # Monitoring
        self.violation_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.safeguard_stats = {
            'total_steps': 0,
            'reward_violations': 0,
            'position_violations': 0,
            'nan_incidents': 0,
            'total_rewards_scaled': 0
        }
        
        logger.info(f"InstitutionalSafeguards initialized:")
        logger.info(f"  - Reward bounds: {self.reward_bounds}")
        logger.info(f"  - Position limit: {self.max_position_size:.1%}")
        logger.info(f"  - Cash reserve: {self.min_cash_reserve:.1%}")
        logger.info(f"  - Reward scaling: {self.reward_scaling}")
        
    def validate_step_output(self, observation: np.ndarray, reward: float, 
                           done: bool, info: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Comprehensive step output validation with institutional safeguards."""
        
        self.safeguard_stats['total_steps'] += 1
        original_reward = reward
        
        # 1. Reward validation with institutional bounds
        if not np.isfinite(reward):
            logger.critical(f"🚨 Non-finite reward detected: {reward}")
            self.safeguard_stats['nan_incidents'] += 1
            reward = 0.0
            info['safeguard_violation'] = 'non_finite_reward'
            
        # 2. Reward bounds checking
        if not (self.reward_bounds[0] <= reward <= self.reward_bounds[1]):
            violation_severity = 'extreme' if (
                reward < self.reward_bounds[0] * 2 or 
                reward > self.reward_bounds[1] * 2
            ) else 'warning'
            
            if violation_severity == 'extreme':
                logger.error(f"🚨 EXTREME reward outside bounds: {reward} (bounds: {self.reward_bounds})")
                self.safeguard_stats['reward_violations'] += 1
                # Cap extreme rewards
                reward = np.clip(reward, self.reward_bounds[0] * 2, self.reward_bounds[1] * 2)
            else:
                logger.warning(f"⚠️ Reward outside normal bounds: {reward}")
                
            self.violation_history.append({
                'type': 'reward_bounds',
                'severity': violation_severity,
                'original_value': original_reward,
                'capped_value': reward,
                'timestamp': datetime.datetime.utcnow()
            })
            
        # 3. Observation validation
        obs_validation = self._validate_observation(observation)
        if not obs_validation['valid']:
            logger.error(f"❌ Observation validation failed: {obs_validation['error']}")
            # Try to fix common issues
            observation = self._fix_observation_issues(observation, obs_validation)
            
        # 4. Position size validation
        if 'portfolio_state' in info:
            position_validation = self._validate_position_sizes(info['portfolio_state'])
            if not position_validation['valid']:
                logger.warning(f"⚠️ Position size warning: {position_validation['warning']}")
                self.safeguard_stats['position_violations'] += 1
                
        # 5. Apply reward scaling with bounds checking
        if self.reward_scaling != 1.0:
            scaled_reward = self.apply_reward_scaling(reward, self.reward_scaling)
            if scaled_reward != reward:
                self.safeguard_stats['total_rewards_scaled'] += 1
            reward = scaled_reward
            
        # 6. Final validation
        assert np.isfinite(reward), f"Final reward validation failed: {reward}"
        assert observation.shape[-1] > 0, f"Empty observation detected"
        
        # Store for monitoring
        self.reward_history.append(reward)
        
        # Add safeguard info
        info.update({
            'safeguards_applied': True,
            'original_reward': original_reward,
            'final_reward': reward,
            'reward_scaling_applied': self.reward_scaling != 1.0,
            'validation_passed': True
        })
        
        return observation, reward, done, info
        
    def apply_reward_scaling(self, raw_reward: float, scaling_factor: float) -> float:
        """Apply reward scaling with bounds checking and monitoring."""
        
        scaled_reward = raw_reward * scaling_factor
        
        # Institutional bounds enforcement (post-scaling)
        if abs(scaled_reward) > 50000:  # Extreme sanity check
            logger.warning(f"⚠️ Unusually large scaled reward: {scaled_reward:.0f} (from {raw_reward:.0f})")
            
        # Alert on approach to bounds
        bounds_buffer = (self.reward_bounds[1] - self.reward_bounds[0]) * self.reward_alert_threshold
        if abs(scaled_reward) > bounds_buffer:
            logger.info(f"📊 Reward approaching bounds: {scaled_reward:.0f} (bounds: {self.reward_bounds})")
            
        return scaled_reward
        
    def _validate_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """Validate observation array for common issues."""
        
        # Shape validation
        if len(observation.shape) == 0:
            return {'valid': False, 'error': 'Scalar observation (expected array)'}
            
        if len(observation.shape) > 2:
            return {'valid': False, 'error': f'High-dimensional observation: {observation.shape}'}
            
        # NaN/Inf checking
        if np.any(np.isnan(observation)):
            nan_count = np.sum(np.isnan(observation))
            return {'valid': False, 'error': f'{nan_count} NaN values in observation'}
            
        if np.any(np.isinf(observation)):
            inf_count = np.sum(np.isinf(observation))
            return {'valid': False, 'error': f'{inf_count} Inf values in observation'}
            
        # Range validation (basic sanity checks)
        if np.any(np.abs(observation) > 1e6):
            extreme_count = np.sum(np.abs(observation) > 1e6)
            return {'valid': False, 'error': f'{extreme_count} extreme values (>1e6) in observation'}
            
        return {'valid': True, 'error': None}
        
    def _fix_observation_issues(self, observation: np.ndarray, validation_result: Dict[str, Any]) -> np.ndarray:
        """Attempt to fix common observation issues."""
        
        error = validation_result['error']
        
        if 'NaN values' in error:
            # Replace NaN with zeros
            observation = np.nan_to_num(observation, nan=0.0)
            logger.info("🔧 Fixed NaN values by replacing with zeros")
            
        elif 'Inf values' in error:
            # Replace Inf with large finite values
            observation = np.nan_to_num(observation, posinf=1e6, neginf=-1e6)
            logger.info("🔧 Fixed Inf values by capping to ±1e6")
            
        elif 'extreme values' in error:
            # Clip extreme values
            observation = np.clip(observation, -1e6, 1e6)
            logger.info("🔧 Clipped extreme values to ±1e6")
            
        return observation
        
    def _validate_position_sizes(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio position sizes against institutional limits."""
        
        if 'positions' not in portfolio_state:
            return {'valid': True, 'warning': None}
            
        positions = portfolio_state['positions']
        total_capital = portfolio_state.get('total_value', 1.0)
        
        # Check individual position sizes
        max_position_pct = 0.0
        max_position_symbol = None
        
        for symbol, position_value in positions.items():
            position_pct = abs(position_value) / total_capital
            if position_pct > max_position_pct:
                max_position_pct = position_pct
                max_position_symbol = symbol
                
        # Check against limits
        if max_position_pct > self.max_position_size:
            return {
                'valid': False,
                'warning': f'Position {max_position_symbol} at {max_position_pct:.1%} exceeds limit {self.max_position_size:.1%}'
            }
            
        # Check cash reserve
        cash_pct = portfolio_state.get('cash_pct', 0.0)
        if cash_pct < self.min_cash_reserve:
            return {
                'valid': False,
                'warning': f'Cash reserve {cash_pct:.1%} below minimum {self.min_cash_reserve:.1%}'
            }
            
        return {'valid': True, 'warning': None}
        
    def get_safeguard_stats(self) -> Dict[str, Any]:
        """Get comprehensive safeguard statistics."""
        
        recent_rewards = list(self.reward_history)[-100:] if self.reward_history else []
        
        stats = self.safeguard_stats.copy()
        stats.update({
            'recent_reward_stats': {
                'mean': np.mean(recent_rewards) if recent_rewards else 0,
                'std': np.std(recent_rewards) if recent_rewards else 0,
                'min': np.min(recent_rewards) if recent_rewards else 0,
                'max': np.max(recent_rewards) if recent_rewards else 0,
                'count': len(recent_rewards)
            },
            'violation_rates': {
                'reward_violation_rate': self.safeguard_stats['reward_violations'] / max(1, self.safeguard_stats['total_steps']),
                'position_violation_rate': self.safeguard_stats['position_violations'] / max(1, self.safeguard_stats['total_steps']),
                'nan_incident_rate': self.safeguard_stats['nan_incidents'] / max(1, self.safeguard_stats['total_steps'])
            },
            'recent_violations': list(self.violation_history)[-10:],
            'institutional_compliance': {
                'reward_stability': self.safeguard_stats['reward_violations'] < self.safeguard_stats['total_steps'] * 0.01,  # <1% violation rate
                'numerical_stability': self.safeguard_stats['nan_incidents'] == 0,  # Zero tolerance
                'position_compliance': self.safeguard_stats['position_violations'] < self.safeguard_stats['total_steps'] * 0.05  # <5% violation rate
            }
        })
        
        return stats
        
    def reset_stats(self):
        """Reset safeguard statistics (for new episodes)."""
        
        self.safeguard_stats = {
            'total_steps': 0,
            'reward_violations': 0,
            'position_violations': 0,
            'nan_incidents': 0,
            'total_rewards_scaled': 0
        }
        logger.info("🔄 Safeguard statistics reset for new episode")


def create_safeguards_from_config(config_path: str) -> InstitutionalSafeguards:
    """Factory function to create safeguards from config file."""
    
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return InstitutionalSafeguards(config)