"""
Transaction Cost Environment Wrapper for Phase 2A
Integrates transaction costs into the trading environment
"""

import gym
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from src.execution.basic_transaction_cost_engine import BasicTransactionCostEngine
from src.gym_env.institutional_safeguards import InstitutionalSafeguards

logger = logging.getLogger(__name__)


class TransactionCostWrapper(gym.Wrapper):
    """Environment wrapper that adds realistic transaction costs to trading environment."""
    
    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env)
        
        self.config = config
        self.transaction_costs_enabled = config.get('environment', {}).get('enable_transaction_costs', True)
        
        # Initialize cost engine
        if self.transaction_costs_enabled:
            self.cost_engine = BasicTransactionCostEngine(config)
        else:
            self.cost_engine = None
            
        # Initialize safeguards (from Phase 1)
        self.safeguards = InstitutionalSafeguards(config)
        
        # Enhanced observation space for transaction cost features
        original_obs_space = env.observation_space
        if hasattr(original_obs_space, 'shape'):
            original_features = original_obs_space.shape[-1]
            # Add 1 feature for transaction cost ratio
            new_features = original_features + 1
            
            # Update observation space
            if len(original_obs_space.shape) == 1:
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(new_features,), 
                    dtype=np.float32
                )
            else:
                new_shape = original_obs_space.shape[:-1] + (new_features,)
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=new_shape,
                    dtype=np.float32
                )
        else:
            # Fallback if observation space is unusual
            self.observation_space = original_obs_space
            
        # Cost tracking
        self.last_trade_cost = 0.0
        self.cumulative_costs = 0.0
        self.trade_count = 0
        self.episode_costs = []
        
        logger.info(f"TransactionCostWrapper initialized:")
        logger.info(f"  - Transaction costs enabled: {self.transaction_costs_enabled}")
        logger.info(f"  - Original obs features: {original_features if 'original_features' in locals() else 'unknown'}")
        logger.info(f"  - New obs features: {new_features if 'new_features' in locals() else 'unknown'}")
        
    def reset(self, **kwargs):
        """Reset environment and cost tracking."""
        
        # Reset underlying environment
        obs = self.env.reset(**kwargs)
        
        # Handle tuple return from newer gym versions
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
            
        # Reset cost tracking
        self.last_trade_cost = 0.0
        self.cumulative_costs = 0.0
        self.trade_count = 0
        
        # Store episode costs for analysis
        if hasattr(self, 'episode_costs') and self.episode_costs:
            episode_cost_summary = {
                'total_costs': sum(self.episode_costs),
                'avg_cost': np.mean(self.episode_costs),
                'max_cost': max(self.episode_costs),
                'num_trades': len(self.episode_costs)
            }
            logger.debug(f"Episode cost summary: {episode_cost_summary}")
            
        self.episode_costs = []
        
        # Enhance observation with cost features
        enhanced_obs = self._enhance_observation(obs, 0.0)  # Zero cost at reset
        
        # Apply institutional safeguards
        enhanced_obs, _, _, enhanced_info = self.safeguards.validate_step_output(
            enhanced_obs, 0.0, False, info
        )
        
        return enhanced_obs if not isinstance(self.env.reset(**kwargs), tuple) else (enhanced_obs, enhanced_info)
        
    def step(self, action):
        """Execute step with transaction cost calculation."""
        
        # Execute action in underlying environment
        step_result = self.env.step(action)
        
        # Handle different step result formats
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
            
        # Calculate transaction costs if enabled
        transaction_cost = 0.0
        cost_breakdown = {}
        
        if self.transaction_costs_enabled and self.cost_engine:
            # Extract trade information from info or estimate from action
            trade_value = self._extract_trade_value(action, info)
            portfolio_value = self._extract_portfolio_value(info)
            market_data = self._extract_market_data(info)
            
            if abs(trade_value) > 1e-6:  # Only calculate costs for actual trades
                cost_breakdown = self.cost_engine.compute_transaction_cost(
                    trade_value, portfolio_value, market_data
                )
                transaction_cost = cost_breakdown['total_cost']
                
                # Update tracking
                self.trade_count += 1
                self.cumulative_costs += transaction_cost
                self.last_trade_cost = transaction_cost
                self.episode_costs.append(transaction_cost)
                
                # Log transaction cost details
                if self.config.get('logging', {}).get('transaction_cost_logs', False):
                    logger.debug(f"Trade cost: ${transaction_cost:.2f} "
                               f"({cost_breakdown.get('cost_as_pct_trade', 0):.3%} of ${abs(trade_value):.0f})")
                    
        # Apply transaction cost to reward
        cost_adjusted_reward = reward - transaction_cost
        
        # Enhance observation with cost features
        cost_ratio = transaction_cost / max(abs(reward), 1.0) if reward != 0 else 0.0
        enhanced_obs = self._enhance_observation(obs, cost_ratio)
        
        # Update info with cost information
        enhanced_info = info.copy()
        enhanced_info.update({
            'transaction_cost': transaction_cost,
            'cost_breakdown': cost_breakdown,
            'cost_adjusted_reward': cost_adjusted_reward,
            'original_reward': reward,
            'cumulative_costs': self.cumulative_costs,
            'trade_count': self.trade_count,
            'cost_ratio': cost_ratio
        })
        
        # Apply institutional safeguards (from Phase 1)
        enhanced_obs, final_reward, done, enhanced_info = self.safeguards.validate_step_output(
            enhanced_obs, cost_adjusted_reward, done, enhanced_info
        )
        
        # Return in appropriate format
        if len(step_result) == 4:
            return enhanced_obs, final_reward, done, enhanced_info
        else:
            return enhanced_obs, final_reward, terminated, truncated, enhanced_info
            
    def _enhance_observation(self, obs: np.ndarray, cost_ratio: float) -> np.ndarray:
        """Add transaction cost features to observation."""
        
        try:
            # Ensure observation is numpy array
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
                
            # Add cost ratio as new feature
            if len(obs.shape) == 1:
                # 1D observation
                enhanced_obs = np.append(obs, cost_ratio)
            elif len(obs.shape) == 2:
                # 2D observation (batch dimension)
                cost_feature = np.full((obs.shape[0], 1), cost_ratio)
                enhanced_obs = np.concatenate([obs, cost_feature], axis=1)
            else:
                # Higher dimensions - add to last axis
                cost_shape = obs.shape[:-1] + (1,)
                cost_feature = np.full(cost_shape, cost_ratio)
                enhanced_obs = np.concatenate([obs, cost_feature], axis=-1)
                
            return enhanced_obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error enhancing observation: {e}")
            # Return original observation if enhancement fails
            return obs
            
    def _extract_trade_value(self, action, info: Dict[str, Any]) -> float:
        """Extract trade value from action or info."""
        
        # Try to get from info first
        if 'trade_value' in info:
            return float(info['trade_value'])
            
        # Try to estimate from action and portfolio info
        if 'portfolio_value' in info and hasattr(action, '__len__'):
            portfolio_value = info['portfolio_value']
            # Assume action represents position change as fraction of portfolio
            if len(action) > 0:
                trade_fraction = float(action[0]) if hasattr(action[0], '__float__') else 0.0
                return abs(trade_fraction * portfolio_value)
                
        # Default estimation based on action magnitude
        if hasattr(action, '__len__'):
            action_magnitude = np.sum(np.abs(action))
        else:
            action_magnitude = abs(float(action))
            
        # Estimate trade value (this is a fallback)
        estimated_trade_value = action_magnitude * 1000  # Rough estimate
        
        return estimated_trade_value
        
    def _extract_portfolio_value(self, info: Dict[str, Any]) -> float:
        """Extract portfolio value from info."""
        
        # Try various keys that might contain portfolio value
        portfolio_keys = ['portfolio_value', 'total_value', 'account_value', 'equity']
        
        for key in portfolio_keys:
            if key in info:
                return float(info[key])
                
        # Fallback to initial capital if available
        initial_capital = self.config.get('environment', {}).get('initial_capital', 50000.0)
        return initial_capital
        
    def _extract_market_data(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market data from info for ADV calculations."""
        
        market_data = {}
        
        # Extract volume if available
        if 'volume' in info:
            market_data['volume'] = info['volume']
        elif 'market_data' in info and 'volume' in info['market_data']:
            market_data['volume'] = info['market_data']['volume']
            
        # Extract other market data
        market_keys = ['price', 'bid', 'ask', 'spread', 'volatility']
        for key in market_keys:
            if key in info:
                market_data[key] = info[key]
            elif 'market_data' in info and key in info['market_data']:
                market_data[key] = info['market_data'][key]
                
        return market_data
        
    def get_cost_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cost statistics."""
        
        wrapper_stats = {
            'wrapper_stats': {
                'cumulative_costs': self.cumulative_costs,
                'trade_count': self.trade_count,
                'last_trade_cost': self.last_trade_cost,
                'avg_cost_per_trade': self.cumulative_costs / max(1, self.trade_count),
                'episode_cost_count': len(self.episode_costs)
            }
        }
        
        # Get engine stats if available
        if self.cost_engine:
            engine_stats = self.cost_engine.get_cost_statistics()
            wrapper_stats.update(engine_stats)
            
        return wrapper_stats