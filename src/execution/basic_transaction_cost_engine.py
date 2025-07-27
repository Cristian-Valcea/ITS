"""
Basic Transaction Cost Engine for Phase 2A
Implements realistic but simple transaction costs before advanced microstructure
"""

import numpy as np
import logging
import datetime
from typing import Dict, Any, Optional, Tuple
from collections import deque
import json

logger = logging.getLogger(__name__)


class BasicTransactionCostEngine:
    """Simple but realistic transaction costs (Phase 2A implementation)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('transaction_costs', {})
        
        # Fixed costs
        self.commission_per_trade = self.config.get('fixed_costs', {}).get('commission_per_trade', 0.50)
        
        # Proportional costs
        proportional = self.config.get('proportional_costs', {})
        self.spread_cost_bps = proportional.get('spread_cost_bps', 1.0)
        self.exchange_fees_bps = proportional.get('exchange_fees_bps', 0.1)
        
        # Capacity penalty configuration
        capacity_config = self.config.get('capacity_penalty', {})
        self.capacity_enabled = capacity_config.get('enabled', True)
        self.daily_capacity_pct = capacity_config.get('daily_capacity_pct', 0.02)
        self.penalty_weight = capacity_config.get('penalty_weight', 0.0001)
        self.max_penalty_pct = capacity_config.get('max_penalty_pct_trade', 0.3)
        
        # ADV scaling configuration
        adv_config = self.config.get('adv_scaling', {})
        self.adv_enabled = adv_config.get('enabled', True)
        self.adv_scaling_factor = adv_config.get('adv_scaling_factor', 1.5)
        self.adv_lookback_days = adv_config.get('adv_lookback_days', 20)
        self.min_adv_threshold = adv_config.get('min_adv_threshold', 100000)
        
        # Metrics streaming
        metrics_config = self.config.get('metrics_streaming', {})
        self.metrics_enabled = metrics_config.get('enabled', True)
        self.stream_frequency = metrics_config.get('stream_frequency', 'every_trade')
        
        # Historical ADV tracking
        self.adv_history = deque(maxlen=self.adv_lookback_days)
        self.current_adv = 1000000  # Default ADV
        
        # Cost tracking for analytics
        self.cost_history = deque(maxlen=10000)
        self.cost_stats = {
            'total_trades': 0,
            'total_fixed_costs': 0.0,
            'total_proportional_costs': 0.0,
            'total_capacity_costs': 0.0,
            'total_all_costs': 0.0,
            'avg_cost_per_trade': 0.0,
            'max_single_trade_cost': 0.0
        }
        
        logger.info(f"BasicTransactionCostEngine initialized:")
        logger.info(f"  - Commission per trade: ${self.commission_per_trade:.2f}")
        logger.info(f"  - Spread cost: {self.spread_cost_bps:.1f} bps")
        logger.info(f"  - Exchange fees: {self.exchange_fees_bps:.1f} bps")
        logger.info(f"  - Daily capacity: {self.daily_capacity_pct:.1%}")
        logger.info(f"  - ADV scaling: {self.adv_enabled} (factor: {self.adv_scaling_factor})")
        logger.info(f"  - Metrics streaming: {self.metrics_enabled}")
        
    def compute_transaction_cost(self, trade_value: float, portfolio_value: float, 
                               market_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute comprehensive transaction costs with detailed breakdown."""
        
        if market_data is None:
            market_data = {}
            
        # Initialize cost breakdown
        cost_breakdown = {
            'fixed_cost': 0.0,
            'spread_cost': 0.0,
            'exchange_fees': 0.0,
            'capacity_cost': 0.0,
            'total_cost': 0.0,
            'cost_as_pct_trade': 0.0,
            'trade_value': trade_value,
            'timestamp': datetime.datetime.utcnow()
        }
        
        # Skip calculation for zero trades
        if abs(trade_value) < 1e-6:
            return cost_breakdown
            
        try:
            # 1. Fixed costs
            cost_breakdown['fixed_cost'] = self.commission_per_trade
            
            # 2. Proportional costs (bid-ask spread + fees)
            abs_trade_value = abs(trade_value)
            cost_breakdown['spread_cost'] = abs_trade_value * self.spread_cost_bps / 10000
            cost_breakdown['exchange_fees'] = abs_trade_value * self.exchange_fees_bps / 10000
            
            # 3. Basic capacity penalty with ADV scaling
            if self.capacity_enabled:
                cost_breakdown['capacity_cost'] = self._compute_basic_capacity_penalty(
                    trade_value, portfolio_value, market_data
                )
                
            # 4. Total cost calculation
            total_cost = (
                cost_breakdown['fixed_cost'] + 
                cost_breakdown['spread_cost'] + 
                cost_breakdown['exchange_fees'] + 
                cost_breakdown['capacity_cost']
            )
            cost_breakdown['total_cost'] = total_cost
            
            # 5. Cost as percentage of trade
            if abs_trade_value > 0:
                cost_breakdown['cost_as_pct_trade'] = total_cost / abs_trade_value
            
            # 6. Update statistics
            self._update_cost_statistics(cost_breakdown)
            
            # 7. Stream metrics if enabled
            if self.metrics_enabled:
                self._stream_cost_metrics(cost_breakdown)
                
            # 8. Validation and logging
            self._validate_cost_calculation(cost_breakdown, trade_value, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error computing transaction cost: {e}")
            # Return safe default
            cost_breakdown['total_cost'] = abs(trade_value) * 0.001  # 10 bps default
            cost_breakdown['cost_as_pct_trade'] = 0.001
            
        return cost_breakdown
        
    def _compute_basic_capacity_penalty(self, trade_value: float, portfolio_value: float, 
                                      market_data: Dict[str, Any]) -> float:
        """Basic capacity penalty with ADV scaling (addressing team feedback)."""
        
        # Base daily capacity
        base_capacity = self.daily_capacity_pct
        
        # ADV scaling (addressing team feedback)
        if self.adv_enabled:
            adv_ratio = self._get_adv_ratio(market_data)
            adjusted_capacity = base_capacity * (1 + adv_ratio * self.adv_scaling_factor)
            
            # Log ADV scaling application
            if self.config.get('logging', {}).get('adv_scaling_logs', False):
                logger.debug(f"ADV scaling: base_capacity={base_capacity:.4f}, "
                           f"adv_ratio={adv_ratio:.2f}, adjusted_capacity={adjusted_capacity:.4f}")
        else:
            adjusted_capacity = base_capacity
            
        # Compute excess over capacity
        daily_turnover = abs(trade_value) / portfolio_value
        excess = max(0, daily_turnover - adjusted_capacity)
        
        if excess <= 0:
            return 0.0
            
        # Simple quadratic penalty
        penalty = self.penalty_weight * (excess ** 2) * portfolio_value
        
        # Apply absolute cap to prevent explosion on large trades (addressing production feedback)
        max_penalty = abs(trade_value) * self.max_penalty_pct
        penalty = min(penalty, max_penalty)
        
        # Log capacity penalty application
        if self.config.get('logging', {}).get('capacity_penalty_logs', False):
            logger.debug(f"Capacity penalty: excess={excess:.4f}, "
                        f"raw_penalty=${penalty:.2f}, capped=${min(penalty, max_penalty):.2f}")
            
        return penalty
        
    def _get_adv_ratio(self, market_data: Dict[str, Any]) -> float:
        """Calculate ADV ratio for capacity scaling."""
        
        # Get current volume from market data
        current_volume = market_data.get('volume', self.current_adv)
        
        # Update ADV history
        if current_volume > 0:
            self.adv_history.append(current_volume)
            
        # Calculate current ADV
        if len(self.adv_history) > 0:
            self.current_adv = np.mean(list(self.adv_history))
        
        # Calculate ratio (current volume / ADV)
        if self.current_adv > self.min_adv_threshold:
            adv_ratio = current_volume / self.current_adv
            # Cap ratio to reasonable bounds
            adv_ratio = np.clip(adv_ratio, 0.1, 5.0)
        else:
            adv_ratio = 1.0  # Default when insufficient data
            
        return adv_ratio
        
    def _update_cost_statistics(self, cost_breakdown: Dict[str, float]):
        """Update running cost statistics for monitoring."""
        
        self.cost_stats['total_trades'] += 1
        self.cost_stats['total_fixed_costs'] += cost_breakdown['fixed_cost']
        self.cost_stats['total_proportional_costs'] += (
            cost_breakdown['spread_cost'] + cost_breakdown['exchange_fees']
        )
        self.cost_stats['total_capacity_costs'] += cost_breakdown['capacity_cost']
        self.cost_stats['total_all_costs'] += cost_breakdown['total_cost']
        
        # Update averages
        total_trades = self.cost_stats['total_trades']
        self.cost_stats['avg_cost_per_trade'] = self.cost_stats['total_all_costs'] / total_trades
        
        # Track maximum
        if cost_breakdown['total_cost'] > self.cost_stats['max_single_trade_cost']:
            self.cost_stats['max_single_trade_cost'] = cost_breakdown['total_cost']
            
        # Store in history
        self.cost_history.append(cost_breakdown.copy())
        
    def _stream_cost_metrics(self, cost_breakdown: Dict[str, float]):
        """Stream cost metrics to database (addressing production feedback)."""
        
        if not self.metrics_enabled:
            return
            
        try:
            # Prepare metrics for streaming
            metrics_payload = {
                'timestamp': cost_breakdown['timestamp'].isoformat(),
                'fixed_cost': cost_breakdown['fixed_cost'],
                'spread_cost': cost_breakdown['spread_cost'], 
                'exchange_fees': cost_breakdown['exchange_fees'],
                'capacity_cost': cost_breakdown['capacity_cost'],
                'total_cost': cost_breakdown['total_cost'],
                'cost_as_pct_trade': cost_breakdown['cost_as_pct_trade'],
                'trade_value': cost_breakdown['trade_value'],
                'cost_breakdown_provided': True
            }
            
            # Log metrics (in production, this would stream to TimescaleDB)
            if self.config.get('logging', {}).get('transaction_cost_logs', False):
                logger.info(f"Cost metrics: {json.dumps(metrics_payload, indent=2)}")
                
            # TODO: In production, implement actual database streaming
            # self._write_to_metrics_db(metrics_payload)
            
        except Exception as e:
            logger.error(f"Error streaming cost metrics: {e}")
            
    def _validate_cost_calculation(self, cost_breakdown: Dict[str, float], 
                                 trade_value: float, portfolio_value: float):
        """Validate cost calculation for sanity and bounds."""
        
        total_cost = cost_breakdown['total_cost']
        cost_pct = cost_breakdown['cost_as_pct_trade']
        
        # Sanity checks
        if not np.isfinite(total_cost):
            logger.error(f"Non-finite total cost: {total_cost}")
            
        if total_cost < 0:
            logger.error(f"Negative total cost: {total_cost}")
            
        # Bounds checking
        validation_config = self.config.get('validation', {}).get('cost_validation', {})
        max_cost_pct = validation_config.get('max_cost_per_trade_pct', 0.5)
        
        if cost_pct > max_cost_pct:
            logger.warning(f"High transaction cost: {cost_pct:.2%} of trade value "
                          f"(limit: {max_cost_pct:.2%})")
            
        # Daily cost accumulation check
        daily_cost_pct = self.cost_stats['total_all_costs'] / portfolio_value
        max_daily_cost_pct = validation_config.get('max_daily_cost_pct', 0.05)
        
        if daily_cost_pct > max_daily_cost_pct:
            logger.warning(f"High daily cost accumulation: {daily_cost_pct:.2%} of portfolio "
                          f"(limit: {max_daily_cost_pct:.2%})")
            
    def get_cost_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cost statistics."""
        
        recent_costs = list(self.cost_history)[-100:] if self.cost_history else []
        
        stats = self.cost_stats.copy()
        
        if recent_costs:
            recent_total_costs = [c['total_cost'] for c in recent_costs]
            recent_cost_pcts = [c['cost_as_pct_trade'] for c in recent_costs]
            
            stats.update({
                'recent_cost_stats': {
                    'mean_cost': np.mean(recent_total_costs),
                    'std_cost': np.std(recent_total_costs),
                    'min_cost': np.min(recent_total_costs),
                    'max_cost': np.max(recent_total_costs),
                    'mean_cost_pct': np.mean(recent_cost_pcts),
                    'count': len(recent_costs)
                },
                'cost_breakdown_pcts': {
                    'fixed_pct': self.cost_stats['total_fixed_costs'] / max(1, self.cost_stats['total_all_costs']),
                    'proportional_pct': self.cost_stats['total_proportional_costs'] / max(1, self.cost_stats['total_all_costs']),
                    'capacity_pct': self.cost_stats['total_capacity_costs'] / max(1, self.cost_stats['total_all_costs'])
                },
                'adv_info': {
                    'current_adv': self.current_adv,
                    'adv_history_length': len(self.adv_history),
                    'adv_enabled': self.adv_enabled
                }
            })
            
        return stats
        
    def reset_daily_stats(self):
        """Reset daily cost statistics (for new trading sessions)."""
        
        # Keep cumulative stats but reset daily accumulators
        daily_stats = {
            'daily_total_costs': self.cost_stats['total_all_costs'],
            'daily_trades': self.cost_stats['total_trades'],
            'session_date': datetime.datetime.utcnow().date()
        }
        
        # Reset main stats for new day
        self.cost_stats = {
            'total_trades': 0,
            'total_fixed_costs': 0.0,
            'total_proportional_costs': 0.0,
            'total_capacity_costs': 0.0,
            'total_all_costs': 0.0,
            'avg_cost_per_trade': 0.0,
            'max_single_trade_cost': 0.0
        }
        
        logger.info(f"🔄 Daily cost statistics reset: {daily_stats}")
        
        return daily_stats


def create_cost_engine_from_config(config_path: str) -> BasicTransactionCostEngine:
    """Factory function to create cost engine from config file."""
    
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return BasicTransactionCostEngine(config)