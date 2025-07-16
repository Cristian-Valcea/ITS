"""
Advanced Curriculum Learning Manager for Risk-Constrained RL

Implements sophisticated episode-based curriculum with performance gates:
- Progressive risk constraint tightening
- Performance-based stage advancement
- AND logic for gate criteria (drawdown AND Sharpe)
- Adaptive learning schedule based on agent competency

Based on recent risk-constrained RL research with "gate" curricula.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    episode_start: int
    episode_end: Optional[int]  # None for final stage
    drawdown_cap: float
    lambda_penalty: float
    
    # Performance gate criteria
    sharpe_gate_threshold: float = -0.5  # Must be > -0.5
    min_episodes_in_stage: int = 10  # Minimum episodes before gate check
    drawdown_gate_threshold: float = None  # Will be calculated in __post_init__
    
    def __post_init__(self):
        """Calculate derived parameters."""
        if self.drawdown_gate_threshold is None:
            self.drawdown_gate_threshold = self.drawdown_cap * 0.75


@dataclass
class EpisodePerformance:
    """Performance metrics for a single episode."""
    episode: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    num_trades: int
    final_portfolio_value: float
    stage_name: str


class CurriculumManager:
    """
    Manages progressive curriculum learning with performance gates.
    
    Implements episode-based progression through risk constraint stages,
    with performance gates requiring both drawdown control AND positive Sharpe.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize curriculum manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize curriculum stages
        self.stages = self._initialize_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Performance tracking
        self.episode_history: List[EpisodePerformance] = []
        self.stage_performance: Dict[str, List[EpisodePerformance]] = {
            stage.name: [] for stage in self.stages
        }
        
        # Gate checking
        self.episodes_in_current_stage = 0
        self.gate_check_window = config.get('gate_check_window', 10)  # Episodes to average over
        self.gate_check_frequency = config.get('gate_check_frequency', 5)  # Check every N episodes
        
        self.logger.info("🎓 Curriculum Learning Manager initialized:")
        self.logger.info(f"   Stages: {len(self.stages)}")
        self.logger.info(f"   Current stage: {self.current_stage.name}")
        self.logger.info(f"   Gate logic: AND (drawdown < {self.current_stage.drawdown_gate_threshold:.1%} AND Sharpe > {self.current_stage.sharpe_gate_threshold:.2f})")
        
    def _initialize_stages(self) -> List[CurriculumStage]:
        """Initialize curriculum stages from configuration."""
        stages = [
            CurriculumStage(
                name="Warm-up",
                episode_start=0,
                episode_end=30,
                drawdown_cap=0.04,  # 4%
                lambda_penalty=0.5,
                min_episodes_in_stage=10
            ),
            CurriculumStage(
                name="Stabilise",
                episode_start=31,
                episode_end=80,
                drawdown_cap=0.03,  # 3%
                lambda_penalty=1.0,
                min_episodes_in_stage=15
            ),
            CurriculumStage(
                name="Tighten",
                episode_start=81,
                episode_end=130,
                drawdown_cap=0.025,  # 2.5%
                lambda_penalty=1.5,
                min_episodes_in_stage=15
            ),
            CurriculumStage(
                name="Final",
                episode_start=131,
                episode_end=None,  # Open-ended
                drawdown_cap=0.02,  # 2%
                lambda_penalty=2.0,
                min_episodes_in_stage=20
            )
        ]
        
        return stages
    
    def get_current_constraints(self) -> Dict[str, float]:
        """Get current risk constraints for the active stage."""
        return {
            'drawdown_cap': self.current_stage.drawdown_cap,
            'lambda_penalty': self.current_stage.lambda_penalty,
            'stage_name': self.current_stage.name,
            'episode_start': self.current_stage.episode_start,
            'episode_end': self.current_stage.episode_end,
            'episodes_in_stage': self.episodes_in_current_stage
        }
    
    def record_episode_performance(self, 
                                 episode: int,
                                 total_return: float,
                                 max_drawdown: float,
                                 sharpe_ratio: float,
                                 volatility: float,
                                 num_trades: int,
                                 final_portfolio_value: float) -> Dict[str, Any]:
        """
        Record performance for completed episode and check for stage advancement.
        
        Returns:
            Dict with stage information and any advancement details
        """
        # Create performance record
        performance = EpisodePerformance(
            episode=episode,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            num_trades=num_trades,
            final_portfolio_value=final_portfolio_value,
            stage_name=self.current_stage.name
        )
        
        # Record performance
        self.episode_history.append(performance)
        self.stage_performance[self.current_stage.name].append(performance)
        self.episodes_in_current_stage += 1
        
        # Log episode performance
        self.logger.info(f"📊 Episode {episode} ({self.current_stage.name}): "
                        f"Return={total_return:.2%}, DD={max_drawdown:.2%}, "
                        f"Sharpe={sharpe_ratio:.3f}, Trades={num_trades}")
        
        # Check for stage advancement
        advancement_info = self._check_stage_advancement(episode)
        
        return {
            'current_stage': self.current_stage.name,
            'episodes_in_stage': self.episodes_in_current_stage,
            'stage_constraints': self.get_current_constraints(),
            'advancement_info': advancement_info,
            'performance': performance
        }
    
    def _check_stage_advancement(self, episode: int) -> Dict[str, Any]:
        """
        Check if agent should advance to next stage based on performance gates.
        
        Uses AND logic: both drawdown AND Sharpe criteria must be met.
        """
        advancement_info = {
            'advanced': False,
            'reason': None,
            'gate_check_performed': False,
            'criteria_met': {},
            'previous_stage': None,
            'new_stage': None
        }
        
        # Don't check if we're in the final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return advancement_info
        
        # Must have minimum episodes in current stage
        if self.episodes_in_current_stage < self.current_stage.min_episodes_in_stage:
            advancement_info['reason'] = f"Need {self.current_stage.min_episodes_in_stage - self.episodes_in_current_stage} more episodes in {self.current_stage.name}"
            return advancement_info
        
        # Check gate criteria every N episodes
        if self.episodes_in_current_stage % self.gate_check_frequency != 0:
            return advancement_info
        
        advancement_info['gate_check_performed'] = True
        
        # Get recent performance for gate evaluation
        recent_episodes = self.stage_performance[self.current_stage.name][-self.gate_check_window:]
        if len(recent_episodes) < min(self.gate_check_window, self.episodes_in_current_stage):
            advancement_info['reason'] = f"Need more episodes for gate evaluation"
            return advancement_info
        
        # Calculate average performance metrics
        avg_drawdown = np.mean([ep.max_drawdown for ep in recent_episodes])
        avg_sharpe = np.mean([ep.sharpe_ratio for ep in recent_episodes])
        
        # Check gate criteria (AND logic)
        drawdown_criterion = avg_drawdown < self.current_stage.drawdown_gate_threshold
        sharpe_criterion = avg_sharpe > self.current_stage.sharpe_gate_threshold
        
        advancement_info['criteria_met'] = {
            'drawdown': {
                'value': avg_drawdown,
                'threshold': self.current_stage.drawdown_gate_threshold,
                'met': drawdown_criterion
            },
            'sharpe': {
                'value': avg_sharpe,
                'threshold': self.current_stage.sharpe_gate_threshold,
                'met': sharpe_criterion
            }
        }
        
        # Log gate check results
        self.logger.info(f"🚪 GATE CHECK - Episode {episode} ({self.current_stage.name}):")
        self.logger.info(f"   Avg Drawdown: {avg_drawdown:.2%} {'✅' if drawdown_criterion else '❌'} (< {self.current_stage.drawdown_gate_threshold:.2%})")
        self.logger.info(f"   Avg Sharpe: {avg_sharpe:.3f} {'✅' if sharpe_criterion else '❌'} (> {self.current_stage.sharpe_gate_threshold:.2f})")
        
        # Advance if both criteria met (AND logic)
        if drawdown_criterion and sharpe_criterion:
            advancement_info['advanced'] = True
            advancement_info['previous_stage'] = self.current_stage.name
            
            # Advance to next stage
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]
            self.episodes_in_current_stage = 0
            
            advancement_info['new_stage'] = self.current_stage.name
            advancement_info['reason'] = "Both gate criteria met (AND logic)"
            
            self.logger.info(f"🎓 STAGE ADVANCEMENT: {advancement_info['previous_stage']} → {self.current_stage.name}")
            self.logger.info(f"   New constraints: DD cap={self.current_stage.drawdown_cap:.1%}, λ={self.current_stage.lambda_penalty}")
            
        else:
            # Determine what's blocking advancement
            if not drawdown_criterion and not sharpe_criterion:
                advancement_info['reason'] = "Both drawdown and Sharpe criteria not met"
            elif not drawdown_criterion:
                advancement_info['reason'] = f"Drawdown too high: {avg_drawdown:.2%} >= {self.current_stage.drawdown_gate_threshold:.2%}"
            else:
                advancement_info['reason'] = f"Sharpe too low: {avg_sharpe:.3f} <= {self.current_stage.sharpe_gate_threshold:.2f}"
        
        return advancement_info
    
    def get_stage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all stages."""
        stats = {
            'current_stage': self.current_stage.name,
            'current_stage_index': self.current_stage_idx,
            'total_episodes': len(self.episode_history),
            'episodes_in_current_stage': self.episodes_in_current_stage,
            'stages': {}
        }
        
        for stage in self.stages:
            stage_episodes = self.stage_performance[stage.name]
            if stage_episodes:
                stage_stats = {
                    'episodes_completed': len(stage_episodes),
                    'avg_return': np.mean([ep.total_return for ep in stage_episodes]),
                    'avg_drawdown': np.mean([ep.max_drawdown for ep in stage_episodes]),
                    'avg_sharpe': np.mean([ep.sharpe_ratio for ep in stage_episodes]),
                    'avg_volatility': np.mean([ep.volatility for ep in stage_episodes]),
                    'avg_trades': np.mean([ep.num_trades for ep in stage_episodes]),
                    'best_sharpe': max([ep.sharpe_ratio for ep in stage_episodes]),
                    'worst_drawdown': max([ep.max_drawdown for ep in stage_episodes])
                }
            else:
                stage_stats = {
                    'episodes_completed': 0,
                    'avg_return': 0.0,
                    'avg_drawdown': 0.0,
                    'avg_sharpe': 0.0,
                    'avg_volatility': 0.0,
                    'avg_trades': 0.0,
                    'best_sharpe': 0.0,
                    'worst_drawdown': 0.0
                }
            
            stats['stages'][stage.name] = {
                'config': {
                    'drawdown_cap': stage.drawdown_cap,
                    'lambda_penalty': stage.lambda_penalty,
                    'episode_range': f"{stage.episode_start}-{stage.episode_end or 'end'}"
                },
                'performance': stage_stats
            }
        
        return stats
    
    def should_terminate_training(self) -> Tuple[bool, str]:
        """
        Check if training should be terminated due to curriculum completion or failure.
        
        Returns:
            Tuple of (should_terminate, reason)
        """
        # Don't terminate - let training continue in final stage
        return False, ""
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get current curriculum progress information."""
        total_stages = len(self.stages)
        progress_pct = (self.current_stage_idx / (total_stages - 1)) * 100
        
        return {
            'current_stage': self.current_stage.name,
            'stage_index': self.current_stage_idx,
            'total_stages': total_stages,
            'progress_percentage': progress_pct,
            'episodes_in_stage': self.episodes_in_current_stage,
            'min_episodes_required': self.current_stage.min_episodes_in_stage,
            'stage_constraints': self.get_current_constraints(),
            'next_stage': self.stages[self.current_stage_idx + 1].name if self.current_stage_idx < total_stages - 1 else None
        }
    
    def reset_for_new_training(self):
        """Reset curriculum manager for new training run."""
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        self.episodes_in_current_stage = 0
        self.episode_history.clear()
        
        for stage_name in self.stage_performance:
            self.stage_performance[stage_name].clear()
        
        self.logger.info("🔄 Curriculum Manager reset for new training run")
        self.logger.info(f"   Starting stage: {self.current_stage.name}")
    
    def export_curriculum_data(self) -> pd.DataFrame:
        """Export curriculum performance data as DataFrame for analysis."""
        if not self.episode_history:
            return pd.DataFrame()
        
        data = []
        for ep in self.episode_history:
            data.append({
                'episode': ep.episode,
                'stage': ep.stage_name,
                'total_return': ep.total_return,
                'max_drawdown': ep.max_drawdown,
                'sharpe_ratio': ep.sharpe_ratio,
                'volatility': ep.volatility,
                'num_trades': ep.num_trades,
                'final_portfolio_value': ep.final_portfolio_value
            })
        
        return pd.DataFrame(data)