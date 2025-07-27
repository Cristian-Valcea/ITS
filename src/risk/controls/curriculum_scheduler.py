"""
Curriculum Scheduler - Episode-based progression system

Implements curriculum learning for risk controls, gradually tightening
risk limits as the agent improves. Supports both episode-based and
performance-based advancement triggers.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    episode_range: tuple  # (start_episode, end_episode)
    dd_limit: float
    penalty_lambda: float
    advance_conditions: Dict[str, Any]


class CurriculumScheduler:
    """
    Manages curriculum progression for risk controls.
    
    Tracks episode performance and advances through curriculum stages
    based on configurable conditions (episode count, performance metrics).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum configuration containing:
                - enabled: Whether curriculum is active
                - use_perf_trigger: Use performance-based triggers
                - logic: "and" or "or" for advancement conditions
                - stages: List of stage configurations
        """
        self.logger = logging.getLogger("CurriculumScheduler")
        
        self.enabled = config.get('enabled', False)
        self.use_perf_trigger = config.get('use_perf_trigger', False)
        self.logic = config.get('logic', 'and')
        
        # Parse stages
        self.stages = []
        if self.enabled:
            self._parse_stages(config.get('stages', []))
        
        # Current state
        self.current_stage_idx = 0
        self.current_episode = 0
        self.episode_metrics = []  # Store metrics for performance evaluation
        
        # Performance tracking for triggers
        self.performance_window = 10  # Episodes to average for performance triggers
        
        if self.enabled:
            self.logger.info(f"CurriculumScheduler initialized - {len(self.stages)} stages, "
                           f"logic: {self.logic}, perf_trigger: {self.use_perf_trigger}")
            self._log_stage_summary()
        else:
            self.logger.info("CurriculumScheduler disabled")
    
    def _parse_stages(self, stages_config: List[Dict[str, Any]]):
        """Parse stage configurations into CurriculumStage objects."""
        for i, stage_config in enumerate(stages_config):
            # Parse episode range
            episodes = stage_config.get('episodes', [0, 99])
            if isinstance(episodes, list) and len(episodes) == 2:
                episode_range = tuple(episodes)
            elif isinstance(episodes, str):
                # Parse "0-19" format
                start, end = map(int, episodes.split('-'))
                episode_range = (start, end)
            else:
                episode_range = (i * 20, (i + 1) * 20 - 1)  # Default ranges
            
            stage = CurriculumStage(
                name=stage_config.get('name', f'stage_{i+1}'),
                episode_range=episode_range,
                dd_limit=stage_config.get('dd_limit', 0.03),
                penalty_lambda=stage_config.get('penalty_lambda', 0.25),
                advance_conditions=stage_config.get('advance_conditions', {})
            )
            
            self.stages.append(stage)
    
    def _log_stage_summary(self):
        """Log summary of all curriculum stages."""
        self.logger.info("📚 Curriculum Stages:")
        for i, stage in enumerate(self.stages):
            self.logger.info(f"  Stage {i+1} ({stage.name}): "
                           f"episodes {stage.episode_range[0]}-{stage.episode_range[1]}, "
                           f"dd_limit={stage.dd_limit:.1%}, λ={stage.penalty_lambda}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current risk configuration based on curriculum stage.
        
        Returns:
            Risk configuration dict for current stage
        """
        if not self.enabled or not self.stages:
            return {}
        
        current_stage = self.stages[self.current_stage_idx]
        
        return {
            'dd_limit': current_stage.dd_limit,
            'penalty_lambda': current_stage.penalty_lambda,
            'curriculum_stage': current_stage.name,
            'curriculum_episode': self.current_episode
        }
    
    def episode_end(self, episode_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process end of episode and check for stage advancement.
        
        Args:
            episode_metrics: Metrics from completed episode containing:
                - sharpe_ratio: Episode Sharpe ratio
                - max_drawdown: Maximum drawdown during episode
                - total_return: Total return for episode
                - volatility: Average volatility
                - episode_length: Number of steps
        
        Returns:
            New risk configuration if stage advanced, None otherwise
        """
        if not self.enabled:
            return None
        
        self.current_episode += 1
        self.episode_metrics.append({
            'episode': self.current_episode,
            'stage': self.current_stage_idx,
            **episode_metrics
        })
        
        # Keep only recent metrics for performance evaluation
        if len(self.episode_metrics) > 100:
            self.episode_metrics = self.episode_metrics[-100:]
        
        # Check if we should advance to next stage
        if self._should_advance_stage():
            return self._advance_stage()
        
        return None
    
    def _should_advance_stage(self) -> bool:
        """Check if conditions are met to advance to next stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_stage = self.stages[self.current_stage_idx]
        
        # Episode-based advancement (always checked)
        episode_condition = self.current_episode >= current_stage.episode_range[1]
        
        if not self.use_perf_trigger:
            return episode_condition
        
        # Performance-based advancement
        perf_conditions = self._check_performance_conditions(current_stage.advance_conditions)
        
        if self.logic == 'and':
            return episode_condition and perf_conditions
        else:  # 'or'
            return episode_condition or perf_conditions
    
    def _check_performance_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if performance conditions are met."""
        if not conditions or len(self.episode_metrics) < self.performance_window:
            return True  # No conditions or insufficient data
        
        # Get recent metrics for evaluation
        recent_metrics = self.episode_metrics[-self.performance_window:]
        
        # Calculate averages
        avg_sharpe = sum(m.get('sharpe_ratio', 0) for m in recent_metrics) / len(recent_metrics)
        max_dd = max(m.get('max_drawdown', 0) for m in recent_metrics)
        min_episodes = len([m for m in recent_metrics if m['stage'] == self.current_stage_idx])
        
        # Check each condition
        results = []
        
        if 'min_sharpe' in conditions:
            results.append(avg_sharpe >= conditions['min_sharpe'])
        
        if 'max_drawdown' in conditions:
            results.append(max_dd <= conditions['max_drawdown'])
        
        if 'min_episodes' in conditions:
            results.append(min_episodes >= conditions['min_episodes'])
        
        # Apply logic
        if not results:
            return True
        
        if self.logic == 'and':
            return all(results)
        else:
            return any(results)
    
    def _advance_stage(self) -> Dict[str, Any]:
        """Advance to next curriculum stage."""
        old_stage = self.stages[self.current_stage_idx]
        self.current_stage_idx = min(self.current_stage_idx + 1, len(self.stages) - 1)
        new_stage = self.stages[self.current_stage_idx]
        
        self.logger.info(f"📈 Curriculum Advanced! {old_stage.name} → {new_stage.name}")
        self.logger.info(f"   Episode {self.current_episode}: "
                        f"dd_limit {old_stage.dd_limit:.1%} → {new_stage.dd_limit:.1%}, "
                        f"λ {old_stage.penalty_lambda} → {new_stage.penalty_lambda}")
        
        # Log recent performance that triggered advancement
        if self.use_perf_trigger and len(self.episode_metrics) >= 5:
            recent = self.episode_metrics[-5:]
            avg_sharpe = sum(m.get('sharpe_ratio', 0) for m in recent) / len(recent)
            avg_dd = sum(m.get('max_drawdown', 0) for m in recent) / len(recent)
            self.logger.info(f"   Recent performance: Sharpe={avg_sharpe:.2f}, DD={avg_dd:.1%}")
        
        return self.get_current_config()
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        if not self.enabled or not self.stages:
            return {'enabled': False}
        
        current_stage = self.stages[self.current_stage_idx]
        
        return {
            'enabled': True,
            'current_stage': self.current_stage_idx + 1,
            'total_stages': len(self.stages),
            'stage_name': current_stage.name,
            'episode': self.current_episode,
            'episode_range': current_stage.episode_range,
            'dd_limit': current_stage.dd_limit,
            'penalty_lambda': current_stage.penalty_lambda,
            'progress_pct': min(100, (self.current_episode / current_stage.episode_range[1]) * 100)
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum metrics for logging."""
        if not self.episode_metrics:
            return {}
        
        # Group by stage
        stage_metrics = {}
        for metric in self.episode_metrics:
            stage = metric['stage']
            if stage not in stage_metrics:
                stage_metrics[stage] = []
            stage_metrics[stage].append(metric)
        
        # Calculate stage summaries
        summary = {}
        for stage_idx, metrics in stage_metrics.items():
            if stage_idx < len(self.stages):
                stage_name = self.stages[stage_idx].name
                avg_sharpe = sum(m.get('sharpe_ratio', 0) for m in metrics) / len(metrics)
                avg_dd = sum(m.get('max_drawdown', 0) for m in metrics) / len(metrics)
                
                summary[stage_name] = {
                    'episodes': len(metrics),
                    'avg_sharpe': avg_sharpe,
                    'avg_drawdown': avg_dd,
                    'completed': stage_idx < self.current_stage_idx
                }
        
        return summary
    
    def save_progress(self, filepath: str):
        """Save curriculum progress to file."""
        progress_data = {
            'current_stage_idx': self.current_stage_idx,
            'current_episode': self.current_episode,
            'episode_metrics': self.episode_metrics[-50:],  # Save last 50 episodes
            'stage_info': self.get_stage_info(),
            'metrics_summary': self.get_metrics_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.logger.info(f"📁 Curriculum progress saved to {filepath}")
    
    def load_progress(self, filepath: str):
        """Load curriculum progress from file."""
        try:
            with open(filepath, 'r') as f:
                progress_data = json.load(f)
            
            self.current_stage_idx = progress_data.get('current_stage_idx', 0)
            self.current_episode = progress_data.get('current_episode', 0)
            self.episode_metrics = progress_data.get('episode_metrics', [])
            
            self.logger.info(f"📁 Curriculum progress loaded from {filepath}")
            self.logger.info(f"   Resumed at episode {self.current_episode}, stage {self.current_stage_idx + 1}")
            
        except FileNotFoundError:
            self.logger.info(f"📁 No curriculum progress file found at {filepath}, starting fresh")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load curriculum progress: {e}")