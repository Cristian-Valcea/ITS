#!/usr/bin/env python3
"""
🎓 Curriculum Scheduler for Progressive Turnover Targets

Implements a curriculum learning system that gradually increases turnover targets
as the agent masters each stage. This prevents overwhelming the agent with
difficult targets from the start.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

class CurriculumScheduler:
    """
    🎓 Curriculum Learning Scheduler for Turnover Targets
    
    Manages progressive difficulty by advancing through stages based on:
    - Minimum episodes completed in current stage
    - Performance thresholds (average reward)
    - Automatic progression when criteria are met
    """
    
    def __init__(self, curriculum_config: Dict[str, Any]):
        """
        Initialize curriculum scheduler.
        
        Args:
            curriculum_config: Configuration dict with 'enabled' and 'stages' keys
        """
        self.logger = logging.getLogger(__name__)
        
        # Parse configuration
        self.enabled = curriculum_config.get('enabled', False)
        self.stages = curriculum_config.get('stages', [])
        
        if not self.enabled:
            self.logger.info("🎓 Curriculum scheduler DISABLED")
            return
        
        if not self.stages:
            self.logger.warning("🎓 Curriculum enabled but no stages defined - disabling")
            self.enabled = False
            return
        
        # Initialize state
        self.current_stage = 0
        self.episodes_in_current_stage = 0
        self.episode_rewards = []  # Track recent rewards for threshold checking
        self.stage_history = []    # Track stage progression
        
        # Validate stages
        self._validate_stages()
        
        # Log initialization
        self.logger.info(f"🎓 Curriculum scheduler initialized with {len(self.stages)} stages")
        self._log_current_stage()
    
    def _validate_stages(self) -> None:
        """Validate stage configuration."""
        for i, stage in enumerate(self.stages):
            if 'target_ratio' not in stage:
                raise ValueError(f"Stage {i} missing 'target_ratio'")
            if 'min_episodes' not in stage:
                raise ValueError(f"Stage {i} missing 'min_episodes'")
            
            # Validate values
            if stage['target_ratio'] <= 0:
                raise ValueError(f"Stage {i} target_ratio must be positive")
            if stage['min_episodes'] <= 0:
                raise ValueError(f"Stage {i} min_episodes must be positive")
    
    def _log_current_stage(self) -> None:
        """Log current stage information."""
        if not self.enabled:
            return
        
        stage = self.stages[self.current_stage]
        self.logger.info(
            f"🎓 CURRICULUM STAGE {self.current_stage + 1}/{len(self.stages)}: "
            f"Target={stage['target_ratio']:.3f}, "
            f"Min Episodes={stage['min_episodes']}, "
            f"Threshold={stage.get('reward_threshold', 'None')}"
        )
    
    def get_current_target_ratio(self) -> float:
        """Get the current target ratio for the active stage."""
        if not self.enabled or not self.stages:
            return 0.5  # Default fallback
        
        return self.stages[self.current_stage]['target_ratio']
    
    def on_episode_end(self, episode_reward: float) -> bool:
        """
        Called at the end of each episode to update curriculum state.
        
        Args:
            episode_reward: Total reward for the completed episode
            
        Returns:
            bool: True if stage advanced, False otherwise
        """
        if not self.enabled:
            return False
        
        # Track episode
        self.episodes_in_current_stage += 1
        self.episode_rewards.append(episode_reward)
        
        # Keep only recent rewards for threshold calculation (last 10 episodes)
        if len(self.episode_rewards) > 10:
            self.episode_rewards.pop(0)
        
        # Check if we should advance to next stage
        return self._check_stage_advancement()
    
    def _check_stage_advancement(self) -> bool:
        """Check if criteria are met to advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            # Already at final stage
            return False
        
        current_stage = self.stages[self.current_stage]
        
        # Check minimum episodes requirement
        if self.episodes_in_current_stage < current_stage['min_episodes']:
            return False
        
        # Check reward threshold (if specified)
        reward_threshold = current_stage.get('reward_threshold')
        if reward_threshold is not None:
            if len(self.episode_rewards) < 5:  # Need some history
                return False
            
            # Calculate average reward over recent episodes
            avg_reward = np.mean(self.episode_rewards[-5:])  # Last 5 episodes
            
            if avg_reward < reward_threshold:
                self.logger.debug(
                    f"🎓 Stage advancement blocked: avg_reward={avg_reward:.1f} < threshold={reward_threshold}"
                )
                return False
        
        # All criteria met - advance stage
        self._advance_stage()
        return True
    
    def _advance_stage(self) -> None:
        """Advance to the next curriculum stage."""
        old_stage = self.current_stage
        old_target = self.stages[old_stage]['target_ratio']
        
        # Record stage completion
        self.stage_history.append({
            'stage': old_stage,
            'episodes_completed': self.episodes_in_current_stage,
            'final_avg_reward': np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else 0.0,
            'target_ratio': old_target
        })
        
        # Advance to next stage
        self.current_stage += 1
        self.episodes_in_current_stage = 0
        self.episode_rewards = []  # Reset for new stage
        
        new_target = self.stages[self.current_stage]['target_ratio']
        
        # Log advancement
        self.logger.info(
            f"🎓 CURRICULUM ADVANCEMENT! "
            f"Stage {old_stage + 1} → {self.current_stage + 1} | "
            f"Target: {old_target:.3f} → {new_target:.3f} | "
            f"Episodes completed: {self.stage_history[-1]['episodes_completed']}"
        )
        
        self._log_current_stage()
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current curriculum progress information."""
        if not self.enabled:
            return {'enabled': False}
        
        current_stage = self.stages[self.current_stage]
        
        # Calculate progress in current stage
        min_episodes = current_stage['min_episodes']
        progress_pct = min(100.0, (self.episodes_in_current_stage / min_episodes) * 100)
        
        # Calculate average recent reward
        avg_reward = np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else 0.0
        
        return {
            'enabled': True,
            'current_stage': self.current_stage + 1,
            'total_stages': len(self.stages),
            'current_target_ratio': current_stage['target_ratio'],
            'episodes_in_stage': self.episodes_in_current_stage,
            'min_episodes_required': min_episodes,
            'stage_progress_pct': progress_pct,
            'reward_threshold': current_stage.get('reward_threshold'),
            'avg_recent_reward': avg_reward,
            'stage_history': self.stage_history.copy()
        }
    
    def force_advance_stage(self) -> bool:
        """
        Force advancement to next stage (for testing/debugging).
        
        Returns:
            bool: True if advanced, False if already at final stage
        """
        if not self.enabled or self.current_stage >= len(self.stages) - 1:
            return False
        
        self.logger.warning(f"🎓 FORCED curriculum advancement from stage {self.current_stage + 1}")
        self._advance_stage()
        return True
    
    def reset_to_stage(self, stage_index: int) -> bool:
        """
        Reset curriculum to a specific stage (for testing/debugging).
        
        Args:
            stage_index: 0-based stage index
            
        Returns:
            bool: True if reset successful, False if invalid stage
        """
        if not self.enabled or stage_index < 0 or stage_index >= len(self.stages):
            return False
        
        self.logger.warning(f"🎓 RESET curriculum to stage {stage_index + 1}")
        self.current_stage = stage_index
        self.episodes_in_current_stage = 0
        self.episode_rewards = []
        
        self._log_current_stage()
        return True
    
    def is_final_stage(self) -> bool:
        """Check if currently at the final curriculum stage."""
        if not self.enabled:
            return True
        
        return self.current_stage >= len(self.stages) - 1
    
    def get_stage_summary(self) -> str:
        """Get a human-readable summary of curriculum progress."""
        if not self.enabled:
            return "Curriculum: DISABLED"
        
        info = self.get_progress_info()
        
        summary = (
            f"Curriculum: Stage {info['current_stage']}/{info['total_stages']} | "
            f"Target: {info['current_target_ratio']:.3f} | "
            f"Progress: {info['episodes_in_stage']}/{info['min_episodes_required']} episodes "
            f"({info['stage_progress_pct']:.1f}%)"
        )
        
        if info['reward_threshold'] is not None:
            summary += f" | Threshold: {info['reward_threshold']:.1f} (avg: {info['avg_recent_reward']:.1f})"
        
        return summary