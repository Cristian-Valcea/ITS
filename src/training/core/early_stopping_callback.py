"""
Early Stopping Callback for Training

Prevents infinite loops by monitoring training progress and stopping
when sufficient episodes have been completed or performance plateaus.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback to prevent infinite training loops.
    
    Monitors episode completion and can stop training based on:
    - Maximum number of episodes
    - Performance plateau detection
    - Time-based limits
    """
    
    def __init__(
        self,
        max_episodes: Optional[int] = None,
        max_training_time_minutes: Optional[int] = None,
        plateau_patience: int = 10,
        min_improvement: float = 0.01,
        check_freq: int = 2000,
        min_episodes_before_stopping: int = 10,
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            max_episodes: Maximum number of episodes before stopping
            max_training_time_minutes: Maximum training time in minutes
            plateau_patience: Number of episodes to wait for improvement
            min_improvement: Minimum improvement to reset patience counter
            check_freq: Check early stopping conditions every N steps
            min_episodes_before_stopping: Minimum episodes before early stopping can trigger
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.max_training_time_minutes = max_training_time_minutes
        self.plateau_patience = plateau_patience
        self.min_improvement = min_improvement
        self.check_freq = check_freq
        self.min_episodes_before_stopping = min_episodes_before_stopping
        
        # Tracking variables
        self.episode_count = 0
        self.best_reward = -np.inf
        self.patience_counter = 0
        self.episode_rewards = []
        self.start_time = None
        self.last_check_step = 0
        
        self._logger = logging.getLogger(__name__)
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        import time
        self.start_time = time.time()
        self._logger.info(f"Early stopping callback initialized:")
        if self.max_episodes:
            self._logger.info(f"  - Max episodes: {self.max_episodes}")
        if self.max_training_time_minutes:
            self._logger.info(f"  - Max training time: {self.max_training_time_minutes} minutes")
        self._logger.info(f"  - Plateau patience: {self.plateau_patience} episodes")
        self._logger.info(f"  - Check frequency: every {self.check_freq} steps")
        self._logger.info(f"  - Min episodes before stopping: {self.min_episodes_before_stopping}")
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Track episode completions
        infos = self.locals.get('infos', [])
        if infos:
            # Iterate over all environments in case of VecEnv
            for info in infos:
                if info and info.get('episode'):
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    self.episode_rewards.append(episode_reward)
                    
                    # Check for improvement
                    if episode_reward > self.best_reward + self.min_improvement:
                        self.best_reward = episode_reward
                        self.patience_counter = 0
                        if self.verbose >= 1:
                            self._logger.info(f"Episode {self.episode_count}: New best reward {episode_reward:.2f}")
                    else:
                        self.patience_counter += 1
        
        # Check stopping conditions only at specified frequency
        current_step = self.num_timesteps
        if current_step - self.last_check_step >= self.check_freq:
            self.last_check_step = current_step
            if self._should_stop():
                return False
                    
        return True
        
    def _should_stop(self) -> bool:
        """Check if training should stop based on configured conditions."""
        # Always allow minimum episodes before any early stopping
        if self.episode_count < self.min_episodes_before_stopping:
            return False
            
        # Check max episodes
        if self.max_episodes and self.episode_count >= self.max_episodes:
            self._logger.info(f"Stopping: Reached max episodes ({self.max_episodes})")
            return True
            
        # Check max training time
        if self.max_training_time_minutes and self.start_time:
            import time
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_training_time_minutes:
                self._logger.info(f"Stopping: Reached max training time ({self.max_training_time_minutes} minutes)")
                return True
                
        # Check plateau (only after minimum episodes)
        if self.patience_counter >= self.plateau_patience:
            self._logger.info(f"Stopping: Performance plateau detected ({self.plateau_patience} episodes without improvement)")
            return True
            
        return False
        
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.start_time:
            import time
            total_time = (time.time() - self.start_time) / 60
            self._logger.info(f"Training completed after {self.episode_count} episodes in {total_time:.1f} minutes")
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
                self._logger.info(f"Average reward (last 10 episodes): {avg_reward:.2f}")