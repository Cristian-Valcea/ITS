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
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            max_episodes: Maximum number of episodes before stopping
            max_training_time_minutes: Maximum training time in minutes
            plateau_patience: Number of episodes to wait for improvement
            min_improvement: Minimum improvement to reset patience counter
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.max_training_time_minutes = max_training_time_minutes
        self.plateau_patience = plateau_patience
        self.min_improvement = min_improvement
        
        # Tracking variables
        self.episode_count = 0
        self.best_reward = -np.inf
        self.patience_counter = 0
        self.episode_rewards = []
        self.start_time = None
        
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
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Check if episode just ended (handle vector environments)
        infos = self.locals.get('infos', [])
        if infos:
            # Iterate over all environments in case of VecEnv
            for info in infos:
                # Initialize episode_reward to prevent UnboundLocalError
                episode_reward = None
                
                if info and info.get('episode'):
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    self.episode_rewards.append(episode_reward)
                    
                    # Check for improvement (episode_reward is guaranteed to be set here)
                    if episode_reward > self.best_reward + self.min_improvement:
                        self.best_reward = episode_reward
                        self.patience_counter = 0
                        if self.verbose >= 1:
                            self._logger.info(f"Episode {self.episode_count}: New best reward {episode_reward:.2f}")
                    else:
                        self.patience_counter += 1
                    
                    # Check stopping conditions
                    if self._should_stop():
                        return False
                    
        return True
        
    def _should_stop(self) -> bool:
        """Check if training should stop based on configured conditions."""
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
                
        # Check plateau
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