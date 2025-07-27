"""
Q-Value Variance Tracking Module

Provides utilities to track Q-value variance (Q_max - Q_min) for DQN-based algorithms.
This helps monitor training stability and detect Q-value estimation issues.
"""

import logging
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from collections import deque


class QValueTracker:
    """
    Tracks Q-value statistics for monitoring training stability.
    
    Monitors:
    - Q-value variance (Q_max - Q_min)
    - Q-value distribution statistics
    - Q-value evolution over time
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.q_values_buffer = deque(maxlen=buffer_size)
        self.q_variance_buffer = deque(maxlen=buffer_size)
        self.q_mean_buffer = deque(maxlen=buffer_size)
        
        self.logger = logging.getLogger(__name__)
        
    def update_q_values(self, model, observation: np.ndarray) -> Optional[float]:
        """
        Extract Q-values from model and update tracking.
        
        Args:
            model: The RL model (DQN, QR-DQN, etc.)
            observation: Current observation
            
        Returns:
            Q-value variance (Q_max - Q_min) or None if not available
        """
        try:
            # Handle different model types
            q_values = self._extract_q_values(model, observation)
            
            if q_values is not None and len(q_values) > 0:
                # Calculate statistics
                q_max = float(np.max(q_values))
                q_min = float(np.min(q_values))
                q_mean = float(np.mean(q_values))
                q_variance = q_max - q_min
                
                # Update buffers
                self.q_values_buffer.append(q_values.copy())
                self.q_variance_buffer.append(q_variance)
                self.q_mean_buffer.append(q_mean)
                
                return q_variance
                
        except Exception as e:
            self.logger.debug(f"Error extracting Q-values: {e}")
            
        return None
    
    def _extract_q_values(self, model, observation: np.ndarray) -> Optional[np.ndarray]:
        """Extract Q-values from different model types."""
        try:
            # Convert observation to tensor if needed
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            else:
                obs_tensor = observation
            
            # Handle different SB3 model types
            if hasattr(model, 'q_net'):
                # DQN, QR-DQN models
                with torch.no_grad():
                    q_values = model.q_net(obs_tensor)
                    if isinstance(q_values, tuple):
                        q_values = q_values[0]  # Take first element if tuple
                    return q_values.cpu().numpy().flatten()
                    
            elif hasattr(model, 'policy') and hasattr(model.policy, 'q_net'):
                # Alternative structure
                with torch.no_grad():
                    q_values = model.policy.q_net(obs_tensor)
                    if isinstance(q_values, tuple):
                        q_values = q_values[0]
                    return q_values.cpu().numpy().flatten()
                    
            elif hasattr(model, 'predict_values'):
                # Value-based methods
                with torch.no_grad():
                    values = model.predict_values(obs_tensor)
                    return values.cpu().numpy().flatten()
                    
            else:
                # Try to call the model directly
                with torch.no_grad():
                    output = model(obs_tensor)
                    if torch.is_tensor(output):
                        return output.cpu().numpy().flatten()
                        
        except Exception as e:
            self.logger.debug(f"Q-value extraction failed: {e}")
            
        return None
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current Q-value statistics."""
        stats = {}
        
        if self.q_variance_buffer:
            stats['q_variance_current'] = self.q_variance_buffer[-1]
            stats['q_variance_mean'] = np.mean(list(self.q_variance_buffer))
            stats['q_variance_std'] = np.std(list(self.q_variance_buffer))
            stats['q_variance_max'] = np.max(list(self.q_variance_buffer))
            
        if self.q_mean_buffer:
            stats['q_mean_current'] = self.q_mean_buffer[-1]
            stats['q_mean_avg'] = np.mean(list(self.q_mean_buffer))
            
        if self.q_values_buffer:
            # Get latest Q-values
            latest_q = self.q_values_buffer[-1]
            stats['q_max_current'] = float(np.max(latest_q))
            stats['q_min_current'] = float(np.min(latest_q))
            
        stats['buffer_size'] = len(self.q_variance_buffer)
        
        return stats
    
    def reset(self):
        """Reset all buffers."""
        self.q_values_buffer.clear()
        self.q_variance_buffer.clear()
        self.q_mean_buffer.clear()


class QValueMonitoringCallback:
    """
    Callback to integrate Q-value tracking with training.
    
    Can be used standalone or integrated into TensorBoardMonitoringCallback.
    """
    
    def __init__(self, model, buffer_size: int = 1000):
        self.model = model
        self.tracker = QValueTracker(buffer_size)
        self.step_count = 0
        
    def on_step(self, observation: np.ndarray) -> Optional[float]:
        """Called at each training step."""
        self.step_count += 1
        return self.tracker.update_q_values(self.model, observation)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        stats = self.tracker.get_current_stats()
        stats['step_count'] = self.step_count
        return stats
    
    def reset(self):
        """Reset tracking."""
        self.tracker.reset()
        self.step_count = 0


# Utility functions for integration
def create_q_value_tracker(model, config: Dict[str, Any]) -> Optional[QValueMonitoringCallback]:
    """
    Create Q-value tracker if model supports it.
    
    Args:
        model: RL model
        config: Configuration dictionary
        
    Returns:
        QValueMonitoringCallback or None if not supported
    """
    try:
        # Check if model supports Q-value extraction
        if hasattr(model, 'q_net') or hasattr(model, 'policy'):
            buffer_size = config.get('monitoring', {}).get('buffer_size', 1000)
            return QValueMonitoringCallback(model, buffer_size)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Q-value tracker creation failed: {e}")
    
    return None


def is_q_value_compatible(algorithm: str) -> bool:
    """Check if algorithm supports Q-value tracking."""
    q_value_algorithms = {
        'DQN', 'DDQN', 'QR-DQN', 'QRDQN', 'Rainbow', 'C51', 'IQN'
    }
    return algorithm in q_value_algorithms


# Export public interface
__all__ = [
    'QValueTracker',
    'QValueMonitoringCallback',
    'create_q_value_tracker',
    'is_q_value_compatible'
]