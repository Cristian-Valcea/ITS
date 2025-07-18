#!/usr/bin/env python3
"""
TensorBoard Export Support for Turnover Penalty System

This module provides comprehensive TensorBoard logging capabilities for:
- Training loss and convergence metrics
- Total reward and episode performance
- Turnover penalty evolution and analysis
- Win rate, Sharpe ratio, drawdown tracking
- Risk management indicators
- Portfolio performance metrics

Usage:
    from src.training.core.tensorboard_exporter import TensorBoardExporter
    
    exporter = TensorBoardExporter(log_dir="runs/experiment_1")
    
    # Log metrics during training
    exporter.log_episode_metrics(
        episode=100,
        total_reward=1250.0,
        turnover_penalty=-520.0,  # Fixed: penalty scales with NAV
        turnover_ratio=0.025,     # Fixed: dimensionless ratio
        turnover_target=0.02,     # 2% target
        portfolio_value=52000.0
    )
    
    # Log training metrics
    exporter.log_training_metrics(
        step=10000,
        loss=0.045,
        q_values=[1.2, 0.8, 1.5],
        learning_rate=0.001
    )
    
    exporter.close()
"""

import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from collections import deque
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

logger = logging.getLogger(__name__)


class TensorBoardExporter:
    """
    Comprehensive TensorBoard exporter for RL trading system metrics.
    
    Provides organized logging of:
    - Training metrics (loss, learning rate, gradients)
    - Performance metrics (rewards, win rate, Sharpe ratio)
    - Turnover penalty metrics (penalty, normalized turnover, excess)
    - Risk metrics (drawdown, volatility, portfolio value)
    - System metrics (buffer sizes, computation time)
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        model_name: str = "RL_Trading_Model",
        comment: str = "",
        flush_secs: int = 120
    ):
        """
        Initialize TensorBoard exporter.
        
        Args:
            log_dir: Directory for TensorBoard logs
            model_name: Name of the model being trained
            comment: Additional comment for the run
            flush_secs: How often to flush logs to disk
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_name}_{timestamp}"
        if comment:
            run_name += f"_{comment}"
        
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=str(self.run_dir),
            flush_secs=flush_secs
        )
        
        # Tracking variables
        self.step_count = 0
        self.episode_count = 0
        
        # Metric buffers for smoothing
        self.reward_buffer = deque(maxlen=100)
        self.win_buffer = deque(maxlen=100)
        self.portfolio_buffer = deque(maxlen=1000)
        
        logger.info(f"TensorBoard exporter initialized: {self.run_dir}")
        
        # Log system info
        self._log_system_info()
    
    def _log_system_info(self) -> None:
        """Log system and configuration information."""
        try:
            import platform
            import torch
            
            # System information
            self.writer.add_text('system/platform', platform.platform())
            self.writer.add_text('system/python_version', platform.python_version())
            self.writer.add_text('system/pytorch_version', torch.__version__)
            
            # GPU information
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.writer.add_text('system/gpu', f"{gpu_name} ({gpu_memory:.1f}GB)")
                self.writer.add_scalar('system/gpu_memory_gb', gpu_memory, 0)
            else:
                self.writer.add_text('system/gpu', 'CPU only')
            
            # Model information
            self.writer.add_text('model/name', self.model_name)
            self.writer.add_text('model/timestamp', datetime.now().isoformat())
            
        except Exception as e:
            logger.warning(f"Failed to log system info: {e}")
    
    def log_training_metrics(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        q_values: Optional[List[float]] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy_loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log training-specific metrics.
        
        Args:
            step: Training step number
            loss: Total training loss
            learning_rate: Current learning rate
            q_values: List of Q-values for variance tracking
            policy_loss: Policy loss (for actor-critic methods)
            value_loss: Value function loss
            entropy_loss: Entropy regularization loss
            gradient_norm: Gradient norm for monitoring
            **kwargs: Additional training metrics
        """
        if loss is not None:
            self.writer.add_scalar('training/loss', loss, step)
        
        if learning_rate is not None:
            self.writer.add_scalar('training/learning_rate', learning_rate, step)
        
        if q_values is not None and len(q_values) > 0:
            q_array = np.array(q_values)
            self.writer.add_scalar('training/q_value_mean', np.mean(q_array), step)
            self.writer.add_scalar('training/q_value_std', np.std(q_array), step)
            self.writer.add_scalar('training/q_value_max', np.max(q_array), step)
            self.writer.add_scalar('training/q_value_min', np.min(q_array), step)
            self.writer.add_scalar('training/q_value_variance', np.max(q_array) - np.min(q_array), step)
        
        if policy_loss is not None:
            self.writer.add_scalar('training/policy_loss', policy_loss, step)
        
        if value_loss is not None:
            self.writer.add_scalar('training/value_loss', value_loss, step)
        
        if entropy_loss is not None:
            self.writer.add_scalar('training/entropy_loss', entropy_loss, step)
        
        if gradient_norm is not None:
            self.writer.add_scalar('training/gradient_norm', gradient_norm, step)
        
        # Log additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'training/{key}', value, step)
        
        self.step_count = max(self.step_count, step)
    
    def log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        episode_length: Optional[int] = None,
        portfolio_value: Optional[float] = None,
        turnover_penalty: Optional[float] = None,
        turnover_ratio: Optional[float] = None,
        turnover_target: Optional[float] = None,
        win: Optional[bool] = None,
        **kwargs
    ) -> None:
        """
        Log episode-level performance metrics.
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            episode_length: Number of steps in episode
            portfolio_value: Final portfolio value
            turnover_penalty: Turnover penalty applied
            turnover_ratio: Turnover ratio (dimensionless, ≤1 for 1× capital)
            turnover_target: Target turnover ratio
            win: Whether episode was profitable
            **kwargs: Additional episode metrics
        """
        # Core performance metrics
        self.writer.add_scalar('episode/total_reward', total_reward, episode)
        self.reward_buffer.append(total_reward)
        
        if episode_length is not None:
            self.writer.add_scalar('episode/length', episode_length, episode)
        
        if portfolio_value is not None:
            self.writer.add_scalar('episode/portfolio_value', portfolio_value, episode)
            self.portfolio_buffer.append(portfolio_value)
        
        # Fixed turnover penalty metrics
        if turnover_penalty is not None:
            self.writer.add_scalar('turnover/penalty_current', turnover_penalty, episode)
        
        if turnover_ratio is not None:
            # Turnover ratio (dimensionless, ≤1 for 1× capital)
            self.writer.add_scalar('turnover/ratio_current', turnover_ratio, episode)
        
        if turnover_target is not None:
            self.writer.add_scalar('turnover/target', turnover_target, episode)
            
            if turnover_ratio is not None:
                excess = turnover_ratio - turnover_target
                relative_excess = excess / turnover_target if turnover_target > 0 else excess
                self.writer.add_scalar('turnover/excess_current', excess, episode)
                self.writer.add_scalar('turnover/relative_excess_current', relative_excess, episode)
        
        # Win tracking
        if win is not None:
            self.win_buffer.append(1.0 if win else 0.0)
        elif total_reward > 0:
            self.win_buffer.append(1.0)
        else:
            self.win_buffer.append(0.0)
        
        # Calculate rolling statistics
        if len(self.reward_buffer) >= 10:
            reward_mean = np.mean(list(self.reward_buffer))
            reward_std = np.std(list(self.reward_buffer))
            self.writer.add_scalar('episode/reward_mean_100', reward_mean, episode)
            self.writer.add_scalar('episode/reward_std_100', reward_std, episode)
            
            # Sharpe ratio (if we have enough variance)
            if reward_std > 0:
                sharpe_ratio = reward_mean / reward_std * np.sqrt(252)  # Annualized
                self.writer.add_scalar('performance/sharpe_ratio', sharpe_ratio, episode)
        
        # Win rate
        if len(self.win_buffer) >= 10:
            win_rate = np.mean(list(self.win_buffer)) * 100
            self.writer.add_scalar('performance/win_rate', win_rate, episode)
        
        # Drawdown calculation
        if len(self.portfolio_buffer) >= 10:
            portfolio_values = np.array(list(self.portfolio_buffer))
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak * 100
            max_drawdown = np.max(drawdown)
            current_drawdown = drawdown[-1]
            
            self.writer.add_scalar('performance/max_drawdown', max_drawdown, episode)
            self.writer.add_scalar('performance/current_drawdown', current_drawdown, episode)
        
        # Log additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'episode/{key}', value, episode)
        
        self.episode_count = max(self.episode_count, episode)
    
    def log_risk_metrics(
        self,
        step: int,
        volatility: Optional[float] = None,
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
        expected_shortfall: Optional[float] = None,
        beta: Optional[float] = None,
        correlation: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log risk management metrics.
        
        Args:
            step: Step number
            volatility: Portfolio volatility
            var_95: Value at Risk (95%)
            var_99: Value at Risk (99%)
            expected_shortfall: Expected shortfall (CVaR)
            beta: Portfolio beta
            correlation: Market correlation
            **kwargs: Additional risk metrics
        """
        if volatility is not None:
            self.writer.add_scalar('risk/volatility', volatility, step)
        
        if var_95 is not None:
            self.writer.add_scalar('risk/var_95', var_95, step)
        
        if var_99 is not None:
            self.writer.add_scalar('risk/var_99', var_99, step)
        
        if expected_shortfall is not None:
            self.writer.add_scalar('risk/expected_shortfall', expected_shortfall, step)
        
        if beta is not None:
            self.writer.add_scalar('risk/beta', beta, step)
        
        if correlation is not None:
            self.writer.add_scalar('risk/correlation', correlation, step)
        
        # Log additional risk metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'risk/{key}', value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log hyperparameters and final metrics for comparison.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        # Convert complex objects to strings for TensorBoard
        hparams_clean = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_clean[key] = value
            else:
                hparams_clean[key] = str(value)
        
        self.writer.add_hparams(hparams_clean, metrics)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """
        Log model computational graph.
        
        Args:
            model: PyTorch model
            input_sample: Sample input tensor
        """
        try:
            self.writer.add_graph(model, input_sample)
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int) -> None:
        """
        Log histogram of values.
        
        Args:
            tag: Tag for the histogram
            values: Values to histogram
            step: Step number
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text information.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Optional step number
        """
        if step is None:
            step = self.step_count
        
        self.writer.add_text(tag, text, step)
    
    def flush(self) -> None:
        """Flush pending logs to disk."""
        self.writer.flush()
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logger.info(f"TensorBoard logs saved to: {self.run_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_tensorboard_exporter(
    log_dir: Union[str, Path] = "runs",
    model_name: str = "RL_Trading_Model",
    comment: str = "",
    **kwargs
) -> TensorBoardExporter:
    """
    Factory function to create TensorBoard exporter.
    
    Args:
        log_dir: Base directory for logs
        model_name: Name of the model
        comment: Additional comment
        **kwargs: Additional arguments for TensorBoardExporter
    
    Returns:
        TensorBoardExporter instance
    """
    return TensorBoardExporter(
        log_dir=log_dir,
        model_name=model_name,
        comment=comment,
        **kwargs
    )


# CLI utility for launching TensorBoard
def launch_tensorboard(log_dir: Union[str, Path] = "runs", port: int = 6006) -> None:
    """
    Launch TensorBoard server.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port for TensorBoard server
    """
    import subprocess
    import webbrowser
    from time import sleep
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory does not exist: {log_dir}")
        return
    
    try:
        # Launch TensorBoard
        cmd = f"tensorboard --logdir {log_dir} --port {port}"
        logger.info(f"Launching TensorBoard: {cmd}")
        
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        sleep(3)
        
        # Open browser
        url = f"http://localhost:{port}"
        logger.info(f"Opening TensorBoard at: {url}")
        webbrowser.open(url)
        
        logger.info("TensorBoard launched successfully!")
        logger.info("Press Ctrl+C to stop the server")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Stopping TensorBoard...")
        process.terminate()
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorBoard utilities")
    parser.add_argument("--launch", action="store_true", help="Launch TensorBoard server")
    parser.add_argument("--log-dir", default="runs", help="Log directory")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    
    args = parser.parse_args()
    
    if args.launch:
        launch_tensorboard(args.log_dir, args.port)
    else:
        print("TensorBoard exporter module loaded successfully!")
        print("Use --launch to start TensorBoard server")