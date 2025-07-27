# src/training/model_performance_validator.py
"""
Model Performance Validation System

Comprehensive benchmarking and validation for dual-ticker models.
Validates transfer learning success and production readiness.
"""

import time
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from stable_baselines3 import RecurrentPPO

try:
    from ..gym_env.dual_ticker_trading_env import DualTickerTradingEnv
    from ..gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
    from .dual_ticker_model_adapter import DualTickerModelAdapter
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from gym_env.dual_ticker_trading_env import DualTickerTradingEnv
    from gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
    from training.dual_ticker_model_adapter import DualTickerModelAdapter


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    steps_per_second: float
    episode_reward_mean: float
    episode_reward_std: float
    episode_length_mean: float
    sharpe_ratio: float
    max_drawdown: float
    turnover_ratio: float
    prediction_latency_ms: float
    memory_usage_mb: float
    gpu_utilization_pct: float


class ModelPerformanceValidator:
    """
    Comprehensive model performance validation system
    
    Key Features:
    - 🔧 Performance benchmarking (>100 steps/sec requirement)
    - 🔧 Transfer learning validation (3→9 actions, 13→26 obs)
    - 🔧 Production readiness assessment
    - 🔧 SLA compliance checking
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Performance SLA thresholds (Day 2 requirements)
        self.sla_thresholds = {
            'min_steps_per_sec': 100,        # Day 2 target: >100 steps/sec
            'max_prediction_latency_ms': 10, # <10ms prediction latency
            'min_episode_reward': 3.0,       # Minimum viable reward
            'max_drawdown_threshold': 0.10,  # 10% max drawdown
            'max_memory_usage_mb': 2048,     # 2GB memory limit
            'min_model_load_time_s': 5.0     # <5s model loading
        }
        
        self.logger.info("🔧 Performance validator initialized")
        self.logger.info(f"📊 SLA thresholds: {self.sla_thresholds}")
    
    def validate_model_adaptation(self, 
                                model: RecurrentPPO,
                                test_env: DualTickerTradingEnv,
                                num_episodes: int = 10) -> Dict[str, Any]:
        """
        Validate successful model adaptation from single-ticker to dual-ticker
        
        Args:
            model: Adapted RecurrentPPO model
            test_env: Dual-ticker test environment
            num_episodes: Number of episodes for validation
            
        Returns:
            Validation results and performance metrics
        """
        
        self.logger.info(f"🔧 Starting model adaptation validation ({num_episodes} episodes)...")
        
        # Performance tracking
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        prediction_times = []
        
        # Memory tracking
        initial_memory = self._get_memory_usage()
        
        try:
            for episode in range(num_episodes):
                episode_reward = 0.0
                episode_length = 0
                
                obs, info = test_env.reset()
                done = False
                
                while not done:
                    # Time prediction latency
                    pred_start = time.perf_counter()
                    action, _states = model.predict(obs, deterministic=True)
                    pred_time = (time.perf_counter() - pred_start) * 1000  # Convert to ms
                    prediction_times.append(pred_time)
                    
                    # Step environment
                    obs, reward, done, truncated, info = test_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if episode % 5 == 0:
                    self.logger.info(f"  Episode {episode+1}/{num_episodes}: reward={episode_reward:.3f}")
        
            # Calculate performance metrics
            total_time = time.time() - start_time
            total_steps = sum(episode_lengths)
            steps_per_second = total_steps / total_time
            
            # Calculate financial metrics
            returns = np.array(episode_rewards)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                steps_per_second=steps_per_second,
                episode_reward_mean=np.mean(episode_rewards),
                episode_reward_std=np.std(episode_rewards),
                episode_length_mean=np.mean(episode_lengths),
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self._calculate_max_drawdown(returns),
                turnover_ratio=self._estimate_turnover(test_env, total_steps),
                prediction_latency_ms=np.mean(prediction_times),
                memory_usage_mb=self._get_memory_usage() - initial_memory,
                gpu_utilization_pct=self._get_gpu_utilization()
            )
            
            # SLA compliance check
            sla_results = self._check_sla_compliance(metrics)
            
            results = {
                'validation_status': 'SUCCESS' if sla_results['overall_pass'] else 'FAILED',
                'metrics': metrics,
                'sla_compliance': sla_results,
                'episode_details': {
                    'rewards': episode_rewards,
                    'lengths': episode_lengths,
                    'prediction_times_ms': prediction_times
                },
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'environment_config': {
                    'bar_size': test_env.bar_size,
                    'observation_shape': test_env.observation_space.shape,
                    'action_space_size': test_env.action_space.n
                }
            }
            
            self.logger.info(f"✅ Validation complete: {results['validation_status']}")
            self.logger.info(f"📊 Performance: {steps_per_second:.1f} steps/sec, "
                           f"reward={metrics.episode_reward_mean:.3f}±{metrics.episode_reward_std:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {
                'validation_status': 'ERROR',
                'error': str(e),
                'validation_timestamp': pd.Timestamp.now().isoformat()
            }
    
    def benchmark_environment_performance(self, 
                                        test_env: DualTickerTradingEnv,
                                        num_steps: int = 10000) -> Dict[str, Any]:
        """
        Benchmark pure environment performance (no model)
        
        Args:
            test_env: Environment to benchmark
            num_steps: Number of steps for benchmarking
            
        Returns:
            Environment performance metrics
        """
        
        self.logger.info(f"🔧 Benchmarking environment performance ({num_steps:,} steps)...")
        
        start_time = time.time()
        step_times = []
        reset_times = []
        
        # Benchmark environment reset
        reset_start = time.perf_counter()
        obs, info = test_env.reset()
        reset_time = (time.perf_counter() - reset_start) * 1000
        reset_times.append(reset_time)
        
        steps_completed = 0
        episodes_completed = 0
        
        try:
            while steps_completed < num_steps:
                # Random action for pure environment benchmarking
                action = test_env.action_space.sample()
                
                # Time single step
                step_start = time.perf_counter()
                obs, reward, done, truncated, info = test_env.step(action)
                step_time = (time.perf_counter() - step_start) * 1000
                step_times.append(step_time)
                
                steps_completed += 1
                
                if done or truncated:
                    episodes_completed += 1
                    # Time reset
                    reset_start = time.perf_counter()
                    obs, info = test_env.reset()
                    reset_time = (time.perf_counter() - reset_start) * 1000
                    reset_times.append(reset_time)
        
            total_time = time.time() - start_time
            steps_per_second = steps_completed / total_time
            
            results = {
                'environment_performance': {
                    'steps_per_second': steps_per_second,
                    'mean_step_time_ms': np.mean(step_times),
                    'mean_reset_time_ms': np.mean(reset_times),
                    'steps_completed': steps_completed,
                    'episodes_completed': episodes_completed,
                    'total_time_seconds': total_time
                },
                'sla_compliance': {
                    'meets_speed_requirement': steps_per_second >= self.sla_thresholds['min_steps_per_sec'],
                    'speed_requirement': self.sla_thresholds['min_steps_per_sec'],
                    'actual_speed': steps_per_second
                },
                'benchmark_timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.logger.info(f"✅ Environment benchmark: {steps_per_second:.1f} steps/sec")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Environment benchmark failed: {e}")
            return {
                'benchmark_status': 'ERROR',
                'error': str(e),
                'benchmark_timestamp': pd.Timestamp.now().isoformat()
            }
    
    def _check_sla_compliance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Check if metrics meet SLA requirements"""
        
        results = {
            'speed_check': metrics.steps_per_second >= self.sla_thresholds['min_steps_per_sec'],
            'latency_check': metrics.prediction_latency_ms <= self.sla_thresholds['max_prediction_latency_ms'],
            'reward_check': metrics.episode_reward_mean >= self.sla_thresholds['min_episode_reward'],
            'drawdown_check': metrics.max_drawdown <= self.sla_thresholds['max_drawdown_threshold'],
            'memory_check': metrics.memory_usage_mb <= self.sla_thresholds['max_memory_usage_mb']
        }
        
        results['overall_pass'] = all(results.values())
        results['passed_checks'] = sum(results.values())
        results['total_checks'] = len(results) - 2  # Exclude summary fields
        
        return results
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        return float(np.max(drawdown))
    
    def _estimate_turnover(self, env: DualTickerTradingEnv, total_steps: int) -> float:
        """Estimate turnover ratio from environment trades"""
        if hasattr(env, 'trade_log') and env.trade_log:
            # Count position changes
            position_changes = len([t for t in env.trade_log if abs(t.get('position_change', 0)) > 0])
            return position_changes / total_steps if total_steps > 0 else 0.0
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0.0
        except:
            return 0.0
    
    def save_validation_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save comprehensive validation report"""
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, PerformanceMetrics):
                return obj.__dict__
            return obj
        
        # Create detailed report
        report = {
            'validation_summary': results.get('validation_status', 'UNKNOWN'),
            'performance_metrics': results.get('metrics'),
            'sla_compliance': results.get('sla_compliance'),
            'environment_config': results.get('environment_config'),
            'validation_timestamp': results.get('validation_timestamp'),
            'validator_version': '1.0.0',
            'sla_thresholds': self.sla_thresholds
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"📋 Validation report saved: {output_path}")


if __name__ == "__main__":
    # Quick validation test
    validator = ModelPerformanceValidator()
    print("✅ Model Performance Validator initialized successfully")