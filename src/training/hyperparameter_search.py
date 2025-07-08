# src/training/hyperparameter_search.py
"""
Distributed Hyperparameter Search with Ray Tune and GPU Support.

This module provides scalable hyperparameter optimization that can run on:
- Multiple GPUs (local or distributed)
- Ray cluster with spare nodes
- Fallback to improved CPU parallelization

Replaces the 14-line Optuna study with concurrency=1 limitation.
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import tempfile
import pickle

# Ray Tune imports (with fallback)
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.suggest.optuna import OptunaSearch
    from ray.tune.suggest.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray Tune not available. Install with: pip install ray[tune] optuna hyperopt")

# Optuna fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_COUNT = 0

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class DistributedHyperparameterSearch:
    """
    Distributed hyperparameter search with GPU support and Ray Tune integration.
    
    Features:
    - Multi-GPU training support
    - Ray cluster distribution
    - Advanced schedulers (ASHA, PBT)
    - Multiple search algorithms (Optuna, HyperOpt)
    - Resource-aware scheduling
    - Fault tolerance and checkpointing
    """
    
    def __init__(
        self,
        search_space: Dict[str, Any],
        metric_name: str = "episode_reward_mean",
        mode: str = "max",
        max_concurrent_trials: Optional[int] = None,
        use_gpu: bool = True,
        ray_address: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize distributed hyperparameter search.
        
        Args:
            search_space: Ray Tune search space configuration
            metric_name: Metric to optimize (e.g., "episode_reward_mean", "risk_adjusted_return")
            mode: Optimization mode ("max" or "min")
            max_concurrent_trials: Maximum concurrent trials (auto-detected if None)
            use_gpu: Whether to use GPU acceleration
            ray_address: Ray cluster address (None for local)
            storage_path: Path for storing results and checkpoints
        """
        self.search_space = search_space
        self.metric_name = metric_name
        self.mode = mode
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.ray_address = ray_address
        
        # Auto-detect resources
        if max_concurrent_trials is None:
            if self.use_gpu:
                max_concurrent_trials = GPU_COUNT
            else:
                max_concurrent_trials = min(os.cpu_count() // 2, 8)  # Conservative CPU usage
        
        self.max_concurrent_trials = max_concurrent_trials
        
        # Setup storage
        if storage_path is None:
            storage_path = f"./hyperparameter_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray if available
        self.ray_initialized = False
        if RAY_AVAILABLE:
            self._initialize_ray()
        
        logger.info(f"Hyperparameter search initialized:")
        logger.info(f"  GPU available: {GPU_AVAILABLE} (count: {GPU_COUNT})")
        logger.info(f"  Ray available: {RAY_AVAILABLE}")
        logger.info(f"  Max concurrent trials: {max_concurrent_trials}")
        logger.info(f"  Storage path: {self.storage_path}")
    
    def _initialize_ray(self):
        """Initialize Ray cluster connection."""
        try:
            if self.ray_address:
                # Connect to existing Ray cluster
                ray.init(address=self.ray_address, ignore_reinit_error=True)
                logger.info(f"Connected to Ray cluster at {self.ray_address}")
            else:
                # Start local Ray cluster
                ray.init(
                    num_cpus=os.cpu_count(),
                    num_gpus=GPU_COUNT if self.use_gpu else 0,
                    ignore_reinit_error=True
                )
                logger.info("Started local Ray cluster")
            
            self.ray_initialized = True
            
            # Log cluster resources
            resources = ray.cluster_resources()
            logger.info(f"Ray cluster resources: {resources}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}")
            self.ray_initialized = False
    
    def create_training_function(self, trainer_class, base_config: Dict[str, Any]):
        """
        Create a Ray Tune training function.
        
        Args:
            trainer_class: TrainerAgent class or similar
            base_config: Base configuration to merge with hyperparameters
            
        Returns:
            Training function compatible with Ray Tune
        """
        def train_model(config: Dict[str, Any]):
            """Training function for Ray Tune."""
            import tempfile
            import shutil
            from pathlib import Path
            
            # Merge base config with hyperparameters
            merged_config = {**base_config, **config}
            
            # Create temporary directory for this trial
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                try:
                    # Initialize trainer with merged config
                    trainer = trainer_class(
                        config=merged_config,
                        training_env=None,  # Will be created by trainer
                        model_save_path=str(temp_path / "model"),
                        log_dir=str(temp_path / "logs")
                    )
                    
                    # Train model
                    model_path = trainer.train()
                    
                    # Get training metrics
                    metrics = trainer.get_training_metrics()
                    
                    # Report metrics to Ray Tune
                    if metrics:
                        # Report primary metric
                        primary_metric = metrics.get(self.metric_name, 0.0)
                        tune.report(**{self.metric_name: primary_metric})
                        
                        # Report additional metrics
                        additional_metrics = {
                            k: v for k, v in metrics.items() 
                            if k != self.metric_name and isinstance(v, (int, float))
                        }
                        if additional_metrics:
                            tune.report(**additional_metrics)
                    else:
                        # Fallback metric
                        tune.report(**{self.metric_name: 0.0})
                
                except Exception as e:
                    logger.error(f"Training failed: {e}")
                    # Report failure
                    tune.report(**{self.metric_name: -float('inf') if self.mode == 'max' else float('inf')})
        
        return train_model
    
    def run_ray_tune_search(
        self,
        training_function: Callable,
        num_samples: int = 50,
        max_training_iterations: int = 100,
        scheduler_type: str = "asha",
        search_algorithm: str = "optuna"
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search using Ray Tune.
        
        Args:
            training_function: Function to train model with given config
            num_samples: Number of hyperparameter combinations to try
            max_training_iterations: Maximum training iterations per trial
            scheduler_type: Scheduler type ("asha", "pbt", "fifo")
            search_algorithm: Search algorithm ("optuna", "hyperopt", "random")
            
        Returns:
            Dictionary with best configuration and results
        """
        if not self.ray_initialized:
            raise RuntimeError("Ray not initialized. Cannot run Ray Tune search.")
        
        # Configure scheduler
        if scheduler_type == "asha":
            scheduler = ASHAScheduler(
                metric=self.metric_name,
                mode=self.mode,
                max_t=max_training_iterations,
                grace_period=10,
                reduction_factor=2
            )
        elif scheduler_type == "pbt":
            scheduler = PopulationBasedTraining(
                metric=self.metric_name,
                mode=self.mode,
                perturbation_interval=20,
                hyperparam_mutations={
                    # Define mutation ranges for PBT
                    "learning_rate": lambda: np.random.uniform(1e-5, 1e-2),
                    "batch_size": [32, 64, 128, 256],
                }
            )
        else:  # fifo
            scheduler = None
        
        # Configure search algorithm
        if search_algorithm == "optuna" and OPTUNA_AVAILABLE:
            search_alg = OptunaSearch(
                metric=self.metric_name,
                mode=self.mode,
                sampler=TPESampler(seed=42)
            )
        elif search_algorithm == "hyperopt":
            search_alg = HyperOptSearch(
                metric=self.metric_name,
                mode=self.mode,
                random_state_seed=42
            )
        else:
            search_alg = None  # Random search
        
        # Configure reporter
        reporter = CLIReporter(
            metric_columns=[self.metric_name, "training_iteration"],
            max_progress_rows=10,
            max_error_rows=5
        )
        
        # Configure resources per trial
        resources_per_trial = {
            "cpu": max(1, os.cpu_count() // self.max_concurrent_trials),
            "gpu": 1 if self.use_gpu else 0
        }
        
        # Run hyperparameter search
        logger.info(f"Starting Ray Tune search with {num_samples} trials")
        logger.info(f"Resources per trial: {resources_per_trial}")
        logger.info(f"Scheduler: {scheduler_type}, Search algorithm: {search_algorithm}")
        
        analysis = tune.run(
            training_function,
            config=self.search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=reporter,
            resources_per_trial=resources_per_trial,
            local_dir=str(self.storage_path),
            name="hyperparameter_search",
            stop={"training_iteration": max_training_iterations},
            checkpoint_freq=10,
            keep_checkpoints_num=3,
            verbose=1
        )
        
        # Get best results
        best_trial = analysis.get_best_trial(self.metric_name, self.mode)
        best_config = best_trial.config
        best_result = best_trial.last_result
        
        results = {
            "best_config": best_config,
            "best_result": best_result,
            "best_metric_value": best_result[self.metric_name],
            "analysis": analysis,
            "total_trials": len(analysis.trials),
            "successful_trials": len([t for t in analysis.trials if t.status == "TERMINATED"])
        }
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Hyperparameter search completed!")
        logger.info(f"Best {self.metric_name}: {results['best_metric_value']:.4f}")
        logger.info(f"Best config: {best_config}")
        
        return results
    
    def run_optuna_search(
        self,
        objective_function: Callable,
        num_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Fallback hyperparameter search using Optuna (improved from original 14-line version).
        
        Args:
            objective_function: Function that takes trial and returns metric value
            num_trials: Number of trials to run
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs (improved from concurrency=1)
            
        Returns:
            Dictionary with best configuration and results
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available. Install with: pip install optuna")
        
        # Create study with improved configuration
        study = optuna.create_study(
            direction="maximize" if self.mode == "max" else "minimize",
            sampler=TPESampler(seed=42, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=f"hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Starting Optuna search with {num_trials} trials")
        logger.info(f"Parallel jobs: {n_jobs} (improved from concurrency=1)")
        
        # Run optimization with parallel execution
        study.optimize(
            objective_function,
            n_trials=num_trials,
            timeout=timeout,
            n_jobs=n_jobs,  # This is the key improvement from concurrency=1
            show_progress_bar=True
        )
        
        # Get results
        best_trial = study.best_trial
        results = {
            "best_config": best_trial.params,
            "best_metric_value": best_trial.value,
            "study": study,
            "total_trials": len(study.trials),
            "successful_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Optuna search completed!")
        logger.info(f"Best {self.metric_name}: {results['best_metric_value']:.4f}")
        logger.info(f"Best config: {results['best_config']}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save search results to disk."""
        # Save JSON summary
        json_results = {
            "best_config": results["best_config"],
            "best_metric_value": float(results["best_metric_value"]),
            "total_trials": results["total_trials"],
            "successful_trials": results["successful_trials"],
            "timestamp": datetime.now().isoformat(),
            "search_space": self.search_space,
            "metric_name": self.metric_name,
            "mode": self.mode
        }
        
        json_path = self.storage_path / "best_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save full results (pickle)
        pickle_path = self.storage_path / "full_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {self.storage_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.ray_initialized:
            try:
                ray.shutdown()
                logger.info("Ray cluster shut down")
            except Exception as e:
                logger.warning(f"Error shutting down Ray: {e}")


def create_dqn_search_space() -> Dict[str, Any]:
    """Create search space for DQN hyperparameters."""
    return {
        # Learning parameters
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "buffer_size": tune.choice([10000, 50000, 100000, 200000]),
        "learning_starts": tune.choice([1000, 5000, 10000]),
        "target_update_interval": tune.choice([1000, 5000, 10000]),
        
        # Network architecture
        "net_arch": tune.choice([
            [64, 64],
            [128, 128],
            [256, 256],
            [128, 64],
            [256, 128, 64]
        ]),
        
        # Exploration
        "exploration_fraction": tune.uniform(0.1, 0.3),
        "exploration_final_eps": tune.uniform(0.01, 0.1),
        
        # Training parameters
        "train_freq": tune.choice([1, 4, 8]),
        "gradient_steps": tune.choice([1, 2, 4]),
        "gamma": tune.uniform(0.95, 0.999),
        "tau": tune.uniform(0.001, 0.01),
        
        # Risk management (if using enhanced risk callback)
        "risk_config": {
            "early_stop_threshold": tune.uniform(0.6, 0.9),
            "liquidity_penalty_multiplier": tune.uniform(1.5, 5.0),
            "risk_weights": {
                "drawdown_pct": tune.uniform(0.2, 0.4),
                "ulcer_index": tune.uniform(0.15, 0.35),
                "kyle_lambda": tune.uniform(0.25, 0.5),
                "feed_staleness": tune.uniform(0.1, 0.25)
            }
        }
    }


def create_ppo_search_space() -> Dict[str, Any]:
    """Create search space for PPO hyperparameters."""
    return {
        # Learning parameters
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "n_steps": tune.choice([1024, 2048, 4096]),
        "n_epochs": tune.choice([3, 5, 10, 20]),
        
        # PPO specific
        "clip_range": tune.uniform(0.1, 0.3),
        "clip_range_vf": tune.choice([None, 0.1, 0.2, 0.3]),
        "ent_coef": tune.loguniform(1e-8, 1e-1),
        "vf_coef": tune.uniform(0.1, 1.0),
        "max_grad_norm": tune.uniform(0.3, 2.0),
        
        # Network architecture
        "net_arch": tune.choice([
            [64, 64],
            [128, 128],
            [256, 256],
            [128, 64],
            [256, 128, 64]
        ]),
        
        # Training parameters
        "gamma": tune.uniform(0.95, 0.999),
        "gae_lambda": tune.uniform(0.9, 0.99),
        
        # Risk management
        "risk_config": {
            "early_stop_threshold": tune.uniform(0.6, 0.9),
            "liquidity_penalty_multiplier": tune.uniform(1.5, 5.0)
        }
    }


def example_training_function(config: Dict[str, Any]):
    """
    Example training function for hyperparameter search.
    Replace this with your actual TrainerAgent integration.
    """
    # This would be replaced with actual TrainerAgent
    from src.training.trainer_agent import TrainerAgent
    
    # Simulate training (replace with actual training)
    import time
    import random
    
    # Simulate variable training time
    time.sleep(random.uniform(1, 5))
    
    # Simulate training results
    episode_reward_mean = random.uniform(0, 100) * (1 - config.get("learning_rate", 0.001))
    risk_adjusted_return = episode_reward_mean * (1 - config.get("risk_config", {}).get("early_stop_threshold", 0.7))
    
    # Report metrics
    tune.report(
        episode_reward_mean=episode_reward_mean,
        risk_adjusted_return=risk_adjusted_return,
        training_iteration=1
    )


def main():
    """Example usage of distributed hyperparameter search."""
    print("🚀 Distributed Hyperparameter Search")
    print("Solving: Hyper-param search runs on laptop CPU with concurrency=1")
    print("Solution: Ray Tune with GPU support and distributed execution")
    print()
    
    # Create search space
    search_space = create_dqn_search_space()
    
    # Initialize distributed search
    search = DistributedHyperparameterSearch(
        search_space=search_space,
        metric_name="episode_reward_mean",
        mode="max",
        use_gpu=True,  # Use GPU if available
        max_concurrent_trials=None  # Auto-detect
    )
    
    if RAY_AVAILABLE and search.ray_initialized:
        print("✅ Using Ray Tune for distributed search")
        
        # Create training function
        training_fn = search.create_training_function(
            trainer_class=None,  # Would be TrainerAgent
            base_config={"algorithm": "DQN", "total_timesteps": 100000}
        )
        
        # Run search
        results = search.run_ray_tune_search(
            training_function=example_training_function,  # Replace with training_fn
            num_samples=20,
            scheduler_type="asha",
            search_algorithm="optuna"
        )
        
    elif OPTUNA_AVAILABLE:
        print("✅ Using improved Optuna search (parallel execution)")
        
        def objective(trial):
            # Sample hyperparameters
            config = {}
            for key, space in search_space.items():
                if key == "learning_rate":
                    config[key] = trial.suggest_float(key, 1e-5, 1e-2, log=True)
                elif key == "batch_size":
                    config[key] = trial.suggest_categorical(key, [32, 64, 128, 256])
                # Add more parameter sampling as needed
            
            # Simulate training
            return random.uniform(0, 100)
        
        results = search.run_optuna_search(
            objective_function=objective,
            num_trials=20,
            n_jobs=min(4, os.cpu_count())  # Parallel execution
        )
    
    else:
        print("❌ Neither Ray Tune nor Optuna available")
        print("Install with: pip install ray[tune] optuna hyperopt")
        return
    
    # Cleanup
    search.cleanup()
    
    print("\n🎯 Key Improvements Achieved:")
    print("✅ Multi-GPU support for parallel training")
    print("✅ Ray cluster distribution across spare nodes")
    print("✅ Advanced schedulers (ASHA, PBT) for efficient search")
    print("✅ Multiple search algorithms (Optuna, HyperOpt)")
    print("✅ Improved Optuna with n_jobs > 1 (vs concurrency=1)")
    print("✅ Resource-aware scheduling and fault tolerance")
    print()
    print("🏆 Hyperparameter search is now scalable and GPU-accelerated!")


if __name__ == "__main__":
    main()