# examples/distributed_hyperparameter_search_example.py
"""
Distributed Hyperparameter Search Example.

This example demonstrates how to replace the 14-line Optuna study with concurrency=1
with a scalable Ray Tune solution that can leverage GPUs and spare nodes.

Usage:
    # Local GPU search
    python examples/distributed_hyperparameter_search_example.py --mode local_gpu
    
    # Ray cluster search
    python examples/distributed_hyperparameter_search_example.py --mode ray_cluster --ray_address ray://head-node:10001
    
    # Improved Optuna (parallel)
    python examples/distributed_hyperparameter_search_example.py --mode optuna_parallel
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.hyperparameter_search import (
    DistributedHyperparameterSearch,
    create_dqn_search_space,
    create_ppo_search_space
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_trainer_agent():
    """Create a mock TrainerAgent for demonstration."""
    
    class MockTrainerAgent:
        """Mock TrainerAgent that simulates training with different hyperparameters."""
        
        def __init__(self, config: Dict[str, Any], training_env=None, model_save_path=None, log_dir=None):
            self.config = config
            self.model_save_path = model_save_path
            self.log_dir = log_dir
            self.training_metrics = {}
            
        def train(self) -> str:
            """Simulate training process."""
            import time
            import random
            import numpy as np
            
            # Simulate training time based on complexity
            batch_size = self.config.get('batch_size', 64)
            learning_rate = self.config.get('learning_rate', 0.001)
            net_arch = self.config.get('net_arch', [64, 64])
            
            # Simulate training time (larger networks take longer)
            complexity_factor = sum(net_arch) / 128.0
            training_time = random.uniform(2, 8) * complexity_factor
            time.sleep(min(training_time, 10))  # Cap at 10 seconds for demo
            
            # Simulate performance based on hyperparameters
            # Better hyperparameters lead to better performance
            base_performance = random.uniform(50, 100)
            
            # Learning rate effect (too high or too low is bad)
            lr_penalty = abs(np.log10(learning_rate) + 3) / 2  # Optimal around 1e-3
            lr_factor = max(0.5, 1.0 - lr_penalty * 0.1)
            
            # Batch size effect (moderate sizes work better)
            bs_factor = 1.0 if 64 <= batch_size <= 128 else 0.9
            
            # Network architecture effect (deeper can be better but diminishing returns)
            arch_factor = min(1.0, 0.8 + len(net_arch) * 0.1)
            
            # Risk configuration effect
            risk_config = self.config.get('risk_config', {})
            risk_threshold = risk_config.get('early_stop_threshold', 0.75)
            risk_factor = 1.0 - (risk_threshold - 0.75) * 0.5  # Balanced threshold is better
            
            # Calculate final performance
            final_performance = base_performance * lr_factor * bs_factor * arch_factor * risk_factor
            
            # Add some noise
            final_performance += random.uniform(-5, 5)
            
            # Store metrics
            self.training_metrics = {
                'episode_reward_mean': final_performance,
                'risk_adjusted_return': final_performance * risk_factor,
                'training_time': training_time,
                'final_loss': random.uniform(0.1, 1.0) / lr_factor,
                'exploration_rate': random.uniform(0.01, 0.1),
                'risk_violations': max(0, int(random.uniform(0, 10) * (1 - risk_factor)))
            }
            
            return self.model_save_path or "mock_model.pkl"
        
        def get_training_metrics(self) -> Dict[str, float]:
            """Return training metrics."""
            return self.training_metrics
    
    return MockTrainerAgent

def run_local_gpu_search():
    """Run hyperparameter search on local GPUs."""
    print("🔥 Running Local GPU Hyperparameter Search")
    print("=" * 50)
    
    # Create search space
    search_space = create_dqn_search_space()
    
    # Initialize search with GPU support
    search = DistributedHyperparameterSearch(
        search_space=search_space,
        metric_name="episode_reward_mean",
        mode="max",
        use_gpu=True,
        max_concurrent_trials=None  # Auto-detect based on GPU count
    )
    
    # Create training function
    trainer_class = create_mock_trainer_agent()
    base_config = {
        "algorithm": "DQN",
        "total_timesteps": 100000,
        "verbose": 1
    }
    
    training_fn = search.create_training_function(trainer_class, base_config)
    
    # Run search
    results = search.run_ray_tune_search(
        training_function=training_fn,
        num_samples=15,  # Reduced for demo
        max_training_iterations=50,
        scheduler_type="asha",
        search_algorithm="optuna"
    )
    
    print(f"\n🏆 Best Configuration Found:")
    print(f"  Reward: {results['best_metric_value']:.2f}")
    print(f"  Config: {results['best_config']}")
    print(f"  Trials: {results['successful_trials']}/{results['total_trials']}")
    
    search.cleanup()
    return results

def run_ray_cluster_search(ray_address: str):
    """Run hyperparameter search on Ray cluster."""
    print(f"🌐 Running Ray Cluster Hyperparameter Search")
    print(f"Cluster address: {ray_address}")
    print("=" * 50)
    
    # Create search space
    search_space = create_ppo_search_space()  # Try PPO for variety
    
    # Initialize search with Ray cluster
    search = DistributedHyperparameterSearch(
        search_space=search_space,
        metric_name="risk_adjusted_return",  # Optimize for risk-adjusted returns
        mode="max",
        use_gpu=True,
        ray_address=ray_address,
        max_concurrent_trials=8  # Scale based on cluster size
    )
    
    # Create training function
    trainer_class = create_mock_trainer_agent()
    base_config = {
        "algorithm": "PPO",
        "total_timesteps": 200000,
        "verbose": 1
    }
    
    training_fn = search.create_training_function(trainer_class, base_config)
    
    # Run search with Population Based Training
    results = search.run_ray_tune_search(
        training_function=training_fn,
        num_samples=25,
        max_training_iterations=100,
        scheduler_type="pbt",  # Population Based Training for cluster
        search_algorithm="hyperopt"
    )
    
    print(f"\n🏆 Best Configuration Found:")
    print(f"  Risk-Adjusted Return: {results['best_metric_value']:.2f}")
    print(f"  Config: {results['best_config']}")
    print(f"  Trials: {results['successful_trials']}/{results['total_trials']}")
    
    search.cleanup()
    return results

def run_improved_optuna_search():
    """Run improved Optuna search with parallel execution."""
    print("⚡ Running Improved Optuna Search (Parallel)")
    print("Improvement: n_jobs > 1 (vs original concurrency=1)")
    print("=" * 50)
    
    # Create search space
    search_space = create_dqn_search_space()
    
    # Initialize search
    search = DistributedHyperparameterSearch(
        search_space=search_space,
        metric_name="episode_reward_mean",
        mode="max",
        use_gpu=False  # Optuna fallback doesn't use GPU directly
    )
    
    # Create objective function
    trainer_class = create_mock_trainer_agent()
    
    def objective(trial):
        """Optuna objective function."""
        # Sample hyperparameters from search space
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
            "net_arch": trial.suggest_categorical("net_arch", [
                [64, 64], [128, 128], [256, 256], [128, 64], [256, 128, 64]
            ]),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.3),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "risk_config": {
                "early_stop_threshold": trial.suggest_float("early_stop_threshold", 0.6, 0.9),
                "liquidity_penalty_multiplier": trial.suggest_float("liquidity_penalty_multiplier", 1.5, 5.0)
            }
        }
        
        # Train model
        trainer = trainer_class(config)
        trainer.train()
        metrics = trainer.get_training_metrics()
        
        return metrics.get("episode_reward_mean", 0.0)
    
    # Run parallel Optuna search
    results = search.run_optuna_search(
        objective_function=objective,
        num_trials=20,
        n_jobs=min(4, os.cpu_count())  # Key improvement: parallel execution
    )
    
    print(f"\n🏆 Best Configuration Found:")
    print(f"  Reward: {results['best_metric_value']:.2f}")
    print(f"  Config: {results['best_config']}")
    print(f"  Trials: {results['successful_trials']}/{results['total_trials']}")
    
    return results

def compare_search_methods():
    """Compare different search methods."""
    print("📊 Hyperparameter Search Method Comparison")
    print("=" * 60)
    
    methods = {
        "Original (14-line Optuna)": {
            "concurrency": 1,
            "gpu_support": False,
            "distributed": False,
            "advanced_schedulers": False,
            "estimated_time": "Very Slow"
        },
        "Improved Optuna": {
            "concurrency": "n_jobs > 1",
            "gpu_support": False,
            "distributed": False,
            "advanced_schedulers": False,
            "estimated_time": "Moderate"
        },
        "Ray Tune Local GPU": {
            "concurrency": "GPU count",
            "gpu_support": True,
            "distributed": False,
            "advanced_schedulers": True,
            "estimated_time": "Fast"
        },
        "Ray Tune Cluster": {
            "concurrency": "Cluster size",
            "gpu_support": True,
            "distributed": True,
            "advanced_schedulers": True,
            "estimated_time": "Very Fast"
        }
    }
    
    for method, features in methods.items():
        print(f"\n{method}:")
        for feature, value in features.items():
            status = "✅" if value not in [False, 1, "Very Slow"] else "❌" if value is False else "⚠️"
            print(f"  {status} {feature}: {value}")
    
    print("\n💡 Recommendations:")
    print("• Local development: Use Ray Tune Local GPU")
    print("• Production training: Use Ray Tune Cluster")
    print("• Limited resources: Use Improved Optuna (n_jobs > 1)")
    print("• Quick experiments: Use Ray Tune with ASHA scheduler")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Distributed Hyperparameter Search Example")
    parser.add_argument(
        "--mode",
        choices=["local_gpu", "ray_cluster", "optuna_parallel", "compare"],
        default="compare",
        help="Search mode to run"
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        help="Ray cluster address (e.g., ray://head-node:10001)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Distributed Hyperparameter Search")
    print("Solving: Hyper-param search runs on laptop CPU with concurrency=1")
    print("Solution: Ray Tune with GPU support and distributed execution")
    print()
    
    try:
        if args.mode == "local_gpu":
            results = run_local_gpu_search()
            
        elif args.mode == "ray_cluster":
            if not args.ray_address:
                print("❌ Ray cluster address required for cluster mode")
                print("Example: --ray_address ray://head-node:10001")
                return
            results = run_ray_cluster_search(args.ray_address)
            
        elif args.mode == "optuna_parallel":
            results = run_improved_optuna_search()
            
        elif args.mode == "compare":
            compare_search_methods()
            return
        
        print("\n🎯 Key Improvements Achieved:")
        print("✅ Replaced concurrency=1 with parallel/distributed execution")
        print("✅ Added GPU acceleration for training")
        print("✅ Implemented advanced schedulers (ASHA, PBT)")
        print("✅ Added Ray cluster support for spare nodes")
        print("✅ Improved resource utilization and fault tolerance")
        print()
        print("🏆 Hyperparameter search is now scalable and efficient!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Search interrupted by user")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()