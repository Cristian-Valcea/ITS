"""
Hyperparameter Search Core Module

Contains hyperparameter optimization logic extracted from TrainerAgent.
This module handles:
- Training callback creation and management
- Hyperparameter optimization with Optuna/Ray
- Search space definition and validation
- Optimization result analysis

This is an internal module - use src.training.TrainerAgent for public API.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import numpy as np

# Optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Core dependencies
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback


def create_training_callbacks(
    run_dir: Path,
    run_name: str,
    config: Dict[str, Any],
    eval_env: Optional[Any] = None,
    risk_advisor: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> List[Any]:
    """
    Create training callbacks for model training.
    
    Creates callbacks for early stopping, checkpointing, evaluation, and risk management
    to align with TrainerCore functionality during hyperparameter optimization.
    
    Args:
        run_dir: Directory for saving training artifacts
        run_name: Name of the training run
        config: Training configuration
        eval_env: Optional evaluation environment
        risk_advisor: Optional risk advisor instance
        logger: Optional logger instance
        
    Returns:
        List of configured training callbacks
    """
    logger = logger or logging.getLogger(__name__)
    
    callbacks = []
    
    # Add early stopping callback to prevent infinite loops (align with TrainerCore)
    try:
        from .early_stopping_callback import EarlyStoppingCallback
        max_episodes = config.get("max_episodes", 50)  # Reasonable default
        max_training_time = config.get("max_training_time_minutes", 30)  # 30 minute limit
        early_stopping = EarlyStoppingCallback(
            max_episodes=max_episodes,
            max_training_time_minutes=max_training_time,
            plateau_patience=10,
            verbose=1
        )
        callbacks.append(early_stopping)
        logger.info("Early stopping callback added to prevent infinite training")
    except ImportError:
        logger.warning("EarlyStoppingCallback not available, skipping early stopping")
    
    # Add checkpoint callback
    if config.get('checkpoint_freq', 0) > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"{run_name}_checkpoint"
        )
        callbacks.append(checkpoint_callback)
    
    # Add evaluation callback if eval_env provided
    if eval_env is not None:
        eval_freq = config.get('eval_freq', 1000)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    # Add risk-aware callback if risk_advisor provided
    if risk_advisor is not None:
        try:
            from .risk_callbacks import RiskAwareCallback
            risk_config = config.get('risk_config', {})
            risk_callback = RiskAwareCallback(
                risk_advisor=risk_advisor,
                penalty_weight=risk_config.get("penalty_weight", 0.1),
                early_stop_threshold=risk_config.get("early_stop_threshold", 0.8),
                log_freq=config.get("log_interval", 100)
            )
            callbacks.append(risk_callback)
            logger.info("Risk-aware callback added to training")
        except ImportError:
            logger.warning("RiskAwareCallback not available, skipping risk callback")
    
    logger.info(f"Created {len(callbacks)} training callbacks for {run_name}")
    
    return callbacks


def run_hyperparameter_study(
    objective_function: Callable,
    search_space: Dict[str, Any],
    n_trials: int = 100,
    study_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """
    Run hyperparameter optimization study using Optuna.
    
    Args:
        objective_function: Function to optimize (should accept trial as first argument)
        search_space: Dictionary defining the search space
        n_trials: Number of optimization trials
        study_name: Optional name for the study
        logger: Optional logger instance
        
    Returns:
        Dictionary with optimization results or None if failed
    """
    logger = logger or logging.getLogger(__name__)
    
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is not available. Install with: pip install optuna")
        return None
    
    try:
        study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting hyperparameter study '{study_name}' with {n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # Assuming we want to maximize the objective
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)  # Reproducible results
        )
        
        # Define objective wrapper that uses the search space
        def optuna_objective(trial):
            # Sample parameters from search space
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                elif param_config['type'] == 'choice':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                else:
                    logger.warning(f"Unknown parameter type: {param_config['type']}")
            
            # Call the objective function with sampled parameters
            return objective_function(trial, params)
        
        # Run optimization
        study.optimize(optuna_objective, n_trials=n_trials)
        
        # Compile results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'completed_at': datetime.now().isoformat(),
            'trials_data': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        logger.info(f"Hyperparameter study completed. Best value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return results
        
    except Exception as e:
        logger.error(f"Hyperparameter study failed: {e}")
        return None


def define_search_space(
    algorithm: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Define hyperparameter search space for a given algorithm.
    
    Args:
        algorithm: RL algorithm name (e.g., 'DQN', 'PPO')
        config: Base configuration
        logger: Optional logger instance
        
    Returns:
        Dictionary defining the search space
    """
    logger = logger or logging.getLogger(__name__)
    
    search_spaces = {
        'DQN': {
            'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'buffer_size': {'type': 'choice', 'choices': [10000, 50000, 100000]},
            'learning_starts': {'type': 'choice', 'choices': [1000, 5000, 10000]},
            'batch_size': {'type': 'choice', 'choices': [32, 64, 128, 256]},
            'tau': {'type': 'uniform', 'low': 0.001, 'high': 0.1},
            'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999},
            'exploration_fraction': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'exploration_final_eps': {'type': 'uniform', 'low': 0.01, 'high': 0.1}
        },
        'PPO': {
            'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'n_steps': {'type': 'choice', 'choices': [128, 256, 512, 1024, 2048]},
            'batch_size': {'type': 'choice', 'choices': [32, 64, 128, 256]},
            'n_epochs': {'type': 'choice', 'choices': [3, 5, 10, 20]},
            'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999},
            'gae_lambda': {'type': 'uniform', 'low': 0.8, 'high': 1.0},
            'clip_range': {'type': 'uniform', 'low': 0.1, 'high': 0.4},
            'ent_coef': {'type': 'loguniform', 'low': 1e-8, 'high': 1e-1}
        }
    }
    
    if algorithm not in search_spaces:
        logger.warning(f"No predefined search space for algorithm: {algorithm}")
        return {}
        
    search_space = search_spaces[algorithm]
    logger.info(f"Defined search space for {algorithm} with {len(search_space)} parameters")
    
    return search_space


def validate_hyperparameters(
    params: Dict[str, Any],
    algorithm: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate hyperparameter values for a given algorithm.
    
    Args:
        params: Hyperparameter dictionary to validate
        algorithm: RL algorithm name
        logger: Optional logger instance
        
    Returns:
        True if parameters are valid, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    # Common validations
    if 'learning_rate' in params:
        lr = params['learning_rate']
        if not (1e-6 <= lr <= 1.0):
            logger.error(f"Invalid learning rate: {lr}")
            return False
            
    if 'gamma' in params:
        gamma = params['gamma']
        if not (0.0 <= gamma <= 1.0):
            logger.error(f"Invalid gamma: {gamma}")
            return False
            
    if 'batch_size' in params:
        batch_size = params['batch_size']
        if batch_size <= 0 or not isinstance(batch_size, int):
            logger.error(f"Invalid batch size: {batch_size}")
            return False
            
    # Algorithm-specific validations
    if algorithm == 'DQN':
        if 'buffer_size' in params and params['buffer_size'] <= 0:
            logger.error(f"Invalid buffer size: {params['buffer_size']}")
            return False
            
        if 'tau' in params and not (0.0 <= params['tau'] <= 1.0):
            logger.error(f"Invalid tau: {params['tau']}")
            return False
            
    elif algorithm == 'PPO':
        if 'n_steps' in params and params['n_steps'] <= 0:
            logger.error(f"Invalid n_steps: {params['n_steps']}")
            return False
            
        if 'clip_range' in params and not (0.0 <= params['clip_range'] <= 1.0):
            logger.error(f"Invalid clip_range: {params['clip_range']}")
            return False
            
    logger.info(f"Hyperparameter validation passed for {algorithm}")
    return True


def analyze_optimization_results(
    results: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Analyze hyperparameter optimization results.
    
    Args:
        results: Optimization results dictionary
        logger: Optional logger instance
        
    Returns:
        Dictionary with analysis results
    """
    logger = logger or logging.getLogger(__name__)
    
    analysis = {
        'best_params': results.get('best_params', {}),
        'best_value': results.get('best_value', 0.0),
        'n_trials': results.get('n_trials', 0),
        'improvement_over_default': 0.0,
        'parameter_importance': {},
        'convergence_analysis': {}
    }
    
    # TODO: Add more sophisticated analysis
    # - Parameter importance analysis
    # - Convergence analysis
    # - Statistical significance tests
    
    logger.info(f"Optimization analysis completed for {analysis['n_trials']} trials")
    return analysis


def save_optimization_results(
    results: Dict[str, Any],
    save_path: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Save hyperparameter optimization results to file.
    
    Args:
        results: Results dictionary to save
        save_path: Path to save the results
        logger: Optional logger instance
        
    Returns:
        True if saved successfully, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        import json
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp if not present
        if 'saved_at' not in results:
            results['saved_at'] = datetime.now().isoformat()
            
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Optimization results saved to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")
        return False