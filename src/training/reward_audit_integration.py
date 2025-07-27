# src/training/reward_audit_integration.py
"""
Integration example for Reward-P&L Audit System with IntradayJules training pipeline.

This module shows how to integrate the RewardPnLAudit callback with the existing
training infrastructure to ensure reward-P&L alignment during model training.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reward_pnl_audit import RewardPnLAudit
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

logger = logging.getLogger(__name__)


def _build_default_paths(model_save_path: str) -> Dict[str, str]:
    """
    Build default sub-folder paths for callbacks.
    
    Args:
        model_save_path: Base path for saving models and results
    
    Returns:
        Dict containing default paths for each callback type
    """
    return {
        'reward_audit': f"{model_save_path}/reward_audit",
        'checkpoints': f"{model_save_path}/checkpoints", 
        'best_model': f"{model_save_path}/best_model",
        'eval_logs': f"{model_save_path}/eval_logs"
    }


def create_comprehensive_callback_list(
    model_save_path: str,
    eval_env=None,
    audit_config: Optional[Dict[str, Any]] = None,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    eval_config: Optional[Dict[str, Any]] = None,
    callback_order: str = "audit_first"
) -> CallbackList:
    """
    Create a comprehensive callback list including reward-P&L audit.
    
    Args:
        model_save_path: Base path for saving models and results
        eval_env: Environment for evaluation callback
        audit_config: Configuration for reward-P&L audit
        checkpoint_config: Configuration for model checkpointing
        eval_config: Configuration for evaluation callback
        callback_order: Order of callbacks - "audit_first" (sees all steps including eval) 
                       or "audit_last" (only sees training steps)
    
    Returns:
        CallbackList: Combined callbacks for training
    """
    callbacks = []
    
    # Build default paths once
    default_paths = _build_default_paths(model_save_path)
    
    # Default configurations with default paths
    audit_config = audit_config or {}
    checkpoint_config = checkpoint_config or {}
    eval_config = eval_config or {}
    
    # 1. Reward-P&L Audit Callback (CRITICAL for production readiness)
    audit_callback = RewardPnLAudit(
        output_dir=audit_config.get('output_dir', default_paths['reward_audit']),
        min_correlation_threshold=audit_config.get('min_correlation_threshold', 0.6),
        alert_episodes=audit_config.get('alert_episodes', 10),
        save_plots=audit_config.get('save_plots', True),
        verbose=audit_config.get('verbose', True),
        fail_fast=audit_config.get('fail_fast', False)
    )
    
    # 2. Model Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_config.get('save_freq', 10000),
        save_path=checkpoint_config.get('save_path', default_paths['checkpoints']),
        name_prefix=checkpoint_config.get('name_prefix', 'model_checkpoint'),
        save_replay_buffer=checkpoint_config.get('save_replay_buffer', True),
        save_vecnormalize=checkpoint_config.get('save_vecnormalize', True),
        verbose=checkpoint_config.get('verbose', 1)
    )
    
    # 3. Evaluation Callback (if eval environment provided)
    eval_callback = None
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=eval_config.get('best_model_save_path', default_paths['best_model']),
            log_path=eval_config.get('log_path', default_paths['eval_logs']),
            eval_freq=eval_config.get('eval_freq', 5000),
            n_eval_episodes=eval_config.get('n_eval_episodes', 10),
            deterministic=eval_config.get('deterministic', True),
            render=eval_config.get('render', False),
            verbose=eval_config.get('verbose', 1)
        )
    
    # Order callbacks based on callback_order parameter
    if callback_order == "audit_first":
        # Audit sees all steps including evaluation rollouts
        callbacks.append(audit_callback)
        callbacks.append(checkpoint_callback)
        if eval_callback is not None:
            callbacks.append(eval_callback)
    elif callback_order == "audit_last":
        # Audit only sees training steps, not evaluation rollouts
        callbacks.append(checkpoint_callback)
        if eval_callback is not None:
            callbacks.append(eval_callback)
        callbacks.append(audit_callback)
    else:
        raise ValueError(f"Invalid callback_order: {callback_order}. Must be 'audit_first' or 'audit_last'")
    
    logger.info(f"📋 Created callback list with {len(callbacks)} callbacks (order: {callback_order}):")
    for i, callback in enumerate(callbacks, 1):
        logger.info(f"  {i}. {callback.__class__.__name__}")
    
    return CallbackList(callbacks)


def enhanced_training_with_audit(
    model,
    total_timesteps: int,
    model_save_path: str,
    eval_env=None,
    audit_strict: bool = False,
    callback_order: str = "audit_first",
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced training function with comprehensive reward-P&L auditing.
    
    Args:
        model: SB3 model to train
        total_timesteps: Number of training timesteps
        model_save_path: Path to save model and audit results
        eval_env: Optional evaluation environment
        audit_strict: Whether to use strict audit settings (fail_fast=True)
                     NOTE: This takes precedence over audit_config['fail_fast']
        callback_order: Order of callbacks - "audit_first" or "audit_last"
        **kwargs: Additional arguments for callback configuration
    
    Returns:
        Dict containing training results and audit summary
    """
    logger.info("🚀 Starting enhanced training with Reward-P&L audit...")
    
    # Configure audit settings with explicit precedence rules
    audit_config = kwargs.get('audit_config', {}).copy()  # Copy to avoid modifying original
    
    # Apply audit_strict precedence rules
    if audit_strict:
        logger.info("🔒 Audit strict mode enabled - applying strict validation settings")
        strict_overrides = {
            'min_correlation_threshold': 0.7,
            'alert_episodes': 5,
            'fail_fast': True
        }
        
        # Log any overrides
        for key, strict_value in strict_overrides.items():
            if key in audit_config and audit_config[key] != strict_value:
                logger.warning(
                    f"⚠️ audit_strict=True overriding audit_config['{key}'] "
                    f"from {audit_config[key]} to {strict_value}"
                )
            audit_config[key] = strict_value
    else:
        # Apply defaults only if not specified
        audit_defaults = {
            'min_correlation_threshold': 0.5,
            'alert_episodes': 10,
            'fail_fast': False
        }
        for key, default_value in audit_defaults.items():
            audit_config.setdefault(key, default_value)
    
    # Always set these defaults regardless of strict mode
    audit_config.setdefault('save_plots', True)
    audit_config.setdefault('verbose', True)
    
    # Build default paths
    default_paths = _build_default_paths(model_save_path)
    audit_config.setdefault('output_dir', default_paths['reward_audit'])
    
    # Create comprehensive callback list
    callback_list = create_comprehensive_callback_list(
        model_save_path=model_save_path,
        eval_env=eval_env,
        audit_config=audit_config,
        checkpoint_config=kwargs.get('checkpoint_config', {}),
        eval_config=kwargs.get('eval_config', {}),
        callback_order=callback_order
    )
    
    try:
        # Train the model with audit monitoring
        logger.info(f"🏋️ Training model for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            **{k: v for k, v in kwargs.items() if k not in ['audit_config', 'checkpoint_config', 'eval_config']}
        )
        
        # Save final model
        final_model_path = f"{model_save_path}/final_model"
        model.save(final_model_path)
        logger.info(f"💾 Final model saved to: {final_model_path}")
        
        # Load and analyze audit results
        audit_results = analyze_audit_results(f"{model_save_path}/reward_audit")
        
        return {
            'success': True,
            'model_path': final_model_path,
            'audit_results': audit_results,
            'total_timesteps': total_timesteps
        }
        
    except ValueError as e:
        if "correlation" in str(e).lower():
            logger.error("🚨 TRAINING STOPPED: Reward-P&L correlation too low!")
            logger.error("This indicates the reward function is not aligned with profitability.")
            logger.error("Please revise the reward function before continuing.")
            
            return {
                'success': False,
                'error': 'reward_pnl_misalignment',
                'message': str(e),
                'recommendation': 'Revise reward function to better align with P&L'
            }
        else:
            raise
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return {
            'success': False,
            'error': 'training_failure',
            'message': str(e)
        }


def analyze_audit_results(audit_dir: str) -> Dict[str, Any]:
    """
    Analyze audit results and provide summary.
    
    Args:
        audit_dir: Directory containing audit results
    
    Returns:
        Dict containing audit analysis
    """
    audit_path = Path(audit_dir)
    
    try:
        # Load final metrics if available
        final_metrics_path = audit_path / "final_metrics.json"
        if final_metrics_path.exists():
            import json
            with open(final_metrics_path, 'r') as f:
                final_metrics = json.load(f)
        else:
            final_metrics = {}
        
        # Load episode data if available
        csv_path = audit_path / "reward_pnl_audit.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            episode_analysis = {
                'total_episodes': len(df),
                'mean_step_correlation': df['step_correlation'].mean(),
                'median_step_correlation': df['step_correlation'].median(),
                'min_step_correlation': df['step_correlation'].min(),
                'max_step_correlation': df['step_correlation'].max(),
                'episodes_below_threshold': (df['step_correlation'] < 0.5).sum(),
                'total_reward': df['total_reward'].sum(),
                'total_pnl': df['total_realized_pnl'].sum(),
                'episode_level_correlation': df[['total_reward', 'total_realized_pnl']].corr().iloc[0, 1]
            }
        else:
            episode_analysis = {}
        
        # Combine results
        audit_summary = {
            'audit_completed': True,
            'audit_directory': str(audit_path),
            'final_metrics': final_metrics,
            'episode_analysis': episode_analysis,
            'files_generated': [f.name for f in audit_path.glob('*') if f.is_file()]
        }
        
        # Add recommendation
        mean_corr = episode_analysis.get('mean_step_correlation', 0.0)
        episode_corr = episode_analysis.get('episode_level_correlation', 0.0)
        
        if mean_corr >= 0.8 and episode_corr >= 0.8:
            audit_summary['recommendation'] = "EXCELLENT - Ready for deployment"
            audit_summary['deployment_ready'] = True
        elif mean_corr >= 0.6 and episode_corr >= 0.6:
            audit_summary['recommendation'] = "GOOD - Consider minor optimizations"
            audit_summary['deployment_ready'] = True
        elif mean_corr >= 0.4 or episode_corr >= 0.4:
            audit_summary['recommendation'] = "MODERATE - Improve reward alignment"
            audit_summary['deployment_ready'] = False
        else:
            audit_summary['recommendation'] = "POOR - Significant revision needed"
            audit_summary['deployment_ready'] = False
        
        return audit_summary
        
    except Exception as e:
        logger.error(f"Error analyzing audit results: {e}")
        return {
            'audit_completed': False,
            'error': str(e)
        }


def create_reward_audit_report(audit_results: Dict[str, Any], output_path: str) -> None:
    """
    Create a comprehensive reward audit report.
    
    Args:
        audit_results: Results from analyze_audit_results
        output_path: Path to save the report
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("INTRADAYJULES REWARD-P&L AUDIT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        if not audit_results.get('audit_completed', False):
            f.write("❌ AUDIT FAILED\n")
            f.write(f"Error: {audit_results.get('error', 'Unknown error')}\n")
            return
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Deployment Ready: {'✅ YES' if audit_results.get('deployment_ready', False) else '❌ NO'}\n")
        f.write(f"Recommendation: {audit_results.get('recommendation', 'Unknown')}\n\n")
        
        # Detailed Analysis
        episode_analysis = audit_results.get('episode_analysis', {})
        if episode_analysis:
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Step Correlation: {episode_analysis.get('mean_step_correlation', 0):.3f}\n")
            f.write(f"Episode-Level Correlation: {episode_analysis.get('episode_level_correlation', 0):.3f}\n")
            f.write(f"Total Episodes: {episode_analysis.get('total_episodes', 0)}\n")
            f.write(f"Episodes Below Threshold: {episode_analysis.get('episodes_below_threshold', 0)}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Reward: {episode_analysis.get('total_reward', 0):.4f}\n")
            f.write(f"Total P&L: ${episode_analysis.get('total_pnl', 0):.2f}\n\n")
        
        # Files Generated
        files = audit_results.get('files_generated', [])
        if files:
            f.write("GENERATED FILES\n")
            f.write("-" * 20 + "\n")
            for file in files:
                f.write(f"- {file}\n")
            f.write("\n")
        
        # Next Steps
        f.write("NEXT STEPS\n")
        f.write("-" * 20 + "\n")
        if audit_results.get('deployment_ready', False):
            f.write("✅ Model is ready for deployment\n")
            f.write("- Review diagnostic plots for any edge cases\n")
            f.write("- Consider additional validation on out-of-sample data\n")
        else:
            f.write("❌ Model requires additional work before deployment\n")
            f.write("- Revise reward function to better align with P&L\n")
            f.write("- Consider adjusting transaction costs or position sizing\n")
            f.write("- Re-run training with improved reward function\n")
    
    logger.info(f"📋 Audit report saved to: {report_path}")


# Example usage
def example_usage():
    """Example of how to use the reward audit integration."""
    print("""
    # Example: Enhanced training with reward-P&L audit
    
    from stable_baselines3 import DQN
    from src.training.reward_audit_integration import enhanced_training_with_audit
    
    # Create model and environment
    model = DQN("MlpPolicy", env, verbose=1)
    
    # Example 1: Strict production training
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=100000,
        model_save_path="models/nvda_dqn_audited",
        eval_env=eval_env,
        audit_strict=True,  # Overrides any audit_config settings
        callback_order="audit_first"  # Audit sees all steps including eval
    )
    
    # Example 2: Custom configuration with explicit precedence
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=100000,
        model_save_path="models/nvda_dqn_custom",
        eval_env=eval_env,
        audit_strict=False,  # Use custom config
        callback_order="audit_last",  # Audit only sees training steps
        audit_config={
            'min_correlation_threshold': 0.6,
            'alert_episodes': 8,
            'fail_fast': False
        }
    )
    
    # Example 3: Strict mode with warning about overrides
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=100000,
        model_save_path="models/nvda_dqn_strict",
        audit_strict=True,  # This will override the config below
        audit_config={
            'min_correlation_threshold': 0.4,  # Will be overridden to 0.7
            'fail_fast': False  # Will be overridden to True
        }
    )
    # ⚠️ Warning logged: audit_strict=True overriding audit_config settings
    
    # Check results
    if results['success']:
        print("✅ Training completed successfully!")
        print(f"Model saved to: {results['model_path']}")
        print(f"Deployment ready: {results['audit_results']['deployment_ready']}")
    else:
        print(f"❌ Training failed: {results['message']}")
        if results['error'] == 'reward_pnl_misalignment':
            print("🔧 Action required: Revise reward function")
    """)


if __name__ == "__main__":
    example_usage()