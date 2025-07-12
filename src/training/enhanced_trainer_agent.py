# src/training/enhanced_trainer_agent.py
"""
Enhanced TrainerAgent with MLflow Experiment Registry Integration.

This module extends the existing TrainerAgent with comprehensive experiment tracking
and model versioning capabilities using MLflow and optional S3 storage.

Key Enhancements:
- MLflow experiment registry integration
- Semantic versioning (v2025-07-06--18h51) instead of UUIDs
- Production-ready model deployment pipeline
- Model validation and testing before registration
- S3/cloud storage for model artifacts
- Experiment comparison and rollback capabilities
- Automated model lifecycle management

Usage:
    # Create enhanced trainer with registry
    config = {
        'algorithm': 'DQN',
        'total_timesteps': 20000,
        'model_save_dir': 'models/',
        'experiment_registry': {
            'tracking_uri': 'http://mlflow-server:5000',
            's3_bucket': 'my-models-bucket',
            'enable_wandb': True
        }
    }
    
    trainer = EnhancedTrainerAgent(config)
    
    # Train and register model
    model_version = trainer.train_and_register(
        training_env=env,
        experiment_name="momentum_strategy_v2",
        tags={"strategy": "momentum", "market": "equity"}
    )
    
    # Production deployment
    trainer.deploy_to_production(model_version.version_id)
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np

# Import base TrainerAgent
from .trainer_agent import TrainerAgent
from .experiment_registry import ExperimentRegistry, create_experiment_registry, ModelVersionInfo

# Optional dependencies for enhanced features
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class EnhancedTrainerAgent(TrainerAgent):
    """
    Enhanced TrainerAgent with MLflow experiment registry integration.
    
    This class extends the base TrainerAgent with:
    - MLflow experiment tracking and model registry
    - Semantic versioning for production deployment
    - Model validation and testing pipeline
    - S3/cloud storage integration
    - Automated model lifecycle management
    - Production deployment capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Enhanced TrainerAgent with experiment registry.
        
        Args:
            config: Configuration dictionary with additional registry settings:
                - experiment_registry: Registry configuration
                - auto_register: Whether to auto-register trained models
                - validation_config: Model validation settings
                - deployment_config: Production deployment settings
        """
        # Initialize base TrainerAgent
        super().__init__(config)
        
        # Registry configuration
        self.registry_config = config.get('experiment_registry', {})
        self.auto_register = config.get('auto_register', True)
        self.validation_config = config.get('validation_config', {})
        self.deployment_config = config.get('deployment_config', {})
        
        # Initialize experiment registry
        self.registry = None
        if self.registry_config or MLFLOW_AVAILABLE:
            try:
                self.registry = create_experiment_registry(self.registry_config)
                self.logger.info("✅ Experiment registry initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize experiment registry: {e}")
                self.logger.warning("Falling back to traditional model saving")
        
        # Enhanced logging
        self.logger.info("🚀 EnhancedTrainerAgent initialized")
        if self.registry:
            self.logger.info(f"📊 MLflow tracking: {self.registry.config.tracking_uri}")
            if self.registry.config.s3_bucket:
                self.logger.info(f"☁️  S3 storage: {self.registry.config.s3_bucket}")
            if self.registry.config.enable_wandb:
                self.logger.info(f"🏃 W&B integration: {self.registry.config.wandb_project}")
    
    def train_and_register(self,
                          training_env,
                          experiment_name: str,
                          model_name: Optional[str] = None,
                          tags: Optional[Dict[str, str]] = None,
                          metrics: Optional[Dict[str, float]] = None,
                          validate_before_register: bool = True,
                          **train_kwargs) -> Optional[ModelVersionInfo]:
        """
        Train model and register it in the experiment registry.
        
        Args:
            training_env: Training environment
            experiment_name: Name of the experiment
            model_name: Optional model name (defaults to algorithm)
            tags: Optional tags for the model
            metrics: Optional additional metrics to log
            validate_before_register: Whether to validate model before registration
            **train_kwargs: Additional arguments for training
            
        Returns:
            ModelVersionInfo if registration successful, None otherwise
        """
        self.logger.info(f"🎯 Starting training and registration for experiment: {experiment_name}")
        
        # Train the model using base TrainerAgent
        bundle_path = self.train(training_env, **train_kwargs)
        
        if not bundle_path:
            self.logger.error("❌ Training failed, cannot register model")
            return None
        
        # Register the trained model if registry is available
        if self.registry and self.auto_register:
            return self._register_trained_model(
                bundle_path=bundle_path,
                experiment_name=experiment_name,
                model_name=model_name,
                tags=tags,
                metrics=metrics,
                validate_before_register=validate_before_register
            )
        else:
            self.logger.warning("⚠️  No experiment registry available, model saved locally only")
            return None
    
    def _register_trained_model(self,
                               bundle_path: str,
                               experiment_name: str,
                               model_name: Optional[str] = None,
                               tags: Optional[Dict[str, str]] = None,
                               metrics: Optional[Dict[str, float]] = None,
                               validate_before_register: bool = True) -> Optional[ModelVersionInfo]:
        """Register a trained model in the experiment registry."""
        try:
            bundle_dir = Path(bundle_path)
            
            # Find TorchScript bundle (policy.pt + metadata.json)
            torchscript_dirs = list(bundle_dir.glob("*_torchscript"))
            if not torchscript_dirs:
                self.logger.error("❌ No TorchScript bundle found for registration")
                return None
            
            torchscript_dir = torchscript_dirs[0]
            model_path = torchscript_dir / "policy.pt"
            metadata_path = torchscript_dir / "metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                self.logger.error("❌ Missing policy.pt or metadata.json files")
                return None
            
            # Prepare tags
            if tags is None:
                tags = {}
            
            # Add training context tags
            tags.update({
                'training_algorithm': self.algorithm_name,
                'training_timesteps': str(self.config.get('total_timesteps', 'unknown')),
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'bundle_path': str(bundle_path)
            })
            
            # Prepare metrics (extract from training if available)
            if metrics is None:
                metrics = {}
            
            # Try to extract metrics from training logs or tensorboard
            training_metrics = self._extract_training_metrics(bundle_dir)
            metrics.update(training_metrics)
            
            # Register model
            self.logger.info("📝 Registering model in experiment registry...")
            
            model_version_info = self.registry.register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                experiment_name=experiment_name,
                model_name=model_name,
                tags=tags,
                metrics=metrics,
                validate_model=validate_before_register
            )
            
            self.logger.info(f"✅ Model registered successfully: {model_version_info.version_id}")
            self.logger.info(f"🏷️  Version: {model_version_info.version_id}")
            self.logger.info(f"🔬 Experiment: {experiment_name}")
            self.logger.info(f"📊 Status: {model_version_info.status}")
            
            # Log validation results if available
            if model_version_info.validation_results:
                validation = model_version_info.validation_results
                if validation.get('passed', False):
                    self.logger.info("✅ Model validation: PASSED")
                else:
                    self.logger.warning("⚠️  Model validation: FAILED")
                    for error in validation.get('errors', []):
                        self.logger.warning(f"   - {error}")
            
            return model_version_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            import traceback
            self.logger.debug(f"Registration error traceback: {traceback.format_exc()}")
            return None
    
    def _extract_training_metrics(self, bundle_dir: Path) -> Dict[str, float]:
        """Extract training metrics from logs or tensorboard."""
        metrics = {}
        
        try:
            # Try to read from tensorboard logs
            tensorboard_dir = bundle_dir.parent / "logs" / "tensorboard"
            if tensorboard_dir.exists():
                # This would require tensorboard log parsing
                # For now, return empty metrics
                pass
            
            # Try to extract from monitor logs
            monitor_dir = bundle_dir.parent / "logs" / "monitor_logs"
            if monitor_dir.exists():
                monitor_files = list(monitor_dir.glob("*.monitor.csv"))
                if monitor_files:
                    # Parse monitor file for episode rewards
                    import pandas as pd
                    try:
                        df = pd.read_csv(monitor_files[0])
                        if 'r' in df.columns:  # reward column
                            metrics['mean_episode_reward'] = float(df['r'].mean())
                            metrics['max_episode_reward'] = float(df['r'].max())
                            metrics['min_episode_reward'] = float(df['r'].min())
                            metrics['std_episode_reward'] = float(df['r'].std())
                            metrics['total_episodes'] = len(df)
                    except Exception as e:
                        self.logger.debug(f"Failed to parse monitor logs: {e}")
            
        except Exception as e:
            self.logger.debug(f"Failed to extract training metrics: {e}")
        
        return metrics
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersionInfo]:
        """
        Get information about a specific model version.
        
        Args:
            version_id: Semantic version ID (e.g., "v2025-07-06--18h51")
            
        Returns:
            ModelVersionInfo if found, None otherwise
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return None
        
        return self.registry.get_model_version(version_id)
    
    def list_model_versions(self,
                           experiment_name: Optional[str] = None,
                           algorithm: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 20) -> List[ModelVersionInfo]:
        """
        List model versions with optional filtering.
        
        Args:
            experiment_name: Filter by experiment name
            algorithm: Filter by algorithm
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of ModelVersionInfo objects
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return []
        
        return self.registry.list_model_versions(
            experiment_name=experiment_name,
            algorithm=algorithm,
            status=status,
            limit=limit
        )
    
    def download_model_for_deployment(self, version_id: str, deployment_dir: Optional[str] = None) -> Optional[str]:
        """
        Download a model version for production deployment.
        
        Args:
            version_id: Semantic version ID
            deployment_dir: Optional deployment directory
            
        Returns:
            Path to downloaded model directory
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return None
        
        self.logger.info(f"📥 Downloading model for deployment: {version_id}")
        
        model_path = self.registry.download_model(version_id, deployment_dir)
        
        if model_path:
            self.logger.info(f"✅ Model downloaded: {model_path}")
            
            # Verify deployment files
            policy_path = Path(model_path) / "policy.pt"
            metadata_path = Path(model_path) / "metadata.json"
            
            if policy_path.exists() and metadata_path.exists():
                self.logger.info("✅ Deployment files verified")
                return model_path
            else:
                self.logger.error("❌ Missing deployment files")
                return None
        else:
            self.logger.error(f"❌ Failed to download model: {version_id}")
            return None
    
    def deploy_to_production(self, version_id: str, deployment_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Deploy a model version to production.
        
        Args:
            version_id: Semantic version ID
            deployment_config: Optional deployment configuration
            
        Returns:
            True if deployment successful
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return False
        
        self.logger.info(f"🚀 Deploying model to production: {version_id}")
        
        # Get model information
        model_info = self.registry.get_model_version(version_id)
        if not model_info:
            self.logger.error(f"❌ Model version not found: {version_id}")
            return False
        
        # Check if model is validated
        if model_info.status not in ["validated", "production"]:
            self.logger.warning(f"⚠️  Model {version_id} status is '{model_info.status}', not validated")
            
            # Ask for confirmation in production
            if not self.deployment_config.get('allow_unvalidated', False):
                self.logger.error("❌ Cannot deploy unvalidated model to production")
                return False
        
        # Merge deployment config
        final_deployment_config = self.deployment_config.copy()
        if deployment_config:
            final_deployment_config.update(deployment_config)
        
        # Promote to production
        success = self.registry.promote_to_production(version_id, final_deployment_config)
        
        if success:
            self.logger.info(f"✅ Model deployed to production: {version_id}")
            
            # Optional: Download to production directory
            if final_deployment_config.get('download_to_production_dir'):
                prod_dir = final_deployment_config.get('production_dir', 'production/models')
                self.download_model_for_deployment(version_id, prod_dir)
            
            return True
        else:
            self.logger.error(f"❌ Failed to deploy model: {version_id}")
            return False
    
    def rollback_production(self, previous_version_id: str) -> bool:
        """
        Rollback production to a previous model version.
        
        Args:
            previous_version_id: Version ID to rollback to
            
        Returns:
            True if rollback successful
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return False
        
        self.logger.info(f"🔄 Rolling back production to: {previous_version_id}")
        
        # Verify the previous version exists and was in production
        model_info = self.registry.get_model_version(previous_version_id)
        if not model_info:
            self.logger.error(f"❌ Previous version not found: {previous_version_id}")
            return False
        
        # Deploy the previous version
        rollback_config = {
            'rollback': True,
            'rollback_from': 'current_production',
            'rollback_reason': 'Manual rollback'
        }
        
        return self.deploy_to_production(previous_version_id, rollback_config)
    
    def compare_model_versions(self, version_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            version_ids: List of version IDs to compare
            
        Returns:
            Comparison results
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return {}
        
        self.logger.info(f"📊 Comparing model versions: {version_ids}")
        
        comparison = self.registry.compare_models(version_ids)
        
        # Log comparison summary
        if "error" not in comparison:
            self.logger.info("📈 Model Comparison Results:")
            for metric, values in comparison.get("metrics_comparison", {}).items():
                self.logger.info(f"   {metric}:")
                for version_id, value in values.items():
                    if value is not None:
                        self.logger.info(f"     {version_id}: {value:.4f}")
        
        return comparison
    
    def get_production_model_info(self) -> Optional[ModelVersionInfo]:
        """
        Get information about the current production model.
        
        Returns:
            ModelVersionInfo for current production model, None if not found
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return None
        
        # Find current production model
        production_models = self.registry.list_model_versions(status="production", limit=1)
        
        if production_models:
            return production_models[0]
        else:
            self.logger.info("ℹ️  No production model found")
            return None
    
    def validate_model_version(self, version_id: str) -> Dict[str, Any]:
        """
        Validate a specific model version.
        
        Args:
            version_id: Version ID to validate
            
        Returns:
            Validation results
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return {"error": "No registry available"}
        
        self.logger.info(f"🧪 Validating model version: {version_id}")
        
        # Download model for validation
        model_path = self.registry.download_model(version_id)
        if not model_path:
            return {"error": "Failed to download model"}
        
        # Load metadata
        metadata_path = Path(model_path) / "metadata.json"
        if not metadata_path.exists():
            return {"error": "Metadata file not found"}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Run validation
        policy_path = Path(model_path) / "policy.pt"
        validation_results = self.registry._validate_model(policy_path, metadata)
        
        # Log results
        if validation_results.get('passed', False):
            self.logger.info(f"✅ Model validation passed: {version_id}")
        else:
            self.logger.warning(f"⚠️  Model validation failed: {version_id}")
            for error in validation_results.get('errors', []):
                self.logger.warning(f"   - {error}")
        
        return validation_results
    
    def cleanup_old_models(self, retention_days: int = 90) -> int:
        """
        Clean up old model versions.
        
        Args:
            retention_days: Days to retain models
            
        Returns:
            Number of models cleaned up
        """
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return 0
        
        self.logger.info(f"🧹 Cleaning up models older than {retention_days} days")
        
        cleanup_count = self.registry.cleanup_old_versions(retention_days)
        
        self.logger.info(f"✅ Cleaned up {cleanup_count} old model versions")
        return cleanup_count
    
    def generate_model_report(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive model report.
        
        Args:
            experiment_name: Optional experiment name filter
            
        Returns:
            Model report dictionary
        """
        if not self.registry:
            return {"error": "No experiment registry available"}
        
        self.logger.info("📋 Generating model report...")
        
        # Get all models
        all_models = self.registry.list_model_versions(
            experiment_name=experiment_name,
            limit=100
        )
        
        # Categorize by status
        status_counts = {}
        algorithm_counts = {}
        
        for model in all_models:
            status_counts[model.status] = status_counts.get(model.status, 0) + 1
            algorithm_counts[model.algorithm] = algorithm_counts.get(model.algorithm, 0) + 1
        
        # Find production model
        production_models = [m for m in all_models if m.status == "production"]
        current_production = production_models[0] if production_models else None
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "total_models": len(all_models),
            "status_breakdown": status_counts,
            "algorithm_breakdown": algorithm_counts,
            "current_production": {
                "version_id": current_production.version_id if current_production else None,
                "algorithm": current_production.algorithm if current_production else None,
                "created_at": current_production.created_at.isoformat() if current_production else None
            },
            "recent_models": [
                {
                    "version_id": m.version_id,
                    "algorithm": m.algorithm,
                    "status": m.status,
                    "created_at": m.created_at.isoformat()
                }
                for m in all_models[:10]  # Last 10 models
            ]
        }
        
        self.logger.info(f"📊 Report generated: {report['total_models']} models found")
        return report


def create_enhanced_trainer_agent(config: Dict[str, Any]) -> EnhancedTrainerAgent:
    """
    Create an EnhancedTrainerAgent with recommended experiment registry settings.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Configured EnhancedTrainerAgent instance
    """
    # Add recommended registry settings
    enhanced_config = config.copy()
    
    if 'experiment_registry' not in enhanced_config:
        enhanced_config['experiment_registry'] = {
            'tracking_uri': 'sqlite:///mlflow.db',
            'auto_validate': True,
            'retention_days': 90
        }
    
    # Set auto-register default
    enhanced_config.setdefault('auto_register', True)
    
    # Set validation config defaults
    enhanced_config.setdefault('validation_config', {
        'validate_before_register': True,
        'run_performance_tests': True
    })
    
    return EnhancedTrainerAgent(enhanced_config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'algorithm': 'DQN',
        'total_timesteps': 10000,
        'model_save_dir': 'models/',
        'experiment_registry': {
            'tracking_uri': 'sqlite:///test_mlflow.db',
            's3_bucket': None,  # Use local storage for testing
            'enable_wandb': False,
            'auto_validate': True
        },
        'auto_register': True
    }
    
    # Create enhanced trainer
    trainer = create_enhanced_trainer_agent(config)
    
    # Example operations (would require actual environment)
    print("Enhanced TrainerAgent Example:")
    print("1. Create trainer ✓")
    print("2. Train and register model (requires environment)")
    print("3. List model versions")
    print("4. Deploy to production")
    print("5. Generate model report")
    
    # Generate model report
    report = trainer.generate_model_report()
    print(f"Model report: {report['total_models']} models found")
    
    # List recent models
    models = trainer.list_model_versions(limit=5)
    print(f"Recent models: {len(models)} found")
    
    for model in models:
        print(f"  - {model.version_id} ({model.algorithm}) - {model.status}")