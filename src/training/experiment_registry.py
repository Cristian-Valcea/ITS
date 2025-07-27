# src/training/experiment_registry.py
"""
MLflow-based Experiment Registry for TrainerAgent model versioning.

This module provides a comprehensive experiment registry system that allows
production systems to request specific model versions by semantic identifiers
rather than UUIDs.

Key Features:
- MLflow integration for experiment tracking
- Semantic versioning (v2025-07-06--18h51)
- Model artifact management with S3/local storage
- Production-ready model deployment pipeline
- Experiment comparison and rollback capabilities
- Automated model validation and testing

Usage:
    # Initialize registry
    registry = ExperimentRegistry(
        tracking_uri="http://mlflow-server:5000",
        s3_bucket="my-models-bucket"
    )
    
    # Register new model
    model_version = registry.register_model(
        model_path="models/DQN_20250706_185123/policy.pt",
        metadata_path="models/DQN_20250706_185123/metadata.json",
        experiment_name="intraday_trading_dqn",
        tags={"environment": "production", "strategy": "momentum"}
    )
    
    # Retrieve model for production
    model_info = registry.get_model_version("v2025-07-06--18h51")
    policy_path = registry.download_model(model_info.version_id)
"""

import os
import json
import shutil
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import tempfile

# MLflow dependencies
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# AWS S3 dependencies (optional)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

# Alternative: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class ModelVersionInfo:
    """Information about a registered model version."""
    version_id: str  # Semantic version like "v2025-07-06--18h51"
    run_id: str  # MLflow run ID
    experiment_id: str  # MLflow experiment ID
    model_uri: str  # MLflow model URI
    artifact_uri: str  # Direct artifact URI (S3/local)
    algorithm: str  # DQN, PPO, etc.
    created_at: datetime
    tags: Dict[str, str]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    status: str  # "registered", "validated", "production", "archived"
    validation_results: Optional[Dict[str, Any]] = None
    deployment_info: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiment registry."""
    tracking_uri: str = "sqlite:///mlflow.db"  # MLflow tracking server
    s3_bucket: Optional[str] = None  # S3 bucket for artifacts
    s3_prefix: str = "models"  # S3 prefix for model artifacts
    local_cache_dir: str = "cache/models"  # Local cache directory
    default_experiment: str = "intraday_trading"  # Default experiment name
    enable_wandb: bool = False  # Enable Weights & Biases integration
    wandb_project: str = "intraday-jules"  # W&B project name
    auto_validate: bool = True  # Auto-validate registered models
    retention_days: int = 90  # Days to retain old model versions


class ExperimentRegistry:
    """
    MLflow-based experiment registry for model versioning and deployment.
    
    This class provides a comprehensive system for:
    - Registering trained models with semantic versioning
    - Storing model artifacts in S3 or local storage
    - Tracking experiments with MLflow
    - Validating models before production deployment
    - Managing model lifecycle (register -> validate -> deploy -> archive)
    """
    
    def __init__(self, config: Union[ExperimentConfig, Dict[str, Any]]):
        """
        Initialize the experiment registry.
        
        Args:
            config: ExperimentConfig instance or dictionary with configuration
        """
        if isinstance(config, dict):
            self.config = ExperimentConfig(**config)
        else:
            self.config = config
        
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for experiment registry. Install with: pip install mlflow")
        
        # Initialize MLflow
        self._initialize_mlflow()
        
        # Initialize S3 client if configured
        self.s3_client = None
        if self.config.s3_bucket and S3_AVAILABLE:
            self._initialize_s3()
        
        # Initialize W&B if configured
        if self.config.enable_wandb and WANDB_AVAILABLE:
            self._initialize_wandb()
        
        # Create local cache directory
        Path(self.config.local_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ExperimentRegistry initialized with tracking URI: {self.config.tracking_uri}")
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self.mlflow_client = MlflowClient()
            
            # Test connection
            experiments = self.mlflow_client.search_experiments()
            self.logger.info(f"Connected to MLflow: {len(experiments)} experiments found")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def _initialize_s3(self):
        """Initialize S3 client for artifact storage."""
        try:
            self.s3_client = boto3.client('s3')
            
            # Test S3 connection
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            self.logger.info(f"Connected to S3 bucket: {self.config.s3_bucket}")
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Configure AWS CLI or environment variables.")
            self.s3_client = None
        except ClientError as e:
            self.logger.error(f"S3 bucket access failed: {e}")
            self.s3_client = None
        except Exception as e:
            self.logger.error(f"S3 initialization failed: {e}")
            self.s3_client = None
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases integration."""
        try:
            wandb.init(
                project=self.config.wandb_project,
                job_type="model_registry",
                reinit=True
            )
            self.logger.info(f"Connected to W&B project: {self.config.wandb_project}")
        except Exception as e:
            self.logger.error(f"W&B initialization failed: {e}")
    
    def register_model(self,
                      model_path: Union[str, Path],
                      metadata_path: Union[str, Path],
                      experiment_name: str,
                      model_name: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      validate_model: bool = None) -> ModelVersionInfo:
        """
        Register a new model version in the experiment registry.
        
        Args:
            model_path: Path to the model file (policy.pt)
            metadata_path: Path to the metadata file (metadata.json)
            experiment_name: Name of the experiment
            model_name: Optional model name (defaults to algorithm from metadata)
            tags: Optional tags for the model
            metrics: Optional metrics to log
            parameters: Optional parameters to log
            validate_model: Whether to validate the model (defaults to config.auto_validate)
            
        Returns:
            ModelVersionInfo with details about the registered model
        """
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Generate semantic version
        version_id = self._generate_version_id(metadata.get('created_at'))
        
        # Set defaults
        if model_name is None:
            model_name = f"{metadata.get('algo', 'unknown')}_policy"
        if tags is None:
            tags = {}
        if metrics is None:
            metrics = {}
        if parameters is None:
            parameters = {}
        if validate_model is None:
            validate_model = self.config.auto_validate
        
        # Add metadata to parameters
        parameters.update({
            'algorithm': metadata.get('algo'),
            'framework': metadata.get('framework'),
            'export_method': metadata.get('export_method'),
            'model_size_kb': model_path.stat().st_size / 1024
        })
        
        # Add system tags
        tags.update({
            'version_id': version_id,
            'model_type': 'policy',
            'framework': metadata.get('framework', 'torchscript'),
            'registered_at': datetime.now().isoformat()
        })
        
        try:
            # Create or get experiment
            experiment = self._get_or_create_experiment(experiment_name)
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version_id):
                run_id = mlflow.active_run().info.run_id
                
                # Log parameters and metrics
                mlflow.log_params(parameters)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)
                
                # Log model artifacts
                artifacts_dir = "model_artifacts"
                mlflow.log_artifact(str(model_path), artifacts_dir)
                mlflow.log_artifact(str(metadata_path), artifacts_dir)
                
                # Get artifact URI
                artifact_uri = mlflow.get_artifact_uri(artifacts_dir)
                model_uri = f"runs:/{run_id}/{artifacts_dir}"
                
                # Upload to S3 if configured
                s3_uri = None
                if self.s3_client:
                    s3_uri = self._upload_to_s3(model_path, metadata_path, version_id)
                
                # Create model version info
                model_version_info = ModelVersionInfo(
                    version_id=version_id,
                    run_id=run_id,
                    experiment_id=experiment.experiment_id,
                    model_uri=model_uri,
                    artifact_uri=s3_uri or artifact_uri,
                    algorithm=metadata.get('algo', 'unknown'),
                    created_at=datetime.now(),
                    tags=tags,
                    metrics=metrics,
                    parameters=parameters,
                    status="registered"
                )
                
                # Log to W&B if enabled
                if self.config.enable_wandb and WANDB_AVAILABLE:
                    self._log_to_wandb(model_version_info, model_path, metadata_path)
                
                # Validate model if requested
                if validate_model:
                    validation_results = self._validate_model(model_path, metadata)
                    model_version_info.validation_results = validation_results
                    
                    if validation_results.get('passed', False):
                        model_version_info.status = "validated"
                        mlflow.set_tag("validation_status", "passed")
                    else:
                        mlflow.set_tag("validation_status", "failed")
                
                # Register model in MLflow Model Registry
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=None,  # We're logging artifacts manually
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to register in MLflow Model Registry: {e}")
                
                self.logger.info(f"Model registered successfully: {version_id}")
                return model_version_info
                
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersionInfo]:
        """
        Get information about a specific model version.
        
        Args:
            version_id: Semantic version ID (e.g., "v2025-07-06--18h51")
            
        Returns:
            ModelVersionInfo if found, None otherwise
        """
        try:
            # Search for runs with the version_id tag
            runs = self.mlflow_client.search_runs(
                experiment_ids=[],  # Search all experiments
                filter_string=f"tags.version_id = '{version_id}'",
                max_results=1
            )
            
            if not runs:
                self.logger.warning(f"Model version not found: {version_id}")
                return None
            
            run = runs[0]
            
            # Construct ModelVersionInfo from run data
            model_version_info = ModelVersionInfo(
                version_id=version_id,
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                model_uri=f"runs:/{run.info.run_id}/model_artifacts",
                artifact_uri=run.info.artifact_uri,
                algorithm=run.data.params.get('algorithm', 'unknown'),
                created_at=datetime.fromtimestamp(run.info.start_time / 1000),
                tags=run.data.tags,
                metrics=run.data.metrics,
                parameters=run.data.params,
                status=run.data.tags.get('status', 'registered')
            )
            
            return model_version_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model version {version_id}: {e}")
            return None
    
    def list_model_versions(self,
                          experiment_name: Optional[str] = None,
                          algorithm: Optional[str] = None,
                          status: Optional[str] = None,
                          limit: int = 50) -> List[ModelVersionInfo]:
        """
        List model versions with optional filtering.
        
        Args:
            experiment_name: Filter by experiment name
            algorithm: Filter by algorithm (DQN, PPO, etc.)
            status: Filter by status (registered, validated, production, archived)
            limit: Maximum number of results
            
        Returns:
            List of ModelVersionInfo objects
        """
        try:
            # Build filter string
            filters = []
            if algorithm:
                filters.append(f"params.algorithm = '{algorithm}'")
            if status:
                filters.append(f"tags.status = '{status}'")
            
            filter_string = " and ".join(filters) if filters else ""
            
            # Get experiment IDs
            experiment_ids = []
            if experiment_name:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_ids = [experiment.experiment_id]
            else:
                experiments = self.mlflow_client.search_experiments()
                experiment_ids = [exp.experiment_id for exp in experiments]
            
            # Search runs
            runs = self.mlflow_client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            # Convert to ModelVersionInfo objects
            model_versions = []
            for run in runs:
                version_id = run.data.tags.get('version_id')
                if version_id:  # Only include runs with version_id
                    model_version_info = ModelVersionInfo(
                        version_id=version_id,
                        run_id=run.info.run_id,
                        experiment_id=run.info.experiment_id,
                        model_uri=f"runs:/{run.info.run_id}/model_artifacts",
                        artifact_uri=run.info.artifact_uri,
                        algorithm=run.data.params.get('algorithm', 'unknown'),
                        created_at=datetime.fromtimestamp(run.info.start_time / 1000),
                        tags=run.data.tags,
                        metrics=run.data.metrics,
                        parameters=run.data.params,
                        status=run.data.tags.get('status', 'registered')
                    )
                    model_versions.append(model_version_info)
            
            return model_versions
            
        except Exception as e:
            self.logger.error(f"Failed to list model versions: {e}")
            return []
    
    def download_model(self, version_id: str, local_path: Optional[str] = None) -> Optional[str]:
        """
        Download a model version to local storage.
        
        Args:
            version_id: Semantic version ID
            local_path: Optional local path (defaults to cache directory)
            
        Returns:
            Path to downloaded model directory, or None if failed
        """
        model_info = self.get_model_version(version_id)
        if not model_info:
            return None
        
        if local_path is None:
            local_path = Path(self.config.local_cache_dir) / version_id
        else:
            local_path = Path(local_path)
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download from MLflow
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_info.model_uri,
                dst_path=str(local_path)
            )
            
            self.logger.info(f"Model downloaded: {version_id} -> {local_path}")
            return str(local_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download model {version_id}: {e}")
            
            # Try S3 fallback if available
            if self.s3_client and model_info.artifact_uri.startswith('s3://'):
                return self._download_from_s3(version_id, local_path)
            
            return None
    
    def promote_to_production(self, version_id: str, deployment_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Promote a model version to production status.
        
        Args:
            version_id: Semantic version ID
            deployment_config: Optional deployment configuration
            
        Returns:
            True if promotion successful
        """
        model_info = self.get_model_version(version_id)
        if not model_info:
            self.logger.error(f"Model version not found: {version_id}")
            return False
        
        if model_info.status != "validated":
            self.logger.warning(f"Model {version_id} is not validated. Current status: {model_info.status}")
        
        try:
            # Update run tags
            self.mlflow_client.set_tag(model_info.run_id, "status", "production")
            self.mlflow_client.set_tag(model_info.run_id, "promoted_at", datetime.now().isoformat())
            
            if deployment_config:
                self.mlflow_client.set_tag(
                    model_info.run_id, 
                    "deployment_config", 
                    json.dumps(deployment_config)
                )
            
            self.logger.info(f"Model promoted to production: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {version_id}: {e}")
            return False
    
    def archive_model(self, version_id: str, reason: str = "Manual archive") -> bool:
        """
        Archive a model version.
        
        Args:
            version_id: Semantic version ID
            reason: Reason for archiving
            
        Returns:
            True if archiving successful
        """
        model_info = self.get_model_version(version_id)
        if not model_info:
            self.logger.error(f"Model version not found: {version_id}")
            return False
        
        try:
            # Update run tags
            self.mlflow_client.set_tag(model_info.run_id, "status", "archived")
            self.mlflow_client.set_tag(model_info.run_id, "archived_at", datetime.now().isoformat())
            self.mlflow_client.set_tag(model_info.run_id, "archive_reason", reason)
            
            self.logger.info(f"Model archived: {version_id} (reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to archive model {version_id}: {e}")
            return False
    
    def compare_models(self, version_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            version_ids: List of version IDs to compare
            
        Returns:
            Comparison results dictionary
        """
        models = []
        for version_id in version_ids:
            model_info = self.get_model_version(version_id)
            if model_info:
                models.append(model_info)
        
        if len(models) < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        # Compare metrics
        all_metrics = set()
        for model in models:
            all_metrics.update(model.metrics.keys())
        
        comparison = {
            "models": [model.version_id for model in models],
            "metrics_comparison": {},
            "parameters_comparison": {},
            "summary": {}
        }
        
        # Metrics comparison
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = {}
            for model in models:
                comparison["metrics_comparison"][metric][model.version_id] = model.metrics.get(metric)
        
        # Parameters comparison
        all_params = set()
        for model in models:
            all_params.update(model.parameters.keys())
        
        for param in all_params:
            comparison["parameters_comparison"][param] = {}
            for model in models:
                comparison["parameters_comparison"][param][model.version_id] = model.parameters.get(param)
        
        return comparison
    
    def cleanup_old_versions(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old model versions based on retention policy.
        
        Args:
            retention_days: Days to retain (defaults to config.retention_days)
            
        Returns:
            Number of versions cleaned up
        """
        if retention_days is None:
            retention_days = self.config.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Get all model versions
        all_versions = self.list_model_versions(limit=1000)
        
        cleanup_count = 0
        for version in all_versions:
            # Don't clean up production models
            if version.status == "production":
                continue
            
            # Check if older than retention period
            if version.created_at < cutoff_date:
                if self.archive_model(version.version_id, "Automatic cleanup - retention policy"):
                    cleanup_count += 1
        
        self.logger.info(f"Cleaned up {cleanup_count} old model versions")
        return cleanup_count
    
    def _generate_version_id(self, created_at: Optional[str] = None) -> str:
        """Generate semantic version ID."""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        else:
            dt = datetime.now()
        
        return f"v{dt.strftime('%Y-%m-%d--%Hh%M')}"
    
    def _get_or_create_experiment(self, experiment_name: str):
        """Get or create MLflow experiment."""
        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
                experiment = self.mlflow_client.get_experiment(experiment_id)
            return experiment
        except Exception as e:
            self.logger.error(f"Failed to get/create experiment {experiment_name}: {e}")
            raise
    
    def _upload_to_s3(self, model_path: Path, metadata_path: Path, version_id: str) -> Optional[str]:
        """Upload model artifacts to S3."""
        if not self.s3_client:
            return None
        
        try:
            s3_prefix = f"{self.config.s3_prefix}/{version_id}"
            
            # Upload model file
            model_key = f"{s3_prefix}/policy.pt"
            self.s3_client.upload_file(str(model_path), self.config.s3_bucket, model_key)
            
            # Upload metadata file
            metadata_key = f"{s3_prefix}/metadata.json"
            self.s3_client.upload_file(str(metadata_path), self.config.s3_bucket, metadata_key)
            
            s3_uri = f"s3://{self.config.s3_bucket}/{s3_prefix}"
            self.logger.info(f"Uploaded to S3: {s3_uri}")
            return s3_uri
            
        except Exception as e:
            self.logger.error(f"S3 upload failed: {e}")
            return None
    
    def _download_from_s3(self, version_id: str, local_path: Path) -> Optional[str]:
        """Download model artifacts from S3."""
        if not self.s3_client:
            return None
        
        try:
            s3_prefix = f"{self.config.s3_prefix}/{version_id}"
            
            # Download model file
            model_key = f"{s3_prefix}/policy.pt"
            model_local_path = local_path / "policy.pt"
            self.s3_client.download_file(self.config.s3_bucket, model_key, str(model_local_path))
            
            # Download metadata file
            metadata_key = f"{s3_prefix}/metadata.json"
            metadata_local_path = local_path / "metadata.json"
            self.s3_client.download_file(self.config.s3_bucket, metadata_key, str(metadata_local_path))
            
            self.logger.info(f"Downloaded from S3: {version_id}")
            return str(local_path)
            
        except Exception as e:
            self.logger.error(f"S3 download failed: {e}")
            return None
    
    def _log_to_wandb(self, model_info: ModelVersionInfo, model_path: Path, metadata_path: Path):
        """Log model to Weights & Biases."""
        try:
            # Create W&B artifact
            artifact = wandb.Artifact(
                name=f"policy_{model_info.version_id}",
                type="model",
                metadata=model_info.parameters
            )
            
            # Add files to artifact
            artifact.add_file(str(model_path), name="policy.pt")
            artifact.add_file(str(metadata_path), name="metadata.json")
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            # Log metrics
            wandb.log(model_info.metrics)
            
        except Exception as e:
            self.logger.error(f"W&B logging failed: {e}")
    
    def _validate_model(self, model_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model before registration."""
        validation_results = {
            "passed": False,
            "checks": {},
            "errors": []
        }
        
        try:
            # Check file exists and is readable
            if not model_path.exists():
                validation_results["errors"].append("Model file does not exist")
                return validation_results
            
            validation_results["checks"]["file_exists"] = True
            
            # Check file size
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            validation_results["checks"]["file_size_mb"] = file_size_mb
            
            if file_size_mb > 100:  # Warn if model is very large
                validation_results["errors"].append(f"Model file is very large: {file_size_mb:.1f} MB")
            
            # Try to load the model (basic check)
            try:
                import torch
                model = torch.jit.load(str(model_path))
                validation_results["checks"]["model_loadable"] = True
                
                # Check if model can be evaluated
                model.eval()
                validation_results["checks"]["model_evaluable"] = True
                
            except Exception as e:
                validation_results["errors"].append(f"Model loading failed: {e}")
                validation_results["checks"]["model_loadable"] = False
            
            # Check metadata consistency
            required_fields = ["algo", "framework", "created_at"]
            for field in required_fields:
                if field not in metadata:
                    validation_results["errors"].append(f"Missing metadata field: {field}")
                else:
                    validation_results["checks"][f"metadata_{field}"] = True
            
            # Overall validation result
            validation_results["passed"] = len(validation_results["errors"]) == 0
            
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {e}")
        
        return validation_results


def create_experiment_registry(config: Optional[Dict[str, Any]] = None) -> ExperimentRegistry:
    """
    Create an ExperimentRegistry with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ExperimentRegistry instance
    """
    default_config = {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        "s3_bucket": os.getenv("MODEL_S3_BUCKET"),
        "s3_prefix": "models",
        "local_cache_dir": "cache/models",
        "default_experiment": "intraday_trading",
        "enable_wandb": os.getenv("ENABLE_WANDB", "false").lower() == "true",
        "wandb_project": "intraday-jules",
        "auto_validate": True,
        "retention_days": 90
    }
    
    if config:
        default_config.update(config)
    
    return ExperimentRegistry(ExperimentConfig(**default_config))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create registry
    registry = create_experiment_registry({
        "tracking_uri": "sqlite:///test_mlflow.db",
        "s3_bucket": None,  # Use local storage for testing
        "enable_wandb": False
    })
    
    # Example: Register a model (would normally come from TrainerAgent)
    print("Example usage of ExperimentRegistry:")
    print("1. Create registry ✓")
    print("2. Register model (requires actual model files)")
    print("3. List models")
    print("4. Get specific version")
    print("5. Promote to production")
    
    # List existing models
    models = registry.list_model_versions(limit=10)
    print(f"Found {len(models)} existing model versions")
    
    for model in models[:3]:  # Show first 3
        print(f"  - {model.version_id} ({model.algorithm}) - {model.status}")