# src/deployment/model_deployment_service.py
"""
Production Model Deployment Service with Semantic Versioning.

This service provides a production-ready interface for deploying and managing
ML models with semantic versioning. Production systems can request specific
model versions like "give me policy v2025-07-06--18h51" instead of UUIDs.

Key Features:
- Semantic version-based model retrieval
- Hot model swapping without downtime
- Model health monitoring and validation
- Rollback capabilities
- A/B testing support
- Performance monitoring and SLA tracking
- Automated failover to previous versions

Usage:
    # Initialize deployment service
    service = ModelDeploymentService(
        registry_config={'tracking_uri': 'http://mlflow-server:5000'},
        deployment_config={'health_check_interval': 30}
    )
    
    # Deploy specific version
    success = service.deploy_model_version("v2025-07-06--18h51")
    
    # Get current production model
    model = service.get_current_model()
    prediction = model.predict(observation)
    
    # Rollback if needed
    service.rollback_to_previous_version()
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import tempfile
import shutil

# Core dependencies
import torch
import numpy as np

# Import experiment registry
try:
    from ..training.experiment_registry import ExperimentRegistry, create_experiment_registry, ModelVersionInfo
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# Optional monitoring dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class DeploymentConfig:
    """Configuration for model deployment service."""
    # Model management
    deployment_dir: str = "production/models"
    backup_dir: str = "production/backups"
    max_model_cache: int = 5  # Number of models to keep in memory
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    performance_sla_ms: float = 100.0  # milliseconds
    error_rate_threshold: float = 0.05  # 5% error rate
    
    # Deployment safety
    validation_required: bool = True
    canary_deployment: bool = False
    canary_traffic_percent: float = 10.0
    rollback_on_failure: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    log_predictions: bool = False
    
    # Registry connection
    registry_config: Dict[str, Any] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for deployed models."""
    version_id: str
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    error_rate: float = 0.0
    last_prediction_time: Optional[datetime] = None
    deployment_time: Optional[datetime] = None


@dataclass
class DeploymentStatus:
    """Current deployment status."""
    current_version: Optional[str] = None
    previous_version: Optional[str] = None
    deployment_time: Optional[datetime] = None
    status: str = "idle"  # idle, deploying, active, error, rollback
    health_status: str = "unknown"  # healthy, degraded, unhealthy
    canary_version: Optional[str] = None
    canary_traffic_percent: float = 0.0


class ProductionModel:
    """Wrapper for production model with monitoring and safety features."""
    
    def __init__(self, model_path: str, version_id: str, metadata: Dict[str, Any]):
        """
        Initialize production model wrapper.
        
        Args:
            model_path: Path to the model file
            version_id: Semantic version ID
            metadata: Model metadata
        """
        self.model_path = model_path
        self.version_id = version_id
        self.metadata = metadata
        self.logger = logging.getLogger(f"ProductionModel.{version_id}")
        
        # Load the model
        self.model = None
        self.load_time = None
        self.is_loaded = False
        
        # Performance tracking
        self.metrics = ModelPerformanceMetrics(version_id=version_id)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the TorchScript model."""
        try:
            start_time = time.time()
            self.model = torch.jit.load(self.model_path)
            self.model.eval()
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            self.logger.info(f"✅ Model loaded: {self.version_id} ({self.load_time:.3f}s)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {self.version_id}: {e}")
            self.is_loaded = False
            raise
    
    def predict(self, observation: np.ndarray, timeout_ms: float = 100.0) -> Optional[np.ndarray]:
        """
        Make prediction with monitoring and safety checks.
        
        Args:
            observation: Input observation
            timeout_ms: Prediction timeout in milliseconds
            
        Returns:
            Prediction array or None if failed
        """
        if not self.is_loaded:
            self.logger.error("❌ Model not loaded")
            self.metrics.failed_predictions += 1
            return None
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.from_numpy(observation).float()
            else:
                obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Make prediction with timeout
            with torch.no_grad():
                prediction = self.model(obs_tensor)
            
            # Convert back to numpy
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, success=True)
            
            # Check SLA
            if latency_ms > timeout_ms:
                self.logger.warning(f"⚠️  Prediction exceeded timeout: {latency_ms:.1f}ms > {timeout_ms}ms")
            
            return prediction
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, success=False)
            self.logger.error(f"❌ Prediction failed: {e}")
            return None
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """Update performance metrics."""
        self.metrics.total_predictions += 1
        self.metrics.last_prediction_time = datetime.now()
        
        if success:
            self.metrics.successful_predictions += 1
        else:
            self.metrics.failed_predictions += 1
        
        # Update latency metrics
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (self.metrics.total_predictions - 1) + latency_ms) /
            self.metrics.total_predictions
        )
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
        
        # Update error rate
        self.metrics.error_rate = (
            self.metrics.failed_predictions / self.metrics.total_predictions
            if self.metrics.total_predictions > 0 else 0.0
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status."""
        return {
            'version_id': self.version_id,
            'is_loaded': self.is_loaded,
            'load_time_s': self.load_time,
            'metrics': asdict(self.metrics),
            'memory_usage_mb': self._get_memory_usage(),
            'last_health_check': datetime.now().isoformat()
        }
    
    def _get_memory_usage(self) -> float:
        """Get model memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            else:
                return 0.0
        except:
            return 0.0


class ModelDeploymentService:
    """
    Production model deployment service with semantic versioning.
    
    This service manages the deployment and lifecycle of ML models in production,
    providing semantic version-based retrieval, health monitoring, and automated
    rollback capabilities.
    """
    
    def __init__(self, config: Union[DeploymentConfig, Dict[str, Any]]):
        """
        Initialize the model deployment service.
        
        Args:
            config: DeploymentConfig instance or dictionary
        """
        if isinstance(config, dict):
            self.config = DeploymentConfig(**config)
        else:
            self.config = config
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize experiment registry
        self.registry = None
        if self.config.registry_config and REGISTRY_AVAILABLE:
            try:
                self.registry = create_experiment_registry(self.config.registry_config)
                self.logger.info("✅ Connected to experiment registry")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to registry: {e}")
        
        # Create directories
        Path(self.config.deployment_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Model management
        self.current_model: Optional[ProductionModel] = None
        self.previous_model: Optional[ProductionModel] = None
        self.canary_model: Optional[ProductionModel] = None
        self.model_cache: Dict[str, ProductionModel] = {}
        
        # Deployment status
        self.status = DeploymentStatus()
        
        # Monitoring
        self.health_check_thread = None
        self.health_check_running = False
        
        # Metrics
        if self.config.enable_metrics and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # Start health monitoring
        if self.config.health_check_interval > 0:
            self.start_health_monitoring()
        
        self.logger.info("🚀 ModelDeploymentService initialized")
    
    def deploy_model_version(self, version_id: str, force: bool = False) -> bool:
        """
        Deploy a specific model version to production.
        
        Args:
            version_id: Semantic version ID (e.g., "v2025-07-06--18h51")
            force: Force deployment even if validation fails
            
        Returns:
            True if deployment successful
        """
        self.logger.info(f"🚀 Deploying model version: {version_id}")
        
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return False
        
        try:
            # Update status
            self.status.status = "deploying"
            
            # Get model information
            model_info = self.registry.get_model_version(version_id)
            if not model_info:
                self.logger.error(f"❌ Model version not found: {version_id}")
                self.status.status = "error"
                return False
            
            # Validate model if required
            if self.config.validation_required and not force:
                if model_info.status not in ["validated", "production"]:
                    self.logger.error(f"❌ Model {version_id} is not validated (status: {model_info.status})")
                    self.status.status = "error"
                    return False
            
            # Download model
            model_path = self._download_model_for_deployment(version_id)
            if not model_path:
                self.logger.error(f"❌ Failed to download model: {version_id}")
                self.status.status = "error"
                return False
            
            # Load model metadata
            metadata_path = Path(model_path) / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create production model
            policy_path = Path(model_path) / "policy.pt"
            new_model = ProductionModel(str(policy_path), version_id, metadata)
            
            # Test the new model
            if not self._test_model_deployment(new_model):
                self.logger.error(f"❌ Model deployment test failed: {version_id}")
                if not force:
                    self.status.status = "error"
                    return False
            
            # Backup current model
            if self.current_model:
                self.previous_model = self.current_model
                self.status.previous_version = self.current_model.version_id
            
            # Deploy new model
            self.current_model = new_model
            self.status.current_version = version_id
            self.status.deployment_time = datetime.now()
            self.status.status = "active"
            
            # Add to cache
            self.model_cache[version_id] = new_model
            self._cleanup_model_cache()
            
            # Update registry
            if model_info.status != "production":
                self.registry.promote_to_production(version_id, {
                    'deployed_at': datetime.now().isoformat(),
                    'deployment_service': 'ModelDeploymentService'
                })
            
            self.logger.info(f"✅ Model deployed successfully: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            self.status.status = "error"
            import traceback
            self.logger.debug(f"Deployment error traceback: {traceback.format_exc()}")
            return False
    
    def get_current_model(self) -> Optional[ProductionModel]:
        """Get the current production model."""
        return self.current_model
    
    def predict(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """
        Make prediction using current production model.
        
        Args:
            observation: Input observation
            
        Returns:
            Prediction or None if failed
        """
        if not self.current_model:
            self.logger.error("❌ No model deployed")
            return None
        
        # Handle canary deployment
        if self.canary_model and self._should_use_canary():
            return self.canary_model.predict(observation, self.config.performance_sla_ms)
        else:
            return self.current_model.predict(observation, self.config.performance_sla_ms)
    
    def rollback_to_previous_version(self) -> bool:
        """
        Rollback to the previous model version.
        
        Returns:
            True if rollback successful
        """
        if not self.previous_model:
            self.logger.error("❌ No previous model available for rollback")
            return False
        
        self.logger.info(f"🔄 Rolling back to previous version: {self.previous_model.version_id}")
        
        try:
            # Swap models
            temp_model = self.current_model
            self.current_model = self.previous_model
            self.previous_model = temp_model
            
            # Update status
            self.status.current_version = self.current_model.version_id
            self.status.previous_version = self.previous_model.version_id if self.previous_model else None
            self.status.deployment_time = datetime.now()
            self.status.status = "active"
            
            self.logger.info(f"✅ Rollback successful: {self.current_model.version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def deploy_canary_version(self, version_id: str, traffic_percent: float = None) -> bool:
        """
        Deploy a canary version for A/B testing.
        
        Args:
            version_id: Canary version ID
            traffic_percent: Percentage of traffic to route to canary
            
        Returns:
            True if canary deployment successful
        """
        if traffic_percent is None:
            traffic_percent = self.config.canary_traffic_percent
        
        self.logger.info(f"🧪 Deploying canary version: {version_id} ({traffic_percent}% traffic)")
        
        if not self.registry:
            self.logger.error("❌ No experiment registry available")
            return False
        
        try:
            # Download and create canary model
            model_path = self._download_model_for_deployment(version_id)
            if not model_path:
                return False
            
            metadata_path = Path(model_path) / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            policy_path = Path(model_path) / "policy.pt"
            canary_model = ProductionModel(str(policy_path), version_id, metadata)
            
            # Test canary model
            if not self._test_model_deployment(canary_model):
                self.logger.error(f"❌ Canary model test failed: {version_id}")
                return False
            
            # Deploy canary
            self.canary_model = canary_model
            self.status.canary_version = version_id
            self.status.canary_traffic_percent = traffic_percent
            
            self.logger.info(f"✅ Canary deployed: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Canary deployment failed: {e}")
            return False
    
    def promote_canary_to_production(self) -> bool:
        """
        Promote canary version to production.
        
        Returns:
            True if promotion successful
        """
        if not self.canary_model:
            self.logger.error("❌ No canary model to promote")
            return False
        
        self.logger.info(f"⬆️  Promoting canary to production: {self.canary_model.version_id}")
        
        # Backup current model
        if self.current_model:
            self.previous_model = self.current_model
            self.status.previous_version = self.current_model.version_id
        
        # Promote canary
        self.current_model = self.canary_model
        self.status.current_version = self.canary_model.version_id
        self.status.deployment_time = datetime.now()
        
        # Clear canary
        self.canary_model = None
        self.status.canary_version = None
        self.status.canary_traffic_percent = 0.0
        
        self.logger.info(f"✅ Canary promoted to production: {self.current_model.version_id}")
        return True
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        status_dict = asdict(self.status)
        
        # Add model health information
        if self.current_model:
            status_dict['current_model_health'] = self.current_model.get_health_status()
        
        if self.canary_model:
            status_dict['canary_model_health'] = self.canary_model.get_health_status()
        
        # Add system health
        status_dict['system_health'] = self._get_system_health()
        
        return status_dict
    
    def list_available_versions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available model versions for deployment.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of available model versions
        """
        if not self.registry:
            return []
        
        models = self.registry.list_model_versions(
            status="validated",  # Only show validated models
            limit=limit
        )
        
        return [
            {
                'version_id': model.version_id,
                'algorithm': model.algorithm,
                'created_at': model.created_at.isoformat(),
                'status': model.status,
                'metrics': model.metrics
            }
            for model in models
        ]
    
    def start_health_monitoring(self):
        """Start health monitoring thread."""
        if self.health_check_running:
            return
        
        self.health_check_running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        self.logger.info("✅ Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health monitoring thread."""
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        self.logger.info("🛑 Health monitoring stopped")
    
    def _download_model_for_deployment(self, version_id: str) -> Optional[str]:
        """Download model for deployment."""
        if not self.registry:
            return None
        
        deployment_path = Path(self.config.deployment_dir) / version_id
        
        # Check if already downloaded
        if deployment_path.exists():
            policy_path = deployment_path / "policy.pt"
            metadata_path = deployment_path / "metadata.json"
            
            if policy_path.exists() and metadata_path.exists():
                self.logger.info(f"📁 Using cached model: {version_id}")
                return str(deployment_path)
        
        # Download from registry
        self.logger.info(f"📥 Downloading model: {version_id}")
        return self.registry.download_model(version_id, str(deployment_path))
    
    def _test_model_deployment(self, model: ProductionModel) -> bool:
        """Test model deployment with sample data."""
        try:
            # Create sample observation (adjust based on your model's input shape)
            sample_obs = np.random.randn(10)  # Adjust size as needed
            
            # Test prediction
            prediction = model.predict(sample_obs, timeout_ms=self.config.performance_sla_ms * 2)
            
            if prediction is None:
                return False
            
            # Check latency
            if model.metrics.avg_latency_ms > self.config.performance_sla_ms:
                self.logger.warning(f"⚠️  Model latency exceeds SLA: {model.metrics.avg_latency_ms:.1f}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model test failed: {e}")
            return False
    
    def _should_use_canary(self) -> bool:
        """Determine if canary model should be used for this request."""
        if not self.canary_model or self.status.canary_traffic_percent <= 0:
            return False
        
        # Simple random routing based on traffic percentage
        import random
        return random.random() * 100 < self.status.canary_traffic_percent
    
    def _cleanup_model_cache(self):
        """Clean up model cache to stay within limits."""
        if len(self.model_cache) <= self.config.max_model_cache:
            return
        
        # Remove oldest models (simple LRU)
        sorted_models = sorted(
            self.model_cache.items(),
            key=lambda x: x[1].metrics.last_prediction_time or datetime.min
        )
        
        models_to_remove = len(self.model_cache) - self.config.max_model_cache
        
        for version_id, _ in sorted_models[:models_to_remove]:
            if version_id not in [self.status.current_version, self.status.previous_version]:
                del self.model_cache[version_id]
                self.logger.info(f"🗑️  Removed model from cache: {version_id}")
    
    def _health_check_loop(self):
        """Health monitoring loop."""
        while self.health_check_running:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _perform_health_check(self):
        """Perform health check on current model."""
        if not self.current_model:
            self.status.health_status = "unhealthy"
            return
        
        try:
            # Check model health
            health = self.current_model.get_health_status()
            
            # Check error rate
            error_rate = health['metrics']['error_rate']
            avg_latency = health['metrics']['avg_latency_ms']
            
            if error_rate > self.config.error_rate_threshold:
                self.status.health_status = "degraded"
                self.logger.warning(f"⚠️  High error rate: {error_rate:.1%}")
                
                # Auto-rollback if configured
                if self.config.rollback_on_failure and self.previous_model:
                    self.logger.warning("🔄 Auto-rollback triggered due to high error rate")
                    self.rollback_to_previous_version()
                    
            elif avg_latency > self.config.performance_sla_ms * 1.5:
                self.status.health_status = "degraded"
                self.logger.warning(f"⚠️  High latency: {avg_latency:.1f}ms")
                
            else:
                self.status.health_status = "healthy"
            
        except Exception as e:
            self.status.health_status = "unhealthy"
            self.logger.error(f"❌ Health check error: {e}")
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'deployment_service_status': 'running'
        }
        
        if PSUTIL_AVAILABLE:
            try:
                health.update({
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                })
            except:
                pass
        
        return health
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            
            # Prediction metrics
            self.prediction_counter = Counter(
                'model_predictions_total',
                'Total number of predictions',
                ['version_id', 'status']
            )
            
            self.prediction_latency = Histogram(
                'model_prediction_latency_seconds',
                'Prediction latency in seconds',
                ['version_id']
            )
            
            self.error_rate_gauge = Gauge(
                'model_error_rate',
                'Current model error rate',
                ['version_id']
            )
            
            # Start metrics server
            start_http_server(self.config.metrics_port)
            self.logger.info(f"📊 Prometheus metrics server started on port {self.config.metrics_port}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Prometheus metrics: {e}")


def create_deployment_service(config: Optional[Dict[str, Any]] = None) -> ModelDeploymentService:
    """
    Create a ModelDeploymentService with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ModelDeploymentService instance
    """
    default_config = {
        'deployment_dir': 'production/models',
        'backup_dir': 'production/backups',
        'health_check_interval': 30,
        'performance_sla_ms': 100.0,
        'validation_required': True,
        'rollback_on_failure': True,
        'enable_metrics': True,
        'registry_config': {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
            's3_bucket': os.getenv('MODEL_S3_BUCKET')
        }
    }
    
    if config:
        default_config.update(config)
    
    return ModelDeploymentService(DeploymentConfig(**default_config))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create deployment service
    service = create_deployment_service({
        'registry_config': {
            'tracking_uri': 'sqlite:///test_mlflow.db',
            's3_bucket': None
        },
        'health_check_interval': 10
    })
    
    print("Model Deployment Service Example:")
    print("1. Service initialized ✓")
    print("2. List available versions")
    print("3. Deploy specific version")
    print("4. Monitor health")
    print("5. Rollback if needed")
    
    # List available versions
    versions = service.list_available_versions(limit=5)
    print(f"Available versions: {len(versions)}")
    
    # Get deployment status
    status = service.get_deployment_status()
    print(f"Deployment status: {status['status']}")
    
    # Example: Deploy a version (would require actual model)
    # success = service.deploy_model_version("v2025-07-06--18h51")
    # print(f"Deployment success: {success}")
    
    # Stop service
    service.stop_health_monitoring()
    print("Service stopped ✓")