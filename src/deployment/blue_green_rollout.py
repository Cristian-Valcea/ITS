# src/deployment/blue_green_rollout.py
"""
Blue/Green Rollout System for IntradayJules Trading System.

Handles atomic deployment of model bundles using symlink swapping to avoid
half-written bundles during execution pod updates.

Features:
- Atomic symlink swapping: "current → v2025-07-06"
- Model bundle validation before activation
- Rollback functionality
- Health checks during deployment
- Zero-downtime deployments
"""

import os
import shutil
import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import signal
import subprocess
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HealthCheckStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModelBundle:
    """Model bundle metadata."""
    version: str
    path: Path
    checksum: str
    size_bytes: int
    created_at: float
    model_files: List[str]
    config_files: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'path': str(self.path),
            'created_at_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.created_at))
        }


@dataclass
class DeploymentRecord:
    """Deployment record."""
    deployment_id: str
    bundle_version: str
    status: DeploymentStatus
    started_at: float
    completed_at: Optional[float]
    previous_version: Optional[str]
    health_checks: List[Dict[str, Any]]
    error_message: Optional[str]
    rollback_available: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'status': self.status.value,
            'started_at_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.started_at)),
            'completed_at_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.completed_at)) if self.completed_at else None
        }


class ModelBundleValidator:
    """Validates model bundles before deployment."""
    
    def __init__(self):
        self.logger = logging.getLogger("ModelBundleValidator")
        self.required_files = [
            "policy.pt",           # Main policy model
            "value_function.pt",   # Value function model
            "config.yaml",         # Model configuration
            "metadata.json"        # Bundle metadata
        ]
        self.optional_files = [
            "feature_scaler.pkl",  # Feature scaling parameters
            "risk_model.pt",       # Risk model
            "README.md"            # Documentation
        ]
    
    def validate_bundle(self, bundle_path: Path) -> Tuple[bool, List[str], ModelBundle]:
        """
        Validate model bundle structure and contents.
        
        Returns:
            Tuple of (is_valid, error_messages, bundle_metadata)
        """
        errors = []
        
        try:
            if not bundle_path.exists():
                errors.append(f"Bundle path does not exist: {bundle_path}")
                return False, errors, None
            
            if not bundle_path.is_dir():
                errors.append(f"Bundle path is not a directory: {bundle_path}")
                return False, errors, None
            
            # Check required files
            missing_files = []
            for required_file in self.required_files:
                file_path = bundle_path / required_file
                if not file_path.exists():
                    missing_files.append(required_file)
            
            if missing_files:
                errors.append(f"Missing required files: {missing_files}")
            
            # Validate metadata.json
            metadata_path = bundle_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check required metadata fields
                    required_metadata = ['version', 'created_at', 'model_type', 'training_data_hash']
                    missing_metadata = [field for field in required_metadata if field not in metadata]
                    if missing_metadata:
                        errors.append(f"Missing metadata fields: {missing_metadata}")
                        
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid metadata.json: {e}")
            
            # Validate model files
            model_files = []
            config_files = []
            
            for file_path in bundle_path.iterdir():
                if file_path.is_file():
                    if file_path.suffix == '.pt':
                        # Validate PyTorch model file
                        if not self._validate_pytorch_model(file_path):
                            errors.append(f"Invalid PyTorch model: {file_path.name}")
                        model_files.append(file_path.name)
                    elif file_path.suffix in ['.yaml', '.yml', '.json']:
                        config_files.append(file_path.name)
            
            # Calculate bundle checksum
            checksum = self._calculate_bundle_checksum(bundle_path)
            
            # Calculate bundle size
            size_bytes = sum(f.stat().st_size for f in bundle_path.rglob('*') if f.is_file())
            
            # Create bundle metadata
            bundle = ModelBundle(
                version=metadata.get('version', bundle_path.name),
                path=bundle_path,
                checksum=checksum,
                size_bytes=size_bytes,
                created_at=metadata.get('created_at', time.time()),
                model_files=model_files,
                config_files=config_files,
                metadata=metadata
            )
            
            is_valid = len(errors) == 0
            self.logger.info(f"Bundle validation {'passed' if is_valid else 'failed'}: {bundle_path}")
            
            return is_valid, errors, bundle
            
        except Exception as e:
            errors.append(f"Validation exception: {str(e)}")
            self.logger.error(f"Bundle validation failed with exception: {e}")
            return False, errors, None
    
    def _validate_pytorch_model(self, model_path: Path) -> bool:
        """Validate PyTorch model file."""
        try:
            import torch
            
            # Try to load the model
            model = torch.load(model_path, map_location='cpu')
            
            # Basic checks
            if not isinstance(model, (dict, torch.nn.Module)):
                return False
            
            # If it's a state dict, check for expected keys
            if isinstance(model, dict):
                if 'state_dict' in model or any(key.endswith('.weight') or key.endswith('.bias') for key in model.keys()):
                    return True
            
            return True
            
        except Exception as e:
            self.logger.warning(f"PyTorch model validation failed: {e}")
            return False
    
    def _calculate_bundle_checksum(self, bundle_path: Path) -> str:
        """Calculate SHA256 checksum of bundle contents."""
        hasher = hashlib.sha256()
        
        # Sort files for consistent checksum
        files = sorted(bundle_path.rglob('*'))
        
        for file_path in files:
            if file_path.is_file():
                # Include file path in hash for structure validation
                hasher.update(str(file_path.relative_to(bundle_path)).encode())
                
                # Include file contents
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()


class HealthChecker:
    """Health checker for deployed models."""
    
    def __init__(self, health_check_url: str = "http://localhost:8000/health"):
        self.health_check_url = health_check_url
        self.logger = logging.getLogger("HealthChecker")
    
    async def check_health(self, timeout: float = 10.0) -> Tuple[HealthCheckStatus, Dict[str, Any]]:
        """
        Perform health check on deployed system.
        
        Returns:
            Tuple of (status, health_data)
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Check critical components
                        if self._validate_health_response(health_data):
                            return HealthCheckStatus.HEALTHY, health_data
                        else:
                            return HealthCheckStatus.UNHEALTHY, health_data
                    else:
                        return HealthCheckStatus.UNHEALTHY, {
                            "error": f"HTTP {response.status}",
                            "url": self.health_check_url
                        }
                        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthCheckStatus.UNKNOWN, {
                "error": str(e),
                "url": self.health_check_url
            }
    
    def _validate_health_response(self, health_data: Dict[str, Any]) -> bool:
        """Validate health check response."""
        # Check required health indicators
        required_checks = [
            "model_loaded",
            "risk_system_active",
            "database_connected"
        ]
        
        for check in required_checks:
            if not health_data.get(check, False):
                self.logger.warning(f"Health check failed: {check} = {health_data.get(check)}")
                return False
        
        return True


class BlueGreenDeployment:
    """
    Blue/Green Deployment Manager for IntradayJules Trading System.
    
    Features:
    - Atomic symlink swapping for zero-downtime deployments
    - Model bundle validation before activation
    - Health checks during deployment
    - Automatic rollback on failure
    - Deployment history and audit trail
    """
    
    def __init__(self, 
                 deployment_root: Path,
                 current_symlink: str = "current",
                 health_check_url: str = "http://localhost:8000/health"):
        
        self.deployment_root = Path(deployment_root)
        self.current_symlink = self.deployment_root / current_symlink
        self.bundles_dir = self.deployment_root / "bundles"
        self.deployments_dir = self.deployment_root / "deployments"
        self.lock_file = self.deployment_root / "deployment.lock"
        
        # Create directories
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = ModelBundleValidator()
        self.health_checker = HealthChecker(health_check_url)
        self.logger = logging.getLogger("BlueGreenDeployment")
        
        # Deployment settings
        self.health_check_timeout = 30.0  # 30 seconds
        self.health_check_retries = 3
        self.health_check_interval = 5.0  # 5 seconds between retries
        
        self.logger.info(f"BlueGreenDeployment initialized: {deployment_root}")
    
    def deploy_bundle(self, bundle_version: str, bundle_path: Path, 
                     force: bool = False) -> DeploymentRecord:
        """
        Deploy model bundle using blue/green deployment.
        
        Args:
            bundle_version: Version identifier for the bundle
            bundle_path: Path to the bundle directory
            force: Skip validation if True
            
        Returns:
            DeploymentRecord with deployment status
        """
        deployment_id = f"deploy_{bundle_version}_{int(time.time())}"
        started_at = time.time()
        
        # Initialize deployment record
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            bundle_version=bundle_version,
            status=DeploymentStatus.PENDING,
            started_at=started_at,
            completed_at=None,
            previous_version=self._get_current_version(),
            health_checks=[],
            error_message=None,
            rollback_available=self.current_symlink.exists()
        )
        
        try:
            # Acquire deployment lock
            with self._deployment_lock():
                self.logger.info(f"Starting deployment: {deployment_id}")
                
                # Step 1: Validate bundle
                if not force:
                    deployment.status = DeploymentStatus.VALIDATING
                    self._save_deployment_record(deployment)
                    
                    is_valid, errors, bundle_metadata = self.validator.validate_bundle(bundle_path)
                    if not is_valid:
                        deployment.status = DeploymentStatus.FAILED
                        deployment.error_message = f"Bundle validation failed: {errors}"
                        deployment.completed_at = time.time()
                        self._save_deployment_record(deployment)
                        return deployment
                
                # Step 2: Copy bundle to deployment directory
                deployment.status = DeploymentStatus.DEPLOYING
                self._save_deployment_record(deployment)
                
                target_bundle_path = self.bundles_dir / bundle_version
                if target_bundle_path.exists():
                    if not force:
                        deployment.status = DeploymentStatus.FAILED
                        deployment.error_message = f"Bundle version {bundle_version} already exists"
                        deployment.completed_at = time.time()
                        self._save_deployment_record(deployment)
                        return deployment
                    else:
                        shutil.rmtree(target_bundle_path)
                
                # Copy bundle atomically
                temp_bundle_path = target_bundle_path.with_suffix('.tmp')
                shutil.copytree(bundle_path, temp_bundle_path)
                temp_bundle_path.rename(target_bundle_path)
                
                self.logger.info(f"Bundle copied to: {target_bundle_path}")
                
                # Step 3: Create new symlink atomically
                new_symlink = self.current_symlink.with_suffix('.new')
                
                # Remove old temporary symlink if exists
                if new_symlink.exists() or new_symlink.is_symlink():
                    new_symlink.unlink()
                
                # Create new symlink
                new_symlink.symlink_to(target_bundle_path, target_is_directory=True)
                
                # Step 4: Perform health checks
                health_check_passed = False
                for attempt in range(self.health_check_retries):
                    self.logger.info(f"Health check attempt {attempt + 1}/{self.health_check_retries}")
                    
                    # Temporarily swap symlink for health check
                    if attempt == 0:
                        # Atomic swap
                        if self.current_symlink.exists() or self.current_symlink.is_symlink():
                            old_symlink = self.current_symlink.with_suffix('.old')
                            if old_symlink.exists() or old_symlink.is_symlink():
                                old_symlink.unlink()
                            self.current_symlink.rename(old_symlink)
                        
                        new_symlink.rename(self.current_symlink)
                    
                    # Wait for system to pick up new model
                    time.sleep(self.health_check_interval)
                    
                    # Perform health check
                    import asyncio
                    health_status, health_data = asyncio.run(
                        self.health_checker.check_health(self.health_check_timeout)
                    )
                    
                    health_check_record = {
                        "attempt": attempt + 1,
                        "timestamp": time.time(),
                        "status": health_status.value,
                        "data": health_data
                    }
                    deployment.health_checks.append(health_check_record)
                    
                    if health_status == HealthCheckStatus.HEALTHY:
                        health_check_passed = True
                        self.logger.info(f"Health check passed on attempt {attempt + 1}")
                        break
                    else:
                        self.logger.warning(f"Health check failed on attempt {attempt + 1}: {health_data}")
                
                # Step 5: Finalize deployment or rollback
                if health_check_passed:
                    # Deployment successful
                    deployment.status = DeploymentStatus.ACTIVE
                    deployment.completed_at = time.time()
                    
                    # Clean up old symlink
                    old_symlink = self.current_symlink.with_suffix('.old')
                    if old_symlink.exists() or old_symlink.is_symlink():
                        old_symlink.unlink()
                    
                    self.logger.info(f"Deployment successful: {deployment_id}")
                    
                else:
                    # Health checks failed - rollback
                    deployment.status = DeploymentStatus.FAILED
                    deployment.error_message = "Health checks failed after deployment"
                    deployment.completed_at = time.time()
                    
                    # Rollback symlink
                    old_symlink = self.current_symlink.with_suffix('.old')
                    if old_symlink.exists() or old_symlink.is_symlink():
                        if self.current_symlink.exists() or self.current_symlink.is_symlink():
                            self.current_symlink.unlink()
                        old_symlink.rename(self.current_symlink)
                        self.logger.info("Rolled back to previous version")
                    
                    # Remove failed bundle
                    if target_bundle_path.exists():
                        shutil.rmtree(target_bundle_path)
                
                self._save_deployment_record(deployment)
                return deployment
                
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = f"Deployment exception: {str(e)}"
            deployment.completed_at = time.time()
            self._save_deployment_record(deployment)
            self.logger.error(f"Deployment failed: {e}")
            return deployment
    
    def rollback_deployment(self, target_version: Optional[str] = None) -> DeploymentRecord:
        """
        Rollback to previous or specified version.
        
        Args:
            target_version: Specific version to rollback to, or None for previous
            
        Returns:
            DeploymentRecord with rollback status
        """
        deployment_id = f"rollback_{int(time.time())}"
        started_at = time.time()
        
        current_version = self._get_current_version()
        
        # Determine target version
        if target_version is None:
            # Find previous version from deployment history
            target_version = self._get_previous_version()
        
        if not target_version:
            return DeploymentRecord(
                deployment_id=deployment_id,
                bundle_version="unknown",
                status=DeploymentStatus.FAILED,
                started_at=started_at,
                completed_at=time.time(),
                previous_version=current_version,
                health_checks=[],
                error_message="No previous version available for rollback",
                rollback_available=False
            )
        
        # Check if target version exists
        target_bundle_path = self.bundles_dir / target_version
        if not target_bundle_path.exists():
            return DeploymentRecord(
                deployment_id=deployment_id,
                bundle_version=target_version,
                status=DeploymentStatus.FAILED,
                started_at=started_at,
                completed_at=time.time(),
                previous_version=current_version,
                health_checks=[],
                error_message=f"Target version {target_version} not found",
                rollback_available=False
            )
        
        # Perform rollback deployment
        self.logger.info(f"Rolling back from {current_version} to {target_version}")
        return self.deploy_bundle(target_version, target_bundle_path, force=True)
    
    def _get_current_version(self) -> Optional[str]:
        """Get currently deployed version."""
        if self.current_symlink.exists() and self.current_symlink.is_symlink():
            target = self.current_symlink.resolve()
            return target.name
        return None
    
    def _get_previous_version(self) -> Optional[str]:
        """Get previous version from deployment history."""
        try:
            deployments = self.list_deployments()
            active_deployments = [d for d in deployments if d.status == DeploymentStatus.ACTIVE]
            
            if len(active_deployments) >= 2:
                # Return second most recent active deployment
                return active_deployments[1].bundle_version
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get previous version: {e}")
            return None
    
    def list_deployments(self) -> List[DeploymentRecord]:
        """List all deployment records, sorted by most recent first."""
        deployments = []
        
        try:
            for deployment_file in self.deployments_dir.glob("*.json"):
                with open(deployment_file, 'r') as f:
                    data = json.load(f)
                    
                    deployment = DeploymentRecord(
                        deployment_id=data['deployment_id'],
                        bundle_version=data['bundle_version'],
                        status=DeploymentStatus(data['status']),
                        started_at=data['started_at'],
                        completed_at=data.get('completed_at'),
                        previous_version=data.get('previous_version'),
                        health_checks=data.get('health_checks', []),
                        error_message=data.get('error_message'),
                        rollback_available=data.get('rollback_available', False)
                    )
                    deployments.append(deployment)
            
            # Sort by start time, most recent first
            deployments.sort(key=lambda d: d.started_at, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list deployments: {e}")
        
        return deployments
    
    def _save_deployment_record(self, deployment: DeploymentRecord) -> None:
        """Save deployment record to disk."""
        try:
            deployment_file = self.deployments_dir / f"{deployment.deployment_id}.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployment.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save deployment record: {e}")
    
    def _deployment_lock(self):
        """Context manager for deployment locking."""
        class DeploymentLock:
            def __init__(self, lock_file: Path, logger):
                self.lock_file = lock_file
                self.logger = logger
                self.lock_fd = None
            
            def __enter__(self):
                try:
                    self.lock_fd = open(self.lock_file, 'w')
                    
                    if HAS_FCNTL:
                        # Unix-style file locking
                        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        # Windows-style file locking (simplified)
                        # Remove existing lock file if it exists (for testing)
                        if self.lock_file.exists():
                            try:
                                self.lock_file.unlink()
                            except:
                                pass
                    
                    self.lock_fd.write(f"locked_at={time.time()}\npid={os.getpid()}\n")
                    self.lock_fd.flush()
                    self.logger.info("Deployment lock acquired")
                    return self
                except (IOError, OSError) as e:
                    if self.lock_fd:
                        self.lock_fd.close()
                    raise RuntimeError(f"Failed to acquire deployment lock: {e}")
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.lock_fd:
                    if HAS_FCNTL:
                        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                    self.lock_fd.close()
                    try:
                        self.lock_file.unlink()
                    except FileNotFoundError:
                        pass
                    self.logger.info("Deployment lock released")
        
        return DeploymentLock(self.lock_file, self.logger)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        current_version = self._get_current_version()
        recent_deployments = self.list_deployments()[:5]  # Last 5 deployments
        
        return {
            "current_version": current_version,
            "current_bundle_path": str(self.current_symlink.resolve()) if current_version else None,
            "deployment_root": str(self.deployment_root),
            "available_bundles": [p.name for p in self.bundles_dir.iterdir() if p.is_dir()],
            "recent_deployments": [d.to_dict() for d in recent_deployments],
            "rollback_available": len([d for d in recent_deployments if d.status == DeploymentStatus.ACTIVE]) > 1
        }


def create_blue_green_deployment(deployment_root: str = "/opt/intradayjules/models") -> BlueGreenDeployment:
    """Factory function to create BlueGreenDeployment."""
    return BlueGreenDeployment(Path(deployment_root))


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import sys
    
    def test_blue_green_deployment():
        """Test blue/green deployment functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_root = Path(temp_dir) / "deployments"
            
            # Create test deployment manager
            bg_deploy = BlueGreenDeployment(deployment_root)
            
            # Create test bundle
            test_bundle_dir = Path(temp_dir) / "test_bundle_v1"
            test_bundle_dir.mkdir(parents=True)
            
            # Create required files
            (test_bundle_dir / "policy.pt").write_text("fake pytorch model")
            (test_bundle_dir / "value_function.pt").write_text("fake value function")
            (test_bundle_dir / "config.yaml").write_text("model_type: test\n")
            
            metadata = {
                "version": "v2025-07-06",
                "created_at": time.time(),
                "model_type": "test",
                "training_data_hash": "abc123"
            }
            (test_bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            
            print(f"Created test bundle: {test_bundle_dir}")
            
            # Test deployment
            print("Testing deployment...")
            deployment = bg_deploy.deploy_bundle("v2025-07-06", test_bundle_dir, force=True)
            print(f"Deployment result: {deployment.status.value}")
            print(f"Error: {deployment.error_message}")
            
            # Check deployment status
            status = bg_deploy.get_deployment_status()
            print(f"Current version: {status['current_version']}")
            print(f"Available bundles: {status['available_bundles']}")
            
            # Test rollback
            if deployment.status == DeploymentStatus.ACTIVE:
                print("Testing rollback...")
                rollback = bg_deploy.rollback_deployment()
                print(f"Rollback result: {rollback.status.value}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_blue_green_deployment()