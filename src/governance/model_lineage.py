# src/governance/model_lineage.py
"""
Model Lineage Tracking Implementation.

Provides complete model lineage tracking including dataset hashing,
training data provenance, and model artifact versioning for regulatory
compliance and reproducibility.

Features:
- Dataset SHA-256 hashing for data provenance
- Complete training pipeline lineage
- Model artifact versioning and signing
- Reproducibility validation
- Regulatory compliance reporting
"""

import json
import hashlib
import time
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetFingerprint:
    """Dataset fingerprint for lineage tracking."""
    dataset_name: str
    dataset_path: str
    sha256_hash: str
    row_count: int
    column_count: int
    columns: List[str]
    data_types: Dict[str, str]
    date_range: Dict[str, str]  # start_date, end_date
    size_bytes: int
    created_timestamp: str
    source_systems: List[str]
    transformations_applied: List[str]


@dataclass
class ModelLineage:
    """Complete model lineage record."""
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    training_timestamp: str
    training_duration_seconds: float
    
    # Dataset lineage
    training_datasets: List[DatasetFingerprint]
    validation_datasets: List[DatasetFingerprint]
    test_datasets: List[DatasetFingerprint]
    
    # Training configuration
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    feature_columns: List[str]
    target_columns: List[str]
    
    # Model artifacts
    model_file_hash: str
    model_file_path: str
    model_size_bytes: int
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Environment and dependencies
    python_version: str
    dependencies: Dict[str, str]  # package_name -> version
    hardware_info: Dict[str, Any]
    
    # Governance
    created_by: str
    approved_by: Optional[str] = None
    approval_timestamp: Optional[str] = None
    compliance_tags: List[str] = None
    
    # Reproducibility
    random_seed: Optional[int] = None
    reproducibility_hash: Optional[str] = None


class DatasetHasher:
    """
    Dataset hashing utility for data lineage tracking.
    
    Provides deterministic hashing of datasets to track data provenance
    and ensure model training reproducibility.
    """
    
    def __init__(self):
        self.chunk_size = 8192  # For large file processing
    
    def hash_dataframe(self, df: pd.DataFrame, dataset_name: str = None) -> DatasetFingerprint:
        """
        Generate fingerprint for pandas DataFrame.
        
        Args:
            df: DataFrame to hash
            dataset_name: Name of the dataset
            
        Returns:
            DatasetFingerprint with complete dataset metadata
        """
        try:
            # Sort DataFrame for deterministic hashing
            df_sorted = df.sort_index().sort_index(axis=1)
            
            # Calculate SHA-256 hash
            hasher = hashlib.sha256()
            
            # Hash column names and types
            columns_info = [(col, str(df_sorted[col].dtype)) for col in df_sorted.columns]
            hasher.update(json.dumps(columns_info, sort_keys=True).encode())
            
            # Hash data in chunks for memory efficiency
            for chunk_start in range(0, len(df_sorted), 1000):
                chunk = df_sorted.iloc[chunk_start:chunk_start + 1000]
                chunk_bytes = chunk.to_csv(index=False).encode()
                hasher.update(chunk_bytes)
            
            sha256_hash = hasher.hexdigest()
            
            # Extract metadata
            date_columns = df_sorted.select_dtypes(include=['datetime64']).columns
            date_range = {}
            if len(date_columns) > 0:
                first_date_col = date_columns[0]
                date_range = {
                    'start_date': str(df_sorted[first_date_col].min()),
                    'end_date': str(df_sorted[first_date_col].max())
                }
            
            fingerprint = DatasetFingerprint(
                dataset_name=dataset_name or "unnamed_dataset",
                dataset_path="in_memory_dataframe",
                sha256_hash=sha256_hash,
                row_count=len(df_sorted),
                column_count=len(df_sorted.columns),
                columns=list(df_sorted.columns),
                data_types={col: str(dtype) for col, dtype in df_sorted.dtypes.items()},
                date_range=date_range,
                size_bytes=df_sorted.memory_usage(deep=True).sum(),
                created_timestamp=datetime.now(timezone.utc).isoformat(),
                source_systems=["pandas_dataframe"],
                transformations_applied=[]
            )
            
            logger.info(f"Dataset fingerprint generated: {dataset_name} -> {sha256_hash[:16]}...")
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to hash DataFrame: {e}")
            raise
    
    def hash_csv_file(self, file_path: Path, dataset_name: str = None) -> DatasetFingerprint:
        """
        Generate fingerprint for CSV file.
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name of the dataset
            
        Returns:
            DatasetFingerprint with complete file metadata
        """
        try:
            # Read CSV to get metadata
            df = pd.read_csv(file_path)
            
            # Calculate file hash
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            
            # Get file stats
            file_stats = file_path.stat()
            
            # Extract date range if possible
            date_columns = df.select_dtypes(include=['datetime64']).columns
            date_range = {}
            if len(date_columns) > 0:
                first_date_col = date_columns[0]
                date_range = {
                    'start_date': str(df[first_date_col].min()),
                    'end_date': str(df[first_date_col].max())
                }
            
            fingerprint = DatasetFingerprint(
                dataset_name=dataset_name or file_path.stem,
                dataset_path=str(file_path.absolute()),
                sha256_hash=file_hash,
                row_count=len(df),
                column_count=len(df.columns),
                columns=list(df.columns),
                data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                date_range=date_range,
                size_bytes=file_stats.st_size,
                created_timestamp=datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat(),
                source_systems=[f"csv_file:{file_path.name}"],
                transformations_applied=[]
            )
            
            logger.info(f"CSV file fingerprint generated: {file_path.name} -> {file_hash[:16]}...")
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to hash CSV file {file_path}: {e}")
            raise
    
    def hash_parquet_file(self, file_path: Path, dataset_name: str = None) -> DatasetFingerprint:
        """Generate fingerprint for Parquet file."""
        try:
            # Read Parquet to get metadata
            df = pd.read_parquet(file_path)
            
            # Calculate file hash
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            file_stats = file_path.stat()
            
            # Extract date range
            date_columns = df.select_dtypes(include=['datetime64']).columns
            date_range = {}
            if len(date_columns) > 0:
                first_date_col = date_columns[0]
                date_range = {
                    'start_date': str(df[first_date_col].min()),
                    'end_date': str(df[first_date_col].max())
                }
            
            fingerprint = DatasetFingerprint(
                dataset_name=dataset_name or file_path.stem,
                dataset_path=str(file_path.absolute()),
                sha256_hash=file_hash,
                row_count=len(df),
                column_count=len(df.columns),
                columns=list(df.columns),
                data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                date_range=date_range,
                size_bytes=file_stats.st_size,
                created_timestamp=datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat(),
                source_systems=[f"parquet_file:{file_path.name}"],
                transformations_applied=[]
            )
            
            logger.info(f"Parquet file fingerprint generated: {file_path.name} -> {file_hash[:16]}...")
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to hash Parquet file {file_path}: {e}")
            raise


class ModelLineageTracker:
    """
    Model lineage tracking system.
    
    Tracks complete model lineage including datasets, training configuration,
    model artifacts, and governance information for regulatory compliance.
    """
    
    def __init__(self, lineage_storage_path: str = "./model_lineage"):
        self.storage_path = Path(lineage_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.dataset_hasher = DatasetHasher()
        
        # Create lineage index
        self.index_file = self.storage_path / "lineage_index.json"
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump([], f)
        
        logger.info(f"Model lineage tracker initialized: {self.storage_path}")
    
    def start_training_session(self, 
                             model_name: str,
                             model_type: str,
                             training_config: Dict[str, Any],
                             created_by: str) -> str:
        """
        Start model training session and generate model ID.
        
        Returns:
            str: Unique model ID for this training session
        """
        timestamp = datetime.now(timezone.utc)
        model_id = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(training_config)) % 10000:04d}"
        
        # Create session directory
        session_dir = self.storage_path / model_id
        session_dir.mkdir(exist_ok=True)
        
        # Save initial session info
        session_info = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'training_config': training_config,
            'created_by': created_by,
            'session_started': timestamp.isoformat(),
            'status': 'TRAINING_STARTED'
        }
        
        with open(session_dir / "session_info.json", 'w') as f:
            json.dump(session_info, f, indent=2)
        
        logger.info(f"Training session started: {model_id}")
        return model_id
    
    def record_dataset_usage(self, 
                           model_id: str,
                           dataset: pd.DataFrame,
                           dataset_name: str,
                           dataset_type: str,  # 'training', 'validation', 'test'
                           transformations: List[str] = None) -> DatasetFingerprint:
        """
        Record dataset usage in model training.
        
        Args:
            model_id: Model training session ID
            dataset: Dataset used in training
            dataset_name: Name of the dataset
            dataset_type: Type of dataset usage
            transformations: List of transformations applied
            
        Returns:
            DatasetFingerprint: Generated dataset fingerprint
        """
        try:
            # Generate dataset fingerprint
            fingerprint = self.dataset_hasher.hash_dataframe(dataset, dataset_name)
            
            # Add transformation information
            if transformations:
                fingerprint.transformations_applied = transformations
            
            # Save dataset fingerprint
            session_dir = self.storage_path / model_id
            dataset_file = session_dir / f"dataset_{dataset_type}_{dataset_name}.json"
            
            with open(dataset_file, 'w') as f:
                json.dump(asdict(fingerprint), f, indent=2)
            
            logger.info(f"Dataset usage recorded: {model_id} -> {dataset_name} ({dataset_type})")
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to record dataset usage: {e}")
            raise
    
    def record_model_artifact(self, 
                            model_id: str,
                            model_file_path: Path,
                            model_version: str = "1.0.0") -> str:
        """
        Record model artifact with hash and metadata.
        
        Args:
            model_id: Model training session ID
            model_file_path: Path to model file
            model_version: Model version
            
        Returns:
            str: Model file hash
        """
        try:
            # Calculate model file hash
            hasher = hashlib.sha256()
            with open(model_file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            
            model_hash = hasher.hexdigest()
            file_stats = model_file_path.stat()
            
            # Copy model file to lineage storage
            session_dir = self.storage_path / model_id
            model_copy_path = session_dir / f"model_{model_version}.pt"
            shutil.copy2(model_file_path, model_copy_path)
            
            # Save model artifact metadata
            artifact_metadata = {
                'model_id': model_id,
                'model_version': model_version,
                'original_path': str(model_file_path.absolute()),
                'lineage_path': str(model_copy_path.absolute()),
                'sha256_hash': model_hash,
                'size_bytes': file_stats.st_size,
                'created_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(session_dir / f"model_artifact_{model_version}.json", 'w') as f:
                json.dump(artifact_metadata, f, indent=2)
            
            logger.info(f"Model artifact recorded: {model_id} -> {model_hash[:16]}...")
            return model_hash
            
        except Exception as e:
            logger.error(f"Failed to record model artifact: {e}")
            raise
    
    def complete_training_session(self,
                                model_id: str,
                                model_version: str,
                                hyperparameters: Dict[str, Any],
                                training_metrics: Dict[str, float],
                                validation_metrics: Dict[str, float],
                                test_metrics: Dict[str, float] = None,
                                feature_columns: List[str] = None,
                                target_columns: List[str] = None,
                                random_seed: int = None) -> ModelLineage:
        """
        Complete model training session and generate full lineage record.
        
        Returns:
            ModelLineage: Complete model lineage record
        """
        try:
            session_dir = self.storage_path / model_id
            
            # Load session info
            with open(session_dir / "session_info.json", 'r') as f:
                session_info = json.load(f)
            
            # Load dataset fingerprints
            training_datasets = []
            validation_datasets = []
            test_datasets = []
            
            for dataset_file in session_dir.glob("dataset_*.json"):
                with open(dataset_file, 'r') as f:
                    fingerprint_data = json.load(f)
                    fingerprint = DatasetFingerprint(**fingerprint_data)
                
                if "training" in dataset_file.name:
                    training_datasets.append(fingerprint)
                elif "validation" in dataset_file.name:
                    validation_datasets.append(fingerprint)
                elif "test" in dataset_file.name:
                    test_datasets.append(fingerprint)
            
            # Load model artifact info
            model_artifact_file = session_dir / f"model_artifact_{model_version}.json"
            with open(model_artifact_file, 'r') as f:
                artifact_info = json.load(f)
            
            # Get environment info
            dependencies = self._get_dependencies()
            hardware_info = self._get_hardware_info()
            
            # Calculate training duration
            start_time = datetime.fromisoformat(session_info['session_started'])
            end_time = datetime.now(timezone.utc)
            training_duration = (end_time - start_time).total_seconds()
            
            # Generate reproducibility hash
            reproducibility_data = {
                'hyperparameters': hyperparameters,
                'training_datasets': [fp.sha256_hash for fp in training_datasets],
                'validation_datasets': [fp.sha256_hash for fp in validation_datasets],
                'random_seed': random_seed,
                'feature_columns': sorted(feature_columns or []),
                'target_columns': sorted(target_columns or [])
            }
            reproducibility_hash = hashlib.sha256(
                json.dumps(reproducibility_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Create complete lineage record
            lineage = ModelLineage(
                model_id=model_id,
                model_name=session_info['model_name'],
                model_version=model_version,
                model_type=session_info['model_type'],
                training_timestamp=session_info['session_started'],
                training_duration_seconds=training_duration,
                
                training_datasets=training_datasets,
                validation_datasets=validation_datasets,
                test_datasets=test_datasets,
                
                hyperparameters=hyperparameters,
                training_config=session_info['training_config'],
                feature_columns=feature_columns or [],
                target_columns=target_columns or [],
                
                model_file_hash=artifact_info['sha256_hash'],
                model_file_path=artifact_info['lineage_path'],
                model_size_bytes=artifact_info['size_bytes'],
                
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics or {},
                
                python_version=dependencies.get('python', 'unknown'),
                dependencies=dependencies,
                hardware_info=hardware_info,
                
                created_by=session_info['created_by'],
                compliance_tags=['MODEL_LINEAGE', 'TRAINING_COMPLETE'],
                
                random_seed=random_seed,
                reproducibility_hash=reproducibility_hash
            )
            
            # Save complete lineage record
            lineage_file = session_dir / "complete_lineage.json"
            with open(lineage_file, 'w') as f:
                json.dump(asdict(lineage), f, indent=2)
            
            # Update index
            self._update_lineage_index(lineage)
            
            logger.info(f"Training session completed: {model_id}")
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to complete training session: {e}")
            raise
    
    def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get complete model lineage by model ID."""
        try:
            lineage_file = self.storage_path / model_id / "complete_lineage.json"
            if not lineage_file.exists():
                return None
            
            with open(lineage_file, 'r') as f:
                lineage_data = json.load(f)
                return ModelLineage(**lineage_data)
                
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return None
    
    def validate_reproducibility(self, model_id: str, 
                               new_training_datasets: List[DatasetFingerprint]) -> bool:
        """
        Validate if model can be reproduced with given datasets.
        
        Args:
            model_id: Original model ID
            new_training_datasets: New dataset fingerprints
            
        Returns:
            bool: True if reproducible, False otherwise
        """
        try:
            original_lineage = self.get_model_lineage(model_id)
            if not original_lineage:
                return False
            
            # Compare dataset hashes
            original_hashes = {fp.sha256_hash for fp in original_lineage.training_datasets}
            new_hashes = {fp.sha256_hash for fp in new_training_datasets}
            
            datasets_match = original_hashes == new_hashes
            
            logger.info(f"Reproducibility validation: {model_id} -> {'PASS' if datasets_match else 'FAIL'}")
            return datasets_match
            
        except Exception as e:
            logger.error(f"Failed to validate reproducibility: {e}")
            return False
    
    def generate_lineage_report(self, model_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive lineage report for compliance."""
        try:
            # Load index
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # Filter by model IDs if specified
            if model_ids:
                index = [entry for entry in index if entry['model_id'] in model_ids]
            
            # Generate report
            report = {
                'report_generated': datetime.now(timezone.utc).isoformat(),
                'total_models': len(index),
                'models': [],
                'dataset_usage_summary': {},
                'compliance_summary': {
                    'models_with_complete_lineage': 0,
                    'models_with_dataset_hashes': 0,
                    'models_with_reproducibility_hash': 0
                }
            }
            
            for entry in index:
                lineage = self.get_model_lineage(entry['model_id'])
                if lineage:
                    model_summary = {
                        'model_id': lineage.model_id,
                        'model_name': lineage.model_name,
                        'model_version': lineage.model_version,
                        'training_timestamp': lineage.training_timestamp,
                        'created_by': lineage.created_by,
                        'approved_by': lineage.approved_by,
                        'dataset_count': len(lineage.training_datasets) + len(lineage.validation_datasets),
                        'model_file_hash': lineage.model_file_hash,
                        'reproducibility_hash': lineage.reproducibility_hash,
                        'compliance_tags': lineage.compliance_tags
                    }
                    report['models'].append(model_summary)
                    
                    # Update compliance summary
                    if lineage.model_file_hash:
                        report['compliance_summary']['models_with_complete_lineage'] += 1
                    if lineage.training_datasets:
                        report['compliance_summary']['models_with_dataset_hashes'] += 1
                    if lineage.reproducibility_hash:
                        report['compliance_summary']['models_with_reproducibility_hash'] += 1
                    
                    # Track dataset usage
                    for dataset in lineage.training_datasets + lineage.validation_datasets:
                        dataset_name = dataset.dataset_name
                        if dataset_name not in report['dataset_usage_summary']:
                            report['dataset_usage_summary'][dataset_name] = 0
                        report['dataset_usage_summary'][dataset_name] += 1
            
            logger.info(f"Lineage report generated: {len(index)} models")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate lineage report: {e}")
            raise
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current Python environment dependencies."""
        try:
            import sys
            import pkg_resources
            
            dependencies = {
                'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
            
            # Get installed packages
            for dist in pkg_resources.working_set:
                dependencies[dist.project_name] = dist.version
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Failed to get dependencies: {e}")
            return {'python': 'unknown'}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for reproducibility."""
        try:
            import platform
            import psutil
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'python_implementation': platform.python_implementation()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get hardware info: {e}")
            return {'platform': 'unknown'}
    
    def _update_lineage_index(self, lineage: ModelLineage):
        """Update lineage index with new model."""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # Add new entry
            index.append({
                'model_id': lineage.model_id,
                'model_name': lineage.model_name,
                'model_version': lineage.model_version,
                'training_timestamp': lineage.training_timestamp,
                'created_by': lineage.created_by,
                'model_file_hash': lineage.model_file_hash,
                'reproducibility_hash': lineage.reproducibility_hash
            })
            
            # Sort by timestamp
            index.sort(key=lambda x: x['training_timestamp'], reverse=True)
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update lineage index: {e}")
            raise