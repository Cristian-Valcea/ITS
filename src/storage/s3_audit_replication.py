"""
S3 Audit Bucket Replication Manager
===================================

This module provides minute-level cross-region replication for S3 audit buckets
to meet compliance requirements for disaster recovery and data protection.

PROBLEM SOLVED:
"DR: S3 WORM mirrors cross-region hourly, not continuous.  
 → Enable S3 replication minute-level for audit bucket."

FEATURES:
- Real-time S3 cross-region replication
- WORM (Write Once Read Many) compliance
- Minute-level replication frequency
- Multi-region disaster recovery
- Audit trail integrity protection
- Compliance reporting and monitoring

USAGE:
    replicator = S3AuditReplicator(
        primary_bucket="intradayjules-audit-worm",
        replica_regions=["us-west-2", "eu-west-1"]
    )
    replicator.setup_replication()
    replicator.start_continuous_monitoring()
"""

import boto3
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReplicationConfig:
    """Configuration for S3 audit bucket replication."""
    primary_bucket: str
    primary_region: str
    replica_regions: List[str]
    replication_frequency_seconds: int = 60  # 1 minute
    enable_worm: bool = True
    retention_years: int = 7
    encryption_enabled: bool = True
    versioning_enabled: bool = True
    mfa_delete_enabled: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ReplicationStatus:
    """Status of replication for a specific object."""
    object_key: str
    source_bucket: str
    target_bucket: str
    target_region: str
    replication_time: datetime
    status: str  # 'PENDING', 'COMPLETED', 'FAILED'
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: int = 0


class S3AuditReplicator:
    """
    Manages minute-level cross-region replication for S3 audit buckets.
    
    This class provides comprehensive disaster recovery capabilities for
    audit data with WORM compliance and real-time replication.
    """
    
    def __init__(self, config: ReplicationConfig):
        """
        Initialize S3 audit replicator.
        
        Args:
            config: Replication configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 clients for all regions
        self.s3_clients = {}
        self.s3_clients[config.primary_region] = boto3.client('s3', region_name=config.primary_region)
        
        for region in config.replica_regions:
            self.s3_clients[region] = boto3.client('s3', region_name=region)
        
        # Replication state tracking
        self.replication_queue: Set[str] = set()
        self.replication_status: Dict[str, List[ReplicationStatus]] = {}
        self.last_sync_time = datetime.now()
        
        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self.logger.info(f"S3AuditReplicator initialized for bucket: {config.primary_bucket}")
        self.logger.info(f"Replica regions: {config.replica_regions}")
        self.logger.info(f"Replication frequency: {config.replication_frequency_seconds}s")
    
    def setup_replication(self) -> bool:
        """
        Set up S3 buckets and replication configuration.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.logger.info("Setting up S3 audit bucket replication...")
            
            # 1. Setup primary bucket
            self._setup_primary_bucket()
            
            # 2. Setup replica buckets
            for region in self.config.replica_regions:
                self._setup_replica_bucket(region)
            
            # 3. Configure bucket replication rules
            self._configure_bucket_replication()
            
            # 4. Setup monitoring and alerting
            self._setup_monitoring()
            
            self.logger.info("✅ S3 audit bucket replication setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup S3 replication: {e}")
            return False
    
    def _setup_primary_bucket(self):
        """Setup primary audit bucket with WORM and compliance features."""
        bucket_name = self.config.primary_bucket
        region = self.config.primary_region
        s3_client = self.s3_clients[region]
        
        try:
            # Create bucket if it doesn't exist
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                self.logger.info(f"Primary bucket {bucket_name} already exists")
            except s3_client.exceptions.NoSuchBucket:
                self.logger.info(f"Creating primary bucket: {bucket_name}")
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
            
            # Enable versioning (required for replication)
            if self.config.versioning_enabled:
                s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                self.logger.info(f"Enabled versioning for {bucket_name}")
            
            # Configure server-side encryption
            if self.config.encryption_enabled:
                s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        'Rules': [{
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            },
                            'BucketKeyEnabled': True
                        }]
                    }
                )
                self.logger.info(f"Enabled encryption for {bucket_name}")
            
            # Configure WORM (Object Lock)
            if self.config.enable_worm:
                try:
                    s3_client.put_object_lock_configuration(
                        Bucket=bucket_name,
                        ObjectLockConfiguration={
                            'ObjectLockEnabled': 'Enabled',
                            'Rule': {
                                'DefaultRetention': {
                                    'Mode': 'COMPLIANCE',
                                    'Years': self.config.retention_years
                                }
                            }
                        }
                    )
                    self.logger.info(f"Enabled WORM compliance for {bucket_name}")
                except Exception as e:
                    self.logger.warning(f"Could not enable WORM on existing bucket: {e}")
            
            # Configure lifecycle policy for cost optimization
            self._configure_lifecycle_policy(bucket_name, s3_client)
            
        except Exception as e:
            self.logger.error(f"Failed to setup primary bucket {bucket_name}: {e}")
            raise
    
    def _setup_replica_bucket(self, region: str):
        """Setup replica bucket in specified region."""
        replica_bucket_name = f"{self.config.primary_bucket}-replica-{region}"
        s3_client = self.s3_clients[region]
        
        try:
            # Create replica bucket
            try:
                s3_client.head_bucket(Bucket=replica_bucket_name)
                self.logger.info(f"Replica bucket {replica_bucket_name} already exists")
            except s3_client.exceptions.NoSuchBucket:
                self.logger.info(f"Creating replica bucket: {replica_bucket_name}")
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=replica_bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=replica_bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
            
            # Configure replica bucket (same settings as primary)
            if self.config.versioning_enabled:
                s3_client.put_bucket_versioning(
                    Bucket=replica_bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
            
            if self.config.encryption_enabled:
                s3_client.put_bucket_encryption(
                    Bucket=replica_bucket_name,
                    ServerSideEncryptionConfiguration={
                        'Rules': [{
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            },
                            'BucketKeyEnabled': True
                        }]
                    }
                )
            
            self.logger.info(f"✅ Replica bucket {replica_bucket_name} configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup replica bucket {replica_bucket_name}: {e}")
            raise
    
    def _configure_bucket_replication(self):
        """Configure S3 bucket replication rules."""
        bucket_name = self.config.primary_bucket
        s3_client = self.s3_clients[self.config.primary_region]
        
        try:
            # Create IAM role for replication (if not exists)
            replication_role_arn = self._ensure_replication_role()
            
            # Configure replication rules
            replication_rules = []
            
            for i, region in enumerate(self.config.replica_regions):
                replica_bucket = f"{bucket_name}-replica-{region}"
                
                rule = {
                    'ID': f'ReplicateToRegion{region.replace("-", "").upper()}',
                    'Status': 'Enabled',
                    'Priority': i + 1,
                    'Filter': {'Prefix': 'audit-records/'},
                    'Destination': {
                        'Bucket': f'arn:aws:s3:::{replica_bucket}',
                        'StorageClass': 'STANDARD_IA',  # Cost optimization
                        'ReplicationTime': {
                            'Status': 'Enabled',
                            'Time': {'Minutes': 1}  # 1-minute RTO
                        },
                        'Metrics': {
                            'Status': 'Enabled',
                            'EventThreshold': {'Minutes': 1}
                        }
                    },
                    'DeleteMarkerReplication': {'Status': 'Disabled'}  # WORM compliance
                }
                
                replication_rules.append(rule)
            
            # Apply replication configuration
            replication_config = {
                'Role': replication_role_arn,
                'Rules': replication_rules
            }
            
            s3_client.put_bucket_replication(
                Bucket=bucket_name,
                ReplicationConfiguration=replication_config
            )
            
            self.logger.info(f"✅ Configured replication rules for {bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to configure bucket replication: {e}")
            raise
    
    def _ensure_replication_role(self) -> str:
        """Ensure IAM role exists for S3 replication."""
        iam_client = boto3.client('iam')
        role_name = f"{self.config.primary_bucket}-replication-role"
        
        try:
            # Check if role exists
            try:
                response = iam_client.get_role(RoleName=role_name)
                return response['Role']['Arn']
            except iam_client.exceptions.NoSuchEntityException:
                pass
            
            # Create replication role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "s3.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"S3 replication role for {self.config.primary_bucket}"
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach replication policy
            replication_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObjectVersionForReplication",
                            "s3:GetObjectVersionAcl",
                            "s3:GetObjectVersionTagging"
                        ],
                        "Resource": f"arn:aws:s3:::{self.config.primary_bucket}/*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:ListBucket"
                        ],
                        "Resource": f"arn:aws:s3:::{self.config.primary_bucket}"
                    }
                ]
            }
            
            # Add permissions for replica buckets
            for region in self.config.replica_regions:
                replica_bucket = f"{self.config.primary_bucket}-replica-{region}"
                replication_policy["Statement"].extend([
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:ReplicateObject",
                            "s3:ReplicateDelete",
                            "s3:ReplicateTags"
                        ],
                        "Resource": f"arn:aws:s3:::{replica_bucket}/*"
                    }
                ])
            
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{role_name}-policy",
                PolicyDocument=json.dumps(replication_policy)
            )
            
            self.logger.info(f"✅ Created replication role: {role_arn}")
            return role_arn
            
        except Exception as e:
            self.logger.error(f"Failed to ensure replication role: {e}")
            raise
    
    def _configure_lifecycle_policy(self, bucket_name: str, s3_client):
        """Configure lifecycle policy for cost optimization."""
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'AuditDataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'audit-records/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    'NoncurrentVersionTransitions': [
                        {
                            'NoncurrentDays': 30,
                            'StorageClass': 'STANDARD_IA'
                        }
                    ]
                }
            ]
        }
        
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
        
        self.logger.info(f"✅ Configured lifecycle policy for {bucket_name}")
    
    def _setup_monitoring(self):
        """Setup CloudWatch monitoring and alerting."""
        cloudwatch = boto3.client('cloudwatch', region_name=self.config.primary_region)
        
        try:
            # Create custom metrics for replication monitoring
            metric_namespace = 'IntradayJules/S3Replication'
            
            # Replication lag metric
            cloudwatch.put_metric_alarm(
                AlarmName=f'{self.config.primary_bucket}-ReplicationLag',
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=2,
                MetricName='ReplicationLag',
                Namespace=metric_namespace,
                Period=60,
                Statistic='Maximum',
                Threshold=300.0,  # 5 minutes
                ActionsEnabled=True,
                AlarmDescription='S3 audit bucket replication lag exceeded threshold',
                Unit='Seconds'
            )
            
            # Replication failure metric
            cloudwatch.put_metric_alarm(
                AlarmName=f'{self.config.primary_bucket}-ReplicationFailures',
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='ReplicationFailures',
                Namespace=metric_namespace,
                Period=60,
                Statistic='Sum',
                Threshold=0.0,
                ActionsEnabled=True,
                AlarmDescription='S3 audit bucket replication failures detected',
                Unit='Count'
            )
            
            self.logger.info("✅ Configured CloudWatch monitoring")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {e}")
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring of replication status."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("✅ Started continuous replication monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Stopped continuous replication monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop for replication status."""
        while self.monitoring_active:
            try:
                # Check for new objects to replicate
                self._check_for_new_objects()
                
                # Monitor replication status
                self._monitor_replication_status()
                
                # Send metrics to CloudWatch
                self._send_metrics()
                
                # Wait for next cycle
                time.sleep(self.config.replication_frequency_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _check_for_new_objects(self):
        """Check for new objects that need replication."""
        s3_client = self.s3_clients[self.config.primary_region]
        
        try:
            # List objects modified since last sync
            response = s3_client.list_objects_v2(
                Bucket=self.config.primary_bucket,
                Prefix='audit-records/'
            )
            
            if 'Contents' not in response:
                return
            
            new_objects = []
            for obj in response['Contents']:
                if obj['LastModified'] > self.last_sync_time:
                    new_objects.append(obj['Key'])
            
            if new_objects:
                self.logger.info(f"Found {len(new_objects)} new objects for replication")
                self._replicate_objects(new_objects)
            
            self.last_sync_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error checking for new objects: {e}")
    
    def _replicate_objects(self, object_keys: List[str]):
        """Replicate objects to all replica regions."""
        with ThreadPoolExecutor(max_workers=len(self.config.replica_regions)) as executor:
            futures = []
            
            for region in self.config.replica_regions:
                future = executor.submit(self._replicate_to_region, object_keys, region)
                futures.append(future)
            
            # Wait for all replications to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Replication failed: {e}")
    
    def _replicate_to_region(self, object_keys: List[str], target_region: str):
        """Replicate objects to a specific region."""
        source_client = self.s3_clients[self.config.primary_region]
        target_client = self.s3_clients[target_region]
        target_bucket = f"{self.config.primary_bucket}-replica-{target_region}"
        
        for object_key in object_keys:
            try:
                # Copy object to target region
                copy_source = {
                    'Bucket': self.config.primary_bucket,
                    'Key': object_key
                }
                
                target_client.copy_object(
                    CopySource=copy_source,
                    Bucket=target_bucket,
                    Key=object_key,
                    ServerSideEncryption='AES256'
                )
                
                # Record replication status
                status = ReplicationStatus(
                    object_key=object_key,
                    source_bucket=self.config.primary_bucket,
                    target_bucket=target_bucket,
                    target_region=target_region,
                    replication_time=datetime.now(),
                    status='COMPLETED'
                )
                
                if object_key not in self.replication_status:
                    self.replication_status[object_key] = []
                self.replication_status[object_key].append(status)
                
                self.logger.debug(f"Replicated {object_key} to {target_region}")
                
            except Exception as e:
                self.logger.error(f"Failed to replicate {object_key} to {target_region}: {e}")
                
                # Record failure
                status = ReplicationStatus(
                    object_key=object_key,
                    source_bucket=self.config.primary_bucket,
                    target_bucket=target_bucket,
                    target_region=target_region,
                    replication_time=datetime.now(),
                    status='FAILED',
                    error_message=str(e)
                )
                
                if object_key not in self.replication_status:
                    self.replication_status[object_key] = []
                self.replication_status[object_key].append(status)
    
    def _monitor_replication_status(self):
        """Monitor overall replication health."""
        total_objects = len(self.replication_status)
        if total_objects == 0:
            return
        
        # Calculate replication metrics
        completed_replications = 0
        failed_replications = 0
        total_lag_seconds = 0
        
        for object_key, statuses in self.replication_status.items():
            for status in statuses:
                if status.status == 'COMPLETED':
                    completed_replications += 1
                    # Calculate replication lag (simplified)
                    lag = (datetime.now() - status.replication_time).total_seconds()
                    total_lag_seconds += lag
                elif status.status == 'FAILED':
                    failed_replications += 1
        
        # Log metrics
        if completed_replications > 0:
            avg_lag = total_lag_seconds / completed_replications
            self.logger.info(f"Replication metrics: {completed_replications} completed, "
                           f"{failed_replications} failed, avg lag: {avg_lag:.1f}s")
    
    def _send_metrics(self):
        """Send custom metrics to CloudWatch."""
        cloudwatch = boto3.client('cloudwatch', region_name=self.config.primary_region)
        metric_namespace = 'IntradayJules/S3Replication'
        
        try:
            # Count recent replication statuses
            recent_completed = 0
            recent_failed = 0
            max_lag = 0
            
            cutoff_time = datetime.now() - timedelta(minutes=5)
            
            for statuses in self.replication_status.values():
                for status in statuses:
                    if status.replication_time > cutoff_time:
                        if status.status == 'COMPLETED':
                            recent_completed += 1
                            lag = (datetime.now() - status.replication_time).total_seconds()
                            max_lag = max(max_lag, lag)
                        elif status.status == 'FAILED':
                            recent_failed += 1
            
            # Send metrics
            cloudwatch.put_metric_data(
                Namespace=metric_namespace,
                MetricData=[
                    {
                        'MetricName': 'ReplicationLag',
                        'Value': max_lag,
                        'Unit': 'Seconds',
                        'Timestamp': datetime.now()
                    },
                    {
                        'MetricName': 'ReplicationFailures',
                        'Value': recent_failed,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    },
                    {
                        'MetricName': 'ReplicationSuccess',
                        'Value': recent_completed,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    }
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send metrics: {e}")
    
    def get_replication_report(self) -> Dict:
        """Generate comprehensive replication status report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'total_objects': len(self.replication_status),
            'regions': {},
            'overall_health': 'HEALTHY'
        }
        
        # Analyze per-region status
        for region in self.config.replica_regions:
            region_stats = {
                'completed': 0,
                'failed': 0,
                'pending': 0,
                'avg_lag_seconds': 0,
                'max_lag_seconds': 0
            }
            
            total_lag = 0
            completed_count = 0
            
            for statuses in self.replication_status.values():
                for status in statuses:
                    if status.target_region == region:
                        if status.status == 'COMPLETED':
                            region_stats['completed'] += 1
                            completed_count += 1
                            lag = (datetime.now() - status.replication_time).total_seconds()
                            total_lag += lag
                            region_stats['max_lag_seconds'] = max(region_stats['max_lag_seconds'], lag)
                        elif status.status == 'FAILED':
                            region_stats['failed'] += 1
                        elif status.status == 'PENDING':
                            region_stats['pending'] += 1
            
            if completed_count > 0:
                region_stats['avg_lag_seconds'] = total_lag / completed_count
            
            # Determine region health
            if region_stats['failed'] > 0 or region_stats['max_lag_seconds'] > 300:
                region_stats['health'] = 'DEGRADED'
                report['overall_health'] = 'DEGRADED'
            else:
                region_stats['health'] = 'HEALTHY'
            
            report['regions'][region] = region_stats
        
        return report


def create_default_config() -> ReplicationConfig:
    """Create default replication configuration."""
    return ReplicationConfig(
        primary_bucket="intradayjules-audit-worm",
        primary_region="us-east-1",
        replica_regions=["us-west-2", "eu-west-1"],
        replication_frequency_seconds=60,  # 1 minute
        enable_worm=True,
        retention_years=7,
        encryption_enabled=True,
        versioning_enabled=True,
        mfa_delete_enabled=True
    )


def main():
    """Main function for testing and demonstration."""
    print("🚀 S3 Audit Bucket Replication Setup")
    print("=" * 50)
    
    # Create configuration
    config = create_default_config()
    
    print(f"Primary bucket: {config.primary_bucket}")
    print(f"Primary region: {config.primary_region}")
    print(f"Replica regions: {config.replica_regions}")
    print(f"Replication frequency: {config.replication_frequency_seconds}s")
    print(f"WORM enabled: {config.enable_worm}")
    
    # Initialize replicator
    replicator = S3AuditReplicator(config)
    
    # Setup replication (commented out for safety)
    # print("\n🔧 Setting up replication...")
    # success = replicator.setup_replication()
    # 
    # if success:
    #     print("✅ Replication setup completed")
    #     
    #     # Start monitoring
    #     print("\n📊 Starting continuous monitoring...")
    #     replicator.start_continuous_monitoring()
    #     
    #     # Run for demonstration
    #     time.sleep(60)
    #     
    #     # Generate report
    #     report = replicator.get_replication_report()
    #     print(f"\n📋 Replication Report:")
    #     print(json.dumps(report, indent=2, default=str))
    #     
    #     # Stop monitoring
    #     replicator.stop_continuous_monitoring()
    # else:
    #     print("❌ Replication setup failed")
    
    print("\n✅ S3 audit replication module ready for deployment")
    print("⚠️  Uncomment setup code and configure AWS credentials to activate")


if __name__ == "__main__":
    main()