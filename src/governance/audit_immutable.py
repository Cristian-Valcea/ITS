# src/governance/audit_immutable.py
"""
Immutable Audit Trail Implementation.

Provides WORM (Write-Once-Read-Many) storage for audit logs to ensure
regulatory compliance and prevent tampering with audit records.

Features:
- Stream audit logs to immutable storage (S3 with object lock, Kafka)
- Cryptographic integrity verification
- Tamper-evident audit chains
- Regulatory compliance reporting
"""

import json
import hashlib
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from kafka import KafkaProducer
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Immutable audit record structure."""
    timestamp: str
    event_type: str
    component: str
    user_id: str
    session_id: str
    action: str
    details: Dict[str, Any]
    risk_impact: str
    compliance_tags: List[str]
    previous_hash: Optional[str] = None
    record_hash: Optional[str] = None
    sequence_number: Optional[int] = None


class ImmutableAuditSink:
    """
    Immutable audit sink that streams audit records to WORM storage.
    
    Ensures audit records cannot be modified after creation through:
    - Cryptographic hashing chains
    - WORM storage backends
    - Tamper detection mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sequence_counter = 0
        self.last_hash = "0" * 64  # Genesis hash
        self.storage_backends = []
        
        # Initialize storage backends
        if config.get('s3_worm_enabled', False):
            self.storage_backends.append(
                S3WORMStorage(config.get('s3_config', {}))
            )
        
        if config.get('kafka_enabled', False):
            self.storage_backends.append(
                KafkaAuditStorage(config.get('kafka_config', {}))
            )
        
        # Fallback local WORM storage
        self.storage_backends.append(
            LocalWORMStorage(config.get('local_worm_path', './audit_worm'))
        )
        
        logger.info(f"Initialized immutable audit sink with {len(self.storage_backends)} backends")
    
    async def write_audit_record(self, 
                                event_type: str,
                                component: str, 
                                user_id: str,
                                session_id: str,
                                action: str,
                                details: Dict[str, Any],
                                risk_impact: str = "LOW",
                                compliance_tags: List[str] = None) -> str:
        """
        Write immutable audit record to WORM storage.
        
        Returns:
            str: Record hash for verification
        """
        if compliance_tags is None:
            compliance_tags = []
        
        # Create audit record
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            component=component,
            user_id=user_id,
            session_id=session_id,
            action=action,
            details=details,
            risk_impact=risk_impact,
            compliance_tags=compliance_tags,
            previous_hash=self.last_hash,
            sequence_number=self.sequence_counter
        )
        
        # Calculate record hash
        record_data = asdict(record)
        record_data.pop('record_hash', None)  # Exclude hash from hash calculation
        record_json = json.dumps(record_data, sort_keys=True)
        record.record_hash = hashlib.sha256(record_json.encode()).hexdigest()
        
        # Update chain
        self.last_hash = record.record_hash
        self.sequence_counter += 1
        
        # Write to all storage backends
        write_tasks = []
        for backend in self.storage_backends:
            write_tasks.append(backend.write_record(record))
        
        try:
            await asyncio.gather(*write_tasks)
            logger.info(f"Audit record written: {record.record_hash[:16]}...")
            return record.record_hash
        except Exception as e:
            logger.error(f"Failed to write audit record: {e}")
            raise
    
    async def verify_audit_chain(self, start_sequence: int = 0, end_sequence: int = None) -> bool:
        """
        Verify integrity of audit chain.
        
        Returns:
            bool: True if chain is valid, False if tampered
        """
        try:
            # Use primary storage backend for verification
            primary_backend = self.storage_backends[0]
            records = await primary_backend.read_records(start_sequence, end_sequence)
            
            previous_hash = "0" * 64 if start_sequence == 0 else None
            
            for record in records:
                # Verify hash chain
                if previous_hash and record.previous_hash != previous_hash:
                    logger.error(f"Hash chain broken at sequence {record.sequence_number}")
                    return False
                
                # Verify record hash
                record_data = asdict(record)
                record_data.pop('record_hash')
                record_json = json.dumps(record_data, sort_keys=True)
                calculated_hash = hashlib.sha256(record_json.encode()).hexdigest()
                
                if calculated_hash != record.record_hash:
                    logger.error(f"Record hash mismatch at sequence {record.sequence_number}")
                    return False
                
                previous_hash = record.record_hash
            
            logger.info(f"Audit chain verified: sequences {start_sequence}-{end_sequence or 'latest'}")
            return True
            
        except Exception as e:
            logger.error(f"Audit chain verification failed: {e}")
            return False
    
    async def generate_compliance_report(self, 
                                       start_date: datetime,
                                       end_date: datetime,
                                       compliance_tags: List[str] = None) -> Dict[str, Any]:
        """Generate compliance report for regulatory audits."""
        try:
            primary_backend = self.storage_backends[0]
            records = await primary_backend.read_records_by_date(start_date, end_date)
            
            # Filter by compliance tags if specified
            if compliance_tags:
                records = [r for r in records if any(tag in r.compliance_tags for tag in compliance_tags)]
            
            # Generate report
            report = {
                'report_generated': datetime.now(timezone.utc).isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_records': len(records),
                'compliance_tags_filter': compliance_tags,
                'records_by_component': {},
                'records_by_risk_impact': {},
                'chain_integrity_verified': await self.verify_audit_chain(),
                'records': [asdict(r) for r in records]
            }
            
            # Aggregate statistics
            for record in records:
                # By component
                component = record.component
                if component not in report['records_by_component']:
                    report['records_by_component'][component] = 0
                report['records_by_component'][component] += 1
                
                # By risk impact
                risk_impact = record.risk_impact
                if risk_impact not in report['records_by_risk_impact']:
                    report['records_by_risk_impact'][risk_impact] = 0
                report['records_by_risk_impact'][risk_impact] += 1
            
            logger.info(f"Compliance report generated: {len(records)} records")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise


class WORMAuditStorage:
    """Base class for WORM (Write-Once-Read-Many) audit storage."""
    
    async def write_record(self, record: AuditRecord) -> bool:
        """Write audit record to WORM storage."""
        raise NotImplementedError
    
    async def read_records(self, start_sequence: int, end_sequence: int = None) -> List[AuditRecord]:
        """Read audit records by sequence number."""
        raise NotImplementedError
    
    async def read_records_by_date(self, start_date: datetime, end_date: datetime) -> List[AuditRecord]:
        """Read audit records by date range."""
        raise NotImplementedError


class S3WORMStorage(WORMAuditStorage):
    """S3 WORM storage with object lock for immutable audit records."""
    
    def __init__(self, config: Dict[str, Any]):
        self.bucket_name = config['bucket_name']
        self.prefix = config.get('prefix', 'audit-records/')
        self.region = config.get('region', 'us-east-1')
        
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # Verify bucket has object lock enabled
        try:
            response = self.s3_client.get_object_lock_configuration(Bucket=self.bucket_name)
            if response['ObjectLockConfiguration']['ObjectLockEnabled'] != 'Enabled':
                raise ValueError(f"S3 bucket {self.bucket_name} does not have object lock enabled")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ObjectLockConfigurationNotFoundError':
                raise ValueError(f"S3 bucket {self.bucket_name} does not have object lock configured")
            raise
        
        logger.info(f"S3 WORM storage initialized: {self.bucket_name}")
    
    async def write_record(self, record: AuditRecord) -> bool:
        """Write audit record to S3 with object lock."""
        try:
            key = f"{self.prefix}{record.sequence_number:010d}_{record.record_hash}.json"
            
            # Write with object lock (WORM)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(asdict(record), indent=2),
                ContentType='application/json',
                ObjectLockMode='COMPLIANCE',
                ObjectLockRetainUntilDate=datetime.now(timezone.utc).replace(year=datetime.now().year + 7),  # 7 year retention
                Metadata={
                    'sequence-number': str(record.sequence_number),
                    'record-hash': record.record_hash,
                    'component': record.component,
                    'risk-impact': record.risk_impact
                }
            )
            
            logger.debug(f"S3 WORM record written: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write S3 WORM record: {e}")
            raise
    
    async def read_records(self, start_sequence: int, end_sequence: int = None) -> List[AuditRecord]:
        """Read audit records from S3 by sequence number."""
        try:
            records = []
            
            # List objects in sequence range
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract sequence number from key
                    try:
                        seq_str = key.split('/')[-1].split('_')[0]
                        seq_num = int(seq_str)
                        
                        if seq_num >= start_sequence and (end_sequence is None or seq_num <= end_sequence):
                            # Read record
                            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                            record_data = json.loads(response['Body'].read())
                            records.append(AuditRecord(**record_data))
                    except (ValueError, IndexError):
                        continue
            
            # Sort by sequence number
            records.sort(key=lambda r: r.sequence_number)
            return records
            
        except Exception as e:
            logger.error(f"Failed to read S3 WORM records: {e}")
            raise
    
    async def read_records_by_date(self, start_date: datetime, end_date: datetime) -> List[AuditRecord]:
        """Read audit records from S3 by date range."""
        # For simplicity, read all records and filter by date
        # In production, consider using S3 Select or partitioning by date
        all_records = await self.read_records(0)
        
        filtered_records = []
        for record in all_records:
            record_date = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
            if start_date <= record_date <= end_date:
                filtered_records.append(record)
        
        return filtered_records


class KafkaAuditStorage(WORMAuditStorage):
    """Kafka storage for real-time audit streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        self.topic = config['topic']
        self.bootstrap_servers = config['bootstrap_servers']
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
        
        logger.info(f"Kafka audit storage initialized: {self.topic}")
    
    async def write_record(self, record: AuditRecord) -> bool:
        """Write audit record to Kafka topic."""
        try:
            future = self.producer.send(
                self.topic,
                key=str(record.sequence_number),
                value=asdict(record)
            )
            
            # Wait for send to complete
            future.get(timeout=10)
            logger.debug(f"Kafka audit record sent: {record.sequence_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write Kafka audit record: {e}")
            raise
    
    async def read_records(self, start_sequence: int, end_sequence: int = None) -> List[AuditRecord]:
        """Read audit records from Kafka (limited implementation)."""
        # Note: Kafka is primarily for streaming, not querying
        # For full audit queries, use S3 or database storage
        logger.warning("Kafka storage is for streaming only, use S3 for audit queries")
        return []
    
    async def read_records_by_date(self, start_date: datetime, end_date: datetime) -> List[AuditRecord]:
        """Read audit records by date (not supported for Kafka)."""
        logger.warning("Date-based queries not supported for Kafka storage")
        return []


class LocalWORMStorage(WORMAuditStorage):
    """Local filesystem WORM storage for development/testing."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file
        self.index_file = self.storage_path / 'audit_index.json'
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump([], f)
        
        logger.info(f"Local WORM storage initialized: {self.storage_path}")
    
    async def write_record(self, record: AuditRecord) -> bool:
        """Write audit record to local WORM storage."""
        try:
            # Write record file
            record_file = self.storage_path / f"{record.sequence_number:010d}_{record.record_hash}.json"
            with open(record_file, 'w') as f:
                json.dump(asdict(record), f, indent=2)
            
            # Make file read-only (simple WORM simulation)
            record_file.chmod(0o444)
            
            # Update index
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            index.append({
                'sequence_number': record.sequence_number,
                'record_hash': record.record_hash,
                'timestamp': record.timestamp,
                'filename': record_file.name
            })
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.debug(f"Local WORM record written: {record_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write local WORM record: {e}")
            raise
    
    async def read_records(self, start_sequence: int, end_sequence: int = None) -> List[AuditRecord]:
        """Read audit records from local storage."""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            records = []
            for entry in index:
                seq_num = entry['sequence_number']
                if seq_num >= start_sequence and (end_sequence is None or seq_num <= end_sequence):
                    record_file = self.storage_path / entry['filename']
                    with open(record_file, 'r') as f:
                        record_data = json.load(f)
                        records.append(AuditRecord(**record_data))
            
            records.sort(key=lambda r: r.sequence_number)
            return records
            
        except Exception as e:
            logger.error(f"Failed to read local WORM records: {e}")
            raise
    
    async def read_records_by_date(self, start_date: datetime, end_date: datetime) -> List[AuditRecord]:
        """Read audit records by date range."""
        all_records = await self.read_records(0)
        
        filtered_records = []
        for record in all_records:
            record_date = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
            if start_date <= record_date <= end_date:
                filtered_records.append(record)
        
        return filtered_records


# Integration with existing audit system
class EnhancedAuditSink:
    """Enhanced audit sink with immutable storage capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.immutable_sink = ImmutableAuditSink(config.get('immutable_config', {}))
        self.legacy_enabled = config.get('legacy_audit_enabled', True)
        
        if self.legacy_enabled:
            # Import existing audit sink
            from ..risk.audit_sink import AuditSink
            self.legacy_sink = AuditSink(config.get('legacy_config', {}))
    
    async def audit_risk_decision(self, decision_data: Dict[str, Any]) -> str:
        """Audit risk decision with immutable storage."""
        
        # Write to immutable storage
        record_hash = await self.immutable_sink.write_audit_record(
            event_type="RISK_DECISION",
            component="RiskAgent",
            user_id=decision_data.get('user_id', 'system'),
            session_id=decision_data.get('session_id', 'unknown'),
            action=decision_data.get('decision', 'UNKNOWN'),
            details=decision_data,
            risk_impact=decision_data.get('risk_level', 'MEDIUM'),
            compliance_tags=['RISK_MANAGEMENT', 'TRADING_DECISION']
        )
        
        # Also write to legacy system if enabled
        if self.legacy_enabled:
            await self.legacy_sink.audit_risk_decision(decision_data)
        
        return record_hash
    
    async def audit_model_deployment(self, deployment_data: Dict[str, Any]) -> str:
        """Audit model deployment with immutable storage."""
        
        record_hash = await self.immutable_sink.write_audit_record(
            event_type="MODEL_DEPLOYMENT",
            component="ModelDeployment",
            user_id=deployment_data.get('deployed_by', 'system'),
            session_id=deployment_data.get('deployment_session', 'unknown'),
            action="DEPLOY_MODEL",
            details=deployment_data,
            risk_impact="HIGH",
            compliance_tags=['MODEL_GOVERNANCE', 'DEPLOYMENT', 'FOUR_EYES']
        )
        
        return record_hash
    
    async def generate_regulatory_report(self, 
                                       start_date: datetime,
                                       end_date: datetime,
                                       report_type: str = "FULL") -> Dict[str, Any]:
        """Generate regulatory compliance report."""
        
        compliance_tags = {
            'RISK': ['RISK_MANAGEMENT', 'TRADING_DECISION'],
            'MODEL': ['MODEL_GOVERNANCE', 'DEPLOYMENT'],
            'FULL': None  # All records
        }.get(report_type, None)
        
        return await self.immutable_sink.generate_compliance_report(
            start_date, end_date, compliance_tags
        )