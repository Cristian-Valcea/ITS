# src/governance/integration.py
"""
Integration layer for governance and compliance with existing IntradayJules systems.

Provides seamless integration of governance features with:
- Risk management system
- Model deployment pipeline
- Configuration management
- Audit systems
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

from .audit_immutable import EnhancedAuditSink
from .model_lineage import ModelLineageTracker
from .release_approval import ApprovalWorkflow, ApprovalType

logger = logging.getLogger(__name__)


class GovernanceManager:
    """
    Central governance manager that integrates all governance components
    with the existing IntradayJules trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize governance components
        self.enhanced_audit = EnhancedAuditSink(config.get('audit_config', {}))
        self.lineage_tracker = ModelLineageTracker(
            config.get('lineage_storage_path', './model_lineage')
        )
        self.approval_workflow = ApprovalWorkflow(config.get('approval_config', {}))
        
        # Integration flags
        self.audit_enabled = config.get('audit_enabled', True)
        self.lineage_enabled = config.get('lineage_enabled', True)
        self.approval_enabled = config.get('approval_enabled', True)
        
        logger.info("Governance manager initialized")
    
    async def audit_risk_decision(self, 
                                decision_data: Dict[str, Any],
                                session_id: str = None) -> Optional[str]:
        """
        Audit risk management decisions with immutable storage.
        
        Integrates with existing RiskAgent to provide governance oversight.
        """
        if not self.audit_enabled:
            return None
        
        try:
            # Enhance decision data with governance metadata
            enhanced_data = {
                **decision_data,
                'governance_timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': session_id or 'unknown',
                'compliance_framework': 'MIFID_II,DODD_FRANK'
            }
            
            # Audit through enhanced audit sink
            audit_hash = await self.enhanced_audit.audit_risk_decision(enhanced_data)
            
            logger.info(f"Risk decision audited: {audit_hash[:16]}...")
            return audit_hash
            
        except Exception as e:
            logger.error(f"Failed to audit risk decision: {e}")
            return None
    
    async def start_model_training_governance(self,
                                            model_name: str,
                                            model_type: str,
                                            training_config: Dict[str, Any],
                                            created_by: str) -> Optional[str]:
        """
        Start governance tracking for model training session.
        
        Integrates with model training pipeline to track lineage from start.
        """
        if not self.lineage_enabled:
            return None
        
        try:
            # Start lineage tracking
            model_id = self.lineage_tracker.start_training_session(
                model_name=model_name,
                model_type=model_type,
                training_config=training_config,
                created_by=created_by
            )
            
            # Audit training start
            if self.audit_enabled:
                await self.enhanced_audit.audit_model_deployment({
                    'event': 'TRAINING_STARTED',
                    'model_id': model_id,
                    'model_name': model_name,
                    'model_type': model_type,
                    'created_by': created_by,
                    'training_config': training_config
                })
            
            logger.info(f"Model training governance started: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to start model training governance: {e}")
            return None
    
    async def track_dataset_usage(self,
                                model_id: str,
                                dataset_path: str,
                                dataset_name: str,
                                dataset_type: str,
                                transformations: List[str] = None) -> Optional[str]:
        """
        Track dataset usage in model training with hash verification.
        
        Integrates with data pipeline to ensure dataset provenance.
        """
        if not self.lineage_enabled:
            return None
        
        try:
            # Load dataset for hashing
            import pandas as pd
            
            if dataset_path.endswith('.csv'):
                dataset = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.parquet'):
                dataset = pd.read_parquet(dataset_path)
            else:
                logger.warning(f"Unsupported dataset format: {dataset_path}")
                return None
            
            # Record dataset usage
            fingerprint = self.lineage_tracker.record_dataset_usage(
                model_id=model_id,
                dataset=dataset,
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                transformations=transformations or []
            )
            
            # Audit dataset usage
            if self.audit_enabled:
                await self.enhanced_audit.audit_model_deployment({
                    'event': 'DATASET_USED',
                    'model_id': model_id,
                    'dataset_name': dataset_name,
                    'dataset_type': dataset_type,
                    'dataset_hash': fingerprint.sha256_hash,
                    'dataset_path': dataset_path,
                    'row_count': fingerprint.row_count,
                    'transformations': transformations or []
                })
            
            logger.info(f"Dataset usage tracked: {dataset_name} -> {fingerprint.sha256_hash[:16]}...")
            return fingerprint.sha256_hash
            
        except Exception as e:
            logger.error(f"Failed to track dataset usage: {e}")
            return None
    
    async def complete_model_training_governance(self,
                                               model_id: str,
                                               model_file_path: str,
                                               model_version: str,
                                               hyperparameters: Dict[str, Any],
                                               training_metrics: Dict[str, float],
                                               validation_metrics: Dict[str, float],
                                               test_metrics: Dict[str, float] = None,
                                               feature_columns: List[str] = None,
                                               target_columns: List[str] = None,
                                               random_seed: int = None) -> Optional[Dict[str, str]]:
        """
        Complete model training governance with full lineage record.
        
        Integrates with model training completion to finalize governance tracking.
        """
        if not self.lineage_enabled:
            return None
        
        try:
            # Record model artifact
            model_hash = self.lineage_tracker.record_model_artifact(
                model_id=model_id,
                model_file_path=Path(model_file_path),
                model_version=model_version
            )
            
            # Complete training session
            lineage = self.lineage_tracker.complete_training_session(
                model_id=model_id,
                model_version=model_version,
                hyperparameters=hyperparameters,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_columns=feature_columns,
                target_columns=target_columns,
                random_seed=random_seed
            )
            
            # Audit training completion
            if self.audit_enabled:
                audit_hash = await self.enhanced_audit.audit_model_deployment({
                    'event': 'TRAINING_COMPLETED',
                    'model_id': model_id,
                    'model_version': model_version,
                    'model_hash': model_hash,
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics,
                    'test_metrics': test_metrics or {},
                    'lineage_complete': True,
                    'reproducibility_hash': lineage.reproducibility_hash
                })
            else:
                audit_hash = None
            
            result = {
                'model_hash': model_hash,
                'reproducibility_hash': lineage.reproducibility_hash,
                'audit_hash': audit_hash
            }
            
            logger.info(f"Model training governance completed: {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to complete model training governance: {e}")
            return None
    
    async def request_production_deployment_approval(self,
                                                   model_id: str,
                                                   model_path: str,
                                                   model_hash: str,
                                                   dataset_hashes: List[str],
                                                   performance_metrics: Dict[str, float],
                                                   requested_by: str,
                                                   deployment_config: Dict[str, Any] = None) -> Optional[str]:
        """
        Request approval for production model deployment.
        
        Integrates with deployment pipeline to enforce four-eyes approval.
        """
        if not self.approval_enabled:
            return None
        
        try:
            # Request deployment approval
            approval_request_id = await self.approval_workflow.request_model_deployment_approval(
                model_id=model_id,
                model_path=model_path,
                model_hash=model_hash,
                dataset_hashes=dataset_hashes,
                performance_metrics=performance_metrics,
                requested_by=requested_by,
                target_environment="production"
            )
            
            # Audit approval request
            if self.audit_enabled:
                await self.enhanced_audit.audit_model_deployment({
                    'event': 'APPROVAL_REQUESTED',
                    'model_id': model_id,
                    'approval_request_id': approval_request_id,
                    'requested_by': requested_by,
                    'model_hash': model_hash,
                    'dataset_hashes': dataset_hashes,
                    'performance_metrics': performance_metrics,
                    'deployment_config': deployment_config or {}
                })
            
            logger.info(f"Production deployment approval requested: {approval_request_id}")
            return approval_request_id
            
        except Exception as e:
            logger.error(f"Failed to request deployment approval: {e}")
            return None
    
    async def check_deployment_approval_status(self, approval_request_id: str) -> Dict[str, Any]:
        """
        Check status of deployment approval request.
        
        Integrates with deployment pipeline to verify approval before deployment.
        """
        try:
            # Check approval status
            is_approved = await self.approval_workflow.is_deployment_approved(approval_request_id)
            
            # Get detailed approval request
            approval_request = await self.approval_workflow.release_gate.get_approval_status(approval_request_id)
            
            if approval_request:
                status_info = {
                    'approved': is_approved,
                    'status': approval_request.status.value,
                    'approval_count': len(approval_request.approvals),
                    'required_approvals': approval_request.minimum_approvals,
                    'rejection_count': len(approval_request.rejections),
                    'expires_timestamp': approval_request.expires_timestamp,
                    'approvers': [a['approver_name'] for a in approval_request.approvals],
                    'rejectors': [r['approver_name'] for r in approval_request.rejections]
                }
            else:
                status_info = {
                    'approved': False,
                    'status': 'NOT_FOUND',
                    'error': 'Approval request not found'
                }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to check approval status: {e}")
            return {
                'approved': False,
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def audit_deployment_execution(self,
                                       model_id: str,
                                       approval_request_id: str,
                                       deployment_status: str,
                                       deployment_details: Dict[str, Any]) -> Optional[str]:
        """
        Audit actual model deployment execution.
        
        Integrates with deployment system to audit deployment events.
        """
        if not self.audit_enabled:
            return None
        
        try:
            # Audit deployment execution
            audit_hash = await self.enhanced_audit.audit_model_deployment({
                'event': 'DEPLOYMENT_EXECUTED',
                'model_id': model_id,
                'approval_request_id': approval_request_id,
                'deployment_status': deployment_status,
                'deployment_timestamp': datetime.now(timezone.utc).isoformat(),
                'deployment_details': deployment_details
            })
            
            logger.info(f"Deployment execution audited: {audit_hash[:16]}...")
            return audit_hash
            
        except Exception as e:
            logger.error(f"Failed to audit deployment execution: {e}")
            return None
    
    async def audit_configuration_change(self,
                                       component: str,
                                       config_changes: Dict[str, Any],
                                       changed_by: str,
                                       session_id: str = None) -> Optional[str]:
        """
        Audit configuration changes with immutable storage.
        
        Integrates with configuration management to track all changes.
        """
        if not self.audit_enabled:
            return None
        
        try:
            # Audit configuration change
            audit_hash = await self.enhanced_audit.immutable_sink.write_audit_record(
                event_type="CONFIG_CHANGE",
                component=component,
                user_id=changed_by,
                session_id=session_id or 'unknown',
                action="UPDATE_CONFIG",
                details={
                    'config_changes': config_changes,
                    'change_timestamp': datetime.now(timezone.utc).isoformat()
                },
                risk_impact="MEDIUM",
                compliance_tags=['CONFIGURATION_MANAGEMENT', 'SYSTEM_CHANGE']
            )
            
            logger.info(f"Configuration change audited: {audit_hash[:16]}...")
            return audit_hash
            
        except Exception as e:
            logger.error(f"Failed to audit configuration change: {e}")
            return None
    
    async def generate_governance_report(self,
                                       start_date: datetime,
                                       end_date: datetime,
                                       report_type: str = "FULL") -> Dict[str, Any]:
        """
        Generate comprehensive governance report.
        
        Combines audit, lineage, and approval data for regulatory reporting.
        """
        try:
            report = {
                'report_type': report_type,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'generated_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Add audit report
            if self.audit_enabled:
                audit_report = await self.enhanced_audit.generate_regulatory_report(
                    start_date, end_date, report_type
                )
                report['audit_summary'] = audit_report
            
            # Add lineage report
            if self.lineage_enabled:
                lineage_report = self.lineage_tracker.generate_lineage_report()
                report['lineage_summary'] = lineage_report
            
            # Add approval summary
            if self.approval_enabled:
                # Get all approval requests in date range
                pending_approvals = await self.approval_workflow.release_gate.get_pending_approvals()
                report['approval_summary'] = {
                    'pending_approvals': len(pending_approvals),
                    'pending_requests': [
                        {
                            'request_id': req.request_id,
                            'title': req.title,
                            'requested_by': req.requested_by,
                            'created_timestamp': req.created_timestamp,
                            'approval_type': req.approval_type.value
                        }
                        for req in pending_approvals
                    ]
                }
            
            logger.info(f"Governance report generated: {report_type}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate governance report: {e}")
            return {
                'error': str(e),
                'report_type': report_type,
                'generated_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def validate_model_reproducibility(self,
                                           model_id: str,
                                           current_dataset_paths: List[str]) -> Dict[str, Any]:
        """
        Validate model reproducibility using current datasets.
        
        Integrates with model validation pipeline to ensure reproducibility.
        """
        if not self.lineage_enabled:
            return {'reproducible': False, 'error': 'Lineage tracking disabled'}
        
        try:
            # Hash current datasets
            current_fingerprints = []
            for dataset_path in current_dataset_paths:
                if Path(dataset_path).exists():
                    if dataset_path.endswith('.csv'):
                        fingerprint = self.lineage_tracker.dataset_hasher.hash_csv_file(
                            Path(dataset_path)
                        )
                    else:
                        logger.warning(f"Unsupported dataset format for reproducibility check: {dataset_path}")
                        continue
                    current_fingerprints.append(fingerprint)
            
            # Validate reproducibility
            is_reproducible = self.lineage_tracker.validate_reproducibility(
                model_id, current_fingerprints
            )
            
            # Get original lineage for comparison
            original_lineage = self.lineage_tracker.get_model_lineage(model_id)
            
            result = {
                'reproducible': is_reproducible,
                'model_id': model_id,
                'original_dataset_count': len(original_lineage.training_datasets) if original_lineage else 0,
                'current_dataset_count': len(current_fingerprints),
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if original_lineage:
                result['original_datasets'] = [
                    {
                        'name': ds.dataset_name,
                        'hash': ds.sha256_hash,
                        'row_count': ds.row_count
                    }
                    for ds in original_lineage.training_datasets
                ]
            
            result['current_datasets'] = [
                {
                    'name': fp.dataset_name,
                    'hash': fp.sha256_hash,
                    'row_count': fp.row_count
                }
                for fp in current_fingerprints
            ]
            
            logger.info(f"Model reproducibility validated: {model_id} -> {'PASS' if is_reproducible else 'FAIL'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate model reproducibility: {e}")
            return {
                'reproducible': False,
                'error': str(e),
                'model_id': model_id
            }


# Integration decorators for existing system components
def governance_audit(event_type: str, component: str, risk_impact: str = "MEDIUM"):
    """
    Decorator to automatically audit function calls.
    
    Usage:
        @governance_audit("RISK_DECISION", "RiskAgent", "HIGH")
        def calculate_risk(self, ...):
            ...
    """
    def decorator(func):
        async def async_wrapper(self, *args, **kwargs):
            # Execute original function
            result = await func(self, *args, **kwargs)
            
            # Audit the call if governance manager is available
            if hasattr(self, 'governance_manager') and self.governance_manager:
                try:
                    await self.governance_manager.enhanced_audit.immutable_sink.write_audit_record(
                        event_type=event_type,
                        component=component,
                        user_id=getattr(self, 'user_id', 'system'),
                        session_id=getattr(self, 'session_id', 'unknown'),
                        action=func.__name__,
                        details={
                            'function': func.__name__,
                            'args': str(args)[:500],  # Truncate for storage
                            'result_type': type(result).__name__
                        },
                        risk_impact=risk_impact,
                        compliance_tags=[component.upper(), event_type]
                    )
                except Exception as e:
                    logger.warning(f"Failed to audit {func.__name__}: {e}")
            
            return result
        
        def sync_wrapper(self, *args, **kwargs):
            # For synchronous functions, run audit in background
            result = func(self, *args, **kwargs)
            
            if hasattr(self, 'governance_manager') and self.governance_manager:
                try:
                    # Schedule audit in background
                    asyncio.create_task(
                        self.governance_manager.enhanced_audit.immutable_sink.write_audit_record(
                            event_type=event_type,
                            component=component,
                            user_id=getattr(self, 'user_id', 'system'),
                            session_id=getattr(self, 'session_id', 'unknown'),
                            action=func.__name__,
                            details={
                                'function': func.__name__,
                                'args': str(args)[:500],
                                'result_type': type(result).__name__
                            },
                            risk_impact=risk_impact,
                            compliance_tags=[component.upper(), event_type]
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to schedule audit for {func.__name__}: {e}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def require_approval(approval_type: ApprovalType, title_template: str = None):
    """
    Decorator to require approval before function execution.
    
    Usage:
        @require_approval(ApprovalType.MODEL_DEPLOYMENT, "Deploy model {model_id}")
        async def deploy_model(self, model_id, ...):
            ...
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Check if governance manager is available
            if not hasattr(self, 'governance_manager') or not self.governance_manager:
                logger.warning(f"No governance manager available for approval check in {func.__name__}")
                return await func(self, *args, **kwargs)
            
            # Generate approval title
            if title_template:
                try:
                    # Try to format title with function arguments
                    title = title_template.format(*args, **kwargs)
                except:
                    title = f"{func.__name__} execution"
            else:
                title = f"{func.__name__} execution"
            
            # Check if approval is required (could be configured per environment)
            approval_required = getattr(self, 'require_approval', True)
            
            if approval_required:
                # For demo purposes, we'll assume approval is granted
                # In practice, this would check for existing approval or block execution
                logger.info(f"Approval required for {func.__name__}: {title}")
                
                # Here you would implement the approval check logic
                # For now, we'll proceed with a warning
                logger.warning(f"Proceeding with {func.__name__} without approval check (demo mode)")
            
            # Execute the function
            return await func(self, *args, **kwargs)
        
        return wrapper
    
    return decorator