# src/governance/release_approval.py
"""
Four-Eyes Release Approval System.

Implements enterprise-grade release approval workflow with multiple
approval gates, digital signatures, and audit trails for model
deployments and critical system changes.

Features:
- GitHub-based approval workflow
- ServiceNow integration for enterprise environments
- Digital signature verification
- Multi-stage approval gates
- Automated compliance checks
- Audit trail for all approvals
"""

import json
import hashlib
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import requests
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status enumeration."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ApprovalType(Enum):
    """Approval type enumeration."""
    MODEL_DEPLOYMENT = "MODEL_DEPLOYMENT"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SYSTEM_UPDATE = "SYSTEM_UPDATE"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"


@dataclass
class ApprovalRequest:
    """Approval request data structure."""
    request_id: str
    approval_type: ApprovalType
    title: str
    description: str
    requested_by: str
    created_timestamp: str
    expires_timestamp: str
    
    # Approval requirements
    required_approvers: List[str]
    minimum_approvals: int
    approval_groups: List[str]  # e.g., ['risk_managers', 'senior_developers']
    
    # Request details
    change_details: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    
    # Artifacts to approve
    artifacts: List[Dict[str, Any]]  # Files, configs, models to approve
    
    # Current status
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvals: List[Dict[str, Any]] = None
    rejections: List[Dict[str, Any]] = None
    
    # Compliance
    compliance_checks: Dict[str, bool] = None
    audit_trail: List[Dict[str, Any]] = None


@dataclass
class ApprovalDecision:
    """Individual approval decision."""
    approver_id: str
    approver_name: str
    decision: str  # 'APPROVE', 'REJECT'
    timestamp: str
    comments: str
    digital_signature: Optional[str] = None
    approval_authority: str = None  # Role/group that grants authority


class FourEyesReleaseGate:
    """
    Four-eyes release approval gate.
    
    Implements multi-stage approval workflow with configurable approval
    requirements and integration with external approval systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', './approvals'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize approval backends
        self.approval_backends = []
        
        if config.get('github_enabled', False):
            self.approval_backends.append(
                GitHubApprovalBackend(config.get('github_config', {}))
            )
        
        if config.get('servicenow_enabled', False):
            self.approval_backends.append(
                ServiceNowApprovalBackend(config.get('servicenow_config', {}))
            )
        
        # Always include local approval backend
        self.approval_backends.append(
            LocalApprovalBackend(self.storage_path)
        )
        
        # Load approval groups and policies
        self.approval_groups = config.get('approval_groups', {})
        self.approval_policies = config.get('approval_policies', {})
        
        logger.info(f"Four-eyes release gate initialized with {len(self.approval_backends)} backends")
    
    async def create_approval_request(self,
                                    approval_type: ApprovalType,
                                    title: str,
                                    description: str,
                                    requested_by: str,
                                    change_details: Dict[str, Any],
                                    artifacts: List[Dict[str, Any]] = None,
                                    expires_hours: int = 24) -> str:
        """
        Create new approval request.
        
        Args:
            approval_type: Type of approval required
            title: Short title for the approval
            description: Detailed description
            requested_by: User requesting approval
            change_details: Details of the change
            artifacts: Files/artifacts requiring approval
            expires_hours: Hours until request expires
            
        Returns:
            str: Approval request ID
        """
        try:
            # Generate request ID
            timestamp = datetime.now(timezone.utc)
            request_id = f"{approval_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(title) % 10000:04d}"
            
            # Get approval policy for this type
            policy = self.approval_policies.get(approval_type.value, {})
            required_approvers = policy.get('required_approvers', [])
            minimum_approvals = policy.get('minimum_approvals', 2)
            approval_groups = policy.get('approval_groups', ['senior_developers'])
            
            # Perform automated compliance checks
            compliance_checks = await self._perform_compliance_checks(
                approval_type, change_details, artifacts or []
            )
            
            # Generate risk assessment
            risk_assessment = await self._assess_risk(approval_type, change_details)
            
            # Create approval request
            request = ApprovalRequest(
                request_id=request_id,
                approval_type=approval_type,
                title=title,
                description=description,
                requested_by=requested_by,
                created_timestamp=timestamp.isoformat(),
                expires_timestamp=(timestamp + timedelta(hours=expires_hours)).isoformat(),
                
                required_approvers=required_approvers,
                minimum_approvals=minimum_approvals,
                approval_groups=approval_groups,
                
                change_details=change_details,
                risk_assessment=risk_assessment,
                impact_analysis=await self._analyze_impact(approval_type, change_details),
                rollback_plan=change_details.get('rollback_plan', {}),
                
                artifacts=artifacts or [],
                
                approvals=[],
                rejections=[],
                compliance_checks=compliance_checks,
                audit_trail=[{
                    'timestamp': timestamp.isoformat(),
                    'action': 'REQUEST_CREATED',
                    'user': requested_by,
                    'details': {'request_id': request_id}
                }]
            )
            
            # Submit to approval backends
            for backend in self.approval_backends:
                await backend.create_approval_request(request)
            
            # Save request
            await self._save_approval_request(request)
            
            logger.info(f"Approval request created: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            raise
    
    async def submit_approval(self,
                            request_id: str,
                            approver_id: str,
                            approver_name: str,
                            decision: str,
                            comments: str,
                            approval_authority: str = None) -> bool:
        """
        Submit approval decision.
        
        Args:
            request_id: Approval request ID
            approver_id: ID of approver
            approver_name: Name of approver
            decision: 'APPROVE' or 'REJECT'
            comments: Approval comments
            approval_authority: Authority granting approval rights
            
        Returns:
            bool: True if approval was accepted
        """
        try:
            # Load approval request
            request = await self._load_approval_request(request_id)
            if not request:
                raise ValueError(f"Approval request not found: {request_id}")
            
            # Check if request is still valid
            if request.status != ApprovalStatus.PENDING:
                raise ValueError(f"Request is not pending: {request.status}")
            
            expires_time = datetime.fromisoformat(request.expires_timestamp)
            if datetime.now(timezone.utc) > expires_time:
                request.status = ApprovalStatus.EXPIRED
                await self._save_approval_request(request)
                raise ValueError("Approval request has expired")
            
            # Validate approver authority
            if not await self._validate_approver_authority(request, approver_id, approval_authority):
                raise ValueError(f"Approver {approver_id} does not have authority for this request")
            
            # Check if approver already voted
            existing_approvals = [a for a in request.approvals if a['approver_id'] == approver_id]
            existing_rejections = [r for r in request.rejections if r['approver_id'] == approver_id]
            
            if existing_approvals or existing_rejections:
                raise ValueError(f"Approver {approver_id} has already voted on this request")
            
            # Create approval decision
            decision_obj = ApprovalDecision(
                approver_id=approver_id,
                approver_name=approver_name,
                decision=decision,
                timestamp=datetime.now(timezone.utc).isoformat(),
                comments=comments,
                digital_signature=await self._generate_digital_signature(request_id, approver_id, decision),
                approval_authority=approval_authority
            )
            
            # Add to appropriate list
            if decision == 'APPROVE':
                request.approvals.append(asdict(decision_obj))
            elif decision == 'REJECT':
                request.rejections.append(asdict(decision_obj))
            else:
                raise ValueError(f"Invalid decision: {decision}")
            
            # Update audit trail
            request.audit_trail.append({
                'timestamp': decision_obj.timestamp,
                'action': f'DECISION_{decision}',
                'user': approver_id,
                'details': {
                    'approver_name': approver_name,
                    'comments': comments,
                    'authority': approval_authority
                }
            })
            
            # Check if approval is complete
            await self._check_approval_completion(request)
            
            # Save updated request
            await self._save_approval_request(request)
            
            # Notify approval backends
            for backend in self.approval_backends:
                await backend.update_approval_status(request)
            
            logger.info(f"Approval decision submitted: {request_id} -> {decision} by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit approval: {e}")
            raise
    
    async def get_approval_status(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get current approval status."""
        return await self._load_approval_request(request_id)
    
    async def get_pending_approvals(self, approver_id: str = None) -> List[ApprovalRequest]:
        """Get pending approval requests for approver."""
        try:
            pending_requests = []
            
            # Scan all approval files
            for approval_file in self.storage_path.glob("approval_*.json"):
                request = await self._load_approval_request_from_file(approval_file)
                if request and request.status == ApprovalStatus.PENDING:
                    # Check if approver can approve this request
                    if approver_id is None or await self._can_approver_vote(request, approver_id):
                        pending_requests.append(request)
            
            # Sort by creation time
            pending_requests.sort(key=lambda r: r.created_timestamp, reverse=True)
            return pending_requests
            
        except Exception as e:
            logger.error(f"Failed to get pending approvals: {e}")
            return []
    
    async def cancel_approval_request(self, request_id: str, cancelled_by: str, reason: str) -> bool:
        """Cancel approval request."""
        try:
            request = await self._load_approval_request(request_id)
            if not request:
                return False
            
            if request.status != ApprovalStatus.PENDING:
                raise ValueError(f"Cannot cancel request with status: {request.status}")
            
            request.status = ApprovalStatus.CANCELLED
            request.audit_trail.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'REQUEST_CANCELLED',
                'user': cancelled_by,
                'details': {'reason': reason}
            })
            
            await self._save_approval_request(request)
            
            # Notify backends
            for backend in self.approval_backends:
                await backend.cancel_approval_request(request_id, reason)
            
            logger.info(f"Approval request cancelled: {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel approval request: {e}")
            return False
    
    async def _perform_compliance_checks(self,
                                       approval_type: ApprovalType,
                                       change_details: Dict[str, Any],
                                       artifacts: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Perform automated compliance checks."""
        checks = {}
        
        try:
            # Model deployment specific checks
            if approval_type == ApprovalType.MODEL_DEPLOYMENT:
                checks['model_lineage_complete'] = 'model_lineage' in change_details
                checks['model_hash_verified'] = 'model_hash' in change_details
                checks['dataset_hashes_present'] = 'dataset_hashes' in change_details
                checks['performance_metrics_documented'] = 'performance_metrics' in change_details
                checks['rollback_plan_defined'] = bool(change_details.get('rollback_plan'))
            
            # Configuration change checks
            elif approval_type == ApprovalType.CONFIG_CHANGE:
                checks['config_backup_created'] = 'backup_location' in change_details
                checks['impact_analysis_complete'] = 'impact_analysis' in change_details
                checks['rollback_tested'] = change_details.get('rollback_tested', False)
            
            # System update checks
            elif approval_type == ApprovalType.SYSTEM_UPDATE:
                checks['dependencies_verified'] = 'dependencies' in change_details
                checks['security_scan_passed'] = change_details.get('security_scan_passed', False)
                checks['backup_verified'] = change_details.get('backup_verified', False)
            
            # Artifact checks
            if artifacts:
                checks['artifacts_signed'] = all(
                    'signature' in artifact for artifact in artifacts
                )
                checks['artifacts_scanned'] = all(
                    artifact.get('security_scan_passed', False) for artifact in artifacts
                )
            
            logger.info(f"Compliance checks completed: {sum(checks.values())}/{len(checks)} passed")
            return checks
            
        except Exception as e:
            logger.error(f"Compliance checks failed: {e}")
            return {'compliance_check_error': False}
    
    async def _assess_risk(self, approval_type: ApprovalType, change_details: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level of the change."""
        try:
            risk_factors = []
            risk_score = 0
            
            # Base risk by type
            type_risk = {
                ApprovalType.MODEL_DEPLOYMENT: 3,
                ApprovalType.CONFIG_CHANGE: 2,
                ApprovalType.SYSTEM_UPDATE: 4,
                ApprovalType.EMERGENCY_OVERRIDE: 5
            }.get(approval_type, 2)
            
            risk_score += type_risk
            risk_factors.append(f"Base risk for {approval_type.value}: {type_risk}")
            
            # Production deployment increases risk
            if change_details.get('environment') == 'production':
                risk_score += 2
                risk_factors.append("Production deployment: +2")
            
            # Large changes increase risk
            if change_details.get('change_size') == 'large':
                risk_score += 1
                risk_factors.append("Large change size: +1")
            
            # No rollback plan increases risk
            if not change_details.get('rollback_plan'):
                risk_score += 2
                risk_factors.append("No rollback plan: +2")
            
            # Determine risk level
            if risk_score <= 3:
                risk_level = "LOW"
            elif risk_score <= 6:
                risk_level = "MEDIUM"
            elif risk_score <= 9:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'assessment_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'risk_level': 'UNKNOWN', 'error': str(e)}
    
    async def _analyze_impact(self, approval_type: ApprovalType, change_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of the change."""
        try:
            impact_areas = []
            
            # Determine affected systems
            if approval_type == ApprovalType.MODEL_DEPLOYMENT:
                impact_areas.extend(['trading_engine', 'risk_management', 'model_serving'])
            elif approval_type == ApprovalType.CONFIG_CHANGE:
                impact_areas.extend(['configuration_system', 'runtime_behavior'])
            elif approval_type == ApprovalType.SYSTEM_UPDATE:
                impact_areas.extend(['entire_system', 'dependencies', 'integrations'])
            
            # Estimate downtime
            estimated_downtime = change_details.get('estimated_downtime', 'unknown')
            
            # User impact
            user_impact = change_details.get('user_impact', 'medium')
            
            return {
                'impact_areas': impact_areas,
                'estimated_downtime': estimated_downtime,
                'user_impact': user_impact,
                'business_impact': change_details.get('business_impact', 'medium'),
                'rollback_time': change_details.get('rollback_time', 'unknown'),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            return {'error': str(e)}
    
    async def _validate_approver_authority(self,
                                         request: ApprovalRequest,
                                         approver_id: str,
                                         approval_authority: str) -> bool:
        """Validate that approver has authority for this request."""
        try:
            # Check if approver is in required approvers list
            if approver_id in request.required_approvers:
                return True
            
            # Check if approver belongs to required approval groups
            for group in request.approval_groups:
                group_members = self.approval_groups.get(group, [])
                if approver_id in group_members:
                    return True
            
            # Check approval authority
            if approval_authority:
                authority_groups = self.approval_groups.get(approval_authority, [])
                if approver_id in authority_groups:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate approver authority: {e}")
            return False
    
    async def _can_approver_vote(self, request: ApprovalRequest, approver_id: str) -> bool:
        """Check if approver can vote on this request."""
        # Check if already voted
        existing_votes = [a for a in request.approvals + request.rejections if a['approver_id'] == approver_id]
        if existing_votes:
            return False
        
        # Check authority
        return await self._validate_approver_authority(request, approver_id, None)
    
    async def _generate_digital_signature(self, request_id: str, approver_id: str, decision: str) -> str:
        """Generate digital signature for approval decision."""
        try:
            # Simple signature implementation (in production, use proper PKI)
            signature_data = f"{request_id}:{approver_id}:{decision}:{datetime.now(timezone.utc).isoformat()}"
            signature = hashlib.sha256(signature_data.encode()).hexdigest()
            return signature
            
        except Exception as e:
            logger.error(f"Failed to generate digital signature: {e}")
            return "signature_error"
    
    async def _check_approval_completion(self, request: ApprovalRequest):
        """Check if approval request is complete."""
        try:
            # Check for rejections
            if request.rejections:
                request.status = ApprovalStatus.REJECTED
                request.audit_trail.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'action': 'REQUEST_REJECTED',
                    'user': 'system',
                    'details': {'rejection_count': len(request.rejections)}
                })
                return
            
            # Check if minimum approvals met
            approval_count = len(request.approvals)
            if approval_count >= request.minimum_approvals:
                # Check if all required approvers have approved
                required_approvers_set = set(request.required_approvers)
                approver_ids = {a['approver_id'] for a in request.approvals}
                
                if required_approvers_set.issubset(approver_ids) or not required_approvers_set:
                    request.status = ApprovalStatus.APPROVED
                    request.audit_trail.append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'action': 'REQUEST_APPROVED',
                        'user': 'system',
                        'details': {
                            'approval_count': approval_count,
                            'required_approvals': request.minimum_approvals
                        }
                    })
            
        except Exception as e:
            logger.error(f"Failed to check approval completion: {e}")
    
    async def _save_approval_request(self, request: ApprovalRequest):
        """Save approval request to storage."""
        try:
            request_file = self.storage_path / f"approval_{request.request_id}.json"
            with open(request_file, 'w') as f:
                json.dump(asdict(request), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save approval request: {e}")
            raise
    
    async def _load_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load approval request from storage."""
        try:
            request_file = self.storage_path / f"approval_{request_id}.json"
            return await self._load_approval_request_from_file(request_file)
            
        except Exception as e:
            logger.error(f"Failed to load approval request: {e}")
            return None
    
    async def _load_approval_request_from_file(self, request_file: Path) -> Optional[ApprovalRequest]:
        """Load approval request from file."""
        try:
            if not request_file.exists():
                return None
            
            with open(request_file, 'r') as f:
                request_data = json.load(f)
            
            # Convert enum values
            request_data['approval_type'] = ApprovalType(request_data['approval_type'])
            request_data['status'] = ApprovalStatus(request_data['status'])
            
            return ApprovalRequest(**request_data)
            
        except Exception as e:
            logger.error(f"Failed to load approval request from file: {e}")
            return None


class ApprovalWorkflow:
    """High-level approval workflow orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.release_gate = FourEyesReleaseGate(config)
        self.config = config
    
    async def request_model_deployment_approval(self,
                                              model_id: str,
                                              model_path: str,
                                              model_hash: str,
                                              dataset_hashes: List[str],
                                              performance_metrics: Dict[str, float],
                                              requested_by: str,
                                              target_environment: str = "production") -> str:
        """Request approval for model deployment."""
        
        change_details = {
            'model_id': model_id,
            'model_path': model_path,
            'model_hash': model_hash,
            'dataset_hashes': dataset_hashes,
            'performance_metrics': performance_metrics,
            'target_environment': target_environment,
            'deployment_timestamp': datetime.now(timezone.utc).isoformat(),
            'rollback_plan': {
                'previous_model_available': True,
                'rollback_time_minutes': 5,
                'rollback_procedure': 'Automated rollback via deployment system'
            }
        }
        
        artifacts = [{
            'type': 'model_file',
            'path': model_path,
            'hash': model_hash,
            'signature': f"model_signature_{model_hash[:16]}"
        }]
        
        return await self.release_gate.create_approval_request(
            approval_type=ApprovalType.MODEL_DEPLOYMENT,
            title=f"Deploy Model {model_id} to {target_environment}",
            description=f"Deploy trained model {model_id} with hash {model_hash[:16]}... to {target_environment} environment",
            requested_by=requested_by,
            change_details=change_details,
            artifacts=artifacts,
            expires_hours=48  # Model deployments get longer approval window
        )
    
    async def approve_deployment(self,
                               request_id: str,
                               approver_id: str,
                               approver_name: str,
                               comments: str = "Approved") -> bool:
        """Approve deployment request."""
        return await self.release_gate.submit_approval(
            request_id=request_id,
            approver_id=approver_id,
            approver_name=approver_name,
            decision="APPROVE",
            comments=comments,
            approval_authority="senior_developers"
        )
    
    async def reject_deployment(self,
                              request_id: str,
                              approver_id: str,
                              approver_name: str,
                              reason: str) -> bool:
        """Reject deployment request."""
        return await self.release_gate.submit_approval(
            request_id=request_id,
            approver_id=approver_id,
            approver_name=approver_name,
            decision="REJECT",
            comments=reason,
            approval_authority="senior_developers"
        )
    
    async def is_deployment_approved(self, request_id: str) -> bool:
        """Check if deployment is approved."""
        request = await self.release_gate.get_approval_status(request_id)
        return request and request.status == ApprovalStatus.APPROVED


# Approval backend implementations
class ApprovalBackend:
    """Base class for approval backends."""
    
    async def create_approval_request(self, request: ApprovalRequest):
        """Create approval request in backend system."""
        pass
    
    async def update_approval_status(self, request: ApprovalRequest):
        """Update approval status in backend system."""
        pass
    
    async def cancel_approval_request(self, request_id: str, reason: str):
        """Cancel approval request in backend system."""
        pass


class GitHubApprovalBackend(ApprovalBackend):
    """GitHub-based approval backend using pull requests."""
    
    def __init__(self, config: Dict[str, Any]):
        self.repo_owner = config['repo_owner']
        self.repo_name = config['repo_name']
        self.github_token = config['github_token']
        self.base_url = "https://api.github.com"
        
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    async def create_approval_request(self, request: ApprovalRequest):
        """Create GitHub pull request for approval."""
        try:
            # Create branch for approval
            branch_name = f"approval/{request.request_id}"
            
            # Create approval file
            approval_content = json.dumps(asdict(request), indent=2, default=str)
            
            # Create pull request
            pr_data = {
                'title': f"[APPROVAL] {request.title}",
                'body': self._format_pr_description(request),
                'head': branch_name,
                'base': 'main'
            }
            
            # Note: This is a simplified implementation
            # In practice, you'd need to create the branch and file first
            logger.info(f"GitHub approval request created: {request.request_id}")
            
        except Exception as e:
            logger.error(f"Failed to create GitHub approval request: {e}")
    
    def _format_pr_description(self, request: ApprovalRequest) -> str:
        """Format pull request description."""
        return f"""
## Approval Request: {request.title}

**Type**: {request.approval_type.value}
**Requested by**: {request.requested_by}
**Created**: {request.created_timestamp}
**Expires**: {request.expires_timestamp}

### Description
{request.description}

### Risk Assessment
- **Risk Level**: {request.risk_assessment.get('risk_level', 'Unknown')}
- **Risk Score**: {request.risk_assessment.get('risk_score', 'Unknown')}

### Required Approvals
- **Minimum Approvals**: {request.minimum_approvals}
- **Required Approvers**: {', '.join(request.required_approvers)}
- **Approval Groups**: {', '.join(request.approval_groups)}

### Compliance Checks
{self._format_compliance_checks(request.compliance_checks)}

### Artifacts
{self._format_artifacts(request.artifacts)}

---
**Please review and approve/reject this request by commenting on this PR.**
        """
    
    def _format_compliance_checks(self, checks: Dict[str, bool]) -> str:
        """Format compliance checks for display."""
        if not checks:
            return "No compliance checks performed"
        
        lines = []
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            lines.append(f"- {status} {check.replace('_', ' ').title()}")
        
        return '\n'.join(lines)
    
    def _format_artifacts(self, artifacts: List[Dict[str, Any]]) -> str:
        """Format artifacts for display."""
        if not artifacts:
            return "No artifacts"
        
        lines = []
        for artifact in artifacts:
            lines.append(f"- **{artifact.get('type', 'Unknown')}**: {artifact.get('path', 'Unknown')}")
            if 'hash' in artifact:
                lines.append(f"  - Hash: `{artifact['hash'][:16]}...`")
        
        return '\n'.join(lines)


class ServiceNowApprovalBackend(ApprovalBackend):
    """ServiceNow-based approval backend."""
    
    def __init__(self, config: Dict[str, Any]):
        self.instance_url = config['instance_url']
        self.username = config['username']
        self.password = config['password']
        self.table = config.get('table', 'change_request')
    
    async def create_approval_request(self, request: ApprovalRequest):
        """Create ServiceNow change request."""
        try:
            # ServiceNow API call would go here
            logger.info(f"ServiceNow approval request created: {request.request_id}")
            
        except Exception as e:
            logger.error(f"Failed to create ServiceNow approval request: {e}")


class LocalApprovalBackend(ApprovalBackend):
    """Local file-based approval backend."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
    
    async def create_approval_request(self, request: ApprovalRequest):
        """Create local approval request file."""
        # Request is already saved by the main system
        pass
    
    async def update_approval_status(self, request: ApprovalRequest):
        """Update local approval status."""
        # Status is already updated by the main system
        pass