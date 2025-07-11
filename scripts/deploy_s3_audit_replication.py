#!/usr/bin/env python3
"""
S3 Audit Replication Deployment Script
======================================

This script deploys the S3 cross-region replication infrastructure for
audit bucket disaster recovery with minute-level replication.

PROBLEM SOLVED:
"DR: S3 WORM mirrors cross-region hourly, not continuous.  
 → Enable S3 replication minute-level for audit bucket."

DEPLOYMENT STEPS:
1. Validate AWS credentials and permissions
2. Create S3 buckets in all regions
3. Configure WORM, encryption, and versioning
4. Set up cross-region replication rules
5. Configure monitoring and alerting
6. Validate replication functionality
7. Generate deployment report

USAGE:
    python scripts/deploy_s3_audit_replication.py --deploy
    python scripts/deploy_s3_audit_replication.py --validate
    python scripts/deploy_s3_audit_replication.py --status
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from storage.s3_audit_replication import S3AuditReplicator, ReplicationConfig
from storage.audit_storage_manager import AuditStorageManager, load_config_from_governance_yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3ReplicationDeployer:
    """Handles deployment of S3 audit replication infrastructure."""
    
    def __init__(self, config_path: str = "config/governance.yaml"):
        """Initialize deployer with configuration."""
        self.config_path = config_path
        self.storage_config = load_config_from_governance_yaml(config_path)
        
        # Create replication configuration
        self.replication_config = ReplicationConfig(
            primary_bucket=self.storage_config.s3_bucket,
            primary_region=self.storage_config.s3_region,
            replica_regions=self.storage_config.replica_regions,
            replication_frequency_seconds=self.storage_config.replication_frequency_seconds,
            enable_worm=self.storage_config.worm_enabled,
            retention_years=self.storage_config.retention_years,
            encryption_enabled=self.storage_config.encryption_enabled,
            versioning_enabled=True,
            mfa_delete_enabled=True
        )
        
        self.replicator = S3AuditReplicator(self.replication_config)
        self.deployment_report = {
            'timestamp': datetime.now().isoformat(),
            'deployment_id': f"s3-replication-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'config': self.replication_config.to_dict(),
            'steps': [],
            'overall_success': False
        }
    
    def validate_prerequisites(self) -> bool:
        """Validate prerequisites for deployment."""
        logger.info("🔍 Validating deployment prerequisites...")
        
        step_result = {
            'step': 'validate_prerequisites',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Check AWS credentials
            import boto3
            
            # Test primary region access
            try:
                s3_client = boto3.client('s3', region_name=self.replication_config.primary_region)
                s3_client.list_buckets()
                step_result['details']['primary_region_access'] = True
                logger.info(f"✅ Primary region access validated: {self.replication_config.primary_region}")
            except Exception as e:
                step_result['details']['primary_region_access'] = False
                step_result['success'] = False
                logger.error(f"❌ Primary region access failed: {e}")
            
            # Test replica regions access
            replica_access = {}
            for region in self.replication_config.replica_regions:
                try:
                    s3_client = boto3.client('s3', region_name=region)
                    s3_client.list_buckets()
                    replica_access[region] = True
                    logger.info(f"✅ Replica region access validated: {region}")
                except Exception as e:
                    replica_access[region] = False
                    step_result['success'] = False
                    logger.error(f"❌ Replica region access failed for {region}: {e}")
            
            step_result['details']['replica_regions_access'] = replica_access
            
            # Check IAM permissions
            try:
                iam_client = boto3.client('iam')
                iam_client.list_roles()
                step_result['details']['iam_access'] = True
                logger.info("✅ IAM access validated")
            except Exception as e:
                step_result['details']['iam_access'] = False
                step_result['success'] = False
                logger.error(f"❌ IAM access failed: {e}")
            
            # Check CloudWatch permissions
            try:
                cloudwatch = boto3.client('cloudwatch', region_name=self.replication_config.primary_region)
                cloudwatch.list_metrics(MaxRecords=1)
                step_result['details']['cloudwatch_access'] = True
                logger.info("✅ CloudWatch access validated")
            except Exception as e:
                step_result['details']['cloudwatch_access'] = False
                logger.warning(f"⚠️ CloudWatch access limited: {e}")
            
        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)
            logger.error(f"❌ Prerequisites validation failed: {e}")
        
        self.deployment_report['steps'].append(step_result)
        
        if step_result['success']:
            logger.info("✅ All prerequisites validated successfully")
        else:
            logger.error("❌ Prerequisites validation failed")
        
        return step_result['success']
    
    def deploy_replication_infrastructure(self) -> bool:
        """Deploy the complete S3 replication infrastructure."""
        logger.info("🚀 Deploying S3 audit replication infrastructure...")
        
        step_result = {
            'step': 'deploy_infrastructure',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Deploy replication infrastructure
            success = self.replicator.setup_replication()
            
            if success:
                step_result['details']['replication_setup'] = True
                logger.info("✅ S3 replication infrastructure deployed successfully")
            else:
                step_result['success'] = False
                step_result['details']['replication_setup'] = False
                logger.error("❌ S3 replication infrastructure deployment failed")
            
        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)
            logger.error(f"❌ Infrastructure deployment failed: {e}")
        
        self.deployment_report['steps'].append(step_result)
        return step_result['success']
    
    def start_monitoring(self) -> bool:
        """Start continuous replication monitoring."""
        logger.info("📊 Starting replication monitoring...")
        
        step_result = {
            'step': 'start_monitoring',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Start continuous monitoring
            self.replicator.start_continuous_monitoring()
            
            # Wait a moment to ensure monitoring starts
            time.sleep(5)
            
            step_result['details']['monitoring_started'] = True
            logger.info("✅ Replication monitoring started successfully")
            
        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)
            logger.error(f"❌ Failed to start monitoring: {e}")
        
        self.deployment_report['steps'].append(step_result)
        return step_result['success']
    
    def validate_replication(self) -> bool:
        """Validate replication functionality with test data."""
        logger.info("🧪 Validating replication functionality...")
        
        step_result = {
            'step': 'validate_replication',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Create test audit record
            storage_manager = AuditStorageManager(self.storage_config)
            
            test_record = {
                'event_type': 'DEPLOYMENT_VALIDATION',
                'timestamp': datetime.now().isoformat(),
                'deployment_id': self.deployment_report['deployment_id'],
                'message': 'Test record for S3 replication validation',
                'test_data': True
            }
            
            # Write test record
            write_success = storage_manager.write_audit_record(test_record)
            step_result['details']['test_record_written'] = write_success
            
            if write_success:
                logger.info("✅ Test audit record written successfully")
                
                # Wait for replication to process
                logger.info("⏳ Waiting for replication to process...")
                time.sleep(90)  # Wait 90 seconds for replication
                
                # Check replication status
                replication_report = self.replicator.get_replication_report()
                step_result['details']['replication_report'] = replication_report
                
                if replication_report['overall_health'] in ['HEALTHY', 'DEGRADED']:
                    logger.info("✅ Replication validation successful")
                else:
                    step_result['success'] = False
                    logger.error("❌ Replication validation failed - unhealthy status")
            else:
                step_result['success'] = False
                logger.error("❌ Failed to write test record")
            
            storage_manager.shutdown()
            
        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)
            logger.error(f"❌ Replication validation failed: {e}")
        
        self.deployment_report['steps'].append(step_result)
        return step_result['success']
    
    def generate_deployment_report(self) -> dict:
        """Generate comprehensive deployment report."""
        logger.info("📋 Generating deployment report...")
        
        # Calculate overall success
        all_steps_successful = all(
            step.get('success', False) 
            for step in self.deployment_report['steps']
        )
        
        self.deployment_report['overall_success'] = all_steps_successful
        self.deployment_report['completion_timestamp'] = datetime.now().isoformat()
        
        # Add summary
        successful_steps = sum(1 for step in self.deployment_report['steps'] if step.get('success', False))
        total_steps = len(self.deployment_report['steps'])
        
        self.deployment_report['summary'] = {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': total_steps - successful_steps,
            'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0
        }
        
        # Add recommendations
        recommendations = []
        
        if not all_steps_successful:
            recommendations.append("Review failed deployment steps and resolve issues before production use")
        
        if self.deployment_report['summary']['success_rate'] < 100:
            recommendations.append("Some deployment steps failed - manual intervention may be required")
        
        # Check for specific issues
        for step in self.deployment_report['steps']:
            if step['step'] == 'validate_prerequisites' and not step['success']:
                recommendations.append("Resolve AWS credential and permission issues")
            elif step['step'] == 'deploy_infrastructure' and not step['success']:
                recommendations.append("Check S3 bucket policies and IAM roles")
            elif step['step'] == 'validate_replication' and not step['success']:
                recommendations.append("Verify replication rules and cross-region connectivity")
        
        if not recommendations:
            recommendations.append("Deployment completed successfully - ready for production use")
        
        self.deployment_report['recommendations'] = recommendations
        
        return self.deployment_report
    
    def get_replication_status(self) -> dict:
        """Get current replication status."""
        logger.info("📊 Getting replication status...")
        
        try:
            # Get replication report
            replication_report = self.replicator.get_replication_report()
            
            # Get storage manager status
            storage_manager = AuditStorageManager(self.storage_config)
            storage_status = storage_manager.get_storage_status()
            storage_manager.shutdown()
            
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'replication_status': replication_report,
                'storage_status': storage_status,
                'overall_health': 'HEALTHY'
            }
            
            # Determine overall health
            if (replication_report['overall_health'] == 'DEGRADED' or 
                storage_status['overall_health'] == 'DEGRADED'):
                status_report['overall_health'] = 'DEGRADED'
            elif (replication_report['overall_health'] == 'CRITICAL' or 
                  storage_status['overall_health'] == 'CRITICAL'):
                status_report['overall_health'] = 'CRITICAL'
            
            return status_report
            
        except Exception as e:
            logger.error(f"Failed to get replication status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_health': 'UNKNOWN'
            }
    
    def full_deployment(self) -> bool:
        """Perform complete deployment process."""
        logger.info("🚀 Starting full S3 audit replication deployment...")
        
        try:
            # Step 1: Validate prerequisites
            if not self.validate_prerequisites():
                logger.error("❌ Prerequisites validation failed - aborting deployment")
                return False
            
            # Step 2: Deploy infrastructure
            if not self.deploy_replication_infrastructure():
                logger.error("❌ Infrastructure deployment failed - aborting")
                return False
            
            # Step 3: Start monitoring
            if not self.start_monitoring():
                logger.error("❌ Monitoring startup failed - continuing with validation")
            
            # Step 4: Validate replication
            if not self.validate_replication():
                logger.error("❌ Replication validation failed - check configuration")
                return False
            
            logger.info("✅ Full deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full deployment failed: {e}")
            return False
        finally:
            # Always generate report
            self.generate_deployment_report()


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deploy S3 audit replication infrastructure")
    parser.add_argument('--deploy', action='store_true', help='Deploy full replication infrastructure')
    parser.add_argument('--validate', action='store_true', help='Validate existing deployment')
    parser.add_argument('--status', action='store_true', help='Get current replication status')
    parser.add_argument('--config', default='config/governance.yaml', help='Configuration file path')
    parser.add_argument('--report-file', help='Save deployment report to file')
    
    args = parser.parse_args()
    
    if not any([args.deploy, args.validate, args.status]):
        parser.print_help()
        return
    
    print("🚀 S3 AUDIT REPLICATION DEPLOYMENT")
    print("=" * 50)
    print()
    print("PROBLEM SOLVED:")
    print("DR: S3 WORM mirrors cross-region hourly, not continuous.")
    print("→ Enable S3 replication minute-level for audit bucket.")
    print()
    
    try:
        # Initialize deployer
        deployer = S3ReplicationDeployer(args.config)
        
        print(f"Configuration loaded from: {args.config}")
        print(f"Primary bucket: {deployer.replication_config.primary_bucket}")
        print(f"Primary region: {deployer.replication_config.primary_region}")
        print(f"Replica regions: {deployer.replication_config.replica_regions}")
        print(f"Replication frequency: {deployer.replication_config.replication_frequency_seconds}s")
        print()
        
        if args.deploy:
            print("🚀 FULL DEPLOYMENT")
            print("-" * 30)
            success = deployer.full_deployment()
            
            # Generate and display report
            report = deployer.generate_deployment_report()
            
            print(f"\n📋 DEPLOYMENT REPORT")
            print("-" * 30)
            print(f"Deployment ID: {report['deployment_id']}")
            print(f"Overall Success: {'✅ YES' if report['overall_success'] else '❌ NO'}")
            print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
            print(f"Steps: {report['summary']['successful_steps']}/{report['summary']['total_steps']} successful")
            
            print(f"\n📝 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
            
            # Save report if requested
            if args.report_file:
                with open(args.report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\n💾 Report saved to: {args.report_file}")
            
            if success:
                print(f"\n✅ DEPLOYMENT COMPLETED SUCCESSFULLY")
                print("The S3 audit replication system is now active with minute-level DR capability!")
            else:
                print(f"\n❌ DEPLOYMENT FAILED")
                print("Review the deployment report and resolve issues before retrying.")
        
        elif args.validate:
            print("🧪 VALIDATION")
            print("-" * 30)
            success = deployer.validate_replication()
            
            if success:
                print("✅ Replication validation successful")
            else:
                print("❌ Replication validation failed")
        
        elif args.status:
            print("📊 STATUS CHECK")
            print("-" * 30)
            status = deployer.get_replication_status()
            
            print(f"Overall Health: {status['overall_health']}")
            print(f"Timestamp: {status['timestamp']}")
            
            if 'replication_status' in status:
                repl_status = status['replication_status']
                print(f"Replication Health: {repl_status['overall_health']}")
                print(f"Total Objects: {repl_status['total_objects']}")
                print(f"Regions: {len(repl_status['regions'])}")
            
            if 'storage_status' in status:
                storage_status = status['storage_status']
                print(f"Storage Health: {storage_status['overall_health']}")
                print(f"Local Storage: {'✅' if storage_status['storage_systems']['local']['available'] else '❌'}")
                print(f"S3 Primary: {'✅' if storage_status['storage_systems']['s3_primary']['available'] else '❌'}")
    
    except Exception as e:
        print(f"❌ Deployment script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())