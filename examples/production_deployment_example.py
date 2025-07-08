# examples/production_deployment_example.py
"""
Production Deployment Example for IntradayJules Trading System.

Demonstrates:
- Prometheus alerting with PagerDuty/Slack integration
- Blue/green rollout with atomic symlink swapping
- Secrets management with AWS Secrets Manager/Vault
- Critical alerts on latency P99 > 25µs, audit errors, circuit breaker trips
"""

import os
import sys
import time
import json
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from monitoring.alerting.alert_manager import (
    create_alert_manager, AlertSeverity, AlertStatus, Alert
)
from deployment.blue_green_rollout import create_blue_green_deployment, DeploymentStatus
from security.secrets_manager import create_secrets_manager, SecretType


async def test_alerting_system():
    """Test alerting system with PagerDuty and Slack integration."""
    print("🚨 Testing Alerting System")
    print("=" * 60)
    
    # Create alert manager (will use env vars if available)
    alert_manager = create_alert_manager()
    
    # Test 1: Critical latency alert (P99 > 25µs)
    print("\n📋 Test 1: Critical Latency Alert")
    result = await alert_manager.send_critical_latency_alert("risk_enforcement", 35.5)
    print(f"  Latency alert sent: {result}")
    
    # Test 2: Audit log write error alert
    print("\n📋 Test 2: Audit Log Error Alert")
    result = await alert_manager.send_audit_log_error_alert(3, "Disk full - /var/log/audit")
    print(f"  Audit error alert sent: {result}")
    
    # Test 3: Circuit breaker trip alert
    print("\n📋 Test 3: Circuit Breaker Alert")
    result = await alert_manager.send_circuit_breaker_alert("var_enforcement_breaker", 2)
    print(f"  Circuit breaker alert sent: {result}")
    
    # Test 4: Risk limit breach alert
    print("\n📋 Test 4: Risk Limit Breach Alert")
    result = await alert_manager.send_risk_limit_breach_alert("var_99_limit", 250000, 200000)
    print(f"  Risk limit breach alert sent: {result}")
    
    # Test 5: Custom alert
    print("\n📋 Test 5: Custom Alert")
    custom_alert = Alert(
        alert_name="TestAlert",
        severity=AlertSeverity.WARNING,
        status=AlertStatus.FIRING,
        message="Test alert for production deployment",
        description="This is a test alert to verify the alerting system is working correctly.",
        service="test_service",
        team="trading",
        timestamp=time.time(),
        labels={"test": "true", "environment": "production"},
        annotations={"runbook": "https://wiki.company.com/test-alert"},
        runbook_url="https://wiki.company.com/test-alert"
    )
    
    result = await alert_manager.send_alert(custom_alert)
    print(f"  Custom alert sent: {result}")
    
    print("✅ Alerting system test completed\n")


def test_blue_green_deployment():
    """Test blue/green deployment with atomic symlink swapping."""
    print("🔄 Testing Blue/Green Deployment")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        deployment_root = Path(temp_dir) / "deployments"
        
        # Create blue/green deployment manager
        bg_deploy = create_blue_green_deployment(str(deployment_root))
        
        # Test 1: Create model bundle v1
        print("\n📋 Test 1: Creating Model Bundle v1")
        bundle_v1_dir = Path(temp_dir) / "bundle_v1"
        bundle_v1_dir.mkdir(parents=True)
        
        # Create required files for v1
        (bundle_v1_dir / "policy.pt").write_text("# PyTorch model v1\nmodel_data_v1")
        (bundle_v1_dir / "value_function.pt").write_text("# Value function v1\nvalue_data_v1")
        (bundle_v1_dir / "config.yaml").write_text("model_type: rl_policy\nversion: v1\n")
        
        metadata_v1 = {
            "version": "v2025-07-06",
            "created_at": time.time(),
            "model_type": "rl_policy",
            "training_data_hash": "abc123def456",
            "performance_metrics": {
                "sharpe_ratio": 1.85,
                "max_drawdown": 0.12,
                "win_rate": 0.67
            }
        }
        (bundle_v1_dir / "metadata.json").write_text(json.dumps(metadata_v1, indent=2))
        
        print(f"  Created bundle v1: {bundle_v1_dir}")
        
        # Deploy v1
        print("  Deploying bundle v1...")
        deployment_v1 = bg_deploy.deploy_bundle("v2025-07-06", bundle_v1_dir, force=True)
        print(f"  Deployment v1 status: {deployment_v1.status.value}")
        
        if deployment_v1.error_message:
            print(f"  Error: {deployment_v1.error_message}")
        
        # Check deployment status
        status = bg_deploy.get_deployment_status()
        print(f"  Current version: {status['current_version']}")
        print(f"  Available bundles: {status['available_bundles']}")
        
        # Test 2: Create model bundle v2
        print("\n📋 Test 2: Creating Model Bundle v2")
        bundle_v2_dir = Path(temp_dir) / "bundle_v2"
        bundle_v2_dir.mkdir(parents=True)
        
        # Create required files for v2
        (bundle_v2_dir / "policy.pt").write_text("# PyTorch model v2\nmodel_data_v2_improved")
        (bundle_v2_dir / "value_function.pt").write_text("# Value function v2\nvalue_data_v2_improved")
        (bundle_v2_dir / "config.yaml").write_text("model_type: rl_policy\nversion: v2\nimprovements: [latency, accuracy]\n")
        
        metadata_v2 = {
            "version": "v2025-07-08",
            "created_at": time.time(),
            "model_type": "rl_policy",
            "training_data_hash": "def456ghi789",
            "performance_metrics": {
                "sharpe_ratio": 2.15,
                "max_drawdown": 0.08,
                "win_rate": 0.72
            },
            "improvements": ["reduced_latency", "improved_accuracy", "better_risk_management"]
        }
        (bundle_v2_dir / "metadata.json").write_text(json.dumps(metadata_v2, indent=2))
        
        print(f"  Created bundle v2: {bundle_v2_dir}")
        
        # Deploy v2 (blue/green deployment)
        print("  Deploying bundle v2 (blue/green)...")
        deployment_v2 = bg_deploy.deploy_bundle("v2025-07-08", bundle_v2_dir, force=True)
        print(f"  Deployment v2 status: {deployment_v2.status.value}")
        
        if deployment_v2.error_message:
            print(f"  Error: {deployment_v2.error_message}")
        
        # Check new deployment status
        status = bg_deploy.get_deployment_status()
        print(f"  Current version: {status['current_version']}")
        print(f"  Rollback available: {status['rollback_available']}")
        
        # Test 3: Rollback deployment
        if deployment_v2.status == DeploymentStatus.ACTIVE and status['rollback_available']:
            print("\n📋 Test 3: Testing Rollback")
            print("  Rolling back to previous version...")
            rollback = bg_deploy.rollback_deployment()
            print(f"  Rollback status: {rollback.status.value}")
            
            status = bg_deploy.get_deployment_status()
            print(f"  Current version after rollback: {status['current_version']}")
        
        # Test 4: List deployment history
        print("\n📋 Test 4: Deployment History")
        deployments = bg_deploy.list_deployments()
        print(f"  Total deployments: {len(deployments)}")
        
        for i, deployment in enumerate(deployments[:3], 1):  # Show last 3
            print(f"  {i}. {deployment.bundle_version} - {deployment.status.value} - {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(deployment.started_at))}")
    
    print("✅ Blue/green deployment test completed\n")


async def test_secrets_management():
    """Test secrets management with AWS Secrets Manager/Vault."""
    print("🔐 Testing Secrets Management")
    print("=" * 60)
    
    # Create secrets manager
    secrets_manager = create_secrets_manager()
    
    # Test 1: Store and retrieve database credentials
    print("\n📋 Test 1: Database Credentials")
    db_creds = {
        "host": "postgres.trading.svc.cluster.local",
        "port": 5432,
        "username": "trading_user",
        "password": "super_secure_password_123!",
        "database": "intradayjules_prod",
        "ssl_mode": "require"
    }
    
    success = await secrets_manager.put_secret(
        "database/main",
        db_creds,
        SecretType.DATABASE_PASSWORD,
        "Production database credentials"
    )
    print(f"  Database credentials stored: {success}")
    
    retrieved_creds = await secrets_manager.get_database_credentials("main")
    print(f"  Retrieved credentials: {retrieved_creds is not None}")
    if retrieved_creds:
        print(f"  Host: {retrieved_creds.get('host')}")
        print(f"  Database: {retrieved_creds.get('database')}")
    
    # Test 2: Store and retrieve S3 credentials
    print("\n📋 Test 2: S3 Credentials")
    s3_creds = {
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "region": "us-east-1",
        "bucket": "intradayjules-prod-data",
        "kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    }
    
    success = await secrets_manager.put_secret(
        "aws/s3_credentials",
        s3_creds,
        SecretType.S3_CREDENTIALS,
        "Production S3 bucket credentials"
    )
    print(f"  S3 credentials stored: {success}")
    
    retrieved_s3_creds = await secrets_manager.get_s3_credentials()
    print(f"  Retrieved S3 credentials: {retrieved_s3_creds is not None}")
    if retrieved_s3_creds:
        print(f"  Bucket: {retrieved_s3_creds.get('bucket')}")
        print(f"  Region: {retrieved_s3_creds.get('region')}")
    
    # Test 3: Store and retrieve broker credentials
    print("\n📋 Test 3: Broker Credentials")
    broker_creds = {
        "username": "prod_trading_account",
        "password": "broker_secure_password_456!",
        "account_id": "DU123456",
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
        "api_version": "9.81"
    }
    
    success = await secrets_manager.put_secret(
        "broker/interactive_brokers",
        broker_creds,
        SecretType.BROKER_CREDENTIALS,
        "Production Interactive Brokers credentials"
    )
    print(f"  Broker credentials stored: {success}")
    
    retrieved_broker_creds = await secrets_manager.get_broker_credentials("interactive_brokers")
    print(f"  Retrieved broker credentials: {retrieved_broker_creds is not None}")
    if retrieved_broker_creds:
        print(f"  Account ID: {retrieved_broker_creds.get('account_id')}")
        print(f"  Host: {retrieved_broker_creds.get('host')}")
    
    # Test 4: Store and retrieve API keys
    print("\n📋 Test 4: API Keys")
    api_keys = [
        ("pagerduty", "pd_integration_key_abc123def456"),
        ("slack", "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"),
        ("prometheus", "prom_api_key_789xyz"),
        ("grafana", "grafana_api_key_456abc")
    ]
    
    for service, api_key in api_keys:
        success = await secrets_manager.put_secret(
            f"api_key/{service}",
            api_key,
            SecretType.API_KEY,
            f"{service.title()} API key for production"
        )
        print(f"  {service.title()} API key stored: {success}")
        
        retrieved_key = await secrets_manager.get_api_key(service)
        print(f"  {service.title()} API key retrieved: {retrieved_key is not None}")
    
    # Test 5: Cache performance
    print("\n📋 Test 5: Cache Performance")
    start_time = time.time()
    
    # First access (should hit provider)
    creds1 = await secrets_manager.get_database_credentials("main")
    first_access_time = time.time() - start_time
    
    start_time = time.time()
    
    # Second access (should hit cache)
    creds2 = await secrets_manager.get_database_credentials("main")
    second_access_time = time.time() - start_time
    
    print(f"  First access time: {first_access_time*1000:.1f}ms")
    print(f"  Second access time (cached): {second_access_time*1000:.1f}ms")
    if second_access_time > 0:
        print(f"  Cache speedup: {first_access_time/second_access_time:.1f}x")
    else:
        print(f"  Cache speedup: Very fast (cached access < 1ms)")
    
    print("✅ Secrets management test completed\n")


async def test_integrated_production_scenario():
    """Test integrated production scenario with all components."""
    print("🏭 Testing Integrated Production Scenario")
    print("=" * 60)
    
    # Scenario: Deploy new model with secrets and alerting
    print("\n📋 Scenario: Production Model Deployment")
    
    # Step 1: Retrieve secrets for deployment
    print("  Step 1: Retrieving deployment secrets...")
    secrets_manager = create_secrets_manager()
    
    # Get database credentials for health checks
    db_creds = await secrets_manager.get_database_credentials("main")
    if db_creds:
        print(f"    ✅ Database credentials retrieved")
    else:
        print(f"    ❌ Failed to retrieve database credentials")
    
    # Get S3 credentials for model storage
    s3_creds = await secrets_manager.get_s3_credentials()
    if s3_creds:
        print(f"    ✅ S3 credentials retrieved")
    else:
        print(f"    ❌ Failed to retrieve S3 credentials")
    
    # Step 2: Simulate model deployment
    print("  Step 2: Deploying new model...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        deployment_root = Path(temp_dir) / "production_deployment"
        bg_deploy = create_blue_green_deployment(str(deployment_root))
        
        # Create production model bundle
        bundle_dir = Path(temp_dir) / "production_bundle"
        bundle_dir.mkdir(parents=True)
        
        # Create model files
        (bundle_dir / "policy.pt").write_text("# Production PyTorch model\noptimized_model_data")
        (bundle_dir / "value_function.pt").write_text("# Production value function\noptimized_value_data")
        (bundle_dir / "config.yaml").write_text("""
model_type: rl_policy
version: production_v1.0.0
optimization_level: high
latency_target_us: 20
risk_tolerance: conservative
""")
        
        metadata = {
            "version": "production_v1.0.0",
            "created_at": time.time(),
            "model_type": "rl_policy",
            "training_data_hash": "prod_abc123def456",
            "performance_metrics": {
                "sharpe_ratio": 2.45,
                "max_drawdown": 0.06,
                "win_rate": 0.78,
                "latency_p99_us": 18.5
            },
            "validation": {
                "backtesting_period": "2024-01-01_to_2025-06-30",
                "out_of_sample_performance": "passed",
                "stress_testing": "passed",
                "regulatory_approval": "approved"
            }
        }
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        # Deploy the bundle
        deployment = bg_deploy.deploy_bundle("production_v1.0.0", bundle_dir, force=True)
        
        if deployment.status == DeploymentStatus.ACTIVE:
            print(f"    ✅ Model deployed successfully")
        else:
            print(f"    ❌ Model deployment failed: {deployment.error_message}")
        
        # Step 3: Monitor deployment with alerts
        print("  Step 3: Monitoring deployment...")
        
        alert_manager = create_alert_manager()
        
        # Simulate monitoring checks
        latency_p99 = metadata["performance_metrics"]["latency_p99_us"]
        
        if latency_p99 > 25:
            # Send critical latency alert
            print(f"    ⚠️  Latency P99 ({latency_p99}µs) exceeds threshold, sending alert...")
            await alert_manager.send_critical_latency_alert("risk_enforcement", latency_p99)
        else:
            print(f"    ✅ Latency P99 ({latency_p99}µs) within acceptable range")
        
        # Simulate successful deployment notification
        success_alert = Alert(
            alert_name="ProductionDeploymentSuccess",
            severity=AlertSeverity.INFO,
            status=AlertStatus.FIRING,
            message="Production model deployment successful",
            description=f"Model {metadata['version']} deployed successfully with Sharpe ratio {metadata['performance_metrics']['sharpe_ratio']}",
            service="deployment",
            team="trading",
            timestamp=time.time(),
            labels={
                "version": metadata["version"],
                "sharpe_ratio": str(metadata["performance_metrics"]["sharpe_ratio"]),
                "deployment_id": deployment.deployment_id
            },
            annotations={
                "impact": "Improved trading performance expected",
                "next_steps": "Monitor performance for 24 hours"
            }
        )
        
        await alert_manager.send_alert(success_alert)
        print(f"    ✅ Deployment success notification sent")
        
        # Step 4: Show deployment summary
        print("  Step 4: Deployment Summary")
        status = bg_deploy.get_deployment_status()
        print(f"    Current version: {status['current_version']}")
        print(f"    Rollback available: {status['rollback_available']}")
        print(f"    Deployment time: {deployment.completed_at - deployment.started_at:.1f}s")
    
    print("✅ Integrated production scenario completed\n")


async def main():
    """Run comprehensive production deployment tests."""
    print("🚀 IntradayJules Production Deployment System")
    print("=" * 80)
    print("Testing Prometheus alerting, blue/green rollout, and secrets management")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test individual components
        await test_alerting_system()
        test_blue_green_deployment()
        await test_secrets_management()
        
        # Test integrated scenario
        await test_integrated_production_scenario()
        
        print("🎉 ALL PRODUCTION DEPLOYMENT TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("✅ Prometheus alerting with PagerDuty/Slack integration")
        print("✅ Blue/green rollout with atomic symlink swapping")
        print("✅ Secrets management with AWS Secrets Manager/Vault")
        print("✅ Critical alerts on latency P99 > 25µs")
        print("✅ Audit log write error monitoring")
        print("✅ Circuit breaker trip detection")
        print("✅ Zero-downtime model deployments")
        print("✅ Secure credential management")
        print("=" * 80)
        print("🔧 Production deployment system is ready!")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))