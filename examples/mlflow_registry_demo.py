# examples/mlflow_registry_demo.py
"""
MLflow Experiment Registry Demo

This script demonstrates the complete workflow of the new MLflow-based
experiment registry system, showing how production teams can now request
models by semantic version instead of UUIDs.

Usage:
    python examples/mlflow_registry_demo.py
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_model_files(temp_dir: Path) -> tuple:
    """Create sample model files for demonstration."""
    import torch
    import torch.nn as nn
    
    # Create a simple model
    class SimplePolicy(nn.Module):
        def __init__(self, input_size=10, output_size=3):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Create and save model
    model = SimplePolicy()
    model.eval()
    
    # Convert to TorchScript
    sample_input = torch.randn(1, 10)
    scripted_model = torch.jit.trace(model, sample_input)
    
    # Save model file
    model_path = temp_dir / "policy.pt"
    scripted_model.save(str(model_path))
    
    # Create metadata
    metadata = {
        "algo": "DQN",
        "framework": "torchscript",
        "export_method": "trace",
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "input_shape": [10],
        "output_shape": [3],
        "model_size_bytes": model_path.stat().st_size
    }
    
    metadata_path = temp_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Created sample model files in {temp_dir}")
    return str(model_path), str(metadata_path)

def demo_experiment_registry():
    """Demonstrate experiment registry functionality."""
    print("🔬 MLflow Experiment Registry Demo")
    print("=" * 50)
    
    try:
        # Import registry components
        from src.training.experiment_registry import create_experiment_registry
        from src.deployment.model_deployment_service import create_deployment_service
        
        # Create temporary directory for demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Setup demo configuration
            config = {
                'tracking_uri': f'sqlite:///{temp_path}/demo_mlflow.db',
                's3_bucket': None,  # Use local storage for demo
                'local_cache_dir': str(temp_path / 'cache'),
                'enable_wandb': False,
                'auto_validate': True
            }
            
            print(f"📁 Demo directory: {temp_path}")
            print()
            
            # 1. Initialize Experiment Registry
            print("1️⃣  Initializing Experiment Registry...")
            registry = create_experiment_registry(config)
            print("✅ Registry initialized")
            print()
            
            # 2. Create Sample Model Files
            print("2️⃣  Creating sample model files...")
            model_path, metadata_path = create_sample_model_files(temp_path)
            print("✅ Sample model created")
            print()
            
            # 3. Register Model with Semantic Versioning
            print("3️⃣  Registering model in experiment registry...")
            model_version_info = registry.register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                experiment_name="demo_experiment",
                tags={
                    "environment": "demo",
                    "strategy": "momentum",
                    "demo": "true"
                },
                metrics={
                    "mean_reward": 150.5,
                    "episode_length": 200.0,
                    "success_rate": 0.85
                }
            )
            
            if model_version_info:
                print(f"✅ Model registered with semantic version: {model_version_info.version_id}")
                print(f"   Status: {model_version_info.status}")
                print(f"   Algorithm: {model_version_info.algorithm}")
                print(f"   Run ID: {model_version_info.run_id}")
            else:
                print("❌ Model registration failed")
                return
            print()
            
            # 4. List Available Models
            print("4️⃣  Listing available model versions...")
            models = registry.list_model_versions(experiment_name="demo_experiment", limit=5)
            print(f"📋 Found {len(models)} model versions:")
            for model in models:
                print(f"   - {model.version_id} ({model.algorithm}) - {model.status}")
            print()
            
            # 5. Retrieve Model by Semantic Version
            print("5️⃣  Retrieving model by semantic version...")
            retrieved_model = registry.get_model_version(model_version_info.version_id)
            if retrieved_model:
                print(f"✅ Retrieved model: {retrieved_model.version_id}")
                print(f"   Created: {retrieved_model.created_at}")
                print(f"   Metrics: {len(retrieved_model.metrics)} metrics")
                print(f"   Tags: {len(retrieved_model.tags)} tags")
            else:
                print("❌ Failed to retrieve model")
            print()
            
            # 6. Download Model for Deployment
            print("6️⃣  Downloading model for deployment...")
            download_path = registry.download_model(
                model_version_info.version_id,
                str(temp_path / "deployment")
            )
            if download_path:
                print(f"✅ Model downloaded to: {download_path}")
                
                # Verify files
                policy_file = Path(download_path) / "policy.pt"
                metadata_file = Path(download_path) / "metadata.json"
                
                if policy_file.exists() and metadata_file.exists():
                    print(f"   📄 policy.pt: {policy_file.stat().st_size} bytes")
                    print(f"   📄 metadata.json: {metadata_file.stat().st_size} bytes")
                else:
                    print("⚠️  Some files missing in download")
            else:
                print("❌ Failed to download model")
            print()
            
            # 7. Model Comparison (create second model for comparison)
            print("7️⃣  Creating second model for comparison...")
            model_path_2, metadata_path_2 = create_sample_model_files(temp_path / "model2")
            
            # Modify metadata for second model
            with open(metadata_path_2, 'r') as f:
                metadata_2 = json.load(f)
            metadata_2['algo'] = 'PPO'  # Different algorithm
            with open(metadata_path_2, 'w') as f:
                json.dump(metadata_2, f, indent=2)
            
            model_version_info_2 = registry.register_model(
                model_path=model_path_2,
                metadata_path=metadata_path_2,
                experiment_name="demo_experiment",
                tags={"environment": "demo", "strategy": "mean_reversion"},
                metrics={"mean_reward": 145.2, "episode_length": 180.0, "success_rate": 0.82}
            )
            
            if model_version_info_2:
                print(f"✅ Second model registered: {model_version_info_2.version_id}")
                
                # Compare models
                comparison = registry.compare_models([
                    model_version_info.version_id,
                    model_version_info_2.version_id
                ])
                
                print("📊 Model Comparison:")
                for metric, values in comparison.get('metrics_comparison', {}).items():
                    print(f"   {metric}:")
                    for version_id, value in values.items():
                        if value is not None:
                            print(f"     {version_id}: {value}")
            print()
            
            # 8. Demonstrate Production Deployment Service
            print("8️⃣  Demonstrating production deployment service...")
            deployment_config = {
                'registry_config': config,
                'deployment_dir': str(temp_path / 'production'),
                'health_check_interval': 0,  # Disable for demo
                'validation_required': True
            }
            
            deployment_service = create_deployment_service(deployment_config)
            print("✅ Deployment service initialized")
            
            # List available versions for deployment
            available_versions = deployment_service.list_available_versions(limit=5)
            print(f"📋 Available for deployment: {len(available_versions)} versions")
            
            # Deploy the first model
            if available_versions:
                version_to_deploy = available_versions[0]['version_id']
                print(f"🚀 Deploying version: {version_to_deploy}")
                
                success = deployment_service.deploy_model_version(version_to_deploy)
                if success:
                    print("✅ Deployment successful")
                    
                    # Get deployment status
                    status = deployment_service.get_deployment_status()
                    print(f"📊 Current version: {status['current_version']}")
                    print(f"📊 Status: {status['status']}")
                    print(f"📊 Health: {status['health_status']}")
                    
                    # Test prediction
                    current_model = deployment_service.get_current_model()
                    if current_model:
                        import numpy as np
                        sample_obs = np.random.randn(10)
                        prediction = current_model.predict(sample_obs)
                        
                        if prediction is not None:
                            print(f"✅ Test prediction successful: shape {prediction.shape}")
                        else:
                            print("⚠️  Test prediction failed")
                else:
                    print("❌ Deployment failed")
            print()
            
            # 9. Demonstrate CLI-like Operations
            print("9️⃣  Demonstrating CLI-like operations...")
            
            # Promote to production
            if model_version_info.status == "validated":
                success = registry.promote_to_production(
                    model_version_info.version_id,
                    {"deployed_by": "demo_script", "deployment_time": datetime.now().isoformat()}
                )
                if success:
                    print(f"✅ Promoted to production: {model_version_info.version_id}")
                else:
                    print("❌ Promotion failed")
            
            # Generate model report
            print("📋 Generating model report...")
            
            # Get all models for report
            all_models = registry.list_model_versions(limit=100)
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_models": len(all_models),
                "status_breakdown": {},
                "algorithm_breakdown": {}
            }
            
            for model in all_models:
                # Count by status
                status = model.status
                report["status_breakdown"][status] = report["status_breakdown"].get(status, 0) + 1
                
                # Count by algorithm
                algo = model.algorithm
                report["algorithm_breakdown"][algo] = report["algorithm_breakdown"].get(algo, 0) + 1
            
            print("📊 Model Report Summary:")
            print(f"   Total models: {report['total_models']}")
            print(f"   Status breakdown: {report['status_breakdown']}")
            print(f"   Algorithm breakdown: {report['algorithm_breakdown']}")
            print()
            
            # 10. Cleanup Demo
            print("🔟 Demo cleanup...")
            deployment_service.stop_health_monitoring()
            print("✅ Demo completed successfully!")
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages:")
        print("  pip install mlflow torch")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    print("\n🖥️  CLI Usage Examples")
    print("=" * 30)
    
    cli_examples = [
        "# List available models",
        "python -m src.cli.model_cli list --status validated --limit 10",
        "",
        "# Get model information",
        "python -m src.cli.model_cli info v2025-07-06--18h51",
        "",
        "# Deploy specific version",
        "python -m src.cli.model_cli deploy v2025-07-06--18h51",
        "",
        "# Check deployment status",
        "python -m src.cli.model_cli status",
        "",
        "# Rollback to previous version",
        "python -m src.cli.model_cli rollback",
        "",
        "# Compare model versions",
        "python -m src.cli.model_cli compare v2025-07-06--18h51 v2025-07-05--14h30",
        "",
        "# Generate model report",
        "python -m src.cli.model_cli report --experiment momentum_strategy",
        "",
        "# Validate model",
        "python -m src.cli.model_cli validate v2025-07-06--18h51",
        "",
        "# Clean up old models",
        "python -m src.cli.model_cli cleanup --retention-days 90 --dry-run"
    ]
    
    for example in cli_examples:
        if example.startswith("#"):
            print(f"\n{example}")
        elif example == "":
            continue
        else:
            print(f"  {example}")

def main():
    """Main demo function."""
    print("🚀 MLflow Experiment Registry Demo")
    print("Solving: TrainerAgent exports policy.pt + metadata.json bundles")
    print("Problem: No experiment registry – you zip bundles to S3 with UUID path")
    print("Solution: MLflow registry with semantic versioning (v2025-07-06--18h51)")
    print()
    
    # Run the main demo
    demo_experiment_registry()
    
    # Show CLI examples
    demo_cli_usage()
    
    print("\n🎯 Key Benefits Achieved:")
    print("✅ Semantic versioning instead of UUIDs")
    print("✅ MLflow experiment tracking and model registry")
    print("✅ Production deployment with health monitoring")
    print("✅ CLI tools for operations teams")
    print("✅ Model comparison and analytics")
    print("✅ Automated validation and lifecycle management")
    print()
    print("🏆 Production teams can now request: 'give me policy v2025-07-06--18h51'")

if __name__ == "__main__":
    main()