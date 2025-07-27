# src/cli/model_cli.py
"""
Command Line Interface for Model Management.

This CLI provides production teams with easy access to model versioning,
deployment, and management capabilities. Teams can request specific model
versions using semantic identifiers instead of UUIDs.

Usage Examples:
    # List available models
    python -m src.cli.model_cli list --status validated --limit 10
    
    # Deploy specific version
    python -m src.cli.model_cli deploy v2025-07-06--18h51
    
    # Get current production model
    python -m src.cli.model_cli status
    
    # Rollback to previous version
    python -m src.cli.model_cli rollback
    
    # Compare model versions
    python -m src.cli.model_cli compare v2025-07-06--18h51 v2025-07-05--14h30
    
    # Generate model report
    python -m src.cli.model_cli report --experiment momentum_strategy
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import model management components
try:
    from src.training.experiment_registry import create_experiment_registry, ModelVersionInfo
    from src.deployment.model_deployment_service import create_deployment_service
    from src.training.enhanced_trainer_agent import create_enhanced_trainer_agent
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    COMPONENTS_AVAILABLE = False

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


class ModelCLI:
    """Command Line Interface for model management."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Model CLI.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.registry = None
        self.deployment_service = None
        self.trainer_agent = None
        
        if COMPONENTS_AVAILABLE:
            try:
                self.registry = create_experiment_registry(self.config.get('registry', {}))
                self.deployment_service = create_deployment_service(self.config.get('deployment', {}))
                self.logger.info("✅ Model CLI initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize components: {e}")
        else:
            self.logger.error("❌ Required components not available")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        default_config = {
            'registry': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                's3_bucket': os.getenv('MODEL_S3_BUCKET'),
                'enable_wandb': os.getenv('ENABLE_WANDB', 'false').lower() == 'true'
            },
            'deployment': {
                'deployment_dir': 'production/models',
                'health_check_interval': 30,
                'performance_sla_ms': 100.0
            },
            'cli': {
                'default_limit': 20,
                'date_format': '%Y-%m-%d %H:%M:%S',
                'output_format': 'table'  # table, json, csv
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                print(f"⚠️  Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for CLI."""
        logger = logging.getLogger('ModelCLI')
        
        # Set level based on environment
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def list_models(self, 
                   experiment: Optional[str] = None,
                   algorithm: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = None,
                   output_format: str = None) -> None:
        """
        List available model versions.
        
        Args:
            experiment: Filter by experiment name
            algorithm: Filter by algorithm
            status: Filter by status
            limit: Maximum number of results
            output_format: Output format (table, json, csv)
        """
        if not self.registry:
            print("❌ Registry not available")
            return
        
        if limit is None:
            limit = self.config['cli']['default_limit']
        
        if output_format is None:
            output_format = self.config['cli']['output_format']
        
        print(f"📋 Listing model versions (limit: {limit})")
        if experiment:
            print(f"   Experiment: {experiment}")
        if algorithm:
            print(f"   Algorithm: {algorithm}")
        if status:
            print(f"   Status: {status}")
        print()
        
        try:
            models = self.registry.list_model_versions(
                experiment_name=experiment,
                algorithm=algorithm,
                status=status,
                limit=limit
            )
            
            if not models:
                print("No models found matching criteria")
                return
            
            self._display_models(models, output_format)
            
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            self.logger.error(f"List models error: {e}")
    
    def get_model_info(self, version_id: str, output_format: str = None) -> None:
        """
        Get detailed information about a specific model version.
        
        Args:
            version_id: Model version ID
            output_format: Output format
        """
        if not self.registry:
            print("❌ Registry not available")
            return
        
        if output_format is None:
            output_format = self.config['cli']['output_format']
        
        print(f"🔍 Getting model information: {version_id}")
        print()
        
        try:
            model_info = self.registry.get_model_version(version_id)
            
            if not model_info:
                print(f"❌ Model version not found: {version_id}")
                return
            
            self._display_model_details(model_info, output_format)
            
        except Exception as e:
            print(f"❌ Failed to get model info: {e}")
            self.logger.error(f"Get model info error: {e}")
    
    def deploy_model(self, version_id: str, force: bool = False) -> None:
        """
        Deploy a model version to production.
        
        Args:
            version_id: Model version ID to deploy
            force: Force deployment even if validation fails
        """
        if not self.deployment_service:
            print("❌ Deployment service not available")
            return
        
        print(f"🚀 Deploying model version: {version_id}")
        if force:
            print("⚠️  Force deployment enabled")
        print()
        
        try:
            success = self.deployment_service.deploy_model_version(version_id, force=force)
            
            if success:
                print(f"✅ Model deployed successfully: {version_id}")
                
                # Show deployment status
                status = self.deployment_service.get_deployment_status()
                print(f"📊 Deployment Status:")
                print(f"   Current Version: {status['current_version']}")
                print(f"   Status: {status['status']}")
                print(f"   Health: {status['health_status']}")
                
            else:
                print(f"❌ Deployment failed: {version_id}")
                
        except Exception as e:
            print(f"❌ Deployment error: {e}")
            self.logger.error(f"Deploy model error: {e}")
    
    def get_deployment_status(self, output_format: str = None) -> None:
        """
        Get current deployment status.
        
        Args:
            output_format: Output format
        """
        if not self.deployment_service:
            print("❌ Deployment service not available")
            return
        
        if output_format is None:
            output_format = self.config['cli']['output_format']
        
        print("📊 Current Deployment Status")
        print()
        
        try:
            status = self.deployment_service.get_deployment_status()
            
            if output_format == 'json':
                print(json.dumps(status, indent=2, default=str))
            else:
                self._display_deployment_status(status)
                
        except Exception as e:
            print(f"❌ Failed to get deployment status: {e}")
            self.logger.error(f"Get deployment status error: {e}")
    
    def rollback_deployment(self) -> None:
        """Rollback to previous model version."""
        if not self.deployment_service:
            print("❌ Deployment service not available")
            return
        
        print("🔄 Rolling back to previous model version")
        print()
        
        try:
            success = self.deployment_service.rollback_to_previous_version()
            
            if success:
                print("✅ Rollback successful")
                
                # Show new status
                status = self.deployment_service.get_deployment_status()
                print(f"📊 Current Version: {status['current_version']}")
                
            else:
                print("❌ Rollback failed")
                
        except Exception as e:
            print(f"❌ Rollback error: {e}")
            self.logger.error(f"Rollback error: {e}")
    
    def compare_models(self, version_ids: List[str], output_format: str = None) -> None:
        """
        Compare multiple model versions.
        
        Args:
            version_ids: List of version IDs to compare
            output_format: Output format
        """
        if not self.registry:
            print("❌ Registry not available")
            return
        
        if len(version_ids) < 2:
            print("❌ Need at least 2 model versions for comparison")
            return
        
        if output_format is None:
            output_format = self.config['cli']['output_format']
        
        print(f"📊 Comparing model versions: {', '.join(version_ids)}")
        print()
        
        try:
            comparison = self.registry.compare_models(version_ids)
            
            if "error" in comparison:
                print(f"❌ Comparison failed: {comparison['error']}")
                return
            
            self._display_model_comparison(comparison, output_format)
            
        except Exception as e:
            print(f"❌ Comparison error: {e}")
            self.logger.error(f"Compare models error: {e}")
    
    def generate_report(self, 
                       experiment: Optional[str] = None,
                       output_file: Optional[str] = None,
                       output_format: str = None) -> None:
        """
        Generate comprehensive model report.
        
        Args:
            experiment: Filter by experiment name
            output_file: Optional output file path
            output_format: Output format
        """
        if not self.registry:
            print("❌ Registry not available")
            return
        
        if output_format is None:
            output_format = self.config['cli']['output_format']
        
        print("📋 Generating model report")
        if experiment:
            print(f"   Experiment: {experiment}")
        print()
        
        try:
            # Get enhanced trainer for report generation
            if not self.trainer_agent:
                self.trainer_agent = create_enhanced_trainer_agent({
                    'experiment_registry': self.config['registry']
                })
            
            report = self.trainer_agent.generate_model_report(experiment)
            
            if "error" in report:
                print(f"❌ Report generation failed: {report['error']}")
                return
            
            # Display or save report
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"✅ Report saved to: {output_file}")
            else:
                self._display_model_report(report, output_format)
                
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            self.logger.error(f"Generate report error: {e}")
    
    def validate_model(self, version_id: str) -> None:
        """
        Validate a specific model version.
        
        Args:
            version_id: Model version ID to validate
        """
        if not self.trainer_agent:
            self.trainer_agent = create_enhanced_trainer_agent({
                'experiment_registry': self.config['registry']
            })
        
        print(f"🧪 Validating model version: {version_id}")
        print()
        
        try:
            validation_results = self.trainer_agent.validate_model_version(version_id)
            
            if "error" in validation_results:
                print(f"❌ Validation failed: {validation_results['error']}")
                return
            
            # Display validation results
            if validation_results.get('passed', False):
                print("✅ Model validation: PASSED")
            else:
                print("❌ Model validation: FAILED")
            
            print("\n📋 Validation Details:")
            for check, result in validation_results.get('checks', {}).items():
                status = "✅" if result else "❌"
                print(f"   {status} {check}: {result}")
            
            if validation_results.get('errors'):
                print("\n⚠️  Validation Errors:")
                for error in validation_results['errors']:
                    print(f"   - {error}")
                    
        except Exception as e:
            print(f"❌ Validation error: {e}")
            self.logger.error(f"Validate model error: {e}")
    
    def cleanup_old_models(self, retention_days: int = 90, dry_run: bool = False) -> None:
        """
        Clean up old model versions.
        
        Args:
            retention_days: Days to retain models
            dry_run: Show what would be cleaned up without actually doing it
        """
        if not self.registry:
            print("❌ Registry not available")
            return
        
        print(f"🧹 Cleaning up models older than {retention_days} days")
        if dry_run:
            print("   (DRY RUN - no changes will be made)")
        print()
        
        try:
            if dry_run:
                # Show what would be cleaned up
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                all_models = self.registry.list_model_versions(limit=1000)
                
                cleanup_candidates = [
                    model for model in all_models
                    if model.status != "production" and model.created_at < cutoff_date
                ]
                
                print(f"📊 Found {len(cleanup_candidates)} models for cleanup:")
                for model in cleanup_candidates[:10]:  # Show first 10
                    print(f"   - {model.version_id} ({model.algorithm}) - {model.created_at.strftime('%Y-%m-%d')}")
                
                if len(cleanup_candidates) > 10:
                    print(f"   ... and {len(cleanup_candidates) - 10} more")
                    
            else:
                cleanup_count = self.registry.cleanup_old_versions(retention_days)
                print(f"✅ Cleaned up {cleanup_count} old model versions")
                
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            self.logger.error(f"Cleanup error: {e}")
    
    def _display_models(self, models: List[ModelVersionInfo], output_format: str) -> None:
        """Display list of models in specified format."""
        if output_format == 'json':
            model_data = [
                {
                    'version_id': model.version_id,
                    'algorithm': model.algorithm,
                    'status': model.status,
                    'created_at': model.created_at.isoformat(),
                    'metrics': model.metrics
                }
                for model in models
            ]
            print(json.dumps(model_data, indent=2))
            
        elif output_format == 'csv' and PANDAS_AVAILABLE:
            df = pd.DataFrame([
                {
                    'Version ID': model.version_id,
                    'Algorithm': model.algorithm,
                    'Status': model.status,
                    'Created At': model.created_at.strftime(self.config['cli']['date_format']),
                    'Metrics': len(model.metrics)
                }
                for model in models
            ])
            print(df.to_csv(index=False))
            
        else:  # table format
            if TABULATE_AVAILABLE:
                table_data = [
                    [
                        model.version_id,
                        model.algorithm,
                        model.status,
                        model.created_at.strftime('%Y-%m-%d %H:%M'),
                        len(model.metrics)
                    ]
                    for model in models
                ]
                
                headers = ['Version ID', 'Algorithm', 'Status', 'Created At', 'Metrics']
                print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))
            else:
                # Simple text format
                for model in models:
                    print(f"📦 {model.version_id}")
                    print(f"   Algorithm: {model.algorithm}")
                    print(f"   Status: {model.status}")
                    print(f"   Created: {model.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   Metrics: {len(model.metrics)}")
                    print()
    
    def _display_model_details(self, model: ModelVersionInfo, output_format: str) -> None:
        """Display detailed model information."""
        if output_format == 'json':
            model_data = {
                'version_id': model.version_id,
                'algorithm': model.algorithm,
                'status': model.status,
                'created_at': model.created_at.isoformat(),
                'tags': model.tags,
                'metrics': model.metrics,
                'parameters': model.parameters
            }
            print(json.dumps(model_data, indent=2))
        else:
            print(f"📦 Model Version: {model.version_id}")
            print(f"🔬 Algorithm: {model.algorithm}")
            print(f"📊 Status: {model.status}")
            print(f"📅 Created: {model.created_at.strftime(self.config['cli']['date_format'])}")
            print(f"🏷️  Run ID: {model.run_id}")
            print()
            
            if model.tags:
                print("🏷️  Tags:")
                for key, value in model.tags.items():
                    print(f"   {key}: {value}")
                print()
            
            if model.metrics:
                print("📈 Metrics:")
                for key, value in model.metrics.items():
                    print(f"   {key}: {value}")
                print()
            
            if model.parameters:
                print("⚙️  Parameters:")
                for key, value in model.parameters.items():
                    print(f"   {key}: {value}")
                print()
    
    def _display_deployment_status(self, status: Dict[str, Any]) -> None:
        """Display deployment status."""
        print(f"📊 Current Version: {status.get('current_version', 'None')}")
        print(f"📊 Previous Version: {status.get('previous_version', 'None')}")
        print(f"📊 Status: {status.get('status', 'Unknown')}")
        print(f"📊 Health: {status.get('health_status', 'Unknown')}")
        
        if status.get('deployment_time'):
            print(f"📊 Deployed At: {status['deployment_time']}")
        
        if status.get('canary_version'):
            print(f"🧪 Canary Version: {status['canary_version']} ({status.get('canary_traffic_percent', 0)}% traffic)")
        
        # Show model health if available
        if 'current_model_health' in status:
            health = status['current_model_health']
            metrics = health.get('metrics', {})
            
            print("\n📈 Current Model Performance:")
            print(f"   Total Predictions: {metrics.get('total_predictions', 0)}")
            print(f"   Success Rate: {(1 - metrics.get('error_rate', 0)) * 100:.1f}%")
            print(f"   Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
            
            if metrics.get('last_prediction_time'):
                print(f"   Last Prediction: {metrics['last_prediction_time']}")
    
    def _display_model_comparison(self, comparison: Dict[str, Any], output_format: str) -> None:
        """Display model comparison results."""
        if output_format == 'json':
            print(json.dumps(comparison, indent=2))
        else:
            print("📊 Model Comparison Results")
            print(f"Models: {', '.join(comparison['models'])}")
            print()
            
            # Metrics comparison
            if comparison.get('metrics_comparison'):
                print("📈 Metrics Comparison:")
                for metric, values in comparison['metrics_comparison'].items():
                    print(f"   {metric}:")
                    for version_id, value in values.items():
                        if value is not None:
                            print(f"     {version_id}: {value:.4f}")
                        else:
                            print(f"     {version_id}: N/A")
                print()
            
            # Parameters comparison
            if comparison.get('parameters_comparison'):
                print("⚙️  Parameters Comparison:")
                for param, values in comparison['parameters_comparison'].items():
                    print(f"   {param}:")
                    for version_id, value in values.items():
                        print(f"     {version_id}: {value}")
                print()
    
    def _display_model_report(self, report: Dict[str, Any], output_format: str) -> None:
        """Display model report."""
        if output_format == 'json':
            print(json.dumps(report, indent=2))
        else:
            print("📋 Model Report")
            print(f"Generated: {report.get('generated_at', 'Unknown')}")
            if report.get('experiment_name'):
                print(f"Experiment: {report['experiment_name']}")
            print()
            
            print(f"📊 Total Models: {report.get('total_models', 0)}")
            print()
            
            # Status breakdown
            if report.get('status_breakdown'):
                print("📈 Status Breakdown:")
                for status, count in report['status_breakdown'].items():
                    print(f"   {status}: {count}")
                print()
            
            # Algorithm breakdown
            if report.get('algorithm_breakdown'):
                print("🔬 Algorithm Breakdown:")
                for algorithm, count in report['algorithm_breakdown'].items():
                    print(f"   {algorithm}: {count}")
                print()
            
            # Current production
            prod_info = report.get('current_production', {})
            if prod_info.get('version_id'):
                print("🚀 Current Production:")
                print(f"   Version: {prod_info['version_id']}")
                print(f"   Algorithm: {prod_info['algorithm']}")
                print(f"   Deployed: {prod_info['created_at']}")
                print()
            
            # Recent models
            if report.get('recent_models'):
                print("📦 Recent Models:")
                for model in report['recent_models'][:5]:
                    print(f"   {model['version_id']} ({model['algorithm']}) - {model['status']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list --status validated --limit 10
  %(prog)s deploy v2025-07-06--18h51
  %(prog)s status
  %(prog)s rollback
  %(prog)s compare v2025-07-06--18h51 v2025-07-05--14h30
  %(prog)s report --experiment momentum_strategy
        """
    )
    
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-format', choices=['table', 'json', 'csv'], 
                       help='Output format (default: table)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List model versions')
    list_parser.add_argument('--experiment', help='Filter by experiment name')
    list_parser.add_argument('--algorithm', help='Filter by algorithm')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--limit', type=int, help='Maximum number of results')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get model information')
    info_parser.add_argument('version_id', help='Model version ID')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model version')
    deploy_parser.add_argument('version_id', help='Model version ID to deploy')
    deploy_parser.add_argument('--force', action='store_true', 
                              help='Force deployment even if validation fails')
    
    # Status command
    subparsers.add_parser('status', help='Get deployment status')
    
    # Rollback command
    subparsers.add_parser('rollback', help='Rollback to previous version')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('version_ids', nargs='+', help='Version IDs to compare')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate model report')
    report_parser.add_argument('--experiment', help='Filter by experiment name')
    report_parser.add_argument('--output-file', help='Save report to file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model version')
    validate_parser.add_argument('version_id', help='Model version ID to validate')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old models')
    cleanup_parser.add_argument('--retention-days', type=int, default=90,
                               help='Days to retain models (default: 90)')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be cleaned up without doing it')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    try:
        cli = ModelCLI(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize CLI: {e}")
        return
    
    # Execute command
    try:
        if args.command == 'list':
            cli.list_models(
                experiment=args.experiment,
                algorithm=args.algorithm,
                status=args.status,
                limit=args.limit,
                output_format=args.output_format
            )
        
        elif args.command == 'info':
            cli.get_model_info(args.version_id, args.output_format)
        
        elif args.command == 'deploy':
            cli.deploy_model(args.version_id, args.force)
        
        elif args.command == 'status':
            cli.get_deployment_status(args.output_format)
        
        elif args.command == 'rollback':
            cli.rollback_deployment()
        
        elif args.command == 'compare':
            cli.compare_models(args.version_ids, args.output_format)
        
        elif args.command == 'report':
            cli.generate_report(
                experiment=args.experiment,
                output_file=args.output_file,
                output_format=args.output_format
            )
        
        elif args.command == 'validate':
            cli.validate_model(args.version_id)
        
        elif args.command == 'cleanup':
            cli.cleanup_old_models(args.retention_days, args.dry_run)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Command failed: {e}")
        logging.getLogger('ModelCLI').error(f"Command error: {e}")


if __name__ == '__main__':
    main()