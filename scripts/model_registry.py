#!/usr/bin/env python3
"""
Model Registry Management Script

Handles versioning, promotion, and comparison of trained models.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ModelRegistry:
    """Manages model versions and promotes models based on performance."""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(self, 
                      model_path: str, 
                      version: str, 
                      metadata: Dict,
                      performance: Dict,
                      description: str = "") -> bool:
        """Register a new model version."""
        
        version_dir = self.registry_path / version
        if version_dir.exists():
            print(f"❌ Version {version} already exists!")
            return False
        
        # Create version directory
        version_dir.mkdir(parents=True)
        model_dir = version_dir / "model"
        model_dir.mkdir()
        
        # Copy model files
        model_source = Path(model_path)
        if model_source.is_dir():
            shutil.copytree(model_source, model_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(model_source, model_dir)
        
        # Save metadata
        metadata['version'] = version
        metadata['registered_date'] = datetime.now().isoformat()
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save performance
        with open(version_dir / "performance.json", 'w') as f:
            json.dump(performance, f, indent=2)
        
        # Create README
        readme_content = f"""# Model {version}

## Description
{description}

## Performance Summary
- Average Return: {performance.get('training_performance', {}).get('average_return_pct', 'N/A')}%
- Win Rate: {performance.get('training_performance', {}).get('win_rate_pct', 'N/A')}%
- Episodes: {performance.get('training_performance', {}).get('total_episodes', 'N/A')}

## Registration Date
{metadata['registered_date']}
"""
        with open(version_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Model {version} registered successfully!")
        return True
    
    def list_models(self) -> List[str]:
        """List all registered model versions."""
        versions = []
        for item in self.registry_path.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                versions.append(item.name)
        return sorted(versions)
    
    def get_latest(self) -> Optional[str]:
        """Get the latest model version."""
        latest_file = self.registry_path / "LATEST.txt"
        if latest_file.exists():
            return latest_file.read_text().strip()
        return None
    
    def promote_model(self, version: str) -> bool:
        """Promote a model version to latest."""
        version_dir = self.registry_path / version
        if not version_dir.exists():
            print(f"❌ Version {version} not found!")
            return False
        
        # Update LATEST.txt
        latest_file = self.registry_path / "LATEST.txt"
        latest_file.write_text(version)
        
        print(f"✅ Model {version} promoted to latest!")
        return True
    
    def compare_models(self, version1: str, version2: str) -> Dict:
        """Compare performance between two model versions."""
        
        def load_performance(version: str) -> Dict:
            perf_file = self.registry_path / version / "performance.json"
            if perf_file.exists():
                with open(perf_file) as f:
                    return json.load(f)
            return {}
        
        perf1 = load_performance(version1)
        perf2 = load_performance(version2)
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics': {}
        }
        
        # Compare key metrics
        metrics = ['average_return_pct', 'win_rate_pct', 'average_trades_per_episode', 'average_turnover_ratio']
        
        for metric in metrics:
            val1 = perf1.get('training_performance', {}).get(metric, 0)
            val2 = perf2.get('training_performance', {}).get(metric, 0)
            
            comparison['metrics'][metric] = {
                version1: val1,
                version2: val2,
                'improvement': val2 - val1,
                'improvement_pct': ((val2 - val1) / abs(val1) * 100) if val1 != 0 else 0
            }
        
        return comparison
    
    def commit_model(self, version: str, message: str = None) -> bool:
        """Commit model to git with proper tagging."""
        if not message:
            message = f"Add model {version}"
        
        try:
            # Add files to git
            subprocess.run(['git', 'add', f'models/registry/{version}/'], check=True)
            subprocess.run(['git', 'add', 'models/registry/LATEST.txt'], check=True)
            
            # Commit
            subprocess.run(['git', 'commit', '-m', message], check=True)
            
            # Tag the commit
            subprocess.run(['git', 'tag', '-a', version, '-m', f'Model version {version}'], check=True)
            
            print(f"✅ Model {version} committed and tagged!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return False


def main():
    """CLI interface for model registry."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all registered models')
    
    # Latest command
    subparsers.add_parser('latest', help='Show latest model version')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote model to latest')
    promote_parser.add_argument('version', help='Version to promote')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('version1', help='First version')
    compare_parser.add_argument('version2', help='Second version')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Commit model to git')
    commit_parser.add_argument('version', help='Version to commit')
    commit_parser.add_argument('--message', '-m', help='Commit message')
    
    args = parser.parse_args()
    registry = ModelRegistry()
    
    if args.command == 'list':
        models = registry.list_models()
        print("📋 Registered Models:")
        for model in models:
            print(f"  - {model}")
    
    elif args.command == 'latest':
        latest = registry.get_latest()
        print(f"🏆 Latest Model: {latest}")
    
    elif args.command == 'promote':
        registry.promote_model(args.version)
    
    elif args.command == 'compare':
        comparison = registry.compare_models(args.version1, args.version2)
        print(f"📊 Comparison: {args.version1} vs {args.version2}")
        for metric, data in comparison['metrics'].items():
            print(f"  {metric}:")
            print(f"    {args.version1}: {data[args.version1]:.2f}")
            print(f"    {args.version2}: {data[args.version2]:.2f}")
            print(f"    Improvement: {data['improvement']:+.2f} ({data['improvement_pct']:+.1f}%)")
    
    elif args.command == 'commit':
        registry.commit_model(args.version, args.message)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()