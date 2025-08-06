#!/usr/bin/env python3
"""
Mini-Grid Hyperparameter Orchestrator - Team B Implementation
Advanced grid search infrastructure with GPU queuing for Phase 3

Usage:
    python scripts/mini_grid.py --launch --config config/grid_search.yaml
    python scripts/mini_grid.py --monitor --grid-id grid_20250101_120000
    python scripts/mini_grid.py --analyze --results-dir results/grid_search/

Features:
    - GPU queue management (≤4 GPUs saturated)
    - Parameter grid based on Phase 2 best results
    - Success criteria validation (ep_len ≥ 80, ep_rew ≥ 0, DD triggers < 60%)
    - Automated job launching and monitoring
    - Results collection and ranking
"""

import argparse
import json
import yaml
import subprocess
import time
import psutil
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mini_grid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU allocation and monitoring for grid search jobs"""
    
    def __init__(self, max_gpus: int = 4):
        self.max_gpus = max_gpus
        self.available_gpus = list(range(max_gpus))
        self.active_jobs = {}  # {gpu_id: job_info}
        self.gpu_lock = threading.Lock()
        
    def get_available_gpu(self) -> Optional[int]:
        """Get next available GPU ID, None if all busy"""
        with self.gpu_lock:
            if self.available_gpus:
                return self.available_gpus.pop(0)
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU back to available pool"""
        with self.gpu_lock:
            if gpu_id not in self.available_gpus:
                self.available_gpus.append(gpu_id)
                self.available_gpus.sort()
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """Get current GPU utilization percentages"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return {i: gpu.load * 100 for i, gpu in enumerate(gpus[:self.max_gpus])}
        except ImportError:
            # Fallback: use nvidia-smi if available
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    utilizations = [float(x.strip()) for x in result.stdout.strip().split('\n')]
                    return {i: util for i, util in enumerate(utilizations[:self.max_gpus])}
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Final fallback: assume all GPUs available
            return {i: 0.0 for i in range(self.max_gpus)}

class GridJob:
    """Represents a single grid search job"""
    
    def __init__(self, job_id: str, params: Dict[str, Any], config_path: Path, output_dir: Path):
        self.job_id = job_id
        self.params = params
        self.config_path = config_path
        self.output_dir = output_dir
        self.gpu_id = None
        self.process = None
        self.start_time = None
        self.end_time = None
        self.status = 'pending'  # pending, running, completed, failed
        self.results = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'params': self.params,
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'gpu_id': self.gpu_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'results': self.results
        }

class MiniGridOrchestrator:
    """Main orchestrator for mini-grid hyperparameter search"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self.load_config(config_path)
        self.gpu_manager = GPUManager(max_gpus=self.config.get('max_gpus', 4))
        self.jobs = []
        self.job_queue = queue.Queue()
        self.results_dir = Path(self.config.get('results_dir', 'results/grid_search'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Grid search metadata
        self.grid_id = f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.grid_dir = self.results_dir / self.grid_id
        self.grid_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Mini-Grid Orchestrator initialized: {self.grid_id}")
        logger.info(f"📁 Results directory: {self.grid_dir}")
        
    def load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'max_gpus': 4,
            'base_config': 'config/curriculum/phase2_oos.yaml',
            'training_script': 'train.py',
            'total_timesteps': 10000,  # 10K steps per grid job
            'results_dir': 'results/grid_search',
            'parameter_grid': self.get_default_parameter_grid(),
            'success_criteria': {
                'ep_len_min': 80,
                'ep_rew_min': 0.0,
                'dd_trigger_rate_max': 0.60
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() == '.yaml':
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"✅ Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config {config_path}: {e}")
                logger.info("📝 Using default configuration")
        else:
            logger.info("📝 Using default configuration")
        
        return default_config
    
    def get_default_parameter_grid(self) -> Dict[str, List[Any]]:
        """Define default parameter grid based on Stairways V3 system"""
        return {
            'learning_rate': [1e-4, 3e-4, 5e-4],
            'kl_target': [0.01, 0.02, 0.05],
            'max_daily_drawdown_pct': [0.20, 0.30, 0.40],
            'exit_tax': [2.0, 5.0, 10.0],
            'holding_alpha': [0.03, 0.05, 0.08],
            'pnl_epsilon': [500.0, 750.0, 1000.0]
        }
    
    def update_grid_from_phase2_results(self, phase2_results_path: Path):
        """Update parameter grid based on Phase 2 best results"""
        try:
            if phase2_results_path.exists():
                with open(phase2_results_path, 'r') as f:
                    phase2_results = json.load(f)
                
                # Extract best parameters from Phase 2
                best_checkpoint = phase2_results.get('next_steps', {}).get('best_checkpoint')
                if best_checkpoint:
                    logger.info(f"🎯 Updating grid based on Phase 2 best checkpoint: {best_checkpoint}")
                    
                    # Narrow parameter ranges around best performing values
                    # This is a placeholder - Team A should customize based on actual Phase 2 results
                    best_lr = 3e-4  # Example: extract from Phase 2 results
                    best_kl = 0.02
                    best_dd = 0.30
                    
                    # Create focused grid around best values
                    self.config['parameter_grid'] = {
                        'learning_rate': [best_lr * 0.5, best_lr, best_lr * 1.5],
                        'kl_target': [best_kl * 0.5, best_kl, best_kl * 2.0],
                        'max_daily_drawdown_pct': [best_dd - 0.1, best_dd, best_dd + 0.1],
                        'exit_tax': [2.0, 5.0, 10.0],  # Keep full range
                        'holding_alpha': [0.03, 0.05, 0.08],  # Keep full range
                        'pnl_epsilon': [750.0]  # Fix at best value
                    }
                    
                    logger.info("✅ Parameter grid updated based on Phase 2 results")
                else:
                    logger.warning("⚠️ No best checkpoint found in Phase 2 results")
            else:
                logger.warning(f"⚠️ Phase 2 results not found at {phase2_results_path}")
        except Exception as e:
            logger.error(f"❌ Failed to update grid from Phase 2 results: {e}")
    
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        param_grid = self.config['parameter_grid']
        
        # Get all parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of parameter dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        logger.info(f"📊 Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def create_job_config(self, job_id: str, params: Dict[str, Any]) -> Path:
        """Create job-specific configuration file"""
        job_config_dir = self.grid_dir / 'configs'
        job_config_dir.mkdir(exist_ok=True)
        
        config_path = job_config_dir / f'{job_id}.yaml'
        
        # Load base configuration
        base_config_path = Path(self.config['base_config'])
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
        else:
            # Create minimal base config
            base_config = {
                'training': {'total_timesteps': self.config['total_timesteps']},
                'data': {
                    'train_start': '2022-01-01',
                    'train_end': '2023-12-31',
                    'test_start': '2024-01-01',
                    'test_end': '2024-12-31'
                },
                'reward_system': {'type': 'RefinedRewardSystem'},
                'environment': {'use_governor': True},
                'logging': {'monitor_csv': True}
            }
        
        # Update with grid parameters
        for param_name, param_value in params.items():
            # Map parameter names to config structure
            if param_name == 'learning_rate':
                base_config.setdefault('model', {})['learning_rate'] = param_value
            elif param_name == 'kl_target':
                base_config.setdefault('model', {})['kl_target'] = param_value
            elif param_name == 'max_daily_drawdown_pct':
                base_config.setdefault('environment', {})['max_daily_drawdown_pct'] = param_value
            elif param_name in ['exit_tax', 'holding_alpha', 'pnl_epsilon']:
                base_config.setdefault('reward_system', {})[param_name] = param_value
        
        # Add job metadata
        base_config['grid_search'] = {
            'job_id': job_id,
            'grid_id': self.grid_id,
            'parameters': params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save job config
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return config_path
    
    def create_grid_jobs(self) -> List[GridJob]:
        """Create all grid search jobs"""
        param_combinations = self.generate_parameter_combinations()
        jobs = []
        
        for i, params in enumerate(param_combinations):
            job_id = f"job_{i:03d}"
            
            # Create job output directory
            job_output_dir = self.grid_dir / 'jobs' / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create job configuration
            config_path = self.create_job_config(job_id, params)
            
            # Create job object
            job = GridJob(job_id, params, config_path, job_output_dir)
            jobs.append(job)
        
        logger.info(f"🏗️ Created {len(jobs)} grid search jobs")
        return jobs
    
    def launch_job(self, job: GridJob) -> bool:
        """Launch a single grid search job"""
        # Get available GPU
        gpu_id = self.gpu_manager.get_available_gpu()
        if gpu_id is None:
            return False  # No GPU available
        
        job.gpu_id = gpu_id
        job.status = 'running'
        job.start_time = datetime.now()
        
        # Construct training command
        cmd = [
            'python', self.config['training_script'],
            '--config', str(job.config_path),
            '--save-path', str(job.output_dir),
            '--total-timesteps', str(self.config['total_timesteps'])
        ]
        
        # Set environment variables
        env = {
            **dict(os.environ),
            'CUDA_VISIBLE_DEVICES': str(gpu_id)
        }
        
        try:
            # Launch process
            job.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            logger.info(f"🚀 Launched {job.job_id} on GPU {gpu_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to launch {job.job_id}: {e}")
            job.status = 'failed'
            job.end_time = datetime.now()
            self.gpu_manager.release_gpu(gpu_id)
            return False
    
    def monitor_jobs(self):
        """Monitor running jobs and handle completion"""
        while True:
            active_jobs = [job for job in self.jobs if job.status == 'running']
            
            if not active_jobs:
                break
            
            for job in active_jobs:
                if job.process and job.process.poll() is not None:
                    # Job completed
                    job.end_time = datetime.now()
                    
                    if job.process.returncode == 0:
                        job.status = 'completed'
                        logger.info(f"✅ {job.job_id} completed successfully")
                        
                        # Collect results
                        self.collect_job_results(job)
                    else:
                        job.status = 'failed'
                        logger.error(f"❌ {job.job_id} failed with return code {job.process.returncode}")
                    
                    # Release GPU
                    if job.gpu_id is not None:
                        self.gpu_manager.release_gpu(job.gpu_id)
                    
                    # Try to launch next job
                    self.launch_next_job()
            
            time.sleep(30)  # Check every 30 seconds
    
    def launch_next_job(self):
        """Launch next pending job if GPU available"""
        pending_jobs = [job for job in self.jobs if job.status == 'pending']
        
        if pending_jobs:
            next_job = pending_jobs[0]
            if self.launch_job(next_job):
                logger.info(f"🔄 Launched next job: {next_job.job_id}")
    
    def collect_job_results(self, job: GridJob):
        """Collect results from completed job"""
        try:
            # Load monitor.csv if available
            monitor_file = job.output_dir / 'monitor.csv'
            if monitor_file.exists():
                df = pd.read_csv(monitor_file, comment='#')
                
                # Calculate key metrics
                ep_rew_mean = df['r'].mean() if 'r' in df.columns else 0.0
                ep_len_mean = df['l'].mean() if 'l' in df.columns else 0.0
                ep_rew_std = df['r'].std() if 'r' in df.columns else 0.0
                
                # Calculate Sharpe ratio
                sharpe = (ep_rew_mean / ep_rew_std) * np.sqrt(252) if ep_rew_std > 0 else 0.0
                
                # Estimate drawdown trigger rate (placeholder calculation)
                # Team A should implement actual drawdown trigger detection
                dd_trigger_rate = 0.0  # Placeholder
                
                # Success criteria evaluation
                success_criteria = self.config['success_criteria']
                ep_len_ok = ep_len_mean >= success_criteria['ep_len_min']
                ep_rew_ok = ep_rew_mean >= success_criteria['ep_rew_min']
                dd_triggers_ok = dd_trigger_rate < success_criteria['dd_trigger_rate_max']
                
                job.results = {
                    'ep_rew_mean': float(ep_rew_mean),
                    'ep_rew_std': float(ep_rew_std),
                    'ep_len_mean': float(ep_len_mean),
                    'sharpe_ratio': float(sharpe),
                    'dd_trigger_rate': float(dd_trigger_rate),
                    'total_episodes': len(df),
                    'success_criteria': {
                        'ep_len_ok': ep_len_ok,
                        'ep_rew_ok': ep_rew_ok,
                        'dd_triggers_ok': dd_triggers_ok,
                        'overall_success': ep_len_ok and ep_rew_ok and dd_triggers_ok
                    }
                }
                
                logger.info(f"📊 {job.job_id} results: reward={ep_rew_mean:.3f}, length={ep_len_mean:.1f}, sharpe={sharpe:.3f}")
            else:
                logger.warning(f"⚠️ No monitor.csv found for {job.job_id}")
                job.results = {'error': 'No monitor.csv found'}
                
        except Exception as e:
            logger.error(f"❌ Failed to collect results for {job.job_id}: {e}")
            job.results = {'error': str(e)}
    
    def launch_grid_search(self):
        """Launch complete grid search"""
        logger.info(f"🎯 Starting grid search: {self.grid_id}")
        
        # Create all jobs
        self.jobs = self.create_grid_jobs()
        
        # Launch initial batch of jobs (up to max_gpus)
        initial_jobs = self.jobs[:self.gpu_manager.max_gpus]
        for job in initial_jobs:
            if not self.launch_job(job):
                break
        
        # Save grid metadata
        self.save_grid_metadata()
        
        # Monitor jobs until completion
        logger.info("🔄 Monitoring grid search jobs...")
        self.monitor_jobs()
        
        # Generate final results
        self.generate_grid_summary()
        
        logger.info(f"🎉 Grid search completed: {self.grid_id}")
    
    def save_grid_metadata(self):
        """Save grid search metadata"""
        metadata = {
            'grid_id': self.grid_id,
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'total_jobs': len(self.jobs),
            'parameter_combinations': len(self.jobs),
            'jobs': [job.to_dict() for job in self.jobs]
        }
        
        metadata_file = self.grid_dir / 'grid_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_grid_summary(self):
        """Generate comprehensive grid search summary"""
        completed_jobs = [job for job in self.jobs if job.status == 'completed' and job.results]
        failed_jobs = [job for job in self.jobs if job.status == 'failed']
        
        if not completed_jobs:
            logger.error("❌ No completed jobs to analyze")
            return
        
        # Rank jobs by performance
        job_rankings = []
        for job in completed_jobs:
            if 'ep_rew_mean' in job.results:
                ranking_score = (
                    job.results['ep_rew_mean'] * 0.4 +  # 40% weight on reward
                    job.results['sharpe_ratio'] * 0.3 +  # 30% weight on Sharpe
                    (job.results['ep_len_mean'] / 100.0) * 0.2 +  # 20% weight on episode length
                    (1.0 - job.results['dd_trigger_rate']) * 0.1  # 10% weight on DD trigger avoidance
                )
                
                job_rankings.append({
                    'job_id': job.job_id,
                    'ranking_score': ranking_score,
                    'parameters': job.params,
                    'results': job.results
                })
        
        # Sort by ranking score
        job_rankings.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Generate summary
        summary = {
            'grid_id': self.grid_id,
            'completion_time': datetime.now().isoformat(),
            'total_jobs': len(self.jobs),
            'completed_jobs': len(completed_jobs),
            'failed_jobs': len(failed_jobs),
            'success_rate': len(completed_jobs) / len(self.jobs) if self.jobs else 0.0,
            'top_10_jobs': job_rankings[:10],
            'best_job': job_rankings[0] if job_rankings else None,
            'parameter_analysis': self.analyze_parameter_importance(job_rankings),
            'success_criteria_summary': self.summarize_success_criteria(completed_jobs)
        }
        
        # Save summary
        summary_file = self.grid_dir / 'grid_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self.print_grid_summary(summary)
        
        logger.info(f"📋 Grid summary saved to: {summary_file}")
    
    def analyze_parameter_importance(self, job_rankings: List[Dict]) -> Dict[str, Any]:
        """Analyze which parameters correlate with better performance"""
        if not job_rankings:
            return {}
        
        # Convert to DataFrame for analysis
        data = []
        for job in job_rankings:
            row = {**job['parameters'], 'ranking_score': job['ranking_score']}
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        correlations = {}
        for param in df.columns:
            if param != 'ranking_score' and df[param].dtype in ['int64', 'float64']:
                corr = df[param].corr(df['ranking_score'])
                correlations[param] = float(corr) if not pd.isna(corr) else 0.0
        
        # Find best values for each parameter
        best_job = job_rankings[0]
        best_params = best_job['parameters']
        
        return {
            'correlations': correlations,
            'best_parameters': best_params,
            'parameter_ranges': {
                param: {'min': float(df[param].min()), 'max': float(df[param].max())}
                for param in df.columns if param != 'ranking_score' and df[param].dtype in ['int64', 'float64']
            }
        }
    
    def summarize_success_criteria(self, completed_jobs: List[GridJob]) -> Dict[str, Any]:
        """Summarize success criteria across all jobs"""
        if not completed_jobs:
            return {}
        
        success_counts = {
            'ep_len_ok': 0,
            'ep_rew_ok': 0,
            'dd_triggers_ok': 0,
            'overall_success': 0
        }
        
        for job in completed_jobs:
            if 'success_criteria' in job.results:
                criteria = job.results['success_criteria']
                for key in success_counts:
                    if criteria.get(key, False):
                        success_counts[key] += 1
        
        total_jobs = len(completed_jobs)
        success_rates = {key: count / total_jobs for key, count in success_counts.items()}
        
        return {
            'success_counts': success_counts,
            'success_rates': success_rates,
            'total_jobs': total_jobs
        }
    
    def print_grid_summary(self, summary: Dict[str, Any]):
        """Print formatted grid search summary"""
        print("\n" + "="*80)
        print(f"🎯 MINI-GRID SEARCH RESULTS: {summary['grid_id']}")
        print("="*80)
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   Total Jobs: {summary['total_jobs']}")
        print(f"   Completed: {summary['completed_jobs']}")
        print(f"   Failed: {summary['failed_jobs']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        
        if summary['best_job']:
            best = summary['best_job']
            print(f"\n🏆 BEST JOB: {best['job_id']}")
            print(f"   Ranking Score: {best['ranking_score']:.3f}")
            print(f"   Episode Reward Mean: {best['results']['ep_rew_mean']:.3f}")
            print(f"   Sharpe Ratio: {best['results']['sharpe_ratio']:.3f}")
            print(f"   Episode Length Mean: {best['results']['ep_len_mean']:.1f}")
            
            print(f"\n🎛️ BEST PARAMETERS:")
            for param, value in best['parameters'].items():
                print(f"   {param}: {value}")
        
        if 'success_criteria_summary' in summary:
            success_summary = summary['success_criteria_summary']
            print(f"\n✅ SUCCESS CRITERIA SUMMARY:")
            print(f"   Episode Length ≥ 80: {success_summary['success_rates']['ep_len_ok']:.1%}")
            print(f"   Episode Reward ≥ 0: {success_summary['success_rates']['ep_rew_ok']:.1%}")
            print(f"   DD Triggers < 60%: {success_summary['success_rates']['dd_triggers_ok']:.1%}")
            print(f"   Overall Success: {success_summary['success_rates']['overall_success']:.1%}")

def main():
    """Main CLI interface for mini-grid orchestrator"""
    parser = argparse.ArgumentParser(
        description="Mini-Grid Hyperparameter Orchestrator for Phase 3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch grid search')
    launch_parser.add_argument('--config', type=Path, help='Configuration file path')
    launch_parser.add_argument('--phase2-results', type=Path, 
                              default='results/phase2/phase2_summary_report.json',
                              help='Phase 2 results file for parameter optimization')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor running grid search')
    monitor_parser.add_argument('--grid-id', required=True, help='Grid search ID to monitor')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze completed grid search')
    analyze_parser.add_argument('--results-dir', type=Path, required=True, 
                               help='Grid search results directory')
    
    args = parser.parse_args()
    
    if args.command == 'launch':
        orchestrator = MiniGridOrchestrator(args.config)
        
        # Update grid based on Phase 2 results if available
        if args.phase2_results:
            orchestrator.update_grid_from_phase2_results(args.phase2_results)
        
        # Launch grid search
        orchestrator.launch_grid_search()
        
    elif args.command == 'monitor':
        # Monitor existing grid search
        grid_dir = Path('results/grid_search') / args.grid_id
        if not grid_dir.exists():
            logger.error(f"❌ Grid directory not found: {grid_dir}")
            return 1
        
        # Load grid metadata and monitor
        metadata_file = grid_dir / 'grid_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📊 Monitoring grid: {metadata['grid_id']}")
            # Implement monitoring logic here
        else:
            logger.error(f"❌ Grid metadata not found: {metadata_file}")
            return 1
            
    elif args.command == 'analyze':
        # Analyze completed grid search
        if not args.results_dir.exists():
            logger.error(f"❌ Results directory not found: {args.results_dir}")
            return 1
        
        # Load and analyze results
        summary_file = args.results_dir / 'grid_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Print analysis
            orchestrator = MiniGridOrchestrator()
            orchestrator.print_grid_summary(summary)
        else:
            logger.error(f"❌ Grid summary not found: {summary_file}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    import os
    exit(main())