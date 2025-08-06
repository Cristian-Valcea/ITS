#!/usr/bin/env python3
"""
Seed Variance & Temporal Robustness Testing
Phase 3: 4-seed × 2-split validation runs
"""

import os
import sys
import subprocess
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class SeedVarianceRunner:
    """
    Manages multi-seed, multi-split training runs for robustness testing
    """
    
    def __init__(self, 
                 base_config: str = "config/curriculum/phase2_oos.yaml",
                 base_model: str = "train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/",
                 steps: int = 10000,
                 max_parallel: int = 3):
        
        self.base_config = base_config
        self.base_model = base_model  
        self.steps = steps
        self.max_parallel = max_parallel
        
        # Define temporal splits
        self.splits = [
            {
                "name": "2022_to_2023", 
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01", 
                "test_end": "2023-12-31"
            },
            {
                "name": "2023_to_2024",
                "train_start": "2023-01-01", 
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31"
            }
        ]
        
        # Seeds for variance testing
        self.seeds = [0, 1, 2, 3]
        
        # Results storage
        self.results = []
        self.active_jobs = []
        self.completed_jobs = []
        self.failed_jobs = []
        
        print(f"🎲 Seed Variance Runner initialized")
        print(f"   Seeds: {self.seeds}")
        print(f"   Splits: {len(self.splits)}")
        print(f"   Total jobs: {len(self.seeds) * len(self.splits)}")
        print(f"   Max parallel: {self.max_parallel}")
    
    def generate_run_name(self, seed: int, split_name: str) -> str:
        """Generate unique run directory name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"train_runs/seed_variance_{split_name}_seed{seed}_{timestamp}"
    
    def create_training_command(self, seed: int, split: Dict) -> Dict:
        """Create training command for specific seed and split"""
        
        run_name = self.generate_run_name(seed, split["name"])
        
        # GPU assignment (cycle through available GPUs 0-2)
        gpu_id = seed % 3
        
        cmd = [
            "python", "train.py",
            "--config", self.base_config,
            "--seed", str(seed),
            "--steps", str(self.steps),
            "--resume", self.base_model,
            "--train-start", split["train_start"],
            "--train-end", split["train_end"], 
            "--test-start", split["test_start"],
            "--test-end", split["test_end"],
            "--output-dir", run_name,
            "--use-governor"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        job_info = {
            "seed": seed,
            "split": split["name"],
            "run_name": run_name,
            "gpu_id": gpu_id,
            "command": cmd,
            "env": env,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "process": None
        }
        
        return job_info
    
    def start_job(self, job_info: Dict) -> bool:
        """Start a single training job"""
        
        try:
            print(f"🚀 Starting job: {job_info['split']} seed={job_info['seed']} GPU={job_info['gpu_id']}")
            print(f"   Command: {' '.join(job_info['command'])}")
            
            # Create output directory
            os.makedirs(job_info['run_name'], exist_ok=True)
            
            # Start process
            log_file = Path(job_info['run_name']) / "training.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    job_info['command'],
                    env=job_info['env'],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(project_root)
                )
            
            job_info["process"] = process
            job_info["status"] = "running"
            job_info["start_time"] = time.time()
            job_info["log_file"] = str(log_file)
            
            self.active_jobs.append(job_info)
            
            print(f"   ✅ Job started (PID: {process.pid})")
            print(f"   📝 Logs: {log_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start job: {e}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            self.failed_jobs.append(job_info)
            return False
    
    def check_job_completion(self):
        """Check for completed jobs and update status"""
        
        completed_this_check = []
        
        for job in self.active_jobs[:]:  # Copy list to allow modification
            if job["process"].poll() is not None:  # Process finished
                job["end_time"] = time.time()
                job["duration_minutes"] = (job["end_time"] - job["start_time"]) / 60.0
                
                if job["process"].returncode == 0:
                    job["status"] = "completed"
                    self.completed_jobs.append(job)
                    print(f"✅ Job completed: {job['split']} seed={job['seed']} "
                          f"({job['duration_minutes']:.1f} min)")
                else:
                    job["status"] = "failed"
                    job["return_code"] = job["process"].returncode
                    self.failed_jobs.append(job)
                    print(f"❌ Job failed: {job['split']} seed={job['seed']} "
                          f"(code: {job['process'].returncode})")
                
                self.active_jobs.remove(job)
                completed_this_check.append(job)
        
        return completed_this_check
    
    def run_all_jobs(self) -> List[Dict]:
        """Run all seed variance jobs with parallel execution"""
        
        print("\n🎯 Starting seed variance testing...")
        print("="*50)
        
        # Generate all job configurations
        all_jobs = []
        for split in self.splits:
            for seed in self.seeds:
                job = self.create_training_command(seed, split)
                all_jobs.append(job)
        
        print(f"📋 Generated {len(all_jobs)} jobs")
        
        # Execute jobs with parallelization
        job_queue = all_jobs[:]
        
        while job_queue or self.active_jobs:
            
            # Start new jobs if slots available
            while len(self.active_jobs) < self.max_parallel and job_queue:
                next_job = job_queue.pop(0)
                self.start_job(next_job)
            
            # Check for completions
            self.check_job_completion()
            
            # Status update
            total_jobs = len(all_jobs)
            completed = len(self.completed_jobs)
            failed = len(self.failed_jobs)
            active = len(self.active_jobs)
            
            if active > 0:
                print(f"📊 Progress: {completed}/{total_jobs} completed, "
                      f"{failed} failed, {active} running")
                time.sleep(30)  # Check every 30 seconds
        
        print("\n🏁 All jobs finished!")
        print(f"   Completed: {len(self.completed_jobs)}")
        print(f"   Failed: {len(self.failed_jobs)}")
        
        return all_jobs
    
    def analyze_seed_variance(self, output_file: Optional[str] = None) -> Dict:
        """Analyze variance across seeds and temporal splits"""
        
        if not self.completed_jobs:
            return {
                "error": "No completed jobs to analyze",
                "variance_analysis": {}
            }
        
        print("\n📈 Analyzing seed variance...")
        
        # Load results from monitor.csv files
        analysis_results = {}
        
        for split in self.splits:
            split_name = split["name"]
            split_results = []
            
            for job in self.completed_jobs:
                if job["split"] == split_name:
                    
                    monitor_file = Path(job["run_name"]) / "monitor.csv"
                    
                    if monitor_file.exists():
                        try:
                            # Import evaluate_phase2 to reuse evaluation logic
                            sys.path.insert(0, str(Path(__file__).parent))
                            from evaluate_phase2 import evaluate_oos_performance
                            
                            result = evaluate_oos_performance(job["run_name"])
                            result["seed"] = job["seed"]
                            result["job_duration_min"] = job.get("duration_minutes", 0)
                            split_results.append(result)
                            
                        except Exception as e:
                            print(f"⚠️ Error analyzing {job['run_name']}: {e}")
            
            if split_results:
                # Calculate variance statistics
                rewards = [r["ep_rew_mean"] for r in split_results if not r.get("error")]
                sharpes = [r["sharpe_ratio"] for r in split_results if not r.get("error")]
                
                if rewards and sharpes:
                    reward_mean = np.mean(rewards)
                    reward_std = np.std(rewards)
                    reward_cv = reward_std / abs(reward_mean) if reward_mean != 0 else np.inf
                    
                    sharpe_mean = np.mean(sharpes)
                    sharpe_std = np.std(sharpes)
                    sharpe_cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else np.inf
                    
                    # Success criteria: σ/μ < 30%
                    variance_success = reward_cv < 0.30
                    
                    analysis_results[split_name] = {
                        "seed_count": len(split_results),
                        "reward_stats": {
                            "mean": float(reward_mean),
                            "std": float(reward_std),
                            "cv": float(reward_cv),  # Coefficient of variation
                            "values": rewards
                        },
                        "sharpe_stats": {
                            "mean": float(sharpe_mean),
                            "std": float(sharpe_std), 
                            "cv": float(sharpe_cv),
                            "values": sharpes
                        },
                        "variance_success": variance_success,
                        "variance_threshold": 0.30,
                        "individual_results": split_results
                    }
        
        # Overall variance analysis
        all_rewards = []
        all_sharpes = []
        
        for split_data in analysis_results.values():
            all_rewards.extend(split_data["reward_stats"]["values"])
            all_sharpes.extend(split_data["sharpe_stats"]["values"])
        
        overall_reward_cv = np.std(all_rewards) / abs(np.mean(all_rewards)) if all_rewards and np.mean(all_rewards) != 0 else np.inf
        overall_variance_success = overall_reward_cv < 0.30
        
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed_variance_analysis": {
                "overall_success": overall_variance_success,
                "overall_reward_cv": float(overall_reward_cv),
                "variance_threshold": 0.30,
                "splits_analyzed": len(analysis_results),
                "total_successful_jobs": len(self.completed_jobs),
                "total_failed_jobs": len(self.failed_jobs)
            },
            "split_results": analysis_results,
            "job_summary": {
                "completed": [
                    {
                        "seed": job["seed"],
                        "split": job["split"], 
                        "duration_min": job.get("duration_minutes", 0),
                        "run_path": job["run_name"]
                    }
                    for job in self.completed_jobs
                ],
                "failed": [
                    {
                        "seed": job["seed"],
                        "split": job["split"],
                        "error": job.get("error", "Unknown"),
                        "return_code": job.get("return_code")
                    }
                    for job in self.failed_jobs
                ]
            }
        }
        
        # Print summary
        print("\n" + "="*50)
        print("📊 SEED VARIANCE ANALYSIS SUMMARY")
        print("="*50)
        
        for split_name, data in analysis_results.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Reward CV: {data['reward_stats']['cv']:.1%} "
                  f"({'✅' if data['variance_success'] else '❌'} <30%)")
            print(f"  Sharpe CV: {data['sharpe_stats']['cv']:.1%}")
            print(f"  Mean Reward: {data['reward_stats']['mean']:.3f} ± {data['reward_stats']['std']:.3f}")
            print(f"  Mean Sharpe: {data['sharpe_stats']['mean']:.3f} ± {data['sharpe_stats']['std']:.3f}")
        
        print(f"\n🎯 OVERALL VARIANCE SUCCESS: {'✅ PASSED' if overall_variance_success else '❌ FAILED'}")
        print(f"   Overall Reward CV: {overall_reward_cv:.1%} (<30%)")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Analysis saved to: {output_file}")
        
        return summary

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed Variance & Temporal Robustness Testing")
    parser.add_argument('--config', type=str,
                       default='config/curriculum/phase2_oos.yaml',
                       help='Base config file')
    parser.add_argument('--base-model', type=str,
                       default='train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/',
                       help='Base model to resume from')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Training steps per job')
    parser.add_argument('--max-parallel', type=int, default=3,
                       help='Maximum parallel jobs')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='Seeds to test')
    parser.add_argument('--output', type=str,
                       default='seed_variance_analysis.json',
                       help='Output analysis file')
    parser.add_argument('--analyze-only', type=str,
                       help='Skip training, analyze existing results from pattern')
    
    args = parser.parse_args()
    
    print("🎲 Seed Variance Testing Starting...")
    print("="*50)
    
    runner = SeedVarianceRunner(
        base_config=args.config,
        base_model=args.base_model,
        steps=args.steps,
        max_parallel=args.max_parallel
    )
    
    # Override seeds if provided
    runner.seeds = args.seeds
    
    if args.analyze_only:
        # Analysis-only mode (for existing completed runs)
        import glob
        completed_paths = glob.glob(args.analyze_only)
        
        # Mock completed jobs for analysis
        runner.completed_jobs = []
        for path in completed_paths:
            # Extract seed and split from path
            path_parts = Path(path).name.split('_')
            seed = next((int(part.replace('seed', '')) for part in path_parts if part.startswith('seed')), 0)
            split = next((part for part in path_parts if '2022' in part or '2023' in part), 'unknown')
            
            runner.completed_jobs.append({
                "seed": seed,
                "split": split,
                "run_name": path,
                "duration_minutes": 0
            })
        
        print(f"📂 Analyzing {len(runner.completed_jobs)} existing runs")
        
    else:
        # Full training + analysis
        print(f"🚀 Running {len(runner.seeds)} seeds × {len(runner.splits)} splits = {len(runner.seeds) * len(runner.splits)} jobs")
        runner.run_all_jobs()
    
    # Analyze results
    analysis = runner.analyze_seed_variance(args.output)
    
    # Return appropriate exit code
    if analysis.get("seed_variance_analysis", {}).get("overall_success", False):
        print("\n🎉 Seed variance testing PASSED!")
        return 0
    else:
        print("\n⚠️ Seed variance testing FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())