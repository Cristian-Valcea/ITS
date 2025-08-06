#!/usr/bin/env python3
"""
🔬 MICRO-OVERFIT PROBE
Instruments the 5K→10K degradation with fine-grained monitoring
Saves checkpoints every 1K steps and tracks KL divergence + reward metrics
"""

import argparse
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the trainer class
import importlib.util
spec = importlib.util.spec_from_file_location("train_diagnostic", Path(__file__).parent.parent / "train_diagnostic.py")
train_diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_diagnostic)
DiagnosticTrainer = train_diagnostic.DiagnosticTrainer

logger = logging.getLogger(__name__)

class MicroOverfitProbe:
    """Fine-grained monitoring of training dynamics around the 5K→10K degradation point"""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Override settings for micro-probe
        self.config['training']['total_timesteps'] = 8000  # Focus on 5K→8K range
        self.config['training']['save_interval'] = 2       # Save every 1K steps (512*2)
        self.config['training']['log_interval'] = 1        # Log every update
        
        # Setup logging
        log_file = self.output_dir / f"micro_overfit_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"🔬 Micro-overfit probe initialized")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Target steps: 8000 (focus on 5K→8K degradation)")
        
        # Metrics tracking
        self.metrics_history = []
        self.checkpoint_rewards = {}
        
    def run_probe(self):
        """Run the micro-overfit probe with detailed monitoring"""
        logger.info("🚀 Starting micro-overfit probe...")
        
        # Run training with the existing DiagnosticTrainer
        try:
            checkpoint_path = self.output_dir / "micro_probe_final.zip"
            trainer = DiagnosticTrainer(self.config_path)
            
            logger.info("🚀 Starting micro-overfit training (8K steps)...")
            trainer.train(
                total_timesteps=8000,
                save_path=str(checkpoint_path)
            )
            
            logger.info("✅ Micro-overfit probe training completed successfully")
            
            # Parse training logs for analysis
            self.parse_training_logs()
            
        except Exception as e:
            logger.error(f"❌ Micro-overfit probe failed: {e}")
            raise
        
        # Analyze results
        self.analyze_results()
        
    def parse_training_logs(self):
        """Parse training logs to extract metrics for analysis"""
        logger.info("📊 Parsing training logs for metrics...")
        
        # Look for recent log files
        log_files = list(Path("diagnostic_runs").glob("**/logs/*.log"))
        if not log_files:
            logger.warning("No log files found for parsing")
            return
            
        # Get the most recent log file
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"📄 Parsing log file: {latest_log}")
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Parse stable-baselines3 training output
            for line in lines:
                if "| rollout/" in line and "ep_rew_mean" in line:
                    try:
                        # Extract metrics from SB3 log format
                        parts = line.split("|")
                        metrics = {}
                        step = None
                        
                        for part in parts:
                            part = part.strip()
                            if "total_timesteps" in part:
                                step = int(part.split()[-1])
                            elif "ep_rew_mean" in part:
                                metrics['ep_rew_mean'] = float(part.split()[-1])
                            elif "ep_len_mean" in part:
                                metrics['ep_len_mean'] = float(part.split()[-1])
                        
                        # Look for training metrics in subsequent lines
                        if step and metrics:
                            # Add default values for missing metrics
                            metric_record = {
                                'step': step,
                                'ep_rew_mean': metrics.get('ep_rew_mean', 0),
                                'ep_len_mean': metrics.get('ep_len_mean', 0),
                                'policy_loss': 0,  # Will be filled from train/ section
                                'value_loss': 0,
                                'entropy_loss': 0,
                                'approx_kl': 0,
                                'clip_fraction': 0,
                                'explained_variance': 0
                            }
                            self.metrics_history.append(metric_record)
                    except Exception:
                        continue
                            
                elif "| train/" in line:
                    # Parse training metrics and update the last record
                    if self.metrics_history:
                        parts = line.split("|")
                        for part in parts:
                            part = part.strip()
                            if "policy_gradient_loss" in part:
                                self.metrics_history[-1]['policy_loss'] = float(part.split()[-1])
                            elif "value_loss" in part:
                                self.metrics_history[-1]['value_loss'] = float(part.split()[-1])
                            elif "entropy_loss" in part:
                                self.metrics_history[-1]['entropy_loss'] = float(part.split()[-1])
                            elif "approx_kl" in part:
                                self.metrics_history[-1]['approx_kl'] = float(part.split()[-1])
                            elif "clip_fraction" in part:
                                self.metrics_history[-1]['clip_fraction'] = float(part.split()[-1])
                            elif "explained_variance" in part:
                                self.metrics_history[-1]['explained_variance'] = float(part.split()[-1])
                                
        except Exception as e:
            logger.error(f"Error parsing training logs: {e}")
            
        logger.info(f"📊 Parsed {len(self.metrics_history)} metric records from logs")
        
    def analyze_results(self):
        """Analyze the collected metrics and generate insights"""
        logger.info("🔍 Analyzing micro-overfit probe results...")
        
        if not self.metrics_history:
            logger.warning("No metrics collected during probe")
            return
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.metrics_history)
        
        # Save raw metrics
        metrics_file = self.output_dir / "micro_probe_metrics.csv"
        df.to_csv(metrics_file, index=False)
        logger.info(f"📊 Metrics saved to: {metrics_file}")
        
        # Generate analysis plots
        self.plot_training_dynamics(df)
        
        # Detect degradation patterns
        self.detect_degradation_patterns(df)
        
        # Generate summary report
        self.generate_summary_report(df)
        
    def plot_training_dynamics(self, df):
        """Generate plots of training dynamics"""
        logger.info("📈 Generating training dynamics plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Micro-Overfit Probe: Training Dynamics (5K→8K)', fontsize=16)
        
        # Plot 1: Reward progression
        axes[0, 0].plot(df['step'], df['ep_rew_mean'], 'b-', linewidth=2)
        axes[0, 0].axvline(x=5000, color='r', linestyle='--', alpha=0.7, label='5K mark')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Episode Reward Mean')
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: KL Divergence
        axes[0, 1].plot(df['step'], df['approx_kl'], 'g-', linewidth=2)
        axes[0, 1].axvline(x=5000, color='r', linestyle='--', alpha=0.7, label='5K mark')
        axes[0, 1].axhline(y=0.01, color='orange', linestyle=':', alpha=0.7, label='KL=0.01')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Approximate KL Divergence')
        axes[0, 1].set_title('KL Divergence (Policy Stability)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Policy Loss
        axes[1, 0].plot(df['step'], df['policy_loss'], 'purple', linewidth=2)
        axes[1, 0].axvline(x=5000, color='r', linestyle='--', alpha=0.7, label='5K mark')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Policy Gradient Loss')
        axes[1, 0].set_title('Policy Loss Progression')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Value Loss
        axes[1, 1].plot(df['step'], df['value_loss'], 'orange', linewidth=2)
        axes[1, 1].axvline(x=5000, color='r', linestyle='--', alpha=0.7, label='5K mark')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Value Function Loss')
        axes[1, 1].set_title('Value Loss Progression')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "training_dynamics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Training dynamics plot saved to: {plot_file}")
        plt.close()
        
    def detect_degradation_patterns(self, df):
        """Detect specific degradation patterns in the data"""
        logger.info("🔍 Detecting degradation patterns...")
        
        # Find 5K baseline
        baseline_idx = df[df['step'] >= 5000].index[0] if len(df[df['step'] >= 5000]) > 0 else -1
        
        if baseline_idx == -1:
            logger.warning("No data points at or after 5K steps")
            return
            
        baseline_reward = df.loc[baseline_idx, 'ep_rew_mean']
        baseline_kl = df.loc[baseline_idx, 'approx_kl']
        
        logger.info(f"📊 5K Baseline: Reward={baseline_reward:.3f}, KL={baseline_kl:.6f}")
        
        # Check for patterns after 5K
        post_5k = df[df['step'] > 5000]
        
        if len(post_5k) == 0:
            logger.warning("No data points after 5K steps")
            return
            
        # Pattern 1: KL spike
        max_kl = post_5k['approx_kl'].max()
        if max_kl > 0.01:
            kl_spike_step = post_5k[post_5k['approx_kl'] == max_kl]['step'].iloc[0]
            logger.warning(f"🚨 KL SPIKE DETECTED: {max_kl:.6f} at step {kl_spike_step}")
            logger.warning("   → Optimizer taking large policy steps")
            logger.warning("   → Recommendation: Lower learning rate or tighten target_kl")
        
        # Pattern 2: Reward collapse
        min_reward = post_5k['ep_rew_mean'].min()
        if min_reward < baseline_reward - 10:
            reward_collapse_step = post_5k[post_5k['ep_rew_mean'] == min_reward]['step'].iloc[0]
            logger.warning(f"🚨 REWARD COLLAPSE DETECTED: {min_reward:.3f} at step {reward_collapse_step}")
            logger.warning(f"   → Dropped {baseline_reward - min_reward:.3f} from 5K baseline")
            
            # Check if KL was stable during collapse
            collapse_kl = post_5k[post_5k['step'] == reward_collapse_step]['approx_kl'].iloc[0]
            if collapse_kl < 0.01:
                logger.warning("   → KL remained stable during collapse")
                logger.warning("   → Likely reward loophole or environment issue")
            else:
                logger.warning("   → KL was high during collapse")
                logger.warning("   → Likely optimizer instability")
        
        # Pattern 3: Gradual degradation
        if len(post_5k) >= 3:
            reward_trend = np.polyfit(post_5k['step'], post_5k['ep_rew_mean'], 1)[0]
            if reward_trend < -0.001:  # Negative slope
                logger.warning(f"🚨 GRADUAL DEGRADATION DETECTED: Slope={reward_trend:.6f}")
                logger.warning("   → Consistent reward decline after 5K")
                logger.warning("   → May indicate systematic overfitting")
        
    def generate_summary_report(self, df):
        """Generate a summary report of findings"""
        logger.info("📝 Generating summary report...")
        
        report_file = self.output_dir / "micro_probe_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Micro-Overfit Probe Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Config**: {self.config_path}\n\n")
            f.write(f"**Steps Analyzed**: {df['step'].min()} → {df['step'].max()}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            
            baseline_idx = df[df['step'] >= 5000].index[0] if len(df[df['step'] >= 5000]) > 0 else -1
            if baseline_idx != -1:
                baseline_reward = df.loc[baseline_idx, 'ep_rew_mean']
                final_reward = df['ep_rew_mean'].iloc[-1]
                max_kl = df['approx_kl'].max()
                
                f.write(f"- **5K Baseline Reward**: {baseline_reward:.3f}\n")
                f.write(f"- **Final Reward**: {final_reward:.3f}\n")
                f.write(f"- **Reward Change**: {final_reward - baseline_reward:.3f}\n")
                f.write(f"- **Max KL Divergence**: {max_kl:.6f}\n")
                f.write(f"- **Total Episodes**: {len(df)}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            post_5k = df[df['step'] > 5000]
            if len(post_5k) > 0:
                max_kl = post_5k['approx_kl'].max()
                min_reward = post_5k['ep_rew_mean'].min()
                
                if max_kl > 0.01:
                    f.write("- ⚠️ **KL Spike Detected**: Policy instability after 5K steps\n")
                
                if baseline_idx != -1 and min_reward < df.loc[baseline_idx, 'ep_rew_mean'] - 5:
                    f.write("- ⚠️ **Reward Degradation**: Significant performance drop after 5K\n")
                
                if len(post_5k) >= 3:
                    reward_trend = np.polyfit(post_5k['step'], post_5k['ep_rew_mean'], 1)[0]
                    if reward_trend < -0.001:
                        f.write("- ⚠️ **Gradual Decline**: Consistent reward degradation trend\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis above:\n\n")
            f.write("1. **If KL spikes**: Lower learning rate or add target_kl constraint\n")
            f.write("2. **If reward collapses with stable KL**: Investigate reward system logic\n")
            f.write("3. **If gradual decline**: Consider early stopping or regularization\n")
            f.write("4. **Compare with Phase 1B results** to isolate root cause\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- `micro_probe_metrics.csv`: Raw training metrics\n")
            f.write(f"- `training_dynamics.png`: Visualization plots\n")
            f.write(f"- `micro_probe_final.zip`: Final model checkpoint\n")
        
        logger.info(f"📝 Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Micro-overfit probe for 5K→10K degradation")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run probe
    probe = MicroOverfitProbe(args.config, args.output_dir)
    probe.run_probe()
    
    print(f"\n🎉 Micro-overfit probe completed!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()