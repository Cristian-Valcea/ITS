#!/usr/bin/env python3
"""
🚀 STAIRWAYS V4 - 200K REAL DATA TRAINING
Progressive 3-phase training with proper learning iterations (390 updates vs failed 23)

FIXES:
- n_steps: 2048 → 512 (4x more policy updates)
- timesteps: 48K → 200K (4.2x more learning)
- Progressive phases with exploration → exploitation schedule
- Validation gates to prevent over-conservative behavior
"""

import sys
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import our monitoring system
from training_monitor import TrainingMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressCallback(BaseCallback):
    """Custom callback for real-time progress reporting"""
    
    def __init__(self, monitor: TrainingMonitor, phase_name: str, total_steps: int, report_freq: int = 10000):
        super().__init__()
        self.monitor = monitor
        self.phase_name = phase_name
        self.total_steps = total_steps
        self.report_freq = report_freq
        self.last_report_step = 0
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        
        # Report progress every report_freq steps (default 10K)
        if current_step - self.last_report_step >= self.report_freq:
            
            # Get recent episode metrics
            episode_metrics = self.monitor.analyze_episode_rewards()
            
            metrics_str = ""
            if episode_metrics["status"] == "active":
                metrics_str = f"Episodes: {episode_metrics['total_episodes']}, Avg Reward: {episode_metrics['avg_reward']:.3f}"
            
            # Log progress
            self.monitor.log_progress(
                current_step=current_step,
                total_steps=self.total_steps,
                phase=self.phase_name,
                metrics={"episodes": episode_metrics.get("total_episodes", 0), 
                        "avg_reward": episode_metrics.get("avg_reward", 0)}
            )
            
            # Generate milestone report every 20K steps
            if current_step % 20000 == 0:
                self.monitor.save_milestone_report(current_step, self.total_steps, self.phase_name)
            
            self.last_report_step = current_step
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Quick progress update
        current_step = self.num_timesteps
        progress_pct = (current_step / self.total_steps) * 100
        
        if current_step % 5000 == 0:  # Every 5K steps
            print(f"🎯 {self.phase_name}: {current_step:,}/{self.total_steps:,} ({progress_pct:.1f}%)")

class StairwaysV4Trainer:
    """Stairways V4 Progressive Training with Real Data"""
    
    def __init__(self, config_path: str = "config/ppo_200k_stairways_v4.yaml"):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"stairways_v4_200k_{self.timestamp}"
        
        # Setup paths
        self.save_path = Path(f"train_runs/{self.run_name}")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.model = None
        self.training_results = []
        self.monitor = TrainingMonitor(self.run_name)
        self.total_steps_trained = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def _create_environment(self) -> DummyVecEnv:
        """Create training environment with real data"""
        try:
            logger.info("📈 Loading real market data...")
            
            # Database configuration with vault password
            db_password = SecretsHelper.get_timescaledb_password()
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_data',
                'user': 'postgres',
                'password': db_password
            }
            
            # Load real market data
            adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
            market_data = adapter.load_training_data(
                start_date=self.config['environment']['start_date'],
                end_date=self.config['environment']['end_date'],
                symbols=self.config['environment']['symbols'],
                bar_size=self.config['environment']['bar_size'],
                data_split='train'
            )
            
            # Prepare data for environment
            nvda_features = market_data['nvda_features']
            nvda_prices = market_data['nvda_prices']
            msft_features = market_data['msft_features']
            msft_prices = market_data['msft_prices']
            trading_days = market_data['trading_days']
            
            # Combine features (26-dim observation)
            combined_features = np.concatenate([nvda_features, msft_features], axis=1)
            position_features = np.zeros((combined_features.shape[0], 2))
            combined_features = np.concatenate([combined_features, position_features], axis=1)
            
            # Create 4-column price data
            nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
            msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
            
            combined_prices = np.column_stack([
                nvda_prices, nvda_returns, msft_prices, msft_returns
            ])
            
            logger.info(f"📊 Data loaded: {len(trading_days)} timesteps, {combined_features.shape[1]} features")
            
            # Create environment with relaxed termination conditions
            env = DualTickerTradingEnvV3Enhanced(
                processed_feature_data=combined_features,
                processed_price_data=combined_prices,
                trading_days=trading_days,
                initial_capital=self.config['environment']['initial_capital'],
                lookback_window=self.config['environment']['lookback_window'],
                max_episode_steps=self.config['environment']['max_episode_steps'],
                max_daily_drawdown_pct=0.20,  # 20% very permissive for Phase 1 learning
                transaction_cost_pct=self.config['environment']['transaction_cost_pct']
            )
            
            # Monitor environment for logging
            env = Monitor(env, str(self.save_path / "monitor.csv"))
            
            # Wrap for stable-baselines3
            return DummyVecEnv([lambda: env])
            
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            raise
    
    def _create_model(self, phase_config: Dict[str, Any]) -> PPO:
        """Create PPO model with phase-specific configuration"""
        
        # Get learning rate (handle linear decay and scientific notation)
        learning_rate = phase_config.get('learning_rate', self.config['ppo']['learning_rate'])
        if isinstance(learning_rate, str):
            if 'linear_decay' in learning_rate:
                learning_rate = 3e-4  # Default, will be handled by scheduler
            elif learning_rate == '5e-4':
                learning_rate = 5e-4
            else:
                learning_rate = float(learning_rate)
        
        model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            n_steps=self.config['ppo']['n_steps'],  # CRITICAL: 512 not 2048
            batch_size=self.config['ppo']['batch_size'],
            n_epochs=self.config['ppo']['n_epochs'],
            gamma=self.config['ppo']['gamma'],
            gae_lambda=self.config['ppo']['gae_lambda'],
            clip_range=self.config['ppo']['clip_range'],
            ent_coef=phase_config.get('ent_coef', self.config['ppo']['ent_coef']),
            vf_coef=self.config['ppo']['vf_coef'],
            max_grad_norm=self.config['ppo']['max_grad_norm'],
            policy_kwargs={'net_arch': [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=f"tensorboard_logs/{self.run_name}",
            verbose=1
        )
        
        return model
    
    def _run_validation(self, model: PPO, phase_name: str) -> Dict[str, float]:
        """Run validation episode and check gates"""
        
        logger.info(f"🧪 Running validation for {phase_name}...")
        
        # Run 5 validation episodes
        validation_results = []
        
        for episode in range(5):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            actions = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]
                actions.append(action[0])
            
            # Calculate metrics
            hold_rate = sum(1 for a in actions if a == 4) / len(actions) if actions else 1.0
            
            validation_results.append({
                'episode_reward': episode_reward,
                'hold_rate': hold_rate,
                'total_actions': len(actions)
            })
        
        # Aggregate results
        avg_reward = np.mean([r['episode_reward'] for r in validation_results])
        avg_hold_rate = np.mean([r['hold_rate'] for r in validation_results])
        
        results = {
            'avg_reward': avg_reward,
            'hold_rate': avg_hold_rate,
            'phase': phase_name
        }
        
        logger.info(f"📊 {phase_name} Validation: Reward={avg_reward:.3f}, Hold Rate={avg_hold_rate:.1%}")
        
        return results
    
    def _check_validation_gates(self, results: Dict[str, float], phase_name: str) -> bool:
        """Check if validation results pass the gates"""
        
        gates = self.config['validation_gates']
        
        # Check hold rate range
        hold_rate = results['hold_rate']
        hold_range = gates['hold_rate_range']
        
        if not (hold_range[0] <= hold_rate <= hold_range[1]):
            logger.warning(f"⚠️ {phase_name}: Hold rate {hold_rate:.1%} outside target range {hold_range[0]:.1%}-{hold_range[1]:.1%}")
            return False
        
        # Check if model is trading (not stuck at 100% hold)
        if hold_rate > 0.9:
            logger.warning(f"⚠️ {phase_name}: Model is over-conservative (hold rate {hold_rate:.1%})")
            return False
        
        logger.info(f"✅ {phase_name}: Validation gates passed")
        return True
    
    def train_phase(self, phase_name: str, phase_config: Dict[str, Any]) -> bool:
        """Train a single phase"""
        
        logger.info(f"🚀 Starting {phase_name}...")
        logger.info(f"   Timesteps: {phase_config['timesteps']:,}")
        logger.info(f"   Exploration: {phase_config['ent_coef']}")
        logger.info(f"   Target: {phase_config['target']}")
        
        # Create/update model for this phase
        if self.model is None:
            self.model = self._create_model(phase_config)
        else:
            # Update hyperparameters for new phase
            self.model.ent_coef = phase_config['ent_coef']
            if 'learning_rate' in phase_config:
                self.model.learning_rate = phase_config['learning_rate']
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,  # Every 25K steps
            save_path=str(self.save_path / f"{phase_name}_checkpoints"),
            name_prefix=f'{phase_name}_model'
        )
        
        # Progress monitoring callback
        progress_callback = ProgressCallback(
            monitor=self.monitor,
            phase_name=phase_name,
            total_steps=self.total_steps_trained + phase_config['timesteps'],
            report_freq=10000  # Report every 10K steps
        )
        
        # Train phase
        try:
            print(f"\n🚀 {phase_name.upper()} TRAINING STARTED")
            print(f"📊 Watch progress: python training_monitor.py {self.run_name}")
            print(f"📈 TensorBoard: tensorboard --logdir tensorboard_logs/{self.run_name}")
            print("=" * 60)
            
            self.model.learn(
                total_timesteps=phase_config['timesteps'],
                callback=[checkpoint_callback, progress_callback],
                tb_log_name=f"{self.run_name}_{phase_name}",
                reset_num_timesteps=False  # Continue from previous phase
            )
            
            # Update total steps trained
            self.total_steps_trained += phase_config['timesteps']
            
            # Validation
            validation_results = self._run_validation(self.model, phase_name)
            gates_passed = self._check_validation_gates(validation_results, phase_name)
            
            if not gates_passed:
                logger.error(f"❌ {phase_name}: Failed validation gates")
                return False
            
            # Save phase results
            phase_results = {
                'phase': phase_name,
                'timesteps': phase_config['timesteps'],
                'validation': validation_results,
                'gates_passed': gates_passed
            }
            
            self.training_results.append(phase_results)
            
            # Save phase model
            phase_model_path = self.save_path / f"{phase_name}_final_model.zip"
            self.model.save(str(phase_model_path))
            logger.info(f"✅ {phase_name} complete - model saved to {phase_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {phase_name} failed: {e}")
            return False
    
    def run_full_training(self) -> bool:
        """Run complete 3-phase Stairways V4 training"""
        
        logger.info("🎯 STAIRWAYS V4 - 200K REAL DATA TRAINING")
        logger.info("=" * 60)
        logger.info(f"Run: {self.run_name}")
        logger.info(f"Total timesteps: {self.config['training']['total_timesteps']:,}")
        logger.info(f"Expected updates: ~390 (vs failed 48K with only 23 updates)")
        
        # Create environment
        self.env = self._create_environment()
        
        # Phase 1: Warm-up exploration
        phase1_success = self.train_phase("phase_1_warmup", self.config['phases']['phase_1_warmup'])
        if not phase1_success:
            logger.error("❌ Phase 1 failed - stopping training")
            return False
        
        # Phase 2: Core learning
        phase2_success = self.train_phase("phase_2_core", self.config['phases']['phase_2_core'])
        if not phase2_success:
            logger.error("❌ Phase 2 failed - stopping training")
            return False
        
        # Phase 3: Refinement
        phase3_success = self.train_phase("phase_3_refinement", self.config['phases']['phase_3_refinement'])
        if not phase3_success:
            logger.error("❌ Phase 3 failed - stopping training")
            return False
        
        # Save final model
        final_model_path = self.save_path / "stairways_v4_final_model.zip"
        self.model.save(str(final_model_path))
        
        # Generate training report
        self._generate_final_report()
        
        logger.info("🎉 STAIRWAYS V4 TRAINING COMPLETE!")
        logger.info(f"📁 Final model: {final_model_path}")
        logger.info(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs/{self.run_name}")
        
        return True
    
    def _generate_final_report(self):
        """Generate comprehensive training report"""
        
        report_path = self.save_path / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("STAIRWAYS V4 - 200K REAL DATA TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run Name: {self.run_name}\n")
            f.write(f"Total Timesteps: {self.config['training']['total_timesteps']:,}\n")
            f.write(f"Policy Updates: ~390 (vs failed 48K with 23 updates)\n\n")
            
            for result in self.training_results:
                phase = result['phase']
                validation = result['validation']
                f.write(f"{phase.upper()}:\n")
                f.write(f"  Timesteps: {result['timesteps']:,}\n")
                f.write(f"  Average Reward: {validation['avg_reward']:.3f}\n")
                f.write(f"  Hold Rate: {validation['hold_rate']:.1%}\n")
                f.write(f"  Gates Passed: {result['gates_passed']}\n\n")
        
        logger.info(f"📄 Training report saved: {report_path}")

def main():
    """Main execution"""
    
    try:
        trainer = StairwaysV4Trainer()
        success = trainer.run_full_training()
        
        if success:
            print("✅ STAIRWAYS V4 200K TRAINING: SUCCESS")
            return True
        else:
            print("❌ STAIRWAYS V4 200K TRAINING: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training setup failed: {e}")
        print("❌ STAIRWAYS V4 200K TRAINING: SETUP FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)