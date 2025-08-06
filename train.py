#!/usr/bin/env python3
"""
Unified Training Entry Point for Phase 2 OOS Training
Compatible with Team A Phase 2 requirements and existing training scripts
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def setup_logging():
    """Setup training session logging"""
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamped log file
    log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        raise

def create_output_dir(base_name: str, config: dict) -> str:
    """Create output directory for training run"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Include seed in directory name if specified
    seed_suffix = f"_seed{config.get('training', {}).get('seed', 0)}"
    
    output_dir = f"train_runs/{base_name}_{timestamp}{seed_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def run_dual_ticker_training(config: dict, output_dir: str, resume_path: str = None, use_governor: bool = False):
    """
    Run dual-ticker training with the specified configuration
    
    This function integrates the existing dual-ticker training pipeline
    with Phase 2 OOS requirements
    """
    
    logger = logging.getLogger("train")
    logger.info("🚀 Starting dual-ticker training...")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Resume from: {resume_path if resume_path else 'None (fresh training)'}")
    logger.info(f"   Governor enabled: {use_governor}")
    
    try:
        # Import training components
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        
        # Import our dual-ticker components
        from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
        from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
        
        # Extract configuration
        env_config = config.get('environment', {})
        training_config = config.get('training', {})
        ppo_config = config.get('ppo', {})
        reward_config = config.get('reward_system', {})
        
        logger.info("🏗️ Creating training environment...")
        
        # Create data adapter
        data_adapter = DualTickerDataAdapter()
        
        # Create environment
        env = DualTickerTradingEnv(
            data_adapter=data_adapter,
            symbols=env_config.get('symbols', ['NVDA', 'MSFT']),
            initial_capital=env_config.get('initial_capital', 10000.0),
            lookback_window=env_config.get('lookback_window', 50),
            max_episode_steps=env_config.get('max_episode_steps', 390),
            transaction_cost_pct=env_config.get('transaction_cost_pct', 0.001),
            max_drawdown_pct=env_config.get('max_drawdown_pct', 0.50)
        )
        
        # Add monitor wrapper for logging
        monitor_file = os.path.join(output_dir, "monitor.csv")
        env = Monitor(env, monitor_file)
        
        # Vectorize environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Add normalization
        vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True)
        
        logger.info("🤖 Initializing PPO model...")
        
        # Model configuration
        model_config = {
            'policy': 'MlpPolicy',
            'env': vec_env,
            'verbose': 1,
            'tensorboard_log': os.path.join(output_dir, 'tensorboard'),
            **ppo_config  # Merge PPO hyperparameters
        }
        
        # Create or load model
        if resume_path and os.path.exists(resume_path):
            logger.info(f"📂 Loading model from {resume_path}")
            
            # Load the model
            model_file = None
            if os.path.isdir(resume_path):
                # Find model file in directory
                for file in os.listdir(resume_path):
                    if file.endswith('.zip') and 'checkpoint' in file:
                        model_file = os.path.join(resume_path, file)
                        break
            else:
                model_file = resume_path
            
            if model_file and os.path.exists(model_file):
                model = RecurrentPPO.load(model_file, env=vec_env)
                logger.info(f"✅ Model loaded successfully from {model_file}")
            else:
                logger.warning(f"⚠️ Model file not found, creating fresh model")
                model = RecurrentPPO(**model_config)
        else:
            logger.info("🆕 Creating fresh model")
            model = RecurrentPPO(**model_config)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('save_interval', 2500),
            save_path=output_dir,
            name_prefix='checkpoint'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if training_config.get('eval_interval'):
            eval_env = Monitor(DualTickerTradingEnv(
                data_adapter=data_adapter,
                symbols=env_config.get('symbols', ['NVDA', 'MSFT']),
                initial_capital=env_config.get('initial_capital', 10000.0)
            ))
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=output_dir,
                log_path=output_dir,
                eval_freq=training_config.get('eval_interval', 2500),
                n_eval_episodes=training_config.get('eval_episodes', 10),
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Save configuration
        config_save_path = os.path.join(output_dir, 'training_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("🎯 Starting training...")
        logger.info(f"   Total timesteps: {training_config.get('total_timesteps', 50000)}")
        logger.info(f"   Save interval: {training_config.get('save_interval', 2500)}")
        logger.info(f"   Log interval: {training_config.get('log_interval', 10)}")
        
        # Run training
        model.learn(
            total_timesteps=training_config.get('total_timesteps', 50000),
            callback=callbacks,
            log_interval=training_config.get('log_interval', 10),
            reset_num_timesteps=False if resume_path else True
        )
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.zip')
        model.save(final_model_path)
        
        # Save vectorized environment stats
        vec_env.save(os.path.join(output_dir, 'vecnormalize.pkl'))
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Monitor file: {monitor_file}")
        logger.info(f"   Config: {config_save_path}")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'final_model_path': final_model_path,
            'monitor_file': monitor_file
        }
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full error traceback:")
        return {
            'success': False,
            'error': str(e),
            'output_dir': output_dir
        }

def main():
    """Main entry point for training script"""
    
    parser = argparse.ArgumentParser(description="Phase 2 OOS Training Script")
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration YAML')
    parser.add_argument('--steps', type=int, required=True,
                       help='Total training steps')
    
    # Optional arguments
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for training')
    parser.add_argument('--resume', type=str,
                       help='Path to model to resume from')
    parser.add_argument('--use-governor', action='store_true',
                       help='Enable risk governor integration')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')
    
    # Data split arguments (for seed-variance testing)
    parser.add_argument('--train-start', type=str,
                       help='Training data start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str,
                       help='Training data end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str,
                       help='Test data start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str,
                       help='Test data end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger("train")
    
    logger.info("=" * 60)
    logger.info("🎯 PHASE 2 OOS TRAINING SESSION")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Governor: {args.use_governor}")
    logger.info(f"Log file: {log_file}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        
        # Override configuration with command line arguments
        if 'training' not in config:
            config['training'] = {}
        
        config['training']['total_timesteps'] = args.steps
        config['training']['seed'] = args.seed
        
        # Override data dates if provided
        if 'environment' not in config:
            config['environment'] = {}
            
        if args.train_start:
            config['environment']['start_date'] = args.train_start
        if args.train_end:
            config['environment']['end_date'] = args.train_end
        if args.test_start:
            config['environment']['test_start_date'] = args.test_start
        if args.test_end:
            config['environment']['test_end_date'] = args.test_end
        
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Create output directory
    try:
        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            config_name = Path(args.config).stem
            output_dir = create_output_dir(config_name, config)
        
        logger.info(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create output directory: {e}")
        return 1
    
    # Run training
    result = run_dual_ticker_training(
        config=config,
        output_dir=output_dir,
        resume_path=args.resume,
        use_governor=args.use_governor
    )
    
    if result['success']:
        logger.info("✅ Training session completed successfully")
        
        # Print summary for Phase 2 evaluation
        print("\n" + "=" * 50)
        print("📊 TRAINING SESSION SUMMARY")
        print("=" * 50)
        print(f"Status: ✅ SUCCESS")
        print(f"Output Directory: {result['output_dir']}")
        print(f"Final Model: {result['final_model_path']}")
        print(f"Monitor File: {result['monitor_file']}")
        print(f"Steps Completed: {args.steps}")
        print(f"Seed: {args.seed}")
        print("=" * 50)
        
        return 0
    else:
        logger.error("❌ Training session failed")
        print(f"\n❌ TRAINING FAILED: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)