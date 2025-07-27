#!/usr/bin/env python3
"""
Curriculum Learning Training CLI

Run training with curriculum learning enabled. The system will automatically
progress through stages of increasing risk control strictness.

Examples:
    # Basic curriculum training
    python scripts/run_curriculum_training.py --symbol AAPL --days 10
    
    # Custom curriculum with 3 stages
    python scripts/run_curriculum_training.py --symbol NVDA --stages 3 --episodes 150
    
    # Performance-based curriculum advancement
    python scripts/run_curriculum_training.py --symbol TSLA --perf-triggers --days 7
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CurriculumTrainingCLI")

def create_curriculum_stages(num_stages: int, total_episodes: int, perf_triggers: bool):
    """Create curriculum stages configuration."""
    episodes_per_stage = total_episodes // num_stages
    
    stages = []
    
    for i in range(num_stages):
        start_episode = i * episodes_per_stage
        end_episode = (i + 1) * episodes_per_stage - 1
        if i == num_stages - 1:  # Last stage
            end_episode = total_episodes - 1
        
        # Progressive tightening
        dd_limit = 0.05 - (i * 0.01)  # 5% -> 4% -> 3% -> 2%
        penalty_lambda = i * 0.15     # 0.0 -> 0.15 -> 0.3 -> 0.45
        
        stage = {
            'name': f'stage_{i+1}',
            'episodes': [start_episode, end_episode],
            'dd_limit': max(dd_limit, 0.02),  # Minimum 2%
            'penalty_lambda': min(penalty_lambda, 0.5)  # Maximum 0.5
        }
        
        # Add performance conditions if enabled
        if perf_triggers:
            min_episodes = max(5, episodes_per_stage // 4)
            min_sharpe = 0.3 + (i * 0.2)  # 0.3 -> 0.5 -> 0.7 -> 0.9
            
            stage['advance_conditions'] = {
                'min_episodes': min_episodes,
                'min_sharpe': min_sharpe,
                'max_drawdown': dd_limit + 0.005  # Slightly above stage limit
            }
        
        stages.append(stage)
    
    return stages

def create_curriculum_config(args):
    """Create curriculum configuration from CLI arguments."""
    stages = create_curriculum_stages(args.stages, args.episodes, args.perf_triggers)
    
    curriculum_config = {
        'enabled': True,
        'use_perf_trigger': args.perf_triggers,
        'logic': 'and',
        'stages': stages
    }
    
    logger.info(f"🎓 Curriculum Configuration:")
    logger.info(f"   Stages: {args.stages}")
    logger.info(f"   Total Episodes: {args.episodes}")
    logger.info(f"   Performance Triggers: {'Enabled' if args.perf_triggers else 'Disabled'}")
    
    for i, stage in enumerate(stages):
        logger.info(f"   Stage {i+1} ({stage['name']}): "
                   f"episodes {stage['episodes'][0]}-{stage['episodes'][1]}, "
                   f"dd_limit={stage['dd_limit']:.1%}, λ={stage['penalty_lambda']}")
    
    return curriculum_config

def update_config_for_curriculum(curriculum_config, training_steps):
    """Update main config with curriculum settings."""
    config_path = PROJECT_ROOT / "config" / "main_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update risk section with curriculum
    if 'risk' not in config:
        config['risk'] = {}
    
    config['risk'].update({
        'vol_window': 60,
        'penalty_lambda': 0.0,  # Will be overridden by curriculum
        'dd_limit': 0.05,       # Will be overridden by curriculum
        'eval_dd_limit': 0.02,
        'include_risk_features': True,
        'curriculum': curriculum_config
    })
    
    # Update training configuration
    config['training']['total_timesteps'] = training_steps
    config['training']['episodes'] = curriculum_config['stages'][-1]['episodes'][1] + 1
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("📝 Updated main_config.yaml with curriculum settings")

def run_curriculum_training(symbol, start_date, end_date, interval):
    """Run training with curriculum learning."""
    logger.info(f"🎓 Starting curriculum learning training for {symbol}...")
    
    try:
        from src.execution.orchestrator_agent import OrchestratorAgent
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(
            main_config_path="config/main_config.yaml",
            model_params_config_path="config/model_params.yaml"
        )
        
        # Run training pipeline
        result = orchestrator.run_training_pipeline(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cached_data=True,
            run_evaluation_after_train=True
        )
        
        if result:
            logger.info(f"✅ Curriculum training completed successfully!")
            logger.info(f"📁 Model saved at: {result}")
            
            # Log curriculum completion summary
            logger.info("🎓 Curriculum Learning Summary:")
            logger.info("   Check logs above for stage transitions and performance metrics")
            
            return True
        else:
            logger.error("❌ Curriculum training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Curriculum training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic parameters
    parser.add_argument('--symbol', '-s', default='AAPL', help='Stock symbol to trade')
    parser.add_argument('--days', '-d', type=int, default=10, help='Number of days of data to use')
    parser.add_argument('--interval', '-i', default='1min', help='Data interval (1min, 5min, etc.)')
    
    # Curriculum parameters
    parser.add_argument('--stages', type=int, default=4, 
                       help='Number of curriculum stages (default: 4)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Total episodes for curriculum (default: 100)')
    parser.add_argument('--perf-triggers', action='store_true',
                       help='Enable performance-based stage advancement')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=200000,
                       help='Training timesteps (default: 200000)')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with 5000 steps and 20 episodes')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.steps = 5000
        args.episodes = 20
        args.stages = 3
        logger.info("🏃 Quick mode: 5000 steps, 20 episodes, 3 stages")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"📅 Training period: {start_date_str} to {end_date_str}")
    
    # Create and apply curriculum configuration
    curriculum_config = create_curriculum_config(args)
    update_config_for_curriculum(curriculum_config, args.steps)
    
    # Run curriculum training
    success = run_curriculum_training(args.symbol, start_date_str, end_date_str, args.interval)
    
    if success:
        logger.info("🎉 Curriculum learning training completed successfully!")
        logger.info("🎓 The agent has progressed through all curriculum stages!")
        logger.info("📊 Check TensorBoard logs for curriculum progression metrics")
    else:
        logger.error("❌ Curriculum training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()