#!/usr/bin/env python3
"""
Example: Risk-Aware RL Model Training

Demonstrates the new production-grade TrainerAgent with:
- Risk-aware training callbacks
- TorchScript policy bundle export
- Latency SLO validation
- Clean SB3 integration
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainer_agent import create_trainer_agent
from gym_env.intraday_trading_env import IntradayTradingEnv
from shared.constants import CLOSE, MODEL_VERSION_FORMAT
from datetime import datetime


def create_mock_training_data(num_steps: int = 1000, num_features: int = 5):
    """Create mock market data for training."""
    # Generate mock market features
    market_features = np.random.randn(num_steps, num_features).astype(np.float32)
    
    # Generate mock price series
    prices = 100 + np.cumsum(np.random.randn(num_steps) * 0.1)
    dates = pd.date_range(start='2023-01-01', periods=num_steps, freq='1min')
    price_series = pd.Series(prices, index=dates, name=CLOSE)
    
    return market_features, price_series


def create_training_config():
    """Create training configuration."""
    return {
        'model_save_dir': 'models/risk_aware_training',
        'log_dir': 'logs/risk_aware_training',
        
        # Algorithm configuration
        'algorithm': 'DQN',
        'algo_params': {
            'policy': 'MlpPolicy',
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'target_update_interval': 500,
            'exploration_fraction': 0.3,
            'exploration_final_eps': 0.05,
            'verbose': 1,
            'seed': 42,
        },
        
        # Training parameters
        'training_params': {
            'total_timesteps': 50000,
            'log_interval': 100,
            'checkpoint_freq': 5000,
            'use_eval_callback': False,  # Disable for this example
        },
        
        # Risk-aware training configuration
        'risk_config': {
            'enabled': True,
            'policy_yaml': Path('config/risk_limits.yaml'),  # Will create if missing
            'penalty_weight': 0.1,
            'early_stop_threshold': 0.8,
            'log_freq': 500,
        }
    }


def create_minimal_risk_config():
    """Create minimal risk configuration for demo."""
    risk_config = {
        'calculators': {
            'drawdown': {
                'enabled': True,
                'config': {
                    'lookback_window': 100,
                    'max_drawdown_threshold': 0.05
                }
            },
            'turnover': {
                'enabled': True,
                'config': {
                    'lookback_window': 50,
                    'max_turnover_threshold': 2.0
                }
            }
        },
        'policies': [{
            'policy_id': 'training_policy',
            'policy_name': 'Training Risk Policy',
            'rules': [{
                'rule_id': 'max_drawdown',
                'rule_type': 'threshold',
                'metric': 'current_drawdown',
                'threshold': 0.05,
                'action': 'ALERT'
            }]
        }],
        'active_policy': 'training_policy'
    }
    return risk_config


def main():
    """Main training example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("🚀 RISK-AWARE RL TRAINING EXAMPLE")
    print("=" * 60)
    
    try:
        # Create mock training data
        logger.info("Creating mock training data...")
        market_features, price_series = create_mock_training_data(
            num_steps=2000, 
            num_features=6
        )
        
        # Create training environment
        logger.info("Setting up training environment...")
        env = IntradayTradingEnv(
            processed_feature_data=market_features,
            price_data=price_series,
            initial_capital=100000,
            lookback_window=1,
            max_daily_drawdown_pct=0.05,
            transaction_cost_pct=0.001,
            max_episode_steps=200,
            log_trades=True
        )
        
        logger.info(f"Environment created:")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space: {env.action_space}")
        
        # Create training configuration
        config = create_training_config()
        
        # Create minimal risk config if it doesn't exist
        risk_config_path = config['risk_config']['policy_yaml']
        if not risk_config_path.exists():
            logger.info(f"Creating minimal risk config: {risk_config_path}")
            risk_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(risk_config_path, 'w') as f:
                yaml.dump(create_minimal_risk_config(), f, default_flow_style=False)
        
        # Create trainer agent
        logger.info("Creating TrainerAgent...")
        trainer = create_trainer_agent(config)
        
        # Start training
        logger.info("Starting risk-aware training...")
        model_bundle_path = trainer.run(env)
        
        if model_bundle_path:
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"   Model bundle saved: {model_bundle_path}")
            
            # Validate the saved bundle
            bundle_path = Path(model_bundle_path)
            if (bundle_path / "policy.pt").exists():
                logger.info("   ✅ TorchScript policy exported")
            if (bundle_path / "metadata.json").exists():
                logger.info("   ✅ Metadata saved")
            
            # Load and test the policy
            logger.info("Testing saved policy bundle...")
            try:
                from training.policies.sb3_policy import TorchScriptPolicy
                
                policy = TorchScriptPolicy.load_bundle(bundle_path)
                
                # Test prediction
                test_obs = env.reset()
                if isinstance(test_obs, tuple):
                    test_obs = test_obs[0]
                
                action, info = policy.predict(test_obs)
                logger.info(f"   ✅ Policy prediction test: action={action}, info={info}")
                
                # Validate latency SLO
                latency_stats = policy.validate_prediction_latency(test_obs, num_trials=50)
                logger.info(f"   📊 Latency validation:")
                logger.info(f"      Mean: {latency_stats['mean_latency_us']:.1f}µs")
                logger.info(f"      P99: {latency_stats['p99_latency_us']:.1f}µs")
                logger.info(f"      SLO violations: {latency_stats['slo_violation_rate']:.2%}")
                
                if latency_stats['p99_latency_us'] < 100.0:
                    logger.info("   ✅ Policy meets production latency SLO")
                else:
                    logger.warning("   ⚠️ Policy may not meet production latency SLO")
                
            except Exception as e:
                logger.error(f"   ❌ Policy testing failed: {e}")
            
        else:
            logger.error("❌ Training failed")
            return 1
        
        # Cleanup
        env.close()
        
        print("\n" + "=" * 60)
        print("🎯 TRAINING EXAMPLE COMPLETED")
        print(f"📁 Model bundle: {model_bundle_path}")
        print("📊 Check logs/risk_aware_training/ for TensorBoard logs")
        print("🔧 Use the policy bundle in production with ExecutionAgent")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())