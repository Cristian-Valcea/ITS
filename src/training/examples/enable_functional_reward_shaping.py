# src/training/examples/enable_functional_reward_shaping.py
"""
How to enable FUNCTIONAL reward shaping in your training process.

This script shows exactly what you need to change in your configuration
to make the reward shaping actually work (instead of just logging).
"""

import yaml
from pathlib import Path

def create_functional_reward_shaping_config():
    """Create a configuration that enables functional reward shaping."""
    
    config = {
        # Your existing training configuration
        'training': {
            'algorithm': 'RECURRENTPPO',
            'total_timesteps': 50000,
            'max_episodes': 20,
            'max_training_time_minutes': 15,
            'learning_rate': 0.0001,
            'n_steps': 256,
            'batch_size': 32,
            'n_epochs': 4,
            'verbose': 1
        },
        
        # Your existing environment configuration
        'environment': {
            'initial_capital': 100000.0,
            'lookback_window': 15,
            'max_daily_drawdown_pct': 0.02,
            'hourly_turnover_cap': 5.0,
            'transaction_cost_pct': 0.001,
            'reward_scaling': 1.0,
            'enable_kyle_lambda_fills': True
        },
        
        # 🔧 THIS IS THE KEY CHANGE: Enable functional reward shaping
        'risk': {
            # Your existing risk configuration
            'policy_yaml': 'config/risk_policy.yaml',
            'penalty_weight': 0.1,
            'early_stop_threshold': 0.8,
            
            # 🚨 ADD THIS: Enable functional reward shaping
            'reward_shaping': {
                'enabled': True,  # ✅ This makes it actually work!
                'penalty_weight': 0.1  # Weight for risk penalties
            }
        },
        
        # Your existing model configuration
        'model': {
            'policy': 'MlpLstmPolicy',
            'net_arch': [256, 256],
            'activation_fn': 'ReLU',
            'lstm_hidden_size': 64,
            'n_lstm_layers': 1
        },
        
        # Your existing features, data, etc.
        'features': {
            'calculators': ['RSI', 'EMA', 'VWAP', 'Time', 'ATR'],
            'lookback_window': 15,
            'max_indicator_lookback': 170
        },
        
        'data': {
            'symbol': 'NVDA',
            'start_date': '2024-01-01',
            'end_date': '2024-01-05',
            'interval': '1min'
        }
    }
    
    return config


def show_before_after_comparison():
    """Show the difference between broken and functional configurations."""
    
    print("🚨 BEFORE (BROKEN - Only logs penalties):")
    print("=" * 60)
    broken_config = """
risk:
  policy_yaml: 'config/risk_policy.yaml'
  penalty_weight: 0.1
  # ❌ No reward_shaping section = penalties are only logged!
"""
    print(broken_config)
    
    print("\n✅ AFTER (FUNCTIONAL - Actually applies penalties):")
    print("=" * 60)
    functional_config = """
risk:
  policy_yaml: 'config/risk_policy.yaml'
  penalty_weight: 0.1
  # ✅ ADD THIS to make it actually work:
  reward_shaping:
    enabled: true
    penalty_weight: 0.1  # This penalty is actually applied!
"""
    print(functional_config)


def create_example_training_script():
    """Show how to use the functional reward shaping in training."""
    
    script = '''
# example_training_with_functional_reward_shaping.py

from src.training.trainer_agent import TrainerAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv

# 1. Create your configuration with functional reward shaping enabled
config = {
    'training': {
        'algorithm': 'RECURRENTPPO',
        'total_timesteps': 50000,
        # ... your other training params
    },
    'risk': {
        'policy_yaml': 'config/risk_policy.yaml',
        'reward_shaping': {
            'enabled': True,  # ✅ This is the key!
            'penalty_weight': 0.1
        }
    },
    # ... rest of your config
}

# 2. Create trainer (will automatically apply reward shaping)
trainer = TrainerAgent(config)

# 3. Create your environment
env = IntradayTradingEnv(
    # ... your environment parameters
)

# 4. Set environment (reward shaping is applied automatically)
trainer.set_env(env)

# 5. Train (penalties are now actually applied to rewards!)
model_path = trainer.train()

print(f"✅ Training complete with FUNCTIONAL reward shaping!")
print(f"Model saved to: {model_path}")
'''
    
    return script


def main():
    """Show how to enable functional reward shaping."""
    
    print("🔧 HOW TO ENABLE FUNCTIONAL REWARD SHAPING")
    print("=" * 60)
    print("The RiskPenaltyCallback was broken - it only logged penalties.")
    print("Here's how to make it actually work:\n")
    
    # Show the configuration difference
    show_before_after_comparison()
    
    print("\n📝 COMPLETE EXAMPLE CONFIGURATION:")
    print("=" * 60)
    
    # Create and save example configuration
    config = create_functional_reward_shaping_config()
    config_path = Path("config/functional_reward_shaping_example.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Example configuration saved to: {config_path}")
    
    print("\n🐍 EXAMPLE TRAINING SCRIPT:")
    print("=" * 60)
    script = create_example_training_script()
    print(script)
    
    print("\n🎯 WHAT HAPPENS NOW:")
    print("=" * 60)
    print("✅ Risk penalties are actually applied to the reward signal")
    print("✅ Agent learns to avoid risky behavior")
    print("✅ Training is more stable and risk-aware")
    print("✅ Reward-P&L correlation improves")
    print("✅ Better alignment between training rewards and profitability")
    
    print("\n🔍 HOW TO VERIFY IT'S WORKING:")
    print("=" * 60)
    print("Look for these log messages during training:")
    print("  ✅ Applied functional reward shaping (weight: 0.1)")
    print("  ✅ Added FUNCTIONAL RiskPenaltyCallback (applies penalties to rewards)")
    print("  🔍 KYLE LAMBDA IMPACT: Mid=$100.00 → Fill=$100.05 (Impact: 5.0bps)")
    print("  💰 REWARD BREAKDOWN: Gross P&L: $0.001234, Penalty: $0.050000")


if __name__ == "__main__":
    main()