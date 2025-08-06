#!/usr/bin/env python3
"""
🎯 48K REAL DATA TRAINING LAUNCHER
Launch 48,000 timestep training run using real historical market data from database
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database_data():
    """Verify sufficient data is available in database"""
    import psycopg2
    
    try:
        # Use secure password from vault
        db_password = SecretsHelper.get_timescaledb_password()
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_data',
            user='postgres', 
            password=db_password
        )
        cur = conn.cursor()
        
        # Check data availability
        cur.execute("""
            SELECT symbol, COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date
            FROM minute_bars 
            WHERE symbol IN ('NVDA', 'MSFT')
            GROUP BY symbol
            ORDER BY symbol;
        """)
        
        data = cur.fetchall()
        total_records = sum(row[1] for row in data)
        
        logger.info("📊 Database Data Summary:")
        for row in data:
            logger.info(f"  {row[0]}: {row[1]:,} records from {row[2]} to {row[3]}")
        logger.info(f"  Total: {total_records:,} records")
        
        # Minimum requirement for 48K training (need ~50K+ records for proper episodes)
        if total_records < 50000:
            logger.warning(f"⚠️ Insufficient data: {total_records:,} records (need 50K+ for 48K training)")
            return False, total_records
        
        logger.info(f"✅ Sufficient data available: {total_records:,} records")
        return True, total_records
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False, 0
    finally:
        if 'conn' in locals():
            conn.close()

def create_training_config(total_steps=48000):
    """Create training configuration for real data"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"real_data_48k_{timestamp}"
    
    config = {
        # Training Configuration
        'total_timesteps': total_steps,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 2048,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Environment Configuration  
        'symbols': ['NVDA', 'MSFT'],
        'initial_cash': 10000.0,
        'max_position_size': 1000.0,
        'transaction_cost': 0.001,
        'lookback_window': 50,
        'episode_length': 1000,
        
        # Data Configuration
        'data_source': 'timescaledb',
        'start_date': '2022-01-03',
        'end_date': '2024-12-31', 
        'bar_size': '1min',
        'data_split': 'train',
        
        # Model Configuration
        'policy': 'MlpPolicy',
        'policy_kwargs': {
            'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
            'activation_fn': 'torch.nn.ReLU'
        },
        
        # Output Configuration
        'run_name': run_name,
        'save_path': f'train_runs/{run_name}',
        'tensorboard_log': f'tensorboard_logs/{run_name}',
        'checkpoint_freq': 10000,
        'eval_freq': 5000,
        'eval_episodes': 10,
        
        # Real Data Training Flags
        'use_real_data': True,
        'real_data_validation': True,
        'market_hours_only': True
    }
    
    return config

def launch_training():
    """Launch the 48K real data training run"""
    
    logger.info("🚀 48K REAL DATA TRAINING LAUNCHER")
    logger.info("=" * 60)
    
    # Step 1: Check data availability
    logger.info("📊 Step 1: Checking database data availability...")
    data_available, record_count = check_database_data()
    
    if not data_available:
        logger.error("❌ Training cannot proceed - insufficient data")
        logger.info("💡 Please wait for historical data fetch to complete")
        return False
    
    # Step 2: Create training configuration
    logger.info("⚙️ Step 2: Creating training configuration...")
    config = create_training_config()
    logger.info(f"  Run name: {config['run_name']}")
    logger.info(f"  Total timesteps: {config['total_timesteps']:,}")
    logger.info(f"  Data records: {record_count:,}")
    
    # Step 3: Initialize database connection for adapter
    logger.info("🔧 Step 3: Configuring data adapter...")
    try:
        # Use secure password from vault
        db_password = SecretsHelper.get_timescaledb_password()
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': db_password
        }
        
        # Test adapter initialization
        adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
        logger.info("✅ Data adapter configured successfully")
        
    except Exception as e:
        logger.error(f"❌ Data adapter configuration failed: {e}")
        return False
    
    # Step 4: Launch training
    logger.info("🎯 Step 4: Launching training run...")
    
    # Create directories
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(config['tensorboard_log'], exist_ok=True)
    
    # Launch training script
    training_script = f"""
import sys
sys.path.insert(0, '{project_root}')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {db_config}

# Create environment with real data
logger.info("Creating training environment...")
adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)

# Load real market data
logger.info("Loading real market data...")
market_data = adapter.load_training_data(
    start_date='{config['start_date']}',
    end_date='{config['end_date']}',
    symbols={config['symbols']},
    bar_size='{config['bar_size']}',
    data_split='{config['data_split']}'
)

# Extract data arrays for environment
processed_feature_data = market_data['features']
processed_price_data = market_data['prices'] 
trading_days = market_data['trading_days']

logger.info(f"Loaded data shapes: features {processed_feature_data.shape}, prices {processed_price_data.shape}")

# Create environment with real data
env = DualTickerTradingEnvV3Enhanced(
    processed_feature_data=processed_feature_data,
    processed_price_data=processed_price_data,
    trading_days=trading_days,
    initial_capital={config['initial_cash']},
    lookback_window={config['lookback_window']},
    max_episode_steps={config['episode_length']},
    transaction_cost_pct={config['transaction_cost']}
)

# Wrap environment
env = DummyVecEnv([lambda: env])

# Create model
logger.info("Creating PPO model...")
model = PPO(
    policy='{config['policy']}',
    env=env,
    learning_rate={config['learning_rate']},
    n_steps={config['n_steps']},
    batch_size={config['batch_size']},
    n_epochs={config['n_epochs']},
    gamma={config['gamma']},
    gae_lambda={config['gae_lambda']},
    clip_range={config['clip_range']},
    ent_coef={config['ent_coef']},
    vf_coef={config['vf_coef']},
    max_grad_norm={config['max_grad_norm']},
    policy_kwargs={config['policy_kwargs']},
    tensorboard_log='{config['tensorboard_log']}',
    verbose=1
)

# Create callbacks
checkpoint_callback = CheckpointCallback(
    save_freq={config['checkpoint_freq']},
    save_path='{config['save_path']}/checkpoints/',
    name_prefix='real_data_48k_model'
)

# Train model
logger.info("🚀 Starting {config['total_timesteps']:,} timestep training...")
model.learn(
    total_timesteps={config['total_timesteps']},
    callback=[checkpoint_callback],
    tb_log_name='{config['run_name']}'
)

# Save final model
final_model_path = '{config['save_path']}/final_model.zip'
model.save(final_model_path)
logger.info(f"✅ Training complete! Model saved to {{final_model_path}}")

# Save training config
import json
config_path = '{config['save_path']}/training_config.json'
with open(config_path, 'w') as f:
    json.dump({config}, f, indent=2, default=str)
logger.info(f"📊 Config saved to {{config_path}}")
"""
    
    # Write and execute training script
    script_path = f"{config['save_path']}/train_real_data.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    logger.info(f"📝 Training script written to: {script_path}")
    logger.info("🎯 Launching training process...")
    
    # Execute training
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ Training completed successfully!")
            return True
        else:
            logger.error(f"❌ Training failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training execution failed: {e}")
        return False

if __name__ == "__main__":
    success = launch_training()
    if success:
        print("✅ 48K Real Data Training: SUCCESS")
        sys.exit(0)
    else:
        print("❌ 48K Real Data Training: FAILED") 
        sys.exit(1)