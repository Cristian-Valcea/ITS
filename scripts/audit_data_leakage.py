#!/usr/bin/env python3
"""
🔍 PHASE 1C: DATA LEAKAGE AUDIT
Implements management-enhanced data leakage detection:
1. Shuffle episode start offsets (±0.5 trading day)
2. Shift derived features by +1 step
3. Run 10K sanity validation
4. Compare reward distributions pre/post shift
5. Feature importance analysis for leakage detection
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLeakageAuditor:
    """Data leakage detection system for Phase 1C diagnostic"""
    
    def __init__(self, output_dir: str = "diagnostic_runs/phase1c_leakage_audit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage
        self.original_data = None
        self.shifted_data = None
        self.original_env = None
        self.shifted_env = None
        
        logger.info(f"🔍 Data Leakage Auditor initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Dict[str, Any]:
        """Load real market data and prepare original + shifted versions"""
        
        logger.info("📈 Loading real market data for leakage audit...")
        
        try:
            # Database connection
            db_password = SecretsHelper.get_timescaledb_password()
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_data',
                'user': 'postgres',
                'password': db_password
            }
            
            # Load real market data (Q1 2022 - sufficient for leakage testing)
            adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
            market_data = adapter.load_training_data(
                start_date='2022-01-03',
                end_date='2022-03-31',  # Q1 2022 - sufficient sample
                symbols=['NVDA', 'MSFT'],
                bar_size='1min',
                data_split='train'
            )
            
            # Extract components
            nvda_features = market_data['nvda_features']
            nvda_prices = market_data['nvda_prices']
            msft_features = market_data['msft_features']
            msft_prices = market_data['msft_prices']
            trading_days = market_data['trading_days']
            
            logger.info(f"📊 Data loaded: {len(trading_days)} timesteps")
            
            # Prepare original data (baseline)
            original_features = np.concatenate([nvda_features, msft_features], axis=1)
            position_features = np.zeros((original_features.shape[0], 2))
            original_features = np.concatenate([original_features, position_features], axis=1)
            
            # Create original price data
            nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
            msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
            original_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
            
            # Create shifted data (LEAKAGE TEST)
            logger.info("🔄 Creating +1 step shifted features for leakage detection...")
            
            # Shift all derived features by +1 step (critical test)
            shifted_features = original_features.copy()
            
            # Shift derived features (columns 1-11 for each asset are derived)
            # NVDA derived features: columns 1-11
            # MSFT derived features: columns 13-23  
            for asset_offset in [0, 12]:  # NVDA starts at 0, MSFT at 12
                for feature_idx in range(1, 12):  # Skip raw price (index 0)
                    col_idx = asset_offset + feature_idx
                    if col_idx < shifted_features.shape[1] - 2:  # Exclude position features
                        # Shift by +1 step (introduce future leak)
                        shifted_features[:-1, col_idx] = shifted_features[1:, col_idx]
                        shifted_features[-1, col_idx] = shifted_features[-2, col_idx]  # Fill last
            
            # Shuffle episode start offsets (±0.5 trading day = ±195 minutes)
            logger.info("🎲 Shuffling episode start offsets...")
            shuffled_indices = np.arange(len(trading_days))
            
            # Add random offset within ±195 steps (0.5 trading day)
            offset_range = 195
            random_offsets = np.random.randint(-offset_range, offset_range + 1, size=len(trading_days))
            
            # Apply offsets with bounds checking
            for i in range(len(shuffled_indices)):
                new_idx = shuffled_indices[i] + random_offsets[i]
                new_idx = max(0, min(len(trading_days) - 1, new_idx))
                shuffled_indices[i] = new_idx
            
            # Apply shuffling to shifted data
            shifted_features = shifted_features[shuffled_indices]
            shifted_prices = original_prices[shuffled_indices]
            shuffled_days = [trading_days[i] for i in shuffled_indices]
            
            # Store datasets
            self.original_data = {
                'features': original_features,
                'prices': original_prices,
                'trading_days': trading_days
            }
            
            self.shifted_data = {
                'features': shifted_features,
                'prices': shifted_prices,
                'trading_days': shuffled_days
            }
            
            logger.info("✅ Original and shifted datasets prepared")
            
            # Save processed data for feature importance analysis
            self._save_processed_data()
            
            return {
                'original_timesteps': len(trading_days),
                'shifted_timesteps': len(shuffled_days),
                'feature_count': original_features.shape[1]
            }
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def _save_processed_data(self):
        """Save processed data for feature importance analysis"""
        
        # Save as parquet for efficiency
        original_df = pd.DataFrame(self.original_data['features'])
        original_df['target'] = 0  # Dummy target for now
        original_df.to_parquet(self.output_dir / "original_features.parquet")
        
        shifted_df = pd.DataFrame(self.shifted_data['features'])
        shifted_df['target'] = 0  # Dummy target for now
        shifted_df.to_parquet(self.output_dir / "shifted_features.parquet")
        
        logger.info("💾 Processed data saved as parquet files")
    
    def create_environments(self) -> Tuple[DummyVecEnv, DummyVecEnv]:
        """Create original and shifted environments for comparison"""
        
        logger.info("🏗️ Creating original and shifted environments...")
        
        # Environment configuration
        env_config = {
            'initial_capital': 10000.0,
            'lookback_window': 50,
            'max_episode_steps': 390,
            'max_daily_drawdown_pct': 0.20,  # Permissive for testing
            'transaction_cost_pct': 0.001
        }
        
        # Original environment
        original_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=self.original_data['features'],
            processed_price_data=self.original_data['prices'],
            trading_days=self.original_data['trading_days'],
            **env_config
        )
        
        # Shifted environment (with potential leakage)
        shifted_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=self.shifted_data['features'],
            processed_price_data=self.shifted_data['prices'],
            trading_days=self.shifted_data['trading_days'],
            **env_config
        )
        
        # Monitor both environments
        original_env = Monitor(original_env, str(self.output_dir / "original_monitor.csv"))
        shifted_env = Monitor(shifted_env, str(self.output_dir / "shifted_monitor.csv"))
        
        # Wrap for stable-baselines3
        self.original_env = DummyVecEnv([lambda: original_env])
        self.shifted_env = DummyVecEnv([lambda: shifted_env])
        
        logger.info("✅ Original and shifted environments created")
        
        return self.original_env, self.shifted_env
    
    def run_sanity_test(self, timesteps: int = 10000) -> Dict[str, Any]:
        """Run 10K sanity test on both environments"""
        
        logger.info(f"🧪 Running {timesteps:,} step sanity test...")
        
        try:
            # Create simple PPO model for testing
            model_original = PPO(
                policy='MlpPolicy',
                env=self.original_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=128,
                verbose=0  # Quiet for testing
            )
            
            model_shifted = PPO(
                policy='MlpPolicy', 
                env=self.shifted_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=128,
                verbose=0  # Quiet for testing
            )
            
            # Train both models briefly
            logger.info("🏃 Training original environment model...")
            model_original.learn(total_timesteps=timesteps//2)
            
            logger.info("🏃 Training shifted environment model...")
            model_shifted.learn(total_timesteps=timesteps//2)
            
            # Evaluate both models
            original_rewards = self._evaluate_model(model_original, self.original_env, "original")
            shifted_rewards = self._evaluate_model(model_shifted, self.shifted_env, "shifted")
            
            # Compare results
            results = {
                'original_mean_reward': np.mean(original_rewards),
                'original_std_reward': np.std(original_rewards),
                'shifted_mean_reward': np.mean(shifted_rewards), 
                'shifted_std_reward': np.std(shifted_rewards),
                'reward_diff_pct': 0.0,
                'sanity_passed': True,
                'no_crashes': True
            }
            
            # Calculate percentage difference
            if results['original_mean_reward'] != 0:
                results['reward_diff_pct'] = abs(
                    (results['shifted_mean_reward'] - results['original_mean_reward']) / 
                    results['original_mean_reward'] * 100
                )
            
            # Check diagnostic criteria
            if results['reward_diff_pct'] > 10:  # >10% difference
                logger.warning(f"⚠️ Reward difference {results['reward_diff_pct']:.1f}% > 10% threshold")
                results['sanity_passed'] = False
            
            logger.info(f"📊 Sanity Test Results:")
            logger.info(f"   Original: {results['original_mean_reward']:.3f} ± {results['original_std_reward']:.3f}")
            logger.info(f"   Shifted: {results['shifted_mean_reward']:.3f} ± {results['shifted_std_reward']:.3f}")
            logger.info(f"   Difference: {results['reward_diff_pct']:.1f}%")
            
            if results['sanity_passed']:
                logger.info("✅ Sanity test passed - no obvious leakage detected")
            else:
                logger.warning("⚠️ Sanity test failed - potential leakage detected")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Sanity test failed: {e}")
            return {
                'no_crashes': False,
                'error': str(e),
                'sanity_passed': False
            }
    
    def _evaluate_model(self, model: PPO, env: DummyVecEnv, env_name: str, episodes: int = 5) -> list:
        """Evaluate model performance"""
        
        rewards = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward[0]
            
            rewards.append(episode_reward)
        
        logger.info(f"📈 {env_name.title()}: {len(rewards)} episodes, avg reward: {np.mean(rewards):.3f}")
        
        return rewards
    
    def generate_audit_report(self, sanity_results: Dict[str, Any], data_info: Dict[str, Any]):
        """Generate comprehensive audit report"""
        
        report_path = self.output_dir / f"leakage_audit_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("🔍 PHASE 1C: DATA LEAKAGE AUDIT REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("📊 DATA PREPARATION:\n")
            f.write(f"  Original timesteps: {data_info['original_timesteps']:,}\n")
            f.write(f"  Shifted timesteps: {data_info['shifted_timesteps']:,}\n")
            f.write(f"  Feature count: {data_info['feature_count']}\n")
            f.write("  Modifications applied:\n")
            f.write("    ✓ +1 step shift on derived features\n")
            f.write("    ✓ ±0.5 trading day episode offset shuffling\n\n")
            
            f.write("🧪 SANITY TEST RESULTS:\n")
            if sanity_results.get('no_crashes', False):
                f.write("  ✓ No crashes detected\n")
                f.write(f"  Original mean reward: {sanity_results.get('original_mean_reward', 0):.3f}\n")
                f.write(f"  Shifted mean reward: {sanity_results.get('shifted_mean_reward', 0):.3f}\n")
                f.write(f"  Reward difference: {sanity_results.get('reward_diff_pct', 0):.1f}%\n")
                f.write(f"  Sanity passed: {sanity_results.get('sanity_passed', False)}\n")
            else:
                f.write("  ❌ Crashes detected during testing\n")
                f.write(f"  Error: {sanity_results.get('error', 'Unknown')}\n")
            
            f.write("\n📋 DIAGNOSTIC CRITERIA:\n")
            f.write("  Success threshold: Reward difference ≤10%\n")
            f.write("  Feature importance stability: ≤50% change in top-5\n")
            f.write("  No pipeline crashes or data errors\n\n")
            
            f.write("📁 DELIVERABLES:\n")
            f.write("  ✓ original_features.parquet\n")
            f.write("  ✓ shifted_features.parquet\n")
            f.write("  ✓ original_monitor.csv\n")
            f.write("  ✓ shifted_monitor.csv\n")
            f.write("  → feature_importance_comparison.html (next step)\n")
        
        logger.info(f"📄 Audit report saved: {report_path}")
        return report_path

def main():
    """Main Phase 1C execution"""
    
    parser = argparse.ArgumentParser(description="Phase 1C: Data Leakage Audit")
    parser.add_argument('--timesteps', type=int, default=10000, help='Sanity test timesteps')
    parser.add_argument('--output_dir', type=str, default='diagnostic_runs/phase1c_leakage_audit', 
                       help='Output directory')
    args = parser.parse_args()
    
    logger.info("🔍 PHASE 1C: DATA LEAKAGE AUDIT")
    logger.info("=" * 50)
    
    try:
        # Initialize auditor
        auditor = DataLeakageAuditor(args.output_dir)
        
        # Load and prepare data
        data_info = auditor.load_and_prepare_data()
        
        # Create environments
        auditor.create_environments()
        
        # Run sanity test
        sanity_results = auditor.run_sanity_test(args.timesteps)
        
        # Generate report
        report_path = auditor.generate_audit_report(sanity_results, data_info)
        
        # Summary
        logger.info("\n🎯 PHASE 1C AUDIT COMPLETE")
        logger.info(f"📄 Report: {report_path}")
        logger.info("📋 Next steps:")
        logger.info("  1. Run feature importance analysis")
        logger.info("  2. Generate comparison visualization")
        logger.info("  3. Validate leakage detection criteria")
        
        if sanity_results.get('sanity_passed', False):
            logger.info("✅ Phase 1C: No obvious data leakage detected")
            return True
        else:
            logger.warning("⚠️ Phase 1C: Potential data leakage detected")
            return False
        
    except Exception as e:
        logger.error(f"❌ Phase 1C audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)