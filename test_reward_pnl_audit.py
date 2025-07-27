#!/usr/bin/env python3
"""
Test script for the Reward-P&L Audit System

This script demonstrates how to use the RewardPnLAudit callback to ensure
your agent's reward signal aligns with actual P&L, preventing the common
"looks-good-in-training, bad-in-cash" problem.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reward_pnl_audit import RewardPnLAudit, quick_audit_check

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockEnvironment:
    """Mock trading environment for testing the audit system."""
    
    def __init__(self, scenario: str = "aligned"):
        self.scenario = scenario
        self.step_count = 0
        self.max_steps = 100
        self.portfolio_value = 10000.0
        self.position = 0
        self.entry_price = 100.0
        
        # Generate price series
        np.random.seed(42)
        self.prices = 100 + np.cumsum(np.random.normal(0, 0.5, self.max_steps))
        
    def reset(self):
        self.step_count = 0
        self.portfolio_value = 10000.0
        self.position = 0
        return np.random.random(10)  # Mock observation
    
    def step(self, action):
        if self.step_count >= self.max_steps:
            return np.random.random(10), 0, True, True, {}
        
        current_price = self.prices[self.step_count]
        
        # Simulate trading logic
        realized_pnl_step = 0.0
        fees_step = 0.0
        
        if action != 1:  # Not hold
            if self.position != 0:  # Close existing position
                if self.position == 1:  # Close long
                    realized_pnl_step = (current_price - self.entry_price) * 100
                else:  # Close short
                    realized_pnl_step = (self.entry_price - current_price) * 100
                fees_step = current_price * 100 * 0.001  # 0.1% fee
                self.position = 0
            
            if action == 0:  # New short
                self.position = -1
                self.entry_price = current_price
            elif action == 2:  # New long
                self.position = 1
                self.entry_price = current_price
        
        # Calculate unrealized P&L
        unrealized_pnl_step = 0.0
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl_step = (current_price - self.entry_price) * 100
            else:
                unrealized_pnl_step = (self.entry_price - current_price) * 100
        
        # Update portfolio
        net_pnl_step = realized_pnl_step - fees_step
        self.portfolio_value += net_pnl_step
        
        # Generate reward based on scenario
        if self.scenario == "aligned":
            # Reward perfectly aligned with P&L
            reward = net_pnl_step / 1000.0  # Scale down
        elif self.scenario == "misaligned":
            # Reward not aligned with P&L (common mistake)
            reward = np.random.normal(0, 0.1)  # Random reward
        elif self.scenario == "partially_aligned":
            # Partially aligned (some correlation but not perfect)
            reward = 0.7 * (net_pnl_step / 1000.0) + 0.3 * np.random.normal(0, 0.05)
        else:
            reward = 0.0
        
        # Create info dictionary with all required fields
        info = {
            'realized_pnl_step': realized_pnl_step,
            'unrealized_pnl_step': unrealized_pnl_step,
            'total_pnl_step': realized_pnl_step + unrealized_pnl_step,
            'fees_step': fees_step,
            'net_pnl_step': net_pnl_step,
            'raw_reward': reward,
            'scaled_reward': reward * 1.0,  # No scaling in this test
            'portfolio_value': self.portfolio_value,
            'timestamp': pd.Timestamp.now() + pd.Timedelta(minutes=self.step_count)
        }
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return np.random.random(10), reward, done, False, info


class MockCallback:
    """Mock callback to simulate SB3 callback interface."""
    
    def __init__(self, audit_callback):
        self.audit_callback = audit_callback
        self.locals = {}
        self.globals = {}
    
    def simulate_training(self, env, episodes=10):
        """Simulate training episodes."""
        logger.info(f"🚀 Simulating {episodes} episodes with {env.scenario} reward scenario...")
        
        # Initialize callback
        self.audit_callback.init_callback(None)
        self.audit_callback.locals = {}
        self.audit_callback.globals = {}
        self.audit_callback._on_training_start()
        
        for episode in range(episodes):
            obs = env.reset()
            done = False
            episode_rewards = []
            episode_infos = []
            episode_actions = []
            
            while not done:
                # Simple policy: random actions with some bias
                action = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
                obs, reward, done, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                episode_infos.append(info)
                episode_actions.append(action)
                
                # Update callback locals
                self.audit_callback.locals = {
                    'rewards': [reward],
                    'infos': [info],
                    'actions': [action]
                }
                
                self.audit_callback._on_step()
            
            # End of episode
            self.audit_callback._on_rollout_end()
            
            if episode % 5 == 0:
                logger.info(f"Completed episode {episode + 1}/{episodes}")
        
        # End of training
        self.audit_callback._on_training_end()
        logger.info("✅ Training simulation completed!")


def test_aligned_scenario():
    """Test with perfectly aligned reward-P&L scenario."""
    logger.info("\n" + "="*60)
    logger.info("🎯 Testing ALIGNED Reward Scenario")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_callback = RewardPnLAudit(
            output_dir=temp_dir + "/aligned_test",
            min_correlation_threshold=0.7,
            verbose=True,
            fail_fast=False
        )
        
        env = MockEnvironment(scenario="aligned")
        mock_callback = MockCallback(audit_callback)
        mock_callback.simulate_training(env, episodes=20)
        
        # Quick check
        csv_path = Path(temp_dir) / "aligned_test" / "reward_pnl_audit.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mean_corr = df['step_correlation'].mean()
            logger.info(f"✅ Aligned scenario - Mean correlation: {mean_corr:.3f}")
            return mean_corr
        
        return 0.0


def test_misaligned_scenario():
    """Test with misaligned reward-P&L scenario."""
    logger.info("\n" + "="*60)
    logger.info("🚨 Testing MISALIGNED Reward Scenario")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_callback = RewardPnLAudit(
            output_dir=temp_dir + "/misaligned_test",
            min_correlation_threshold=0.5,
            alert_episodes=5,
            verbose=True,
            fail_fast=False  # Don't fail fast for testing
        )
        
        env = MockEnvironment(scenario="misaligned")
        mock_callback = MockCallback(audit_callback)
        mock_callback.simulate_training(env, episodes=15)
        
        # Quick check
        csv_path = Path(temp_dir) / "misaligned_test" / "reward_pnl_audit.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mean_corr = df['step_correlation'].mean()
            logger.info(f"⚠️ Misaligned scenario - Mean correlation: {mean_corr:.3f}")
            return mean_corr
        
        return 0.0


def test_partially_aligned_scenario():
    """Test with partially aligned reward-P&L scenario."""
    logger.info("\n" + "="*60)
    logger.info("🔄 Testing PARTIALLY ALIGNED Reward Scenario")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_callback = RewardPnLAudit(
            output_dir=temp_dir + "/partial_test",
            min_correlation_threshold=0.6,
            verbose=True,
            fail_fast=False
        )
        
        env = MockEnvironment(scenario="partially_aligned")
        mock_callback = MockCallback(audit_callback)
        mock_callback.simulate_training(env, episodes=20)
        
        # Quick check
        csv_path = Path(temp_dir) / "partial_test" / "reward_pnl_audit.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mean_corr = df['step_correlation'].mean()
            logger.info(f"🔄 Partially aligned scenario - Mean correlation: {mean_corr:.3f}")
            return mean_corr
        
        return 0.0


def test_fail_fast_mechanism():
    """Test the fail-fast mechanism with misaligned rewards."""
    logger.info("\n" + "="*60)
    logger.info("🛑 Testing FAIL-FAST Mechanism")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_callback = RewardPnLAudit(
            output_dir=temp_dir + "/fail_fast_test",
            min_correlation_threshold=0.8,  # High threshold
            alert_episodes=3,  # Low episode count
            verbose=True,
            fail_fast=True  # Enable fail-fast
        )
        
        env = MockEnvironment(scenario="misaligned")
        mock_callback = MockCallback(audit_callback)
        
        try:
            mock_callback.simulate_training(env, episodes=10)
            logger.warning("⚠️ Fail-fast mechanism did not trigger (unexpected)")
            return False
        except ValueError as e:
            logger.info(f"✅ Fail-fast mechanism triggered correctly: {str(e)[:100]}...")
            return True


def create_comparison_visualization():
    """Create a comparison visualization of different scenarios."""
    logger.info("\n" + "="*60)
    logger.info("📊 Creating Comparison Visualization")
    logger.info("="*60)
    
    scenarios = ["aligned", "partially_aligned", "misaligned"]
    correlations = []
    
    for scenario in scenarios:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_callback = RewardPnLAudit(
                output_dir=temp_dir + f"/{scenario}_viz",
                min_correlation_threshold=0.5,
                verbose=False,
                save_plots=False
            )
            
            env = MockEnvironment(scenario=scenario)
            mock_callback = MockCallback(audit_callback)
            mock_callback.simulate_training(env, episodes=15)
            
            csv_path = Path(temp_dir) / f"{scenario}_viz" / "reward_pnl_audit.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                correlations.append(df['step_correlation'].mean())
            else:
                correlations.append(0.0)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Bar plot of mean correlations
    plt.subplot(2, 2, 1)
    colors = ['green', 'orange', 'red']
    bars = plt.bar(scenarios, correlations, color=colors, alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.ylabel('Mean Step Correlation')
    plt.title('Reward-P&L Correlation by Scenario')
    plt.legend()
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Create sample data for other plots
    np.random.seed(42)
    episodes = range(1, 16)
    
    # Simulated correlation trends
    plt.subplot(2, 2, 2)
    aligned_trend = 0.85 + np.random.normal(0, 0.05, 15)
    partial_trend = 0.55 + np.random.normal(0, 0.1, 15)
    misaligned_trend = 0.1 + np.random.normal(0, 0.15, 15)
    
    plt.plot(episodes, aligned_trend, 'g-', label='Aligned', marker='o')
    plt.plot(episodes, partial_trend, 'orange', label='Partially Aligned', marker='s')
    plt.plot(episodes, misaligned_trend, 'r-', label='Misaligned', marker='^')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Step Correlation')
    plt.title('Correlation Trends Over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reward vs P&L scatter for aligned scenario
    plt.subplot(2, 2, 3)
    rewards = np.random.normal(0, 0.1, 50)
    pnl = rewards * 1000 + np.random.normal(0, 50, 50)  # Highly correlated
    plt.scatter(rewards, pnl, alpha=0.6, color='green')
    plt.axline((0, 0), slope=1000, linestyle='--', color='red', alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('P&L ($)')
    plt.title('Aligned: Reward vs P&L')
    plt.grid(True, alpha=0.3)
    
    # Reward vs P&L scatter for misaligned scenario
    plt.subplot(2, 2, 4)
    rewards_mis = np.random.normal(0, 0.1, 50)
    pnl_mis = np.random.normal(0, 100, 50)  # No correlation
    plt.scatter(rewards_mis, pnl_mis, alpha=0.6, color='red')
    plt.xlabel('Reward')
    plt.ylabel('P&L ($)')
    plt.title('Misaligned: Reward vs P&L')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reward_pnl_audit_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Comparison visualization saved as 'reward_pnl_audit_comparison.png'")


def main():
    """Run comprehensive tests of the Reward-P&L Audit System."""
    logger.info("🎯 Starting Reward-P&L Audit System Tests...")
    
    # Test different scenarios
    aligned_corr = test_aligned_scenario()
    misaligned_corr = test_misaligned_scenario()
    partial_corr = test_partially_aligned_scenario()
    
    # Test fail-fast mechanism
    fail_fast_worked = test_fail_fast_mechanism()
    
    # Create comparison visualization
    create_comparison_visualization()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Aligned scenario correlation: {aligned_corr:.3f}")
    logger.info(f"🔄 Partially aligned correlation: {partial_corr:.3f}")
    logger.info(f"⚠️ Misaligned scenario correlation: {misaligned_corr:.3f}")
    logger.info(f"🛑 Fail-fast mechanism: {'✅ Working' if fail_fast_worked else '❌ Failed'}")
    
    # Validate results
    if aligned_corr > 0.7 and misaligned_corr < 0.3 and fail_fast_worked:
        logger.info("\n🎉 ALL TESTS PASSED! Reward-P&L Audit System is working correctly!")
        logger.info("\n📖 Usage Instructions:")
        logger.info("1. Import: from src.training.reward_pnl_audit import RewardPnLAudit")
        logger.info("2. Create: audit_cb = RewardPnLAudit(min_correlation_threshold=0.6)")
        logger.info("3. Train: model.learn(total_timesteps=100000, callback=audit_cb)")
        logger.info("4. Check: Results saved in 'reward_pnl_audit/' directory")
        return 0
    else:
        logger.error("❌ Some tests failed - check implementation")
        return 1


if __name__ == "__main__":
    exit(main())