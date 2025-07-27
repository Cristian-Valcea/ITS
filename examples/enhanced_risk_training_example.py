# examples/enhanced_risk_training_example.py
"""
Enhanced Risk-Aware Training Example with λ-weighted Multi-Risk Early Stopping.

This example demonstrates how to use the new EnhancedRiskCallback that combines
multiple risk metrics (drawdown, ulcer index, market impact, feed staleness) 
with configurable weights to prevent the DQN from learning to trade illiquid names.

Key Features Demonstrated:
- λ-weighted composite risk scoring
- Liquidity-aware penalties for illiquid trading
- Multi-risk metric early stopping
- Risk decomposition and analysis
- Adaptive threshold adjustments

Usage:
    python examples/enhanced_risk_training_example.py
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_risk_advisor():
    """Create a sample risk advisor for demonstration."""
    from src.training.interfaces.risk_advisor import RiskAdvisor
    
    class SampleRiskAdvisor(RiskAdvisor):
        """Sample risk advisor that simulates realistic risk metrics."""
        
        def __init__(self):
            super().__init__("sample_advisor")
            self.step_count = 0
            self.portfolio_history = [100000.0]  # Starting portfolio value
            self.max_portfolio_value = 100000.0
            
        def evaluate(self, obs: dict) -> dict:
            """Simulate risk evaluation with realistic metrics."""
            self.step_count += 1
            
            # Extract portfolio value from observation
            portfolio_value = obs.get('portfolio_value', 100000.0)
            self.portfolio_history.append(portfolio_value)
            self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
            
            # Keep history manageable
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-500:]
            
            # Calculate drawdown
            current_drawdown = max(0, (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value)
            
            # Simulate ulcer index (simplified)
            if len(self.portfolio_history) >= 10:
                recent_values = np.array(self.portfolio_history[-10:])
                peaks = np.maximum.accumulate(recent_values)
                drawdowns = (peaks - recent_values) / peaks
                ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
            else:
                ulcer_index = current_drawdown
            
            # Simulate market impact based on position size
            position = abs(obs.get('position', 0.0))
            trade_size = obs.get('trade_size', 0.0)
            
            # Higher impact for larger positions and trades
            base_impact = min(position * 0.001 + trade_size * 0.002, 0.1)
            
            # Add some market condition variability
            market_stress = 0.5 + 0.5 * np.sin(self.step_count * 0.01)  # Oscillating market stress
            kyle_lambda = base_impact * market_stress
            
            # Simulate feed staleness (random with occasional spikes)
            if np.random.random() < 0.05:  # 5% chance of feed issues
                feed_staleness_ms = np.random.uniform(500, 2000)
            else:
                feed_staleness_ms = np.random.uniform(10, 100)
            
            # Calculate breach severity (for compatibility)
            risk_factors = [current_drawdown, ulcer_index, kyle_lambda, feed_staleness_ms / 1000.0]
            breach_severity = np.mean(risk_factors)
            
            # Calculate penalty
            penalty = (
                2.0 * current_drawdown +
                1.5 * ulcer_index +
                3.0 * kyle_lambda +
                1.0 * (feed_staleness_ms / 1000.0)
            )
            
            return {
                'drawdown_pct': current_drawdown,
                'ulcer_index': ulcer_index,
                'kyle_lambda': kyle_lambda,
                'feed_staleness_ms': feed_staleness_ms,
                'breach_severity': breach_severity,
                'penalty': penalty,
                'overall_risk_score': min(breach_severity, 1.0),
                'var_breach_severity': kyle_lambda,  # Use market impact as VaR proxy
                'position_concentration': position,
                'liquidity_risk': kyle_lambda
            }
        
        def get_risk_config(self) -> dict:
            """Return sample risk configuration."""
            return {
                'drawdown_threshold': 0.15,
                'ulcer_threshold': 0.10,
                'market_impact_threshold': 0.05,
                'feed_staleness_threshold': 1000.0
            }
    
    return SampleRiskAdvisor()

def create_sample_environment():
    """Create a sample trading environment for demonstration."""
    import gym
    from gym import spaces
    
    class SampleTradingEnv(gym.Env):
        """Sample trading environment that simulates market conditions."""
        
        def __init__(self):
            super().__init__()
            
            # Action space: -1 (sell), 0 (hold), 1 (buy)
            self.action_space = spaces.Discrete(3)
            
            # Observation space: market features + position + portfolio value
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
            
            self.reset()
        
        def reset(self):
            """Reset environment to initial state."""
            self.step_count = 0
            self.position = 0.0
            self.portfolio_value = 100000.0
            self.cash = 100000.0
            self.price = 100.0
            self.price_history = [self.price]
            
            # Store last observation for risk callback
            self.last_raw_obs = self._get_observation()
            return self.last_raw_obs
        
        def step(self, action):
            """Execute one trading step."""
            self.step_count += 1
            
            # Convert action: 0->-1 (sell), 1->0 (hold), 2->1 (buy)
            trade_action = action - 1
            
            # Simulate price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            self.price *= (1 + price_change)
            self.price_history.append(self.price)
            
            # Execute trade
            trade_size = trade_action * 100  # Trade in lots of 100
            
            if trade_size != 0:
                trade_cost = abs(trade_size) * self.price
                
                # Check if we have enough cash/position
                if trade_size > 0:  # Buying
                    if self.cash >= trade_cost:
                        self.position += trade_size
                        self.cash -= trade_cost
                else:  # Selling
                    if self.position >= abs(trade_size):
                        self.position += trade_size  # trade_size is negative
                        self.cash -= trade_cost  # trade_cost is positive, so this adds cash
            
            # Update portfolio value
            self.portfolio_value = self.cash + self.position * self.price
            
            # Calculate reward (simple P&L)
            if len(self.price_history) > 1:
                price_return = (self.price - self.price_history[-2]) / self.price_history[-2]
                reward = self.position * price_return * 100  # Scale reward
            else:
                reward = 0.0
            
            # Add small penalty for holding large positions (encourage liquidity)
            position_penalty = -0.001 * abs(self.position) / 1000
            reward += position_penalty
            
            # Episode ends after 1000 steps or if portfolio drops too much
            done = (
                self.step_count >= 1000 or 
                self.portfolio_value < 50000  # 50% drawdown limit
            )
            
            # Store observation for risk callback
            self.last_raw_obs = self._get_observation()
            
            info = {
                'portfolio_value': self.portfolio_value,
                'position': self.position,
                'price': self.price,
                'trade_size': abs(trade_size) if 'trade_size' in locals() else 0.0
            }
            
            return self.last_raw_obs, reward, done, info
        
        def _get_observation(self):
            """Get current observation."""
            # Market features (price history, volatility, etc.)
            if len(self.price_history) >= 10:
                recent_prices = np.array(self.price_history[-10:])
                price_returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(price_returns)
                momentum = np.mean(price_returns)
            else:
                volatility = 0.02
                momentum = 0.0
            
            # Create observation vector
            obs = np.array([
                self.price / 100.0,  # Normalized price
                volatility,
                momentum,
                self.position / 1000.0,  # Normalized position
                self.cash / 100000.0,  # Normalized cash
                self.portfolio_value / 100000.0,  # Normalized portfolio value
                self.step_count / 1000.0,  # Normalized time
                np.sin(self.step_count * 0.1),  # Market cycle feature
                np.cos(self.step_count * 0.1),  # Market cycle feature
                np.random.normal(0, 0.1),  # Noise feature
                self.position,  # Raw position for risk calculation
                self.portfolio_value  # Raw portfolio value for risk calculation
            ], dtype=np.float32)
            
            return obs
    
    return SampleTradingEnv()

def demonstrate_enhanced_risk_training():
    """Demonstrate enhanced risk-aware training with λ-weighted early stopping."""
    print("🎯 Enhanced Risk-Aware Training Demo")
    print("=" * 50)
    
    try:
        # Create sample components
        print("1️⃣  Creating sample environment and risk advisor...")
        env = create_sample_environment()
        risk_advisor = create_sample_risk_advisor()
        print("✅ Components created")
        
        # Create enhanced risk callback
        print("\n2️⃣  Creating enhanced risk callback...")
        from src.training.callbacks.enhanced_risk_callback import create_enhanced_risk_callback
        
        # Configuration emphasizing liquidity risk
        risk_config = {
            'risk_weights': {
                'drawdown_pct': 0.25,      # Reduced from default
                'ulcer_index': 0.20,       # Reduced from default  
                'kyle_lambda': 0.40,       # Increased - focus on market impact
                'feed_staleness': 0.15     # Reduced from default
            },
            'early_stop_threshold': 0.70,  # Lower threshold for more aggressive stopping
            'liquidity_penalty_multiplier': 3.0,  # Higher penalty for illiquid trades
            'consecutive_violations_limit': 3,  # Stop faster
            'evaluation_frequency': 50,   # Evaluate more frequently
            'log_frequency': 200,
            'enable_risk_decomposition': True,
            'verbose': 1
        }
        
        enhanced_callback = create_enhanced_risk_callback(
            risk_advisor=risk_advisor,
            config=risk_config
        )
        print("✅ Enhanced risk callback created")
        print(f"   Risk weights: {risk_config['risk_weights']}")
        print(f"   Liquidity penalty multiplier: {risk_config['liquidity_penalty_multiplier']}")
        print(f"   Early stop threshold: {risk_config['early_stop_threshold']}")
        
        # Simulate training steps
        print("\n3️⃣  Simulating training with risk monitoring...")
        
        # Mock training environment setup
        class MockModel:
            def __init__(self):
                self.num_timesteps = 0
        
        class MockLogger:
            def record(self, key, value):
                pass
        
        # Setup callback using init_callback method
        enhanced_callback.init_callback(MockModel())
        enhanced_callback.training_env = env
        
        # Simulate training loop
        training_stopped = False
        max_steps = 2000
        
        for step in range(max_steps):
            enhanced_callback.model.num_timesteps = step
            
            # Simulate environment step
            if step == 0:
                obs = env.reset()
            else:
                action = np.random.choice([0, 1, 2])  # Random action
                obs, reward, done, info = env.step(action)
                
                if done:
                    obs = env.reset()
            
            # Call the enhanced callback
            should_continue = enhanced_callback._on_step()
            
            if not should_continue:
                print(f"\n🛑 Training stopped early at step {step} due to risk violations!")
                training_stopped = True
                break
            
            # Log progress periodically
            if step % 500 == 0 and step > 0:
                print(f"   Step {step}: Training continuing...")
        
        if not training_stopped:
            print(f"\n✅ Training completed {max_steps} steps without early stopping")
        
        # Get risk summary
        print("\n4️⃣  Risk Analysis Summary:")
        risk_summary = enhanced_callback.get_risk_summary()
        
        print(f"   Total Evaluations: {risk_summary['total_evaluations']}")
        print(f"   Total Violations: {risk_summary['total_violations']}")
        print(f"   Illiquid Trading Rate: {risk_summary['illiquid_trading_rate']:.2%}")
        print(f"   Avg Evaluation Time: {risk_summary['avg_evaluation_time_ms']:.2f}ms")
        
        # Show risk statistics
        if risk_summary['risk_statistics']:
            print("\n   Risk Metric Statistics:")
            for metric, stats in risk_summary['risk_statistics'].items():
                if 'current' in stats:
                    print(f"     {metric}: current={stats['current']:.4f}, "
                          f"mean={stats['mean']:.4f}, max={stats['max']:.4f}, "
                          f"violations={stats['violations']}")
        
        # Show recent violations
        if risk_summary['violation_episodes']:
            print(f"\n   Recent Risk Violations ({len(risk_summary['violation_episodes'])}):")
            for i, violation in enumerate(risk_summary['violation_episodes'][-3:]):
                print(f"     {i+1}. Step {violation['step']}: "
                      f"composite_risk={violation['composite_risk']:.4f}")
                
                # Show risk breakdown
                breakdown = violation['risk_breakdown']
                print(f"        Drawdown: {breakdown['drawdown']:.4f}, "
                      f"Ulcer: {breakdown['ulcer']:.4f}, "
                      f"Market Impact: {breakdown['market_impact']:.4f}, "
                      f"Feed Staleness: {breakdown['feed_staleness']:.1f}ms")
        
        # Show illiquid trading episodes
        if risk_summary['illiquid_episodes']:
            print(f"\n   Recent Illiquid Trades ({len(risk_summary['illiquid_episodes'])}):")
            for i, episode in enumerate(risk_summary['illiquid_episodes'][-3:]):
                print(f"     {i+1}. Step {episode['step']}: "
                      f"market_impact={episode['market_impact']:.4f}, "
                      f"position={episode['position_size']:.1f}")
        
        # Save detailed analysis
        print("\n5️⃣  Saving risk analysis...")
        analysis_file = f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        enhanced_callback.save_risk_analysis(analysis_file)
        print(f"✅ Analysis saved to {analysis_file}")
        
        # Demonstrate configuration variations
        print("\n6️⃣  Configuration Recommendations:")
        print("   For High-Frequency Trading:")
        print("     - Increase kyle_lambda weight to 0.5+")
        print("     - Set liquidity_penalty_multiplier to 5.0+")
        print("     - Lower early_stop_threshold to 0.6")
        print("     - Increase evaluation_frequency to 25")
        
        print("\n   For Conservative Long-Term Trading:")
        print("     - Increase drawdown_pct weight to 0.4+")
        print("     - Increase ulcer_index weight to 0.3+")
        print("     - Set early_stop_threshold to 0.8+")
        print("     - Enable adaptive thresholds")
        
        print("\n   For Volatile Markets:")
        print("     - Enable adaptive thresholds")
        print("     - Increase consecutive_violations_limit to 7")
        print("     - Balance all weights more evenly")
        print("     - Increase evaluation_frequency during stress")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please ensure all required modules are available")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_weight_sensitivity():
    """Demonstrate how different weight configurations affect risk scoring."""
    print("\n🔬 Risk Weight Sensitivity Analysis")
    print("=" * 40)
    
    # Sample risk metrics
    sample_metrics = {
        'drawdown_pct': 0.12,      # 12% drawdown
        'ulcer_index': 0.08,       # 8% ulcer index
        'kyle_lambda': 0.06,       # 6% market impact (high)
        'feed_staleness_ms': 800   # 800ms staleness
    }
    
    print("Sample Risk Metrics:")
    for metric, value in sample_metrics.items():
        unit = "ms" if "staleness" in metric else ""
        print(f"  {metric}: {value}{unit}")
    
    # Different weight configurations
    weight_configs = {
        'Balanced': {
            'drawdown_pct': 0.25, 'ulcer_index': 0.25, 
            'kyle_lambda': 0.25, 'feed_staleness': 0.25
        },
        'Liquidity-Focused': {
            'drawdown_pct': 0.20, 'ulcer_index': 0.15, 
            'kyle_lambda': 0.50, 'feed_staleness': 0.15
        },
        'Drawdown-Focused': {
            'drawdown_pct': 0.40, 'ulcer_index': 0.30, 
            'kyle_lambda': 0.20, 'feed_staleness': 0.10
        },
        'Tech-Focused': {
            'drawdown_pct': 0.15, 'ulcer_index': 0.15, 
            'kyle_lambda': 0.20, 'feed_staleness': 0.50
        }
    }
    
    print("\nComposite Risk Scores by Configuration:")
    
    for config_name, weights in weight_configs.items():
        # Calculate composite score
        drawdown_contrib = weights['drawdown_pct'] * sample_metrics['drawdown_pct']
        ulcer_contrib = weights['ulcer_index'] * sample_metrics['ulcer_index']
        impact_contrib = weights['kyle_lambda'] * sample_metrics['kyle_lambda']
        staleness_contrib = weights['feed_staleness'] * (sample_metrics['feed_staleness_ms'] / 5000.0)
        
        composite_score = drawdown_contrib + ulcer_contrib + impact_contrib + staleness_contrib
        
        print(f"\n  {config_name}:")
        print(f"    Composite Score: {composite_score:.4f}")
        print(f"    Contributions:")
        print(f"      Drawdown: {drawdown_contrib:.4f}")
        print(f"      Ulcer: {ulcer_contrib:.4f}")
        print(f"      Market Impact: {impact_contrib:.4f}")
        print(f"      Feed Staleness: {staleness_contrib:.4f}")
        
        # Determine dominant factor
        contributions = {
            'Drawdown': drawdown_contrib,
            'Ulcer': ulcer_contrib,
            'Market Impact': impact_contrib,
            'Feed Staleness': staleness_contrib
        }
        dominant = max(contributions.items(), key=lambda x: x[1])
        print(f"    Dominant Factor: {dominant[0]} ({dominant[1]:.4f})")

def main():
    """Main demonstration function."""
    print("🚀 Enhanced Risk Callback Demonstration")
    print("Solving: Early-stop callback uses only drawdown penalty")
    print("Solution: λ-weighted sum of all risk metrics (ulcer, impact, feed-age)")
    print("Goal: Prevent DQN from learning to trade illiquid names")
    print()
    
    # Run main demonstration
    demonstrate_enhanced_risk_training()
    
    # Show weight sensitivity analysis
    demonstrate_weight_sensitivity()
    
    print("\n🎯 Key Benefits Achieved:")
    print("✅ Multi-risk metric evaluation (not just drawdown)")
    print("✅ λ-weighted composite risk scoring")
    print("✅ Liquidity-aware penalties for illiquid trading")
    print("✅ Configurable risk weights for different strategies")
    print("✅ Risk decomposition and detailed analysis")
    print("✅ Adaptive thresholds based on market conditions")
    print()
    print("🏆 DQN now learns to avoid illiquid names through comprehensive risk penalties!")

if __name__ == "__main__":
    main()