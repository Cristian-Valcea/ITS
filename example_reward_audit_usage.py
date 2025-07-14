#!/usr/bin/env python3
"""
Example: How to use the Reward-P&L Audit System with IntradayJules

This example shows how to integrate the RewardPnLAudit callback into your
existing training pipeline to ensure reward-P&L alignment.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reward_pnl_audit import RewardPnLAudit, quick_audit_check
from src.training.reward_audit_integration import enhanced_training_with_audit

# Example 1: Basic usage with any SB3 model
def basic_usage_example():
    """Basic example of using the audit callback."""
    print("🎯 Example 1: Basic Reward-P&L Audit Usage")
    print("=" * 50)
    
    # Your existing training code would look like this:
    example_code = '''
    from stable_baselines3 import DQN
    from src.training.reward_pnl_audit import RewardPnLAudit
    
    # Create your environment and model as usual
    env = IntradayTradingEnv(...)  # Your trading environment
    model = DQN("MlpPolicy", env, verbose=1)
    
    # Create the audit callback
    audit_callback = RewardPnLAudit(
        output_dir="reward_audit_results",
        min_correlation_threshold=0.6,  # Minimum acceptable correlation
        alert_episodes=10,              # Alert after N low-correlation episodes
        save_plots=True,                # Generate diagnostic plots
        fail_fast=False                 # Set to True for strict validation
    )
    
    # Train with audit monitoring
    model.learn(
        total_timesteps=100000,
        callback=audit_callback
    )
    
    # Results are automatically saved to "reward_audit_results/" directory
    '''
    
    print(example_code)


# Example 2: Enhanced training with comprehensive callbacks
def enhanced_usage_example():
    """Enhanced example with multiple callbacks."""
    print("\n🚀 Example 2: Enhanced Training with Multiple Callbacks")
    print("=" * 60)
    
    example_code = '''
    from src.training.reward_audit_integration import enhanced_training_with_audit
    
    # Example 2a: Strict production training (audit_strict takes precedence)
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=500000,
        model_save_path="models/nvda_dqn_strict",
        eval_env=eval_env,
        audit_strict=True,  # Overrides any audit_config settings
        callback_order="audit_first",  # Audit sees all steps including eval
        
        # These will be overridden by audit_strict=True
        audit_config={
            'min_correlation_threshold': 0.4,  # Overridden to 0.7
            'fail_fast': False  # Overridden to True
        }
    )
    # ⚠️ Warning logged about overrides
    
    # Example 2b: Custom configuration (audit_strict=False)
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=500000,
        model_save_path="models/nvda_dqn_custom",
        eval_env=eval_env,
        audit_strict=False,  # Use custom config
        callback_order="audit_last",  # Audit only sees training steps
        
        # Custom audit configuration
        audit_config={
            'min_correlation_threshold': 0.6,
            'alert_episodes': 8,
            'fail_fast': False
        },
        
        # Checkpoint configuration
        checkpoint_config={
            'save_freq': 10000,
            'save_replay_buffer': True
        }
    )
    
    # Check results
    if results['success']:
        print("✅ Training completed successfully!")
        print(f"Deployment ready: {results['audit_results']['deployment_ready']}")
    else:
        print(f"❌ Training failed: {results['message']}")
        if results['error'] == 'reward_pnl_misalignment':
            print("🔧 Action required: Revise reward function")
    '''
    
    print(example_code)


# Example 3: Post-training analysis
def post_training_analysis_example():
    """Example of post-training analysis."""
    print("\n📊 Example 3: Post-Training Analysis")
    print("=" * 40)
    
    example_code = '''
    from src.training.reward_pnl_audit import quick_audit_check
    
    # Quick analysis of saved audit results
    quick_audit_check("reward_audit_results/reward_pnl_audit.csv")
    
    # This will show:
    # - Mean step-wise correlation
    # - Episode-level correlation  
    # - Scatter plot of reward vs P&L
    # - Correlation trend over episodes
    # - Recommendation for deployment
    '''
    
    print(example_code)


# Example 4: Integration with existing IntradayJules agents
def intradayjules_integration_example():
    """Example of integration with IntradayJules training agents."""
    print("\n🏗️ Example 4: IntradayJules Integration")
    print("=" * 45)
    
    example_code = '''
    # In your trainer agent (e.g., src/agents/trainer_agent.py)
    from src.training.reward_pnl_audit import RewardPnLAudit
    
    class TrainerAgent:
        def train_model(self, config):
            # Create audit callback
            audit_callback = RewardPnLAudit(
                output_dir=f"{config['model_save_path']}/reward_audit",
                min_correlation_threshold=config.get('reward_correlation_threshold', 0.6),
                fail_fast=config.get('strict_reward_validation', False)
            )
            
            # Add to existing callbacks
            callbacks = [audit_callback]
            if hasattr(self, 'checkpoint_callback'):
                callbacks.append(self.checkpoint_callback)
            if hasattr(self, 'eval_callback'):
                callbacks.append(self.eval_callback)
            
            # Train with audit monitoring
            self.model.learn(
                total_timesteps=config['total_timesteps'],
                callback=CallbackList(callbacks)
            )
            
            # Check audit results before deployment
            audit_results = self._analyze_audit_results(
                f"{config['model_save_path']}/reward_audit"
            )
            
            if not audit_results.get('deployment_ready', False):
                raise ValueError(
                    f"Model failed reward-P&L audit: {audit_results['recommendation']}"
                )
    '''
    
    print(example_code)


# Example 5: Configuration for different scenarios
def configuration_examples():
    """Examples of different audit configurations."""
    print("\n⚙️ Example 5: Configuration for Different Scenarios")
    print("=" * 55)
    
    configs = {
        "Development/Testing": {
            'min_correlation_threshold': 0.5,
            'alert_episodes': 10,
            'fail_fast': False,
            'save_plots': True,
            'verbose': True
        },
        
        "Production Validation": {
            'min_correlation_threshold': 0.7,
            'alert_episodes': 5,
            'fail_fast': True,
            'save_plots': True,
            'verbose': True
        },
        
        "Quick Validation": {
            'min_correlation_threshold': 0.6,
            'alert_episodes': 3,
            'fail_fast': True,
            'save_plots': False,
            'verbose': False
        }
    }
    
    for scenario, config in configs.items():
        print(f"\n{scenario}:")
        print(f"  RewardPnLAudit({config})")


def callback_ordering_examples():
    """Examples of callback ordering considerations."""
    print("\n🔄 Example 6: Callback Ordering Considerations")
    print("=" * 50)
    
    example_code = '''
    # Option 1: audit_first (DEFAULT) - Audit sees ALL steps
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=100000,
        model_save_path="models/audit_first",
        eval_env=eval_env,
        callback_order="audit_first"  # Audit → Checkpoint → Eval
    )
    # ✅ Audit tracks correlation during both training AND evaluation
    # ✅ More comprehensive monitoring
    # ⚠️ Evaluation steps might affect correlation metrics
    
    # Option 2: audit_last - Audit only sees training steps
    results = enhanced_training_with_audit(
        model=model,
        total_timesteps=100000,
        model_save_path="models/audit_last", 
        eval_env=eval_env,
        callback_order="audit_last"  # Checkpoint → Eval → Audit
    )
    # ✅ Audit only tracks training correlation (cleaner signal)
    # ✅ Evaluation doesn't interfere with audit metrics
    # ⚠️ Less comprehensive monitoring
    
    # Precedence Rules for audit_strict:
    enhanced_training_with_audit(
        audit_strict=True,  # Takes precedence
        audit_config={
            'fail_fast': False,  # Overridden to True
            'min_correlation_threshold': 0.4  # Overridden to 0.7
        }
    )
    # ⚠️ Warnings logged about overrides
    '''
    
    print(example_code)


def main():
    """Run all examples."""
    print("🎯 REWARD-P&L AUDIT SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    print("This system ensures your agent optimizes for actual P&L,")
    print("preventing 'looks-good-in-training, bad-in-cash' behavior.")
    print("=" * 60)
    
    basic_usage_example()
    enhanced_usage_example()
    post_training_analysis_example()
    intradayjules_integration_example()
    configuration_examples()
    callback_ordering_examples()
    
    print("\n" + "=" * 60)
    print("🎉 SUMMARY: Key Benefits")
    print("=" * 60)
    print("✅ Prevents reward-P&L misalignment")
    print("✅ Early detection of problematic reward functions")
    print("✅ Comprehensive diagnostic plots and reports")
    print("✅ Fail-fast mechanism for production safety")
    print("✅ Integration with TensorBoard/W&B")
    print("✅ Automated deployment readiness assessment")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Add RewardPnLAudit to your training pipeline")
    print("2. Set appropriate correlation thresholds")
    print("3. Review audit reports before deployment")
    print("4. Iterate on reward function if needed")


if __name__ == "__main__":
    main()