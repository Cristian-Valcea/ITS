#!/usr/bin/env python3
"""
📚 FRICTION CURRICULUM CALLBACK
Progressive friction training: 0.5bp → 1.2bp with domain randomization
Prevents over-fitting to unrealistically low trading costs
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)

class FrictionCurriculumCallback(BaseCallback):
    """
    Friction curriculum: progressively increase trading costs during training
    
    Phase 1 (0-10K):   Warm-up at 0.5bp/0.7bp (easy exploration)
    Phase 2 (10K-40K): Linear anneal to 1.0bp/2.0bp (production)
    Phase 3 (40K-60K): Overshoot to 1.2bp/2.5bp (safety margin)
    
    + Domain randomization each episode
    """
    
    def __init__(self, env, total_steps=60000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.total_steps = total_steps
        
        # Curriculum phases
        self.phase1_end = 10000   # Warm-up phase
        self.phase2_end = 40000   # Annealing phase
        self.phase3_end = 60000   # Overshoot phase
        
        # Friction parameters
        self.easy_tc = 0.5
        self.easy_penalty = 0.7
        self.prod_tc = 1.0
        self.prod_penalty = 2.0
        self.hard_tc = 1.2
        self.hard_penalty = 2.5
        
        # Domain randomization bounds
        self.tc_random_range = 0.2      # ±0.2bp randomization
        self.penalty_random_range = 0.4  # ±0.4bp randomization
        
        logger.info("📚 FRICTION CURRICULUM INITIALIZED")
        logger.info(f"   📊 Total steps: {total_steps:,}")
        logger.info(f"   🚀 Phase 1 (0-10K): Easy exploration ({self.easy_tc}bp/{self.easy_penalty}bp)")
        logger.info(f"   📈 Phase 2 (10K-40K): Linear anneal to production ({self.prod_tc}bp/{self.prod_penalty}bp)")
        logger.info(f"   🛡️ Phase 3 (40K-60K): Overshoot for safety ({self.hard_tc}bp/{self.hard_penalty}bp)")
        logger.info(f"   🎲 Domain randomization: ±{self.tc_random_range}bp TC, ±{self.penalty_random_range}bp penalty")
    
    def _get_current_friction(self, step):
        """Calculate current friction parameters based on training step"""
        
        if step <= self.phase1_end:
            # Phase 1: Warm-up (easy exploration)
            tc_bp = self.easy_tc
            penalty_bp = self.easy_penalty
            phase = "WARM-UP"
            
        elif step <= self.phase2_end:
            # Phase 2: Linear annealing to production
            anneal_progress = (step - self.phase1_end) / (self.phase2_end - self.phase1_end)
            tc_bp = self.easy_tc + anneal_progress * (self.prod_tc - self.easy_tc)
            penalty_bp = self.easy_penalty + anneal_progress * (self.prod_penalty - self.easy_penalty)
            phase = "ANNEALING"
            
        else:
            # Phase 3: Overshoot for safety margin
            overshoot_progress = (step - self.phase2_end) / (self.phase3_end - self.phase2_end)
            tc_bp = self.prod_tc + overshoot_progress * (self.hard_tc - self.prod_tc)
            penalty_bp = self.prod_penalty + overshoot_progress * (self.hard_penalty - self.prod_penalty)
            phase = "OVERSHOOT"
        
        return tc_bp, penalty_bp, phase
    
    def _apply_domain_randomization(self, tc_bp, penalty_bp):
        """Apply domain randomization to friction parameters"""
        
        # Randomize transaction cost
        tc_noise = np.random.uniform(-self.tc_random_range, self.tc_random_range)
        randomized_tc = max(0.1, tc_bp + tc_noise)  # Minimum 0.1bp
        
        # Randomize penalty
        penalty_noise = np.random.uniform(-self.penalty_random_range, self.penalty_random_range)
        randomized_penalty = max(0.1, penalty_bp + penalty_noise)  # Minimum 0.1bp
        
        return randomized_tc, randomized_penalty
    
    def _on_step(self) -> bool:
        """Called at each training step to update friction"""
        
        # Get base friction for current step
        tc_bp, penalty_bp, phase = self._get_current_friction(self.num_timesteps)
        
        # Apply domain randomization
        final_tc, final_penalty = self._apply_domain_randomization(tc_bp, penalty_bp)
        
        # Update environment friction
        if hasattr(self.env, 'set_friction'):
            self.env.set_friction(final_tc, final_penalty)
        elif hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            # For vectorized environments
            env = self.env.envs[0]
            if hasattr(env, 'env'):  # Monitor wrapper
                env = env.env
            if hasattr(env, 'set_friction'):
                env.set_friction(final_tc, final_penalty)
        
        # Log progress every 5K steps
        if self.num_timesteps % 5000 == 0:
            progress = self.num_timesteps / self.total_steps
            logger.info(f"📚 CURRICULUM STEP {self.num_timesteps:,} ({progress:.1%})")
            logger.info(f"   🎯 Phase: {phase}")
            logger.info(f"   💰 Base friction: {tc_bp:.2f}bp / {penalty_bp:.2f}bp")
            logger.info(f"   🎲 Randomized: {final_tc:.2f}bp / {final_penalty:.2f}bp")
        
        return True
    
    def _on_training_start(self):
        """Called when training starts"""
        logger.info("📚 FRICTION CURRICULUM TRAINING STARTED")
        
        # Set initial friction (warm-up phase)
        initial_tc, initial_penalty = self._apply_domain_randomization(self.easy_tc, self.easy_penalty)
        
        if hasattr(self.env, 'set_friction'):
            self.env.set_friction(initial_tc, initial_penalty)
        elif hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            env = self.env.envs[0]
            if hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'set_friction'):
                env.set_friction(initial_tc, initial_penalty)
        
        logger.info(f"   🚀 Initial friction: {initial_tc:.2f}bp / {initial_penalty:.2f}bp")
    
    def _on_training_end(self):
        """Called when training ends"""
        logger.info("📚 FRICTION CURRICULUM TRAINING COMPLETED")
        
        # Final friction should be at overshoot level
        final_tc, final_penalty, phase = self._get_current_friction(self.total_steps)
        logger.info(f"   🏁 Final phase: {phase}")
        logger.info(f"   🛡️ Final friction: {final_tc:.2f}bp / {final_penalty:.2f}bp")
        logger.info("   ✅ Model trained with progressive friction curriculum")


class DomainRandomizedTradingEnv:
    """
    Wrapper to add set_friction method to trading environment
    Enables dynamic friction adjustment during training
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.current_tc_bp = getattr(base_env, 'tc_bp', 1.0)
        self.current_trade_penalty_bp = getattr(base_env, 'trade_penalty_bp', 2.0)
    
    def set_friction(self, tc_bp, trade_penalty_bp):
        """Set current friction parameters"""
        self.current_tc_bp = tc_bp
        self.current_trade_penalty_bp = trade_penalty_bp
        
        # Update base environment if it has these attributes
        if hasattr(self.base_env, 'tc_bp'):
            self.base_env.tc_bp = tc_bp
        if hasattr(self.base_env, 'trade_penalty_bp'):
            self.base_env.trade_penalty_bp = trade_penalty_bp
    
    def reset(self, **kwargs):
        """Reset with current friction parameters"""
        return self.base_env.reset(**kwargs)
    
    def step(self, action):
        """Step with current friction parameters"""
        return self.base_env.step(action)
    
    def __getattr__(self, name):
        """Delegate all other attributes to base environment"""
        return getattr(self.base_env, name)