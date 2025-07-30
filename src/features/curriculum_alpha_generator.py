#!/usr/bin/env python3
"""
🎯 CURRICULUM ALPHA GENERATOR
Enhanced alpha patterns for persistent-to-noisy curriculum learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaPattern(Enum):
    """Alpha signal patterns for curriculum learning"""
    PERSISTENT = "persistent"
    PIECEWISE = "piecewise" 
    LOW_NOISE = "low_noise"
    REALISTIC_NOISY = "realistic_noisy"

class CurriculumAlphaGenerator:
    """Generate alpha signals for progressive curriculum learning"""
    
    def __init__(self, pattern: AlphaPattern, seed: int = 42):
        self.pattern = pattern
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"🎯 CurriculumAlphaGenerator initialized:")
        logger.info(f"   Pattern: {pattern.value}")
        logger.info(f"   Seed: {seed}")
    
    def create_persistent_alpha(self, n_periods: int, alpha_strength: float, 
                              bull_episode: bool = True) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Phase 0: Persistent alpha - same direction for entire episode
        
        Args:
            n_periods: Number of time periods
            alpha_strength: Alpha magnitude (e.g., 0.40)
            bull_episode: True for bull episode (+alpha), False for bear (-alpha)
        """
        
        logger.info(f"🎯 Creating PERSISTENT alpha data:")
        logger.info(f"   Periods: {n_periods}")
        logger.info(f"   Alpha strength: {alpha_strength}")
        logger.info(f"   Episode type: {'BULL' if bull_episode else 'BEAR'}")
        
        # Create trading days
        trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
        
        # Persistent alpha signal - same direction throughout episode
        alpha_signal = alpha_strength if bull_episode else -alpha_strength
        alpha_signals = np.full(n_periods, alpha_signal, dtype=np.float32)
        
        # Generate price series with persistent trend
        base_price = 170.0
        prices = [base_price]
        
        for i in range(1, n_periods):
            # Alpha-driven price movement + small noise
            alpha_return = alpha_signal * 0.01  # Convert to return
            noise = np.random.normal(0, 0.0005)  # Very small noise for persistence
            price_change = alpha_return + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Create features: 12 random + persistent alpha
        random_features = np.random.randn(n_periods, 12).astype(np.float32) * 0.05  # Small random features
        alpha_feature = alpha_signals.reshape(-1, 1)
        features = np.hstack([random_features, alpha_feature])
        
        price_series = pd.Series(prices, index=trading_days)
        
        # Metadata
        metadata = {
            'pattern': 'persistent',
            'alpha_strength': alpha_strength,
            'bull_episode': bull_episode,
            'expected_direction': 'UP' if bull_episode else 'DOWN',
            'signal_consistency': 1.0,  # Perfect consistency
            'signal_mean': float(alpha_signal),
            'signal_std': 0.0
        }
        
        logger.info(f"   Expected trend: {metadata['expected_direction']}")
        logger.info(f"   Signal consistency: {metadata['signal_consistency']:.1%}")
        
        return features, price_series, metadata
    
    def create_piecewise_alpha(self, n_periods: int, alpha_strength: float,
                              on_duration: int = 500, off_duration: int = 250) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Phase 1: Piece-wise constant alpha - alternates between signal and neutral
        
        Args:
            n_periods: Number of time periods
            alpha_strength: Alpha magnitude when signal is "on"
            on_duration: Steps with alpha signal
            off_duration: Steps with no signal (neutral)
        """
        
        logger.info(f"🎯 Creating PIECEWISE alpha data:")
        logger.info(f"   Periods: {n_periods}")
        logger.info(f"   Alpha strength: {alpha_strength}")
        logger.info(f"   Pattern: {on_duration} ON / {off_duration} OFF")
        
        # Create trading days
        trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
        
        # Generate piecewise alpha pattern with OFF period indicators
        alpha_signals = []
        off_period_indicators = []  # Track OFF periods for position decay
        cycle_length = on_duration + off_duration
        
        for i in range(n_periods):
            cycle_position = i % cycle_length
            if cycle_position < on_duration:
                # "On" period - use full alpha signal
                alpha_signals.append(alpha_strength)
                off_period_indicators.append(0.0)  # Not OFF period
            else:
                # "Off" period - neutral alpha but mark for position decay
                alpha_signals.append(0.0)
                off_period_indicators.append(1.0)  # OFF period - should decay position
        
        alpha_signals = np.array(alpha_signals, dtype=np.float32)
        off_period_indicators = np.array(off_period_indicators, dtype=np.float32)
        
        # Generate price series following piecewise pattern
        base_price = 170.0
        prices = [base_price]
        
        for i in range(1, n_periods):
            alpha_return = alpha_signals[i] * 0.01
            noise = np.random.normal(0, 0.001)  # Small noise
            price_change = alpha_return + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Create features: 12 random + piecewise alpha + OFF period indicator
        random_features = np.random.randn(n_periods, 12).astype(np.float32) * 0.05
        alpha_feature = alpha_signals.reshape(-1, 1)
        off_indicator_feature = off_period_indicators.reshape(-1, 1)
        features = np.hstack([random_features, alpha_feature, off_indicator_feature])  # 14 features total
        
        price_series = pd.Series(prices, index=trading_days)
        
        # Calculate metadata
        on_periods = np.sum(alpha_signals != 0)
        off_periods = np.sum(alpha_signals == 0)
        
        metadata = {
            'pattern': 'piecewise',
            'alpha_strength': alpha_strength,
            'on_duration': on_duration,
            'off_duration': off_duration,
            'on_periods': int(on_periods),
            'off_periods': int(off_periods),
            'signal_consistency': float(on_periods / n_periods),
            'signal_mean': float(np.mean(alpha_signals)),
            'signal_std': float(np.std(alpha_signals)),
            'has_position_decay_signal': True,
            'off_period_ratio': float(off_periods / n_periods)
        }
        
        logger.info(f"   On periods: {on_periods} ({on_periods/n_periods:.1%})")
        logger.info(f"   Off periods: {off_periods} ({off_periods/n_periods:.1%})")
        logger.info(f"   Signal mean: {metadata['signal_mean']:.3f}")
        
        return features, price_series, metadata
    
    def create_low_noise_alpha(self, n_periods: int, base_alpha: float, 
                              noise_std: float = 0.05) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Phase 2: Low-noise alpha - base signal with small gaussian noise
        
        Args:
            n_periods: Number of time periods
            base_alpha: Base alpha strength (e.g., 0.20)
            noise_std: Standard deviation of noise (e.g., 0.05)
        """
        
        logger.info(f"🎯 Creating LOW-NOISE alpha data:")
        logger.info(f"   Periods: {n_periods}")
        logger.info(f"   Base alpha: {base_alpha}")
        logger.info(f"   Noise std: {noise_std}")
        
        # Create trading days
        trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
        
        # Generate low-noise alpha: base + gaussian noise
        noise = np.random.normal(0, noise_std, n_periods)
        alpha_signals = base_alpha + noise
        alpha_signals = alpha_signals.astype(np.float32)
        
        # Generate price series following noisy alpha
        base_price = 170.0
        prices = [base_price]
        
        for i in range(1, n_periods):
            alpha_return = alpha_signals[i] * 0.01
            market_noise = np.random.normal(0, 0.002)  # Market noise
            price_change = alpha_return + market_noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Create features: 12 random + noisy alpha
        random_features = np.random.randn(n_periods, 12).astype(np.float32) * 0.1
        alpha_feature = alpha_signals.reshape(-1, 1)
        features = np.hstack([random_features, alpha_feature])
        
        price_series = pd.Series(prices, index=trading_days)
        
        # Calculate signal statistics
        positive_signals = np.sum(alpha_signals > 0)
        negative_signals = np.sum(alpha_signals < 0)
        
        metadata = {
            'pattern': 'low_noise',
            'base_alpha': base_alpha,
            'noise_std': noise_std,
            'signal_mean': float(np.mean(alpha_signals)),
            'signal_std': float(np.std(alpha_signals)),
            'positive_signals': int(positive_signals),
            'negative_signals': int(negative_signals),
            'signal_consistency': float(positive_signals / n_periods) if base_alpha > 0 else float(negative_signals / n_periods)
        }
        
        logger.info(f"   Signal mean: {metadata['signal_mean']:.3f}")
        logger.info(f"   Signal std: {metadata['signal_std']:.3f}")
        logger.info(f"   Positive signals: {positive_signals} ({positive_signals/n_periods:.1%})")
        
        return features, price_series, metadata
    
    def create_realistic_noisy_alpha(self, n_periods: int, alpha_strength: float) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Phase 3: Realistic noisy alpha - MA crossover + momentum patterns
        Uses the original alpha_signal_generator approach
        """
        
        logger.info(f"🎯 Creating REALISTIC NOISY alpha data:")
        logger.info(f"   Periods: {n_periods}")
        logger.info(f"   Alpha strength: {alpha_strength}")
        logger.info(f"   Pattern: MA crossover + momentum (realistic)")
        
        # Import the original alpha generator
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path('.') / 'src' / 'features'))
        from alpha_signal_generator import create_toy_alpha_data
        
        # Use original implementation for realistic noisy signals
        features, price_series, metadata = create_toy_alpha_data(
            n_periods=n_periods,
            seed=self.seed,
            alpha_strength=alpha_strength
        )
        
        # Enhance metadata for consistency
        enhanced_metadata = {
            'pattern': 'realistic_noisy',
            'alpha_strength': alpha_strength,
            'signal_mean': metadata.get('signal_mean', 0.0),
            'signal_std': metadata.get('signal_std', 0.0),
            'bullish_signals': metadata.get('bullish_signals', 0),
            'bearish_signals': metadata.get('bearish_signals', 0),
            'neutral_signals': metadata.get('neutral_signals', 0),
            'signal_consistency': 0.5  # Realistic noisy signals have no clear consistency
        }
        
        logger.info(f"   Bullish signals: {enhanced_metadata['bullish_signals']}")
        logger.info(f"   Bearish signals: {enhanced_metadata['bearish_signals']}")
        logger.info(f"   Signal complexity: HIGH (MA + momentum)")
        
        return features, price_series, enhanced_metadata
    
    def generate_alpha_data(self, n_periods: int, **kwargs) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
        """
        Main interface - generate alpha data based on current pattern
        
        Args:
            n_periods: Number of time periods
            **kwargs: Pattern-specific parameters
        """
        
        if self.pattern == AlphaPattern.PERSISTENT:
            alpha_strength = kwargs.get('alpha_strength', 0.40)
            bull_episode = kwargs.get('bull_episode', True)
            return self.create_persistent_alpha(n_periods, alpha_strength, bull_episode)
            
        elif self.pattern == AlphaPattern.PIECEWISE:
            alpha_strength = kwargs.get('alpha_strength', 0.30)
            on_duration = kwargs.get('on_duration', 500)
            off_duration = kwargs.get('off_duration', 250)
            return self.create_piecewise_alpha(n_periods, alpha_strength, on_duration, off_duration)
            
        elif self.pattern == AlphaPattern.LOW_NOISE:
            base_alpha = kwargs.get('base_alpha', 0.20)
            noise_std = kwargs.get('noise_std', 0.05)
            return self.create_low_noise_alpha(n_periods, base_alpha, noise_std)
            
        elif self.pattern == AlphaPattern.REALISTIC_NOISY:
            alpha_strength = kwargs.get('alpha_strength', 0.15)
            return self.create_realistic_noisy_alpha(n_periods, alpha_strength)
            
        else:
            raise ValueError(f"Unknown alpha pattern: {self.pattern}")

def test_curriculum_generators():
    """Test all curriculum alpha generators"""
    
    print("🧪 TESTING CURRICULUM ALPHA GENERATORS")
    
    patterns = [
        (AlphaPattern.PERSISTENT, {'alpha_strength': 0.40, 'bull_episode': True}),
        (AlphaPattern.PIECEWISE, {'alpha_strength': 0.30}),
        (AlphaPattern.LOW_NOISE, {'base_alpha': 0.20, 'noise_std': 0.05}),
        (AlphaPattern.REALISTIC_NOISY, {'alpha_strength': 0.15})
    ]
    
    for pattern, kwargs in patterns:
        print(f"\n{'='*60}")
        print(f"Testing {pattern.value.upper()}")
        print(f"{'='*60}")
        
        generator = CurriculumAlphaGenerator(pattern, seed=42)
        features, prices, metadata = generator.generate_alpha_data(1000, **kwargs)
        
        print(f"✅ Generated: {features.shape[0]} periods, {features.shape[1]} features")
        print(f"✅ Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"✅ Metadata: {metadata}")

if __name__ == "__main__":
    test_curriculum_generators()