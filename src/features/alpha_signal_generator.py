#!/usr/bin/env python3
"""
🎯 ALPHA SIGNAL GENERATOR
Minimal alpha features to test if V3 safety net allows profitable trading
when genuine edge exists

Signals included:
1. Moving Average Crossover (fast MA > slow MA = bullish)
2. Momentum (recent return vs longer-term return)
3. Future Return Leak (ε-fraction for testing purposes)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AlphaSignalGenerator:
    """Generate minimal alpha signals for testing V3 + alpha combination"""
    
    def __init__(
        self,
        # MA Crossover parameters
        fast_ma_window: int = 5,
        slow_ma_window: int = 20,
        ma_signal_strength: float = 0.1,
        
        # Momentum parameters
        short_momentum_window: int = 3,
        long_momentum_window: int = 15,
        momentum_signal_strength: float = 0.05,
        
        # Future return leak (for testing only)
        future_leak_window: int = 5,
        future_leak_strength: float = 0.02,
        enable_future_leak: bool = False,
        
        # Signal combination
        signal_smoothing: float = 0.3,
        min_signal_threshold: float = 0.01,
        
        verbose: bool = False
    ):
        self.fast_ma_window = fast_ma_window
        self.slow_ma_window = slow_ma_window
        self.ma_signal_strength = ma_signal_strength
        
        self.short_momentum_window = short_momentum_window
        self.long_momentum_window = long_momentum_window
        self.momentum_signal_strength = momentum_signal_strength
        
        self.future_leak_window = future_leak_window
        self.future_leak_strength = future_leak_strength
        self.enable_future_leak = enable_future_leak
        
        self.signal_smoothing = signal_smoothing
        self.min_signal_threshold = min_signal_threshold
        self.verbose = verbose
        
        # State tracking
        self.previous_alpha_signal = 0.0
        
        if verbose:
            logger.info(f"🎯 AlphaSignalGenerator initialized:")
            logger.info(f"   📈 MA Crossover: {fast_ma_window}/{slow_ma_window} windows, strength {ma_signal_strength}")
            logger.info(f"   🚀 Momentum: {short_momentum_window}/{long_momentum_window} windows, strength {momentum_signal_strength}")
            if enable_future_leak:
                logger.info(f"   🔮 Future leak: {future_leak_window} window, strength {future_leak_strength} (TESTING ONLY)")
    
    def generate_alpha_features(
        self, 
        price_series: pd.Series,
        original_features: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate alpha features from price series
        
        Returns:
            enhanced_features: Original features + alpha signals
            alpha_metadata: Information about generated signals
        """
        
        n_periods = len(price_series)
        prices = price_series.values
        
        # Initialize alpha signals
        ma_crossover_signal = np.zeros(n_periods)
        momentum_signal = np.zeros(n_periods)
        future_leak_signal = np.zeros(n_periods)
        combined_alpha_signal = np.zeros(n_periods)
        
        # 1. Moving Average Crossover Signal
        if n_periods >= self.slow_ma_window:
            fast_ma = pd.Series(prices).rolling(self.fast_ma_window).mean().values
            slow_ma = pd.Series(prices).rolling(self.slow_ma_window).mean().values
            
            # Signal: +1 when fast > slow (bullish), -1 when fast < slow (bearish)
            for i in range(self.slow_ma_window, n_periods):
                if not (np.isnan(fast_ma[i]) or np.isnan(slow_ma[i])):
                    if fast_ma[i] > slow_ma[i]:
                        ma_crossover_signal[i] = self.ma_signal_strength
                    elif fast_ma[i] < slow_ma[i]:
                        ma_crossover_signal[i] = -self.ma_signal_strength
        
        # 2. Momentum Signal
        if n_periods >= self.long_momentum_window:
            # Short-term momentum vs long-term momentum
            short_returns = pd.Series(prices).pct_change(self.short_momentum_window).values
            long_returns = pd.Series(prices).pct_change(self.long_momentum_window).values
            
            for i in range(self.long_momentum_window, n_periods):
                if not (np.isnan(short_returns[i]) or np.isnan(long_returns[i])):
                    momentum_diff = short_returns[i] - long_returns[i]
                    momentum_signal[i] = momentum_diff * self.momentum_signal_strength
        
        # 3. Future Return Leak (TESTING ONLY)
        if self.enable_future_leak and n_periods >= self.future_leak_window:
            future_returns = pd.Series(prices).pct_change(self.future_leak_window).shift(-self.future_leak_window).values
            
            for i in range(n_periods - self.future_leak_window):
                if not np.isnan(future_returns[i]):
                    future_leak_signal[i] = future_returns[i] * self.future_leak_strength
        
        # 4. Combine Signals with Smoothing
        raw_combined = ma_crossover_signal + momentum_signal + future_leak_signal
        
        # Apply smoothing to prevent signal whipsawing
        for i in range(1, n_periods):
            combined_alpha_signal[i] = (
                self.signal_smoothing * self.previous_alpha_signal + 
                (1 - self.signal_smoothing) * raw_combined[i]
            )
            self.previous_alpha_signal = combined_alpha_signal[i]
        
        # Apply minimum threshold to filter noise
        combined_alpha_signal = np.where(
            np.abs(combined_alpha_signal) < self.min_signal_threshold,
            0.0,
            combined_alpha_signal
        )
        
        # Create enhanced feature matrix
        if original_features is not None:
            # Add alpha signal as additional feature
            alpha_feature = combined_alpha_signal.reshape(-1, 1)
            enhanced_features = np.hstack([original_features, alpha_feature])
        else:
            # Create minimal feature set with just alpha signal
            enhanced_features = combined_alpha_signal.reshape(-1, 1)
        
        # Calculate statistics
        alpha_metadata = {
            'n_periods': n_periods,
            'alpha_signal_mean': np.mean(combined_alpha_signal),
            'alpha_signal_std': np.std(combined_alpha_signal),
            'alpha_signal_range': (np.min(combined_alpha_signal), np.max(combined_alpha_signal)),
            'bullish_signals': np.sum(combined_alpha_signal > self.min_signal_threshold),
            'bearish_signals': np.sum(combined_alpha_signal < -self.min_signal_threshold),
            'neutral_signals': np.sum(np.abs(combined_alpha_signal) <= self.min_signal_threshold),
            'signal_components': {
                'ma_crossover_contribution': np.mean(np.abs(ma_crossover_signal)),
                'momentum_contribution': np.mean(np.abs(momentum_signal)),
                'future_leak_contribution': np.mean(np.abs(future_leak_signal)) if self.enable_future_leak else 0.0
            }
        }
        
        if self.verbose:
            logger.info(f"🎯 Alpha signals generated:")
            logger.info(f"   📊 Signal strength: μ={alpha_metadata['alpha_signal_mean']:.4f}, σ={alpha_metadata['alpha_signal_std']:.4f}")
            logger.info(f"   📈 Bullish signals: {alpha_metadata['bullish_signals']} ({alpha_metadata['bullish_signals']/n_periods:.1%})")
            logger.info(f"   📉 Bearish signals: {alpha_metadata['bearish_signals']} ({alpha_metadata['bearish_signals']/n_periods:.1%})")
            logger.info(f"   ⚪ Neutral signals: {alpha_metadata['neutral_signals']} ({alpha_metadata['neutral_signals']/n_periods:.1%})")
        
        return enhanced_features.astype(np.float32), alpha_metadata
    
    def reset(self):
        """Reset signal generator state"""
        self.previous_alpha_signal = 0.0
        
        if self.verbose:
            logger.info("🔄 AlphaSignalGenerator state reset")

def create_alpha_enhanced_data(
    price_series: pd.Series,
    original_features: np.ndarray = None,
    alpha_config: Dict[str, Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Factory function to create alpha-enhanced feature data
    
    Args:
        price_series: Price data for signal generation
        original_features: Existing features to enhance (optional)
        alpha_config: Configuration for alpha signal generator
        
    Returns:
        enhanced_features: Features with alpha signals
        alpha_metadata: Information about generated signals
    """
    
    if alpha_config is None:
        alpha_config = {
            'fast_ma_window': 5,
            'slow_ma_window': 20,
            'ma_signal_strength': 0.1,
            'momentum_signal_strength': 0.05,
            'enable_future_leak': False,  # Disable by default
            'verbose': True
        }
    
    generator = AlphaSignalGenerator(**alpha_config)
    enhanced_features, metadata = generator.generate_alpha_features(price_series, original_features)
    
    return enhanced_features, metadata

# Convenience function for testing
def create_toy_alpha_data(n_periods: int = 1000, seed: int = 42, alpha_strength: float = 0.1):
    """Create toy data with obvious alpha signal for testing"""
    
    np.random.seed(seed)
    
    # Create realistic price series
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    base_price = 170.0
    returns = np.random.normal(0.0001, 0.01, n_periods)  # Moderate volatility
    prices = base_price * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=trading_days)
    
    # Create basic features (12 features)
    original_features = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Add obvious alpha signal
    alpha_config = {
        'fast_ma_window': 5,
        'slow_ma_window': 15,
        'ma_signal_strength': alpha_strength,
        'momentum_signal_strength': alpha_strength * 0.5,
        'enable_future_leak': True,        # Enable for testing
        'future_leak_strength': alpha_strength * 0.2,
        'verbose': True
    }
    
    enhanced_features, metadata = create_alpha_enhanced_data(
        price_series, original_features, alpha_config
    )
    
    logger.info(f"🎯 Toy alpha data created: {n_periods} periods with {enhanced_features.shape[1]} features")
    logger.info(f"   Alpha strength: {alpha_strength}, Metadata: {metadata['bullish_signals']} bullish, {metadata['bearish_signals']} bearish")
    
    return enhanced_features, price_series, metadata