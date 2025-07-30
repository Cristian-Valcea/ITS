#!/usr/bin/env python3
"""
🎯 DUAL-TICKER TRADING ENVIRONMENT V3
Dual-ticker (NVDA + MSFT) environment with V3 reward system integration

SCAFFOLDING PHILOSOPHY:
- Clone architecture from single-ticker env_v3
- Extend observation space: 12 + 1 + 12 + 1 = 26 dimensions
- Extend action space: 3x3 = 9 portfolio actions  
- Keep V3 reward system UNTOUCHED (proven calibrated)
- Add alpha feature support for both tickers
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gym_env.dual_reward_v3 import DualTickerRewardV3

logger = logging.getLogger(__name__)

class DualTickerTradingEnvV3(gym.Env):
    """
    V3 Dual-Ticker Trading Environment with calibrated reward system
    
    KEY FEATURES:
    - V3 reward system (risk-free baseline + 68bp calibrated impact)
    - 26-dimensional observation space (12+1 per ticker + alpha features)
    - 9-action portfolio matrix (3x3 combinations)
    - Alpha signal support for both tickers
    - Safety rails: 2% DD limit, cost-blind trading prevention
    """
    
    def __init__(
        self,
        # Data parameters
        processed_feature_data: np.ndarray,
        price_data: pd.Series,
        initial_capital: float = 100000.0,
        
        # Environment parameters
        lookback_window: int = 50,
        max_episode_steps: int = 1000,
        max_daily_drawdown_pct: float = 0.02,
        
        # Trading parameters  
        max_position_size: int = 500,        # Max shares per ticker
        transaction_cost_pct: float = 0.0001,  # Additional friction (1bp)
        
        # V3 reward parameters (calibrated - DO NOT MODIFY)
        base_impact_bp: float = 68.0,        # Calibrated impact strength
        impact_exponent: float = 0.5,        # sqrt scaling
        risk_free_rate_annual: float = 0.05, # 5% risk-free rate
        
        # Logging
        log_trades: bool = False,
        verbose: bool = False
    ):
        super().__init__()
        
        # Store parameters
        self.processed_feature_data = processed_feature_data.astype(np.float32)
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.max_episode_steps = max_episode_steps
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_position_size = max_position_size
        self.transaction_cost_pct = transaction_cost_pct
        self.log_trades = log_trades
        self.verbose = verbose
        
        # Data validation
        if len(self.processed_feature_data) != len(self.price_data):
            raise ValueError(f"Feature data length {len(self.processed_feature_data)} != price data length {len(self.price_data)}")
        
        # Determine feature dimensions
        base_features_per_ticker = 12  # Standard OHLCV + technical indicators
        has_alpha_features = self.processed_feature_data.shape[1] > (base_features_per_ticker * 2)
        
        if has_alpha_features:
            # Features: 12 NVDA + 12 MSFT + 1 NVDA alpha + 1 MSFT alpha = 26
            expected_features = (base_features_per_ticker * 2) + 2
            alpha_features = 2
        else:
            # Features: 12 NVDA + 12 MSFT = 24
            expected_features = base_features_per_ticker * 2
            alpha_features = 0
        
        if self.processed_feature_data.shape[1] != expected_features:
            logger.warning(f"Expected {expected_features} features, got {self.processed_feature_data.shape[1]}")
            logger.warning(f"Assuming {base_features_per_ticker} features per ticker + {alpha_features} alpha features")
        
        # Define observation and action spaces
        # Observation: [12 NVDA features + 1 NVDA position + 12 MSFT features + 1 MSFT position] + optional alpha
        obs_dim = expected_features + 2  # +2 for position tracking
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 9 discrete actions (3x3 portfolio matrix)
        # 0=SELL_BOTH, 1=SELL_NVDA_HOLD_MSFT, 2=SELL_NVDA_BUY_MSFT,
        # 3=HOLD_NVDA_SELL_MSFT, 4=HOLD_BOTH, 5=HOLD_NVDA_BUY_MSFT,
        # 6=BUY_NVDA_SELL_MSFT, 7=BUY_NVDA_HOLD_MSFT, 8=BUY_BOTH
        self.action_space = spaces.Discrete(9)
        
        # Action mapping: action_id -> (nvda_action, msft_action)
        self.action_map = {
            0: (0, 0),  # SELL_BOTH
            1: (0, 1),  # SELL_NVDA_HOLD_MSFT  
            2: (0, 2),  # SELL_NVDA_BUY_MSFT
            3: (1, 0),  # HOLD_NVDA_SELL_MSFT
            4: (1, 1),  # HOLD_BOTH
            5: (1, 2),  # HOLD_NVDA_BUY_MSFT
            6: (2, 0),  # BUY_NVDA_SELL_MSFT
            7: (2, 1),  # BUY_NVDA_HOLD_MSFT
            8: (2, 2),  # BUY_BOTH
        }
        
        # Single-ticker action mapping: 0=SELL, 1=HOLD, 2=BUY
        self.ticker_action_sizes = {
            0: -self.max_position_size,  # SELL
            1: 0,                        # HOLD  
            2: self.max_position_size,   # BUY
        }
        
        # Initialize V3 reward system (CALIBRATED - DO NOT MODIFY)
        self.reward_calculator = DualTickerRewardV3(
            risk_free_rate_annual=risk_free_rate_annual,
            base_impact_bp=base_impact_bp,       # 68bp calibrated
            impact_exponent=impact_exponent,     # sqrt scaling
            adv_scaling=40000000.0,              # 40M NVDA ADV
            step_minutes=1.0,                    # 1-minute bars
            verbose=verbose
        )
        
        # Environment state
        self.reset()
        
        if verbose:
            logger.info(f"🎯 DualTickerTradingEnvV3 initialized:")
            logger.info(f"   📊 Data: {len(self.processed_feature_data)} steps, {self.processed_feature_data.shape[1]} features")
            logger.info(f"   📐 Obs space: {self.observation_space.shape}, Action space: {self.action_space.n}")
            logger.info(f"   💰 Capital: ${initial_capital:,.0f}")
            logger.info(f"   🛡️ Max DD: {max_daily_drawdown_pct:.1%}")
            logger.info(f"   🎯 V3 reward: Risk-free baseline with {base_impact_bp}bp calibrated impact")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = self.lookback_window
        self.episode_start_step = self.current_step
        
        # Reset portfolio state
        self.portfolio_value = self.initial_capital
        self.cash_balance = self.initial_capital
        self.nvda_position = 0.0  # Shares held
        self.msft_position = 0.0  # Shares held
        
        # Reset tracking
        self.peak_portfolio_value = self.initial_capital
        self.episode_trades = []
        self.step_count = 0
        
        # Reset V3 reward calculator
        self.reward_calculator.reset()
        
        # Get initial observation
        obs = self._get_observation()
        
        if self.verbose:
            logger.info(f"🔄 DualTickerEnvV3 reset - Episode starting at step {self.current_step}")
        
        return obs, {}
    
    def step(self, action: int):
        """Execute one environment step"""
        
        if self.current_step >= len(self.processed_feature_data) - 1:
            # Episode terminated due to data exhaustion
            obs = self._get_observation()
            info = self.get_info()
            return obs, 0.0, True, False, info
        
        # Store previous state for reward calculation
        prev_portfolio_value = self.portfolio_value
        prev_nvda_position = self.nvda_position
        prev_msft_position = self.msft_position
        
        # Get current prices
        current_prices = self._get_current_prices()
        nvda_price = current_prices['nvda']
        msft_price = current_prices['msft']
        
        # Decode action into individual ticker actions
        nvda_action, msft_action = self.action_map[action]
        
        # Calculate trade sizes
        nvda_trade_shares = self.ticker_action_sizes[nvda_action]
        msft_trade_shares = self.ticker_action_sizes[msft_action]
        
        # Execute trades
        nvda_trade_value = self._execute_trade('NVDA', nvda_trade_shares, nvda_price)
        msft_trade_value = self._execute_trade('MSFT', msft_trade_shares, msft_price)
        
        # Update portfolio value
        self._update_portfolio_value(nvda_price, msft_price)
        
        # Calculate V3 reward
        reward, reward_components = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            curr_portfolio_value=self.portfolio_value,
            nvda_trade_value=abs(nvda_trade_value),
            msft_trade_value=abs(msft_trade_value),
            nvda_position=self.nvda_position,
            msft_position=self.msft_position,
            nvda_price=nvda_price,
            msft_price=msft_price,
            step=self.step_count
        )
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False
        
        # Move to next step
        self.current_step += 1
        self.step_count += 1
        
        # Update peak tracking
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # Log trade if enabled
        if self.log_trades and (nvda_trade_value != 0 or msft_trade_value != 0):
            self._log_trade(action, nvda_trade_value, msft_trade_value, reward, reward_components)
        
        # Get next observation
        obs = self._get_observation()
        info = self.get_info()
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including positions and alpha features"""
        
        # Get base features at current step
        base_features = self.processed_feature_data[self.current_step].copy()
        
        # Add position information (normalized)
        nvda_position_norm = self.nvda_position / self.max_position_size
        msft_position_norm = self.msft_position / self.max_position_size
        
        # Construct observation: [features + nvda_position + msft_position]
        obs = np.concatenate([
            base_features,
            [nvda_position_norm, msft_position_norm]
        ]).astype(np.float32)
        
        return obs
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for both tickers"""
        
        # For now, assume single price series represents NVDA
        # MSFT price derived with realistic ratio (MSFT ~3x NVDA)
        nvda_price = float(self.price_data.iloc[self.current_step])
        msft_price = nvda_price * 3.0  # Realistic MSFT/NVDA price ratio
        
        return {'nvda': nvda_price, 'msft': msft_price}
    
    def _execute_trade(self, ticker: str, trade_shares: float, price: float) -> float:
        """Execute trade for specific ticker"""
        
        if trade_shares == 0:
            return 0.0
        
        trade_value = abs(trade_shares * price)
        
        # Apply additional transaction costs (V3 impact is handled in reward)
        transaction_cost = trade_value * self.transaction_cost_pct
        
        # Update positions and cash
        if ticker == 'NVDA':
            self.nvda_position += trade_shares
            self.cash_balance -= (trade_shares * price + transaction_cost)
        elif ticker == 'MSFT':  
            self.msft_position += trade_shares
            self.cash_balance -= (trade_shares * price + transaction_cost)
        
        # Store trade info
        self.episode_trades.append({
            'step': self.current_step,
            'ticker': ticker,
            'shares': trade_shares,
            'price': price,
            'value': trade_shares * price,
            'cost': transaction_cost
        })
        
        return trade_shares * price  # Return signed trade value
    
    def _update_portfolio_value(self, nvda_price: float, msft_price: float):
        """Update total portfolio value"""
        
        nvda_market_value = self.nvda_position * nvda_price
        msft_market_value = self.msft_position * msft_price
        
        self.portfolio_value = self.cash_balance + nvda_market_value + msft_market_value
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Max steps reached
        if self.step_count >= self.max_episode_steps:
            return True
        
        # Data exhaustion
        if self.current_step >= len(self.processed_feature_data) - 1:
            return True
        
        # Drawdown limit exceeded
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if current_drawdown > self.max_daily_drawdown_pct:
                if self.verbose:
                    logger.info(f"Episode terminated: Drawdown {current_drawdown:.2%} > {self.max_daily_drawdown_pct:.2%}")
                return True
        
        return False
    
    def _log_trade(self, action: int, nvda_trade_value: float, msft_trade_value: float, 
                  reward: float, reward_components):
        """Log trade details"""
        
        action_names = [
            "SELL_BOTH", "SELL_NVDA_HOLD_MSFT", "SELL_NVDA_BUY_MSFT",
            "HOLD_NVDA_SELL_MSFT", "HOLD_BOTH", "HOLD_NVDA_BUY_MSFT", 
            "BUY_NVDA_SELL_MSFT", "BUY_NVDA_HOLD_MSFT", "BUY_BOTH"
        ]
        
        logger.info(f"Step {self.step_count}: {action_names[action]} | "
                   f"NVDA ${nvda_trade_value:+.0f}, MSFT ${msft_trade_value:+.0f} | "
                   f"Portfolio ${self.portfolio_value:,.0f} | Reward {reward:+.2f}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        
        current_prices = self._get_current_prices()
        
        # Calculate returns and drawdown
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        return {
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'nvda_position': self.nvda_position,
            'msft_position': self.msft_position,
            'nvda_price': current_prices['nvda'],
            'msft_price': current_prices['msft'],
            'total_return': total_return,
            'current_drawdown': current_drawdown,
            'peak_portfolio_value': self.peak_portfolio_value,
            'episode_trades': len(self.episode_trades),
            'step_count': self.step_count
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            info = self.get_info()
            print(f"Step: {self.step_count} | Portfolio: ${info['portfolio_value']:,.0f} | "
                  f"Return: {info['total_return']:+.2%} | DD: {info['current_drawdown']:.2%} | "
                  f"NVDA: {info['nvda_position']:.0f}@${info['nvda_price']:.2f} | "
                  f"MSFT: {info['msft_position']:.0f}@${info['msft_price']:.2f}")

def create_dual_ticker_alpha_data(n_periods: int = 2000, seed: int = 42, alpha_strength: float = 0.1):
    """Create dual-ticker data with alpha signals for both assets"""
    
    np.random.seed(seed)
    
    # Create price series for both assets
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # NVDA price series
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.01, n_periods)
    nvda_prices = nvda_base_price * np.exp(np.cumsum(nvda_returns))
    nvda_series = pd.Series(nvda_prices, index=trading_days)
    
    # Create base features: 12 per ticker = 24 total
    nvda_features = np.random.randn(n_periods, 12).astype(np.float32)
    msft_features = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Create alpha signals for both tickers
    alpha_signals_nvda = []
    alpha_signals_msft = []
    
    for i in range(n_periods):
        # Simple cyclical alpha pattern
        if i % 30 < 15:  # First half of cycle
            alpha_nvda = alpha_strength
            alpha_msft = -alpha_strength * 0.5  # Opposite correlation
        else:  # Second half
            alpha_nvda = -alpha_strength
            alpha_msft = alpha_strength * 0.5
        
        # Add noise
        alpha_nvda += np.random.normal(0, alpha_strength * 0.1)
        alpha_msft += np.random.normal(0, alpha_strength * 0.1)
        
        alpha_signals_nvda.append(alpha_nvda)
        alpha_signals_msft.append(alpha_msft)
    
    # Combine features: [12 NVDA + 12 MSFT + 1 NVDA alpha + 1 MSFT alpha] = 26
    alpha_nvda_feature = np.array(alpha_signals_nvda).reshape(-1, 1).astype(np.float32)
    alpha_msft_feature = np.array(alpha_signals_msft).reshape(-1, 1).astype(np.float32)
    
    enhanced_features = np.hstack([
        nvda_features,
        msft_features, 
        alpha_nvda_feature,
        alpha_msft_feature
    ])
    
    logger.info(f"🎯 Dual-ticker alpha data created:")
    logger.info(f"   Periods: {n_periods}")
    logger.info(f"   Features: {enhanced_features.shape} (12+12 base + 2 alpha)")
    logger.info(f"   Alpha strength: {alpha_strength}")
    logger.info(f"   Pattern: 15-step cyclical with noise")
    
    return enhanced_features, nvda_series

# Test function
def test_dual_ticker_env_v3():
    """Test dual-ticker V3 environment"""
    
    logger.info("🧪 Testing DualTickerTradingEnvV3...")
    
    # Create test data
    features, prices = create_dual_ticker_alpha_data(1000, seed=42, alpha_strength=0.1)
    
    # Create environment
    env = DualTickerTradingEnvV3(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=100000,
        max_daily_drawdown_pct=0.02,
        base_impact_bp=68.0,  # Calibrated impact
        verbose=True
    )
    
    # Test basic functionality
    obs, info = env.reset()
    logger.info(f"   Observation shape: {obs.shape}")
    logger.info(f"   Action space: {env.action_space}")
    
    # Test a few steps
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 3 == 0:
            logger.info(f"   Step {step}: Action {action}, Reward {reward:.2f}, Portfolio ${info['portfolio_value']:,.0f}")
        
        if terminated or truncated:
            break
    
    logger.info("✅ DualTickerTradingEnvV3 test completed")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dual_ticker_env_v3()