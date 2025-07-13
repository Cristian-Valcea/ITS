"""
BacktestRunner module for executing backtests.

This module handles:
- Running backtests with loaded models
- Environment interaction and step execution
- Trade logging and portfolio tracking
- Backtest progress monitoring
"""

import logging
from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.gym_env.intraday_trading_env import IntradayTradingEnv


class BacktestRunner:
    """
    Handles execution of backtests for trading model evaluation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the BacktestRunner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_log_interval = config.get('backtest_progress_log_interval', 100)
        
    def run_backtest(
        self, 
        model: Any, 
        evaluation_env: IntradayTradingEnv, 
        deterministic: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Run a backtest with the given model and environment.
        
        Args:
            model: The trading model to evaluate
            evaluation_env: The trading environment for evaluation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (trade_log_df, portfolio_history_series) or (None, None) if failed
        """
        if model is None:
            self.logger.error("No model provided. Cannot run backtest.")
            return None, None
            
        if evaluation_env is None:
            self.logger.error("Evaluation environment not provided. Cannot run backtest.")
            return None, None

        self.logger.info(f"Starting backtest on evaluation environment. Deterministic: {deterministic}")
        
        try:
            # Initialize backtest
            obs, info = evaluation_env.reset()
            terminated = False
            truncated = False
            total_steps = 0
            
            # Debug observation shape
            self.logger.info(f"Initial observation shape: {obs.shape}, type: {type(obs)}")
            self.logger.info(f"Model observation space: {model.observation_space}")
            self.logger.info(f"Environment observation space: {evaluation_env.observation_space}")
            
            # Fix observation shape mismatch - flatten if needed
            if len(obs.shape) > 1 and model.observation_space.shape != obs.shape:
                expected_shape = model.observation_space.shape
                if len(expected_shape) == 1 and np.prod(obs.shape) == expected_shape[0]:
                    self.logger.info(f"Flattening observation from {obs.shape} to {expected_shape}")
                    obs = obs.flatten()
            
            # Log initial state
            initial_portfolio_value = getattr(evaluation_env, 'portfolio_value', 0)
            self.logger.info(f"Backtest initialized. Initial portfolio value: {initial_portfolio_value:.2f}")

            # Run backtest loop
            while not (terminated or truncated):
                # Ensure observation shape matches model expectation
                if len(obs.shape) > 1 and model.observation_space.shape != obs.shape:
                    expected_shape = model.observation_space.shape
                    if len(expected_shape) == 1 and np.prod(obs.shape) == expected_shape[0]:
                        obs = obs.flatten()
                
                # Get action from model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Execute step in environment
                obs, reward, terminated, truncated, info = evaluation_env.step(action)
                total_steps += 1
                
                # Log progress periodically
                if total_steps % self.progress_log_interval == 0:
                    current_portfolio_value = info.get('portfolio_value', 'N/A')
                    self.logger.debug(
                        f"Backtest step {total_steps}, Action: {action}, "
                        f"Reward: {reward:.3f}, Portfolio Value: {current_portfolio_value:.2f}"
                    )

            # Log completion
            final_portfolio_value = getattr(evaluation_env, 'portfolio_value', 0)
            self.logger.info(f"Backtest finished after {total_steps} steps.")
            self.logger.info(f"Final portfolio value: {final_portfolio_value:.2f}")

            # Extract results
            trade_log_df, portfolio_history_s = self._extract_backtest_results(evaluation_env)
            
            return trade_log_df, portfolio_history_s
            
        except Exception as e:
            self.logger.error(f"Exception during backtest: {e}", exc_info=True)
            return None, None
    
    def _extract_backtest_results(
        self, 
        evaluation_env: IntradayTradingEnv
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Extract trade log and portfolio history from the environment.
        
        Args:
            evaluation_env: The trading environment after backtest completion
            
        Returns:
            Tuple of (trade_log_df, portfolio_history_series)
        """
        try:
            # Get trade log
            trade_log_df = evaluation_env.get_trade_log()
            if trade_log_df is not None and trade_log_df.empty:
                self.logger.warning("Backtest completed, but no trades were logged by the environment.")
            elif trade_log_df is not None:
                self.logger.info(f"Retrieved trade log with {len(trade_log_df)} entries.")
            else:
                self.logger.warning("Trade log is None.")

            # Get portfolio history
            portfolio_history_s = evaluation_env.get_portfolio_history()
            if portfolio_history_s is not None and portfolio_history_s.empty:
                self.logger.warning("Portfolio history is empty after backtest.")
            elif portfolio_history_s is not None:
                self.logger.info(f"Retrieved portfolio history with {len(portfolio_history_s)} entries.")
            else:
                self.logger.warning("Portfolio history is None.")

            return trade_log_df, portfolio_history_s
            
        except Exception as e:
            self.logger.error(f"Error extracting backtest results: {e}", exc_info=True)
            return None, None
    
    def validate_backtest_results(
        self, 
        trade_log_df: Optional[pd.DataFrame], 
        portfolio_history_s: Optional[pd.Series]
    ) -> bool:
        """
        Validate the backtest results for completeness and consistency.
        
        Args:
            trade_log_df: Trade log DataFrame
            portfolio_history_s: Portfolio history Series
            
        Returns:
            True if results are valid, False otherwise
        """
        validation_passed = True
        
        # Check trade log
        if trade_log_df is None:
            self.logger.warning("Trade log is None - this may be expected if no trades occurred.")
        elif trade_log_df.empty:
            self.logger.warning("Trade log is empty - no trades were executed during backtest.")
        else:
            # Validate trade log structure
            required_columns = ['timestamp', 'action', 'quantity', 'price']
            missing_columns = [col for col in required_columns if col not in trade_log_df.columns]
            if missing_columns:
                self.logger.error(f"Trade log missing required columns: {missing_columns}")
                validation_passed = False
            else:
                self.logger.info(f"Trade log validation passed: {len(trade_log_df)} trades recorded.")
        
        # Check portfolio history
        if portfolio_history_s is None:
            self.logger.error("Portfolio history is None - this indicates a serious issue.")
            validation_passed = False
        elif portfolio_history_s.empty:
            self.logger.error("Portfolio history is empty - this indicates a serious issue.")
            validation_passed = False
        else:
            # Check for reasonable portfolio values
            if (portfolio_history_s <= 0).any():
                self.logger.warning("Portfolio history contains non-positive values.")
            
            # Check for extreme volatility (optional warning)
            if len(portfolio_history_s) > 1:
                returns = portfolio_history_s.pct_change().dropna()
                if not returns.empty and (abs(returns) > 0.5).any():
                    self.logger.warning("Portfolio history shows extreme daily returns (>50%).")
            
            self.logger.info(f"Portfolio history validation passed: {len(portfolio_history_s)} data points.")
        
        if validation_passed:
            self.logger.info("Backtest results validation completed successfully.")
        else:
            self.logger.error("Backtest results validation failed.")
            
        return validation_passed
    
    def get_backtest_summary(
        self, 
        trade_log_df: Optional[pd.DataFrame], 
        portfolio_history_s: Optional[pd.Series],
        initial_capital: float
    ) -> dict:
        """
        Generate a quick summary of backtest results.
        
        Args:
            trade_log_df: Trade log DataFrame
            portfolio_history_s: Portfolio history Series
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary containing backtest summary
        """
        summary = {
            'total_steps': len(portfolio_history_s) if portfolio_history_s is not None else 0,
            'total_trades': len(trade_log_df) if trade_log_df is not None else 0,
            'initial_capital': initial_capital,
            'final_capital': portfolio_history_s.iloc[-1] if portfolio_history_s is not None and not portfolio_history_s.empty else initial_capital,
            'data_quality': 'Valid' if self.validate_backtest_results(trade_log_df, portfolio_history_s) else 'Invalid'
        }
        
        # Calculate basic return
        if summary['initial_capital'] > 0:
            summary['total_return_pct'] = ((summary['final_capital'] - summary['initial_capital']) / summary['initial_capital']) * 100
        else:
            summary['total_return_pct'] = 0.0
            
        return summary