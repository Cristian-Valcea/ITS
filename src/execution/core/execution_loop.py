"""
Execution Loop Core Module

Contains the main trading execution loop logic extracted from OrchestratorAgent.
This module handles:
- Live trading loop coordination
- Real-time bar processing
- Trading state management
- Event-driven trading logic

This is an internal module - use src.execution.OrchestratorAgent for public API.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import pandas as pd
import numpy as np

# TODO: Import statements will be added during extraction phase


class ExecutionLoop:
    """
    Core execution loop for live trading.
    
    Handles the main trading loop logic, real-time data processing,
    and coordination between different trading components.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the execution loop.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.is_running = False
        self.trading_state: Dict[str, Any] = {}
        
        # Event hooks for extensibility
        self.hooks: Dict[str, Callable] = {}
        
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for a specific event."""
        self.hooks[event] = callback
        
    def _trigger_hook(self, event: str, *args, **kwargs) -> None:
        """Trigger a registered hook if it exists."""
        if event in self.hooks:
            try:
                self.hooks[event](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook {event} failed: {e}")
    
    async def start_live_trading_loop(
        self, 
        symbol: str, 
        data_agent=None, 
        feature_agent=None, 
        risk_agent=None, 
        live_model=None
    ) -> None:
        """
        Start the main live trading loop for a symbol.
        
        Args:
            symbol: Trading symbol
            data_agent: Data agent for fetching market data
            feature_agent: Feature engineering agent
            risk_agent: Risk management agent
            live_model: Trained model for predictions
        """
        self.logger.info(f"Starting live trading loop for {symbol}")
        self.is_running = True
        self._trigger_hook("trading_started", symbol)
        
        try:
            # Initialize trading state for this symbol
            self.trading_state[symbol] = {
                'last_bar_time': None,
                'position': 0,
                'last_action': 0,
                'consecutive_actions': 0
            }
            
            # Main trading loop
            while self.is_running:
                try:
                    # Fetch latest market data
                    if data_agent:
                        latest_data = data_agent.get_latest_bar(symbol)
                        if latest_data is not None and not latest_data.empty:
                            await self._process_new_bar(
                                latest_data, symbol, feature_agent, risk_agent, live_model
                            )
                    
                    # Sleep for a short interval before next iteration
                    await asyncio.sleep(1)  # 1 second interval
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop iteration: {e}")
                    await asyncio.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False
            self._trigger_hook("trading_stopped", symbol)
            
    async def _process_new_bar(
        self, 
        new_bar_df: pd.DataFrame, 
        symbol: str, 
        feature_agent=None, 
        risk_agent=None, 
        live_model=None
    ) -> None:
        """
        Process a new bar of market data.
        
        Args:
            new_bar_df: New market data bar
            symbol: Trading symbol
            feature_agent: Feature engineering agent
            risk_agent: Risk management agent
            live_model: Trained model for predictions
        """
        try:
            # Check if this is a new bar
            current_bar_time = new_bar_df.index[-1] if hasattr(new_bar_df.index[-1], 'timestamp') else pd.Timestamp.now()
            
            if (symbol in self.trading_state and 
                self.trading_state[symbol]['last_bar_time'] == current_bar_time):
                return  # Skip if same bar
                
            # Update last bar time
            self.trading_state[symbol]['last_bar_time'] = current_bar_time
            
            # Engineer features
            if feature_agent:
                features_df = feature_agent.engineer_features(new_bar_df)
                if features_df is None or features_df.empty:
                    self.logger.warning("Failed to engineer features")
                    return
                    
                # Get the latest feature row
                latest_features = features_df.iloc[-1]
                
                # Generate action using model
                if live_model:
                    action = self._generate_action(latest_features, live_model)
                    
                    # Trigger action generated hook
                    self._trigger_hook("action_generated", symbol, action, latest_features)
                    
                    # Update trading state
                    self.trading_state[symbol]['last_action'] = action
                    
        except Exception as e:
            self.logger.error(f"Error processing new bar: {e}")
            
    def _generate_action(self, features: pd.Series, model) -> int:
        """
        Generate trading action using the model.
        
        Args:
            features: Feature vector
            model: Trained model
            
        Returns:
            Action (0=hold, 1=buy, 2=sell)
        """
        try:
            # Convert features to numpy array
            feature_array = features.values.reshape(1, -1)
            
            # Get prediction from model
            action, _ = model.predict(feature_array, deterministic=True)
            
            return int(action[0]) if hasattr(action, '__iter__') else int(action)
            
        except Exception as e:
            self.logger.error(f"Error generating action: {e}")
            return 0  # Default to hold
        
    def stop_live_trading_loop(self) -> None:
        """Stop the live trading loop."""
        self.logger.info("Stopping live trading loop")
        self.is_running = False
        self._trigger_hook("trading_stopped")
        
    def process_incoming_bar(self, new_bar_df: pd.DataFrame, symbol: str) -> None:
        """
        Process a new incoming bar of market data.
        
        This method will be populated with the actual bar processing logic
        during the extraction phase.
        """
        # TODO: Extract from _process_incoming_bar in orchestrator_agent.py
        pass
        
    def update_trading_state(self, symbol: str, state_update: Dict[str, Any]) -> None:
        """Update the trading state for a symbol."""
        if symbol not in self.trading_state:
            self.trading_state[symbol] = {}
        self.trading_state[symbol].update(state_update)
        
    def get_trading_state(self, symbol: str) -> Dict[str, Any]:
        """Get the current trading state for a symbol."""
        return self.trading_state.get(symbol, {})