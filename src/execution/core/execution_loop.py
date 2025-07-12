"""
Execution Loop Core Module

Contains the main trading execution loop logic extracted from OrchestratorAgent.
This module handles:
- Live trading loop coordination
- Real-time bar processing
- Trading state management
- Event-driven trading logic

This is an internal module - use src.execution.OrchestratorAgent for public API.

IMPORTANT SETUP REQUIREMENTS:
1. PYTHONPATH: This module requires the project root to be in PYTHONPATH.
   - For development: pip install -e . (editable install)
   - For production: Ensure PYTHONPATH includes the project root directory.

2. CONFIGURATION: Key config parameters:
   - bar_period_seconds: Bar processing interval (default: 1)
   - execution.feature_max_workers: Thread pool size for feature engineering
   - risk.fail_closed_on_missing_positions: Fail-closed mode for missing position data
   - risk.min_order_size: Minimum order size after throttling (default: 1)

3. INTEGRATION: Before live trading:
   - Wire real position tracker or enable fail_closed_on_missing_positions
   - Register RiskEventHandler via register_risk_event_handler()
   - Configure OrderRouter to handle minimum order sizes gracefully
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
import pandas as pd
import numpy as np
from enum import IntEnum

# Import required types and modules
if TYPE_CHECKING:
    from src.agents.data_agent import DataAgent
    from src.agents.feature_agent import FeatureAgent
    from src.agents.risk_agent import RiskAgent

# Action enumeration for type safety
class TradingAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


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
        self._state_lock = asyncio.Lock()  # Thread safety for trading state
        # NOTE: Single lock guards all symbols - fine for ≤10 symbols
        # For >10 symbols, consider per-symbol locks to reduce serialization
        
        # Event hooks for extensibility
        self.hooks: Dict[str, Callable] = {}
        
        # Configuration parameters
        self.bar_period = config.get('bar_period_seconds', 1)  # Configurable bar period
        
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for a specific event."""
        self.hooks[event] = callback
    
    def register_risk_event_handler(self, handler) -> None:
        """
        Register a RiskEventHandler for emergency stop and risk events.
        
        TODO: Call this method during system initialization to wire risk event handling.
        
        Args:
            handler: RiskEventHandler instance
        """
        self.risk_event_handler = handler
        self.logger.info("Risk event handler registered")
        
    def _trigger_hook(self, event: str, *args, **kwargs) -> None:
        """Trigger a registered hook if it exists (supports both sync and async hooks)."""
        if event in self.hooks:
            try:
                hook = self.hooks[event]
                if asyncio.iscoroutinefunction(hook):
                    # Create task for async hooks to avoid blocking
                    try:
                        loop = asyncio.get_running_loop()  # Python 3.7+ preferred method
                        loop.create_task(hook(*args, **kwargs))
                    except RuntimeError:
                        # No event loop running - skip async hook
                        self.logger.warning(f"Async hook {event} skipped - no event loop")
                else:
                    # Synchronous hook
                    hook(*args, **kwargs)
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
                    
                    # Sleep for configurable interval before next iteration
                    await asyncio.sleep(self.bar_period)
                    
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
        symbol: str,
        new_bar_df: pd.DataFrame, 
        feature_agent=None, 
        risk_agent=None, 
        live_model=None
    ) -> None:
        """
        Process a new bar of market data.
        
        Args:
            symbol: Trading symbol
            new_bar_df: New market data bar
            feature_agent: Feature engineering agent
            risk_agent: Risk management agent
            live_model: Trained model for predictions
        """
        try:
            # Check if this is a new bar - raise error if index is wrong type
            try:
                current_bar_time = new_bar_df.index[-1]
                if not isinstance(current_bar_time, (pd.Timestamp, datetime)):
                    raise ValueError(f"DataFrame index must be datetime-like, got {type(current_bar_time)}")
            except (IndexError, AttributeError) as e:
                raise ValueError(f"Invalid DataFrame index for bar processing: {e}")
            
            if (symbol in self.trading_state and 
                self.trading_state[symbol]['last_bar_time'] == current_bar_time):
                return  # Skip if same bar
                
            # Update last bar time
            self.trading_state[symbol]['last_bar_time'] = current_bar_time
            
            # Engineer features (run in thread pool to avoid blocking event loop)
            if feature_agent:
                try:
                    loop = asyncio.get_running_loop()  # Python 3.7+ preferred method
                except RuntimeError:
                    loop = asyncio.new_event_loop()  # Fallback if no loop running
                    asyncio.set_event_loop(loop)
                
                # Monitor executor queue size for latency issues
                # NOTE: If bars are 1s and features heavy, executor queue can back up
                # Consider monitoring executor._work_queue.qsize() in production
                
                # Use configured thread pool or default
                max_workers = self.config.get('execution', {}).get('feature_max_workers', None)
                if max_workers and not hasattr(self, '_feature_executor'):
                    import concurrent.futures
                    self._feature_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                    
                executor = getattr(self, '_feature_executor', None)
                features_df = await loop.run_in_executor(
                    executor, feature_agent.engineer_features, new_bar_df
                )
                if features_df is None or features_df.empty:
                    self.logger.warning("Failed to engineer features")
                    return
                    
                # Get the latest feature row
                latest_features = features_df.iloc[-1]
                
                # Generate action using model
                if live_model:
                    action = await self._generate_action_async(latest_features, live_model)
                    
                    # Trigger action generated hook
                    self._trigger_hook("action_generated", symbol, action, latest_features)
                    
                    # Execute complete trading pipeline if action is not HOLD
                    if action != TradingAction.HOLD:
                        await self._execute_trading_action(
                            symbol, action, latest_features, risk_agent
                        )
                    
                    # Update trading state with thread safety
                    async with self._state_lock:
                        self.trading_state[symbol]['last_action'] = action
                    
        except Exception as e:
            self.logger.error(f"Error processing new bar: {e}")
            
    async def _generate_action_async(self, features: pd.Series, model) -> TradingAction:
        """
        Generate trading action using the model (async to avoid blocking event loop).
        
        Args:
            features: Feature vector
            model: Trained model
            
        Returns:
            Action (TradingAction enum)
        """
        try:
            # Run model prediction in thread pool to avoid blocking event loop
            try:
                loop = asyncio.get_running_loop()  # Python 3.7+ preferred method
            except RuntimeError:
                loop = asyncio.new_event_loop()  # Fallback if no loop running
                asyncio.set_event_loop(loop)
            
            action = await loop.run_in_executor(
                None, self._predict_with_model, features, model
            )
            return TradingAction(action)
            
        except Exception as e:
            self.logger.error(f"Error generating action: {e}")
            return TradingAction.HOLD  # Default to hold
    
    def _predict_with_model(self, features: pd.Series, model) -> int:
        """Synchronous model prediction (runs in thread pool)."""
        # Convert features to numpy array
        feature_array = features.values.reshape(1, -1)
        
        # Get prediction from model
        action, _ = model.predict(feature_array, deterministic=True)
        
        return int(action[0]) if hasattr(action, '__iter__') else int(action)
    
    async def _execute_trading_action(
        self, 
        symbol: str, 
        action: TradingAction, 
        features: pd.Series, 
        risk_agent=None
    ) -> None:
        """
        Execute complete trading action pipeline.
        
        Args:
            symbol: Trading symbol
            action: Trading action to execute
            features: Current feature vector
            risk_agent: Risk management agent
        """
        try:
            # Convert action to string for risk callbacks
            action_str = "BUY" if action == TradingAction.BUY else "SELL"
            
            # Create trading event
            trading_event = {
                'symbol': symbol,
                'action': action_str,
                'shares': self._calculate_position_size(symbol, action, features),
                'timestamp': pd.Timestamp.now(),
                'features': features.to_dict()
            }
            
            # 1. Pre-trade risk check
            if risk_agent:
                from src.execution.core.risk_callbacks import pre_trade_check
                
                # Get current positions (placeholder - should come from position tracker)
                current_positions = await self._get_current_positions()
                
                is_allowed, reason = pre_trade_check(
                    trading_event, 
                    self.config.get('risk', {}), 
                    current_positions,
                    self.logger
                )
                
                if not is_allowed:
                    self.logger.warning(f"Trade blocked by risk check: {reason}")
                    self._trigger_hook("trade_blocked", symbol, action, reason)
                    return
            
            # 2. Create order
            order = await self._create_order(trading_event)
            
            # 2.5. Apply risk-based order throttling
            from src.execution.core.risk_callbacks import throttle_size
            market_conditions = await self._get_market_conditions(symbol)
            order = throttle_size(order, self.config.get('risk', {}), market_conditions, self.logger)
            
            # 2.6. Check minimum order size after throttling
            min_order_size = self.config.get('risk', {}).get('min_order_size', 1)
            if order['shares'] < min_order_size:
                self.logger.info(f"Order size {order['shares']} below minimum {min_order_size} - skipping")
                self._trigger_hook("order_too_small", symbol, order)
                return
            
            # 3. Send to order router (placeholder)
            order_id = await self._send_to_order_router(order)
            
            # 4. Update P&L tracker (placeholder)
            await self._update_pnl_tracker(symbol, order, order_id)
            
            # 5. Trigger trade executed hook
            self._trigger_hook("trade_executed", symbol, action, order, order_id)
            
        except Exception as e:
            self.logger.error(f"Error executing trading action: {e}")
            self._trigger_hook("trade_error", symbol, action, str(e))
    
    def _calculate_position_size(self, symbol: str, action: TradingAction, features: pd.Series) -> int:
        """Calculate position size for the trade."""
        # Placeholder implementation - should use proper position sizing logic
        base_size = self.config.get('position_sizing', {}).get('base_size', 100)
        
        # Could incorporate volatility, confidence, etc. from features
        return base_size
    
    async def _get_current_positions(self) -> Dict[str, Any]:
        """
        Get current portfolio positions.
        
        IMPORTANT: This is a placeholder implementation.
        
        PRODUCTION DEPLOYMENT OPTIONS:
        1. FAIL-CLOSED: Set config 'risk.fail_closed_on_missing_positions' = True
           - Will block all trades when position data unavailable
           - Safer for production but may miss opportunities
        
        2. FAIL-OPEN: Wire to real position tracker that provides:
           - Current position sizes per symbol  
           - Daily P&L
           - Total portfolio value
        
        Current behavior: Returns empty positions (risk checks less effective)
        """
        # Check if we should fail-closed when positions unavailable
        fail_closed = self.config.get('risk', {}).get('fail_closed_on_missing_positions', False)
        
        if fail_closed:
            # In fail-closed mode, raise exception to block trading
            raise RuntimeError(
                "Position data unavailable and fail_closed_on_missing_positions=True. "
                "Wire real position tracker or set fail_closed_on_missing_positions=False"
            )
        
        # TODO: Wire to real position tracker before live trading
        # For now, return empty dict (fail-open mode)
        self.logger.warning("Using empty positions - risk checks will be less effective")
        return {}
    
    async def _get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market conditions for throttling decisions.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market conditions dict with spread, volatility, volume data
        """
        # Placeholder - should integrate with market data provider
        # TODO: Wire to real market data for spread, volatility, volume
        return {
            'spread': 10,  # basis points - placeholder
            'volatility': 0.01,  # 1% - placeholder  
            'avg_volume': 1000000  # shares - placeholder
        }
    
    async def _create_order(self, trading_event: Dict[str, Any]) -> Dict[str, Any]:
        """Create order dictionary from trading event."""
        return {
            'symbol': trading_event['symbol'],
            'action': trading_event['action'],
            'shares': trading_event['shares'],
            'order_type': 'MARKET',  # Could be configurable
            'timestamp': trading_event['timestamp'],
            'source': 'execution_loop'
        }
    
    async def _send_to_order_router(self, order: Dict[str, Any]) -> str:
        """Send order to order router."""
        # Placeholder - should integrate with actual order router
        order_id = f"ORDER_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"Order sent to router: {order_id} - {order}")
        return order_id
    
    async def _update_pnl_tracker(self, symbol: str, order: Dict[str, Any], order_id: str) -> None:
        """
        Update P&L tracker with new order.
        
        IMPORTANT: This is a placeholder implementation.
        Before live trading, this MUST be connected to a real P&L tracker that:
        - Records all trades with timestamps
        - Calculates realized/unrealized P&L
        - Tracks daily/cumulative performance
        - Provides position-level P&L data
        
        TODO: Wire to real P&L tracker before live trading
        
        Args:
            symbol: Trading symbol
            order: Order details
            order_id: Unique order identifier
        """
        # TODO: Wire to real P&L tracker before live trading
        self.logger.info(f"P&L tracker updated for {symbol}: {order_id}")
        pass
        
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