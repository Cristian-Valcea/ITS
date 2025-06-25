# src/agents/risk_agent.py
import logging
from datetime import datetime, timedelta
from collections import deque

from .base_agent import BaseAgent

class RiskAgent(BaseAgent):
    """
    RiskAgent is responsible for:
    1. Monitoring live equity and portfolio state.
    2. Enforcing risk limits:
        - Maximum daily drawdown (e.g., 2% of start-of-day capital).
        - Turnover caps (e.g., 5x hourly or daily turnover).
    3. Signaling "halt" or "liquidate" if critical limits are breached.
    4. Providing checks for proposed trades against these limits.

    Note: This agent is primarily for LIVE TRADING. During backtesting/training,
    the IntradayTradingEnv handles its own risk checks (like max drawdown).
    However, some logic (like turnover calculation) might be shared or adapted.
    """
    def __init__(self, config: dict):
        """
        Initializes the RiskAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys from `risk_limits.yaml`:
                           'max_daily_drawdown_pct', 'max_hourly_turnover_ratio',
                           'max_daily_turnover_ratio', 'halt_on_breach'.
        """
        super().__init__(agent_name="RiskAgent", config=config)
        
        # Ensure defaults are robust if keys are missing, though config loading should handle this.
        self.max_daily_drawdown_pct = float(self.config.get('max_daily_drawdown_pct', 0.02))
        self.max_hourly_turnover_ratio = float(self.config.get('max_hourly_turnover_ratio', 5.0))
        self.max_daily_turnover_ratio = float(self.config.get('max_daily_turnover_ratio', 20.0))
        self.halt_on_breach = bool(self.config.get('halt_on_breach', True))
        self.liquidate_on_halt = bool(self.config.get('liquidate_on_halt', False))


        # State variables for live monitoring
        self.start_of_day_portfolio_value = None # Use portfolio value as the capital base for ratios
        self.current_portfolio_value = None    # Updated by Orchestrator/DataFeed
        
        self.daily_traded_value = 0.0    # Sum of absolute monetary value of all trades today
        self.hourly_traded_value = 0.0   # Sum for the current rolling hour

        self.last_event_timestamp = None # Tracks time for hourly calculations
        self.trades_this_hour = deque()  # Stores (timestamp, trade_value) for hourly turnover calculation

        self.logger.info("RiskAgent initialized with the following limits:")
        self.logger.info(f"  Max Daily Drawdown: {self.max_daily_drawdown_pct*100:.2f}%")
        self.logger.info(f"  Max Hourly Turnover Ratio: {self.max_hourly_turnover_ratio:.2f}x")
        self.logger.info(f"  Max Daily Turnover Ratio: {self.max_daily_turnover_ratio:.2f}x")
        self.logger.info(f"  Halt on Breach: {self.halt_on_breach}")
        self.logger.info(f"  Liquidate on Halt: {self.liquidate_on_halt}")

    def update_portfolio_value(self, portfolio_value: float, timestamp: datetime = None):
        """
        Called by the Orchestrator to update the RiskAgent's view of current portfolio value.
        
        Args:
            portfolio_value (float): Total value of portfolio (cash + positions).
            timestamp (datetime, optional): Current timestamp. Defaults to datetime.now().
        """
        self.current_portfolio_value = portfolio_value
        self.last_event_timestamp = timestamp or datetime.now()

        if self.start_of_day_portfolio_value is None: # First update of the day
            self.start_of_day_portfolio_value = self.current_portfolio_value
            self.logger.info(f"Start of day portfolio value set to: {self.start_of_day_portfolio_value:.2f}")
        
        # Prune hourly trades based on the new timestamp
        self._update_hourly_turnover(self.last_event_timestamp)


    def reset_daily_limits(self, current_portfolio_value: float = None, timestamp: datetime = None):
        """
        Resets daily counters and sets the capital baseline for the new day.
        Should be called by the Orchestrator at the start of each trading day.
        """
        # Use provided portfolio value or the last known one as the new day's starting point
        sod_portfolio_val = current_portfolio_value if current_portfolio_value is not None else self.current_portfolio_value
        if sod_portfolio_val is None:
            self.logger.warning("Cannot reset daily limits: current portfolio value unknown. RiskAgent may not function correctly.")
            return

        self.start_of_day_portfolio_value = sod_portfolio_val
        self.daily_traded_value = 0.0
        self.hourly_traded_value = 0.0 # Reset this as well, as it's relative to SOD capital
        self.trades_this_hour.clear()  # Clear the deque of hourly trades
        self.last_event_timestamp = timestamp or datetime.now()
        
        self.logger.info(f"Daily risk limits reset. Start of day portfolio value: {self.start_of_day_portfolio_value:.2f} at {self.last_event_timestamp}")


    def _update_hourly_turnover(self, current_time: datetime):
        """Helper to remove trades older than 1 hour from the deque and update sum."""
        if not self.trades_this_hour:
            return
        
        one_hour_ago = current_time - timedelta(hours=1)
        updated_hourly_traded_value = 0
        
        # Rebuild deque and sum only valid trades to avoid float precision issues with subtraction
        new_deque = deque()
        for ts, trade_val in self.trades_this_hour:
            if ts >= one_hour_ago:
                new_deque.append((ts, trade_val))
                updated_hourly_traded_value += trade_val
        
        if len(new_deque) < len(self.trades_this_hour):
            self.logger.debug(f"Pruned {len(self.trades_this_hour) - len(new_deque)} old trades from hourly window.")

        self.trades_this_hour = new_deque
        self.hourly_traded_value = updated_hourly_traded_value


    def record_trade(self, trade_value: float, timestamp: datetime = None):
        """
        Records a trade to update turnover calculations.
        trade_value is the absolute monetary value of the trade (e.g., num_shares * price).
        timestamp is when the trade occurred.
        """
        if self.start_of_day_portfolio_value is None:
            self.logger.error("Cannot record trade: start_of_day_portfolio_value is not set. Call reset_daily_limits() first.")
            return

        current_time = timestamp or datetime.now()
        self.last_event_timestamp = current_time

        abs_trade_value = abs(trade_value)
        if abs_trade_value <= 0:
            self.logger.warning(f"Attempted to record a trade with non-positive value: {trade_value}. Ignoring.")
            return
            
        self.daily_traded_value += abs_trade_value
        
        # Prune hourly trades before adding new one and calculating sum
        self._update_hourly_turnover(current_time) 
        self.trades_this_hour.append((current_time, abs_trade_value))
        self.hourly_traded_value += abs_trade_value # Add new trade to sum
        
        self.logger.debug(f"Trade recorded: Value={abs_trade_value:.2f} at {current_time}. "
                         f"Hourly Traded Value: {self.hourly_traded_value:.2f}, "
                         f"Daily Traded Value: {self.daily_traded_value:.2f}")

    def check_drawdown(self) -> tuple[bool, float]:
        """
        Checks if the maximum daily drawdown limit has been breached.

        Returns:
            tuple[bool, float]: (True if breached, current_drawdown_percentage)
        """
        if self.portfolio_value is None or self.start_of_day_capital is None or self.start_of_day_capital == 0:
            self.logger.warning("Cannot check drawdown: capital values not properly set.")
            return False, 0.0

        current_drawdown_pct = (self.start_of_day_capital - self.portfolio_value) / self.start_of_day_capital
        
        if current_drawdown_pct > self.max_daily_drawdown_pct:
            self.logger.warning(f"DRAWDOWN BREACHED: Current {current_drawdown_pct*100:.2f}% > Limit {self.max_daily_drawdown_pct*100:.2f}%")
            return True, current_drawdown_pct
        
        return False, current_drawdown_pct

    def check_turnover(self, proposed_trade_value: float = 0.0, current_time: datetime = None) -> tuple[bool, bool, float, float]:
        """
        Checks if hourly or daily turnover limits would be breached by a proposed trade.

        Args:
            proposed_trade_value (float): Absolute monetary value of the proposed trade.
            current_time (datetime): The current time, needed for hourly calculations.

        Returns:
            tuple[bool, bool, float, float]:
                - hourly_breached (bool): True if hourly limit is/would be breached.
                - daily_breached (bool): True if daily limit is/would be breached.
                - current_hourly_ratio (float): Current hourly turnover ratio (after proposed trade).
                - current_daily_ratio (float): Current daily turnover ratio (after proposed trade).
        """
        if self.start_of_day_capital is None or self.start_of_day_capital == 0:
            self.logger.warning("Cannot check turnover: start_of_day_capital not set or zero.")
            return False, False, 0.0, 0.0
        
        current_time = current_time or datetime.now() # Use current time if not provided
        self._update_hourly_turnover(current_time) # Ensure hourly window is current

        potential_hourly_value = self.hourly_traded_value + abs(proposed_trade_value)
        potential_daily_value = self.daily_traded_value + abs(proposed_trade_value)

        hourly_ratio = potential_hourly_value / self.start_of_day_capital
        daily_ratio = potential_daily_value / self.start_of_day_capital

        hourly_breached = hourly_ratio > self.max_hourly_turnover_ratio
        daily_breached = daily_ratio > self.max_daily_turnover_ratio

        if hourly_breached:
            self.logger.warning(f"HOURLY TURNOVER BREACH: Potential ratio {hourly_ratio:.2f}x > Limit {self.max_hourly_turnover_ratio:.2f}x")
        if daily_breached:
            self.logger.warning(f"DAILY TURNOVER BREACH: Potential ratio {daily_ratio:.2f}x > Limit {self.max_daily_turnover_ratio:.2f}x")
            
        return hourly_breached, daily_breached, hourly_ratio, daily_ratio

    def assess_trade_risk(self, proposed_trade_value: float, current_time: datetime = None) -> tuple[bool, str]:
        """
        Assesses if a proposed trade is permissible based on current risk status.
        This combines drawdown and turnover checks.

        Args:
            proposed_trade_value (float): Absolute monetary value of the proposed trade.
            current_time (datetime): Current time for turnover calculations.

        Returns:
            tuple[bool, str]: (is_safe_to_trade, reason_if_not_safe)
                              is_safe_to_trade is True if trade is allowed.
        """
        if self.portfolio_value is None or self.start_of_day_capital is None:
            return False, "Capital information not available for risk assessment."

        # 1. Check Drawdown (based on current portfolio value, not affected by proposed trade directly here)
        drawdown_breached, dd_pct = self.check_drawdown()
        if drawdown_breached:
            reason = f"Drawdown limit breached ({dd_pct*100:.2f}% vs {self.max_daily_drawdown_pct*100:.2f}%)."
            if self.halt_on_breach:
                self.logger.critical(f"HALT SIGNAL: {reason}")
                return False, f"HALT: {reason}"
            else:
                self.logger.warning(f"Risk Warning: {reason} (Halt is False)")
                # Allow trade if halt_on_breach is false, but this is risky.
                # Usually, drawdown breach means stop. For this logic, let's assume halt_on_breach applies.

        # 2. Check Turnover (including the proposed trade)
        hourly_turnover_breached, daily_turnover_breached, hr, dr = self.check_turnover(proposed_trade_value, current_time)
        
        if hourly_turnover_breached:
            reason = f"Hourly turnover limit would be breached ({hr:.2f}x vs {self.max_hourly_turnover_ratio:.2f}x)."
            if self.halt_on_breach:
                self.logger.critical(f"HALT SIGNAL: {reason}")
                return False, f"HALT: {reason}"
            else: # Log warning but allow trade if halt_on_breach is false
                 self.logger.warning(f"Risk Warning: {reason} (Halt is False)")


        if daily_turnover_breached:
            reason = f"Daily turnover limit would be breached ({dr:.2f}x vs {self.max_daily_turnover_ratio:.2f}x)."
            if self.halt_on_breach:
                self.logger.critical(f"HALT SIGNAL: {reason}")
                return False, f"HALT: {reason}"
            else:
                 self.logger.warning(f"Risk Warning: {reason} (Halt is False)")
        
        # If any "halt on breach" condition was met and returned False, we don't reach here.
        # If halt_on_breach is False for all, or no limits breached:
        if drawdown_breached or hourly_turnover_breached or daily_turnover_breached:
            if not self.halt_on_breach: # If limits were breached but halt is false
                return True, "Proceeding with trade despite limit breach warning (halt_on_breach=False)."
            # This case should ideally not be reached if halt_on_breach=True and a breach occurred.
            # The first breach with halt_on_breach=True should have returned (False, "HALT: ...").
            # This logic implies that if halt_on_breach is true, any breach is a no-go.
            # If we are here, it means either no breach, or breach occurred but halt_on_breach is false.
            # This needs careful review of desired behavior. Let's refine:
            # If any limit is breached AND halt_on_breach is true for THAT type of limit (not global), then block.
            # For simplicity, current `self.halt_on_breach` is global.

        # Refined logic:
        # If drawdown_breached is True and self.halt_on_breach is True, it should have already returned.
        # If we are here, either no limits are breached, or some are breached but self.halt_on_breach is False.
        
        # Let's assume if any limit is breached, and self.halt_on_breach is True, the trade is blocked.
        # The individual checks above already log warnings.
        final_decision_block = (drawdown_breached or hourly_turnover_breached or daily_turnover_breached) and self.halt_on_breach
        
        if final_decision_block:
            # This should ideally be caught by the first check that breaches and halts.
            # Adding a catch-all.
            return False, "HALT: A risk limit was breached and halt_on_breach is True."

        return True, "Trade is permissible within risk limits."

    def run(self, current_capital: float, portfolio_value: float, proposed_trade_value: float = 0.0, current_time: datetime = None):
        """
        Primary method for RiskAgent in a live context: update state and check overall risk.
        This is more of a status check than an action. `assess_trade_risk` is for pre-trade checks.

        Args:
            current_capital (float): Current cash.
            portfolio_value (float): Current total portfolio value.
            proposed_trade_value (float): Optional, if checking a hypothetical next trade.
            current_time (datetime): Current timestamp.

        Returns:
            dict: A status dictionary with current risk metrics and breach status.
        """
        self.update_capital_and_portfolio(current_capital, portfolio_value)
        current_time = current_time or datetime.now()

        dd_breached, dd_pct = self.check_drawdown()
        ht_breached, dt_breached, hr, dr = self.check_turnover(proposed_trade_value, current_time)

        overall_breach = (dd_breached or ht_breached or dt_breached)
        halt_signal = overall_breach and self.halt_on_breach

        status = {
            "timestamp": current_time.isoformat(),
            "portfolio_value": self.portfolio_value,
            "start_of_day_capital": self.start_of_day_capital,
            "drawdown_percentage": dd_pct,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "drawdown_breached": dd_breached,
            "hourly_turnover_ratio": hr,
            "max_hourly_turnover_ratio": self.max_hourly_turnover_ratio,
            "hourly_turnover_breached": ht_breached,
            "daily_turnover_ratio": dr,
            "max_daily_turnover_ratio": self.max_daily_turnover_ratio,
            "daily_turnover_breached": dt_breached,
            "overall_limit_breached": overall_breach,
            "halt_signal_active": halt_signal,
            "halt_on_breach_config": self.halt_on_breach
        }
        
        if halt_signal:
            self.logger.critical(f"RISK HALT SIGNAL ACTIVE. Status: {status}")
        elif overall_breach:
            self.logger.warning(f"Risk limit breached (halt_on_breach=False). Status: {status}")
            
        return status


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- RiskAgent Configuration ---
    risk_config = {
        'max_daily_drawdown_pct': 0.02,  # 2%
        'max_hourly_turnover_ratio': 1.0, # 1x capital per hour
        'max_daily_turnover_ratio': 3.0,  # 3x capital per day
        'halt_on_breach': True
    }
    risk_agent = RiskAgent(config=risk_config)

    # --- Simulate a Trading Day ---
    sim_current_time = datetime(2023, 1, 1, 9, 30, 0)
    initial_cap = 100000.0
    
    # Start of day
    risk_agent.reset_daily_limits(start_of_day_capital=initial_cap)
    risk_agent.update_capital_and_portfolio(current_capital=initial_cap, portfolio_value=initial_cap)
    print(f"\n--- Simulating Trading Day: {sim_current_time.date()} ---")
    print(f"Initial Capital: {initial_cap:.2f}")

    # Simulate some trades
    # Trade 1: Buy 10k worth of stock
    trade_1_value = 10000.0
    sim_current_time += timedelta(minutes=5)
    safe, reason = risk_agent.assess_trade_risk(trade_1_value, sim_current_time)
    print(f"Trade 1 (Value: {trade_1_value:.2f}) Assessment: Safe={safe}, Reason='{reason}'")
    if safe:
        risk_agent.record_trade(trade_1_value, sim_current_time)
        # Assume capital doesn't change much for this example, only portfolio value might
        risk_agent.update_capital_and_portfolio(current_capital=initial_cap - trade_1_value, portfolio_value=initial_cap)


    # Trade 2: Sell 50k worth of stock (potential turnover breach)
    trade_2_value = 50000.0
    sim_current_time += timedelta(minutes=10)
    safe, reason = risk_agent.assess_trade_risk(trade_2_value, sim_current_time)
    print(f"Trade 2 (Value: {trade_2_value:.2f}) Assessment: Safe={safe}, Reason='{reason}'")
    if safe:
        risk_agent.record_trade(trade_2_value, sim_current_time)
        risk_agent.update_capital_and_portfolio(current_capital=initial_cap - trade_1_value + trade_2_value, portfolio_value=initial_cap) # Simplified capital update

    # Trade 3: Buy another 50k (should definitely breach hourly turnover if previous was allowed)
    trade_3_value = 50000.0
    sim_current_time += timedelta(minutes=10)
    safe, reason = risk_agent.assess_trade_risk(trade_3_value, sim_current_time)
    print(f"Trade 3 (Value: {trade_3_value:.2f}) Assessment: Safe={safe}, Reason='{reason}'")
    if safe: # This should be False if Trade 2 happened and config['halt_on_breach'] is True
        risk_agent.record_trade(trade_3_value, sim_current_time)
    
    # Simulate portfolio value drop (drawdown)
    sim_current_time += timedelta(minutes=30) # 10:25 AM
    new_portfolio_value = initial_cap * 0.97 # 3% drawdown
    risk_agent.update_capital_and_portfolio(current_capital=initial_cap, portfolio_value=new_portfolio_value) # Assuming cash is same, positions lost value
    print(f"Portfolio value dropped to {new_portfolio_value:.2f} at {sim_current_time.time()}")

    # Check risk status
    status_check = risk_agent.run(
        current_capital=risk_agent.current_capital, # Use agent's current view
        portfolio_value=risk_agent.portfolio_value,
        current_time=sim_current_time
    )
    print(f"Risk Status at {sim_current_time.time()}:")
    for k, v in status_check.items(): print(f"  {k}: {v}")


    # Try another trade after drawdown breach
    trade_4_value = 5000.0
    sim_current_time += timedelta(minutes=5) # 10:30 AM
    safe, reason = risk_agent.assess_trade_risk(trade_4_value, sim_current_time)
    print(f"Trade 4 (Value: {trade_4_value:.2f}) Post-Drawdown Assessment: Safe={safe}, Reason='{reason}'")
    # This should be False if halt_on_breach is True due to drawdown.

    # Simulate time passing to next hour to check hourly turnover reset
    sim_current_time = datetime(2023, 1, 1, 10, 55, 0) # Near end of first hour of trades
    risk_agent._update_hourly_turnover(sim_current_time) # Manually trigger for visibility
    print(f"Hourly traded value before hour reset ({sim_current_time.time()}): {risk_agent.hourly_traded_value:.2f}")
    
    sim_current_time = datetime(2023, 1, 1, 11, 5, 0) # Into the next hour
    risk_agent._update_hourly_turnover(sim_current_time) # Should prune trades from 09:30-10:05 window
    print(f"Hourly traded value after hour reset ({sim_current_time.time()}): {risk_agent.hourly_traded_value:.2f} (should be lower or zero)")

    trade_5_value = 20000.0
    safe, reason = risk_agent.assess_trade_risk(trade_5_value, sim_current_time)
    print(f"Trade 5 (Value: {trade_5_value:.2f}) In Next Hour Assessment: Safe={safe}, Reason='{reason}'")
    # This might be allowed if hourly turnover reset, but daily drawdown might still be an issue.

    print("\nRiskAgent example simulation complete.")
