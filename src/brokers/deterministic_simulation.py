#!/usr/bin/env python3
"""
🎯 Deterministic Simulation Mode for IBKR Orders
Replaces hash-based random simulation with predictable test vectors
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SimulationScenario:
    """Deterministic simulation scenario"""
    name: str
    symbol: str
    action: str
    quantity: int
    order_type: str
    status_sequence: List[str]  # Sequence of statuses to simulate
    fill_price: float
    timing_delays: List[float]  # Delay before each status (seconds)
    final_filled_qty: float

class DeterministicSimulator:
    """
    🎯 Deterministic order simulation with predictable outcomes
    
    FIXES ISSUE: Non-deterministic hash() based simulation
    PROVIDES: Reproducible test scenarios for CI/CD
    """
    
    # Predefined test scenarios for reproducible testing
    DEFAULT_SCENARIOS = {
        'msft_quick_fill': SimulationScenario(
            name='MSFT Quick Fill',
            symbol='MSFT',
            action='BUY',
            quantity=1,
            order_type='MKT',
            status_sequence=['PendingSubmit', 'PreSubmitted', 'Filled'],
            fill_price=425.50,
            timing_delays=[0.1, 0.5, 1.0],  # Status changes at these intervals
            final_filled_qty=1
        ),
        
        'nvda_partial_fill': SimulationScenario(
            name='NVDA Partial Fill',
            symbol='NVDA',
            action='BUY', 
            quantity=10,
            order_type='MKT',
            status_sequence=['PendingSubmit', 'PreSubmitted', 'Submitted', 'Filled'],
            fill_price=875.25,
            timing_delays=[0.1, 0.3, 1.0, 2.0],
            final_filled_qty=7  # Partial fill
        ),
        
        'msft_cancelled': SimulationScenario(
            name='MSFT Cancelled Order',
            symbol='MSFT',
            action='SELL',
            quantity=5,
            order_type='LMT',
            status_sequence=['PendingSubmit', 'PreSubmitted', 'Submitted', 'PendingCancel', 'Cancelled'],
            fill_price=0.0,  # No fill
            timing_delays=[0.1, 0.5, 1.0, 0.3, 0.5],
            final_filled_qty=0
        ),
        
        'nvda_rejected': SimulationScenario(
            name='NVDA Rejected Order',
            symbol='NVDA',
            action='BUY',
            quantity=100,
            order_type='MKT',
            status_sequence=['PendingSubmit', 'Cancelled'],  # Immediate rejection
            fill_price=0.0,
            timing_delays=[0.1, 0.2],
            final_filled_qty=0
        ),
        
        'live_trading_sequence': SimulationScenario(
            name='Live Trading Sequence',
            symbol='MSFT',
            action='BUY',
            quantity=3,
            order_type='MKT',
            status_sequence=['PendingSubmit', 'PreSubmitted', 'Submitted', 'Filled'],
            fill_price=428.75,
            timing_delays=[0.1, 0.8, 2.0, 1.5],  # Realistic timing
            final_filled_qty=3
        )
    }
    
    def __init__(self, scenarios: Optional[Dict[str, SimulationScenario]] = None):
        """
        Initialize deterministic simulator
        
        Args:
            scenarios: Custom scenarios, defaults to built-in scenarios
        """
        self.scenarios = scenarios or self.DEFAULT_SCENARIOS
        self.order_counter = 1000  # Start order IDs at 1000
        self.simulation_log = []
        
    def select_scenario(self, symbol: str, action: str, quantity: int, order_type: str = 'MKT') -> SimulationScenario:
        """
        🎯 Select deterministic scenario based on order parameters
        
        REPLACES: hash(symbol) % 100 random behavior
        PROVIDES: Predictable scenario selection
        """
        
        # Deterministic scenario selection logic
        scenario_key = f"{symbol.lower()}_{action.lower()}_{order_type.lower()}"
        
        # Map to predefined scenarios
        scenario_mapping = {
            'msft_buy_mkt': 'msft_quick_fill',
            'msft_sell_mkt': 'msft_quick_fill', 
            'msft_sell_lmt': 'msft_cancelled',
            'nvda_buy_mkt': 'nvda_partial_fill',
            'nvda_sell_mkt': 'nvda_partial_fill',
        }
        
        # Special cases based on quantity
        if quantity >= 100:
            scenario_name = 'nvda_rejected'  # Large orders get rejected
        elif quantity >= 10:
            scenario_name = 'nvda_partial_fill'  # Medium orders get partial fills
        else:
            scenario_name = scenario_mapping.get(scenario_key, 'live_trading_sequence')
        
        scenario = self.scenarios.get(scenario_name, self.scenarios['msft_quick_fill'])
        
        # Customize scenario for this specific order
        customized = SimulationScenario(
            name=f"{scenario.name} ({symbol} {action} {quantity})",
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            status_sequence=scenario.status_sequence.copy(),
            fill_price=scenario.fill_price,
            timing_delays=scenario.timing_delays.copy(),
            final_filled_qty=min(scenario.final_filled_qty, quantity)  # Can't fill more than ordered
        )
        
        logger.info(f"🎯 Selected scenario: {customized.name}")
        return customized
    
    def simulate_order(self, symbol: str, action: str, quantity: int, order_type: str = 'MKT') -> Dict:
        """
        🎯 Execute deterministic order simulation
        
        Returns order result with predictable outcome
        """
        
        scenario = self.select_scenario(symbol, action, quantity, order_type)
        order_id = self.order_counter
        self.order_counter += 1
        
        start_time = time.time()
        
        # Log simulation start
        sim_log = {
            'order_id': order_id,
            'scenario': scenario.name,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_type': order_type,
            'start_time': datetime.now().isoformat(),
            'status_history': []
        }
        
        logger.info(f"🎭 DETERMINISTIC SIMULATION: {scenario.name}")
        print(f"🎭 Simulating: {scenario.name}")
        
        # Simulate status sequence with timing
        current_filled = 0
        current_price = scenario.fill_price
        
        for i, (status, delay) in enumerate(zip(scenario.status_sequence, scenario.timing_delays)):
            # Wait for realistic timing
            if delay > 0:
                time.sleep(delay)
            
            # Calculate progressive fills
            if status == 'Filled':
                current_filled = scenario.final_filled_qty
            elif status in ['Submitted', 'PreSubmitted'] and scenario.final_filled_qty > 0:
                # Partial fills during active status
                current_filled = min(scenario.final_filled_qty * 0.3, quantity)
            
            elapsed = time.time() - start_time
            
            # Log status change
            status_event = {
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'elapsed_seconds': elapsed,
                'filled_qty': current_filled,
                'avg_fill_price': current_price if current_filled > 0 else 0.0
            }
            
            sim_log['status_history'].append(status_event)
            
            print(f"   [{elapsed:5.1f}s] {status} (filled: {current_filled})")
            logger.info(f"Simulation order {order_id}: {status} @ {elapsed:.1f}s")
        
        # Final result
        final_status = scenario.status_sequence[-1]
        total_time = time.time() - start_time
        
        result = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'action': action,
            'order_type': order_type,
            'status': final_status,
            'fill_price': current_price if current_filled > 0 else 0.0,
            'filled_quantity': current_filled,
            'timestamp': datetime.now().isoformat(),
            'mode': 'deterministic_simulation',
            'scenario_name': scenario.name,
            'simulation_time': total_time,
            'is_filled': final_status == 'Filled',
            'is_cancelled': final_status in ['Cancelled', 'ApiCancelled']
        }
        
        sim_log['final_result'] = result
        self.simulation_log.append(sim_log)
        
        print(f"✅ Simulation complete: {final_status}")
        logger.info(f"Simulation order {order_id} complete: {final_status}")
        
        return result
    
    def get_simulation_report(self) -> str:
        """Generate simulation report for testing/debugging"""
        
        report = "🎯 DETERMINISTIC SIMULATION REPORT\n"
        report += "=" * 50 + "\n"
        report += f"Total simulations: {len(self.simulation_log)}\n\n"
        
        for sim in self.simulation_log:
            report += f"Order {sim['order_id']}: {sim['scenario']}\n"
            report += f"  Symbol: {sim['symbol']} {sim['action']} {sim['quantity']}\n"
            report += f"  Status sequence: {len(sim['status_history'])} events\n"
            report += f"  Final: {sim['final_result']['status']}\n"
            if sim['final_result']['is_filled']:
                report += f"  Fill: {sim['final_result']['filled_quantity']} @ ${sim['final_result']['fill_price']}\n"
            report += "\n"
        
        return report
    
    def save_simulation_vectors(self, filename: str):
        """Save simulation scenarios as test vectors for CI/CD"""
        
        vectors = {
            'scenarios': {name: {
                'name': scenario.name,
                'symbol': scenario.symbol,
                'action': scenario.action,
                'quantity': scenario.quantity,
                'order_type': scenario.order_type,
                'status_sequence': scenario.status_sequence,
                'fill_price': scenario.fill_price,
                'timing_delays': scenario.timing_delays,
                'final_filled_qty': scenario.final_filled_qty
            } for name, scenario in self.scenarios.items()},
            'simulation_log': self.simulation_log
        }
        
        with open(filename, 'w') as f:
            json.dump(vectors, f, indent=2)
        
        logger.info(f"📁 Simulation vectors saved to {filename}")

def create_test_simulator() -> DeterministicSimulator:
    """Create simulator with test scenarios for unit testing"""
    
    # Additional test scenarios for comprehensive testing
    test_scenarios = DeterministicSimulator.DEFAULT_SCENARIOS.copy()
    
    # Add edge cases for testing
    test_scenarios['edge_case_zero_fill'] = SimulationScenario(
        name='Edge Case: Zero Fill',
        symbol='TEST',
        action='BUY',
        quantity=1,
        order_type='MKT',
        status_sequence=['PendingSubmit', 'Cancelled'],
        fill_price=0.0,
        timing_delays=[0.1, 0.1],
        final_filled_qty=0
    )
    
    return DeterministicSimulator(test_scenarios)