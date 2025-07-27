# debug_volatility_penalty.py
"""
Debug script to identify why VolatilityPenaltyCalculator always returns 0.0.

This script will trace the entire data flow from observation to penalty calculation
to identify where the drawdown_velocity is getting lost.
"""

import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_risk_advisor_flow():
    """Debug the complete risk advisor flow to find where drawdown_velocity gets lost."""
    
    print("🔍 DEBUGGING VOLATILITY PENALTY CALCULATOR")
    print("=" * 80)
    
    try:
        # Step 1: Load risk configuration
        print("📋 Step 1: Loading risk configuration...")
        risk_config_path = Path("config/risk_limits.yaml")
        
        if not risk_config_path.exists():
            print(f"❌ Risk config not found: {risk_config_path}")
            return False
        
        with open(risk_config_path, 'r') as f:
            risk_config = yaml.safe_load(f)
        
        print(f"✅ Risk config loaded from: {risk_config_path}")
        
        # Check drawdown_velocity configuration
        sensor_config = risk_config.get('sensor_config', {})
        drawdown_vel_config = sensor_config.get('drawdown_velocity', {})
        
        print(f"🔧 Drawdown velocity config: {drawdown_vel_config}")
        
        # Step 2: Create ProductionRiskAdvisor
        print("\n📋 Step 2: Creating ProductionRiskAdvisor...")
        
        from src.training.interfaces.risk_advisor import ProductionRiskAdvisor
        
        risk_advisor = ProductionRiskAdvisor(risk_config_path, "debug_advisor")
        print("✅ ProductionRiskAdvisor created successfully")
        
        # Step 3: Create realistic observation data
        print("\n📋 Step 3: Creating realistic observation data...")
        
        # Simulate portfolio values with drawdown
        portfolio_values = [100000.0]  # Start with $100k
        timestamps = [pd.Timestamp.now()]
        
        # Simulate a drawdown scenario
        for i in range(20):
            # Simulate declining portfolio value (drawdown)
            decline_factor = 0.98 if i < 10 else 0.995  # Faster decline initially
            new_value = portfolio_values[-1] * decline_factor
            portfolio_values.append(new_value)
            timestamps.append(timestamps[-1] + pd.Timedelta(minutes=1))
        
        print(f"📊 Portfolio values: {portfolio_values[:5]}...{portfolio_values[-5:]}")
        print(f"📊 Total drawdown: {(portfolio_values[0] - portfolio_values[-1]) / portfolio_values[0] * 100:.2f}%")
        
        # Create observation dictionary
        obs_dict = {
            'portfolio_value': portfolio_values[-1],
            'portfolio_values': portfolio_values,
            'timestamps': timestamps,
            'current_position': 1,  # Long position
            'current_price': 150.0,
            'daily_pnl': portfolio_values[-1] - portfolio_values[0],
            'current_capital': portfolio_values[-1] * 0.5,  # Assume 50% cash
            'position_quantity': 100.0,
            'entry_price': 145.0,
            'transaction_cost_pct': 0.001,
            'hourly_turnover_cap': 5.0,
            'current_hourly_traded_value': 1000.0,
            'max_daily_drawdown_pct': 0.02
        }
        
        print(f"✅ Observation created with {len(portfolio_values)} portfolio values")
        
        # Step 4: Test DrawdownVelocityCalculator directly
        print("\n📋 Step 4: Testing DrawdownVelocityCalculator directly...")
        
        from src.risk.calculators.drawdown_velocity_calculator import DrawdownVelocityCalculator
        
        # Create calculator with debug config
        calc_config = drawdown_vel_config.get('config', {
            'velocity_window': 10,
            'min_periods': 5
        })
        
        drawdown_calc = DrawdownVelocityCalculator(calc_config)
        print(f"✅ DrawdownVelocityCalculator created with config: {calc_config}")
        
        # Test calculator directly
        calc_data = {
            'portfolio_values': np.array(portfolio_values),
            'timestamps': np.array([ts.timestamp() for ts in timestamps])
        }
        
        calc_result = drawdown_calc.calculate(calc_data)
        print(f"🔍 Direct calculator result: {calc_result.values}")
        
        if calc_result.values.get('velocity', 0.0) == 0.0:
            print("❌ PROBLEM: DrawdownVelocityCalculator returns 0.0 velocity")
            print(f"   Metadata: {calc_result.metadata}")
            
            # Debug the calculation step by step
            print("\n🔍 DEBUGGING CALCULATOR INTERNALS:")
            portfolio_array = np.array(portfolio_values)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (running_max - portfolio_array) / running_max
            
            print(f"   Portfolio values: {portfolio_array[:5]}...{portfolio_array[-5:]}")
            print(f"   Running max: {running_max[:5]}...{running_max[-5:]}")
            print(f"   Drawdown: {drawdown[:5]}...{drawdown[-5:]}")
            print(f"   Max drawdown: {np.max(drawdown):.4f}")
            
            if len(drawdown) >= calc_config['velocity_window']:
                recent_drawdown = drawdown[-calc_config['velocity_window']:]
                drawdown_changes = np.diff(recent_drawdown)
                print(f"   Recent drawdown: {recent_drawdown}")
                print(f"   Drawdown changes: {drawdown_changes}")
        else:
            print(f"✅ DrawdownVelocityCalculator working: velocity = {calc_result.values.get('velocity', 0.0):.6f}")
        
        # Step 5: Test RiskAdvisor evaluation
        print("\n📋 Step 5: Testing RiskAdvisor evaluation...")
        
        risk_metrics = risk_advisor.evaluate(obs_dict)
        print(f"🔍 Risk advisor result: {risk_metrics}")
        
        drawdown_vel = risk_metrics.get('drawdown_velocity', 0.0)
        drawdown_vel_alt = risk_metrics.get('drawdown_vel', 0.0)  # Check alternative key
        
        print(f"🔍 drawdown_velocity: {drawdown_vel}")
        print(f"🔍 drawdown_vel: {drawdown_vel_alt}")
        
        if drawdown_vel == 0.0 and drawdown_vel_alt == 0.0:
            print("❌ PROBLEM: RiskAdvisor returns 0.0 for drawdown velocity")
            
            # Debug the risk agent internals
            print("\n🔍 DEBUGGING RISK AGENT INTERNALS:")
            
            # Check if the risk agent has the calculator
            risk_agent = risk_advisor._risk_agent
            print(f"   Risk agent calculators: {[calc.__class__.__name__ for calc in risk_agent.calculators]}")
            
            # Check if DrawdownVelocityCalculator is present
            drawdown_calcs = [calc for calc in risk_agent.calculators 
                            if 'DrawdownVelocity' in calc.__class__.__name__]
            
            if not drawdown_calcs:
                print("❌ CRITICAL: DrawdownVelocityCalculator not found in risk agent!")
            else:
                print(f"✅ Found DrawdownVelocityCalculator: {drawdown_calcs[0]}")
        else:
            print(f"✅ RiskAdvisor working: drawdown_velocity = {max(drawdown_vel, drawdown_vel_alt):.6f}")
        
        # Step 6: Test penalty calculation
        print("\n📋 Step 6: Testing penalty calculation...")
        
        penalty_weight = 0.1
        penalty = penalty_weight * risk_metrics.get('drawdown_velocity', 0.0)
        
        print(f"🔍 Penalty calculation: {penalty_weight} * {risk_metrics.get('drawdown_velocity', 0.0)} = {penalty}")
        
        if penalty == 0.0:
            print("❌ PROBLEM: Final penalty is 0.0")
            
            # Try alternative keys
            alt_keys = ['drawdown_velocity', 'velocity', 'drawdown_vel']
            for key in alt_keys:
                value = risk_metrics.get(key, 0.0)
                if value != 0.0:
                    print(f"✅ Found non-zero value for key '{key}': {value}")
                    penalty_alt = penalty_weight * value
                    print(f"   Alternative penalty: {penalty_weight} * {value} = {penalty_alt}")
        else:
            print(f"✅ Penalty calculation working: {penalty}")
        
        # Step 7: Summary and recommendations
        print("\n📋 Step 7: Summary and Recommendations")
        print("=" * 80)
        
        if penalty > 0.0 or any(risk_metrics.get(key, 0.0) > 0.0 for key in ['drawdown_velocity', 'drawdown_vel']):
            print("✅ SYSTEM WORKING: Drawdown velocity penalty is being calculated")
            print(f"   Final penalty: {penalty}")
        else:
            print("❌ SYSTEM BROKEN: Drawdown velocity penalty is always 0.0")
            print("\n🔧 RECOMMENDED FIXES:")
            print("1. Check if DrawdownVelocityCalculator is enabled in risk_limits.yaml")
            print("2. Verify portfolio_values data is being passed correctly")
            print("3. Check if timestamps are in correct format")
            print("4. Verify velocity_window and min_periods configuration")
            print("5. Check for data type mismatches (numpy arrays vs lists)")
        
        return penalty > 0.0
        
    except Exception as e:
        print(f"❌ DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_debug_callback():
    """Create a debug version of the risk penalty callback with extensive logging."""
    
    print("\n🔧 CREATING DEBUG CALLBACK")
    print("=" * 40)
    
    debug_callback_code = '''
# Debug version of RiskPenaltyCallback with extensive logging

class DebugRiskPenaltyCallback(BaseCallback):
    """Debug version of RiskPenaltyCallback with extensive logging."""
    
    def __init__(self, risk_advisor, penalty_weight=0.1, verbose=1):
        super().__init__(verbose)
        self.risk_advisor = risk_advisor
        self.penalty_weight = penalty_weight
        self.step_count = 0
        
        # Enhanced logging
        self.logger = logging.getLogger("DebugRiskPenaltyCallback")
        self.logger.setLevel(logging.DEBUG)
        
        print(f"🔧 DebugRiskPenaltyCallback initialized:")
        print(f"   Risk advisor: {risk_advisor}")
        print(f"   Penalty weight: {penalty_weight}")
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        try:
            # Get current observation with extensive logging
            obs = self._get_current_observation()
            
            self.logger.debug(f"Step {self.step_count}: Got observation keys: {list(obs.keys()) if obs else 'None'}")
            
            if obs is None:
                self.logger.warning(f"Step {self.step_count}: No observation available")
                return True
            
            # Log key observation values
            portfolio_value = obs.get('portfolio_value', 'missing')
            current_position = obs.get('current_position', 'missing')
            daily_pnl = obs.get('daily_pnl', 'missing')
            
            self.logger.debug(f"Step {self.step_count}: portfolio_value={portfolio_value}, position={current_position}, pnl={daily_pnl}")
            
            # Evaluate risk with extensive logging
            self.logger.debug(f"Step {self.step_count}: Calling risk_advisor.evaluate()...")
            risk_result = self.risk_advisor.evaluate(obs)
            
            self.logger.debug(f"Step {self.step_count}: Risk evaluation result: {risk_result}")
            
            # Check all possible keys for drawdown velocity
            drawdown_keys = ['drawdown_vel', 'drawdown_velocity', 'velocity']
            drawdown_values = {key: risk_result.get(key, 0.0) for key in drawdown_keys}
            
            self.logger.debug(f"Step {self.step_count}: Drawdown values: {drawdown_values}")
            
            # Calculate penalty
            drawdown_vel = risk_result.get('drawdown_vel', 0.0)
            penalty = self.penalty_weight * drawdown_vel
            
            self.logger.info(f"Step {self.step_count}: PENALTY CALCULATION:")
            self.logger.info(f"   drawdown_vel: {drawdown_vel}")
            self.logger.info(f"   penalty_weight: {self.penalty_weight}")
            self.logger.info(f"   penalty: {penalty}")
            
            if penalty == 0.0:
                self.logger.warning(f"Step {self.step_count}: ⚠️  PENALTY IS ZERO!")
                
                # Try to find non-zero values
                non_zero_values = {k: v for k, v in risk_result.items() if v != 0.0}
                if non_zero_values:
                    self.logger.info(f"   Non-zero risk values: {non_zero_values}")
                else:
                    self.logger.warning(f"   ALL RISK VALUES ARE ZERO: {risk_result}")
            else:
                self.logger.info(f"Step {self.step_count}: ✅ PENALTY CALCULATED: {penalty}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Step {self.step_count}: Error in debug callback: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return True
    
    def _get_current_observation(self):
        """Get current observation from environment."""
        try:
            # Try to get observation from training environment
            if hasattr(self, 'training_env') and self.training_env is not None:
                if hasattr(self.training_env, 'get_attr'):
                    # VecEnv case
                    obs_list = self.training_env.get_attr('_get_info')
                    if obs_list and len(obs_list) > 0:
                        return obs_list[0]
                elif hasattr(self.training_env, '_get_info'):
                    # Direct env case
                    return self.training_env._get_info()
            
            # Try to get from model
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'env'):
                    env = self.model.env
                    if hasattr(env, 'get_attr'):
                        obs_list = env.get_attr('_get_info')
                        if obs_list and len(obs_list) > 0:
                            return obs_list[0]
                    elif hasattr(env, '_get_info'):
                        return env._get_info()
            
            self.logger.warning("Could not get observation from environment")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting observation: {e}")
            return None
'''
    
    print("✅ Debug callback code generated")
    print("   This callback will provide extensive logging of the penalty calculation process")
    
    return debug_callback_code


if __name__ == "__main__":
    print("🚀 STARTING VOLATILITY PENALTY DEBUG")
    print("=" * 80)
    
    # Run the debug flow
    success = debug_risk_advisor_flow()
    
    if success:
        print("\n✅ DEBUG COMPLETED SUCCESSFULLY")
        print("The volatility penalty system is working correctly")
    else:
        print("\n❌ DEBUG FOUND ISSUES")
        print("The volatility penalty system needs fixes")
        
        # Generate debug callback
        debug_code = create_debug_callback()
        
        print("\n🔧 NEXT STEPS:")
        print("1. Run this debug script to identify the exact issue")
        print("2. Use the generated debug callback in your training")
        print("3. Check the detailed logs to see where drawdown_velocity gets lost")
        print("4. Fix the identified issue in the data flow")
    
    print("\n📋 SUMMARY:")
    print("This debug script traces the complete flow:")
    print("  Observation → RiskAdvisor → DrawdownVelocityCalculator → Penalty")
    print("Run it to identify exactly where the drawdown_velocity becomes 0.0")