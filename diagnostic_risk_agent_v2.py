#!/usr/bin/env python3
"""
Diagnostic check to see what calculators are loaded in RiskAgentV2.
"""

import inspect
import sys
import pathlib
import pprint

# Add src to path for imports
sys.path.insert(0, "src")

def main():
    print("🔍 RISK AGENT V2 CALCULATOR DIAGNOSTIC")
    print("=" * 60)
    
    try:
        # Import after path adjustment
        from risk.risk_agent_v2 import create_risk_agent_v2
        
        # Create default RiskAgentV2 instance
        print("📋 Creating RiskAgentV2 with default configuration...")
        
        # Minimal config to see what calculators are loaded
        minimal_config = {
            'calculators': {
                'drawdown': {'enabled': True, 'config': {}},
                'turnover': {'enabled': True, 'config': {}},
                'ulcer_index': {'enabled': True, 'config': {}},
                'drawdown_velocity': {'enabled': True, 'config': {}},
                'expected_shortfall': {'enabled': True, 'config': {}},
                'kyle_lambda': {'enabled': True, 'config': {}},
                'depth_shock': {'enabled': True, 'config': {}},
                'feed_staleness': {'enabled': True, 'config': {}},
                'latency_drift': {'enabled': True, 'config': {}},
                'adv_participation': {'enabled': True, 'config': {}}
            },
            'policies': [{
                'policy_id': 'diagnostic_policy',
                'policy_name': 'Diagnostic Policy',
                'rules': []
            }],
            'active_policy': 'diagnostic_policy'
        }
        
        v2 = create_risk_agent_v2(minimal_config)
        
        # Extract calculator names
        calc_names = [c.__class__.__name__ for c in v2.calculators]
        
        print(f"\n✅ RiskAgentV2 is initialized with {len(calc_names)} calculators:")
        pprint.pp(calc_names)
        
        # Additional diagnostic info
        print(f"\n📊 Calculator Details:")
        for i, calc in enumerate(v2.calculators, 1):
            calc_class = calc.__class__
            module_name = calc_class.__module__
            print(f"   {i}. {calc_class.__name__}")
            print(f"      Module: {module_name}")
            try:
                print(f"      File: {inspect.getfile(calc_class)}")
            except:
                print(f"      File: <unable to determine>")
            
            # Check if it has required methods
            has_calculate = hasattr(calc, 'calculate')
            has_get_fields = hasattr(calc, 'get_required_fields')
            print(f"      Methods: calculate={has_calculate}, get_required_fields={has_get_fields}")
            print()
        
        # Rules engine info
        print(f"🛡️ Rules Engine: {v2.rules_engine.__class__.__name__}")
        try:
            # Try different attribute names for policies
            if hasattr(v2.rules_engine, '_policies'):
                print(f"   Policies registered: {len(v2.rules_engine._policies)}")
            elif hasattr(v2.rules_engine, 'policies'):
                print(f"   Policies registered: {len(v2.rules_engine.policies)}")
            else:
                print(f"   Policies: <unable to determine count>")
        except Exception as e:
            print(f"   Policies: <error accessing: {e}>")
        
        # Limits config info
        print(f"⚙️ Limits Config Keys: {list(v2.limits_config.keys())}")
        
        print("\n" + "=" * 60)
        print("✅ DIAGNOSTIC COMPLETE")
        
    except Exception as e:
        print(f"❌ ERROR during diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()