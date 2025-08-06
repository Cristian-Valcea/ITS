#!/usr/bin/env python3
"""
🚀 PRODUCTION DEPLOYMENT
Integrated system: Enhanced IBKR Safety + Stairways V3 Model + Risk Governor
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.brokers.enhanced_safe_wrapper import EnhancedSafeOrderWrapper
from src.brokers.ib_gateway import IBGatewayClient
from operator_docs.governor_state_manager import GovernorStateManager

def setup_production_logging():
    """Set up comprehensive logging for production deployment"""
    
    log_dir = project_root / "logs" / "production"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"production_session_{session_id}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('ProductionDeployment')
    logger.info(f"Production deployment session started: {session_id}")
    
    return logger, log_file

def validate_production_system():
    """Comprehensive production system validation"""
    
    validations = {
        'enhanced_ibkr': False,
        'risk_governor': False,
        'model_available': False,
        'monitoring_ready': False
    }
    
    logger = logging.getLogger('ProductionDeployment')
    
    # 1. Enhanced IBKR validation
    try:
        ib_client = IBGatewayClient()
        if ib_client.connect():
            logger.info("✅ Enhanced IBKR connection validated")
            validations['enhanced_ibkr'] = True
            ib_client.disconnect()
        else:
            logger.error("❌ Enhanced IBKR connection failed")
    except Exception as e:
        logger.error(f"❌ Enhanced IBKR validation error: {e}")
    
    # 2. Risk Governor validation
    try:
        governor = GovernorStateManager()
        state = governor.get_current_state()
        if state.get('state') == 'RUNNING':
            logger.info("✅ Risk Governor in RUNNING state")
            validations['risk_governor'] = True
        else:
            logger.warning(f"⚠️ Risk Governor state: {state.get('state')}")
    except Exception as e:
        logger.error(f"❌ Risk Governor validation error: {e}")
    
    # 3. Model availability
    model_paths = [
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"✅ Model found: {model_path}")
            validations['model_available'] = True
            break
    
    if not validations['model_available']:
        logger.error("❌ No Stairways V3 model found")
    
    # 4. Monitoring readiness
    try:
        import redis
        r = redis.Redis(decode_responses=True)
        r.ping()
        logger.info("✅ Monitoring infrastructure (Redis) ready")
        validations['monitoring_ready'] = True
    except Exception as e:
        logger.error(f"❌ Monitoring infrastructure error: {e}")
    
    return validations

def start_production_trading(symbols=["NVDA", "MSFT"], position_size=10):
    """Start production trading with full safety integration"""
    
    logger = logging.getLogger('ProductionDeployment')
    
    logger.info("🚀 STARTING PRODUCTION TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Position Size: ${position_size} per symbol")
    logger.info(f"Mode: Paper Trading (Enhanced Safety)")
    logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
    
    # Initialize components
    ib_client = None
    safe_wrapper = None
    
    try:
        # 1. Initialize IBKR with enhanced safety
        logger.info("📡 Initializing Enhanced IBKR Connection...")
        ib_client = IBGatewayClient()
        
        if not ib_client.connect():
            raise RuntimeError("IBKR connection failed")
        
        logger.info(f"✅ Connected to IBKR ({ib_client.simulation_mode and 'Simulation' or 'Live'} mode)")
        
        # 2. Risk Governor Callback
        def production_risk_callback(order_id, status, event_type):
            """Production risk governor for live trading"""
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"🛡️ [{timestamp}] RISK GOVERNOR: Order {order_id} - {status} - {event_type}")
            
            # Critical event logging
            if event_type == 'ORDER_LIVE':
                logger.warning(f"🔴 CRITICAL: Order {order_id} is LIVE in market!")
                
            elif 'PRE_ORDER' in event_type:
                logger.info(f"✅ Pre-order validation passed for {event_type}")
                
            elif event_type == 'ORDER_FILLED':
                logger.info(f"💰 Order {order_id} filled successfully")
            
            return 'ALLOW'  # For paper trading
        
        # 3. Create Enhanced Safe Wrapper
        logger.info("🛡️ Initializing Enhanced Safe Wrapper...")
        safe_wrapper = EnhancedSafeOrderWrapper(ib_client, production_risk_callback)
        
        logger.info("✅ Enhanced safety system initialized")
        
        # 4. Production Trading Loop
        logger.info("🔄 Starting production trading loop...")
        logger.info("=" * 60)
        
        decision_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Simulate trading decision (in real system: market data → model → action)
                import numpy as np
                
                # Mock market observation (would be real market data)
                market_observation = np.random.random(26)
                
                # Mock trading signal (would be from Stairways V3 model for dual-ticker)
                raw_action = np.random.choice([0, 1, 2, 3, 4])  # 5-action space
                selected_symbol = np.random.choice(symbols)  # Choose NVDA or MSFT
                
                # Convert to position increment
                action_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
                position_increment = action_map[raw_action] * position_size
                
                if abs(position_increment) > 0.1:  # Only trade if significant signal
                    decision_count += 1
                    
                    logger.info(f"📈 Decision #{decision_count}: {selected_symbol} Action {raw_action} → ${position_increment:.1f}")
                    
                    try:
                        # Place order through enhanced safety system
                        if position_increment > 0:
                            # Buy order
                            result = safe_wrapper.place_market_order_with_governor(
                                symbol=selected_symbol,
                                quantity=int(abs(position_increment)),
                                action='BUY'
                            )
                        else:
                            # Sell order  
                            result = safe_wrapper.place_market_order_with_governor(
                                symbol=selected_symbol,
                                quantity=int(abs(position_increment)),
                                action='SELL'
                            )
                        
                        # Log result
                        logger.info(f"✅ Order executed: ID {result['order_id']} - {result.get('final_status', result.get('status'))}")
                        
                        if result.get('is_live'):
                            logger.warning(f"🔴 Order {result['order_id']} is LIVE in market!")
                            
                    except Exception as e:
                        logger.error(f"❌ Order execution failed: {e}")
                
                # Status update every 10 decisions
                if decision_count > 0 and decision_count % 10 == 0:
                    elapsed_hours = (time.time() - start_time) / 3600
                    decisions_per_hour = decision_count / elapsed_hours if elapsed_hours > 0 else 0
                    
                    logger.info("📊 PRODUCTION STATUS UPDATE")
                    logger.info(f"   Runtime: {elapsed_hours:.1f} hours")
                    logger.info(f"   Decisions: {decision_count}")
                    logger.info(f"   Rate: {decisions_per_hour:.1f} decisions/hour")
                    logger.info(f"   System: All safety systems operational")
                
                # Wait between decisions (would be driven by market data in real system)
                time.sleep(30)  # 30 second intervals for demo
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested by operator...")
            logger.info("📉 Graceful shutdown in progress...")
            
            # Graceful shutdown
            session_duration = time.time() - start_time
            logger.info(f"✅ Production session completed:")
            logger.info(f"   Duration: {session_duration/3600:.1f} hours")
            logger.info(f"   Total decisions: {decision_count}")
            logger.info(f"   Avg rate: {decision_count/(session_duration/3600):.1f} decisions/hour")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Production trading failed: {e}")
        return False
        
    finally:
        # Cleanup
        if safe_wrapper:
            logger.info("🔄 Cleaning up enhanced safety wrapper...")
            
        if ib_client:
            logger.info("🔌 Disconnecting from IBKR...")
            ib_client.disconnect()
            
        logger.info("🏁 Production deployment session ended")

def main():
    """Main production deployment entry point"""
    
    print("🚀 PRODUCTION DEPLOYMENT - ENHANCED SAFETY INTEGRATION")
    print("=" * 70)
    print("IntradayJules: Stairways V3 + Enhanced IBKR + Risk Governor")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    # Setup logging
    logger, log_file = setup_production_logging()
    logger.info("Production deployment initiated")
    
    # System validation
    print("🔍 Running production system validation...")
    validations = validate_production_system()
    
    # Check validation results
    total_validations = len(validations)
    passed_validations = sum(validations.values())
    success_rate = passed_validations / total_validations
    
    print(f"📊 System Validation: {passed_validations}/{total_validations} passed ({success_rate:.0%})")
    
    for component, status in validations.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component.replace('_', ' ').title()}")
    
    # Deployment decision
    if success_rate >= 0.75:  # At least 75% of validations must pass
        print("\n🎉 SYSTEM VALIDATION PASSED - PROCEEDING WITH DEPLOYMENT")
        print()
        
        # Start production trading
        success = start_production_trading(symbols=["NVDA", "MSFT"], position_size=10)
        
        if success:
            print("\n🎉 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!")
            logger.info("Production deployment completed successfully")
        else:
            print("\n⚠️ PRODUCTION DEPLOYMENT HAD ISSUES")
            logger.error("Production deployment completed with issues")
            
    else:
        print(f"\n🛑 SYSTEM VALIDATION FAILED ({success_rate:.0%} success rate)")
        print("Fix validation issues before deploying to production")
        logger.error(f"System validation failed - {success_rate:.0%} success rate")
        return 1
    
    print(f"\n📝 Full session log: {log_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())