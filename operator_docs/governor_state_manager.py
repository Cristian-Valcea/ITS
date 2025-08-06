#!/usr/bin/env python3
"""
🛡️ Risk Governor State Management
Handles PAUSED → RUNNING transition at 09:25 ET during smoke run
"""

import redis
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GovernorState(Enum):
    PAUSED = "PAUSED"
    RUNNING = "RUNNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MAINTENANCE = "MAINTENANCE"

class GovernorStateManager:
    """
    🛡️ Manages risk governor state transitions for production deployment
    
    Key Functions:
    - PAUSED → RUNNING at 09:25 ET (smoke run window)
    - Emergency stop capability
    - State persistence in Redis
    - Audit trail of all state changes
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize governor state manager"""
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis: {redis_host}:{redis_port}")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
        
        self.state_key = "risk_governor:state"
        self.audit_key = "risk_governor:audit_log"
        self.config_key = "risk_governor:config"
        
        # Initialize default state if not exists
        self._initialize_default_state()
    
    def _initialize_default_state(self):
        """Initialize default governor state"""
        
        if not self.redis_client:
            return
            
        try:
            current_state = self.redis_client.get(self.state_key)
            
            if not current_state:
                # Set initial PAUSED state
                initial_config = {
                    'state': GovernorState.PAUSED.value,
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'updated_by': 'system_init',
                    'reason': 'Initial system startup - awaiting smoke run validation',
                    'circuits': {
                        'position_limit': {'enabled': True, 'threshold': 1000},
                        'daily_loss_limit': {'enabled': True, 'threshold': 50},
                        'order_rate_limit': {'enabled': True, 'threshold': 10}
                    }
                }
                
                self.redis_client.set(self.state_key, json.dumps(initial_config))
                
                # Log initial state
                self._log_state_change(
                    from_state=None,
                    to_state=GovernorState.PAUSED,
                    reason="System initialization",
                    updated_by="system_init"
                )
                
                logger.info("🛡️ Governor initialized in PAUSED state")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize governor state: {e}")
    
    def get_current_state(self) -> Dict:
        """Get current governor state and configuration"""
        
        if not self.redis_client:
            return {'state': 'DISCONNECTED', 'error': 'Redis not available'}
        
        try:
            state_data = self.redis_client.get(self.state_key)
            
            if state_data:
                return json.loads(state_data)
            else:
                return {'state': 'UNKNOWN', 'error': 'State not found in Redis'}
                
        except Exception as e:
            logger.error(f"❌ Failed to get governor state: {e}")
            return {'state': 'ERROR', 'error': str(e)}
    
    def set_state(self, new_state: GovernorState, reason: str, updated_by: str = 'operator') -> bool:
        """
        Set new governor state with audit logging
        
        Args:
            new_state: New governor state
            reason: Reason for state change
            updated_by: Who initiated the change
            
        Returns:
            bool: Success/failure
        """
        
        if not self.redis_client:
            logger.error("❌ Cannot set state - Redis not available")
            return False
        
        try:
            # Get current state
            current_config = self.get_current_state()
            current_state_str = current_config.get('state', 'UNKNOWN')
            
            # Validate state transition
            if not self._validate_state_transition(current_state_str, new_state.value):
                logger.error(f"❌ Invalid state transition: {current_state_str} → {new_state.value}")
                return False
            
            # Update state configuration
            new_config = current_config.copy()
            new_config.update({
                'state': new_state.value,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'updated_by': updated_by,
                'reason': reason,
                'previous_state': current_state_str
            })
            
            # Save new state
            self.redis_client.set(self.state_key, json.dumps(new_config))
            
            # Log state change
            self._log_state_change(
                from_state=current_state_str,
                to_state=new_state,
                reason=reason,
                updated_by=updated_by
            )
            
            logger.info(f"🛡️ Governor state changed: {current_state_str} → {new_state.value}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Updated by: {updated_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set governor state: {e}")
            return False
    
    def _validate_state_transition(self, from_state: str, to_state: str) -> bool:
        """Validate if state transition is allowed"""
        
        # Define allowed transitions
        allowed_transitions = {
            'PAUSED': ['RUNNING', 'MAINTENANCE', 'EMERGENCY_STOP'],
            'RUNNING': ['PAUSED', 'EMERGENCY_STOP', 'MAINTENANCE'],
            'EMERGENCY_STOP': ['PAUSED', 'MAINTENANCE'],
            'MAINTENANCE': ['PAUSED'],
            'UNKNOWN': ['PAUSED']  # Allow recovery from unknown state
        }
        
        allowed = allowed_transitions.get(from_state, [])
        is_valid = to_state in allowed
        
        if not is_valid:
            logger.warning(f"⚠️ Invalid transition attempted: {from_state} → {to_state}")
            logger.info(f"   Allowed from {from_state}: {allowed}")
        
        return is_valid
    
    def _log_state_change(self, from_state: str, to_state: GovernorState, reason: str, updated_by: str):
        """Log state change to audit trail"""
        
        if not self.redis_client:
            return
        
        try:
            audit_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'from_state': from_state,
                'to_state': to_state.value,
                'reason': reason,
                'updated_by': updated_by,
                'system_info': {
                    'hostname': os.uname().nodename,
                    'pid': os.getpid()
                }
            }
            
            # Add to audit log list (keep last 100 entries)
            self.redis_client.lpush(self.audit_key, json.dumps(audit_entry))
            self.redis_client.ltrim(self.audit_key, 0, 99)  # Keep last 100
            
        except Exception as e:
            logger.error(f"❌ Failed to log state change: {e}")
    
    def get_audit_log(self, limit: int = 20) -> list:
        """Get recent state changes from audit log"""
        
        if not self.redis_client:
            return []
        
        try:
            log_entries = self.redis_client.lrange(self.audit_key, 0, limit - 1)
            return [json.loads(entry) for entry in log_entries]
            
        except Exception as e:
            logger.error(f"❌ Failed to get audit log: {e}")
            return []
    
    def smoke_run_transition(self) -> bool:
        """
        Execute the smoke run state transition (PAUSED → RUNNING at 09:25)
        """
        
        logger.info("🚀 Executing smoke run governor transition")
        
        current_state = self.get_current_state()
        
        if current_state.get('state') != 'PAUSED':
            logger.error(f"❌ Cannot transition to RUNNING - current state: {current_state.get('state')}")
            return False
        
        # Transition to RUNNING
        success = self.set_state(
            new_state=GovernorState.RUNNING,
            reason="Smoke run validation completed - transitioning to production",
            updated_by="smoke_run_automation"
        )
        
        if success:
            logger.info("🎉 Governor successfully transitioned to RUNNING")
            logger.info("🚀 System is now in PRODUCTION mode")
        else:
            logger.error("❌ Failed to transition governor to RUNNING")
        
        return success
    
    def emergency_stop(self, reason: str = "Emergency stop triggered") -> bool:
        """Emergency stop - immediately pause all trading"""
        
        logger.warning(f"🚨 EMERGENCY STOP: {reason}")
        
        return self.set_state(
            new_state=GovernorState.EMERGENCY_STOP,
            reason=f"EMERGENCY STOP: {reason}",
            updated_by="emergency_operator"
        )

def main():
    """Command-line interface for governor state management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Risk Governor State Management")
    parser.add_argument('--status', action='store_true', help='Show current governor status')
    parser.add_argument('--set-state', choices=['PAUSED', 'RUNNING', 'EMERGENCY_STOP', 'MAINTENANCE'], 
                       help='Set governor state')
    parser.add_argument('--reason', default='Manual operator change', help='Reason for state change')
    parser.add_argument('--smoke-run', action='store_true', help='Execute smoke run transition')
    parser.add_argument('--emergency-stop', action='store_true', help='Emergency stop')
    parser.add_argument('--audit-log', action='store_true', help='Show recent state changes')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create governor manager
    governor = GovernorStateManager()
    
    if args.status:
        state = governor.get_current_state()
        print(f"\n🛡️ RISK GOVERNOR STATUS")
        print(f"{'='*40}")
        print(f"State: {state.get('state', 'UNKNOWN')}")
        print(f"Last Updated: {state.get('last_updated', 'Unknown')}")
        print(f"Updated By: {state.get('updated_by', 'Unknown')}")
        print(f"Reason: {state.get('reason', 'Unknown')}")
        
        if 'circuits' in state:
            print(f"\n🔧 Circuit Breakers:")
            for circuit, config in state['circuits'].items():
                status = "✅ Enabled" if config.get('enabled') else "❌ Disabled"
                print(f"   {circuit}: {status} (threshold: {config.get('threshold')})")
    
    elif args.set_state:
        new_state = GovernorState(args.set_state)
        success = governor.set_state(new_state, args.reason, 'cli_operator')
        
        if success:
            print(f"✅ Governor state set to: {args.set_state}")
        else:
            print(f"❌ Failed to set governor state to: {args.set_state}")
    
    elif args.smoke_run:
        print("🚀 Executing smoke run governor transition...")
        success = governor.smoke_run_transition()
        
        if success:
            print("✅ Smoke run transition completed - Governor is RUNNING")
        else:
            print("❌ Smoke run transition failed")
    
    elif args.emergency_stop:
        print("🚨 Executing emergency stop...")
        success = governor.emergency_stop("Manual emergency stop via CLI")
        
        if success:
            print("✅ Emergency stop executed - Governor is STOPPED")
        else:
            print("❌ Emergency stop failed")
    
    elif args.audit_log:
        log_entries = governor.get_audit_log(10)
        
        print(f"\n📋 GOVERNOR AUDIT LOG (Last 10 entries)")
        print(f"{'='*60}")
        
        for entry in log_entries:
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            print(f"{timestamp} | {entry['from_state']} → {entry['to_state']}")
            print(f"   Reason: {entry['reason']}")
            print(f"   By: {entry['updated_by']}")
            print()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()