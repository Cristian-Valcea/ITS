"""
Prometheus Monitoring and Alerting System
Real-time metrics collection and alerting for production risk governor
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from prometheus_client.core import CollectorRegistry
import json
import requests
from datetime import datetime, timezone

# Prometheus metrics
REGISTRY = CollectorRegistry()

# Risk Governor Metrics
DECISION_LATENCY = Histogram(
    'risk_governor_decision_latency_seconds',
    'Time taken for risk governor decisions',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY
)

ORDER_EXECUTIONS = Counter(
    'risk_governor_order_executions_total',
    'Total number of order executions', 
    ['symbol', 'status'],
    registry=REGISTRY
)

POSITION_SIZE = Gauge(
    'risk_governor_current_position',
    'Current position size in dollars',
    ['symbol'],
    registry=REGISTRY
)

DAILY_PNL = Gauge(
    'risk_governor_daily_pnl',
    'Daily P&L in dollars',
    registry=REGISTRY
)

DAILY_TURNOVER = Gauge(
    'risk_governor_daily_turnover',
    'Daily turnover in dollars',
    registry=REGISTRY
)

RISK_BUDGET_USAGE = Gauge(
    'risk_governor_risk_budget_usage',
    'Risk budget usage as percentage',
    ['budget_type'],
    registry=REGISTRY
)

HARD_LIMIT_BREACHES = Counter(
    'risk_governor_hard_limit_breaches_total',
    'Total hard limit breaches (should always be 0)',
    ['limit_type'],
    registry=REGISTRY
)

SYSTEM_ERRORS = Counter(
    'risk_governor_system_errors_total',
    'Total system errors',
    ['error_type'],
    registry=REGISTRY
)

MODEL_PREDICTIONS = Counter(
    'risk_governor_model_predictions_total',
    'Total model predictions',
    ['action', 'confidence_bucket'],
    registry=REGISTRY
)

@dataclass
class AlertRule:
    """Alert configuration"""
    name: str
    metric_query: str
    threshold: float
    duration: str  # e.g., "1m", "5m"
    severity: str  # "warning", "critical"
    description: str
    callback: Optional[Callable] = None

class PrometheusMonitor:
    """
    Production monitoring system with Prometheus metrics and alerting
    """
    
    def __init__(self, 
                 metrics_port: int = 8000,
                 alert_webhook: Optional[str] = None,
                 slack_webhook: Optional[str] = None):
        
        self.metrics_port = metrics_port
        self.alert_webhook = alert_webhook  
        self.slack_webhook = slack_webhook
        
        self.logger = logging.getLogger("PrometheusMonitor")
        
        # Alert state tracking
        self.active_alerts = set()
        self.alert_history = []
        self.alert_callbacks = {}
        
        # Metrics collection
        self.metrics_server_started = False
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance tracking
        self.performance_window = []
        self.max_window_size = 1000  # Keep last 1000 measurements
        
        self.logger.info("Prometheus Monitor initialized")
    
    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        if self.metrics_server_started:
            return
        
        try:
            start_http_server(self.metrics_port, registry=REGISTRY)
            self.metrics_server_started = True
            self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    def record_decision_latency(self, latency_ms: float):
        """Record risk governor decision latency"""
        latency_seconds = latency_ms / 1000.0
        DECISION_LATENCY.observe(latency_seconds)
        
        # Track in performance window
        self.performance_window.append({
            'timestamp': time.time(),
            'latency_ms': latency_ms
        })
        
        # Maintain window size
        if len(self.performance_window) > self.max_window_size:
            self.performance_window.pop(0)
        
        # Check latency alert
        if latency_ms > 10.0:  # Alert if > 10ms
            self._trigger_alert("high_latency", f"Decision latency {latency_ms:.2f}ms exceeds 10ms threshold")
    
    def record_order_execution(self, symbol: str, success: bool, execution_details: Optional[Dict] = None):
        """Record order execution"""
        status = "success" if success else "failed"
        ORDER_EXECUTIONS.labels(symbol=symbol, status=status).inc()
        
        if execution_details:
            # Update position gauge
            if "position_after" in execution_details:
                POSITION_SIZE.labels(symbol=symbol).set(execution_details["position_after"])
    
    def update_daily_metrics(self, pnl: float, turnover: float):
        """Update daily P&L and turnover metrics"""
        DAILY_PNL.set(pnl)
        DAILY_TURNOVER.set(turnover)
        
        # Check loss alert
        if pnl < -20.0:  # Alert if loss > $20
            self._trigger_alert("daily_loss", f"Daily loss ${abs(pnl):.2f} exceeds $20 threshold")
    
    def update_risk_budget_usage(self, 
                                position_usage: float, 
                                turnover_usage: float, 
                                loss_usage: float):
        """Update risk budget usage percentages"""
        RISK_BUDGET_USAGE.labels(budget_type="position").set(position_usage)
        RISK_BUDGET_USAGE.labels(budget_type="turnover").set(turnover_usage)
        RISK_BUDGET_USAGE.labels(budget_type="loss").set(loss_usage)
        
        # Alert on high usage
        if position_usage > 0.8:  # 80% usage
            self._trigger_alert("high_position_usage", f"Position budget usage {position_usage:.1%}")
        
        if turnover_usage > 0.8:
            self._trigger_alert("high_turnover_usage", f"Turnover budget usage {turnover_usage:.1%}")
        
        if loss_usage > 0.8:
            self._trigger_alert("high_loss_usage", f"Loss budget usage {loss_usage:.1%}")
    
    def record_hard_limit_breach(self, limit_type: str, details: str):
        """Record hard limit breach (CRITICAL)"""
        HARD_LIMIT_BREACHES.labels(limit_type=limit_type).inc()
        
        # This should NEVER happen - immediate critical alert
        self._trigger_alert(
            "hard_limit_breach", 
            f"CRITICAL: Hard limit breach - {limit_type}: {details}",
            severity="critical"
        )
    
    def record_system_error(self, error_type: str, error_message: str):
        """Record system error"""
        SYSTEM_ERRORS.labels(error_type=error_type).inc()
        
        # Calculate error rate
        current_time = time.time()
        recent_errors = [entry for entry in self.alert_history 
                        if entry.get('type') == 'system_error' and 
                        current_time - entry.get('timestamp', 0) < 300]  # 5 minutes
        
        error_rate = len(recent_errors) / 300  # errors per second
        
        if error_rate > 0.02:  # > 2% error rate
            self._trigger_alert("high_error_rate", f"Error rate {error_rate:.1%} > 2%")
    
    def record_model_prediction(self, action: str, confidence: float):
        """Record model prediction with confidence bucket"""
        if confidence < 0.3:
            confidence_bucket = "low"
        elif confidence < 0.7:
            confidence_bucket = "medium"
        else:
            confidence_bucket = "high"
        
        MODEL_PREDICTIONS.labels(action=action, confidence_bucket=confidence_bucket).inc()
    
    def start_monitoring(self, alert_rules: Optional[List[AlertRule]] = None):
        """Start continuous monitoring with alert rules"""
        if self.monitoring_active:
            return
        
        # Set up default alert rules
        if alert_rules is None:
            alert_rules = self._get_default_alert_rules()
        
        for rule in alert_rules:
            if rule.callback:
                self.alert_callbacks[rule.name] = rule.callback
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Continuous monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check performance metrics
                self._check_performance_alerts()
                
                # Check system health
                self._check_system_health()
                
                # Process any pending alerts
                self._process_alert_queue()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer after errors
    
    def _check_performance_alerts(self):
        """Check performance-based alerts"""
        if len(self.performance_window) < 10:
            return
        
        # Calculate recent average latency
        recent_latencies = [entry['latency_ms'] for entry in self.performance_window[-60:]]  # Last 60 measurements
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        
        if avg_latency > 10.0:
            self._trigger_alert("sustained_high_latency", 
                              f"Average latency {avg_latency:.2f}ms over 1-minute window")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check if metrics are being updated (no flatlines)
            current_time = time.time()
            
            if self.performance_window:
                last_update = self.performance_window[-1]['timestamp']
                if current_time - last_update > 300:  # 5 minutes
                    self._trigger_alert("metrics_flatline", "No metrics updates in 5 minutes")
        
        except Exception as e:
            self.logger.error(f"System health check error: {e}")
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Trigger an alert"""
        alert_key = f"{alert_type}:{message}"
        
        # Avoid duplicate alerts
        if alert_key in self.active_alerts:
            return
        
        self.active_alerts.add(alert_key)
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "datetime": datetime.now(timezone.utc).isoformat()
        }
        
        self.alert_history.append(alert_data)
        
        # Keep alert history manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]  # Keep last 500
        
        self.logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # Send notifications
        self._send_alert_notifications(alert_data)
        
        # Execute callback if configured
        if alert_type in self.alert_callbacks:
            try:
                self.alert_callbacks[alert_type](alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def _send_alert_notifications(self, alert_data: Dict):
        """Send alert notifications via configured channels"""
        
        # Webhook notification
        if self.alert_webhook:
            try:
                response = requests.post(
                    self.alert_webhook,
                    json=alert_data,
                    timeout=5.0
                )
                if response.status_code == 200:
                    self.logger.info("Alert webhook sent successfully")
            except Exception as e:
                self.logger.error(f"Alert webhook failed: {e}")
        
        # Slack notification
        if self.slack_webhook and alert_data["severity"] == "critical":
            try:
                slack_message = {
                    "text": f"🚨 CRITICAL ALERT: {alert_data['message']}",
                    "attachments": [{
                        "color": "danger",
                        "fields": [
                            {"title": "Type", "value": alert_data["type"], "short": True},
                            {"title": "Time", "value": alert_data["datetime"], "short": True}
                        ]
                    }]
                }
                
                response = requests.post(
                    self.slack_webhook,
                    json=slack_message,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    self.logger.info("Slack alert sent successfully")
                    
            except Exception as e:
                self.logger.error(f"Slack alert failed: {e}")
    
    def _process_alert_queue(self):
        """Process and clear resolved alerts"""
        current_time = time.time()
        
        # Clear old alerts (auto-resolve after 1 hour)
        resolved_alerts = set()
        
        for alert_key in self.active_alerts:
            # Find alert in history
            alert_time = None
            for entry in reversed(self.alert_history):
                if f"{entry['type']}:{entry['message']}" == alert_key:
                    alert_time = entry['timestamp']
                    break
            
            if alert_time and current_time - alert_time > 3600:  # 1 hour
                resolved_alerts.add(alert_key)
        
        self.active_alerts -= resolved_alerts
        
        if resolved_alerts:
            self.logger.info(f"Auto-resolved {len(resolved_alerts)} alerts")
    
    def _get_default_alert_rules(self) -> List[AlertRule]:
        """Get default alert rules"""
        return [
            AlertRule(
                name="high_latency",
                metric_query="risk_governor_decision_latency_seconds > 0.01",
                threshold=0.01,
                duration="1m",
                severity="warning",
                description="Decision latency > 10ms"
            ),
            AlertRule(
                name="daily_loss",
                metric_query="risk_governor_daily_pnl < -20",
                threshold=-20.0,
                duration="0s",
                severity="warning",
                description="Daily loss > $20"
            ),
            AlertRule(
                name="hard_limit_breach",
                metric_query="risk_governor_hard_limit_breaches_total > 0",
                threshold=0.0,
                duration="0s",
                severity="critical",
                description="Hard limit breach detected"
            ),
            AlertRule(
                name="high_error_rate",
                metric_query="rate(risk_governor_system_errors_total[5m]) > 0.02",
                threshold=0.02,
                duration="1m",
                severity="warning",
                description="Error rate > 2%"
            )
        ]
    
    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary"""
        current_time = time.time()
        
        # Calculate recent performance
        recent_latencies = []
        if self.performance_window:
            cutoff = current_time - 300  # Last 5 minutes
            recent_latencies = [entry['latency_ms'] for entry in self.performance_window 
                              if entry['timestamp'] > cutoff]
        
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
        max_latency = max(recent_latencies) if recent_latencies else 0
        
        # Count recent alerts
        recent_alerts = [entry for entry in self.alert_history 
                        if current_time - entry['timestamp'] < 3600]  # Last hour
        
        return {
            "timestamp": current_time,
            "metrics_server_running": self.metrics_server_started,
            "monitoring_active": self.monitoring_active,
            "performance": {
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "recent_decisions": len(recent_latencies)
            },
            "alerts": {
                "active_alerts": len(self.active_alerts),
                "recent_alerts": len(recent_alerts),
                "total_alerts": len(self.alert_history)
            },
            "metrics_endpoint": f"http://localhost:{self.metrics_port}/metrics"
        }
    
    def clear_alert(self, alert_type: str, message: str):
        """Manually clear an alert"""
        alert_key = f"{alert_type}:{message}"
        if alert_key in self.active_alerts:
            self.active_alerts.remove(alert_key)
            self.logger.info(f"Manually cleared alert: {alert_key}")

# Integration class for easy setup
class RiskGovernorMonitoring:
    """
    Complete monitoring integration for Risk Governor
    """
    
    def __init__(self, 
                 prometheus_port: int = 8000,
                 alert_webhook: Optional[str] = None,
                 slack_webhook: Optional[str] = None):
        
        self.monitor = PrometheusMonitor(
            metrics_port=prometheus_port,
            alert_webhook=alert_webhook,
            slack_webhook=slack_webhook
        )
        
        self.logger = logging.getLogger("RiskGovernorMonitoring")
    
    def start(self):
        """Start complete monitoring system"""
        self.monitor.start_metrics_server()
        self.monitor.start_monitoring()
        self.logger.info("Risk Governor monitoring started")
    
    def stop(self):
        """Stop monitoring system"""
        self.monitor.stop_monitoring()
        self.logger.info("Risk Governor monitoring stopped")
    
    def record_trading_decision(self, result: Dict):
        """Record a complete trading decision result"""
        # Record latency
        if "total_latency_ms" in result:
            self.monitor.record_decision_latency(result["total_latency_ms"])
        
        # Record model prediction
        if "raw_action" in result and "model_confidence" in result:
            self.monitor.record_model_prediction(result["raw_action"], result["model_confidence"])
        
        # Record execution
        if "execution_result" in result:
            exec_result = result["execution_result"]
            success = exec_result.get("status") in ["EXECUTED", "SIMULATED"]
            symbol = result.get("portfolio_state", {}).get("symbol", "UNKNOWN")
            
            self.monitor.record_order_execution(symbol, success, {
                "position_after": result.get("portfolio_state", {}).get("current_position", 0)
            })
    
    def record_portfolio_state(self, portfolio_state: Dict):
        """Record current portfolio state"""
        pnl = portfolio_state.get("realized_pnl", 0) + portfolio_state.get("unrealized_pnl", 0)
        turnover = portfolio_state.get("daily_turnover", 0)
        
        self.monitor.update_daily_metrics(pnl, turnover)
        
        # Calculate risk budget usage (assuming $100 loss limit, $500 position limit, $2000 turnover limit)
        loss_usage = abs(min(0, pnl)) / 100.0
        position_usage = abs(portfolio_state.get("current_position", 0)) / 500.0
        turnover_usage = turnover / 2000.0
        
        self.monitor.update_risk_budget_usage(position_usage, turnover_usage, loss_usage)

# Example usage
def setup_monitoring(prometheus_port: int = 8000, 
                    slack_webhook: Optional[str] = None) -> RiskGovernorMonitoring:
    """Set up complete monitoring system"""
    
    monitoring = RiskGovernorMonitoring(
        prometheus_port=prometheus_port,
        slack_webhook=slack_webhook
    )
    
    monitoring.start()
    
    logging.getLogger("Monitoring").info(f"Monitoring system ready on port {prometheus_port}")
    
    return monitoring