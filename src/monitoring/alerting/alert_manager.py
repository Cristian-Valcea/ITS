# src/monitoring/alerting/alert_manager.py
"""
Alert Manager for IntradayJules Trading System.

Integrates with PagerDuty and Slack for critical alerts:
- CRITICAL latency P99 > 25 µs
- Audit log write errors
- Circuit breaker trip count ≥ 1
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime, timezone


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert data structure."""
    alert_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    description: str
    service: str
    team: str
    timestamp: float
    labels: Dict[str, str]
    annotations: Dict[str, str]
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            **asdict(self),
            'severity': self.severity.value,
            'status': self.status.value,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        }


class PagerDutyClient:
    """PagerDuty integration client."""
    
    def __init__(self, integration_key: str, api_url: str = "https://events.pagerduty.com/v2/enqueue"):
        self.integration_key = integration_key
        self.api_url = api_url
        self.logger = logging.getLogger("PagerDutyClient")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        if not self.session:
            self.logger.error("PagerDuty client not initialized")
            return False
        
        try:
            # Map alert status to PagerDuty event action
            event_action = "trigger" if alert.status == AlertStatus.FIRING else "resolve"
            
            # Create PagerDuty event payload
            payload = {
                "routing_key": self.integration_key,
                "event_action": event_action,
                "dedup_key": f"{alert.service}_{alert.alert_name}",
                "payload": {
                    "summary": alert.message,
                    "source": f"IntradayJules-{alert.service}",
                    "severity": self._map_severity_to_pagerduty(alert.severity),
                    "component": alert.service,
                    "group": alert.team,
                    "class": "trading_system",
                    "custom_details": {
                        "description": alert.description,
                        "service": alert.service,
                        "team": alert.team,
                        "labels": alert.labels,
                        "annotations": alert.annotations,
                        "runbook_url": alert.runbook_url,
                        "dashboard_url": alert.dashboard_url,
                        "timestamp": alert.timestamp
                    }
                }
            }
            
            # Add links if available
            if alert.runbook_url or alert.dashboard_url:
                payload["payload"]["links"] = []
                if alert.runbook_url:
                    payload["payload"]["links"].append({
                        "href": alert.runbook_url,
                        "text": "Runbook"
                    })
                if alert.dashboard_url:
                    payload["payload"]["links"].append({
                        "href": alert.dashboard_url,
                        "text": "Dashboard"
                    })
            
            # Send to PagerDuty
            async with self.session.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 202:
                    response_data = await response.json()
                    self.logger.info(f"PagerDuty alert sent successfully: {response_data.get('dedup_key')}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"PagerDuty alert failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    def _map_severity_to_pagerduty(self, severity: AlertSeverity) -> str:
        """Map alert severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.INFO: "info"
        }
        return mapping.get(severity, "error")


class SlackClient:
    """Slack integration client."""
    
    def __init__(self, webhook_url: str, channel: str = "#trading-alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.logger = logging.getLogger("SlackClient")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.session:
            self.logger.error("Slack client not initialized")
            return False
        
        try:
            # Create Slack message payload
            color = self._get_alert_color(alert.severity, alert.status)
            
            # Build attachment fields
            fields = [
                {
                    "title": "Service",
                    "value": alert.service,
                    "short": True
                },
                {
                    "title": "Team",
                    "value": alert.team,
                    "short": True
                },
                {
                    "title": "Severity",
                    "value": alert.severity.value.upper(),
                    "short": True
                },
                {
                    "title": "Status",
                    "value": alert.status.value.upper(),
                    "short": True
                }
            ]
            
            # Add important labels as fields
            for key, value in alert.labels.items():
                if key in ['limit_type', 'current_value', 'limit_value', 'breaker_name']:
                    fields.append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            # Build actions (buttons)
            actions = []
            if alert.runbook_url:
                actions.append({
                    "type": "button",
                    "text": "📖 Runbook",
                    "url": alert.runbook_url,
                    "style": "primary"
                })
            if alert.dashboard_url:
                actions.append({
                    "type": "button",
                    "text": "📊 Dashboard",
                    "url": alert.dashboard_url
                })
            
            # Create Slack payload
            payload = {
                "channel": self.channel,
                "username": "IntradayJules Alert Manager",
                "icon_emoji": self._get_alert_emoji(alert.severity),
                "attachments": [
                    {
                        "color": color,
                        "title": alert.message,
                        "text": alert.description,
                        "fields": fields,
                        "actions": actions,
                        "footer": "IntradayJules Trading System",
                        "footer_icon": "https://example.com/trading-icon.png",
                        "ts": int(alert.timestamp)
                    }
                ]
            }
            
            # Send to Slack
            async with self.session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    self.logger.info(f"Slack alert sent successfully: {alert.alert_name}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Slack alert failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _get_alert_color(self, severity: AlertSeverity, status: AlertStatus) -> str:
        """Get Slack attachment color based on severity and status."""
        if status == AlertStatus.RESOLVED:
            return "good"  # Green
        
        color_map = {
            AlertSeverity.CRITICAL: "danger",   # Red
            AlertSeverity.WARNING: "warning",   # Yellow
            AlertSeverity.INFO: "#36a64f"       # Blue
        }
        return color_map.get(severity, "#808080")  # Gray default
    
    def _get_alert_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for alert severity."""
        emoji_map = {
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.INFO: ":information_source:"
        }
        return emoji_map.get(severity, ":question:")


class AlertManager:
    """
    Main Alert Manager for IntradayJules Trading System.
    
    Handles routing of alerts to PagerDuty and Slack based on severity.
    """
    
    def __init__(self, 
                 pagerduty_integration_key: Optional[str] = None,
                 slack_webhook_url: Optional[str] = None,
                 slack_channel: str = "#trading-alerts"):
        
        self.logger = logging.getLogger("AlertManager")
        
        # Initialize clients
        self.pagerduty_client = None
        if pagerduty_integration_key:
            self.pagerduty_client = PagerDutyClient(pagerduty_integration_key)
        
        self.slack_client = None
        if slack_webhook_url:
            self.slack_client = SlackClient(slack_webhook_url, slack_channel)
        
        # Alert routing configuration
        self.pagerduty_severities = {AlertSeverity.CRITICAL}  # Only critical alerts to PagerDuty
        self.slack_severities = {AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO}
        
        # Alert deduplication
        self.recent_alerts: Dict[str, float] = {}
        self.dedup_window = 300  # 5 minutes
        
        self.logger.info("AlertManager initialized")
    
    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert to configured destinations.
        
        Returns:
            Dict with success status for each destination
        """
        results = {}
        
        # Check for duplicate alerts
        alert_key = f"{alert.service}_{alert.alert_name}_{alert.status.value}"
        current_time = time.time()
        
        if alert_key in self.recent_alerts:
            if current_time - self.recent_alerts[alert_key] < self.dedup_window:
                self.logger.info(f"Skipping duplicate alert: {alert_key}")
                return {"deduplicated": True}
        
        self.recent_alerts[alert_key] = current_time
        
        # Clean up old alerts from dedup cache
        self._cleanup_dedup_cache(current_time)
        
        # Send to PagerDuty for critical alerts
        if (self.pagerduty_client and 
            alert.severity in self.pagerduty_severities):
            
            async with self.pagerduty_client as pd_client:
                results["pagerduty"] = await pd_client.send_alert(alert)
        
        # Send to Slack
        if (self.slack_client and 
            alert.severity in self.slack_severities):
            
            async with self.slack_client as slack_client:
                results["slack"] = await slack_client.send_alert(alert)
        
        # Log alert
        self.logger.info(f"Alert processed: {alert.alert_name} ({alert.severity.value}) - Results: {results}")
        
        return results
    
    def _cleanup_dedup_cache(self, current_time: float) -> None:
        """Clean up old entries from deduplication cache."""
        cutoff_time = current_time - self.dedup_window
        keys_to_remove = [
            key for key, timestamp in self.recent_alerts.items()
            if timestamp < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.recent_alerts[key]
    
    async def send_critical_latency_alert(self, service: str, latency_us: float) -> Dict[str, bool]:
        """Send critical latency alert."""
        alert = Alert(
            alert_name="CriticalLatencyP99High",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message=f"CRITICAL: {service} P99 latency > 25µs",
            description=f"{service} P99 latency is {latency_us:.1f}µs, exceeding critical threshold of 25µs. This may impact trading performance.",
            service=service,
            team="trading",
            timestamp=time.time(),
            labels={
                "service": service,
                "latency_us": str(latency_us),
                "threshold_us": "25"
            },
            annotations={
                "impact": "Trading performance degradation",
                "action_required": "Investigate performance bottlenecks"
            },
            runbook_url="https://wiki.company.com/runbooks/trading/latency-high",
            dashboard_url="http://grafana:3000/d/risk-enforcement/risk-enforcement-dashboard"
        )
        
        return await self.send_alert(alert)
    
    async def send_audit_log_error_alert(self, error_count: int, error_message: str) -> Dict[str, bool]:
        """Send audit log write error alert."""
        alert = Alert(
            alert_name="CriticalAuditLogWriteErrors",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="CRITICAL: Audit log write errors detected",
            description=f"{error_count} audit log write errors in the last 5 minutes. Compliance logging is failing. Error: {error_message}",
            service="audit_logging",
            team="trading",
            timestamp=time.time(),
            labels={
                "error_count": str(error_count),
                "error_type": "write_failure"
            },
            annotations={
                "impact": "Regulatory compliance at risk - audit trail incomplete",
                "action_required": "Fix audit logging immediately"
            },
            runbook_url="https://wiki.company.com/runbooks/trading/audit-log-errors"
        )
        
        return await self.send_alert(alert)
    
    async def send_circuit_breaker_alert(self, breaker_name: str, trip_count: int) -> Dict[str, bool]:
        """Send circuit breaker trip alert."""
        alert = Alert(
            alert_name="CriticalCircuitBreakerTripped",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="CRITICAL: Circuit breaker tripped",
            description=f"Circuit breaker '{breaker_name}' has tripped {trip_count} times in the last minute. Trading may be halted.",
            service="circuit_breaker",
            team="trading",
            timestamp=time.time(),
            labels={
                "breaker_name": breaker_name,
                "trip_count": str(trip_count)
            },
            annotations={
                "impact": "Trading operations may be suspended",
                "action_required": "Investigate circuit breaker cause and reset if safe"
            },
            runbook_url="https://wiki.company.com/runbooks/trading/circuit-breaker-trip"
        )
        
        return await self.send_alert(alert)
    
    async def send_risk_limit_breach_alert(self, limit_type: str, current_value: float, limit_value: float) -> Dict[str, bool]:
        """Send risk limit breach alert."""
        alert = Alert(
            alert_name="CriticalRiskLimitBreach",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="CRITICAL: Risk limit breached",
            description=f"Risk limit '{limit_type}' breached. Current value: ${current_value:,.0f}, Limit: ${limit_value:,.0f}",
            service="risk_management",
            team="trading",
            timestamp=time.time(),
            labels={
                "limit_type": limit_type,
                "current_value": f"{current_value:.0f}",
                "limit_value": f"{limit_value:.0f}",
                "breach_percentage": f"{((current_value - limit_value) / limit_value * 100):.1f}"
            },
            annotations={
                "impact": "Risk exposure exceeds acceptable levels",
                "action_required": "Reduce positions or adjust risk limits"
            },
            runbook_url="https://wiki.company.com/runbooks/trading/risk-limit-breach"
        )
        
        return await self.send_alert(alert)


def create_alert_manager() -> AlertManager:
    """Factory function to create AlertManager with environment configuration."""
    pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    slack_channel = os.getenv("SLACK_CHANNEL", "#trading-alerts")
    
    if not pagerduty_key:
        logging.warning("PAGERDUTY_INTEGRATION_KEY not set - PagerDuty alerts disabled")
    
    if not slack_webhook:
        logging.warning("SLACK_WEBHOOK_URL not set - Slack alerts disabled")
    
    return AlertManager(
        pagerduty_integration_key=pagerduty_key,
        slack_webhook_url=slack_webhook,
        slack_channel=slack_channel
    )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_alert_manager():
        """Test alert manager functionality."""
        # Create alert manager (will use env vars if available)
        alert_manager = create_alert_manager()
        
        # Test critical latency alert
        print("Testing critical latency alert...")
        result = await alert_manager.send_critical_latency_alert("risk_enforcement", 35.5)
        print(f"Latency alert result: {result}")
        
        # Test audit log error alert
        print("Testing audit log error alert...")
        result = await alert_manager.send_audit_log_error_alert(3, "Disk full")
        print(f"Audit log alert result: {result}")
        
        # Test circuit breaker alert
        print("Testing circuit breaker alert...")
        result = await alert_manager.send_circuit_breaker_alert("var_enforcement", 2)
        print(f"Circuit breaker alert result: {result}")
        
        # Test risk limit breach alert
        print("Testing risk limit breach alert...")
        result = await alert_manager.send_risk_limit_breach_alert("var_99_limit", 250000, 200000)
        print(f"Risk limit breach alert result: {result}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    asyncio.run(test_alert_manager())