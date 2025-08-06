#!/usr/bin/env python3
"""
🌐 Web-based Paper Trading Launcher
HTML interface for IBKR paper trading with clean slate functionality
"""

import os
import sys
import json
import logging
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.brokers.ibkr_account_manager import IBKRAccountManager
from operator_docs.governor_state_manager import GovernorStateManager

app = Flask(__name__)
app.secret_key = 'intradayjules_secure_key_2025'

# Global state
trading_process = None
trading_active = False
last_status_update = None

# Configure logging
log_dir = project_root / "logs" / "web_launcher"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'web_launcher.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WebTradingLauncher')

def get_system_status():
    """Get comprehensive system status"""
    
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'trading_active': trading_active,
        'ibkr_connection': 'Unknown',
        'risk_governor': 'Unknown',
        'positions': [],
        'open_orders': [],
        'pnl_summary': {},
        'model_status': 'Unknown',
        'alerts': []
    }
    
    # Check IBKR connection and account status
    try:
        manager = IBKRAccountManager()
        if manager.connect():
            status['ibkr_connection'] = 'Connected'
            
            # Get positions
            positions = manager.get_current_positions()
            status['positions'] = positions
            
            # Get open orders
            orders = manager.get_open_orders()
            status['open_orders'] = orders
            
            # Get P&L summary
            if positions:
                pnl = manager.get_pnl_summary()
                status['pnl_summary'] = {
                    'total_positions': pnl['total_positions'],
                    'market_value': pnl['total_market_value'],
                    'unrealized_pnl': pnl['total_unrealized_pnl'],
                    'realized_pnl': pnl['total_realized_pnl']
                }
            
            manager.disconnect()
        else:
            status['ibkr_connection'] = 'Failed'
            status['alerts'].append('IBKR connection failed')
            
    except Exception as e:
        status['ibkr_connection'] = f'Error: {str(e)}'
        status['alerts'].append(f'IBKR error: {str(e)}')
    
    # Check Risk Governor
    try:
        governor = GovernorStateManager()
        gov_state = governor.get_current_state()
        status['risk_governor'] = gov_state.get('state', 'Unknown')
        
        if status['risk_governor'] != 'RUNNING':
            status['alerts'].append(f'Risk Governor not running: {status["risk_governor"]}')
            
    except Exception as e:
        status['risk_governor'] = f'Error: {str(e)}'
        status['alerts'].append(f'Risk Governor error: {str(e)}')
    
    # Check model availability
    model_paths = [
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
    ]
    
    model_found = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            status['model_status'] = f'Available: {os.path.basename(model_path)}'
            model_found = True
            break
    
    if not model_found:
        status['model_status'] = 'Not found'
        status['alerts'].append('Trading model not found')
    
    return status

@app.route('/')
def index():
    """Main trading launcher page"""
    return render_template('trading_launcher.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    global last_status_update
    
    status = get_system_status()
    last_status_update = status
    
    return jsonify(status)

@app.route('/api/reset_account', methods=['POST'])
def api_reset_account():
    """API endpoint to reset IBKR paper account"""
    
    try:
        logger.info("🧹 Starting account reset via web interface")
        
        manager = IBKRAccountManager()
        if not manager.connect():
            return jsonify({
                'success': False,
                'message': 'Failed to connect to IBKR'
            })
        
        # Perform reset
        success = manager.reset_paper_account()
        manager.disconnect()
        
        if success:
            logger.info("✅ Account reset successful")
            return jsonify({
                'success': True,
                'message': 'Account reset successful - clean slate achieved'
            })
        else:
            logger.error("❌ Account reset failed")
            return jsonify({
                'success': False,
                'message': 'Account reset failed - check logs for details'
            })
            
    except Exception as e:
        logger.error(f"❌ Account reset error: {e}")
        return jsonify({
            'success': False,
            'message': f'Account reset error: {str(e)}'
        })

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """API endpoint to start paper trading"""
    global trading_process, trading_active
    
    try:
        data = request.get_json()
        clean_slate = data.get('clean_slate', False)
        symbols = data.get('symbols', ['NVDA', 'MSFT'])
        position_size = data.get('position_size', 10)
        
        logger.info(f"🚀 Starting trading via web interface")
        logger.info(f"   Clean slate: {clean_slate}")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Position size: ${position_size}")
        
        # Step 1: Clean slate if requested
        if clean_slate:
            logger.info("Step 1: Performing account reset...")
            manager = IBKRAccountManager()
            if manager.connect():
                reset_success = manager.reset_paper_account()
                manager.disconnect()
                
                if not reset_success:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to reset account - cannot start with clean slate'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to connect for account reset'
                })
        
        # Step 2: Check if already running
        if trading_active:
            return jsonify({
                'success': False,
                'message': 'Trading is already active'
            })
        
        # Step 3: Start trading process
        logger.info("Step 2: Starting production deployment...")
        
        # Activate venv and run production deployment
        cmd = [
            'bash', '-c',
            f'source venv/bin/activate && python production_deployment.py'
        ]
        
        trading_process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        trading_active = True
        
        logger.info("✅ Trading started successfully")
        
        return jsonify({
            'success': True,
            'message': f'Paper trading started successfully with {len(symbols)} symbols',
            'details': {
                'clean_slate_performed': clean_slate,
                'symbols': symbols,
                'position_size': position_size,
                'process_id': trading_process.pid
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to start trading: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to start trading: {str(e)}'
        })

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    """API endpoint to stop paper trading"""
    global trading_process, trading_active
    
    try:
        if not trading_active or not trading_process:
            return jsonify({
                'success': False,
                'message': 'Trading is not currently active'
            })
        
        logger.info("🛑 Stopping trading via web interface")
        
        # Terminate trading process
        trading_process.terminate()
        
        # Wait for graceful shutdown
        try:
            trading_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Trading process didn't shutdown gracefully, forcing kill")
            trading_process.kill()
            trading_process.wait()
        
        trading_active = False
        trading_process = None
        
        logger.info("✅ Trading stopped successfully")
        
        return jsonify({
            'success': True,
            'message': 'Paper trading stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to stop trading: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to stop trading: {str(e)}'
        })

@app.route('/api/logs')
def api_logs():
    """API endpoint to get recent logs"""
    
    try:
        # Get most recent production log
        log_dir = Path('logs/production')
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                # Read last 50 lines
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                
                return jsonify({
                    'success': True,
                    'log_file': str(latest_log),
                    'lines': [line.strip() for line in recent_lines]
                })
        
        return jsonify({
            'success': False,
            'message': 'No production logs found'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to read logs: {str(e)}'
        })

if __name__ == '__main__':
    # Create templates directory and HTML file
    templates_dir = project_root / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 IntradayJules Paper Trading Launcher</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        .panel {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            border: 1px solid #e9ecef;
        }
        .panel h2 {
            margin: 0 0 20px 0;
            color: #2c3e50;
            font-size: 1.4em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .status-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        .status-item.error {
            border-left-color: #e74c3c;
        }
        .status-item.success {
            border-left-color: #27ae60;
        }
        .status-item.warning {
            border-left-color: #f39c12;
        }
        .status-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .status-value {
            color: #666;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        .form-control {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        .checkbox-group input[type="checkbox"] {
            transform: scale(1.2);
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        .btn-primary {
            background: #3498db;
            color: white;
        }
        .btn-primary:hover {
            background: #2980b9;
        }
        .btn-success {
            background: #27ae60;
            color: white;
        }
        .btn-success:hover {
            background: #229954;
        }
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .btn-warning {
            background: #f39c12;
            color: white;
        }
        .btn-warning:hover {
            background: #e67e22;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .alert {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        .alert-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .alert-error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .alert-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .positions-table th,
        .positions-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .positions-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .log-output {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 12px;
            height: 300px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IntradayJules Paper Trading Launcher</h1>
            <p>Enhanced IBKR Integration with Risk Governor</p>
        </div>
        
        <div class="main-content">
            <!-- System Status Panel -->
            <div class="panel">
                <h2>📊 System Status</h2>
                <div class="status-grid">
                    <div class="status-item" id="ibkr-status">
                        <div class="status-label">IBKR Connection</div>
                        <div class="status-value">Checking...</div>
                    </div>
                    <div class="status-item" id="governor-status">
                        <div class="status-label">Risk Governor</div>
                        <div class="status-value">Checking...</div>
                    </div>
                    <div class="status-item" id="model-status">
                        <div class="status-label">Trading Model</div>
                        <div class="status-value">Checking...</div>
                    </div>
                    <div class="status-item" id="trading-status">
                        <div class="status-label">Trading Status</div>
                        <div class="status-value">Inactive</div>
                    </div>
                </div>
                
                <div id="alerts-container"></div>
                
                <button class="btn btn-primary" onclick="refreshStatus()">
                    <span id="refresh-loading" style="display: none;" class="loading"></span>
                    Refresh Status
                </button>
            </div>
            
            <!-- Trading Control Panel -->
            <div class="panel">
                <h2>🎯 Trading Controls</h2>
                
                <div class="form-group">
                    <label>Trading Symbols</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="nvda-symbol" checked>
                        <label for="nvda-symbol">NVDA</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="msft-symbol" checked>
                        <label for="msft-symbol">MSFT</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="position-size">Position Size ($)</label>
                    <input type="number" id="position-size" class="form-control" value="10" min="1" max="100">
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="clean-slate" checked>
                        <label for="clean-slate">🧹 Clean Slate (Reset account before start)</label>
                    </div>
                    <small style="color: #666;">
                        This will cancel all orders and flatten all positions before starting
                    </small>
                </div>
                
                <div id="control-message"></div>
                
                <button class="btn btn-success" id="start-btn" onclick="startTrading()">
                    <span id="start-loading" style="display: none;" class="loading"></span>
                    🚀 Start Paper Trading
                </button>
                
                <button class="btn btn-danger" id="stop-btn" onclick="stopTrading()" disabled>
                    🛑 Stop Trading
                </button>
                
                <button class="btn btn-warning" onclick="resetAccount()">
                    <span id="reset-loading" style="display: none;" class="loading"></span>
                    🧹 Reset Account Only
                </button>
            </div>
            
            <!-- Positions Panel -->
            <div class="panel">
                <h2>💰 Current Positions</h2>
                <div id="positions-container">
                    <p>No positions loaded. Click "Refresh Status" to update.</p>
                </div>
            </div>
            
            <!-- Orders Panel -->
            <div class="panel">
                <h2>📋 Open Orders</h2>
                <div id="orders-container">
                    <p>No orders loaded. Click "Refresh Status" to update.</p>
                </div>
            </div>
            
            <!-- Logs Panel -->
            <div class="panel full-width">
                <h2>📝 System Logs</h2>
                <button class="btn btn-primary" onclick="refreshLogs()">
                    <span id="logs-loading" style="display: none;" class="loading"></span>
                    Refresh Logs
                </button>
                <div id="log-output" class="log-output">
                    Click "Refresh Logs" to load recent system logs...
                </div>
            </div>
        </div>
    </div>

    <script>
        let statusInterval;
        
        // Auto-refresh status every 30 seconds
        function startStatusUpdates() {
            refreshStatus();
            statusInterval = setInterval(refreshStatus, 30000);
        }
        
        function stopStatusUpdates() {
            if (statusInterval) {
                clearInterval(statusInterval);
            }
        }
        
        async function refreshStatus() {
            const loadingEl = document.getElementById('refresh-loading');
            loadingEl.style.display = 'inline-block';
            
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                updateStatusDisplay(status);
                
            } catch (error) {
                console.error('Failed to refresh status:', error);
                showMessage('Failed to refresh status: ' + error.message, 'error');
            } finally {
                loadingEl.style.display = 'none';
            }
        }
        
        function updateStatusDisplay(status) {
            // Update status items
            updateStatusItem('ibkr-status', status.ibkr_connection);
            updateStatusItem('governor-status', status.risk_governor);
            updateStatusItem('model-status', status.model_status);
            updateStatusItem('trading-status', status.trading_active ? 'Active' : 'Inactive');
            
            // Update alerts
            updateAlerts(status.alerts);
            
            // Update positions
            updatePositions(status.positions, status.pnl_summary);
            
            // Update orders
            updateOrders(status.open_orders);
            
            // Update trading controls
            updateTradingControls(status.trading_active);
        }
        
        function updateStatusItem(itemId, value) {
            const item = document.getElementById(itemId);
            const valueEl = item.querySelector('.status-value');
            valueEl.textContent = value;
            
            // Update styling based on status
            item.className = 'status-item';
            if (value.includes('Connected') || value.includes('RUNNING') || value.includes('Available') || value === 'Active') {
                item.classList.add('success');
            } else if (value.includes('Error') || value.includes('Failed') || value.includes('Not found')) {
                item.classList.add('error');
            } else {
                item.classList.add('warning');
            }
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (alerts && alerts.length > 0) {
                container.innerHTML = alerts.map(alert => 
                    `<div class="alert alert-warning">⚠️ ${alert}</div>`
                ).join('');
            } else {
                container.innerHTML = '<div class="alert alert-success">✅ No active alerts</div>';
            }
        }
        
        function updatePositions(positions, pnlSummary) {
            const container = document.getElementById('positions-container');
            
            if (positions && positions.length > 0) {
                let html = '<table class="positions-table">';
                html += '<tr><th>Symbol</th><th>Position</th><th>Avg Cost</th><th>Market Price</th><th>P&L</th></tr>';
                
                positions.forEach(pos => {
                    html += `<tr>
                        <td><strong>${pos.symbol}</strong></td>
                        <td>${pos.position}</td>
                        <td>$${pos.avg_cost.toFixed(2)}</td>
                        <td>$${pos.market_price.toFixed(2)}</td>
                        <td style="color: ${pos.unrealized_pnl >= 0 ? 'green' : 'red'}">
                            $${pos.unrealized_pnl.toFixed(2)}
                        </td>
                    </tr>`;
                });
                
                html += '</table>';
                
                if (pnlSummary && pnlSummary.total_positions > 0) {
                    html += `<div style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-radius: 4px;">
                        <strong>Total P&L: $${pnlSummary.unrealized_pnl.toFixed(2)}</strong> |
                        Market Value: $${pnlSummary.market_value.toFixed(2)}
                    </div>`;
                }
                
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>✅ No open positions (clean slate)</p>';
            }
        }
        
        function updateOrders(orders) {
            const container = document.getElementById('orders-container');
            
            if (orders && orders.length > 0) {
                let html = '<table class="positions-table">';
                html += '<tr><th>Order ID</th><th>Symbol</th><th>Action</th><th>Quantity</th><th>Type</th><th>Status</th></tr>';
                
                orders.forEach(order => {
                    html += `<tr>
                        <td>${order.order_id}</td>
                        <td>${order.symbol}</td>
                        <td>${order.action}</td>
                        <td>${order.quantity}</td>
                        <td>${order.order_type}</td>
                        <td>${order.status}</td>
                    </tr>`;
                });
                
                html += '</table>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>✅ No open orders</p>';
            }
        }
        
        function updateTradingControls(tradingActive) {
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            startBtn.disabled = tradingActive;
            stopBtn.disabled = !tradingActive;
        }
        
        async function startTrading() {
            const startBtn = document.getElementById('start-btn');
            const loadingEl = document.getElementById('start-loading');
            
            startBtn.disabled = true;
            loadingEl.style.display = 'inline-block';
            
            try {
                const symbols = [];
                if (document.getElementById('nvda-symbol').checked) symbols.push('NVDA');
                if (document.getElementById('msft-symbol').checked) symbols.push('MSFT');
                
                const positionSize = parseInt(document.getElementById('position-size').value);
                const cleanSlate = document.getElementById('clean-slate').checked;
                
                if (symbols.length === 0) {
                    throw new Error('Please select at least one symbol');
                }
                
                const response = await fetch('/api/start_trading', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: symbols,
                        position_size: positionSize,
                        clean_slate: cleanSlate
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showMessage(result.message, 'success');
                    setTimeout(refreshStatus, 2000); // Refresh status after 2 seconds
                } else {
                    showMessage(result.message, 'error');
                }
                
            } catch (error) {
                showMessage('Failed to start trading: ' + error.message, 'error');
            } finally {
                loadingEl.style.display = 'none';
                startBtn.disabled = false;
            }
        }
        
        async function stopTrading() {
            const stopBtn = document.getElementById('stop-btn');
            
            stopBtn.disabled = true;
            
            try {
                const response = await fetch('/api/stop_trading', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showMessage(result.message, 'success');
                    setTimeout(refreshStatus, 2000);
                } else {
                    showMessage(result.message, 'error');
                }
                
            } catch (error) {
                showMessage('Failed to stop trading: ' + error.message, 'error');
            } finally {
                stopBtn.disabled = false;
            }
        }
        
        async function resetAccount() {
            const loadingEl = document.getElementById('reset-loading');
            
            if (!confirm('Are you sure you want to reset the paper trading account? This will cancel all orders and flatten all positions.')) {
                return;
            }
            
            loadingEl.style.display = 'inline-block';
            
            try {
                const response = await fetch('/api/reset_account', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showMessage(result.message, 'success');
                    setTimeout(refreshStatus, 2000);
                } else {
                    showMessage(result.message, 'error');
                }
                
            } catch (error) {
                showMessage('Failed to reset account: ' + error.message, 'error');
            } finally {
                loadingEl.style.display = 'none';
            }
        }
        
        async function refreshLogs() {
            const loadingEl = document.getElementById('logs-loading');
            const logOutput = document.getElementById('log-output');
            
            loadingEl.style.display = 'inline-block';
            
            try {
                const response = await fetch('/api/logs');
                const result = await response.json();
                
                if (result.success) {
                    logOutput.innerHTML = result.lines.join('\\n');
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else {
                    logOutput.innerHTML = 'Failed to load logs: ' + result.message;
                }
                
            } catch (error) {
                logOutput.innerHTML = 'Error loading logs: ' + error.message;
            } finally {
                loadingEl.style.display = 'none';
            }
        }
        
        function showMessage(message, type) {
            const container = document.getElementById('control-message');
            const className = type === 'success' ? 'alert-success' : 
                             type === 'error' ? 'alert-error' : 'alert-warning';
            
            container.innerHTML = `<div class="alert ${className}">${message}</div>`;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                container.innerHTML = '';
            }, 5000);
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            startStatusUpdates();
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            stopStatusUpdates();
        });
    </script>
</body>
</html>'''
    
    with open(templates_dir / 'trading_launcher.html', 'w') as f:
        f.write(html_template)
    
    print("🌐 Starting Web Trading Launcher...")
    print("=" * 50)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"📝 Log File: {log_dir / 'web_launcher.log'}")
    print(f"🎯 Features:")
    print(f"   - Real-time system status monitoring")
    print(f"   - IBKR account reset (clean slate)")
    print(f"   - Paper trading start/stop controls")
    print(f"   - Position and order monitoring")
    print(f"   - Live system logs")
    print("=" * 50)
    print("Press Ctrl+C to stop the web server")
    print()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )