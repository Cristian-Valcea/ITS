#!/usr/bin/env python3
"""
🌐 Fixed Web Trading Launcher
HTML interface with proper threading for IBKR connections
"""

import os
import sys
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
app.secret_key = 'intradayjules_secure_key_2025'

# Global state
trading_process = None
trading_active = False

def run_subprocess_command(cmd, cwd=None):
    """Run subprocess command safely"""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd or str(project_root),
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_system_status():
    """Get system status using subprocess calls instead of direct IBKR connections"""
    
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
    
    # Check IBKR connection via subprocess
    success, stdout, stderr = run_subprocess_command([
        'python', 'src/brokers/ibkr_account_manager.py', '--positions'
    ])
    
    if success:
        status['ibkr_connection'] = 'Connected'
        if "No positions" in stdout:
            status['positions'] = []
        else:
            # Parse positions from output if needed
            status['positions'] = []
    else:
        status['ibkr_connection'] = 'Failed'
        status['alerts'].append('IBKR connection failed')
    
    # Check Risk Governor via subprocess
    success, stdout, stderr = run_subprocess_command([
        'python', 'operator_docs/governor_state_manager.py', '--status'
    ])
    
    if success:
        if "State: RUNNING" in stdout:
            status['risk_governor'] = 'RUNNING'
        elif "State: PAUSED" in stdout:
            status['risk_governor'] = 'PAUSED'
            status['alerts'].append('Risk Governor is PAUSED')
        else:
            status['risk_governor'] = 'Unknown'
    else:
        status['risk_governor'] = 'Error'
        status['alerts'].append('Risk Governor error')
    
    # Check model availability
    model_paths = [
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            status['model_status'] = f'Available: {os.path.basename(model_path)}'
            break
    else:
        status['model_status'] = 'Not found'
        status['alerts'].append('Trading model not found')
    
    return status

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 IntradayJules Trading Launcher</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px; color: #333;
        }
        .container {
            max-width: 1000px; margin: 0 auto; background: white;
            border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white; padding: 30px; text-align: center; border-radius: 12px 12px 0 0;
        }
        .header h1 { font-size: 2.2em; margin-bottom: 10px; }
        .main { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; padding: 30px; }
        .panel {
            background: #f8f9fa; border-radius: 8px; padding: 25px;
            border: 1px solid #e9ecef;
        }
        .panel h2 {
            color: #2c3e50; font-size: 1.3em; margin-bottom: 20px;
            border-bottom: 2px solid #3498db; padding-bottom: 10px;
        }
        .status-item {
            background: white; padding: 15px; margin-bottom: 10px;
            border-radius: 6px; border-left: 4px solid #3498db;
        }
        .status-item.success { border-left-color: #27ae60; }
        .status-item.error { border-left-color: #e74c3c; }
        .status-item.warning { border-left-color: #f39c12; }
        .status-label { font-weight: 600; color: #2c3e50; margin-bottom: 5px; }
        .status-value { color: #666; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; }
        .form-control { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
        .checkbox-group { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .btn {
            padding: 12px 20px; border: none; border-radius: 6px;
            font-size: 14px; font-weight: 600; cursor: pointer;
            margin-right: 10px; margin-bottom: 10px; min-width: 140px;
        }
        .btn-primary { background: #3498db; color: white; }
        .btn-primary:hover { background: #2980b9; }
        .btn-success { background: #27ae60; color: white; }
        .btn-success:hover { background: #229954; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-danger:hover { background: #c0392b; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-warning:hover { background: #e67e22; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .alert {
            padding: 12px; border-radius: 6px; margin-bottom: 15px;
            border: 1px solid transparent;
        }
        .alert-success { background: #d4edda; color: #155724; border-color: #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .alert-warning { background: #fff3cd; color: #856404; border-color: #ffeaa7; }
        .full-width { grid-column: 1 / -1; }
        .loading {
            display: inline-block; width: 16px; height: 16px;
            border: 2px solid #f3f3f3; border-top: 2px solid #3498db;
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IntradayJules Trading Launcher</h1>
            <p>Enhanced IBKR Integration with Clean Slate Support</p>
        </div>
        
        <div class="main">
            <!-- System Status -->
            <div class="panel">
                <h2>📊 System Status</h2>
                <div id="status-container">
                    <div class="status-item">
                        <div class="status-label">Loading...</div>
                    </div>
                </div>
                <div id="alerts-container"></div>
                <button class="btn btn-primary" onclick="refreshStatus()">
                    <span id="refresh-loading" class="loading hidden"></span>
                    🔄 Refresh Status
                </button>
            </div>
            
            <!-- Trading Controls -->
            <div class="panel">
                <h2>🎯 Trading Controls</h2>
                
                <div class="form-group">
                    <label>Symbols (NVDA & MSFT)</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="nvda" checked> <label for="nvda">NVDA</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="msft" checked> <label for="msft">MSFT</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="position-size">Position Size ($)</label>
                    <input type="number" id="position-size" class="form-control" 
                           value="10" min="1" max="100">
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="clean-slate" checked>
                        <label for="clean-slate">🧹 Clean Slate Before Start</label>
                    </div>
                    <small style="color: #666;">Reset account (cancel orders, flatten positions)</small>
                </div>
                
                <div id="message-container"></div>
                
                <div>
                    <button class="btn btn-success" id="start-btn" onclick="startTrading()">
                        <span id="start-loading" class="loading hidden"></span>
                        🚀 Start Trading
                    </button>
                    <button class="btn btn-danger" id="stop-btn" onclick="stopTrading()" disabled>
                        🛑 Stop Trading
                    </button>
                    <button class="btn btn-warning" onclick="resetAccount()">
                        <span id="reset-loading" class="loading hidden"></span>
                        🧹 Reset Account
                    </button>
                </div>
            </div>
            
            <!-- Quick Actions -->
            <div class="panel full-width">
                <h2>⚡ Quick Actions</h2>
                <button class="btn btn-primary" onclick="showPositions()">📊 Show Positions</button>
                <button class="btn btn-primary" onclick="showOrders()">📋 Show Orders</button>
                <button class="btn btn-primary" onclick="showLogs()">📜 Show Logs</button>
                
                <div id="quick-output" style="margin-top: 15px; padding: 15px; background: #2c3e50; color: #ecf0f1; border-radius: 6px; font-family: monospace; font-size: 12px; height: 200px; overflow-y: auto; display: none;">
                </div>
            </div>
        </div>
    </div>

    <script>
        let statusInterval;
        
        function startAutoRefresh() {
            refreshStatus();
            statusInterval = setInterval(refreshStatus, 30000);
        }
        
        async function refreshStatus() {
            const loadingEl = document.getElementById('refresh-loading');
            loadingEl.classList.remove('hidden');
            
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateStatusDisplay(status);
            } catch (error) {
                showMessage('Status refresh failed: ' + error.message, 'error');
            } finally {
                loadingEl.classList.add('hidden');
            }
        }
        
        function updateStatusDisplay(status) {
            const container = document.getElementById('status-container');
            
            const items = [
                { label: 'IBKR Connection', value: status.ibkr_connection },
                { label: 'Risk Governor', value: status.risk_governor },
                { label: 'Trading Model', value: status.model_status },
                { label: 'Trading Status', value: status.trading_active ? 'Active' : 'Inactive' }
            ];
            
            container.innerHTML = items.map(item => {
                const className = getStatusClass(item.value);
                return `<div class="status-item ${className}">
                    <div class="status-label">${item.label}</div>
                    <div class="status-value">${item.value}</div>
                </div>`;
            }).join('');
            
            // Update alerts
            const alertsContainer = document.getElementById('alerts-container');
            if (status.alerts && status.alerts.length > 0) {
                alertsContainer.innerHTML = status.alerts.map(alert => 
                    `<div class="alert alert-warning">⚠️ ${alert}</div>`
                ).join('');
            } else {
                alertsContainer.innerHTML = '<div class="alert alert-success">✅ No alerts</div>';
            }
            
            // Update button states
            document.getElementById('start-btn').disabled = status.trading_active;
            document.getElementById('stop-btn').disabled = !status.trading_active;
        }
        
        function getStatusClass(value) {
            if (value.includes('Connected') || value.includes('RUNNING') || 
                value.includes('Available') || value === 'Active') {
                return 'success';
            } else if (value.includes('Error') || value.includes('Failed') || 
                      value.includes('Not found')) {
                return 'error';
            } else {
                return 'warning';
            }
        }
        
        async function startTrading() {
            const startBtn = document.getElementById('start-btn');
            const loadingEl = document.getElementById('start-loading');
            
            startBtn.disabled = true;
            loadingEl.classList.remove('hidden');
            
            try {
                const symbols = [];
                if (document.getElementById('nvda').checked) symbols.push('NVDA');
                if (document.getElementById('msft').checked) symbols.push('MSFT');
                
                if (symbols.length === 0) {
                    throw new Error('Please select at least one symbol');
                }
                
                const positionSize = parseInt(document.getElementById('position-size').value);
                const cleanSlate = document.getElementById('clean-slate').checked;
                
                const response = await fetch('/api/start_trading', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symbols: symbols,
                        position_size: positionSize,
                        clean_slate: cleanSlate
                    })
                });
                
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
                
            } catch (error) {
                showMessage('Start failed: ' + error.message, 'error');
            } finally {
                loadingEl.classList.add('hidden');
                startBtn.disabled = false;
            }
        }
        
        async function stopTrading() {
            try {
                const response = await fetch('/api/stop_trading', { method: 'POST' });
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
            } catch (error) {
                showMessage('Stop failed: ' + error.message, 'error');
            }
        }
        
        async function resetAccount() {
            if (!confirm('Reset account? This will cancel all orders and flatten positions.')) {
                return;
            }
            
            const loadingEl = document.getElementById('reset-loading');
            loadingEl.classList.remove('hidden');
            
            try {
                const response = await fetch('/api/reset_account', { method: 'POST' });
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
            } catch (error) {
                showMessage('Reset failed: ' + error.message, 'error');
            } finally {
                loadingEl.classList.add('hidden');
            }
        }
        
        async function showPositions() {
            showQuickOutput('Loading positions...');
            try {
                const response = await fetch('/api/positions');
                const result = await response.json();
                showQuickOutput(result.output || 'No positions data');
            } catch (error) {
                showQuickOutput('Error loading positions: ' + error.message);
            }
        }
        
        async function showOrders() {
            showQuickOutput('Loading orders...');
            try {
                const response = await fetch('/api/orders');
                const result = await response.json();
                showQuickOutput(result.output || 'No orders data');
            } catch (error) {
                showQuickOutput('Error loading orders: ' + error.message);
            }
        }
        
        async function showLogs() {
            showQuickOutput('Loading logs...');
            try {
                const response = await fetch('/api/logs');
                const result = await response.json();
                showQuickOutput(result.output || 'No logs available');
            } catch (error) {
                showQuickOutput('Error loading logs: ' + error.message);
            }
        }
        
        function showQuickOutput(text) {
            const output = document.getElementById('quick-output');
            output.style.display = 'block';
            output.textContent = text;
            output.scrollTop = output.scrollHeight;
        }
        
        function showMessage(message, type) {
            const container = document.getElementById('message-container');
            const className = type === 'success' ? 'alert-success' : 
                             type === 'error' ? 'alert-error' : 'alert-warning';
            
            container.innerHTML = `<div class="alert ${className}">${message}</div>`;
            setTimeout(() => container.innerHTML = '', 5000);
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', startAutoRefresh);
        window.addEventListener('beforeunload', () => {
            if (statusInterval) clearInterval(statusInterval);
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    return jsonify(get_system_status())

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    global trading_process, trading_active
    
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['NVDA', 'MSFT'])
        position_size = data.get('position_size', 10)
        clean_slate = data.get('clean_slate', False)
        
        if trading_active:
            return jsonify({'success': False, 'message': 'Trading already active'})
        
        # Clean slate if requested
        if clean_slate:
            success, stdout, stderr = run_subprocess_command([
                'python', 'src/brokers/ibkr_account_manager.py', '--reset'
            ])
            if not success:
                return jsonify({'success': False, 'message': 'Clean slate failed: ' + stderr})
        
        # Start REAL trading with Stairways V3 model
        cmd = ['bash', '-c', 'source venv/bin/activate && echo "yes" | python real_trading_deployment.py']
        trading_process = subprocess.Popen(cmd, cwd=str(project_root))
        trading_active = True
        
        return jsonify({
            'success': True,
            'message': f'Trading started with {len(symbols)} symbols, ${position_size} position size'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    global trading_process, trading_active
    
    try:
        if not trading_active or not trading_process:
            return jsonify({'success': False, 'message': 'Trading not active'})
        
        trading_process.terminate()
        try:
            trading_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            trading_process.kill()
            trading_process.wait()
        
        trading_active = False
        trading_process = None
        
        return jsonify({'success': True, 'message': 'Trading stopped successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reset_account', methods=['POST'])
def api_reset_account():
    try:
        success, stdout, stderr = run_subprocess_command([
            'python', 'src/brokers/ibkr_account_manager.py', '--reset'
        ])
        
        if success:
            return jsonify({'success': True, 'message': 'Account reset successful'})
        else:
            return jsonify({'success': False, 'message': 'Reset failed: ' + stderr})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/positions')
def api_positions():
    try:
        success, stdout, stderr = run_subprocess_command([
            'python', 'src/brokers/ibkr_account_manager.py', '--positions'
        ])
        return jsonify({'success': success, 'output': stdout if success else stderr})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

@app.route('/api/orders')
def api_orders():
    try:
        success, stdout, stderr = run_subprocess_command([
            'python', 'src/brokers/ibkr_account_manager.py', '--orders'
        ])
        return jsonify({'success': success, 'output': stdout if success else stderr})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

@app.route('/api/logs')
def api_logs():
    try:
        log_dir = Path('logs/production')
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-30:] if len(lines) > 30 else lines
                return jsonify({
                    'success': True, 
                    'output': ''.join(recent_lines),
                    'log_file': str(latest_log)
                })
        
        return jsonify({'success': False, 'output': 'No production logs found'})
        
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

if __name__ == '__main__':
    print("🌐 FIXED WEB TRADING LAUNCHER")
    print("=" * 40)
    print("🖥️  URL: http://localhost:5000")
    print("🛡️  Features: Clean slate, position monitoring, trading controls")
    print("⚡ Status: Starting web server...")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)