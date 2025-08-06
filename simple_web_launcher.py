#!/usr/bin/env python3
"""
🌐 Simple Web Trading Launcher
Working web interface on port 8080
"""

import os
import sys
import subprocess
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
trading_process = None
trading_active = False

def run_command(cmd):
    """Run command safely"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root), timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 IntradayJules Trading Launcher</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .status { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .controls { background: #e8f4f8; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .btn { padding: 12px 24px; margin: 5px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .btn:hover { opacity: 0.8; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .checkbox { margin: 10px 0; }
        .input-group { margin: 10px 0; }
        .input-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .input-group input { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100px; }
        .message { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .loading { display: none; }
        .status-item { margin: 5px 0; padding: 8px; background: white; border-radius: 4px; }
        .logs { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 12px; height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IntradayJules Trading Launcher</h1>
            <p>Real Stairways V3 Model + Enhanced IBKR Safety</p>
        </div>
        
        <div class="status">
            <h3>📊 System Status</h3>
            <div id="status-container">
                <div class="status-item">Click "Refresh Status" to check system</div>
            </div>
            <button class="btn btn-primary" onclick="refreshStatus()">🔄 Refresh Status</button>
        </div>
        
        <div class="controls">
            <h3>🎯 Trading Controls</h3>
            
            <div class="checkbox">
                <input type="checkbox" id="nvda" checked> <label for="nvda">NVDA</label>
                <input type="checkbox" id="msft" checked> <label for="msft">MSFT</label>
            </div>
            
            <div class="input-group">
                <label for="position-size">Position Size ($)</label>
                <input type="number" id="position-size" value="10" min="1" max="100">
            </div>
            
            <div class="checkbox">
                <input type="checkbox" id="clean-slate" checked>
                <label for="clean-slate">🧹 Clean Slate (Reset account before start)</label>
            </div>
            
            <div id="message-container"></div>
            
            <div style="margin-top: 15px;">
                <button class="btn btn-success" id="start-btn" onclick="startTrading()">
                    <span id="start-loading" class="loading">⏳</span>
                    🚀 Start Real Trading
                </button>
                
                <button class="btn btn-danger" id="stop-btn" onclick="stopTrading()" disabled>
                    🛑 Stop Trading
                </button>
                
                <button class="btn btn-warning" onclick="resetAccount()">
                    <span id="reset-loading" class="loading">⏳</span>
                    🧹 Reset Account
                </button>
                
                <button class="btn btn-primary" onclick="showPositions()">
                    📊 Show Positions
                </button>
            </div>
        </div>
        
        <div class="status">
            <h3>📜 System Output</h3>
            <div id="output" class="logs">Click buttons above to see system output...</div>
        </div>
    </div>

    <script>
        async function refreshStatus() {
            showOutput('🔄 Checking system status...');
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('status-container').innerHTML = 
                    '<div class="status-item">IBKR: ' + data.ibkr + '</div>' +
                    '<div class="status-item">Risk Governor: ' + data.governor + '</div>' +
                    '<div class="status-item">Model: ' + data.model + '</div>' +
                    '<div class="status-item">Trading: ' + (data.trading_active ? 'Active' : 'Inactive') + '</div>';
                
                // Update button states
                document.getElementById('start-btn').disabled = data.trading_active;
                document.getElementById('stop-btn').disabled = !data.trading_active;
                
                showOutput('✅ Status updated: ' + JSON.stringify(data, null, 2));
            } catch (error) {
                showMessage('Status check failed: ' + error, 'error');
            }
        }
        
        async function startTrading() {
            const startBtn = document.getElementById('start-btn');
            const loading = document.getElementById('start-loading');
            
            startBtn.disabled = true;
            loading.style.display = 'inline';
            
            try {
                const symbols = [];
                if (document.getElementById('nvda').checked) symbols.push('NVDA');
                if (document.getElementById('msft').checked) symbols.push('MSFT');
                
                if (symbols.length === 0) {
                    throw new Error('Please select at least one symbol');
                }
                
                const positionSize = parseInt(document.getElementById('position-size').value);
                const cleanSlate = document.getElementById('clean-slate').checked;
                
                showOutput('🚀 Starting trading with: ' + symbols.join(', ') + ', $' + positionSize + ', clean slate: ' + cleanSlate);
                
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbols, position_size: positionSize, clean_slate: cleanSlate })
                });
                
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
                
            } catch (error) {
                showMessage('Start failed: ' + error, 'error');
            } finally {
                loading.style.display = 'none';
                startBtn.disabled = false;
            }
        }
        
        async function stopTrading() {
            try {
                showOutput('🛑 Stopping trading...');
                const response = await fetch('/api/stop', { method: 'POST' });
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
            } catch (error) {
                showMessage('Stop failed: ' + error, 'error');
            }
        }
        
        async function resetAccount() {
            if (!confirm('Reset account? This will cancel all orders and flatten positions.')) return;
            
            const loading = document.getElementById('reset-loading');
            loading.style.display = 'inline';
            
            try {
                showOutput('🧹 Resetting account...');
                const response = await fetch('/api/reset', { method: 'POST' });
                const result = await response.json();
                showMessage(result.message, result.success ? 'success' : 'error');
                showOutput(result.output || 'Reset completed');
                
                if (result.success) {
                    setTimeout(refreshStatus, 2000);
                }
            } catch (error) {
                showMessage('Reset failed: ' + error, 'error');
            } finally {
                loading.style.display = 'none';
            }
        }
        
        async function showPositions() {
            try {
                showOutput('📊 Loading positions...');
                const response = await fetch('/api/positions');
                const result = await response.json();
                showOutput(result.output || 'No positions data');
            } catch (error) {
                showOutput('❌ Error loading positions: ' + error);
            }
        }
        
        function showMessage(message, type) {
            const container = document.getElementById('message-container');
            container.innerHTML = '<div class="message ' + type + '">' + message + '</div>';
            setTimeout(() => container.innerHTML = '', 5000);
        }
        
        function showOutput(text) {
            const output = document.getElementById('output');
            output.innerHTML = text;
            output.scrollTop = output.scrollHeight;
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshStatus, 30000);
        
        // Initial load
        setTimeout(refreshStatus, 1000);
    </script>
</body>
</html>
    ''')

@app.route('/api/status')
def api_status():
    global trading_active
    
    # Check IBKR
    success, stdout, stderr = run_command(['python', 'src/brokers/ibkr_account_manager.py', '--positions'])
    ibkr_status = '✅ Connected' if success and 'Connected' in stdout else '❌ Failed'
    
    # Check Governor
    success, stdout, stderr = run_command(['python', 'operator_docs/governor_state_manager.py', '--status'])
    if success and 'RUNNING' in stdout:
        governor_status = '✅ RUNNING'
    elif success and 'PAUSED' in stdout:
        governor_status = '⚠️ PAUSED'
    else:
        governor_status = '❌ Error'
    
    # Check Model
    model_paths = [
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
    ]
    
    model_status = '❌ Not found'
    for model_path in model_paths:
        if os.path.exists(model_path):
            model_status = f'✅ {os.path.basename(model_path)}'
            break
    
    return jsonify({
        'ibkr': ibkr_status,
        'governor': governor_status,
        'model': model_status,
        'trading_active': trading_active
    })

@app.route('/api/start', methods=['POST'])
def api_start():
    global trading_process, trading_active
    
    try:
        data = request.get_json()
        clean_slate = data.get('clean_slate', False)
        
        if trading_active:
            return jsonify({'success': False, 'message': 'Trading already active'})
        
        # Clean slate if requested
        if clean_slate:
            success, stdout, stderr = run_command(['python', 'src/brokers/ibkr_account_manager.py', '--reset'])
            if not success:
                return jsonify({'success': False, 'message': 'Clean slate failed: ' + stderr})
        
        # Start real trading
        cmd = ['bash', '-c', 'source venv/bin/activate && echo "yes" | python real_trading_deployment.py']
        trading_process = subprocess.Popen(cmd, cwd=str(project_root))
        trading_active = True
        
        return jsonify({'success': True, 'message': 'Real Stairways V3 trading started!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global trading_process, trading_active
    
    try:
        if not trading_active or not trading_process:
            return jsonify({'success': False, 'message': 'Trading not active'})
        
        trading_process.terminate()
        try:
            trading_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            trading_process.kill()
        
        trading_active = False
        trading_process = None
        
        return jsonify({'success': True, 'message': 'Trading stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    try:
        success, stdout, stderr = run_command(['python', 'src/brokers/ibkr_account_manager.py', '--reset'])
        return jsonify({
            'success': success,
            'message': 'Account reset ' + ('successful' if success else 'failed'),
            'output': stdout if success else stderr
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'output': ''})

@app.route('/api/positions')
def api_positions():
    try:
        success, stdout, stderr = run_command(['python', 'src/brokers/ibkr_account_manager.py', '--positions'])
        return jsonify({'success': success, 'output': stdout if success else stderr})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

if __name__ == '__main__':
    print("🌐 SIMPLE WEB TRADING LAUNCHER")
    print("=" * 40)
    print("🖥️  URL: http://localhost:8080")
    print("🤖 Features: Real Stairways V3 trading")
    print("🧹 Clean slate functionality")
    print("📊 Position monitoring")
    print("=" * 40)
    print("Starting web server...")
    print()
    
    app.run(host='127.0.0.1', port=8080, debug=False)