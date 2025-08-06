#!/usr/bin/env python3
"""
Minimal working launcher - just the essentials
"""

import os
import subprocess
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Launcher</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f0f0f0; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        button { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-blue { background: #007bff; color: white; }
        .btn-green { background: #28a745; color: white; }
        .btn-red { background: #dc3545; color: white; }
        #output { background: #333; color: white; padding: 10px; border-radius: 5px; margin-top: 20px; height: 200px; overflow-y: scroll; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 IntradayJules Trading Launcher</h1>
        
        <button class="btn-blue" onclick="checkStatus()">📊 Check Status</button>
        <button class="btn-green" onclick="startTrading()">🚀 Start Trading</button>
        <button class="btn-red" onclick="resetAccount()">🧹 Reset Account</button>
        
        <div id="output">Click buttons to see output...</div>
    </div>

    <script>
        function log(message) {
            const output = document.getElementById('output');
            const time = new Date().toLocaleTimeString();
            output.innerHTML += '[' + time + '] ' + message + '\\n';
            output.scrollTop = output.scrollHeight;
        }
        
        async function checkStatus() {
            log('📊 Checking system status...');
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                log('✅ Status: ' + JSON.stringify(data, null, 2));
            } catch (error) {
                log('❌ Error: ' + error.message);
            }
        }
        
        async function startTrading() {
            log('🚀 Starting trading...');
            try {
                const response = await fetch('/api/start', { method: 'POST' });
                const data = await response.json();
                log(data.success ? '✅ ' + data.message : '❌ ' + data.message);
            } catch (error) {
                log('❌ Error: ' + error.message);
            }
        }
        
        async function resetAccount() {
            if (!confirm('Reset account?')) return;
            log('🧹 Resetting account...');
            try {
                const response = await fetch('/api/reset', { method: 'POST' });
                const data = await response.json();
                log(data.success ? '✅ ' + data.message : '❌ ' + data.message);
            } catch (error) {
                log('❌ Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/api/status')
def status():
    return jsonify({
        'server': 'running',
        'time': str(__import__('datetime').datetime.now()),
        'message': 'API is working'
    })

@app.route('/api/start', methods=['POST'])
def start():
    return jsonify({
        'success': True,
        'message': 'Trading start command received (demo mode)'
    })

@app.route('/api/reset', methods=['POST'])  
def reset():
    return jsonify({
        'success': True,
        'message': 'Account reset command received (demo mode)'
    })

if __name__ == '__main__':
    print("🌐 MINIMAL LAUNCHER STARTING")
    print("URL: http://localhost:9000")
    print("This is a minimal test version")
    print()
    
    try:
        app.run(host='127.0.0.1', port=9000, debug=False)
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative port...")
        app.run(host='127.0.0.1', port=9001, debug=False)