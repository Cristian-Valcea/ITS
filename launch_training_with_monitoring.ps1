# ========================================================================
# IntradayJules Training Launch with Full Monitoring Suite (PowerShell)
# ========================================================================
# This PowerShell script launches training and all monitoring tools in parallel
# Author: IntradayJules System
# Date: 2025-07-15
# ========================================================================

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🚀 IntradayJules Training Launch with Full Monitoring" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
Set-Location "C:\Projects\IntradayJules"

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
try {
    & "venv_fresh\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host "Please ensure venv_fresh exists and run: venv_fresh\Scripts\Activate.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create logs directory if it doesn't exist
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }
if (!(Test-Path "logs\tensorboard_gpu_fixed")) { New-Item -ItemType Directory -Path "logs\tensorboard_gpu_fixed" }

Write-Host "📁 Log directories prepared..." -ForegroundColor Yellow
Write-Host ""

# ========================================================================
# Function to start processes in new windows
# ========================================================================
function Start-MonitoringProcess {
    param(
        [string]$Title,
        [string]$Command,
        [string]$Description
    )
    
    Write-Host "🔧 Starting $Description..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $Command -WindowStyle Normal
    Start-Sleep -Seconds 1
}

# ========================================================================
# Launch TensorBoard (Port 6006)
# ========================================================================
Write-Host "📊 Starting TensorBoard on http://localhost:6006..." -ForegroundColor Green
Start-Process cmd -ArgumentList "/k", "cd /d C:\Projects\IntradayJules `&`& tensorboard --logdir logs\tensorboard_gpu_fixed --port 6006 --host localhost" -WindowStyle Normal
Start-Sleep -Seconds 3

# ========================================================================
# Launch Log File Monitor
# ========================================================================
Write-Host "📡 Starting real-time log monitor..." -ForegroundColor Green
Start-Process cmd -ArgumentList "/k", "cd /d C:\Projects\IntradayJules `&`& python monitor_live_logs.py" -WindowStyle Normal

# ========================================================================
# Launch Log Tail Monitor
# ========================================================================
Write-Host "📋 Starting log tail monitor..." -ForegroundColor Green
$logTailCommand = @"
Set-Location 'C:\Projects\IntradayJules'
Write-Host '📋 Log Tail Monitor - logs\orchestrator_gpu_fixed.log' -ForegroundColor Cyan
Write-Host '=' * 60 -ForegroundColor Gray
if (Test-Path 'logs\orchestrator_gpu_fixed.log') {
    Get-Content -Path 'logs\orchestrator_gpu_fixed.log' -Wait -Tail 50
} else {
    Write-Host '⏳ Waiting for log file to be created...' -ForegroundColor Yellow
    while (!(Test-Path 'logs\orchestrator_gpu_fixed.log')) {
        Start-Sleep -Seconds 1
    }
    Get-Content -Path 'logs\orchestrator_gpu_fixed.log' -Wait -Tail 50
}
"@
Start-MonitoringProcess -Title "Log-Tail" -Command $logTailCommand -Description "Log Tail Monitor"

# ========================================================================
# Launch API Server for Monitoring Endpoints
# ========================================================================
Write-Host "🌐 Starting API server on http://localhost:8000..." -ForegroundColor Green
Start-Process cmd -ArgumentList "/k", "cd /d C:\Projects\IntradayJules `&`& python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000" -WindowStyle Normal
Start-Sleep -Seconds 5

# ========================================================================
# Launch System Resource Monitor
# ========================================================================
Write-Host "💻 Starting system resource monitor..." -ForegroundColor Green
$resourceMonitorCommand = @"
Write-Host '💻 System Resource Monitor' -ForegroundColor Cyan
Write-Host '=' * 50 -ForegroundColor Gray
while (`$true) {
    `$cpu = Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average
    `$memory = Get-WmiObject -Class Win32_OperatingSystem
    `$memoryUsed = [math]::Round(((`$memory.TotalVisibleMemorySize - `$memory.FreePhysicalMemory) / `$memory.TotalVisibleMemorySize) * 100, 1)
    `$memoryFreeGB = [math]::Round(`$memory.FreePhysicalMemory / 1024 / 1024, 1)
    
    `$disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
    `$diskUsed = [math]::Round(((`$disk.Size - `$disk.FreeSpace) / `$disk.Size) * 100, 1)
    `$diskFreeGB = [math]::Round(`$disk.FreeSpace / 1GB, 1)
    
    Write-Host ("`r⚡ CPU: {0,5:F1}% | 🧠 RAM: {1,5:F1}% ({2}GB free) | 💾 Disk: {3,5:F1}% ({4}GB free)" -f `$cpu, `$memoryUsed, `$memoryFreeGB, `$diskUsed, `$diskFreeGB) -NoNewline
    Start-Sleep -Seconds 2
}
"@
Start-MonitoringProcess -Title "Resource-Monitor" -Command $resourceMonitorCommand -Description "System Resource Monitor"

# ========================================================================
# Launch Risk Audit Monitor
# ========================================================================
Write-Host "🛡️ Starting risk audit monitor..." -ForegroundColor Green
$riskMonitorCommand = @"
Set-Location 'C:\Projects\IntradayJules'
Write-Host '🛡️ Risk Audit Monitor' -ForegroundColor Cyan
Write-Host '=' * 50 -ForegroundColor Gray

`$riskLog = 'logs\risk_audit.jsonl'
if (!(Test-Path `$riskLog)) {
    Write-Host '⏳ Waiting for risk audit log to be created...' -ForegroundColor Yellow
    while (!(Test-Path `$riskLog)) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "📡 Monitoring: `$riskLog" -ForegroundColor Green
Write-Host '🔍 Watching for risk events...' -ForegroundColor Yellow
Write-Host ''

`$lastPosition = 0
if (Test-Path `$riskLog) {
    `$lastPosition = (Get-Item `$riskLog).Length
}

while (`$true) {
    try {
        if (Test-Path `$riskLog) {
            `$currentSize = (Get-Item `$riskLog).Length
            if (`$currentSize -gt `$lastPosition) {
                `$content = Get-Content `$riskLog -Raw
                `$newContent = `$content.Substring(`$lastPosition)
                `$lines = `$newContent -split "`n"
                
                foreach (`$line in `$lines) {
                    if (`$line.Trim()) {
                        try {
                            `$event = `$line | ConvertFrom-Json
                            `$timestamp = `$event.timestamp
                            `$eventType = `$event.event_type
                            Write-Host "🚨 [`$timestamp] `$eventType`: `$line" -ForegroundColor Red
                        } catch {
                            Write-Host "📝 `$line" -ForegroundColor White
                        }
                    }
                }
                `$lastPosition = `$currentSize
            }
        }
        Start-Sleep -Seconds 1
    } catch {
        Write-Host "❌ Error: `$_" -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
}
"@
Start-MonitoringProcess -Title "Risk-Audit" -Command $riskMonitorCommand -Description "Risk Audit Monitor"

# ========================================================================
# Launch Post-Training Visualizer
# ========================================================================
Write-Host "🎨 Starting post-training visualization monitor..." -ForegroundColor Green
$postTrainingCommand = @"
Set-Location 'C:\Projects\IntradayJules'
Write-Host '🎨 Post-Training Visualizer' -ForegroundColor Cyan
Write-Host '=' * 50 -ForegroundColor Gray
Write-Host '⏳ Waiting for training to complete...' -ForegroundColor Yellow
python post_training_visualizer.py
"@
Start-MonitoringProcess -Title "Post-Training-Visualizer" -Command $postTrainingCommand -Description "Post-Training Visualizer"

# Launch Training Progress Monitor
# ========================================================================
Write-Host "📈 Starting training progress monitor..." -ForegroundColor Green
$progressMonitorCommand = @"
Set-Location 'C:\Projects\IntradayJules'
Write-Host '📈 Training Progress Monitor' -ForegroundColor Cyan
Write-Host '=' * 50 -ForegroundColor Gray

`$logFile = 'logs\orchestrator_gpu_fixed.log'
if (!(Test-Path `$logFile)) {
    Write-Host '⏳ Waiting for training log to be created...' -ForegroundColor Yellow
    while (!(Test-Path `$logFile)) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "📡 Monitoring: `$logFile" -ForegroundColor Green
Write-Host '🔍 Watching for training progress...' -ForegroundColor Yellow
Write-Host ''

`$lastPosition = 0
`$episodeCount = 0
`$lastReward = 0.0

while (`$true) {
    try {
        if (Test-Path `$logFile) {
            `$currentSize = (Get-Item `$logFile).Length
            if (`$currentSize -gt `$lastPosition) {
                `$content = Get-Content `$logFile -Raw
                `$newContent = `$content.Substring(`$lastPosition)
                `$lines = `$newContent -split "`n"
                
                foreach (`$line in `$lines) {
                    if (`$line -match 'Episode.*reward') {
                        if (`$line -match 'Episode (\d+)') {
                            `$episodeCount = [int]`$matches[1]
                        }
                        if (`$line -match 'reward[:\s]+([+-]?\d*\.?\d+)') {
                            `$lastReward = [double]`$matches[1]
                        }
                        Write-Host "🎯 Episode `$episodeCount`: Reward = `$(`$lastReward.ToString('F4'))" -ForegroundColor Green
                    }
                    elseif (`$line -match 'KYLE LAMBDA IMPACT') {
                        Write-Host "💰 `$line" -ForegroundColor Magenta
                    }
                    elseif (`$line -match 'turnover|drawdown|risk|violation') {
                        Write-Host "🛡️ `$line" -ForegroundColor Yellow
                    }
                    elseif (`$line -match 'ERROR') {
                        Write-Host "❌ `$line" -ForegroundColor Red
                    }
                    elseif (`$line -match 'WARNING') {
                        Write-Host "⚠️ `$line" -ForegroundColor Yellow
                    }
                }
                `$lastPosition = `$currentSize
            }
        }
        Start-Sleep -Seconds 1
    } catch {
        Write-Host "❌ Error: `$_" -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
}
"@
Start-MonitoringProcess -Title "Training-Progress" -Command $progressMonitorCommand -Description "Training Progress Monitor"

# ========================================================================
# Wait and then launch the main training
# ========================================================================
Write-Host ""
Write-Host "⏳ Waiting 10 seconds for all monitoring tools to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🎯 Starting Main Training Process" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "📊 Command: cd c:/Projects/IntradayJules/src `&`& python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31" -ForegroundColor White
Write-Host ""

# Launch the main training in a new window
Start-Process cmd -ArgumentList "/k", "cd /d C:\Projects\IntradayJules\src `&`& python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31" -WindowStyle Normal

# ========================================================================
# Show monitoring dashboard URLs
# ========================================================================
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "📊 MONITORING DASHBOARD URLS" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🔗 TensorBoard:     http://localhost:6006" -ForegroundColor Blue
Write-Host "🔗 API Monitoring:  http://localhost:8000/docs" -ForegroundColor Blue
Write-Host "🔗 System Status:   http://localhost:8000/api/v1/status" -ForegroundColor Blue
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# ========================================================================
# Control Panel
# ========================================================================
Write-Host "🎮 CONTROL PANEL" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "This window serves as your control panel." -ForegroundColor White
Write-Host ""
Write-Host "📋 Monitoring Windows Launched:" -ForegroundColor Yellow
Write-Host "  - TensorBoard (Port 6006)" -ForegroundColor White
Write-Host "  - Log Monitor" -ForegroundColor White
Write-Host "  - Log Tail" -ForegroundColor White
Write-Host "  - API Server (Port 8000)" -ForegroundColor White
Write-Host "  - Resource Monitor" -ForegroundColor White
Write-Host "  - Risk Audit Monitor" -ForegroundColor White
Write-Host "  - Post-Training Visualizer (waits for completion)" -ForegroundColor White
Write-Host "  - Training Progress Monitor" -ForegroundColor White
Write-Host "  - Main Training Process" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "  - Check TensorBoard at http://localhost:6006 for training metrics" -ForegroundColor White
Write-Host "  - Monitor the various windows for real-time updates" -ForegroundColor White
Write-Host "  - Use Ctrl+C in any window to stop that specific monitor" -ForegroundColor White
Write-Host "  - Close this window to keep everything running" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Training is now running with full monitoring suite!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan

# Keep the control panel open
Read-Host "Press Enter to close this control panel (monitoring will continue running)"