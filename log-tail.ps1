#!/usr/bin/env pwsh
# Log tail script for IntradayJules training logs

param(
    [string]$LogFile = "logs\orchestrator_gpu_fixed.log",
    [int]$Lines = 20
)

$LogPath = Join-Path $PSScriptRoot $LogFile

Write-Host "🔍 Monitoring log file: $LogPath" -ForegroundColor Cyan
Write-Host "📝 Showing last $Lines lines, then following new entries..." -ForegroundColor Green
Write-Host "⏹️  Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
Write-Host ""

# Check if log file exists
if (-not (Test-Path $LogPath)) {
    Write-Host "⚠️  Log file not found: $LogPath" -ForegroundColor Yellow
    Write-Host "💡 The file will be created when training starts with file logging enabled." -ForegroundColor Cyan
    Write-Host "🔄 Waiting for log file to be created..." -ForegroundColor Magenta
    
    # Wait for file to be created
    while (-not (Test-Path $LogPath)) {
        Start-Sleep -Seconds 1
    }
    Write-Host "✅ Log file created! Starting to monitor..." -ForegroundColor Green
}

# Start tailing the log file
try {
    Get-Content -Path $LogPath -Wait -Tail $Lines -ErrorAction Stop
} catch {
    Write-Host "❌ Error reading log file: $_" -ForegroundColor Red
    Write-Host "💡 Make sure the training process is running and file logging is enabled." -ForegroundColor Cyan
}