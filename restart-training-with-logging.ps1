#!/usr/bin/env pwsh
# Script to restart training with proper file logging

param(
    [string]$Symbol = "NVDA",
    [string]$StartDate = "2024-01-01", 
    [string]$EndDate = "2024-01-31",
    [string]$Config = "config/main_config_orchestrator_gpu_fixed.yaml"
)

Write-Host "🔄 Restarting training with file logging enabled..." -ForegroundColor Cyan
Write-Host "📊 Symbol: $Symbol" -ForegroundColor Green
Write-Host "📅 Date range: $StartDate to $EndDate" -ForegroundColor Green
Write-Host "⚙️  Config: $Config" -ForegroundColor Green
Write-Host ""

# Kill existing training processes
Write-Host "🛑 Stopping existing training processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 100MB} | ForEach-Object {
    Write-Host "   Stopping process ID: $($_.Id)" -ForegroundColor Red
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

# Clear the log file
$LogFile = "logs\orchestrator_gpu_fixed.log"
Write-Host "🧹 Clearing log file: $LogFile" -ForegroundColor Magenta
Clear-Content $LogFile -ErrorAction SilentlyContinue

# Start new training process
Write-Host "🚀 Starting new training process with file logging..." -ForegroundColor Green
Set-Location src
$TrainingArgs = @(
    "main.py",
    "train", 
    "--main_config", "../$Config",
    "--symbol", $Symbol,
    "--start_date", $StartDate,
    "--end_date", $EndDate
)

Write-Host "💻 Command: python $($TrainingArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Logs will be written to: $LogFile" -ForegroundColor Green
Write-Host "🔍 Use .\log-tail.ps1 in another terminal to monitor logs" -ForegroundColor Yellow
Write-Host ""

# Start the training process
& "..\venv\Scripts\python.exe" @TrainingArgs