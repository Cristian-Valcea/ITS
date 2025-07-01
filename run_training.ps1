# Quick training script for the Intraday Trading System
# Usage: .\run_training.ps1 [SYMBOL] [START_DATE] [END_DATE] [INTERVAL]
# Example: .\run_training.ps1 AAPL 2023-01-01 2023-01-10 1min

param(
    [string]$Symbol = "AAPL",
    [string]$StartDate = "2023-01-01", 
    [string]$EndDate = "2023-01-03",
    [string]$Interval = "1min"
)

Write-Host "🚀 Starting Intraday Trading System Training..." -ForegroundColor Green
Write-Host "📊 Symbol: $Symbol" -ForegroundColor Cyan
Write-Host "📅 Period: $StartDate to $EndDate" -ForegroundColor Cyan  
Write-Host "⏱️  Interval: $Interval" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Run training
Write-Host "🎯 Starting training pipeline..." -ForegroundColor Yellow
& .\venv\Scripts\python.exe src/main.py train `
  --symbol $Symbol `
  --start_date $StartDate `
  --end_date $EndDate `
  --interval $Interval `
  --main_config config/main_config.yaml `
  --model_params config/model_params.yaml `
  --risk_limits config/risk_limits.yaml

Write-Host ""
Write-Host "✅ Training script completed!" -ForegroundColor Green