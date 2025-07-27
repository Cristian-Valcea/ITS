# PowerShell script to activate the virtual environment
# Usage: .\activate_venv.ps1

Write-Host "🚀 Activating Python virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host "📦 Python version: " -NoNewline -ForegroundColor Cyan
python --version
Write-Host "📍 Current directory: " -NoNewline -ForegroundColor Cyan
Get-Location

Write-Host "`n🔧 Available commands:" -ForegroundColor Yellow
Write-Host "  python src/main.py --help                    # Show help" -ForegroundColor White
Write-Host "  python src/main.py train --symbol AAPL ...   # Train model" -ForegroundColor White
Write-Host "  python src/main.py evaluate --symbol AAPL... # Evaluate model" -ForegroundColor White
Write-Host "  deactivate                                   # Exit virtual environment" -ForegroundColor White
Write-Host ""