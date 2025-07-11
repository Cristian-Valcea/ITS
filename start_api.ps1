# Start the IntradayJules API Server
# Usage: .\start_api.ps1

Write-Host "Starting IntradayJules API Server..." -ForegroundColor Green
Write-Host "Web Interface will be available at: http://127.0.0.1:8000/ui/dashboard" -ForegroundColor Cyan
Write-Host "NVDA DQN Training at: http://127.0.0.1:8000/ui/nvda-dqn" -ForegroundColor Cyan
Write-Host "API Documentation at: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Start the API server
Write-Host "Starting FastAPI server..." -ForegroundColor Yellow
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

Write-Host ""
Write-Host "API server stopped!" -ForegroundColor Green