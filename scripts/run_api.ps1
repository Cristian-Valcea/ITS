# scripts/run_api.ps1
# Launches the FastAPI application using Uvicorn for local development in PowerShell.

# --- Configuration ---
# Ensure this script is run from the project root directory (rl_trading_platform)
# Or adjust paths accordingly.
$ProjectRoot = $PSScriptRoot | Split-Path # Gets the directory of the current script, then its parent
# If scripts/ is directly under project root, this should be okay.
# If not, set $ProjectRoot manually to the absolute path of your project.
# e.g., $ProjectRoot = "C:\Users\YourUser\Projects\rl_trading_platform"

# Default host and port
$Env:API_HOST = if ($Env:API_HOST) { $Env:API_HOST } else { "0.0.0.0" }
$Env:API_PORT = if ($Env:API_PORT) { $Env:API_PORT } else { "8000" }
$Env:API_LOG_LEVEL = if ($Env:API_LOG_LEVEL) { $Env:API_LOG_LEVEL } else { "info" }
# $Env:API_WORKERS = if ($Env:API_WORKERS) { $Env:API_WORKERS } else { "1" } # Uvicorn --reload works best with 1 worker

Write-Host "Project Root detected/set to: $ProjectRoot"

# --- Environment Setup (Optional but Recommended) ---
# Activate virtual environment if you use one
$VenvPath = Join-Path -Path $ProjectRoot -ChildPath "venv" # Assuming venv is in project root
$VenvActivateScript = Join-Path -Path $VenvPath -ChildPath "Scripts\Activate.ps1"

if (Test-Path $VenvActivateScript) {
    Write-Host "Activating virtual environment from $VenvActivateScript..."
    try {
        & $VenvActivateScript
        Write-Host "Virtual environment activated."
    } catch {
        Write-Warning "ERROR: Failed to activate virtual environment. Please ensure it's created and dependencies are installed."
        Write-Warning "Error details: $($_.Exception.Message)"
        # exit 1 # Optionally exit
    }
} else {
    Write-Host "INFO: No virtual environment activation script found at $VenvActivateScript. Using system Python interpreter."
}

# Set PYTHONPATH to include the project root, so 'src' can be imported
# For PowerShell, $env:PYTHONPATH is used.
$CurrentPythonPath = $env:PYTHONPATH
if ($CurrentPythonPath -notlike "*$($ProjectRoot)*") {
    if ([string]::IsNullOrEmpty($CurrentPythonPath)) {
        $env:PYTHONPATH = $ProjectRoot
    } else {
        $env:PYTHONPATH = "$($CurrentPythonPath);$($ProjectRoot)"
    }
    Write-Host "PYTHONPATH set for this session to: $($env:PYTHONPATH)"
} else {
    Write-Host "Project root likely already in PYTHONPATH or PYTHONPATH not modified: $CurrentPythonPath"
}
Write-Host "Current working directory: $(Get-Location)"


Write-Host "Starting FastAPI application with Uvicorn..."
Write-Host "Access at: http://$($Env:API_HOST):$($Env:API_PORT)/docs (Swagger UI) or http://$($Env:API_HOST):$($Env:API_PORT)/redoc (ReDoc)"
Write-Host "API Status: http://$($Env:API_HOST):$($Env:API_PORT}/api/v1/status"

# Note: Uvicorn is a Python module. Ensure it's callable.
# If Python executable is not directly in PATH but venv is activated, 'python -m uvicorn ...' is safer.
# If Python executable path is known ($PythonExecutable from bash script example), use that.
# For simplicity, assuming 'uvicorn' is available in PATH after venv activation or system install.

# It's often more robust to call uvicorn via `python -m uvicorn`
# $PythonExe = "python" # Or use $PythonExecutable if defined earlier
# & $PythonExe -m uvicorn src.api.main:app --reload --host $Env:API_HOST --port $Env:API_PORT --log-level $Env:API_LOG_LEVEL

uvicorn src.api.main:app --reload --host "$($Env:API_HOST)" --port "$($Env:API_PORT)" --log-level "$($Env:API_LOG_LEVEL)"

# To run without reload:
# uvicorn src.api.main:app --host "$($Env:API_HOST)" --port "$($Env:API_PORT)" --log-level "$($Env:API_LOG_LEVEL)"

Write-Host "FastAPI application stopped."
```
