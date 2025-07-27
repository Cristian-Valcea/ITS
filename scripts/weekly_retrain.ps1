# weekly_retrain.ps1
# PowerShell script for automating weekly retraining and evaluation.
# Equivalent to weekly_retrain.sh for Windows environments.

# --- Configuration ---
# IMPORTANT: Set this to your project's root directory using an absolute path.
$ProjectRoot = "C:\path\to\your\rl_trading_platform" 
# Example: $ProjectRoot = "D:\projects\rl_trading_platform"

# Path to your Python executable (use absolute path, especially if in a venv)
$PythonExecutable = "C:\Python39\python.exe" 
# Example for venv: $PythonExecutable = "$($ProjectRoot)\.venv\Scripts\python.exe"

$MainPyScript = Join-Path -Path $ProjectRoot -ChildPath "src\main.py"
$MainConfigFile = Join-Path -Path $ProjectRoot -ChildPath "config\main_config.yaml"
$ModelParamsFile = Join-Path -Path $ProjectRoot -ChildPath "config\model_params.yaml"
$RiskLimitsFile = Join-Path -Path $ProjectRoot -ChildPath "config\risk_limits.yaml"

$LogDir = Join-Path -Path $ProjectRoot -ChildPath "logs"
$CronLogFile = Join-Path -Path $LogDir -ChildPath "powershell_retrain_schedule.log"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# --- Helper Function for Logging ---
function Log-Message {
    param (
        [string]$Message
    )
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "$Timestamp - $Message"
    Write-Output $LogEntry
    Add-Content -Path $CronLogFile -Value $LogEntry
}

Log-Message "--- Starting Weekly Retrain and Evaluation Script (PowerShell) ---"

# --- Environment Setup (Optional but Recommended) ---
# Activate virtual environment if you use one
# Example:
# $VenvActivateScript = Join-Path -Path $ProjectRoot -ChildPath ".venv\Scripts\Activate.ps1"
# if (Test-Path $VenvActivateScript) {
#     Log-Message "Activating virtual environment..."
#     try {
#         & $VenvActivateScript
#         Log-Message "Virtual environment activated."
#     } catch {
#         Log-Message "ERROR: Failed to activate virtual environment. Check path: $VenvActivateScript. Error: $($_.Exception.Message)"
#         # exit 1 # Decide if failure to activate venv is critical
#     }
# } else {
#     Log-Message "INFO: Virtual environment activation script not found at $VenvActivateScript. Proceeding with system Python."
# }


# Set PYTHONPATH if necessary, so Python can find your src modules
# Add the project root to PYTHONPATH environment variable for the current session
$CurrentPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ($CurrentPythonPath -notlike "*$($ProjectRoot)*") {
    $NewPythonPath = if ([string]::IsNullOrEmpty($CurrentPythonPath)) { $ProjectRoot } else { "$($CurrentPythonPath);$($ProjectRoot)" }
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $NewPythonPath, "Process")
    Log-Message "PYTHONPATH set for this session to: $NewPythonPath"
} else {
    Log-Message "Project root already in PYTHONPATH or PYTHONPATH not modified: $CurrentPythonPath"
}


# --- Define Parameters for this Run ---
$TargetSymbol = "SPY"       # Symbol to retrain for
$TrainInterval = "1hour"    # Data interval for training
$EvalInterval = "1hour"     # Data interval for evaluation

# Calculate dynamic date ranges for training and evaluation
$Today = (Get-Date).Date # Get today's date with time component set to 00:00:00

# Training data: from (Today - 2 years) to (Today - 1 day)
$TrainEndDateObj = $Today.AddDays(-1)
$TrainEndDate = $TrainEndDateObj.ToString("yyyy-MM-dd HH:mm:ss") # Format for IBKR if needed, or just date
$TrainStartDate = $TrainEndDateObj.AddYears(-2).ToString("yyyy-MM-dd")

# Evaluation data: from (Today) for the next N days (e.g., 30 days)
$EvalStartDate = $Today.ToString("yyyy-MM-dd")
$EvalEndDate = $Today.AddDays(29).ToString("yyyy-MM-dd HH:mm:ss") # Evaluate for 30 days

Log-Message "Run Parameters:"
Log-Message "  Symbol: $TargetSymbol"
Log-Message "  Training Interval: $TrainInterval"
Log-Message "  Training Start Date: $TrainStartDate"
Log-Message "  Training End Date: $TrainEndDate"
Log-Message "  Evaluation Interval: $EvalInterval"
Log-Message "  Evaluation Start Date: $EvalStartDate"
Log-Message "  Evaluation End Date: $EvalEndDate"

# --- 1. Run Training Pipeline ---
Log-Message "Starting training pipeline for $TargetSymbol..."
$TrainedModelOutputFile = Join-Path -Path $LogDir -ChildPath "last_trained_model_path_ps.txt" # PowerShell specific log
if (Test-Path $TrainedModelOutputFile) {
    Remove-Item $TrainedModelOutputFile -Force
}

# Construct arguments for main.py
$TrainArgs = @(
    "train",
    "--symbol", $TargetSymbol,
    "--start_date", $TrainStartDate,
    "--end_date", $TrainEndDate,
    "--interval", $TrainInterval,
    "--main_config", $MainConfigFile,
    "--model_params", $ModelParamsFile,
    "--risk_limits", $RiskLimitsFile
    # Add other args like --use_cached_data, --continue_from_model if needed
    # e.g., if $UseCachedData { $TrainArgs += "--use_cached_data" }
)

Log-Message "Executing Python for training: $PythonExecutable $MainPyScript $TrainArgs"
try {
    # Execute and capture all output (stdout and stderr)
    # To capture the model path, main.py should print it reliably.
    # For now, we just execute. Model path handling needs to be robust.
    & $PythonExecutable $MainPyScript $TrainArgs # *>&1 | Tee-Object -FilePath (Join-Path $LogDir "training_output.log")
    if ($LASTEXITCODE -ne 0) {
        Log-Message "ERROR: Training pipeline script returned non-zero exit code: $LASTEXITCODE."
        # exit 1 # Decide if to halt script
    } else {
        Log-Message "Training pipeline completed for $TargetSymbol."
    }
} catch {
    Log-Message "ERROR: Exception during training pipeline execution: $($_.Exception.Message)"
    # exit 1
}


# --- Model Path Handling (CRITICAL TODO - Same as bash script) ---
# The script needs a reliable way to get the path of the model trained above.
# Options:
# 1. Modify main.py to write the path to $TrainedModelOutputFile.
# 2. Parse console output (less reliable).
# 3. Assume a naming convention and find the latest model file (less reliable).
# For this skeleton, we'll acknowledge this gap.
$TrainedModelPath = $null
if (Test-Path $TrainedModelOutputFile) {
    $TrainedModelPath = Get-Content $TrainedModelOutputFile -Raw
    if ([string]::IsNullOrWhiteSpace($TrainedModelPath)) {
        Log-Message "ERROR: Trained model path is empty in output file '$TrainedModelOutputFile'."
        $TrainedModelPath = $null
    } else {
        Log-Message "Trained model path from output file: $TrainedModelPath"
    }
} else {
    Log-Message "WARNING: Trained model path output file '$TrainedModelOutputFile' not found."
    Log-Message "Evaluation will rely on default model path in EvaluatorAgent config or fail if none."
}


# --- 2. Run Evaluation Pipeline ---
Log-Message "Starting evaluation pipeline for $TargetSymbol..."
$EvalArgs = @(
    "evaluate",
    "--symbol", $TargetSymbol,
    "--start_date", $EvalStartDate,
    "--end_date", $EvalEndDate,
    "--interval", $EvalInterval,
    "--main_config", $MainConfigFile,
    "--model_params", $ModelParamsFile,
    "--risk_limits", $RiskLimitsFile
)
if (-not [string]::IsNullOrWhiteSpace($TrainedModelPath)) {
    $EvalArgs += "--model_path", $TrainedModelPath
} else {
    Log-Message "WARNING: No trained model path available from training step. Evaluation might use a default model."
}

Log-Message "Executing Python for evaluation: $PythonExecutable $MainPyScript $EvalArgs"
try {
    & $PythonExecutable $MainPyScript $EvalArgs
    if ($LASTEXITCODE -ne 0) {
        Log-Message "ERROR: Evaluation pipeline script returned non-zero exit code: $LASTEXITCODE."
    } else {
        Log-Message "Evaluation pipeline completed for $TargetSymbol."
    }
} catch {
    Log-Message "ERROR: Exception during evaluation pipeline execution: $($_.Exception.Message)"
}


# --- 3. Optional: Model Promotion Logic (Conceptual) ---
Log-Message "Model promotion logic (TODO): Implement checks and promotion if criteria met."
# Similar to bash script: parse report, check metrics, copy model file if good.
# $EvalReportDir = (Get-Item $MainConfigFile | Get-Content | ConvertFrom-Yaml).paths.reports_dir # Example
# $LatestReportFile = Get-ChildItem -Path (Join-Path $ProjectRoot $EvalReportDir) -Filter "${TargetSymbol}*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
# if ($LatestReportFile) {
#     Log-Message "Found evaluation report: $($LatestReportFile.FullName)"
#     # $ReportContent = Get-Content $LatestReportFile.FullName -Raw
#     # if ($ReportContent -match "Sharpe Ratio: (\d+\.?\d*)") { $Sharpe = [double]$Matches[1] }
#     # ... logic ...
# }


# --- Cleanup (Optional) ---
# if (Test-Path $TrainedModelOutputFile) { Remove-Item $TrainedModelOutputFile -Force }

Log-Message "--- Weekly Retrain and Evaluation Script (PowerShell) FINISHED ---"
"" | Add-Content -Path $CronLogFile # Add a blank line

# --- Example for Windows Task Scheduler ---
# To schedule this PowerShell script:
# 1. Open Task Scheduler.
# 2. Create a new Task.
# 3. Trigger: Set to weekly, choose day and time (e.g., every Sunday at 2:00 AM).
# 4. Action: "Start a program".
#    - Program/script: `powershell.exe` (or full path `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`)
#    - Add arguments (optional): `-ExecutionPolicy Bypass -File "C:\path\to\your\rl_trading_platform\scripts\weekly_retrain.ps1"`
#      (Replace path with the actual absolute path to this .ps1 script).
#      The `-ExecutionPolicy Bypass` might be needed if script execution is restricted.
#    - Start in (optional): `C:\path\to\your\rl_trading_platform\scripts\` (or where the script is located, so relative paths in script work as expected if any).
#      However, using absolute paths ($ProjectRoot) within the script is more robust.
# 5. Conditions/Settings: Configure as needed (e.g., run whether user is logged on or not, wake computer).
# 6. Ensure the user account running the task has permissions to:
#    - Execute PowerShell scripts.
#    - Read/write to the project directories (logs, models, data).
#    - Access Python and its dependencies.
#    - Access network if IBKR connection is live.
#
# Logging and Debugging Task Scheduler:
# - Task Scheduler has a history log for each task.
# - The script's own logging (`$CronLogFile`) is crucial.
# - Test the script thoroughly by running it manually in a PowerShell console before scheduling.
#   Ensure `$ProjectRoot` and `$PythonExecutable` are correctly set.
```

**Accompanying `AGENTS.md` (conceptual, if you want to put it in `scripts/AGENTS.md`, you can update the previous one or add this section):**

```markdown
## Regarding `weekly_retrain.ps1` (PowerShell)

This script is the Windows PowerShell equivalent of `weekly_retrain.sh`, designed for automating weekly retraining and evaluation.

**Key Features and Considerations:**
1.  **Configuration:**
    *   `$ProjectRoot`: **Must be set** to the absolute path of your `rl_trading_platform` directory.
    *   `$PythonExecutable`: **Must be set** to the absolute path of your Python interpreter (e.g., system Python or a virtual environment's Python).
2.  **Dynamic Date Calculation:** Uses PowerShell `Get-Date` cmdlets for calculating rolling date windows.
3.  **Pipeline Execution:** Calls `src/main.py` using the specified Python executable.
4.  **Model Path Handling (CRITICAL TODO):**
    *   Similar to the bash script, this PowerShell script needs a robust way to get the path of the newly trained model from the `train` step to pass to the `evaluate` step.
    *   **Recommendation:** Modify `src/main.py` (or `OrchestratorAgent`) in `train` mode to write the saved model's absolute path to a known temporary file (e.g., `logs/last_trained_model_path_ps.txt`). The script attempts to read this.
5.  **Logging:** Logs progress to `logs/powershell_retrain_schedule.log`.
6.  **Environment:**
    *   Includes a conceptual placeholder for activating a PowerShell-based Python virtual environment.
    *   Sets `PYTHONPATH` for the current PowerShell process to help Python find project modules.
7.  **Model Promotion:** Conceptual section for model promotion logic is included.
8.  **Windows Task Scheduler:** Comments at the end of the script provide guidance on how to schedule it using Windows Task Scheduler.

**Pre-requisites for running this script (especially via Task Scheduler):**
*   `$ProjectRoot` and `$PythonExecutable` variables must be correctly configured with absolute paths.
*   PowerShell execution policy might need adjustment (e.g., `Set-ExecutionPolicy RemoteSigned` or using `-ExecutionPolicy Bypass` when calling).
*   The user account running the scheduled task needs appropriate permissions (file system, network if applicable, Python execution).
*   All Python dependencies must be installed in the environment targeted by `$PythonExecutable`.
*   Test the script manually in a PowerShell console thoroughly before scheduling.
```
