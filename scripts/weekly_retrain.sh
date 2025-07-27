#!/bin/bash

# weekly_retrain.sh
# Example script for automating weekly retraining and evaluation.
# This script would be triggered by a cron job.

# --- Configuration ---
# Ensure these paths are correct relative to where the cron job executes,
# or use absolute paths.
PROJECT_ROOT="/path/to/your/rl_trading_platform" # IMPORTANT: Set this to your project's root directory
PYTHON_EXECUTABLE="/usr/bin/python3" # Or path to your virtual environment's python

MAIN_PY_SCRIPT="${PROJECT_ROOT}/src/main.py"
MAIN_CONFIG_FILE="${PROJECT_ROOT}/config/main_config.yaml" # Default main config
MODEL_PARAMS_FILE="${PROJECT_ROOT}/config/model_params.yaml" # Default model params
RISK_LIMITS_FILE="${PROJECT_ROOT}/config/risk_limits.yaml" # Default risk limits

LOG_DIR="${PROJECT_ROOT}/logs"
CRON_LOG_FILE="${LOG_DIR}/cron_retrain_schedule.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# --- Helper Function for Logging ---
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$CRON_LOG_FILE"
}

log_message "--- Starting Weekly Retrain and Evaluation Script ---"

# --- Environment Setup (Optional but Recommended) ---
# Activate virtual environment if you use one
# Example:
# source "${PROJECT_ROOT}/.venv/bin/activate"
# if [ $? -ne 0 ]; then
#     log_message "ERROR: Failed to activate virtual environment. Exiting."
#     exit 1
# fi
# log_message "Virtual environment activated."

# Set PYTHONPATH if necessary, so Python can find your src modules
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
log_message "PYTHONPATH set to: $PYTHONPATH"


# --- Define Parameters for this Run ---
# These might be dynamic based on the date or fetched from a config.
# For this example, let's use placeholders or simple date calculations.

TARGET_SYMBOL="SPY" # Symbol to retrain for
TRAIN_INTERVAL="1hour"  # Data interval for training
EVAL_INTERVAL="1hour"   # Data interval for evaluation

# Calculate dynamic date ranges for training and evaluation
# Example: Train on the last 2 years of data, evaluate on the next 1 month (simulated if future data isn't live)
# This requires `date` command that supports -d option (GNU date).
# Adjust for macOS `date` if needed (install gdate or use different syntax).

# Training data: from (Today - 2 years) to (Today - 1 day)
TRAIN_END_DATE=$(date -d "yesterday" '+%Y-%m-%d %H:%M:%S') # End training data yesterday
TRAIN_START_DATE=$(date -d "yesterday - 2 years" '+%Y-%m-%d')

# Evaluation data: from (Today) for the next N days (e.g., 30 days)
# This is tricky for true out-of-sample if model is for future.
# For backtesting-style walk-forward, eval usually follows train period.
# Let's assume eval period is immediately after training period for this example.
EVAL_START_DATE=$(date -d "today" '+%Y-%m-%d') # Start eval data today
EVAL_END_DATE=$(date -d "today + 29 days" '+%Y-%m-%d %H:%M:%S') # Evaluate for 30 days

log_message "Run Parameters:"
log_message "  Symbol: $TARGET_SYMBOL"
log_message "  Training Interval: $TRAIN_INTERVAL"
log_message "  Training Start Date: $TRAIN_START_DATE"
log_message "  Training End Date: $TRAIN_END_DATE"
log_message "  Evaluation Interval: $EVAL_INTERVAL"
log_message "  Evaluation Start Date: $EVAL_START_DATE"
log_message "  Evaluation End Date: $EVAL_END_DATE"

# --- 1. Run Training Pipeline ---
log_message "Starting training pipeline for $TARGET_SYMBOL..."
# Using default config files specified above. Add --use_cached_data if appropriate.
# The output of training (trained_model_path) needs to be captured if possible,
# or main.py should log it clearly for the next step to find.
# For simplicity, we assume main.py saves models to a predictable location or
# the orchestrator manages model versions.

# We need to capture the path of the trained model.
# One way: main.py prints the path to stdout on the last line if successful.
# Or, modify main.py to write the path to a temporary file.

# Let's assume main.py prints the model path.
TRAINED_MODEL_OUTPUT_FILE="${LOG_DIR}/last_trained_model_path.txt"
rm -f "$TRAINED_MODEL_OUTPUT_FILE" # Clear previous run's output

"$PYTHON_EXECUTABLE" "$MAIN_PY_SCRIPT" train \
    --symbol "$TARGET_SYMBOL" \
    --start_date "$TRAIN_START_DATE" \
    --end_date "$TRAIN_END_DATE" \
    --interval "$TRAIN_INTERVAL" \
    --main_config "$MAIN_CONFIG_FILE" \
    --model_params "$MODEL_PARAMS_FILE" \
    --risk_limits "$RISK_LIMITS_FILE" \
    # Add other relevant args like --use_cached_data, --continue_from_model if needed
    # Example: If Orchestrator's run_training_pipeline prints the path:
    # > "$TRAINED_MODEL_OUTPUT_FILE" # This redirects stdout. Check if main.py prints ONLY the path.
    # A safer way is for main.py to handle this communication.
    # For now, we'll assume the latest model can be found by naming convention or main_config.paths.model_save_dir

# For this script, let's assume the TrainerAgent saves the model with a timestamp,
# and we need a way to find the LATEST model for the symbol.
# This is a simplification; a robust system would have better model versioning.
# For now, let's assume we proceed and Evaluator will pick up the "best" or "latest" model.
# Or, if main.py is modified to output the path:
# CAPTURED_OUTPUT=$("$PYTHON_EXECUTABLE" "$MAIN_PY_SCRIPT" train ... )
# TRAINED_MODEL_PATH=$(echo "$CAPTURED_OUTPUT" | grep "Training pipeline completed. Model saved to:" | awk -F': ' '{print $2}')

# Let's assume the training script will create a model, and the evaluation script
# will need to be pointed to it. For this example, we'll manually construct a potential path
# or assume the evaluator can find the latest.
# A better approach: TrainerAgent writes the path of the saved model to a known temp file.
# Then this script reads it.
# For now, we'll just log the training command. The actual model path handling needs refinement.

if [ $? -ne 0 ]; then
    log_message "ERROR: Training pipeline failed for $TARGET_SYMBOL. Exiting."
    exit 1
fi
log_message "Training pipeline completed for $TARGET_SYMBOL."

# Read the model path if main.py was modified to output it to TRAINED_MODEL_OUTPUT_FILE
# if [ -f "$TRAINED_MODEL_OUTPUT_FILE" ]; then
#     TRAINED_MODEL_PATH=$(cat "$TRAINED_MODEL_OUTPUT_FILE")
#     log_message "Trained model path from output file: $TRAINED_MODEL_PATH"
#     if [ -z "$TRAINED_MODEL_PATH" ]; then
#         log_message "ERROR: Trained model path is empty in output file. Cannot proceed with evaluation."
#         exit 1
#     fi
# else
#     log_message "ERROR: Trained model path output file not found. Cannot proceed with evaluation."
#     log_message "Please ensure main.py (train mode) outputs the saved model path to $TRAINED_MODEL_OUTPUT_FILE."
#     # As a fallback, try to find the latest model in the directory (very brittle)
#     # MODEL_DIR=$(grep 'model_save_dir:' "$MAIN_CONFIG_FILE" | awk '{print $2}' | tr -d '"') # Get from config
#     # TRAINED_MODEL_PATH=$(ls -t "${PROJECT_ROOT}/${MODEL_DIR}"/*.zip | head -1) # Get latest .zip
#     # This is highly dependent on main_config.yaml structure and relative paths.
#     # For now, this script will require manual setting or a more robust discovery.
#     log_message "Evaluation will proceed assuming EvaluatorAgent can find the model or a default is set."
#     # For the purpose of this skeleton, we'll need to provide a placeholder or make an assumption.
#     # Let's assume the user has to manually update the model_path for evaluation or
#     # the evaluator is configured to pick the latest.
#     # This is a key part to make robust in a production system.
#     # For this example, let's assume the model path needs to be known.
#     # We will skip direct use of TRAINED_MODEL_PATH in the eval command for now,
#     # assuming the evaluator has a default or can find it.
#     # If main.py's orchestrator prints "Model saved to: /path/to/model.zip", we can grep that.
#     # This is too complex for a simple cron script without modifying main.py output.
# fi

# --- 2. Run Evaluation Pipeline ---
# The evaluation should use the model trained in the step above.
# If TRAINED_MODEL_PATH was successfully captured, use it.
# Otherwise, EvaluatorAgent might use a default model_path from its config,
# or it needs to be passed.
log_message "Starting evaluation pipeline for $TARGET_SYMBOL..."

# Construct a placeholder model path for the command, assuming it's needed.
# This is where the actual path of the newly trained model should be inserted.
# For this example, if TRAINED_MODEL_PATH is not set, the main.py --model_path argument will be missing,
# and the EvaluatorAgent will rely on its default or fail if none is set.
# Let's make the script illustrative:
PLACEHOLDER_MODEL_FOR_EVAL="PLEASE_SET_MODEL_PATH_HERE_OR_IN_EVAL_CONFIG" 
# If TRAINED_MODEL_PATH is available:
# EVAL_MODEL_ARG="--model_path $TRAINED_MODEL_PATH"
# else
EVAL_MODEL_ARG="" # Relies on EvaluatorAgent's default config for model_path

"$PYTHON_EXECUTABLE" "$MAIN_PY_SCRIPT" evaluate \
    --symbol "$TARGET_SYMBOL" \
    --start_date "$EVAL_START_DATE" \
    --end_date "$EVAL_END_DATE" \
    --interval "$EVAL_INTERVAL" \
    $EVAL_MODEL_ARG \ # This will be empty if TRAINED_MODEL_PATH isn't robustly found
    --main_config "$MAIN_CONFIG_FILE" \
    --model_params "$MODEL_PARAMS_FILE" \
    --risk_limits "$RISK_LIMITS_FILE"
    # Add --use_cached_data if appropriate for evaluation data

if [ $? -ne 0 ]; then
    log_message "ERROR: Evaluation pipeline failed for $TARGET_SYMBOL. Check logs."
    # Don't necessarily exit; training might have been useful.
else
    log_message "Evaluation pipeline completed for $TARGET_SYMBOL."
fi

# --- 3. Optional: Model Promotion Logic ---
# Based on evaluation metrics (e.g., from a report file generated by EvaluatorAgent),
# decide if the newly trained model should be promoted to "production".
# This could involve:
# - Parsing the evaluation report.
# - Comparing metrics against a baseline or previous production model.
# - If better, copying the new model to a 'production_model.zip' path or updating a config.
log_message "Model promotion logic (TODO): Implement checks and promotion if criteria met."
# Example:
# EVAL_REPORT_DIR=$(grep 'reports_dir:' "$MAIN_CONFIG_FILE" | awk '{print $2}' | tr -d '"')
# LATEST_REPORT_FILE=$(ls -t "${PROJECT_ROOT}/${EVAL_REPORT_DIR}"/"${TARGET_SYMBOL}"*.txt | head -1)
# if [ -f "$LATEST_REPORT_FILE" ]; then
#     log_message "Found evaluation report: $LATEST_REPORT_FILE"
#     # score=$(grep 'Sharpe Ratio:' "$LATEST_REPORT_FILE" | awk '{print $3}') # Example parsing
#     # log_message "Parsed Sharpe Ratio: $score"
#     # if (( $(echo "$score > 1.0" | bc -l) )); then # Example criterion
#     #     log_message "Model meets promotion criteria. Promoting model (Conceptual)."
#     #     # cp "$TRAINED_MODEL_PATH" "${PROJECT_ROOT}/${MODEL_DIR}/production_model_${TARGET_SYMBOL}.zip"
#     # else
#     #     log_message "Model does not meet promotion criteria."
#     # fi
# else
#     log_message "WARNING: Evaluation report not found. Cannot run promotion logic."
# fi


# --- Cleanup (Optional) ---
# rm -f "$TRAINED_MODEL_OUTPUT_FILE" # Clean up temp file

log_message "--- Weekly Retrain and Evaluation Script FINISHED ---"
echo "" # Add a blank line in the log for readability

# --- Example Cron Job Entry ---
# This would typically be added to your crontab (e.g., by running `crontab -e`)
#
# SHELL=/bin/bash
# PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin # Ensure PATH is set
#
# # Run weekly_retrain.sh every Sunday at 2:00 AM
# 0 2 * * 0 /path/to/your/rl_trading_platform/scripts/weekly_retrain.sh
#
# Notes for cron:
# 1. Use absolute paths for all scripts and files, as cron jobs have a minimal environment.
#    The PROJECT_ROOT variable in this script is crucial.
# 2. Ensure the script is executable (`chmod +x weekly_retrain.sh`).
# 3. Redirect stdout and stderr from the cron job to a log file for debugging:
#    `0 2 * * 0 /path/to/project/scripts/weekly_retrain.sh >> /path/to/project/logs/cron_job.log 2>&1`
#    (The script itself also logs to CRON_LOG_FILE, so this might be redundant or complementary).
# 4. Manage Python environment: If using a virtual environment, ensure it's activated
#    within the script or PYTHON_EXECUTABLE points to the venv's python.
# 5. Test the script thoroughly from the command line before adding to cron.
```

**Accompanying `AGENTS.md` (conceptual, if you want to put it in `scripts/AGENTS.md`):**

```markdown
# AGENTS.md for rl_trading_platform/scripts/

## Regarding `weekly_retrain.sh`

This script is designed to automate the weekly retraining and evaluation of RL trading models.

**Key Responsibilities:**
1.  **Dynamic Date Calculation:** It calculates rolling date windows for fetching training and evaluation data. Ensure the `date` command syntax is compatible with your system (GNU `date` is assumed for advanced features like `date -d "yesterday - 2 years"`).
2.  **Pipeline Execution:** It calls `src/main.py` to run the `train` and `evaluate` modes.
3.  **Model Path Handling (CRITICAL TODO):**
    *   The current script has a placeholder for capturing the path of the newly trained model from the `train` step and passing it to the `evaluate` step.
    *   **Recommendation for robust implementation:** Modify `src/main.py` (or `OrchestratorAgent`) in `train` mode to write the successfully saved model's absolute path to a well-known temporary file (e.g., `logs/last_trained_model_path.txt`). This script can then reliably read this path.
    *   Alternatively, the `evaluate` mode in `main.py` could be enhanced to find the "latest" model for a given symbol if a specific path isn't provided, but this can be ambiguous.
4.  **Logging:** The script logs its progress to `logs/cron_retrain_schedule.log` within the project. Standard output/error from cron itself should also be managed (see cron job example).
5.  **Environment:** It includes placeholders for activating a Python virtual environment and setting `PYTHONPATH`. These are crucial for correct execution in a cron environment.
6.  **Model Promotion:** A conceptual section for model promotion logic is included. This would involve parsing evaluation metrics and making a decision to deploy the new model.

**Pre-requisites for running this script (especially via cron):**
*   The `PROJECT_ROOT` variable **must** be set correctly to the absolute path of your `rl_trading_platform` directory.
*   `PYTHON_EXECUTABLE` should point to the correct Python interpreter (ideally within a virtual environment).
*   All paths to config files and `main.py` must be correct and accessible from the execution context of the script.
*   The script must be executable (`chmod +x weekly_retrain.sh`).
*   The Python environment must have all necessary dependencies installed (e.g., from `requirements.txt`).
*   The `src/main.py` script and underlying agents must be functional.

**When modifying `src/main.py` or agent behavior:**
*   Consider how changes might affect this script, especially regarding command-line arguments, output, and file paths.
*   If model saving locations or naming conventions change, this script might need updates.
*   If evaluation reports change format, the conceptual model promotion logic would need to adapt.
```
