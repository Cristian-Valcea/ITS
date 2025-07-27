#!/bin/bash

# scripts/run_api.sh
# Launches the FastAPI application using Uvicorn for local development.

# Ensure the script is run from the project root directory (rl_trading_platform)
# Or adjust paths accordingly.

# Default host and port
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
LOG_LEVEL=${API_LOG_LEVEL:-info}
WORKERS=${API_WORKERS:-1} # For Uvicorn, more workers usually for production under Gunicorn/Hypercorn

# Check if virtual environment exists and activate it
VENV_PATH="./.venv" # Assuming venv is in project root
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment from $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate virtual environment. Please ensure it's created and dependencies are installed."
        # exit 1 # Optionally exit if venv activation fails
    fi
else
    echo "INFO: No virtual environment found at $VENV_PATH. Using system Python interpreter."
fi

# Set PYTHONPATH to include the project root, so 'src' can be imported
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
echo "PYTHONPATH set to: $PYTHONPATH"
echo "Current working directory: $(pwd)"


echo "Starting FastAPI application with Uvicorn..."
echo "Access at: http://${HOST}:${PORT}/docs (Swagger UI) or http://${HOST}:${PORT}/redoc (ReDoc)"
echo "API Status: http://${HOST}:${PORT}/api/v1/status"

# uvicorn src.api.main:app --reload --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL" --workers "$WORKERS"
# Using --workers 1 for --reload mode as per Uvicorn recommendation.
# For production, you might use more workers, typically without --reload, and often behind a process manager like Gunicorn.
uvicorn src.api.main:app --reload --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"

# To run without reload (e.g., for more stable testing or if reload causes issues):
# uvicorn src.api.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"

# If you have multiple workers and don't need reload (more for production like setup):
# uvicorn src.api.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL" --workers 4

echo "FastAPI application stopped."
