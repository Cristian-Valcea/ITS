#!/bin/bash
# scripts/feature_store_gc.sh
# Cron job script for IntradayJules Feature Store Garbage Collection
#
# Add to crontab with:
# 0 2 * * * /path/to/IntradayJules/scripts/feature_store_gc.sh
#
# This runs daily at 2 AM to clean up old cache files

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="$PROJECT_ROOT/venv/bin/python"
GC_SCRIPT="$PROJECT_ROOT/src/shared/disk_gc_service.py"

# Default settings (can be overridden by environment variables)
RETENTION_WEEKS="${FEATURE_STORE_GC_RETENTION_WEEKS:-4}"
CACHE_ROOT="${FEATURE_STORE_PATH:-$HOME/.feature_cache}"
LOG_DIR="${FEATURE_STORE_LOG_DIR:-$PROJECT_ROOT/logs}"
DRY_RUN="${FEATURE_STORE_GC_DRY_RUN:-false}"

# Create log directory
mkdir -p "$LOG_DIR"

# Log file with timestamp
LOG_FILE="$LOG_DIR/feature_store_gc_$(date +%Y%m%d_%H%M%S).log"
JSON_OUTPUT="$LOG_DIR/feature_store_gc_$(date +%Y%m%d_%H%M%S).json"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to cleanup old log files (keep last 30 days)
cleanup_old_logs() {
    find "$LOG_DIR" -name "feature_store_gc_*.log" -mtime +30 -delete 2>/dev/null || true
    find "$LOG_DIR" -name "feature_store_gc_*.json" -mtime +30 -delete 2>/dev/null || true
}

# Main execution
main() {
    log "Starting Feature Store Garbage Collection"
    log "Cache root: $CACHE_ROOT"
    log "Retention: $RETENTION_WEEKS weeks"
    log "Dry run: $DRY_RUN"
    log "Log file: $LOG_FILE"
    
    # Check if Python environment exists
    if [[ ! -f "$PYTHON_ENV" ]]; then
        log "ERROR: Python environment not found at $PYTHON_ENV"
        exit 1
    fi
    
    # Check if GC script exists
    if [[ ! -f "$GC_SCRIPT" ]]; then
        log "ERROR: GC script not found at $GC_SCRIPT"
        exit 1
    fi
    
    # Check if cache directory exists
    if [[ ! -d "$CACHE_ROOT" ]]; then
        log "WARNING: Cache directory does not exist: $CACHE_ROOT"
        log "Creating cache directory..."
        mkdir -p "$CACHE_ROOT"
    fi
    
    # Build command arguments
    GC_ARGS=(
        --cache-root "$CACHE_ROOT"
        --retention-weeks "$RETENTION_WEEKS"
        --output-json "$JSON_OUTPUT"
        --verbose
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        GC_ARGS+=(--dry-run)
    fi
    
    # Run garbage collection
    log "Executing: $PYTHON_ENV $GC_SCRIPT ${GC_ARGS[*]}"
    
    if "$PYTHON_ENV" "$GC_SCRIPT" "${GC_ARGS[@]}" >> "$LOG_FILE" 2>&1; then
        log "Garbage collection completed successfully"
        
        # Extract summary from JSON output
        if [[ -f "$JSON_OUTPUT" ]]; then
            if command -v jq >/dev/null 2>&1; then
                log "Summary:"
                jq -r '.summary | to_entries[] | "  \(.key): \(.value)"' "$JSON_OUTPUT" | tee -a "$LOG_FILE"
            else
                log "JSON output saved to: $JSON_OUTPUT"
            fi
        fi
        
        exit_code=0
    else
        log "ERROR: Garbage collection failed"
        exit_code=1
    fi
    
    # Cleanup old log files
    cleanup_old_logs
    
    log "Feature Store GC completed with exit code: $exit_code"
    exit $exit_code
}

# Handle signals
trap 'log "Received signal, terminating..."; exit 130' INT TERM

# Run main function
main "$@"