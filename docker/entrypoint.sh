#!/bin/bash
# Docker entrypoint script that finds the latest model if MODEL_URI is not set

# If MODEL_URI is not set, try to find the latest model
if [ -z "$MODEL_URI" ]; then
    echo "MODEL_URI not set, attempting to find latest model..."
    
    # Find the latest run ID from mlruns
    if [ -d "/app/mlruns" ]; then
        # Try to find the latest run using Python
        set +e  # Temporarily disable exit on error
        LATEST_RUN_ID=$(python3 -c "
import os
import sys
import yaml
from pathlib import Path

try:
    mlruns_path = Path('/app/mlruns')
    if not mlruns_path.exists():
        sys.exit(1)

    # Find experiment directory (numeric IDs only, exclude special dirs)
    exp_dirs = [d for d in mlruns_path.iterdir() 
                if d.is_dir() and d.name != '0' and d.name != 'models' 
                and d.name != '.trash' and d.name.isdigit()]
    if not exp_dirs:
        sys.exit(1)

    exp_dir = exp_dirs[0]
    latest_run = None
    latest_time = 0

    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == 'models' or run_dir.name == 'meta.yaml':
            continue
        meta_file = run_dir / 'meta.yaml'
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = yaml.safe_load(f)
                    end_time = meta.get('end_time', 0)
                    if end_time > latest_time:
                        latest_time = end_time
                        latest_run = run_dir.name
            except Exception:
                continue

    if latest_run:
        print(latest_run)
    else:
        sys.exit(1)
except Exception as e:
    print(f'Error finding model: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)
        PYTHON_EXIT_CODE=$?
        set -e  # Re-enable exit on error
        
        if [ $PYTHON_EXIT_CODE -eq 0 ] && [ -n "$LATEST_RUN_ID" ]; then
            # Try to find the registered model for this run
            # Models are registered in mlruns/<exp_id>/models/
            MODEL_PATH=""
            for exp_dir in /app/mlruns/*/; do
                if [ -d "${exp_dir}models" ]; then
                    for model_dir in "${exp_dir}models"/*/; do
                        if [ -f "${model_dir}meta.yaml" ]; then
                            # Check if this model is from the latest run
                            if grep -q "source_run_id: ${LATEST_RUN_ID}" "${model_dir}meta.yaml" 2>/dev/null; then
                                MODEL_PATH="${model_dir}artifacts"
                                break 2
                            fi
                        fi
                    done
                fi
            done
            
            if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
                MODEL_URI="$MODEL_PATH"
                echo "Found registered model for run: $LATEST_RUN_ID"
            else
                MODEL_URI="runs:/${LATEST_RUN_ID}/model"
                echo "Found latest model run: $LATEST_RUN_ID (using runs:/ URI)"
            fi
            echo "Setting MODEL_URI=$MODEL_URI"
            export MODEL_URI
        else
            echo "Error: Could not find latest model run"
            echo "Please set MODEL_URI environment variable"
            exit 1
        fi
    else
        echo "Error: /app/mlruns directory not found"
        echo "Please set MODEL_URI environment variable"
        exit 1
    fi
fi

echo "Using MODEL_URI=$MODEL_URI"

# Set MLflow tracking URI to local file system
export MLFLOW_TRACKING_URI="file:/app/mlruns"

# Execute the main command
exec "$@"

