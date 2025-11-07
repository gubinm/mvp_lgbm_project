"""Script to view MLFlow metrics from training runs."""

import math
import os
import sys
from pathlib import Path

import mlflow

# Try to find mlruns directory (could be in ml/ or root)
possible_paths = [Path("mlruns"), Path("ml/mlruns"), Path("../mlruns")]
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)

if tracking_uri is None:
    for path in possible_paths:
        if path.exists():
            tracking_uri = f"file:{path.absolute()}"
            break
    if tracking_uri is None:
        tracking_uri = "file:./mlruns"

mlflow.set_tracking_uri(tracking_uri)
experiment_name = os.getenv("MLFLOW_EXPERIMENT", "mvp_lightgbm_price")

# Get the experiment
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        sys.exit(1)

    experiment_id = experiment.experiment_id
    print(f"Experiment: {experiment_name} (ID: {experiment_id})\n")

    # Get all runs for this experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])

    if len(runs) == 0:
        print("No runs found in this experiment.")
        sys.exit(1)

    print(f"Found {len(runs)} run(s):\n")
    print("=" * 80)

    # Display metrics for each run
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", "N/A")
        start_time = run.get("start_time", "N/A")

        print(f"\nRun: {run_name}")
        print(f"Run ID: {run_id}")
        print(f"Start Time: {start_time}")
        print("-" * 80)

        # Display metrics
        metrics = ["rmse", "mae", "mape", "r2"]
        print("Metrics:")
        has_metrics = False
        for metric in metrics:
            value = run.get(f"metrics.{metric}", None)
            if value is not None and not (
                isinstance(value, float) and math.isnan(value)
            ):  # Check for NaN
                has_metrics = True
                if metric == "mape":
                    print(f"  {metric.upper()}: {value:.2f}%")
                elif metric == "r2":
                    print(f"  {metric.upper()}: {value:.3f}")
                else:
                    print(f"  {metric.upper()}: {value:.3f}")
        if not has_metrics:
            print("  (No metrics available - run may have failed or is incomplete)")

        print("=" * 80)

    # Show latest run details
    latest_run = runs.iloc[0]
    latest_run_id = latest_run["run_id"]
    print(f"\nLatest Run ID: {latest_run_id}")
    print("To view in MLFlow UI, run: mlflow ui")
    print("Then navigate to: http://localhost:5000")

except Exception as e:
    print(f"Error accessing MLFlow: {e}")
    sys.exit(1)
