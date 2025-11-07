# MVP LightGBM Price Project

## Train
```bash
pip install -r requirements.txt
cd ml
python train.py --csv ../data/raw/mvp_quotes.csv
```
Env options:

**Linux/Mac (bash):**
```bash
export MLFLOW_TRACKING_URI=file:./mlruns
export MLFLOW_EXPERIMENT=mvp_lightgbm_price
# Optional: export MLFLOW_REGISTER_MODEL=mvp-lightgbm-price
```

**Windows (PowerShell):**
```powershell
$env:MLFLOW_TRACKING_URI="file:./mlruns"
$env:MLFLOW_EXPERIMENT="mvp_lightgbm_price"
# Optional: $env:MLFLOW_REGISTER_MODEL="mvp-lightgbm-price"
```

**Windows (CMD):**
```cmd
set MLFLOW_TRACKING_URI=file:./mlruns
set MLFLOW_EXPERIMENT=mvp_lightgbm_price
REM Optional: set MLFLOW_REGISTER_MODEL=mvp-lightgbm-price
```

## Serve

**Linux/Mac (bash):**
```bash
export MODEL_URI=runs:/0f2260a258f44f3f8a559d440efe92f2/model
export MLFLOW_TRACKING_URI=file:./mlruns
PYTHONPATH=. uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:MODEL_URI="runs:/0f2260a258f44f3f8a559d440efe92f2/model"
$env:MLFLOW_TRACKING_URI="file:./mlruns"
$env:PYTHONPATH="."
uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

**Windows (CMD):**
```cmd
set MODEL_URI=runs:/0f2260a258f44f3f8a559d440efe92f2/model
set MLFLOW_TRACKING_URI=file:./mlruns
set PYTHONPATH=.
uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

Or via Registry:

**Linux/Mac (bash):**
```bash
export MODEL_URI=models:/mvp-lightgbm-price/Production
PYTHONPATH=. uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:MODEL_URI="models:/mvp-lightgbm-price/Production"
$env:PYTHONPATH="."
uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

**Windows (CMD):**
```cmd
set MODEL_URI=models:/mvp-lightgbm-price/Production
set PYTHONPATH=.
uvicorn ml.serve:app --host 0.0.0.0 --port 8000
```

## Docker
### Quick Start

**Build the Docker image:**
```bash
# Linux/Mac
./docker/build.sh

# Windows PowerShell
.\docker\build.ps1
```

**Run the container:**
```bash
# Linux/Mac
./docker/run.sh

# Windows PowerShell
.\docker\run.ps1
```

The API will be available at `http://localhost:8000`

### Quick Start

**Build and run in one command:**
```bash
# Build
docker build -t model-api .

# Run (auto-detects latest model - no MODEL_URI needed!)
docker run -p 8000:8000 model-api
```

That's it! The container automatically finds and loads the latest trained model.

**Run with specific model (optional):**
```bash
# Using registered model path
docker run -p 8000:8000 -e MODEL_URI=/app/mlruns/726614158927855195/models/m-<model_id>/artifacts model-api
```

**Using Docker Compose:**
```bash
docker-compose up --build
```

### Docker Features

- **Automatic model detection**: If `MODEL_URI` is not set, the container will automatically find the latest trained model
- **Health checks**: Built-in health check endpoint at `/healthz`
- **Optimized image**: Only includes the latest model to keep image size small
- **Production ready**: Includes proper error handling and logging

## Tests
This project uses [pytest](https://docs.pytest.org/) for testing.

Run all tests:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_pipeline.py
```

## View MLFlow Metrics

This project uses [MLFlow](https://www.mlflow.org/) to track model training metrics (RMSE, MAE, MAPE, R²).

### Option 1: MLFlow UI (Recommended)

Start the MLFlow UI to view metrics in a web interface:

**Linux/Mac (bash):**
```bash
cd ml
mlflow ui --backend-store-uri file:./mlruns
```

**Windows (PowerShell):**
```powershell
cd ml
mlflow ui --backend-store-uri file:./mlruns
```

Then open your browser to: `http://localhost:5000`

### Option 2: Command Line Script

View metrics from the command line:

**Linux/Mac (bash):**
```bash
python ml/view_metrics.py
# Or if using venv:
./venv/bin/python ml/view_metrics.py
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\python.exe ml/view_metrics.py
```

### Option 3: Programmatic Access

Access metrics programmatically using the MLFlow API:

```python
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
experiment = mlflow.get_experiment_by_name("mvp_lightgbm_price")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get latest run metrics
latest_run = runs.iloc[0]
print(f"RMSE: {latest_run['metrics.rmse']:.3f}")
print(f"MAE: {latest_run['metrics.mae']:.3f}")
print(f"MAPE: {latest_run['metrics.mape']:.2f}%")
print(f"R²: {latest_run['metrics.r2']:.3f}")
```

## Linting
This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

Check for linting issues:
```bash
ruff check ml/
```

Auto-fix linting issues:
```bash
ruff check --fix ml/
```

Format code:
```bash
ruff format ml/
```

Check and format in one command:
```bash
ruff check --fix ml/ && ruff format ml/
```
