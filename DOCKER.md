# Docker Deployment Guide

This guide explains how to build and deploy the MVP LightGBM Price Prediction API using Docker.

## Prerequisites

- Docker installed and running
- Python 3.11+ (for running helper scripts)
- MLflow tracking data in `./mlruns/` directory

## Quick Start

### Option 1: Using Helper Scripts (Recommended)

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

### Option 2: Manual Docker Commands

**1. Build the image:**
```bash
docker build -t mvp-lightgbm-price:latest .
```

**2. Find the latest model run ID:**
```bash
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
exp = mlflow.get_experiment_by_name('mvp_lightgbm_price')
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=['start_time DESC'], max_results=1)
print(runs.iloc[0]['run_id'])
"
```

**3. Run the container:**
```bash
docker run -p 8000:8000 \
  -e MODEL_URI="runs:/<run_id>/model" \
  mvp-lightgbm-price:latest
```

Or let the container auto-detect the latest model:
```bash
docker run -p 8000:8000 mvp-lightgbm-price:latest
```

### Option 3: Docker Compose

```bash
docker-compose up --build
```

## How It Works

1. **Dockerfile**: Builds a Python 3.11 slim image with all dependencies
2. **Entrypoint Script**: Automatically finds the latest model if `MODEL_URI` is not set
3. **Model Loading**: The FastAPI app loads the model from MLflow using the `MODEL_URI` environment variable
4. **Health Check**: Built-in health check at `/healthz` endpoint

## Environment Variables

- `MODEL_URI`: MLflow model URI (e.g., `runs:/<run_id>/model`). If not set, the entrypoint script will try to find the latest model automatically.
- `PYTHONPATH`: Set to `/app` by default
- `PORT`: Internal port (default: 8000)

## Testing the Container

**Health check:**
```bash
curl http://localhost:8000/healthz
```

**API documentation:**
```bash
# Open in browser
http://localhost:8000/docs
```

**Make a prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "rfq_id": 1,
      "customer_tier": "A",
      "material": "steel",
      "thickness_mm": 5.0,
      "length_mm": 1000.0,
      "width_mm": 500.0,
      "material_cost_rub": 1000.0,
      "labor_minutes_per_unit": 10.0,
      "labor_cost_rub": 500.0,
      "qty": 10
    }
  ]'
```

## Image Size Optimization

The `.dockerignore` file excludes:
- Old MLflow runs (only latest run is included)
- Training data
- Virtual environments
- Test files
- Development files

This keeps the Docker image size small while including everything needed for production.

## Troubleshooting

**Container fails to start:**
- Check that `mlruns/` directory exists and contains model data
- Verify `MODEL_URI` is set correctly or that the entrypoint script can find a model
- Check container logs: `docker logs <container_id>`

**Model not found:**
- Ensure the latest model run is included in the Docker image
- Check `.dockerignore` isn't excluding the model directory
- Manually set `MODEL_URI` environment variable

**Port already in use:**
- Change the port mapping: `docker run -p 8001:8000 ...`
- Or stop the existing container using port 8000

## Production Deployment

For production deployment:

1. **Use a specific model version:**
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_URI="runs:/<specific_run_id>/model" \
     mvp-lightgbm-price:latest
   ```

2. **Use Docker Compose with environment variables:**
   ```yaml
   environment:
     MODEL_URI: "runs:/<run_id>/model"
   ```

3. **Deploy to cloud platforms:**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - Kubernetes

4. **Add monitoring and logging:**
   - Set up health check monitoring
   - Configure log aggregation
   - Add metrics collection (Prometheus, etc.)

