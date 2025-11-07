FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml/ ./ml/
COPY pyproject.toml ./

# Copy MLflow runs (models) - this will be filtered by .dockerignore
# Only the latest model run will be included
COPY mlruns/ ./mlruns/

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install PyYAML for entrypoint script
RUN pip install --no-cache-dir pyyaml

# Set MODEL_URI to point to the latest model
# This can be overridden at runtime with -e MODEL_URI=...
# If not set, entrypoint.sh will try to find the latest model automatically
ENV MODEL_URI=""

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Run the application
CMD ["uvicorn", "ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
