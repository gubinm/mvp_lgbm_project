#!/bin/bash
# Run Docker container with the latest model

set -e

IMAGE_NAME=${IMAGE_NAME:-"model-api"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
PORT=${PORT:-8000}

echo "Running container: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "API will be available at: http://localhost:${PORT}"
echo "Model will be auto-detected by the container"
echo ""

docker run -it --rm \
    -p ${PORT}:8000 \
    ${IMAGE_NAME}:${IMAGE_TAG}

