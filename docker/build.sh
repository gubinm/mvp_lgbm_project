#!/bin/bash
# Build Docker image
# The entrypoint script will automatically detect the latest model at runtime

set -e

IMAGE_NAME=${IMAGE_NAME:-"model-api"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Note: Model will be auto-detected when container starts"

# Build the image (entrypoint will auto-detect the model)
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "Docker image built successfully!"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"

