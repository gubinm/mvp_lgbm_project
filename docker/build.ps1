# Build Docker image (PowerShell script)
# The entrypoint script will automatically detect the latest model at runtime

$ErrorActionPreference = "Stop"

$imageName = if ($env:IMAGE_NAME) { $env:IMAGE_NAME } else { "model-api" }
$imageTag = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }

Write-Host "Building Docker image: ${imageName}:${imageTag}"
Write-Host "Note: Model will be auto-detected when container starts"

# Build the image (entrypoint will auto-detect the model)
docker build `
    -t "${imageName}:${imageTag}" `
    .

Write-Host ""
Write-Host "Docker image built successfully!" -ForegroundColor Green
Write-Host "  Image: ${imageName}:${imageTag}"
Write-Host ""
Write-Host "To run the container:"
Write-Host "  docker run -p 8000:8000 ${imageName}:${imageTag}"

