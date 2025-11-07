# Run Docker container with the latest model (PowerShell script)

$ErrorActionPreference = "Stop"

$imageName = if ($env:IMAGE_NAME) { $env:IMAGE_NAME } else { "model-api" }
$imageTag = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }
$port = if ($env:PORT) { $env:PORT } else { 8000 }

Write-Host "Running container: ${imageName}:${imageTag}"
Write-Host "API will be available at: http://localhost:${port}"
Write-Host "Model will be auto-detected by the container"
Write-Host ""

docker run -it --rm `
    -p ${port}:8000 `
    ${imageName}:${imageTag}

