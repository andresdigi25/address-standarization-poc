#!/bin/bash

# Stop execution if any command fails
set -e

echo "Running XGBoost Address Matching Example..."

# Create necessary directories
mkdir -p output

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for required files
if [ ! -f xgboost_address_matching.py ]; then
    echo "ERROR: xgboost_address_matching.py not found in current directory!"
    exit 1
fi

if [ ! -f Dockerfile ]; then
    echo "ERROR: Dockerfile not found in current directory!"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t xgboost-address-matching .

# Run the Docker container
echo "Running address matching example..."
docker run -v "$(pwd)/output:/app/output" xgboost-address-matching

echo "Execution completed! Check the 'output' directory for results."