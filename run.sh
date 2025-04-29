#!/bin/bash

# Check if Docker is installed
if ! command -v docker &>/dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &>/dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if the image exists, if not build it
if ! docker image inspect dsl-perf-prediction:latest &>/dev/null; then
    echo "Building Docker image..."
    docker compose build
fi

# If no arguments provided, start interactive session
if [ $# -eq 0 ]; then
    echo "Starting interactive session..."
    docker compose run palladio-approximator
else
    # Run the provided command
    echo "Running command: $@"
    docker compose run palladio-approximator "$@"
fi
