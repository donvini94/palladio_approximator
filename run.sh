#!/bin/bash

if [[ "$1" == "build" ]]; then
    # Build using docker build instead of docker-compose
    docker build -t palladio_approximator .
    exit 0
fi

if [[ "$1" == "bash" ]]; then
    # Start a bash shell using the working GPU syntax
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/mlruns":/app/mlruns \
        -w /app \
        palladio_approximator bash
    exit 0
fi

if [[ "$1" == "check-gpu" ]]; then
    # Check GPU using the working syntax
    echo "Checking nvidia-smi..."
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        palladio_approximator nvidia-smi

    echo -e "\nChecking PyTorch GPU access..."
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        palladio_approximator python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    exit 0
fi

if [[ "$1" == "train" ]]; then
    # Default to RF model with TFIDF embeddings in summary mode
    MODEL=${2:-rf}
    EMBEDDING=${3:-tfidf}
    MODE=${4:-summary}

    # Run training using the working GPU syntax
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/mlruns":/app/mlruns \
        -w /app \
        palladio_approximator python3 train.py \
        --model $MODEL \
        --embedding $EMBEDDING \
        --prediction_mode $MODE
    exit 0
fi

if [[ "$1" == "mlflow" ]]; then
    # Start MLflow server on port 5000
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/mlruns":/app/mlruns \
        -p 5000:5000 \
        -w /app \
        palladio_approximator mlflow ui --host 0.0.0.0 --port 5000
    exit 0
fi

# Print usage if no valid command was provided
echo "Usage: ./run.sh [command]"
echo "Commands:"
echo "  build              Build the Docker image"
echo "  bash               Start a bash shell in the container"
echo "  check-gpu          Check if GPU is available and working"
echo "  mlflow             Start MLflow UI on port 5000"
echo "  train [model] [embedding] [mode]   Train a model"
echo "    model: rf (default), ridge, lasso"
echo "    embedding: tfidf (default), bert, longformer"
echo "    mode: summary (default), timeseries"
echo ""
echo "Examples:"
echo "  ./run.sh train rf bert summary"
echo "  ./run.sh train ridge tfidf timeseries"
