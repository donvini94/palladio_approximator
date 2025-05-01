#!/bin/bash

if [[ "$1" == "build" ]]; then
    docker compose build
    exit 0
fi

if [[ "$1" == "bash" ]]; then
    docker compose run --rm palladio bash
    exit 0
fi

if [[ "$1" == "check-gpu" ]]; then
    docker compose run --rm palladio nvidia-smi
    docker compose run --rm palladio python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    exit 0
fi

if [[ "$1" == "train" ]]; then
    # Default to RF model with TFIDF embeddings in summary mode
    MODEL=${2:-rf}
    EMBEDDING=${3:-tfidf}
    MODE=${4:-summary}

    docker compose run --rm palladio python3 train.py \
        --model $MODEL \
        --embedding $EMBEDDING \
        --prediction_mode $MODE
    exit 0
fi

if [[ "$1" == "mlflow" ]]; then
    # Start MLflow server on port 5000
    docker compose run --rm -p 5000:5000 palladio mlflow ui --host 0.0.0.0 --port 5000
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
