#!/bin/bash

if [[ "$1" == "build" ]]; then
    # Build using docker build instead of docker-compose
    docker build -t registry.dumusstbereitsein.de/palladio_approximator .
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
        registry.dumusstbereitsein.de/palladio_approximator bash
    exit 0
fi

if [[ "$1" == "check-gpu" ]]; then
    # Check GPU using the working syntax
    echo "Checking nvidia-smi..."
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        registry.dumusstbereitsein.de/palladio_approximator nvidia-smi

    echo -e "\nChecking PyTorch GPU access..."
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        registry.dumusstbereitsein.de/palladio_approximator python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
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
        registry.dumusstbereitsein.de/palladio_approximator python3 train.py \
        --model $MODEL \
        --embedding $EMBEDDING \
        --prediction_mode $MODE
    exit 0
fi

if [[ "$1" == "batch-experiments" ]]; then
    # Get optional settings
    MODES=${2:-"summary"}
    EMBEDDINGS=${3:-"tfidf bert llama"}
    MODELS=${4:-"rf ridge lasso torch"}

    echo "====================================="
    echo "Starting batch experiments"
    echo "====================================="
    echo "Modes: $MODES"
    echo "Embeddings: $EMBEDDINGS"
    echo "Models: $MODELS"
    echo "====================================="

    # Store the start time
    START_TIME=$(date +%s)

    # Create a timestamp for this batch run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="batch_experiments_${TIMESTAMP}.log"

    echo "Experiment batch started at $(date)" | tee -a "$LOG_FILE"

    # Track success/failure
    TOTAL=0
    SUCCESS=0

    # Run each experiment combination
    for MODE in $MODES; do
        for EMBEDDING in $EMBEDDINGS; do
            for MODEL in $MODELS; do

                # Skip torch model with tfidf for very large feature spaces
                if [[ "$MODEL" == "torch" && "$EMBEDDING" == "tfidf" ]]; then
                    echo "Skipping torch + tfidf combination due to memory constraints" | tee -a "$LOG_FILE"
                    continue
                fi

                TOTAL=$((TOTAL + 1))

                # Print experiment info
                echo -e "\n====================================="
                echo "Starting experiment [$(date)]:"
                echo "Mode: $MODE"
                echo "Embedding: $EMBEDDING"
                echo "Model: $MODEL"
                echo "=====================================" | tee -a "$LOG_FILE"

                # Customize hyperparameters based on model type
                EXTRA_ARGS=""

                # Add model-specific hyperparameters
                if [[ "$MODEL" == "rf" ]]; then
                    # Try different numbers of estimators for Random Forest
                    for N_ESTIMATORS in 100 200 300; do
                        echo "Running RF with n_estimators=$N_ESTIMATORS" | tee -a "$LOG_FILE"

                        # Run training with parameters
                        docker run --rm \
                            --device=nvidia.com/gpu=all \
                            -v "$(pwd)":/app \
                            -v "$(pwd)/data":/app/data \
                            -v "$(pwd)/mlruns":/app/mlruns \
                            -w /app \
                            registry.dumusstbereitsein.de/palladio_approximator python3 train.py \
                            --model $MODEL \
                            --embedding $EMBEDDING \
                            --prediction_mode $MODE \
                            --n_estimators $N_ESTIMATORS \
                            --save_features

                        if [ $? -eq 0 ]; then
                            echo "✅ Experiment successful: $MODEL (n_estimators=$N_ESTIMATORS) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                            SUCCESS=$((SUCCESS + 1))
                        else
                            echo "❌ Experiment failed: $MODEL (n_estimators=$N_ESTIMATORS) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                        fi
                    done
                elif [[ "$MODEL" == "ridge" || "$MODEL" == "lasso" ]]; then
                    # Try different alpha values for linear models
                    for ALPHA in 0.1 1.0 10.0; do
                        echo "Running $MODEL with alpha=$ALPHA" | tee -a "$LOG_FILE"

                        # Run training with parameters
                        docker run --rm \
                            --device=nvidia.com/gpu=all \
                            -v "$(pwd)":/app \
                            -v "$(pwd)/data":/app/data \
                            -v "$(pwd)/mlruns":/app/mlruns \
                            -w /app \
                            registry.dumusstbereitsein.de/palladio_approximator python3 train.py \
                            --model $MODEL \
                            --embedding $EMBEDDING \
                            --prediction_mode $MODE \
                            --alpha $ALPHA \
                            --save_features

                        if [ $? -eq 0 ]; then
                            echo "✅ Experiment successful: $MODEL (alpha=$ALPHA) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                            SUCCESS=$((SUCCESS + 1))
                        else
                            echo "❌ Experiment failed: $MODEL (alpha=$ALPHA) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                        fi
                    done
                elif [[ "$MODEL" == "torch" ]]; then
                    # Try different hyperparameters for neural networks
                    # Vary batch size and epochs
                    for BATCH_SIZE in 64 128; do
                        for EPOCHS in 100 200; do
                            echo "Running torch with batch_size=$BATCH_SIZE, epochs=$EPOCHS" | tee -a "$LOG_FILE"

                            # Run training with PyTorch parameters
                            docker run --rm \
                                --device=nvidia.com/gpu=all \
                                -v "$(pwd)":/app \
                                -v "$(pwd)/data":/app/data \
                                -v "$(pwd)/mlruns":/app/mlruns \
                                -w /app \
                                registry.dumusstbereitsein.de/palladio_approximator python3 train.py \
                                --model $MODEL \
                                --embedding $EMBEDDING \
                                --prediction_mode $MODE \
                                --batch_size $BATCH_SIZE \
                                --epochs $EPOCHS \
                                --use_gpu \
                                --save_features

                            if [ $? -eq 0 ]; then
                                echo "✅ Experiment successful: $MODEL (batch_size=$BATCH_SIZE, epochs=$EPOCHS) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                                SUCCESS=$((SUCCESS + 1))
                            else
                                echo "❌ Experiment failed: $MODEL (batch_size=$BATCH_SIZE, epochs=$EPOCHS) + $EMBEDDING + $MODE" | tee -a "$LOG_FILE"
                            fi
                        done
                    done
                fi
            done

            # After each embedding, clean up any lingering processes/memory to ensure fresh start for next embedding
            echo "Finished all models for embedding $EMBEDDING in mode $MODE" | tee -a "$LOG_FILE"
        done
    done

    # Calculate and show total execution time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    echo -e "\n====================================="
    echo "Batch experiments completed!"
    echo "====================================="
    echo "Total experiments: $TOTAL"
    echo "Successful: $SUCCESS"
    echo "Failed: $((TOTAL - SUCCESS))"
    echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Log file: $LOG_FILE"
    echo "====================================="

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
        registry.dumusstbereitsein.de/palladio_approximator mlflow ui --host 0.0.0.0 --port 5000
    exit 0
fi

if [[ "$1" == "summarize" ]]; then
    # Optional filter flags
    FILTER_MODEL=""
    FILTER_EMBEDDING=""
    FORMAT="both"
    SORT_BY="test_r2_score"

    # Parse any additional arguments
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
        --model)
            FILTER_MODEL="--filter-model $2"
            shift 2
            ;;
        --embedding)
            FILTER_EMBEDDING="--filter-embedding $2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --sort)
            SORT_BY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
        esac
    done

    # First ensure MLflow is running in the background
    echo "Checking if MLflow server is running..."
    if ! curl -s http://localhost:5000 >/dev/null; then
        echo "Starting MLflow server in the background..."
        docker run --rm -d \
            --device=nvidia.com/gpu=all \
            -v "$(pwd)":/app \
            -v "$(pwd)/data":/app/data \
            -v "$(pwd)/mlruns":/app/mlruns \
            -p 5000:5000 \
            -w /app \
            --name mlflow_server \
            registry.dumusstbereitsein.de/palladio_approximator mlflow ui --host 0.0.0.0 --port 5000

        # Wait for MLflow to start
        echo "Waiting for MLflow server to start..."
        sleep 5
    fi

    # Run the summarize script
    echo "Generating experiment summary..."
    docker run --rm -it \
        --device=nvidia.com/gpu=all \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/mlruns":/app/mlruns \
        --network host \
        -w /app \
        registry.dumusstbereitsein.de/palladio_approximator python3 summarize_experiments.py \
        --format "$FORMAT" \
        --sort "$SORT_BY" \
        $FILTER_MODEL \
        $FILTER_EMBEDDING

    exit 0
fi

# Print usage if no valid command was provided
echo "Usage: ./run.sh [command]"
echo "Commands:"
echo "  build                Build the Docker image"
echo "  bash                 Start a bash shell in the container"
echo "  check-gpu            Check if GPU is available and working"
echo "  mlflow               Start MLflow UI on port 5000"
echo "  train [model] [embedding] [mode]   Train a model"
echo "    model: rf (default), ridge, lasso, torch"
echo "    embedding: tfidf (default), bert, llama"
echo "    mode: summary (default)"
echo "  batch-experiments [modes] [embeddings] [models]   Run a series of experiments with all combinations"
echo "    modes: space-separated list (default: \"summary\")"
echo "    embeddings: space-separated list (default: \"tfidf bert llama\")"
echo "    models: space-separated list (default: \"rf ridge lasso torch\")"
echo "  summarize [options]  Generate a summary of experiment results from MLflow"
echo "    --model MODEL      Filter results by model type"
echo "    --embedding EMB    Filter results by embedding type"
echo "    --format FORMAT    Output format: csv, markdown, or both (default: both)"
echo "    --sort METRIC      Metric to sort by (default: test_r2_score)"
echo ""
echo "Examples:"
echo "  ./run.sh train rf bert summary"
echo "  ./run.sh train ridge tfidf summary"
echo "  ./run.sh batch-experiments \"summary\" \"bert llama\" \"rf torch\""
echo "  ./run.sh summarize --model rf --embedding bert"
echo "  ./run.sh summarize --sort val_mse --format markdown"
echo ""
echo "Convenience Scripts:"
echo "  ./run_experiments.sh --name quick    Run a predefined experiment batch (quick, full, rf-only, etc.)"
echo "  ./run_experiments.sh --custom       Run experiments with custom parameters"
