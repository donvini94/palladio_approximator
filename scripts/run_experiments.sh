#!/bin/bash
# Script to run different experiment batches

# Default options
BATCH_NAME="full"

function print_usage() {
    echo "Usage: ./run_experiments.sh [options]"
    echo "Options:"
    echo "  --name NAME      Experiment batch name (default: full)"
    echo "                   Available presets: full, quick, rf-only, torch-only, embeddings-compare"
    echo "  --custom         Run with custom parameters (prompts for inputs)"
    echo ""
    echo "Examples:"
    echo "  ./run_experiments.sh --name quick"
    echo "  ./run_experiments.sh --name rf-only"
    echo "  ./run_experiments.sh --custom"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --name)
            BATCH_NAME="$2"
            shift
            shift
            ;;
        --custom)
            BATCH_NAME="custom"
            shift
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Set parameters based on batch name
# Get the absolute path to run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

case $BATCH_NAME in
    "full")
        echo "Running FULL experiment batch (all models, all embeddings)"
        bash "${RUN_SCRIPT}" batch-experiments "summary" "tfidf bert llama" "rf ridge lasso torch svm"
        ;;
    "quick")
        echo "Running QUICK experiment batch (limited parameters)"
        bash "${RUN_SCRIPT}" batch-experiments "summary" "bert" "rf torch"
        ;;
    "rf-only")
        echo "Running RF-ONLY experiment batch (Random Forest with all embeddings)"
        bash "${RUN_SCRIPT}" batch-experiments "summary" "tfidf bert llama" "rf"
        ;;
    "torch-only")
        echo "Running TORCH-ONLY experiment batch (PyTorch with bert and llama embeddings)"
        bash "${RUN_SCRIPT}" batch-experiments "summary" "bert llama" "torch"
        ;;
    "embeddings-compare")
        echo "Running EMBEDDINGS-COMPARE batch (comparing all embeddings with RF model)"
        bash "${RUN_SCRIPT}" batch-experiments "summary" "tfidf bert llama" "rf"
        ;;
    "custom")
        # Prompt for custom parameters
        echo "Enter custom experiment parameters:"
        
        read -p "Prediction modes (space-separated, e.g., 'summary'): " MODES
        MODES=${MODES:-"summary"}
        
        read -p "Embeddings (space-separated, e.g., 'tfidf bert llama'): " EMBEDDINGS
        EMBEDDINGS=${EMBEDDINGS:-"tfidf bert llama"}
        
        read -p "Models (space-separated, e.g., 'rf ridge lasso torch svm'): " MODELS
        MODELS=${MODELS:-"rf ridge lasso torch svm"}
        
        echo "Running CUSTOM experiment batch:"
        echo "  Modes: $MODES"
        echo "  Embeddings: $EMBEDDINGS"
        echo "  Models: $MODELS"
        
        bash "${RUN_SCRIPT}" batch-experiments "$MODES" "$EMBEDDINGS" "$MODELS"
        ;;
    *)
        echo "Unknown batch name: $BATCH_NAME"
        print_usage
        ;;
esac

echo "Experiment batch '$BATCH_NAME' completed!"