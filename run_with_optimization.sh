#!/bin/bash
# Example script to run hyperparameter optimization for different model types

# Make script exit on first error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Ensure we're in the right directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Function to print section header
print_header() {
    echo -e "\n${GREEN}===== $1 =====${NC}\n"
}

# Function to print warning
print_warning() {
    echo -e "\n${YELLOW}WARNING: $1${NC}\n"
}

# Check if environment is set up
if ! command -v python &>/dev/null; then
    print_warning "Python not found. You might need to set up the environment first."
    echo "Try: ./run.sh build"
    exit 1
fi

# Check if the data directory exists
if [ ! -d "data" ]; then
    print_warning "Data directory not found. Please ensure your data is available."
    exit 1
fi

# Parse arguments
MODEL=${1:-torch}             # Default to Neural net
EMBEDDING=${2:-bert}          # Default to BERT embeddings
PREDICTION_MODE=${3:-summary} # Default to summary prediction mode
N_TRIALS=${4:-20}             # Default to 20 optimization trials

# Validate arguments
case $MODEL in
rf | ridge | lasso | torch) ;;
*)
    print_warning "Invalid model: $MODEL. Choose from: rf, ridge, lasso, torch"
    exit 1
    ;;
esac

case $EMBEDDING in
tfidf | bert | llama) ;;
*)
    print_warning "Invalid embedding: $EMBEDDING. Choose from: tfidf, bert, llama"
    exit 1
    ;;
esac

case $PREDICTION_MODE in
summary) ;;
*)
    print_warning "Invalid prediction mode: $PREDICTION_MODE. Currently only 'summary' is supported."
    exit 1
    ;;
esac

# Run with hyperparameter optimization
print_header "Running $MODEL model with $EMBEDDING embeddings and $N_TRIALS optimization trials"

# Build command with appropriate options
CMD="python train.py --model $MODEL --embedding $EMBEDDING --prediction_mode $PREDICTION_MODE --optimize_hyperparameters --n_trials $N_TRIALS --load_features"

# Add model-specific arguments
case $MODEL in
torch)
    # For PyTorch, set a reasonable epoch count for final training
    CMD="$CMD --epochs 200"
    ;;
rf)
    # For Random Forest, start with a moderate number of estimators
    CMD="$CMD --n_estimators 100"
    ;;
ridge | lasso)
    # For linear models, start with a default alpha
    CMD="$CMD --alpha 1.0"
    ;;
esac

# Add embedding-specific arguments
case $EMBEDDING in
llama)
    # For LLaMA, use 4-bit quantization for memory efficiency
    CMD="$CMD --use_4bit_llama --use_precomputed_embeddings"
    ;;
esac

# Run the command
echo "Running command: $CMD"
eval $CMD

print_header "Optimization and training completed"
echo "Results are available in the models/saved directory and MLflow"
echo "Run './run.sh mlflow' to view results in the MLflow UI"
