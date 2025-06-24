#!/bin/bash
# Script to run a comprehensive hyperparameter optimization and model evaluation for cluster execution
# This script runs multiple experiments in sequence with hyperparameter optimization enabled

# Ensure we are in the script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up color formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
N_TRIALS=50  # More trials for thorough optimization on a cluster
OPTIMIZE_MODE="all"  # Options: all, fast, test
USE_MLFLOW="--use_mlflow"  # Default to using MLflow
USE_GPU="--use_gpu"
OUTPUT_DIR="cluster_experiments_$(date +%Y%m%d_%H%M%S)"

# Function to print usage information
function print_usage() {
    echo "Usage: ./run_cluster_experiments.sh [options]"
    echo "Options:"
    echo "  --trials N           Number of optimization trials per model (default: 50)"
    echo "  --mode MODE          Optimization mode: all, fast, test (default: all)"
    echo "  --no-mlflow          Disable MLflow logging"
    echo "  --no-gpu             Disable GPU usage"
    echo "  --output-dir DIR     Output directory for logs (default: cluster_experiments_TIMESTAMP)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_cluster_experiments.sh --trials 100"
    echo "  ./run_cluster_experiments.sh --mode fast"
    echo "  ./run_cluster_experiments.sh --no-mlflow --no-gpu"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --mode)
            OPTIMIZE_MODE="$2"
            shift 2
            ;;
        --no-mlflow)
            USE_MLFLOW="--no_mlflow"
            shift
            ;;
        --no-gpu)
            USE_GPU=""
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
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

# Validate optimization mode
case $OPTIMIZE_MODE in
    all|fast|test)
        # Valid modes
        ;;
    *)
        echo -e "${YELLOW}Invalid optimization mode: $OPTIMIZE_MODE${NC}"
        print_usage
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
SUMMARY_FILE="$OUTPUT_DIR/experiment_summary.md"

# Log start time and configuration
echo -e "${GREEN}Starting cluster experiments at $(date)${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}Configuration:${NC}" | tee -a "$LOG_FILE"
echo -e "  Optimization Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo -e "  Optimization Mode: $OPTIMIZE_MODE" | tee -a "$LOG_FILE"
echo -e "  Using MLflow: ${USE_MLFLOW/--no_mlflow/No}" | tee -a "$LOG_FILE"
echo -e "  Using GPU: ${USE_GPU:+Yes}" | tee -a "$LOG_FILE"
echo -e "  Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"

# Create experiment summary header
cat > "$SUMMARY_FILE" << EOF
# Cluster Experiment Results

## Configuration
- **Optimization Trials**: $N_TRIALS
- **Optimization Mode**: $OPTIMIZE_MODE
- **Using MLflow**: ${USE_MLFLOW/--no_mlflow/No}
- **Using GPU**: ${USE_GPU:+Yes}
- **Started**: $(date)

## Experiment Results

| Model | Embedding | Optimization Time | Training Time | Val MSE | Val MAE | Val R² | Status |
|-------|-----------|-------------------|---------------|---------|---------|--------|--------|
EOF

# Set up experiment combinations based on mode
if [[ "$OPTIMIZE_MODE" == "test" ]]; then
    # Testing mode - just one quick experiment
    MODELS=("rf")
    EMBEDDINGS=("bert")
    PREDICTION_MODES=("summary")
    N_TRIALS=5  # Override to a small number for testing
elif [[ "$OPTIMIZE_MODE" == "fast" ]]; then
    # Fast mode - limited set of experiments
    MODELS=("rf" "torch")
    EMBEDDINGS=("bert" "llama")
    PREDICTION_MODES=("summary")
elif [[ "$OPTIMIZE_MODE" == "all" ]]; then
    # All mode - comprehensive experiments
    MODELS=("rf" "ridge" "lasso" "torch")
    EMBEDDINGS=("tfidf" "bert" "llama")
    PREDICTION_MODES=("summary")
else
    echo "Unknown mode: $OPTIMIZE_MODE"
    exit 1
fi

# Track total execution time
TOTAL_START_TIME=$(date +%s)

# Run experiments
TOTAL_EXPERIMENTS=${#MODELS[@]}*${#EMBEDDINGS[@]}*${#PREDICTION_MODES[@]}
COMPLETED=0
SUCCEEDED=0

for PREDICTION_MODE in "${PREDICTION_MODES[@]}"; do
    for EMBEDDING in "${EMBEDDINGS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            # Skip torch model with tfidf for memory constraints
            if [[ "$MODEL" == "torch" && "$EMBEDDING" == "tfidf" ]]; then
                echo -e "${YELLOW}Skipping torch + tfidf combination due to memory constraints${NC}" | tee -a "$LOG_FILE"
                COMPLETED=$((COMPLETED + 1))
                continue
            fi

            # Experiment identifier
            EXPERIMENT_ID="${MODEL}_${EMBEDDING}_${PREDICTION_MODE}"
            EXPERIMENT_LOG="$OUTPUT_DIR/${EXPERIMENT_ID}.log"
            
            echo -e "\n${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
            echo -e "${GREEN}Starting experiment ($COMPLETED/$TOTAL_EXPERIMENTS): $EXPERIMENT_ID${NC}" | tee -a "$LOG_FILE"
            echo -e "${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
            
            EXPERIMENT_START_TIME=$(date +%s)
            
            # Construct command with appropriate options
            CMD="python train.py --model $MODEL --embedding $EMBEDDING --prediction_mode $PREDICTION_MODE"
            CMD="$CMD --optimize_hyperparameters --n_trials $N_TRIALS --optimization_metric val_mse"
            CMD="$CMD $USE_MLFLOW $USE_GPU"
            
            # Add model-specific arguments for better starting points
            case $MODEL in
                rf)
                    CMD="$CMD --n_estimators 100"
                    ;;
                ridge|lasso)
                    CMD="$CMD --alpha 1.0"
                    ;;
                torch)
                    CMD="$CMD --batch_size 64 --epochs 200"
                    ;;
            esac
            
            # Add embedding-specific arguments
            case $EMBEDDING in
                llama)
                    CMD="$CMD --use_4bit_llama --use_precomputed_embeddings"
                    ;;
            esac
            
            # Run experiment and log results
            echo -e "${BLUE}Running command: $CMD${NC}" | tee -a "$LOG_FILE"
            
            # Execute the command
            START_TIME=$(date +%s)
            $CMD > "$EXPERIMENT_LOG" 2>&1
            RESULT=$?
            END_TIME=$(date +%s)
            
            # Calculate duration
            OPTIMIZATION_DURATION=$((END_TIME - START_TIME))
            HOURS=$((OPTIMIZATION_DURATION / 3600))
            MINUTES=$(((OPTIMIZATION_DURATION % 3600) / 60))
            SECONDS=$((OPTIMIZATION_DURATION % 60))
            DURATION_STR="${HOURS}h ${MINUTES}m ${SECONDS}s"
            
            # Determine if experiment succeeded
            if [ $RESULT -eq 0 ]; then
                STATUS="✅ Success"
                SUCCEEDED=$((SUCCEEDED + 1))
                
                # Extract metrics from log file (would be better with direct MLflow API in a production setting)
                VAL_MSE=$(grep -o "val_mse.*" "$EXPERIMENT_LOG" | tail -1 | sed 's/.*: //')
                VAL_MAE=$(grep -o "val_mae.*" "$EXPERIMENT_LOG" | tail -1 | sed 's/.*: //')
                VAL_R2=$(grep -o "val_r2.*" "$EXPERIMENT_LOG" | tail -1 | sed 's/.*: //')
                
                # Get training time separate from optimization time
                TRAIN_TIME=$(grep -o "Training completed in.*" "$EXPERIMENT_LOG" | tail -1 | sed 's/.*in //')
                
                echo -e "${GREEN}✅ Experiment successful: $EXPERIMENT_ID${NC}" | tee -a "$LOG_FILE"
                echo -e "   Optimization Time: $DURATION_STR" | tee -a "$LOG_FILE"
                echo -e "   Training Time: ${TRAIN_TIME:-unknown}" | tee -a "$LOG_FILE"
                echo -e "   Validation MSE: ${VAL_MSE:-N/A}" | tee -a "$LOG_FILE"
                echo -e "   Validation MAE: ${VAL_MAE:-N/A}" | tee -a "$LOG_FILE"
                echo -e "   Validation R²: ${VAL_R2:-N/A}" | tee -a "$LOG_FILE"
            else
                STATUS="❌ Failed"
                VAL_MSE="N/A"
                VAL_MAE="N/A" 
                VAL_R2="N/A"
                TRAIN_TIME="N/A"
                
                echo -e "${YELLOW}❌ Experiment failed: $EXPERIMENT_ID${NC}" | tee -a "$LOG_FILE"
                echo -e "   Check log file for details: $EXPERIMENT_LOG" | tee -a "$LOG_FILE"
            fi
            
            # Add result to summary markdown
            echo "| $MODEL | $EMBEDDING | $DURATION_STR | ${TRAIN_TIME:-N/A} | ${VAL_MSE:-N/A} | ${VAL_MAE:-N/A} | ${VAL_R2:-N/A} | $STATUS |" >> "$SUMMARY_FILE"
            
            COMPLETED=$((COMPLETED + 1))
            
            # Clean up if using GPU to prevent memory issues
            if [[ -n "$USE_GPU" ]]; then
                echo -e "${BLUE}Cleaning up GPU memory...${NC}" | tee -a "$LOG_FILE"
                # On some systems you might need additional cleanup commands here
                # for CUDA like: nvidia-smi --gpu-reset or similar
            fi
        done
    done
done

# Calculate total execution time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Add summary to markdown file
cat >> "$SUMMARY_FILE" << EOF

## Summary
- **Total Experiments**: $TOTAL_EXPERIMENTS
- **Successful**: $SUCCEEDED
- **Failed**: $((TOTAL_EXPERIMENTS - SUCCEEDED))
- **Total Execution Time**: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s
- **Completed**: $(date)

EOF

# Final summary
echo -e "\n${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}Cluster experiments completed!${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "Total experiments: $TOTAL_EXPERIMENTS" | tee -a "$LOG_FILE"
echo -e "Successful: $SUCCEEDED" | tee -a "$LOG_FILE"
echo -e "Failed: $((TOTAL_EXPERIMENTS - SUCCEEDED))" | tee -a "$LOG_FILE"
echo -e "Total execution time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
echo -e "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo -e "Summary file: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo -e "${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"

# Run MLflow summarization if MLflow was used
if [[ "$USE_MLFLOW" != "--no_mlflow" ]]; then
    echo -e "${BLUE}Generating MLflow experiment summary...${NC}" | tee -a "$LOG_FILE"
    python scripts/summarize_experiments.py --format markdown --sort val_r2 > "$OUTPUT_DIR/mlflow_summary.md"
fi

echo -e "${GREEN}All tasks completed! Check $OUTPUT_DIR for all results.${NC}"