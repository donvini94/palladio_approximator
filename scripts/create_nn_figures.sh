#!/bin/bash

# Create Thesis Figures - Neural Network Hyperparameter Analysis
# This script generates all the consolidated plots for NN hyperparameter analysis

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
OUTPUT_DIR="thesis_figures/nn_analysis"
METRICS="metrics.test_mse_avg metrics.test_mae_avg metrics.test_r2_avg metrics.test_rmse_avg metrics.val_mse_avg metrics.val_mae_avg metrics.val_r2_avg"
EXPERIMENT_NAME=""

# Function to show usage
show_usage() {
    echo "Usage: ./create_nn_figures.sh [options]"
    echo ""
    echo "Generate thesis-quality consolidated plots for Neural Network hyperparameter analysis"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR      Output directory for figures (default: thesis_figures/nn_analysis)"
    echo "  --metrics METRICS     Space-separated list of metrics to analyze"
    echo "                        (default: metrics.test_mse_avg metrics.test_mae_avg metrics.test_r2_avg etc.)"
    echo "  --experiment NAME     Specific MLflow experiment to analyze (optional)"
    echo "  --test-metrics        Use only test metrics (recommended for thesis)"
    echo "  --val-metrics         Use only validation metrics"
    echo "  --all-metrics         Use all available metrics (default)"
    echo ""
    echo "Note: Each plot shows both normalized and non-normalized results for comparison"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./create_nn_figures.sh"
    echo "  ./create_nn_figures.sh --test-metrics"
    echo "  ./create_nn_figures.sh --output-dir my_figures --experiment 'NN Analysis'"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --test-metrics)
            METRICS="metrics.test_mse_avg metrics.test_mae_avg metrics.test_r2_avg metrics.test_rmse_avg"
            shift
            ;;
        --val-metrics)
            METRICS="metrics.val_mse_avg metrics.val_mae_avg metrics.val_r2_avg metrics.val_rmse_avg"
            shift
            ;;
        --all-metrics)
            METRICS="metrics.test_mse_avg metrics.test_mae_avg metrics.test_r2_avg metrics.test_rmse_avg metrics.val_mse_avg metrics.val_mae_avg metrics.val_r2_avg metrics.val_rmse_avg"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "generate_nn_figures.py" ]; then
    print_error "generate_nn_figures.py not found!"
    exit 1
fi

# Check if MLflow data exists
if [ ! -d "mlruns" ]; then
    print_warning "mlruns directory not found. Make sure you have run Neural Network experiments first."
fi

print_info "==========================================="
print_info "Creating Thesis Figures - NN Analysis"
print_info "==========================================="
print_info "Output directory: $OUTPUT_DIR"
print_info "Metrics: $METRICS"
if [ -n "$EXPERIMENT_NAME" ]; then
    print_info "Experiment: $EXPERIMENT_NAME"
fi
print_info "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python3 generate_nn_figures.py --output-dir \"$OUTPUT_DIR\""

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment-name \"$EXPERIMENT_NAME\""
fi

CMD="$CMD --metrics $METRICS"

# Run the figure generation
print_info "Starting figure generation..."
START_TIME=$(date +%s)

if eval "$CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "✅ Figure generation completed successfully!"
    print_info "Duration: ${DURATION} seconds"
    print_info "Figures saved to: $OUTPUT_DIR"
    
    # Count generated figures
    if [ -d "$OUTPUT_DIR" ]; then
        FIGURE_COUNT=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
        print_success "Generated $FIGURE_COUNT figures"
        
        if [ -f "$OUTPUT_DIR/nn_summary_statistics.csv" ]; then
            print_success "Summary statistics saved to nn_summary_statistics.csv"
        fi
    fi
    
    print_info ""
    print_info "Next steps for thesis:"
    print_info "1. Review consolidated figures in $OUTPUT_DIR"
    print_info "2. Organize figures by hyperparameter (learning_rate, dropout_rate, batch_size, epochs)"
    print_info "3. Create sections like 'Effect of Learning Rate'"
    print_info "4. Each figure shows normalization comparison with different colors"
    print_info "5. Compare across embedding types (TF-IDF, BERT) in separate figures"
    
else
    print_error "❌ Figure generation failed!"
    exit 1
fi