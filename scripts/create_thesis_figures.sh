#!/bin/bash

# Create Thesis Figures - Hyperparameter Analysis
# This script generates all the plots for SVM, Neural Network, and Linear model hyperparameter analysis

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
MODEL_TYPE="svm"  # Default model type
OUTPUT_DIR="thesis_figures/svm_analysis"
METRICS="metrics.test_mse_avg metrics.test_mae_avg metrics.test_r2_avg metrics.test_rmse_avg metrics.val_mse_avg metrics.val_mae_avg metrics.val_r2_avg"
EXPERIMENT_NAME=""

# Function to show usage
show_usage() {
    echo "Usage: ./create_thesis_figures.sh [options]"
    echo ""
    echo "Generate thesis-quality consolidated plots for hyperparameter analysis"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE     Model type to analyze: svm, nn, linear, rf, or all (default: svm)"
    echo "  --output-dir DIR      Output directory for figures (default: thesis_figures/{model_type}_analysis)"
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
    echo "  ./create_thesis_figures.sh --model-type svm"
    echo "  ./create_thesis_figures.sh --model-type nn --test-metrics"
    echo "  ./create_thesis_figures.sh --model-type linear --output-dir my_figures"
    echo "  ./create_thesis_figures.sh --model-type rf"
    echo "  ./create_thesis_figures.sh --model-type all"
    echo ""
}

# Parse command line arguments
OUTPUT_DIR_SET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            OUTPUT_DIR_SET=true
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

# Set default output directory if not specified
if [ "$OUTPUT_DIR_SET" = false ]; then
    case $MODEL_TYPE in
        svm)
            OUTPUT_DIR="thesis_figures/svm_analysis"
            ;;
        nn)
            OUTPUT_DIR="thesis_figures/nn_analysis"
            ;;
        linear)
            OUTPUT_DIR="thesis_figures/linear_analysis"
            ;;
        rf)
            OUTPUT_DIR="thesis_figures/rf_analysis"
            ;;
        all)
            OUTPUT_DIR="thesis_figures"
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            echo "Valid options: svm, nn, linear, rf, all"
            exit 1
            ;;
    esac
fi

# Function to run figure generation for a specific model type
run_figure_generation() {
    local model_type=$1
    local output_subdir=$2
    
    case $model_type in
        svm)
            script_name="generate_svm_figures.py"
            ;;
        nn)
            script_name="generate_nn_figures.py"
            ;;
        linear)
            script_name="generate_linear_figures.py"
            ;;
        rf)
            script_name="generate_rf_figures.py"
            ;;
        *)
            print_error "Unknown model type: $model_type"
            return 1
            ;;
    esac
    
    # Check if Python script exists
    if [ ! -f "scripts/$script_name" ]; then
        print_error "scripts/$script_name not found!"
        return 1
    fi
    
    print_info "----------------------------------------"
    print_info "Creating figures for $model_type model"
    print_info "----------------------------------------"
    print_info "Script: $script_name"
    print_info "Output directory: $output_subdir"
    print_info "Metrics: $METRICS"
    if [ -n "$EXPERIMENT_NAME" ]; then
        print_info "Experiment: $EXPERIMENT_NAME"
    fi
    print_info "----------------------------------------"
    
    # Create output subdirectory
    mkdir -p "$output_subdir"
    
    # Build the command
    CMD="python3 scripts/$script_name --output-dir \"$output_subdir\""
    
    if [ -n "$EXPERIMENT_NAME" ]; then
        CMD="$CMD --experiment-name \"$EXPERIMENT_NAME\""
    fi
    
    CMD="$CMD --metrics $METRICS"
    
    # Run the figure generation
    print_info "Starting $model_type figure generation..."
    START_TIME=$(date +%s)
    
    if eval "$CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        print_success "✅ $model_type figure generation completed successfully!"
        print_info "Duration: ${DURATION} seconds"
        
        # Count generated figures
        if [ -d "$output_subdir" ]; then
            FIGURE_COUNT=$(find "$output_subdir" -name "*.png" | wc -l)
            print_success "Generated $FIGURE_COUNT figures"
            
            if [ -f "$output_subdir/${model_type}_summary_statistics.csv" ]; then
                print_success "Summary statistics saved to ${model_type}_summary_statistics.csv"
            fi
        fi
        
        return 0
    else
        print_error "❌ $model_type figure generation failed!"
        return 1
    fi
}

# Check if MLflow data exists
if [ ! -d "mlruns" ]; then
    print_warning "mlruns directory not found. Make sure you have run experiments first."
fi

print_info "========================================="
print_info "Creating Thesis Figures - Model Analysis"
print_info "========================================="
print_info "Model type(s): $MODEL_TYPE"
print_info "Base output directory: $OUTPUT_DIR"
print_info "========================================="

# Run figure generation based on model type
case $MODEL_TYPE in
    svm|nn|linear|rf)
        if run_figure_generation "$MODEL_TYPE" "$OUTPUT_DIR"; then
            print_success "✅ All figure generation completed successfully!"
            
            print_info ""
            print_info "Next steps for thesis:"
            print_info "1. Review consolidated figures in $OUTPUT_DIR"
            case $MODEL_TYPE in
                svm)
                    print_info "2. Organize figures by hyperparameter (C, kernel, epsilon, gamma)"
                    print_info "3. Create sections like 'Effect of Regularization Parameter C'"
                    ;;
                nn)
                    print_info "2. Organize figures by hyperparameter (learning_rate, dropout_rate, batch_size, epochs)"
                    print_info "3. Create sections like 'Effect of Learning Rate'"
                    ;;
                linear)
                    print_info "2. Organize figures by hyperparameter (alpha, model_type)"
                    print_info "3. Create sections like 'Effect of Regularization Parameter Alpha'"
                    ;;
                rf)
                    print_info "2. Organize figures by hyperparameter (n_estimators, max_depth)"
                    print_info "3. Create sections like 'Effect of Number of Trees'"
                    ;;
            esac
            print_info "4. Each figure shows normalization comparison with different colors"
            print_info "5. Compare across embedding types (TF-IDF, BERT) in separate figures"
        else
            exit 1
        fi
        ;;
    all)
        # Generate figures for all model types
        all_success=true
        for model in svm nn linear rf; do
            model_output_dir="$OUTPUT_DIR/${model}_analysis"
            if ! run_figure_generation "$model" "$model_output_dir"; then
                all_success=false
            fi
        done
        
        if [ "$all_success" = true ]; then
            print_success "✅ All figure generation completed successfully!"
            
            print_info ""
            print_info "Next steps for thesis:"
            print_info "1. Review consolidated figures in $OUTPUT_DIR"
            print_info "   - SVM figures: $OUTPUT_DIR/svm_analysis"
            print_info "   - Neural Network figures: $OUTPUT_DIR/nn_analysis" 
            print_info "   - Linear model figures: $OUTPUT_DIR/linear_analysis"
            print_info "   - Random Forest figures: $OUTPUT_DIR/rf_analysis"
            print_info "2. Organize by model type and hyperparameters"
            print_info "3. Each figure shows normalization comparison with different colors"
            print_info "4. Compare across embedding types in separate figures"
        else
            print_error "❌ Some figure generation failed!"
            exit 1
        fi
        ;;
    *)
        print_error "Unknown model type: $MODEL_TYPE"
        echo "Valid options: svm, nn, linear, all"
        exit 1
        ;;
esac