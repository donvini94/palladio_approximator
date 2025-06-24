#!/bin/bash

# Thesis Experiments - Predefined parameter sweeps for systematic analysis
# This script runs curated experiment sets perfect for thesis sections

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SWEEP_SCRIPT="bash scripts/run_parameter_sweep.sh"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to print colored output
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

# Function to show usage
show_usage() {
    echo "Usage: ./run_thesis_experiments.sh --experiment-set <set> [options]"
    echo ""
    echo "Experiment Sets:"
    echo ""
    echo "  svm-analysis              Complete SVM hyperparameter analysis"
    echo "    ├── svm-regularization    Impact of C parameter"
    echo "    ├── svm-kernels          Kernel comparison"
    echo "    ├── svm-epsilon          SVR epsilon optimization"
    echo "    └── svm-gamma            RBF kernel gamma tuning"
    echo ""
    echo "  model-comparison          Model architecture comparison"
    echo "    ├── model-types          RF vs SVM vs Neural vs Linear"
    echo "    └── embedding-impact     TF-IDF vs BERT vs LLaMA"
    echo ""
    echo "  preprocessing-analysis    Data preprocessing impact"
    echo "    └── normalization-impact Target normalization effects"
    echo ""
    echo "  rf-optimization          Random Forest optimization"
    echo "    ├── rf-trees             Number of estimators tuning"
    echo "    └── rf-depth             Tree depth optimization"
    echo ""
    echo "  linear-analysis          Linear model analysis"
    echo "    └── linear-regularization Ridge/Lasso alpha tuning"
    echo ""
    echo "  neural-optimization      Neural network optimization"
    echo "    ├── nn-batch-size        Batch size impact"
    echo "    ├── nn-epochs            Training duration analysis"
    echo "    ├── nn-learning-rate     Learning rate optimization"
    echo "    └── nn-dropout           Dropout regularization analysis"
    echo ""
    echo "  complete-analysis        All experiments (long running!)"
    echo ""
    echo "Options:"
    echo "  --dry-run                Show what would be executed"
    echo "  --baseline-model MODEL   Override baseline model"
    echo "  --baseline-embedding EMB Override baseline embedding"
    echo "  --n-runs N               Number of times to run each experiment (default: 3)"
    echo ""
    echo "Examples:"
    echo "  ./run_thesis_experiments.sh --experiment-set svm-analysis"
    echo "  ./run_thesis_experiments.sh --experiment-set model-comparison --dry-run"
    echo "  ./run_thesis_experiments.sh --experiment-set normalization-impact"
    echo ""
}

# Function to run experiment set
run_experiment_set() {
    local set_name="$1"
    local dry_run="$2"
    local baseline_model="$3"
    local baseline_embedding="$4"
    local n_runs="$5"

    local dry_run_flag=""
    if [ "$dry_run" = "true" ]; then
        dry_run_flag="--dry-run"
    fi

    local baseline_args=""
    if [ -n "$baseline_model" ]; then
        baseline_args="$baseline_args --baseline-model $baseline_model"
    fi
    if [ -n "$baseline_embedding" ]; then
        baseline_args="$baseline_args --baseline-embedding $baseline_embedding"
    fi

    local n_runs_flag="--n-runs $n_runs"

    case "$set_name" in
    "svm-regularization")
        print_info "Running SVM regularization parameter (C) analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param C --values "0.01,0.1,1.0,10.0,100.0" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "svm-kernels")
        print_info "Running SVM kernel comparison analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param kernel --values "linear,rbf,poly,sigmoid" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "svm-epsilon")
        print_info "Running SVM epsilon parameter analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param epsilon --values "0.001,0.01,0.1,1.0" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "svm-gamma")
        print_info "Running SVM gamma parameter analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param gamma --values "scale,auto" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "svm-analysis")
        print_info "Running complete SVM analysis (4 parameter sweeps)..."
        run_experiment_set "svm-regularization" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "svm-kernels" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "svm-epsilon" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "svm-gamma" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        ;;

    "model-types")
        print_info "Running model architecture comparison..."
        $SWEEP_SCRIPT --sweep-type single-param --param model --values "rf,svm,ridge,lasso,torch" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "embedding-impact")
        print_info "Running embedding comparison analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param embedding --values "tfidf,bert,llama" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "model-comparison")
        print_info "Running complete model comparison analysis..."
        run_experiment_set "model-types" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "embedding-impact" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        ;;

    "normalization-impact")
        print_info "Running target normalization impact analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param normalize_targets --values "true,false" $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "preprocessing-analysis")
        print_info "Running preprocessing impact analysis..."
        run_experiment_set "normalization-impact" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        ;;

    "rf-trees")
        print_info "Running Random Forest trees optimization..."
        $SWEEP_SCRIPT --sweep-type single-param --param n_estimators --values "5,10,20" --baseline-model rf $baseline_args $dry_run_flag $n_runs_flag
        ;;
        
    "rf-depth")
        print_info "Running Random Forest depth optimization..."
        $SWEEP_SCRIPT --sweep-type single-param --param max_depth --values "3,5,7" --baseline-model rf $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "rf-optimization")
        print_info "Running Random Forest optimization..."
        run_experiment_set "rf-trees" "$dry_run" "rf" "$baseline_embedding" "$n_runs"
        run_experiment_set "rf-depth" "$dry_run" "rf" "$baseline_embedding" "$n_runs"
        ;;

    "linear-regularization")
        print_info "Running linear model regularization analysis..."
        print_info "Testing Ridge regression..."
        $SWEEP_SCRIPT --sweep-type single-param --param alpha --values "0.001,0.01,0.1,1.0,10.0,100.0" --baseline-model ridge $baseline_args $dry_run_flag $n_runs_flag
        print_info "Testing Lasso regression..."
        $SWEEP_SCRIPT --sweep-type single-param --param alpha --values "0.001,0.01,0.1,1.0,10.0,100.0" --baseline-model lasso $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "linear-analysis")
        print_info "Running linear model analysis..."
        run_experiment_set "linear-regularization" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        ;;

    "nn-batch-size")
        print_info "Running neural network batch size analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param batch_size --values "32,64,128,256,512" --baseline-model torch $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "nn-epochs")
        print_info "Running neural network training duration analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param epochs --values "50,100,200,300" --baseline-model torch $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "nn-learning-rate")
        print_info "Running neural network learning rate analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param learning_rate --values "0.0001,0.0005,0.001,0.005,0.01" --baseline-model torch $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "nn-dropout")
        print_info "Running neural network dropout analysis..."
        $SWEEP_SCRIPT --sweep-type single-param --param dropout_rate --values "0.1,0.2,0.3,0.4,0.5" --baseline-model torch $baseline_args $dry_run_flag $n_runs_flag
        ;;

    "neural-optimization")
        print_info "Running neural network optimization..."
        run_experiment_set "nn-batch-size" "$dry_run" "torch" "$baseline_embedding" "$n_runs"
        run_experiment_set "nn-epochs" "$dry_run" "torch" "$baseline_embedding" "$n_runs"
        run_experiment_set "nn-learning-rate" "$dry_run" "torch" "$baseline_embedding" "$n_runs"
        run_experiment_set "nn-dropout" "$dry_run" "torch" "$baseline_embedding" "$n_runs"
        ;;

    "complete-analysis")
        print_info "Running complete thesis analysis (WARNING: This will take hours!)..."
        run_experiment_set "svm-analysis" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "model-comparison" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "preprocessing-analysis" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "rf-optimization" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "linear-analysis" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        run_experiment_set "neural-optimization" "$dry_run" "$baseline_model" "$baseline_embedding" "$n_runs"
        ;;

    *)
        print_error "Unknown experiment set: $set_name"
        show_usage
        exit 1
        ;;
    esac
}

# Parse command line arguments
EXPERIMENT_SET=""
DRY_RUN="false"
BASELINE_MODEL=""
BASELINE_EMBEDDING=""
N_RUNS="3"

while [[ $# -gt 0 ]]; do
    case $1 in
    --experiment-set)
        EXPERIMENT_SET="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN="true"
        shift
        ;;
    --baseline-model)
        BASELINE_MODEL="$2"
        shift 2
        ;;
    --baseline-embedding)
        BASELINE_EMBEDDING="$2"
        shift 2
        ;;
    --n-runs)
        N_RUNS="$2"
        shift 2
        ;;
    -h | --help)
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

# Validate required arguments
if [ -z "$EXPERIMENT_SET" ]; then
    echo "Error: Missing required --experiment-set argument"
    show_usage
    exit 1
fi

# Check if sweep script exists
if [ ! -f "scripts/run_parameter_sweep.sh" ]; then
    print_error "Parameter sweep script not found: scripts/run_parameter_sweep.sh"
    exit 1
fi

# Start experiments
START_TIME=$(date +%s)

print_info "========================================="
print_info "Thesis Experiments Started"
print_info "========================================="
print_info "Experiment set: $EXPERIMENT_SET"
print_info "Dry run: $DRY_RUN"
if [ -n "$BASELINE_MODEL" ]; then
    print_info "Baseline model: $BASELINE_MODEL"
fi
if [ -n "$BASELINE_EMBEDDING" ]; then
    print_info "Baseline embedding: $BASELINE_EMBEDDING"
fi
print_info "Started at: $(date)"
print_info "========================================="

# Run experiments
run_experiment_set "$EXPERIMENT_SET" "$DRY_RUN" "$BASELINE_MODEL" "$BASELINE_EMBEDDING" "$N_RUNS"

# Calculate execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Summary
print_info "========================================="
print_success "Thesis Experiments Completed"
print_info "========================================="
print_info "Experiment set: $EXPERIMENT_SET"
print_info "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
print_info "Completed at: $(date)"
print_info "========================================="

print_info "Next steps:"
print_info "1. Run: scripts/run.sh summarize --format both"
print_info "2. Check: scripts/run.sh mlflow (for detailed analysis)"
print_info "3. Generate figures with visualization scripts"

exit 0
