#!/bin/bash

# Run Parameter Sweep - Systematic single-parameter experiments for thesis analysis
# This script runs controlled experiments where only ONE parameter changes at a time
# to isolate the impact of each variable for scientific analysis.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="registry.dumusstbereitsein.de/palladio_approximator"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="parameter_sweep_${TIMESTAMP}.log"

# Default baseline configuration (everything else stays constant)
DEFAULT_MODEL="svm"
DEFAULT_EMBEDDING="tfidf"
DEFAULT_PREDICTION_MODE="summary"
DEFAULT_C="1.0"
DEFAULT_EPSILON="0.1"
DEFAULT_KERNEL="rbf"
DEFAULT_GAMMA="scale"
DEFAULT_DEGREE="3"
DEFAULT_N_ESTIMATORS="10"
DEFAULT_MAX_DEPTH="5"
DEFAULT_ALPHA="1.0"
DEFAULT_BATCH_SIZE="128"
DEFAULT_EPOCHS="100"
DEFAULT_LEARNING_RATE="0.0008"
DEFAULT_DROPOUT_RATE="0.3"
DEFAULT_NORMALIZE_TARGETS="false"

# Experiment tracking
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to show usage
show_usage() {
    echo "Usage: ./run_parameter_sweep.sh --sweep-type <type> --param <parameter> --values <comma-separated-values> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --sweep-type single-param    Single parameter sweep (recommended for thesis)"
    echo "  --param PARAMETER           Parameter to sweep (see list below)"
    echo "  --values VALUE1,VALUE2,...  Comma-separated values to test"
    echo ""
    echo "Optional Arguments:"
    echo "  --baseline-model MODEL      Baseline model type (default: svm)"
    echo "  --baseline-embedding EMB    Baseline embedding (default: tfidf)"
    echo "  --dry-run                   Show what would be executed without running"
    echo "  --parallel N                Run N experiments in parallel (default: 1)"
    echo "  --n-runs N                  Number of times to run each experiment (default: 3)"
    echo ""
    echo "Available Parameters to Sweep:"
    echo ""
    echo "  Model Selection:"
    echo "    model                     Model type: rf,svm,ridge,lasso,torch"
    echo "    embedding                 Embedding type: tfidf,bert,llama"
    echo "    normalize_targets         Target normalization: true,false"
    echo ""
    echo "  SVM Parameters:"
    echo "    C                         Regularization: 0.01,0.1,1.0,10.0,100.0"
    echo "    epsilon                   SVR epsilon: 0.001,0.01,0.1,1.0"
    echo "    kernel                    Kernel type: linear,rbf,poly,sigmoid"
    echo "    gamma                     Kernel coefficient: scale,auto"
    echo "    degree                    Poly kernel degree: 2,3,4,5"
    echo ""
    echo "  Random Forest Parameters:"
    echo "    n_estimators              Number of trees: 5,10,20"
    echo "    max_depth                 Maximum tree depth: 3,5,7"
    echo ""
    echo "  Linear Model Parameters:"
    echo "    alpha                     Regularization: 0.001,0.01,0.1,1.0,10.0,100.0"
    echo ""
    echo "  Neural Network Parameters:"
    echo "    batch_size                Batch size: 32,64,128,256,512"
    echo "    epochs                    Training epochs: 50,100,200,300"
    echo "    learning_rate             Learning rate: 0.0001,0.0005,0.001,0.005,0.01"
    echo "    dropout_rate              Dropout rate: 0.1,0.2,0.3,0.4,0.5"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Impact of SVM C parameter (everything else constant)"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param C --values \"0.1,1.0,10.0,100.0\""
    echo ""
    echo "  # Impact of kernel type on SVM performance"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param kernel --values \"linear,rbf,poly,sigmoid\""
    echo ""
    echo "  # Embedding comparison across models"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param embedding --values \"tfidf,bert,llama\" --baseline-model rf"
    echo ""
    echo "  # Target normalization impact"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param normalize_targets --values \"true,false\""
    echo ""
    echo "  # RF tree count optimization"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param n_estimators --values \"5,10,20\" --baseline-model rf"
    echo ""
    echo "  # RF tree depth optimization"
    echo "  ./run_parameter_sweep.sh --sweep-type single-param --param max_depth --values \"3,5,7\" --baseline-model rf"
    echo ""
}

# Function to build experiment command
build_experiment_command() {
    local model="$1"
    local embedding="$2"
    local normalize_targets="$3"
    local extra_args="$4"

    local cmd="docker run --rm \
        --device=nvidia.com/gpu=all \
        -v \"$(pwd)\":/app \
        -v \"$(pwd)/data\":/app/data \
        -v \"$(pwd)/mlruns\":/app/mlruns \
        -w /app \
        $DOCKER_IMAGE python3 train.py \
        --model $model \
        --embedding $embedding \
        --prediction_mode $DEFAULT_PREDICTION_MODE"

    if [ "$normalize_targets" = "true" ]; then
        cmd="$cmd --normalize_targets"
    fi

    cmd="$cmd $extra_args"

    echo "$cmd"
}

# Function to run single experiment
run_experiment() {
    local experiment_name="$1"
    local command="$2"
    local dry_run="$3"

    print_info "Running experiment: $experiment_name"
    print_info "Command: $command"

    if [ "$dry_run" = "true" ]; then
        print_warning "[DRY RUN] Would execute: $command"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        return 0
    fi

    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

    # Execute the command
    if eval "$command"; then
        print_success "✅ Experiment successful: $experiment_name"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        return 0
    else
        print_error "❌ Experiment failed: $experiment_name"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        return 1
    fi
}

# Function to run single parameter sweep
run_single_param_sweep() {
    local param="$1"
    local values="$2"
    local baseline_model="$3"
    local baseline_embedding="$4"
    local dry_run="$5"
    local n_runs="$6"

    print_info "Starting single parameter sweep for: $param"
    print_info "Baseline model: $baseline_model, embedding: $baseline_embedding"
    print_info "Values to test: $values"
    print_info "Number of runs per experiment: $n_runs"

    # Convert comma-separated values to array
    IFS=',' read -ra VALUE_ARRAY <<<"$values"

    for value in "${VALUE_ARRAY[@]}"; do
        value=$(echo "$value" | xargs) # trim whitespace

        # Set up baseline configuration
        local model="$baseline_model"
        local embedding="$baseline_embedding"
        local normalize_targets="$DEFAULT_NORMALIZE_TARGETS"
        local extra_args=""

        # Modify the specific parameter being swept
        case "$param" in
        "model")
            model="$value"
            ;;
        "embedding")
            embedding="$value"
            ;;
        "normalize_targets")
            normalize_targets="$value"
            ;;
        "C")
            extra_args="--C $value"
            ;;
        "epsilon")
            extra_args="--epsilon $value"
            ;;
        "kernel")
            extra_args="--kernel $value"
            ;;
        "gamma")
            extra_args="--gamma $value"
            ;;
        "degree")
            extra_args="--degree $value"
            ;;
        "n_estimators")
            extra_args="--n_estimators $value"
            ;;
        "max_depth")
            extra_args="--max_depth $value"
            ;;
        "alpha")
            extra_args="--alpha $value"
            ;;
        "batch_size")
            extra_args="--batch_size $value"
            ;;
        "epochs")
            extra_args="--epochs $value"
            ;;
        "learning_rate")
            extra_args="--learning_rate $value"
            ;;
        "dropout_rate")
            extra_args="--dropout_rate $value"
            ;;
        *)
            print_error "Unknown parameter: $param"
            return 1
            ;;
        esac

        # Build experiment name
        local experiment_name="${param}=${value}_model=${model}_emb=${embedding}_norm=${normalize_targets}"

        # Add baseline parameters to extra_args based on model type
        if [ "$param" != "C" ] && [ "$param" != "epsilon" ] && [ "$param" != "kernel" ] && [ "$param" != "gamma" ] && [ "$param" != "degree" ]; then
            case "$model" in
            "svm")
                if [ "$param" != "C" ]; then extra_args="$extra_args --C $DEFAULT_C"; fi
                if [ "$param" != "epsilon" ]; then extra_args="$extra_args --epsilon $DEFAULT_EPSILON"; fi
                if [ "$param" != "kernel" ]; then extra_args="$extra_args --kernel $DEFAULT_KERNEL"; fi
                if [ "$param" != "gamma" ]; then extra_args="$extra_args --gamma $DEFAULT_GAMMA"; fi
                if [ "$param" != "degree" ]; then extra_args="$extra_args --degree $DEFAULT_DEGREE"; fi
                ;;
            "rf")
                if [ "$param" != "n_estimators" ]; then extra_args="$extra_args --n_estimators $DEFAULT_N_ESTIMATORS"; fi
                if [ "$param" != "max_depth" ]; then extra_args="$extra_args --max_depth $DEFAULT_MAX_DEPTH"; fi
                ;;
            "ridge" | "lasso")
                if [ "$param" != "alpha" ]; then extra_args="$extra_args --alpha $DEFAULT_ALPHA"; fi
                ;;
            "torch")
                if [ "$param" != "batch_size" ]; then extra_args="$extra_args --batch_size $DEFAULT_BATCH_SIZE"; fi
                if [ "$param" != "epochs" ]; then extra_args="$extra_args --epochs $DEFAULT_EPOCHS"; fi
                if [ "$param" != "learning_rate" ]; then extra_args="$extra_args --learning_rate $DEFAULT_LEARNING_RATE"; fi
                if [ "$param" != "dropout_rate" ]; then extra_args="$extra_args --dropout_rate $DEFAULT_DROPOUT_RATE"; fi
                ;;
            esac
        fi

        # Run experiment multiple times for statistical reliability
        for run_num in $(seq 1 $n_runs); do
            local run_experiment_name="${experiment_name}_run${run_num}"
            local command=$(build_experiment_command "$model" "$embedding" "$normalize_targets" "$extra_args")

            print_info "Running experiment $run_num/$n_runs for parameter combination: ${param}=${value}"
            run_experiment "$run_experiment_name" "$command" "$dry_run"

            # Add small delay between runs
            if [ "$dry_run" != "true" ]; then
                sleep 2
            fi
        done
    done
}

# Parse command line arguments
SWEEP_TYPE=""
PARAM=""
VALUES=""
BASELINE_MODEL="$DEFAULT_MODEL"
BASELINE_EMBEDDING="$DEFAULT_EMBEDDING"
DRY_RUN="false"
PARALLEL="1"
N_RUNS="3"

while [[ $# -gt 0 ]]; do
    case $1 in
    --sweep-type)
        SWEEP_TYPE="$2"
        shift 2
        ;;
    --param)
        PARAM="$2"
        shift 2
        ;;
    --values)
        VALUES="$2"
        shift 2
        ;;
    --baseline-model)
        BASELINE_MODEL="$2"
        shift 2
        ;;
    --baseline-embedding)
        BASELINE_EMBEDDING="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN="true"
        shift
        ;;
    --parallel)
        PARALLEL="$2"
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
if [ -z "$SWEEP_TYPE" ] || [ -z "$PARAM" ] || [ -z "$VALUES" ]; then
    echo "Error: Missing required arguments"
    show_usage
    exit 1
fi

if [ "$SWEEP_TYPE" != "single-param" ]; then
    echo "Error: Only 'single-param' sweep type is currently supported"
    exit 1
fi

# Start sweep
START_TIME=$(date +%s)

print_info "========================================="
print_info "Parameter Sweep Started"
print_info "========================================="
print_info "Sweep type: $SWEEP_TYPE"
print_info "Parameter: $PARAM"
print_info "Values: $VALUES"
print_info "Baseline model: $BASELINE_MODEL"
print_info "Baseline embedding: $BASELINE_EMBEDDING"
print_info "Dry run: $DRY_RUN"
print_info "Log file: $LOG_FILE"
print_info "Started at: $(date)"
print_info "========================================="

# Run the sweep
case "$SWEEP_TYPE" in
"single-param")
    run_single_param_sweep "$PARAM" "$VALUES" "$BASELINE_MODEL" "$BASELINE_EMBEDDING" "$DRY_RUN" "$N_RUNS"
    ;;
*)
    print_error "Unsupported sweep type: $SWEEP_TYPE"
    exit 1
    ;;
esac

# Calculate execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Summary
print_info "========================================="
print_info "Parameter Sweep Completed"
print_info "========================================="
print_info "Total experiments: $TOTAL_EXPERIMENTS"
print_success "Successful: $SUCCESSFUL_EXPERIMENTS"
if [ "$FAILED_EXPERIMENTS" -gt 0 ]; then
    print_error "Failed: $FAILED_EXPERIMENTS"
else
    print_info "Failed: $FAILED_EXPERIMENTS"
fi
print_info "Execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
print_info "Log file: $LOG_FILE"
print_info "========================================="

# Exit with appropriate code
if [ "$FAILED_EXPERIMENTS" -gt 0 ]; then
    exit 1
else
    exit 0
fi
