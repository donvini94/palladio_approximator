"""
Main training script for Palladio Approximator.

This script orchestrates the entire training pipeline:
1. Parsing command line arguments
2. Loading or generating embeddings/features
3. Training a model (random forest, linear, or neural network)
4. Evaluating the model on validation and test sets
5. Saving the model and tracking the experiment with MLflow
"""

import sys
import time
import traceback
import uuid
import psutil
import socket
from datetime import datetime

# Load utilities
from utils.config import (
    parse_args,
    setup_environment,
    get_checkpoint_paths,
    get_model_path,
)
from utils.feature_manager import get_features
from utils.model_trainer import train_model, save_model, save_embedding_model
from utils.mlflow_logger import (
    setup_mlflow,
    log_common_parameters,
    log_torch_model_metrics,
    log_evaluation_results,
    end_mlflow_run,
)


def log_system_info(args):
    """Log system information for reproducibility"""
    print("=== System Information ===")
    print(f"Hostname: {socket.gethostname()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=True)}")

    # GPU information if available
    if args.use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(
                    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
                )
                print(f"CUDA Version: {torch.version.cuda}")
            else:
                print("GPU: Not available or not enabled")
        except ImportError:
            print("PyTorch not available, GPU info cannot be displayed")

    # Memory information
    mem_info = psutil.virtual_memory()
    print(f"Total Memory: {mem_info.total / (1024**3):.2f} GB")
    print(f"Available Memory: {mem_info.available / (1024**3):.2f} GB")
    print("=========================")


def get_experiment_id(args):
    """Generate a unique experiment ID that combines hyperparameters and timestamp.

    Args:
        args: Command line arguments

    Returns:
        str: Unique experiment ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a base experiment name from the key parameters
    base_name = f"{args.model}_{args.embedding}_{args.prediction_mode}"

    # Add experiment type indicators
    if args.compare_architectures:
        base_name += "_arch_compare"
    elif args.optimize_architecture:
        base_name += f"_arch_opt_{args.architecture}"
    elif args.optimize_hyperparameters:
        base_name += "_hyper_opt"
    else:
        # Standard training
        if args.model == "torch":
            base_name += f"_{args.architecture}"

    # Add key hyperparameters based on model type
    if args.model == "rf":
        base_name += f"_n{args.n_estimators}"
    elif args.model in ["ridge", "lasso"]:
        base_name += f"_a{args.alpha}"
    elif args.model == "svm":
        base_name += f"_C{args.C}_eps{args.epsilon}_{args.kernel}"
    elif args.model == "torch":
        base_name += f"_b{args.batch_size}_e{args.epochs}"

    # Add a short UUID for absolute uniqueness
    short_uuid = str(uuid.uuid4())[:8]

    return f"{base_name}_{timestamp}_{short_uuid}"


def log_experiment_results_to_mlflow(
    args,
    model,
    val_results,
    test_results,
    experiment_id,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
    """Helper function to handle MLflow logging for all experiment types."""
    if not args.use_mlflow:
        return

    try:
        # Log PyTorch-specific metrics if applicable
        if args.model == "torch" and hasattr(model, "training_metrics"):
            log_torch_model_metrics(model)

        # Get model path for logging
        model_path = get_model_path(args, experiment_id)

        # Log evaluation results with context data
        log_evaluation_results(
            val_results,
            test_results,
            model,
            model_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        print("MLflow logging completed successfully")

    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")
        # Don't fail the entire experiment if MLflow logging fails
        pass


def main():
    """Main training workflow."""
    # Track total execution time
    start_time = time.time()
    print(f"=== Starting execution at {time.strftime('%H:%M:%S')} ===")

    # Setup environment
    setup_environment()

    # Parse command-line arguments
    args = parse_args()

    # Log system info
    log_system_info(args)

    # Generate a unique experiment ID
    experiment_id = get_experiment_id(args)
    print(f"Experiment ID: {experiment_id}")

    # Get embedding model path
    feature_path, embedding_model_path = get_checkpoint_paths(args)

    # Print key paths
    print(f"Feature checkpoint path: {feature_path}")
    print(f"Embedding model path: {embedding_model_path}")

    try:
        print(f"Starting feature extraction/loading for {args.embedding} embeddings...")
        # Get features (load from disk or extract new ones)
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = get_features(
            args
        )

        # Log dataset shapes
        print(f"Dataset shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")

        # Setup MLflow if enabled - do this before training
        if args.use_mlflow:
            print("Setting up MLflow tracking...")
            setup_mlflow()
            log_common_parameters(args)

        # Train the model (or run experiments)
        if args.compare_architectures:
            print(f"Running architecture comparison experiment...")
        elif args.optimize_architecture:
            print(f"Running architecture optimization experiment...")
        else:
            print(f"Training {args.model} model...")

        model, val_results, test_results = train_model(
            args, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Handle different experiment types for logging and saving
        if args.compare_architectures:
            print(
                "Architecture comparison completed. Results saved to experiments/architecture_comparison/"
            )
            # For comparison experiments, we don't have a single model to log
            # The comparison results are already saved by train_model
            if args.use_mlflow:
                # Log the experiment type and summary info
                import mlflow

                mlflow.log_param("experiment_type", "architecture_comparison")
                mlflow.log_param(
                    "architectures_compared", ",".join(args.architectures_to_compare)
                )
                # The detailed results are saved as artifacts by the comparison function

        elif args.optimize_architecture:
            print(
                "Architecture optimization completed. Results saved to experiments/architecture_optimization/"
            )

            # Save the optimized model
            model_path = get_model_path(args, experiment_id)
            print(f"Saving optimized model to: {model_path}")
            save_model(model, model_path)

            # Log to MLflow - optimization results have regular model/val_results/test_results
            log_experiment_results_to_mlflow(
                args,
                model,
                val_results,
                test_results,
                experiment_id,
                X_train,
                y_train,
                X_val,
                y_val,
            )

        else:
            # Standard training - log everything normally
            log_experiment_results_to_mlflow(
                args,
                model,
                val_results,
                test_results,
                experiment_id,
                X_train,
                y_train,
                X_val,
                y_val,
            )

            # Save model for standard training
            model_path = get_model_path(args, experiment_id)
            print(f"Saving model to: {model_path}")
            save_model(model, model_path)

        # Save embedding model if needed
        if args.save_features and embedding_model is not None:
            print(f"Saving embedding model to: {embedding_model_path}")
            save_embedding_model(embedding_model, embedding_model_path, args)

        # End MLflow run if it was started
        if args.use_mlflow:
            end_mlflow_run()

        # Report total execution time
        elapsed_time = time.time() - start_time
        print(f"=== Total execution time: {elapsed_time:.2f} seconds ===")
        print(f"=== Experiment {experiment_id} completed successfully ===")

        # Return explicit success code for batch script monitoring
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

        # End MLflow run on error too
        if args.use_mlflow:
            try:
                end_mlflow_run()
            except:
                pass

        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
