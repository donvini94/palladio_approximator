"""
Main training script for Palladio Approximator.

This script orchestrates the entire training pipeline:
1. Parsing command line arguments
2. Loading or generating embeddings/features
3. Training a model (random forest, linear, or neural network)
4. Evaluating the model on validation and test sets
5. Saving the model and tracking the experiment with MLflow
"""
import os
import sys
import time
import traceback
import joblib
import uuid
import psutil
import socket
from datetime import datetime

# Load utilities
from utils.config import parse_args, setup_environment, get_checkpoint_paths, get_model_path
from utils.feature_manager import get_features
from utils.model_trainer import train_model, save_model, save_embedding_model_if_needed
from utils.mlflow_logger import (
    setup_mlflow, 
    log_common_parameters,
    log_torch_model_metrics,
    log_evaluation_results,
    end_mlflow_run
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
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
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
    
    # Add key hyperparameters based on model type
    if args.model == "rf":
        base_name += f"_n{args.n_estimators}"
    elif args.model in ["ridge", "lasso"]:
        base_name += f"_a{args.alpha}"
    elif args.model == "torch":
        base_name += f"_b{args.batch_size}_e{args.epochs}"
    
    # Add a short UUID for absolute uniqueness
    short_uuid = str(uuid.uuid4())[:8]
    
    return f"{base_name}_{timestamp}_{short_uuid}"


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
    checkpoint_path, embedding_model_path = get_checkpoint_paths(args)
    
    # Print key paths
    print(f"Feature checkpoint path: {checkpoint_path}")
    print(f"Embedding model path: {embedding_model_path}")
    
    try:
        print(f"Starting feature extraction/loading for {args.embedding} embeddings...")
        # Get features (load from disk or extract new ones)
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = get_features(args)
        
        # Log dataset shapes
        print(f"Dataset shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Setup MLflow if enabled
        if args.use_mlflow:
            print("Setting up MLflow tracking...")
            setup_mlflow()
            log_common_parameters(args)
        
        # Train the model
        print(f"Training {args.model} model...")
        model, val_results, test_results = train_model(
            args, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Log metrics to MLflow if enabled
        if args.use_mlflow:
            print("Logging results to MLflow...")
            # Log PyTorch-specific metrics if applicable
            if args.model == "torch" and hasattr(model, "training_metrics"):
                log_torch_model_metrics(model)
                
            # Log evaluation results
            model_path = get_model_path(args, experiment_id)
            log_evaluation_results(val_results, test_results, model, model_path)
            end_mlflow_run()
        
        # Save the model with experiment ID in name
        model_path = get_model_path(args, experiment_id)
        print(f"Saving model to: {model_path}")
        save_model(model, model_path)
        
        # Save embedding model if needed
        if args.save_features and embedding_model is not None:
            print(f"Saving embedding model to: {embedding_model_path}")
            save_embedding_model_if_needed(embedding_model, embedding_model_path, args)
        
        # Report total execution time
        elapsed_time = time.time() - start_time
        print(f"=== Total execution time: {elapsed_time:.2f} seconds ===")
        print(f"=== Experiment {experiment_id} completed successfully ===")
        
        # Return explicit success code for batch script monitoring
        return 0
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        # Save work in progress
        print("Attempting to save partial results...")
        try:
            interrupted_file = f"interrupted_state_{experiment_id}.pkl"
            joblib.dump({"args": args, "interrupted": True, "timestamp": datetime.now()}, interrupted_file)
            print(f"Saved interrupt state to {interrupted_file}")
        except Exception as e:
            print(f"Failed to save interrupt state: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())