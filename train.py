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


def main():
    """Main training workflow."""
    # Track total execution time
    start_time = time.time()
    print(f"=== Starting execution at {time.strftime('%H:%M:%S')} ===")
    
    # Setup environment
    setup_environment()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Get embedding model path
    _, embedding_model_path = get_checkpoint_paths(args)
    
    try:
        # Get features (load from disk or extract new ones)
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = get_features(args)
        
        # Setup MLflow if enabled
        if args.use_mlflow:
            setup_mlflow()
            log_common_parameters(args)
        
        # Train the model
        model, val_results, test_results = train_model(
            args, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Log metrics to MLflow if enabled
        if args.use_mlflow:
            # Log PyTorch-specific metrics if applicable
            if args.model == "torch" and hasattr(model, "training_metrics"):
                log_torch_model_metrics(model)
                
            # Log evaluation results
            model_path = get_model_path(args)
            log_evaluation_results(val_results, test_results, model, model_path)
            end_mlflow_run()
        
        # Save the model
        model_path = get_model_path(args)
        save_model(model, model_path)
        
        # Save embedding model if needed
        save_embedding_model_if_needed(embedding_model, embedding_model_path, args)
        
        # Report total execution time
        elapsed_time = time.time() - start_time
        print(f"=== Total execution time: {elapsed_time:.2f} seconds ===")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        # Save work in progress
        print("Attempting to save partial results...")
        try:
            joblib.dump({"args": args, "interrupted": True}, "interrupted_state.pkl")
            print("Saved interrupt state to interrupted_state.pkl")
        except Exception as e:
            print(f"Failed to save interrupt state: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()