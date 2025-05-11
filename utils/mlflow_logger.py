"""
MLflow integration module for experiment tracking.
"""
import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch


def setup_mlflow(experiment_name="dsl-performance-prediction"):
    """Set up MLflow experiment tracking.

    Args:
        experiment_name (str): Name of the experiment
    """
    # Import mlflow again to avoid scoping issues
    import mlflow
    
    print("Setting up MLflow...")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def log_common_parameters(args):
    """Log common parameters for all model types.

    Args:
        args: Command line arguments
    """
    # Import mlflow again to avoid scoping issues
    import mlflow
    
    # Log common parameters for all models
    params = {
        "model": args.model,
        "embedding": args.embedding,
        "prediction_mode": args.prediction_mode,
    }
    
    # Add parameters specific to different model types
    if args.model in ("rf", "ridge", "lasso"):
        if args.model == "rf":
            params.update({
                "model_type": "rf",
                "n_estimators": args.n_estimators,
            })
            
            # Add optimized parameters if available
            for param in ['max_depth', 'min_samples_split', 'min_samples_leaf']:
                if hasattr(args, param):
                    params[param] = getattr(args, param)
                    
        else:
            params.update({
                "model_type": args.model,
                "alpha": args.alpha,
            })
            
            # Add optimized parameters if available
            for param in ['solver', 'max_iter']:
                if hasattr(args, param):
                    params[param] = getattr(args, param)
                    
    elif args.model == "torch":
        params.update({
            "model_type": "torch",  # For compatibility with visualize_training_metrics
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })
        
        # Add optimized parameters if available
        for param in ['learning_rate', 'dropout_rate']:
            if hasattr(args, param):
                params[param] = getattr(args, param)
                
        # Add hidden_dims as string if available
        if hasattr(args, 'hidden_dims'):
            params['hidden_dims'] = str(args.hidden_dims)
    
    # Log hyperparameter optimization settings
    if hasattr(args, 'optimize_hyperparameters'):
        params.update({
            "optimize_hyperparameters": args.optimize_hyperparameters,
            "n_trials": args.n_trials if args.optimize_hyperparameters else None,
            "optimization_metric": args.optimization_metric if args.optimize_hyperparameters else None,
        })
    
    mlflow.log_params(params)


def log_torch_model_metrics(model):
    """Log detailed training metrics from PyTorch model.

    Args:
        model: Trained PyTorch model with training_metrics attribute
    """
    # Import mlflow again to avoid scoping issues
    import mlflow
    import mlflow.pytorch
    import numpy as np
    
    if not hasattr(model, "training_metrics"):
        print("Warning: PyTorch model doesn't have training_metrics attribute")
        return
        
    # Log additional parameters (not already logged)
    additional_params = {
        "input_dim": model.training_metrics.get("input_dim", 0),
        "output_dim": model.training_metrics.get("output_dim", 0),
        "hidden_dims": str(model.training_metrics.get("hidden_dims", [])),
        "learning_rate": model.training_metrics.get("learning_rate", 0),
        "optimizer": model.training_metrics.get("optimizer", "unknown"),
        "early_stopping": model.training_metrics.get("early_stopping", 0),
        "total_epochs": model.training_metrics.get("total_epochs", 0),
        "total_steps": model.training_metrics.get("total_steps", 0),
    }
    mlflow.log_params(additional_params)

    # Log per-epoch metrics
    history = model.training_metrics.get("history", {})
    for epoch in range(len(history.get("train_loss", []))):
        epoch_metrics = {
            "epoch": epoch + 1,
        }
        
        # Add all available metrics for this epoch
        for metric_name in ["train_loss", "val_loss", "train_mse", "val_mse", 
                           "train_mae", "val_mae", "learning_rate"]:
            if metric_name in history and epoch < len(history[metric_name]):
                epoch_metrics[metric_name] = history[metric_name][epoch]
        
        # Log with epoch as the step
        mlflow.log_metrics(epoch_metrics, step=epoch)
    
    # Log batch-level metrics if available
    batch_history = model.training_metrics.get("batch_history", {})
    if batch_history and "train_loss_batches" in batch_history:
        # Log selected batches for visualization (avoid logging every batch)
        batch_losses = batch_history["train_loss_batches"]
        total_batches = len(batch_losses)
        
        # Log at most 100 evenly spaced batches to avoid overwhelming MLflow
        if total_batches > 100:
            indices = np.linspace(0, total_batches-1, 100, dtype=int)
            for i, batch_idx in enumerate(indices):
                mlflow.log_metric("batch_train_loss", batch_losses[batch_idx], step=batch_idx)
        else:
            for batch_idx, loss in enumerate(batch_losses):
                mlflow.log_metric("batch_train_loss", loss, step=batch_idx)

    # Log best metrics with explicit best_ prefix to avoid conflicts
    best_metrics = {
        f"best_{k}": v
        for k, v in model.training_metrics.get("best_metrics", {}).items()
    }
    mlflow.log_metrics(best_metrics)

    # Log final metrics
    final_metrics = {
        "training_time_seconds": model.training_metrics.get("training_time_seconds", 0),
        "final_val_loss": model.training_metrics.get("final_val_loss", 0),
        "final_train_loss": model.training_metrics.get("final_train_loss", 0),
    }
    mlflow.log_metrics(final_metrics)

    # Standardize model logging - prefer using mlflow's native PyTorch support
    try:
        if "pytorch_model" in model.training_metrics:
            pytorch_model = model.training_metrics["pytorch_model"]
            try:
                # Use native PyTorch model logging with proper error handling
                mlflow.pytorch.log_model(pytorch_model, "pytorch_model")
                
                # No need to log the saved file separately
                # The model is already saved as an MLflow artifact
            except Exception as e:
                # Provide more detailed error message
                mlflow.log_param("model_logging_error", str(e))
                print(f"Error logging PyTorch model: {str(e)}")
                print("This could be due to dependencies not being properly captured.")
    except Exception as e:
        # Log the error as a parameter for visibility
        mlflow.log_param("model_processing_error", str(e))
        print(f"Error processing PyTorch model for logging: {str(e)}")


def log_evaluation_results(val_results, test_results, model=None, model_path=None):
    """Log evaluation results to MLflow.

    Args:
        val_results (dict): Validation evaluation metrics
        test_results (dict): Test evaluation metrics
        model: Trained model object to log
        model_path (str, optional): Path to the saved model for artifact logging
    """
    # Import mlflow again to avoid scoping issues
    import mlflow
    import mlflow.sklearn
    
    # Log the evaluation metrics
    mlflow.log_metrics(val_results)
    mlflow.log_metrics(test_results)
    
    # Log the model as an artifact if path is provided
    if model_path and os.path.exists(model_path):
        mlflow.log_artifact(model_path)
    
    # Also log the model to the sklearn flavor if possible
    if model is not None:
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"Warning: Could not log model with sklearn flavor: {e}")


def end_mlflow_run():
    """End the current MLflow run."""
    # Import mlflow again to avoid scoping issues
    import mlflow
    
    mlflow.end_run()