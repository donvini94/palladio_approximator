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
        else:
            params.update({
                "model_type": args.model,
                "alpha": args.alpha,
            })
    elif args.model == "torch":
        params.update({
            "model_type": "torch",  # For compatibility with visualize_training_metrics
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })
    
    mlflow.log_params(params)


def log_torch_model_metrics(model):
    """Log detailed training metrics from PyTorch model.

    Args:
        model: Trained PyTorch model with training_metrics attribute
    """
    # Import mlflow again to avoid scoping issues
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    
    if not hasattr(model, "training_metrics"):
        print("Warning: PyTorch model doesn't have training_metrics attribute")
        return
        
    # Log additional parameters (not already logged)
    additional_params = {
        "input_dim": model.training_metrics["input_dim"],
        "output_dim": model.training_metrics["output_dim"],
        "hidden_dims": str(model.training_metrics["hidden_dims"]),
        "learning_rate": model.training_metrics["learning_rate"],
        "optimizer": model.training_metrics["optimizer"],
        "early_stopping": model.training_metrics["early_stopping"],
    }
    mlflow.log_params(additional_params)

    # Log per-epoch metrics
    history = model.training_metrics["history"]
    for epoch in range(len(history["train_loss"])):
        mlflow.log_metrics(
            {
                "epoch": epoch + 1,
                "train_loss": history["train_loss"][epoch],
                "val_loss": history["val_loss"][epoch],
                "val_mse": history["val_mse"][epoch],
                "val_mae": history["val_mae"][epoch],
                "learning_rate": history["learning_rate"][epoch],
            },
            step=epoch,
        )

    # Log best metrics with explicit best_ prefix to avoid conflicts
    best_metrics = {
        f"best_{k}": v
        for k, v in model.training_metrics["best_metrics"].items()
    }
    mlflow.log_metrics(best_metrics)

    # Log final metrics
    mlflow.log_metrics(
        {
            "training_time_seconds": model.training_metrics["training_time_seconds"],
            "final_val_loss": model.training_metrics["final_val_loss"],
            "total_epochs": model.training_metrics["total_epochs"],
        }
    )

    # Log the PyTorch model file
    try:
        model_path = model.training_metrics.get("model_path")
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        # Try to log PyTorch model directly if it's available
        if "pytorch_model" in model.training_metrics:
            try:
                pytorch_model = model.training_metrics["pytorch_model"]
                mlflow.pytorch.log_model(pytorch_model, "pytorch_model")
            except Exception as e:
                print(f"Warning: Could not log PyTorch model directly: {e}")
    except Exception as e:
        print(f"Warning: Could not log model file: {e}")


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