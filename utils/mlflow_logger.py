import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch


def setup_mlflow(experiment_name="thesis"):
    """Set up MLflow experiment tracking.

    Args:
        experiment_name (str): Name of the experiment
    """
    print("Setting up MLflow...")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def log_common_parameters(args):
    """Log common parameters for all model types.

    Args:
        args: Command line arguments
    """
    # Log common parameters for all models
    params = {
        "model": args.model,
        "embedding": args.embedding,
        "prediction_mode": args.prediction_mode,
    }

    # Add parameters specific to different model types
    if args.model in ("rf", "ridge", "lasso"):
        if args.model == "rf":
            params.update(
                {
                    "model_type": "rf",
                    "n_estimators": args.n_estimators,
                }
            )

            # Add optimized parameters if available
            for param in ["max_depth", "min_samples_split", "min_samples_leaf"]:
                if hasattr(args, param):
                    params[param] = getattr(args, param)

        else:
            params.update(
                {
                    "model_type": args.model,
                    "alpha": args.alpha,
                }
            )

            # Add optimized parameters if available
            for param in ["solver", "max_iter"]:
                if hasattr(args, param):
                    params[param] = getattr(args, param)

    elif args.model == "torch":
        params.update(
            {
                "model_type": "torch",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": getattr(args, "architecture", "embedding_regressor"),
            }
        )

        # Add optimized parameters if available
        for param in ["learning_rate", "dropout_rate"]:
            if hasattr(args, param):
                params[param] = getattr(args, param)

        # Add hidden_dims as string if available
        if hasattr(args, "hidden_dims"):
            params["hidden_dims"] = str(args.hidden_dims)

    # Log hyperparameter optimization settings
    if hasattr(args, "optimize_hyperparameters"):
        params.update(
            {
                "optimize_hyperparameters": args.optimize_hyperparameters,
                "n_trials": args.n_trials if args.optimize_hyperparameters else None,
                "optimization_metric": (
                    args.optimization_metric if args.optimize_hyperparameters else None
                ),
            }
        )

    # Log experiment type flags
    if hasattr(args, "compare_architectures"):
        params["compare_architectures"] = args.compare_architectures
    if hasattr(args, "optimize_architecture"):
        params["optimize_architecture"] = args.optimize_architecture

    mlflow.log_params(params)


def log_torch_model_metrics(model):
    """Log detailed training metrics from PyTorch model.

    Args:
        model: Trained PyTorch model with training_metrics attribute
    """
    import numpy as np

    if not hasattr(model, "training_metrics"):
        print("Warning: PyTorch model doesn't have training_metrics attribute")
        return

    training_metrics = model.training_metrics

    try:
        # Log basic training parameters
        basic_params = {}
        for key in ["input_dim", "output_dim", "total_parameters", "architecture_type"]:
            if key in training_metrics:
                basic_params[key] = training_metrics[key]

        if basic_params:
            mlflow.log_params(basic_params)

        # Log training summary metrics
        summary_metrics = {}
        for key in ["training_time_seconds", "best_val_loss", "total_epochs"]:
            if key in training_metrics and isinstance(
                training_metrics[key], (int, float)
            ):
                summary_metrics[key] = training_metrics[key]

        if summary_metrics:
            mlflow.log_metrics(summary_metrics)

        # Log training history as time series
        if "history" in training_metrics:
            history = training_metrics["history"]

            # Get the length of training history
            train_loss = history.get("train_loss", [])
            max_epochs = len(train_loss)

            if max_epochs > 0:
                for epoch in range(max_epochs):
                    epoch_metrics = {}

                    # Log available metrics for this epoch
                    for metric_name in ["train_loss", "val_loss", "learning_rate"]:
                        if metric_name in history and epoch < len(history[metric_name]):
                            value = history[metric_name][epoch]
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                epoch_metrics[metric_name] = value

                    if epoch_metrics:
                        mlflow.log_metrics(epoch_metrics, step=epoch)

        print("✓ PyTorch model metrics logged successfully")

    except Exception as e:
        print(f"Warning: Error logging PyTorch model metrics: {e}")
        mlflow.log_param("torch_logging_error", str(e))


def log_evaluation_results(
    val_results,
    test_results,
    model=None,
    model_path=None,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
    """Log evaluation results to MLflow with robust error handling.

    Args:
        val_results: Validation evaluation metrics (dict or EvaluationResults)
        test_results: Test evaluation metrics (dict or EvaluationResults)
        model: Trained model object to log
        model_path (str, optional): Path to the saved model for artifact logging
        X_train, y_train: Training data (for baseline calculation)
        X_val, y_val: Validation data (for context metrics)
    """
    try:
        # Extract metrics with robust error handling
        val_metrics = _extract_metrics_safe(val_results, "val")
        test_metrics = _extract_metrics_safe(test_results, "test")

        # Log the evaluation metrics
        if val_metrics:
            mlflow.log_metrics(val_metrics)
            print(f"✓ Logged {len(val_metrics)} validation metrics")

        if test_metrics:
            mlflow.log_metrics(test_metrics)
            print(f"✓ Logged {len(test_metrics)} test metrics")

        # Log confidence intervals if available
        _log_confidence_intervals_safe(val_results, "val")
        _log_confidence_intervals_safe(test_results, "test")

        # Log the model as an artifact if path is provided
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path)
            print("✓ Model artifact logged")

        # Try to log the model with sklearn flavor
        if model is not None:
            try:
                # Check if it's a sklearn-compatible model
                if hasattr(model, "predict") and not hasattr(model, "training_metrics"):
                    mlflow.sklearn.log_model(model, "model")
                    print("✓ Model logged with sklearn flavor")
            except Exception as e:
                print(f"Note: Could not log model with sklearn flavor: {e}")

        # Log basic dataset statistics
        _log_dataset_stats(X_train, y_train, X_val, y_val)

        print("✓ MLflow evaluation results logged successfully")

    except Exception as e:
        print(f"ERROR in MLflow logging: {e}")
        import traceback

        traceback.print_exc()

        # Log the error for debugging
        try:
            mlflow.log_param("logging_error", str(e))
        except:
            pass


def _extract_metrics_safe(results, prefix=""):
    """Safely extract metrics from evaluation results with multiple fallback methods."""
    if results is None:
        return {}

    # Method 1: Try the standard .metrics attribute
    if hasattr(results, "metrics") and isinstance(results.metrics, dict):
        return results.metrics

    # Method 2: If it's already a dictionary
    if isinstance(results, dict):
        return results

    # Method 3: Try to convert object attributes to dict
    if hasattr(results, "__dict__"):
        try:
            metrics = {}
            for key, value in results.__dict__.items():
                if not key.startswith("_") and isinstance(value, (int, float, bool)):
                    metrics[key] = value
            if metrics:
                return metrics
        except:
            pass

    # Method 4: Try common metric attribute names
    common_metrics = ["mse", "mae", "r2", "rmse"]
    metrics = {}

    for metric in common_metrics:
        # Try with and without prefix
        for attr_name in [
            f"{prefix}_{metric}",
            metric,
            f"{metric}_avg",
            f"{prefix}_{metric}_avg",
        ]:
            if hasattr(results, attr_name):
                value = getattr(results, attr_name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[attr_name] = value

    return metrics


def _log_confidence_intervals_safe(results, split_name):
    """Safely log confidence intervals if they exist."""
    try:
        if hasattr(results, "confidence_intervals") and results.confidence_intervals:
            confidence_intervals = results.confidence_intervals
            for metric_name, interval in confidence_intervals.items():
                if isinstance(interval, (list, tuple)) and len(interval) == 2:
                    ci_lower, ci_upper = interval
                    if isinstance(ci_lower, (int, float)) and isinstance(
                        ci_upper, (int, float)
                    ):
                        mlflow.log_metric(f"{metric_name}_ci_lower", ci_lower)
                        mlflow.log_metric(f"{metric_name}_ci_upper", ci_upper)
                        mlflow.log_metric(
                            f"{metric_name}_ci_width", ci_upper - ci_lower
                        )
    except Exception as e:
        print(f"Note: Could not log confidence intervals for {split_name}: {e}")


def _log_dataset_stats(X_train, y_train, X_val, y_val):
    """Log basic dataset statistics."""
    try:
        import numpy as np

        stats = {}

        if X_train is not None:
            stats["n_train_samples"] = X_train.shape[0]
            stats["n_features"] = X_train.shape[1]

        if X_val is not None:
            stats["n_val_samples"] = X_val.shape[0]

        if y_train is not None:
            if len(y_train.shape) > 1:
                stats["n_targets"] = y_train.shape[1]
                stats["train_target_mean"] = float(np.mean(y_train))
                stats["train_target_std"] = float(np.std(y_train))
            else:
                stats["n_targets"] = 1
                stats["train_target_mean"] = float(np.mean(y_train))
                stats["train_target_std"] = float(np.std(y_train))

        if y_val is not None:
            if len(y_val.shape) > 1:
                stats["val_target_mean"] = float(np.mean(y_val))
                stats["val_target_std"] = float(np.std(y_val))
            else:
                stats["val_target_mean"] = float(np.mean(y_val))
                stats["val_target_std"] = float(np.std(y_val))

        if stats:
            mlflow.log_params(stats)
            print(f"✓ Logged {len(stats)} dataset statistics")

    except Exception as e:
        print(f"Note: Could not log dataset statistics: {e}")


def end_mlflow_run():
    """End the current MLflow run."""
    try:
        mlflow.end_run()
        print("✓ MLflow run ended successfully")
    except Exception as e:
        print(f"Warning: Error ending MLflow run: {e}")


# Legacy function for backwards compatibility
def _extract_metrics(results):
    """Legacy function - use _extract_metrics_safe instead."""
    return _extract_metrics_safe(results)
