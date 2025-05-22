import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch


def setup_mlflow(experiment_name="palladio-approximation"):
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
        for metric_name in [
            "train_loss",
            "val_loss",
            "train_mse",
            "val_mse",
            "train_mae",
            "val_mae",
            "learning_rate",
        ]:
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
            indices = np.linspace(0, total_batches - 1, 100, dtype=int)
            for i, batch_idx in enumerate(indices):
                mlflow.log_metric(
                    "batch_train_loss", batch_losses[batch_idx], step=batch_idx
                )
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
            except Exception as e:
                # Provide more detailed error message
                mlflow.log_param("model_logging_error", str(e))
                print(f"Error logging PyTorch model: {str(e)}")
                print("This could be due to dependencies not being properly captured.")
    except Exception as e:
        # Log the error as a parameter for visibility
        mlflow.log_param("model_processing_error", str(e))
        print(f"Error processing PyTorch model for logging: {str(e)}")


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
    """Log evaluation results to MLflow, handling both old dict format and new EvaluationResults.

    Args:
        val_results: Validation evaluation metrics (dict or EvaluationResults)
        test_results: Test evaluation metrics (dict or EvaluationResults)
        model: Trained model object to log
        model_path (str, optional): Path to the saved model for artifact logging
        X_train, y_train: Training data (for baseline calculation)
        X_val, y_val: Validation data (for context metrics)
    """
    import numpy as np
    import os

    try:
        # Convert EvaluationResults to dictionaries if needed
        val_metrics = _extract_metrics(val_results)
        test_metrics = _extract_metrics(test_results)

        # Log the evaluation metrics
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        # Log confidence intervals if available
        if (
            hasattr(val_results, "confidence_intervals")
            and val_results.confidence_intervals
        ):
            _log_confidence_intervals(val_results.confidence_intervals, "val")
        if (
            hasattr(test_results, "confidence_intervals")
            and test_results.confidence_intervals
        ):
            _log_confidence_intervals(test_results.confidence_intervals, "test")

        # Log the model as an artifact if path is provided
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        # Also log the model to the sklearn flavor if possible
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"Warning: Could not log model with sklearn flavor: {e}")

        # Add context metrics if training data is provided
        if (
            X_train is not None
            and y_train is not None
            and X_val is not None
            and y_val is not None
            and model is not None
        ):
            _log_context_metrics(model, X_train, y_train, X_val, y_val, val_metrics)

    except Exception as e:
        # Log any errors that occur during metric processing
        print(f"ERROR in MLflow logging: {e}")
        import traceback

        traceback.print_exc()
        mlflow.log_param("logging_error", str(e))


def _extract_metrics(results):
    """Extract metrics dictionary from results (handles both old and new format)."""
    if hasattr(results, "metrics"):
        return results.metrics
    elif isinstance(results, dict):
        return results
    else:
        raise ValueError(f"Cannot extract metrics from {type(results)}")


def _log_confidence_intervals(confidence_intervals, split_name):
    """Log confidence intervals as separate metrics."""
    for metric_name, (ci_lower, ci_upper) in confidence_intervals.items():
        # Log CI bounds
        mlflow.log_metric(f"{metric_name}_ci_lower", ci_lower)
        mlflow.log_metric(f"{metric_name}_ci_upper", ci_upper)
        # Log CI width
        ci_width = ci_upper - ci_lower
        mlflow.log_metric(f"{metric_name}_ci_width", ci_width)


def _log_context_metrics(model, X_train, y_train, X_val, y_val, val_metrics):
    """Log context metrics including baselines and interpretations."""
    from utils.metrics_context import (
        get_baseline_metrics,
        create_metrics_interpretation,
        create_performance_visualization,
    )
    import tempfile
    import json
    import numpy as np

    try:
        # Calculate baseline metrics
        baseline_metrics = get_baseline_metrics(X_train, y_train, X_val, y_val)

        # Log baseline metrics
        for key, value in baseline_metrics.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    mlflow.log_metric(f"{key}_{i}", v)
            else:
                mlflow.log_metric(key, value)

        # Calculate target statistics
        target_stats = {}
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            # Multi-output case
            target_stats["mean"] = np.mean(y_val, axis=0).tolist()
            target_stats["median"] = np.median(y_val, axis=0).tolist()
            target_stats["std"] = np.std(y_val, axis=0).tolist()
            target_stats["variance"] = np.var(y_val, axis=0).tolist()
        else:
            # Single output case
            target_stats["mean"] = float(np.mean(y_val))
            target_stats["median"] = float(np.median(y_val))
            target_stats["std"] = float(np.std(y_val))
            target_stats["variance"] = float(np.var(y_val))

        # Log target statistics
        mlflow.log_params({f"target_stat_{k}": str(v) for k, v in target_stats.items()})

        # Calculate prediction errors for validation set
        predictions = model.predict(X_val)

        # Filter out extreme outliers from errors to avoid visualization issues
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            # Take first output dimension for visualization - multi-output case
            raw_errors = predictions[:, 0] - y_val[:, 0]
        else:
            # Single output case
            raw_errors = predictions.flatten() - y_val.flatten()

        # Apply outlier filtering - values more than 5 std from mean will be capped
        error_mean = np.mean(raw_errors)
        error_std = np.std(raw_errors)
        lower_bound = error_mean - 5 * error_std
        upper_bound = error_mean + 5 * error_std
        errors = np.clip(raw_errors, lower_bound, upper_bound)

        # Create model_metrics for interpretation
        multi_output = len(y_val.shape) > 1 and y_val.shape[1] > 1

        if multi_output:
            # Calculate MSE for each output dimension
            mse = np.mean((y_val - predictions) ** 2, axis=0)
            mae = np.mean(np.abs(y_val - predictions), axis=0)

            # Convert to list if numpy array
            if hasattr(mse, "tolist"):
                mse = mse.tolist()
            if hasattr(mae, "tolist"):
                mae = mae.tolist()
        else:
            # Single output case
            mse = np.mean((y_val.flatten() - predictions.flatten()) ** 2)
            mae = np.mean(np.abs(y_val.flatten() - predictions.flatten()))

        # Create a more complete model_metrics with correct keys
        model_metrics = {
            "val_mse": mse,
            "val_mae": mae,
            "prediction_errors": errors,
        }

        # Also add all metrics from val_metrics if they're available
        model_metrics.update(val_metrics)

        # Create metrics interpretation
        interpretation = create_metrics_interpretation(
            model_metrics, baseline_metrics, target_stats
        )

        # Save interpretation to a file and log as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save interpretation as JSON
            interp_path = os.path.join(tmpdir, "metrics_interpretation.json")
            with open(interp_path, "w") as f:
                json.dump(interpretation, f, indent=2)

            # Log interpretation file
            mlflow.log_artifact(interp_path)

            # Create performance visualizations and log them
            viz_dir = os.path.join(tmpdir, "performance_context")
            viz_paths = create_performance_visualization(
                model_metrics, baseline_metrics, target_stats, viz_dir
            )

            # Log all visualizations
            for viz_path in viz_paths:
                mlflow.log_artifact(viz_path)

            # Create a markdown summary for easy viewing
            summary_path = os.path.join(tmpdir, "performance_summary.md")
            with open(summary_path, "w") as f:
                f.write("# Model Performance Summary\n\n")

                f.write("## Overall Assessment\n\n")
                f.write(interpretation["performance_summary"])
                f.write("\n\n")

                f.write("## Metrics Context\n\n")
                for metric, info in interpretation["metrics_context"].items():
                    # Check if info is a dictionary before proceeding
                    if not isinstance(info, dict):
                        continue

                    f.write(f"### {metric}\n")

                    # Only include value if it exists
                    if "value" in info and info["value"] is not None:
                        f.write(f"- Value: {info['value']}\n")

                    # Include available baselines and other metrics
                    for key, val in info.items():
                        if key != "interpretation" and key != "value":
                            f.write(f"- {key}: {val}\n")

                    # Add interpretation
                    if "interpretation" in info:
                        f.write(f"- **Interpretation**: {info['interpretation']}\n")

                    f.write("\n")

                f.write("## Domain-Specific Interpretation\n\n")
                if interpretation["domain_interpretation"]:
                    for key, val in interpretation["domain_interpretation"].items():
                        f.write(f"### {key}\n")
                        f.write(f"{val}\n\n")
                else:
                    f.write("No domain-specific interpretation available.\n\n")

            # Log summary markdown
            mlflow.log_artifact(summary_path)

            # Also add key interpretation as run tags for easy filtering/searching
            try:
                if (
                    "performance_summary" in interpretation
                    and interpretation["performance_summary"]
                ):
                    summary = interpretation["performance_summary"]
                    if "shows " in summary and " performance" in summary:
                        level = summary.split("shows ")[1].split(" performance")[0]
                        mlflow.set_tag("performance_level", level)
                    else:
                        # Use a generic tag instead
                        mlflow.set_tag("performance_summary_available", "true")
            except Exception as e:
                print(f"Warning: Could not extract performance level for tagging: {e}")

    except Exception as e:
        print(f"Warning: Could not log context metrics: {e}")
        mlflow.log_param("context_metrics_error", str(e))


def end_mlflow_run():
    """End the current MLflow run."""
    mlflow.end_run()
