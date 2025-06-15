import time
import joblib
import json
import os
import mlflow
from models import torch_model
from models.rf_model import train_random_forest
from models.linear_model import train_linear_model
from models.svm_model import train_svm
from models.torch_model import (
    train_torch_regressor,
    compare_architectures,
    optimize_architecture_hyperparameters,
)
from evaluate import evaluate_model, results_to_dict
from utils.config import get_device, get_model_path


def train_model(args, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model based on the selected type, with support for architecture experiments.

    Args:
        args: Command line arguments
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels

    Returns:
        tuple: (trained_model, val_results, test_results) or (comparison_results, None, None) for comparisons
    """

    print(f"=== Starting model training at {time.strftime('%H:%M:%S')} ===")
    print(
        f"Training {args.model} model on data with shape: X_train={X_train.shape}, y_train={y_train.shape}"
    )

    # Handle architecture comparison experiments
    if args.compare_architectures and args.model == "torch":
        print(f"=== Running Architecture Comparison Experiment ===")
        return run_architecture_comparison(
            args, X_train, y_train, X_val, y_val, X_test, y_test
        )

    # Handle architecture-specific hyperparameter optimization
    if args.optimize_architecture and args.model == "torch":
        print(f"=== Running Architecture-Specific Hyperparameter Optimization ===")
        return run_architecture_optimization(
            args, X_train, y_train, X_val, y_val, X_test, y_test
        )

    # Standard hyperparameter optimization if requested
    if args.optimize_hyperparameters:
        print(
            f"=== Running hyperparameter optimization with {args.n_trials} trials ==="
        )
        # Import here to avoid circular imports
        from utils.hyperparameter_optimization import run_hyperparameter_optimization

        # Run optimization to get best hyperparameters
        best_params, optimization_history = run_hyperparameter_optimization(
            args, X_train, y_train, X_val, y_val
        )

        print(f"Optimization complete. Best parameters: {best_params}")

        # Update args with best hyperparameters
        for param, value in best_params.items():
            if hasattr(args, param):
                print(f"Setting optimized parameter: {param} = {value}")
                setattr(args, param, value)

    # Train model based on selected type (using optimized parameters if available)
    if args.model == "rf":
        # Extract RF-specific parameters from args
        rf_params = {"n_estimators": args.n_estimators}

        # Add additional parameters if they were optimized
        for param in ["max_depth", "min_samples_split", "min_samples_leaf"]:
            if hasattr(args, param):
                rf_params[param] = getattr(args, param)

        model = train_random_forest(X_train, y_train, **rf_params)

    elif args.model == "torch":
        device = get_device(args)
        torch_params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
            "architecture_type": getattr(args, "architecture", "embedding_regressor"),
        }

        # Add additional parameters if they were optimized
        for param in ["learning_rate", "hidden_dims", "dropout_rate"]:
            if hasattr(args, param):
                torch_params[param] = getattr(args, param)

        model = train_torch_regressor(X_train, y_train, **torch_params)

    elif args.model in ("ridge", "lasso"):
        print("Training linear model...")

        # Extract linear model-specific parameters from args
        linear_params = {"model_type": args.model, "alpha": args.alpha}

        # Add additional parameters if they were optimized
        for param in ["solver", "max_iter"]:
            if hasattr(args, param):
                linear_params[param] = getattr(args, param)

        model = train_linear_model(X_train, y_train, **linear_params)

    elif args.model == "svm":
        print("Training SVM model...")

        # Extract SVM-specific parameters from args
        svm_params = {"C": getattr(args, "C", 1.0), "epsilon": getattr(args, "epsilon", 0.1)}

        # Add additional parameters if they were optimized
        for param in ["kernel", "gamma", "degree"]:
            if hasattr(args, param):
                svm_params[param] = getattr(args, param)

        model = train_svm(X_train, y_train, **svm_params)
    else:
        raise ValueError(
            "Unsupported model type. Choose 'rf', 'torch', 'ridge', 'lasso', or 'svm'"
        )

    print(f"=== Model training completed at {time.strftime('%H:%M:%S')} ===")

    # Evaluate the model using enhanced evaluation
    print("Evaluating model...")

    # Use enhanced evaluation with bootstrap confidence intervals
    val_results = evaluate_model(
        model, X_val, y_val, split_name="val", verbose=False, bootstrap_ci=True
    )
    test_results = evaluate_model(
        model, X_test, y_test, split_name="test", verbose=True, bootstrap_ci=True
    )

    # Print results for console output
    print(f"Validation results: {val_results}")
    print(f"Test results: {test_results}")

    return model, val_results, test_results


def run_architecture_comparison(args, X_train, y_train, X_val, y_val, X_test, y_test):
    """Run architecture comparison experiment.

    Args:
        args: Command line arguments
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits

    Returns:
        tuple: (comparison_results, None, None)
    """
    device = get_device(args)
    architectures = args.architectures_to_compare

    print(f"Comparing architectures: {architectures}")
    print(f"Device: {device}")

    # Run comparison
    results = compare_architectures(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        architectures=architectures,
        device=device,
        verbose=True,
    )

    # Create summary of results
    summary = {}
    for arch_name, arch_results in results.items():
        if "error" in arch_results:
            summary[arch_name] = {"error": arch_results["error"]}
        else:
            # Extract key metrics for comparison
            test_results = arch_results["test_results"]
            test_metrics = (
                test_results.metrics
                if hasattr(test_results, "metrics")
                else test_results
            )

            training_time = arch_results["training_metrics"]["training_time_seconds"]
            total_params = arch_results["training_metrics"]["total_parameters"]

            summary[arch_name] = {
                "test_mse": test_metrics.get(
                    "test_mse_avg", test_metrics.get("test_mse", "N/A")
                ),
                "test_r2": test_metrics.get(
                    "test_r2_avg", test_metrics.get("test_r2", "N/A")
                ),
                "training_time": training_time,
                "parameters": total_params,
            }

    # Print comparison summary
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Architecture':<15} {'Test MSE':<12} {'Test R²':<12} {'Time (s)':<10} {'Parameters':<12}"
    )
    print("-" * 80)

    for arch_name, metrics in summary.items():
        if "error" in metrics:
            print(
                f"{arch_name:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<12}"
            )
        else:
            mse = (
                f"{metrics['test_mse']:.6f}"
                if isinstance(metrics["test_mse"], (int, float))
                else str(metrics["test_mse"])
            )
            r2 = (
                f"{metrics['test_r2']:.6f}"
                if isinstance(metrics["test_r2"], (int, float))
                else str(metrics["test_r2"])
            )
            time_str = f"{metrics['training_time']:.1f}"
            params = f"{metrics['parameters']:,}"
            print(f"{arch_name:<15} {mse:<12} {r2:<12} {time_str:<10} {params:<12}")

    print("=" * 80)

    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/architecture_comparison"
    os.makedirs(results_dir, exist_ok=True)

    results_file = f"{results_dir}/comparison_{args.embedding}_{timestamp}.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for arch_name, arch_results in results.items():
        if "error" in arch_results:
            json_results[arch_name] = {"error": arch_results["error"]}
        else:
            # Convert evaluation results to dict format for JSON serialization
            test_results = arch_results["test_results"]
            val_results = arch_results["val_results"]

            # Use results_to_dict if available, otherwise extract metrics
            try:
                test_results_dict = (
                    results_to_dict(test_results)
                    if hasattr(test_results, "metrics")
                    else test_results
                )
                val_results_dict = (
                    results_to_dict(val_results)
                    if hasattr(val_results, "metrics")
                    else val_results
                )
            except:
                # Fallback to direct metrics extraction
                test_results_dict = (
                    test_results.metrics
                    if hasattr(test_results, "metrics")
                    else test_results
                )
                val_results_dict = (
                    val_results.metrics
                    if hasattr(val_results, "metrics")
                    else val_results
                )

            json_results[arch_name] = {
                "test_results": test_results_dict,
                "val_results": val_results_dict,
                "training_metrics": {
                    k: v
                    for k, v in arch_results["training_metrics"].items()
                    if k != "pytorch_model"  # Exclude non-serializable model
                },
            }

    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "detailed_results": json_results,
                "experiment_config": {
                    "embedding": args.embedding,
                    "architectures_compared": architectures,
                    "data_shapes": {
                        "X_train": list(X_train.shape),
                        "y_train": list(y_train.shape),
                        "X_test": list(X_test.shape),
                        "y_test": list(y_test.shape),
                    },
                },
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {results_file}")

    # Log to MLflow if enabled
    if args.use_mlflow:
        try:
            # Log the comparison summary
            for arch_name, metrics in summary.items():
                if "error" not in metrics:
                    mlflow.log_metric(f"{arch_name}_test_mse", metrics["test_mse"])
                    mlflow.log_metric(f"{arch_name}_test_r2", metrics["test_r2"])
                    mlflow.log_metric(
                        f"{arch_name}_training_time", metrics["training_time"]
                    )
                    mlflow.log_metric(f"{arch_name}_parameters", metrics["parameters"])

            # Log the results file as an artifact
            mlflow.log_artifact(results_file)

        except Exception as e:
            print(f"Warning: MLflow logging failed for comparison: {e}")

    # Return results in expected format (model, val_results, test_results)
    # For comparison, we return the results dict as the "model"
    return results, None, None


def run_architecture_optimization(args, X_train, y_train, X_val, y_val, X_test, y_test):
    """Run architecture-specific hyperparameter optimization.

    Args:
        args: Command line arguments
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits

    Returns:
        tuple: (best_model, val_results, test_results)
    """
    architecture_type = args.architecture
    n_trials = args.architecture_trials
    metric = getattr(args, "optimization_metric", "val_mse")

    print(f"Optimizing hyperparameters for {architecture_type} architecture")
    print(f"Trials: {n_trials}, Metric: {metric}")

    # Run architecture-specific optimization
    best_params, study = optimize_architecture_hyperparameters(
        X_train,
        y_train,
        X_val,
        y_val,
        architecture_type=architecture_type,
        n_trials=n_trials,
        metric=metric,
    )

    print(f"Best parameters found: {best_params}")
    print(f"Best value: {study.best_value}")

    # Train final model with best parameters
    device = get_device(args)

    print("Training final model with optimized parameters...")
    final_model = train_torch_regressor(
        X_train,
        y_train,
        architecture_type=architecture_type,
        device=device,
        epochs=args.epochs,  # Use full epochs for final model
        verbose=True,
        **best_params,
    )

    # Evaluate final model
    val_results = evaluate_model(
        final_model, X_val, y_val, split_name="val", verbose=False, bootstrap_ci=True
    )
    test_results = evaluate_model(
        final_model, X_test, y_test, split_name="test", verbose=True, bootstrap_ci=True
    )

    # Save optimization results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/architecture_optimization"
    os.makedirs(results_dir, exist_ok=True)

    optimization_file = f"{results_dir}/optimization_{architecture_type}_{args.embedding}_{timestamp}.json"

    # Extract test metrics for JSON serialization
    try:
        test_metrics = (
            results_to_dict(test_results)
            if hasattr(test_results, "metrics")
            else test_results
        )
        val_metrics = (
            results_to_dict(val_results)
            if hasattr(val_results, "metrics")
            else val_results
        )
    except:
        test_metrics = (
            test_results.metrics if hasattr(test_results, "metrics") else test_results
        )
        val_metrics = (
            val_results.metrics if hasattr(val_results, "metrics") else val_results
        )

    optimization_results = {
        "best_params": best_params,
        "best_validation_value": study.best_value,
        "final_test_results": test_metrics,
        "final_val_results": val_metrics,
        "architecture_type": architecture_type,
        "n_trials": n_trials,
        "optimization_metric": metric,
        "experiment_config": {
            "embedding": args.embedding,
            "epochs": args.epochs,
            "data_shapes": {
                "X_train": list(X_train.shape),
                "y_train": list(y_train.shape),
                "X_test": list(X_test.shape),
                "y_test": list(y_test.shape),
            },
        },
    }

    with open(optimization_file, "w") as f:
        json.dump(optimization_results, f, indent=2)

    print(f"Optimization results saved to: {optimization_file}")

    # Log to MLflow if enabled
    if args.use_mlflow:
        try:
            # Log optimization-specific parameters
            mlflow.log_params({f"optimized_{k}": v for k, v in best_params.items()})
            mlflow.log_param("architecture_type", architecture_type)
            mlflow.log_param("optimization_trials", n_trials)
            mlflow.log_param("optimization_metric", metric)
            mlflow.log_metric("best_optimization_value", study.best_value)

            # Log the optimization results file
            mlflow.log_artifact(optimization_file)

        except Exception as e:
            print(f"Warning: MLflow logging failed for optimization: {e}")

    # Update args with best parameters for potential additional MLflow logging
    for param, value in best_params.items():
        setattr(args, f"optimized_{param}", value)

    return final_model, val_results, test_results


def save_model(model, model_path):
    """Save the trained model to disk.

    Args:
        model: Trained model to save
        model_path (str): Path to save the model
    """
    # Handle case where model is actually comparison results
    if isinstance(model, dict) and any(
        "test_results" in str(v) for v in model.values()
    ):
        print("Skipping model save for architecture comparison results")
        return

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def save_embedding_model(embedding_model, embedding_model_path, args):
    """Save the embedding model if needed.

    Args:
        embedding_model: Embedding model to save
        embedding_model_path (str): Path to save the embedding model
        args: Command line arguments
    """
    if (
        embedding_model is not None
        and not args.load_features
        and not args.save_features
    ):
        print(f"Saving embedding model to {embedding_model_path}...")
        joblib.dump(embedding_model, embedding_model_path)
