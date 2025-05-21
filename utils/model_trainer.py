"""
Model training orchestration module.
"""

import time
import joblib
from models.rf_model import train_random_forest
from models.linear_model import train_linear_model
from models.torch_model import train_torch_regressor
from evaluate import evaluate_model
from utils.config import get_device, get_model_path


def train_model(args, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model based on the selected type.

    Args:
        args: Command line arguments
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels

    Returns:
        tuple: (trained_model, val_results, test_results)
    """

    print(f"=== Starting model training at {time.strftime('%H:%M:%S')} ===")
    print(
        f"Training {args.model} model on data with shape: X_train={X_train.shape}, y_train={y_train.shape}"
    )

    # Perform hyperparameter optimization if requested
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
        print(f"Training PyTorch model on {device}...")

        # Extract PyTorch-specific parameters from args
        torch_params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
        }

        # Add additional parameters if they were optimized
        for param in ["learning_rate", "hidden_dims", "dropout_rate"]:
            if hasattr(args, param):
                torch_params[param] = getattr(args, param)

        model = train_torch_regressor(X_train, y_train, **torch_params)

        # Store model path in training_metrics for later use
        if hasattr(model, "training_metrics"):
            model.training_metrics["model_path"] = get_model_path(args)

    elif args.model in ("ridge", "lasso"):
        print("Training linear model...")

        # Extract linear model-specific parameters from args
        linear_params = {"model_type": args.model, "alpha": args.alpha}

        # Add additional parameters if they were optimized
        for param in ["solver", "max_iter"]:
            if hasattr(args, param):
                linear_params[param] = getattr(args, param)

        model = train_linear_model(X_train, y_train, **linear_params)
    else:
        raise ValueError(
            "Unsupported model type. Choose 'rf', 'torch', 'ridge', or 'lasso'"
        )

    print(f"=== Model training completed at {time.strftime('%H:%M:%S')} ===")

    # Evaluate the model
    print("Evaluating model...")
    val_results = evaluate_model(model, X_val, y_val, split_name="val")
    test_results = evaluate_model(model, X_test, y_test, split_name="test")
    print(f"Validation results: {val_results}")
    print(f"Test results: {test_results}")

    return model, val_results, test_results


def save_model(model, model_path):
    """Save the trained model to disk.

    Args:
        model: Trained model to save
        model_path (str): Path to save the model
    """
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
