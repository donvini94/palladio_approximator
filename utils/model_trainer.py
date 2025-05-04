"""
Model training orchestration module.
"""
import torch
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
    print(f"Training {args.model} model on data with shape: X_train={X_train.shape}, y_train={y_train.shape}")

    # Train model based on selected type
    if args.model == "rf":
        model = train_random_forest(X_train, y_train, n_estimators=args.n_estimators)
    
    elif args.model == "torch":
        device = get_device(args)
        print(f"Training PyTorch model on {device}...")
        model = train_torch_regressor(
            X_train,
            y_train,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
        # Store model path in training_metrics for later use
        if hasattr(model, "training_metrics"):
            model.training_metrics["model_path"] = get_model_path(args)
    
    elif args.model in ("ridge", "lasso"):
        print("Training linear model...")
        model = train_linear_model(
            X_train, y_train, model_type=args.model, alpha=args.alpha
        )
    else:
        raise ValueError("Unsupported model type. Choose 'rf', 'torch', 'ridge', or 'lasso'")

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


def save_embedding_model_if_needed(embedding_model, embedding_model_path, args):
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