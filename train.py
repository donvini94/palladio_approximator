import os
# Set tokenizers parallelism explicitly to prevent warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from dataset import load_dataset
from feature_extraction import (
    build_tfidf_features,
    build_bert_features,
)
from models.rf_model import train_random_forest
from models.linear_model import train_linear_model
from models.torch_model import train_torch_regressor

from evaluate import evaluate_model
import joblib
import torch
import numpy as np
import time
import sys
import warnings


def main(args):
    start_time = time.time()
    print(f"=== Starting execution at {time.strftime('%H:%M:%S')} ===")

    # Determine checkpoint path
    checkpoint_path = f"{args.embedding}_{args.prediction_mode}_features_checkpoint.pkl"
    embedding_model_path = f"{args.embedding}_{args.prediction_mode}_embedding.pkl"

    # Initialize variables to avoid UnboundLocalError
    embedding_model = None

    # First check if we should load features from disk
    if args.load_features and os.path.exists(checkpoint_path):
        print(f"Loading features from {checkpoint_path}...")
        try:
            feature_checkpoint = joblib.load(checkpoint_path)
            X_train = feature_checkpoint["X_train"]
            y_train = feature_checkpoint["y_train"]
            X_val = feature_checkpoint["X_val"]
            y_val = feature_checkpoint["y_val"]
            X_test = feature_checkpoint["X_test"]
            y_test = feature_checkpoint["y_test"]

            # Load embedding model if available
            if os.path.exists(embedding_model_path):
                print(f"Loading embedding model from {embedding_model_path}...")
                embedding_model = joblib.load(embedding_model_path)
            else:
                print("Embedding model not found. Only features will be available.")

            print(
                f"Loaded features successfully: X_train={X_train.shape}, y_train={y_train.shape}"
            )
            print(f"X_val={X_val.shape}, X_test={X_test.shape}")

            # Skip straight to model training
            features_loaded = True

        except Exception as e:
            print(f"Error loading features: {e}")
            print("Falling back to generating features from scratch")
            features_loaded = False
    else:
        if args.load_features:
            print(
                f"Feature checkpoint {checkpoint_path} not found. Generating features."
            )
        features_loaded = False

    # Only proceed with feature extraction if we couldn't load them
    if not features_loaded:
        if args.prediction_mode == "summary":
            print("Loading dataset...")
            train_samples, val_samples, test_samples = load_dataset(args.data_dir)
            print(f"Dataset loaded. Train samples: {len(train_samples)}")

            # Feature extraction
            print(f"Building features using {args.embedding} embedding...")
            if args.embedding == "tfidf":
                # For torch, we want to use full feature set with moderate dimensionality reduction
                if args.model == "torch":
                    # No limit on max features to capture all important terms
                    max_features = None
                    # Keep more components for better representation quality
                    n_components = 2000
                    # Only apply SVD if it maintains high explained variance
                    apply_truncated_svd = True
                    print(f"Using high-quality parameters for torch: max_features={max_features}, n_components={n_components}")
                else:
                    max_features = 10000
                    n_components = 1000
                    apply_truncated_svd = True
                
                X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = (
                    build_tfidf_features(
                        train_samples, 
                        val_samples, 
                        test_samples, 
                        max_features=max_features, 
                        apply_truncated_svd=apply_truncated_svd, 
                        n_components=n_components
                    )
                )
            elif args.embedding == "bert":
                device = (
                    "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
                )
                print(f"Using device: {device}")
                X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
                    build_bert_features(
                        train_samples, val_samples, test_samples, device=device
                    )
                )
                embedding_model = (tokenizer, model)

                print("Feature extraction completed successfully")
                print(
                    f"Feature shapes: X_train={X_train.shape}, y_train={y_train.shape}"
                )

                # Free up memory
                if device == "cuda":
                    print("Clearing GPU cache...")
                    torch.cuda.empty_cache()
            else:
                raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'bert'")

        else:
            raise ValueError("Unsupported prediction mode. Choose 'summary' ")

        # Save embeddings as checkpoint if requested
        if args.save_features:
            print(f"Saving extracted features to {checkpoint_path}...")
            feature_checkpoint = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }
            joblib.dump(feature_checkpoint, checkpoint_path)
            print(f"Features saved successfully")

            # Also save the embedding model for completeness
            if embedding_model is not None:
                print(f"Saving embedding model to {embedding_model_path}...")
                joblib.dump(embedding_model, embedding_model_path)

    if args.use_mlflow:
        import mlflow

        print("Setting up MLflow...")
        mlflow.set_experiment("dsl-performance-prediction")
        mlflow.start_run()
        mlflow.log_params(
            {
                "model_type": args.model,
                "embedding": args.embedding,
                "prediction_mode": args.prediction_mode,
                "n_estimators": args.n_estimators,
                "alpha": args.alpha,
            }
        )

    print(f"=== Starting model training at {time.strftime('%H:%M:%S')} ===")
    print(
        f"Training {args.model} model on data with shape: X_train={X_train.shape}, y_train={y_train.shape}"
    )

    # Train model based on selected type
    if args.model == "rf":
        model = train_random_forest(X_train, y_train, n_estimators=args.n_estimators)

    elif args.model == "torch":

        device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
        print(f"Training PyTorch model on {device}...")
        model = train_torch_regressor(
            X_train,
            y_train,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.model in ("ridge", "lasso"):
        print("Training linear model...")
        model = train_linear_model(
            X_train, y_train, model_type=args.model, alpha=args.alpha
        )
    else:
        raise ValueError(
            "Unsupported model type. Choose 'rf', 'torch', 'ridge', or 'lasso'"
        )

    print(f"=== Model training completed at {time.strftime('%H:%M:%S')} ===")
    print("Evaluating model...")
    val_results = evaluate_model(model, X_val, y_val, split_name="val")
    test_results = evaluate_model(model, X_test, y_test, split_name="test")
    print(f"Validation results: {val_results}")
    print(f"Test results: {test_results}")

    if args.use_mlflow:
        print("Logging results to MLflow...")
        mlflow.log_metrics(val_results)
        mlflow.log_metrics(test_results)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

    print("Saving model...")
    model_path = f"{args.model}_{args.prediction_mode}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save embedding model if we generated it but didn't save it yet
    if (
        embedding_model is not None
        and not args.load_features
        and not args.save_features
    ):
        print(f"Saving embedding model to {embedding_model_path}...")
        joblib.dump(embedding_model, embedding_model_path)

    elapsed_time = time.time() - start_time
    print(f"=== Total execution time: {elapsed_time:.2f} seconds ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--model", type=str, choices=["rf", "ridge", "lasso", "torch"], default="rf"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["tfidf", "bert"],
        default="bert",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["summary"],
        default="summary",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)

    # PyTorch model parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)

    # GPU-related flags
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for model training when possible",
    )
    parser.add_argument(
        "--no_cuda",
        dest="use_cuda",
        action="store_false",
        help="Disable CUDA for embedding extraction",
    )

    # Feature checkpointing
    parser.add_argument(
        "--save_features", action="store_true", help="Save extracted features to disk"
    )
    parser.add_argument(
        "--load_features",
        action="store_true",
        help="Load features from disk instead of extracting",
    )

    # MLflow
    parser.add_argument(
        "--no_mlflow", dest="use_mlflow", action="store_false", help="Disable MLflow"
    )

    parser.set_defaults(
        use_cuda=True,
        use_mlflow=True,
        use_gpu=True,
        save_features=True,
        load_features=False,
    )

    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    try:
        main(args)
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
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
