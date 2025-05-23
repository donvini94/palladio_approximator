"""
Configuration module for handling command line arguments and setup.
"""

import os
import argparse


def parse_args():
    """Parse command line arguments for the model training pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--model", type=str, choices=["rf", "ridge", "lasso", "torch"], default="torch"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["tfidf", "bert", "llama"],
        default="bert",
        help="Embedding model type to use (tfidf, bert, or llama)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=[
            "embedding_regressor",  # Original architecture
            "standard",
            "residual",
            "attention",
            "ensemble",
            "variational",
            "adaptive",
        ],
        default="embedding_regressor",
        help="Neural network architecture type",
    )

    # NEW: Architecture experiment options
    parser.add_argument(
        "--compare_architectures",
        action="store_true",
        help="Compare multiple architectures on the same dataset",
    )
    parser.add_argument(
        "--architectures_to_compare",
        nargs="+",
        choices=[
            "embedding_regressor",
            "standard",
            "residual",
            "attention",
            "ensemble",
            "variational",
            "adaptive",
        ],
        default=["embedding_regressor", "standard", "residual", "attention"],
        help="List of architectures to compare (used with --compare_architectures)",
    )
    parser.add_argument(
        "--optimize_architecture",
        action="store_true",
        help="Optimize hyperparameters for the specified architecture",
    )
    parser.add_argument(
        "--architecture_trials",
        type=int,
        default=30,
        help="Number of trials for architecture-specific hyperparameter optimization",
    )

    # LLaMA model options
    parser.add_argument(
        "--llama_model",
        type=str,
        default="codellama/CodeLlama-7b-hf",
        help="LLaMA model to use (only for --embedding=llama)",
    )
    parser.add_argument(
        "--llama_batch_size",
        type=int,
        default=4,
        help="Batch size for LLaMA processing (smaller values use less memory)",
    )
    parser.add_argument(
        "--no_half_precision",
        action="store_true",
        help="Disable half precision (FP16) for LLaMA models, uses more memory but may be more accurate",
    )
    parser.add_argument(
        "--use_8bit_llama",
        action="store_true",
        help="Use 8-bit quantization for LLaMA models (reduces memory usage by ~50%)",
    )
    parser.add_argument(
        "--use_4bit_llama",
        action="store_true",
        help="Use 4-bit quantization for LLaMA models (reduces memory usage by ~75%, recommended for 7B+ models)",
    )

    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["summary"],
        default="summary",
    )

    # Model hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)

    # PyTorch model parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)  # Updated default

    # Hyperparameter optimization parameters
    parser.add_argument(
        "--optimize_hyperparameters",
        action="store_true",
        help="Run hyperparameter optimization using Optuna before training",
    )
    parser.add_argument(
        "--no_optimize_hyperparameters",
        dest="optimize_hyperparameters",
        action="store_false",
        help="Disable hyperparameter optimization",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--optimization_metric",
        type=str,
        choices=["val_mse", "val_mae", "val_r2"],
        default="val_mse",
        help="Metric to optimize during hyperparameter tuning",
    )

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

    # Add no_ version for dataset loading options
    parser.add_argument(
        "--no_save_dataset",
        dest="save_dataset",
        action="store_false",
        help="Don't save the dataset splits to disk after loading",
    )
    parser.add_argument(
        "--no_load_dataset",
        dest="load_dataset",
        action="store_false",
        help="Don't load the dataset splits from disk (regenerate)",
    )

    parser.set_defaults(
        use_cuda=True,
        use_mlflow=True,
        use_gpu=True,
        save_features=True,
        load_features=True,
        save_dataset=True,
        load_dataset=True,
        optimize_hyperparameters=False,
        compare_architectures=False,
        optimize_architecture=False,
    )

    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    return args


def setup_environment():
    """Set up environment variables and initial configuration."""
    # Set memory-efficient PyTorch configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:512"
    )
    # Set tokenizers parallelism explicitly to prevent warnings with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_device(args):
    """Determine the appropriate device (CPU/GPU) based on args and availability.

    Args:
        args: Command line arguments with use_cuda flag

    Returns:
        str: 'cuda' or 'cpu' depending on availability and settings
    """
    import torch

    return "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"


def get_checkpoint_paths(args):
    """Generate paths for feature checkpoints and embedding models.

    Args:
        args: Command line arguments with embedding and prediction_mode

    Returns:
        tuple: (feature_checkpoint_path, embedding_model_path)
    """
    feature_path = (
        f"features/{args.embedding}_{args.prediction_mode}_features_checkpoint.pkl"
    )
    embedding_model_path = (
        f"embeddings/{args.embedding}_{args.prediction_mode}_embedding.pkl"
    )

    return feature_path, embedding_model_path


def get_model_path(args, experiment_id=None):
    """Generate the model save path based on args and optional experiment ID.

    Args:
        args: Command line arguments with model and prediction_mode
        experiment_id: Optional unique identifier for the experiment

    Returns:
        str: Path where the trained model should be saved
    """
    # Create models directory if it doesn't exist
    os.makedirs("models/saved", exist_ok=True)

    if experiment_id:
        # Use the experiment ID to create a unique filename
        return f"models/saved/{experiment_id}_model.pkl"
    else:
        # Use the basic format for backwards compatibility
        return f"{args.model}_{args.prediction_mode}_model.pkl"
