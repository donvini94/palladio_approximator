import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import mlflow
import joblib
from pathlib import Path
import re

# Set style for consistent, publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)
FIGURE_DIR = "figures"


def create_dirs():
    """Create necessary directories for saving figures"""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIGURE_DIR, "performance"), exist_ok=True)
    os.makedirs(os.path.join(FIGURE_DIR, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(FIGURE_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(FIGURE_DIR, "learning_curves"), exist_ok=True)
    os.makedirs(os.path.join(FIGURE_DIR, "attention"), exist_ok=True)


def performance_comparison_dashboard(runs=None, metric="test_mse", sort_by=None):
    """
    Create visualizations that compare the performance of different model combinations.

    Args:
        runs: List of MLflow run data. If None, fetches all runs in the current experiment.
        metric: Metric to compare (e.g., 'test_mse', 'val_r2')
        sort_by: Column to sort by. If None, sorts by the metric.

    Returns:
        Dashboard data as DataFrame
    """
    create_dirs()

    if runs is None:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("dsl-performance-prediction")
        if experiment is None:
            print(
                "No MLflow experiment found. Create one first by running training with --use_mlflow"
            )
            return None

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # Extract metrics and parameters
    data = []
    for run in runs:
        if metric not in run.data.metrics:
            continue

        row = {
            "run_id": run.info.run_id,
            "embedding": run.data.params.get("embedding", "unknown"),
            "model_type": run.data.params.get("model_type", "unknown"),
            "prediction_mode": run.data.params.get("prediction_mode", "unknown"),
            metric: run.data.metrics.get(metric, np.nan),
        }

        # Collect all metrics for this run
        for m_key, m_value in run.data.metrics.items():
            if m_key != metric:
                row[m_key] = m_value

        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        print(f"No runs found with metric {metric}")
        return None

    # Sort by specified column or by metric
    if sort_by:
        df = df.sort_values(by=sort_by)
    else:
        df = df.sort_values(by=metric)

    # Create bar plot for model comparison
    plt.figure(figsize=(12, 6))

    # Create a compound label for each run
    df["run_label"] = df.apply(
        lambda x: f"{x['embedding']} + {x['model_type']}", axis=1
    )

    # Create grouped bar chart
    ax = sns.barplot(x="run_label", y=metric, data=df, palette="viridis")

    # Format the chart
    plt.title(f"Performance Comparison by Model Configuration ({metric})")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model Configuration")
    plt.ylabel(metric.replace("_", " ").title())
    plt.tight_layout()

    # Add values on top of each bar
    for i, v in enumerate(df[metric]):
        ax.text(
            i,
            v + 0.01 * max(df[metric]),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    # Save the figure
    plt.savefig(
        os.path.join(FIGURE_DIR, "performance", f"model_comparison_{metric}.png"),
        dpi=300,
    )
    plt.close()

    # Create heatmap for multi-metric comparison if multiple metrics exist
    metrics = [
        col for col in df.columns if col.startswith("test_") or col.startswith("val_")
    ]

    if len(metrics) > 1:
        # Pivot the data to create a heatmap (embedding + model vs metrics)
        pivot_df = df.pivot_table(index="run_label", values=metrics)

        # Normalize the values for better visualization
        norm_df = pivot_df.copy()
        for col in norm_df.columns:
            if "r2" in col:  # Higher is better for R2
                norm_df[col] = (norm_df[col] - norm_df[col].min()) / (
                    norm_df[col].max() - norm_df[col].min()
                )
            else:  # Lower is better for MSE, MAE
                norm_df[col] = 1 - (norm_df[col] - norm_df[col].min()) / (
                    norm_df[col].max() - norm_df[col].min()
                )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            norm_df, annot=pivot_df.round(4), cmap="viridis", fmt=".4f", linewidths=0.5
        )
        plt.title("Multi-Metric Performance Comparison")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURE_DIR, "performance", "metric_heatmap.png"), dpi=300
        )
        plt.close()

    return df


def prediction_error_analysis(model, X_test, y_test, embedding_type, model_type):
    """
    Create scatter plots showing predicted vs. actual response times

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        embedding_type: Type of embedding used (tfidf, bert, etc.)
        model_type: Type of model used (rf, torch, etc.)

    Returns:
        Dictionary with error analysis metrics
    """
    create_dirs()

    # Get predictions
    predictions = model.predict(X_test)

    # Ensure shapes match
    if len(predictions.shape) == 1 and len(y_test.shape) > 1:
        predictions = predictions.reshape(-1, 1)

    # Calculate errors
    mse = mean_squared_error(y_test, predictions, multioutput="raw_values")
    mae = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    r2 = r2_score(y_test, predictions, multioutput="raw_values")

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # If multi-output, use first dimension for visualization
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        plt.scatter(y_test[:, 0], predictions[:, 0], alpha=0.6)
        plt.xlabel("Actual Response Time")
        plt.ylabel("Predicted Response Time")
        plt.title(
            f"Actual vs. Predicted Response Times ({embedding_type} + {model_type})"
        )

        # Add diagonal perfect prediction line
        min_val = min(np.min(y_test[:, 0]), np.min(predictions[:, 0]))
        max_val = max(np.max(y_test[:, 0]), np.max(predictions[:, 0]))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        # Add error statistics
        plt.annotate(
            f"MSE: {mse[0]:.4f}\nMAE: {mae[0]:.4f}\nR²: {r2[0]:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )

        # Calculate and plot residuals
        residuals = predictions[:, 0] - y_test[:, 0]

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR, "predictions", f"{embedding_type}_{model_type}_scatter.png"
            ),
            dpi=300,
        )
        plt.close()

        # Create residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions[:, 0], residuals, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Response Time")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot ({embedding_type} + {model_type})")

        # Add a histogram of residuals on the side
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "predictions",
                f"{embedding_type}_{model_type}_residuals.png",
            ),
            dpi=300,
        )
        plt.close()

        # Create histogram of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color="r", linestyle="--")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Residuals ({embedding_type} + {model_type})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "predictions",
                f"{embedding_type}_{model_type}_residual_dist.png",
            ),
            dpi=300,
        )
        plt.close()

    else:
        # Single output case
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.xlabel("Actual Response Time")
        plt.ylabel("Predicted Response Time")
        plt.title(
            f"Actual vs. Predicted Response Times ({embedding_type} + {model_type})"
        )

        # Add diagonal perfect prediction line
        min_val = min(np.min(y_test), np.min(predictions))
        max_val = max(np.max(y_test), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        # Add error statistics
        plt.annotate(
            f"MSE: {mse.item():.4f}\nMAE: {mae.item():.4f}\nR²: {r2.item():.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR, "predictions", f"{embedding_type}_{model_type}_scatter.png"
            ),
            dpi=300,
        )
        plt.close()

        # Calculate and plot residuals
        residuals = predictions.flatten() - y_test.flatten()

        # Create residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Response Time")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot ({embedding_type} + {model_type})")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "predictions",
                f"{embedding_type}_{model_type}_residuals.png",
            ),
            dpi=300,
        )
        plt.close()

        # Create histogram of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color="r", linestyle="--")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Residuals ({embedding_type} + {model_type})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "predictions",
                f"{embedding_type}_{model_type}_residual_dist.png",
            ),
            dpi=300,
        )
        plt.close()

    # Return error metrics
    return {
        "mse": mse.tolist() if hasattr(mse, "tolist") else float(mse),
        "mae": mae.tolist() if hasattr(mae, "tolist") else float(mae),
        "r2": r2.tolist() if hasattr(r2, "tolist") else float(r2),
    }


def generate_learning_curves(
    model_type, embedding_type, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
):
    """
    Generate learning curves showing how performance improves with more training data.
    This function runs multiple training iterations with different data sizes.

    Args:
        model_type: Type of model to train (rf, torch, etc.)
        embedding_type: Type of embedding to use (tfidf, bert, etc.)
        train_sizes: List of training data proportions to use

    Returns:
        DataFrame with learning curve data
    """
    create_dirs()
    from train import main
    import argparse
    from dataset import load_dataset

    class Args:
        def __init__(self):
            self.data_dir = "data"
            self.model = None
            self.embedding = None
            self.prediction_mode = "summary"
            self.n_estimators = 100
            self.alpha = 1.0
            self.epochs = 100
            self.batch_size = 64
            self.use_cuda = True
            self.use_mlflow = False
            self.save_features = False
            self.load_features = False
            self.use_gpu = True
            self.llama_model = "codellama/CodeLlama-7b-hf"
            self.llama_batch_size = 4
            self.no_half_precision = False
            self.use_8bit_llama = False
            self.use_4bit_llama = False

    # Load all available data
    train_samples, val_samples, test_samples = load_dataset("data")

    # Prepare result storage
    results = []

    # Train models with varying amounts of data
    for size in train_sizes:
        n_samples = int(len(train_samples) * size)
        if n_samples < 10:  # Ensure minimum number of samples
            n_samples = 10

        print(
            f"\n=== Training with {n_samples}/{len(train_samples)} samples ({size:.2%}) ==="
        )

        # Create args object
        args = Args()
        args.model = model_type
        args.embedding = embedding_type

        # Train the model with the subset of data
        from feature_extraction import (
            build_tfidf_features,
            build_bert_features,
            build_llama_features,
        )
        from models.rf_model import train_random_forest
        from models.linear_model import train_linear_model
        from models.torch_model import train_torch_regressor
        from evaluate import evaluate_model

        # Use only a subset of training data
        sub_train_samples = train_samples[:n_samples]

        # Extract features
        if embedding_type == "tfidf":
            X_train, y_train, X_val, y_val, X_test, y_test, _ = build_tfidf_features(
                sub_train_samples, val_samples, test_samples
            )
        elif embedding_type == "bert":
            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            X_train, y_train, X_val, y_val, X_test, y_test, _, _ = build_bert_features(
                sub_train_samples, val_samples, test_samples, device=device
            )
        elif embedding_type == "llama":
            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            X_train, y_train, X_val, y_val, X_test, y_test, _, _ = build_llama_features(
                sub_train_samples,
                val_samples,
                test_samples,
                model_name=args.llama_model,
                device=device,
                use_half_precision=not args.no_half_precision,
                use_8bit=args.use_8bit_llama,
                use_4bit=args.use_4bit_llama,
            )

        # Train model
        if model_type == "rf":
            model = train_random_forest(
                X_train, y_train, n_estimators=args.n_estimators
            )
        elif model_type == "torch":
            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            model = train_torch_regressor(
                X_train,
                y_train,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        elif model_type in ("ridge", "lasso"):
            model = train_linear_model(
                X_train, y_train, model_type=model_type, alpha=args.alpha
            )

        # Evaluate
        train_metrics = evaluate_model(
            model, X_train, y_train, split_name="train", verbose=False
        )
        val_metrics = evaluate_model(
            model, X_val, y_val, split_name="val", verbose=False
        )
        test_metrics = evaluate_model(
            model, X_test, y_test, split_name="test", verbose=False
        )

        # Add to results
        results.append(
            {
                "train_size": size,
                "n_samples": n_samples,
                **train_metrics,
                **val_metrics,
                **test_metrics,
            }
        )

        # Free memory
        del model, X_train, y_train, X_val, y_val, X_test, y_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Plot learning curves
    metrics = ["mse", "mae", "r2"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Plot training curve
        plt.plot(
            df["n_samples"],
            df[f"train_{metric}"],
            marker="o",
            label=f"Training {metric}",
        )

        # Plot validation curve
        plt.plot(
            df["n_samples"],
            df[f"val_{metric}"],
            marker="s",
            label=f"Validation {metric}",
        )

        # Plot test curve
        plt.plot(
            df["n_samples"], df[f"test_{metric}"], marker="^", label=f"Test {metric}"
        )

        plt.xscale("log")
        plt.xlabel("Number of Training Samples")
        plt.ylabel(metric.upper())
        plt.title(
            f"Learning Curve - {metric.upper()} ({embedding_type} + {model_type})"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "learning_curves",
                f"{embedding_type}_{model_type}_{metric}_curve.png",
            ),
            dpi=300,
        )
        plt.close()

    # Save results
    df.to_csv(
        os.path.join(
            FIGURE_DIR,
            "learning_curves",
            f"{embedding_type}_{model_type}_learning_curve_data.csv",
        ),
        index=False,
    )

    return df


def visualize_embedding_space(
    X, labels=None, method="pca", embedding_type="unknown", n_components=2
):
    """
    Visualize the embedding space using dimensionality reduction.

    Args:
        X: Embedding matrix
        labels: Labels or categories for coloring points
        method: Dimensionality reduction method ('pca' or 'tsne')
        embedding_type: Type of embedding used
        n_components: Number of components for visualization (2 or 3)
    """
    create_dirs()

    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        title_prefix = "PCA"
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        title_prefix = "t-SNE"
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    # Convert sparse matrix if needed
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = X

    # Apply reduction
    X_reduced = reducer.fit_transform(X_dense)

    # Visualization
    if n_components == 2:
        plt.figure(figsize=(10, 8))

        if labels is not None:
            # Use labels for coloring
            scatter = plt.scatter(
                X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7
            )
            plt.colorbar(scatter, label="Label Value")
        else:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)

        plt.title(f"{title_prefix} Visualization of {embedding_type} Embeddings")
        plt.xlabel(f"{title_prefix} Component 1")
        plt.ylabel(f"{title_prefix} Component 2")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR, "embeddings", f"{embedding_type}_{method.lower()}_2d.png"
            ),
            dpi=300,
        )
        plt.close()

    elif n_components == 3:
        # 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if labels is not None:
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                X_reduced[:, 2],
                c=labels,
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Label Value")
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.7)

        ax.set_title(f"{title_prefix} Visualization of {embedding_type} Embeddings")
        ax.set_xlabel(f"{title_prefix} Component 1")
        ax.set_ylabel(f"{title_prefix} Component 2")
        ax.set_zlabel(f"{title_prefix} Component 3")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURE_DIR, "embeddings", f"{embedding_type}_{method.lower()}_3d.png"
            ),
            dpi=300,
        )
        plt.close()

    # Calculate explained variance for PCA
    if method.lower() == "pca":
        return {
            "explained_variance_ratio": reducer.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(
                reducer.explained_variance_ratio_
            ).tolist(),
        }

    return {"reduced_data": X_reduced}


def visualize_training_metrics(
    run_id=None, metric_names=None, output_dir="figures/training"
):
    """
    Visualize training metrics from MLflow for neural network training.
    Creates line charts showing the progression of metrics over epochs.

    Args:
        run_id: Specific MLflow run ID to visualize (if None, uses the latest run)
        metric_names: List of metric names to visualize (if None, uses default metrics)
        output_dir: Directory to save the visualizations

    Returns:
        DataFrame with metrics data
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)

    if metric_names is None:
        # Default metrics to visualize
        metric_names = [
            "train_loss",
            "val_loss",
            "train_mse",
            "val_mse",
            "train_mae",
            "val_mae",
            "learning_rate",
        ]

    # Connect to MLflow
    client = mlflow.tracking.MlflowClient()

    # Get experiment - try with the name used in train.py
    experiment = mlflow.get_experiment_by_name("dsl-performance-prediction")

    # Try alternative experiment names if not found
    if experiment is None:
        print("Trying alternative experiment names...")
        for exp_name in [
            "dsl_performance_prediction",
            "palladio-performance-prediction",
        ]:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is not None:
                print(f"Found experiment: {exp_name}")
                break

    # Check if still None
    if experiment is None:
        print(
            "No MLflow experiment found. Create one first by running training with --use_mlflow"
        )
        return None

    # Get run
    if run_id is None:
        # Get the latest run with PyTorch model - try different parameter names for model type
        filter_strings = [
            "params.model_type = 'torch_regressor'",
            "params.model_type = 'torch'",
            "params.model = 'torch'",
        ]

        runs = []
        for filter_string in filter_strings:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"],
            )
            if runs:
                print(f"Found runs using filter: {filter_string}")
                break
        if not runs:
            print("No PyTorch model runs found")
            return None
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    # Get metrics
    metrics_data = {}
    for metric_name in metric_names:
        # Get metric history
        metric_history = client.get_metric_history(run_id, metric_name)

        # Skip if no data for this metric
        if not metric_history:
            continue

        # Extract step and value
        steps = [m.step for m in metric_history]
        values = [m.value for m in metric_history]

        # Store in dictionary
        metrics_data[metric_name] = {"steps": steps, "values": values}

    # For each metric, create a line chart
    for metric_name, data in metrics_data.items():
        if not data["steps"]:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(data["steps"], data["values"], marker="o", linestyle="-", alpha=0.7)

        # Format title and labels
        title = f"{metric_name.replace('_', ' ').title()} over Training Epochs"
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.replace("_", " ").title())

        # For learning rate, use log scale
        if "learning_rate" in metric_name.lower():
            plt.yscale("log")

        # Add grid for readability
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_history.png"), dpi=300)
        plt.close()

    # Also create a combined plot for loss metrics
    loss_metrics = [
        m
        for m in metrics_data.keys()
        if "loss" in m.lower() and not m.startswith("batch_")
    ]
    if loss_metrics:
        plt.figure(figsize=(12, 6))

        for metric in loss_metrics:
            plt.plot(
                metrics_data[metric]["steps"],
                metrics_data[metric]["values"],
                marker="o",
                linestyle="-",
                alpha=0.7,
                label=metric.replace("_", " ").title(),
            )

        plt.title("Loss Metrics over Training Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_loss_history.png"), dpi=300)
        plt.close()

    # Create batch-level plot if available
    batch_metrics = client.get_metric_history(run_id, "batch_train_loss")
    if batch_metrics:
        plt.figure(figsize=(12, 6))

        # Extract batch indices and values
        batch_indices = [m.step for m in batch_metrics]
        batch_values = [m.value for m in batch_metrics]

        plt.plot(batch_indices, batch_values, alpha=0.7, linewidth=1)

        # Format title and labels
        plt.title("Training Loss at Batch Level")
        plt.xlabel("Batch")
        plt.ylabel("Loss")

        # Add grid for readability
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_loss_history.png"), dpi=300)
        plt.close()

    # Create a DataFrame with all metrics for easier analysis
    df_data = {
        "epoch": list(
            range(1, max([max(data["steps"]) for data in metrics_data.values()]) + 1)
        )
    }
    for metric_name, data in metrics_data.items():
        # Create a dictionary with epoch as key and value as value
        value_dict = dict(zip(data["steps"], data["values"]))
        # Add to DataFrame
        df_data[metric_name] = [
            value_dict.get(i, None) for i in range(1, max(data["steps"]) + 1)
        ]

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)

    return df


def create_experiment_summary():
    """
    Create a comprehensive summary of all experiments run with MLflow.

    Returns:
        DataFrame with experiment summary
    """
    create_dirs()

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("dsl-performance-prediction")

    if experiment is None:
        print("No MLflow experiment found.")
        return None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs:
        print("No runs found in the experiment.")
        return None

    # Extract data from all runs
    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time": (
                pd.to_datetime(run.info.end_time, unit="ms")
                if run.info.end_time
                else None
            ),
        }

        # Add all parameters
        for param_key, param_val in run.data.params.items():
            row[f"param_{param_key}"] = param_val

        # Add all metrics
        for metric_key, metric_val in run.data.metrics.items():
            row[f"metric_{metric_key}"] = metric_val

        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate run time
    if "start_time" in df.columns and "end_time" in df.columns:
        df["duration_minutes"] = (
            df["end_time"] - df["start_time"]
        ).dt.total_seconds() / 60

    # Save summary to CSV
    df.to_csv(os.path.join(FIGURE_DIR, "experiment_summary.csv"), index=False)

    # Create a pivot table for model performance
    if "param_model_type" in df.columns and "param_embedding" in df.columns:
        # Find all available test metrics
        test_metrics = [col for col in df.columns if col.startswith("metric_test_")]

        if test_metrics:
            for metric in test_metrics:
                metric_name = metric.replace("metric_", "")

                # Create pivot table
                pivot = df.pivot_table(
                    values=metric,
                    index="param_model_type",
                    columns="param_embedding",
                    aggfunc="mean",
                )

                # Plot heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    pivot, annot=True, cmap="viridis", fmt=".4f", linewidths=0.5
                )
                plt.title(f"Average {metric_name} by Model Type and Embedding")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        FIGURE_DIR, "performance", f"heatmap_{metric_name}.png"
                    ),
                    dpi=300,
                )
                plt.close()

    return df


def prediction_time_analysis(
    models, X_test, model_names, embedding_type, batch_size=32
):
    """
    Analyze prediction time for different models.

    Args:
        models: List of trained models
        X_test: Test features
        model_names: List of model names
        embedding_type: Type of embedding used
        batch_size: Batch size for prediction

    Returns:
        DataFrame with timing results
    """
    create_dirs()
    import time

    assert len(models) == len(
        model_names
    ), "Number of models must match number of model names"

    # Prepare results storage
    results = []

    # Convert sparse matrix if needed
    if hasattr(X_test, "toarray"):
        X_dense = X_test.toarray()
    else:
        X_dense = X_test

    # Time predictions for each model
    for model, name in zip(models, model_names):
        # Time batch prediction
        start_time = time.time()
        _ = model.predict(X_test)
        batch_time = time.time() - start_time

        # Time individual predictions
        individual_times = []
        for i in range(min(100, len(X_dense))):  # Limit to 100 samples
            x = X_dense[i : i + 1]
            start_time = time.time()
            _ = model.predict(x)
            individual_times.append(time.time() - start_time)

        avg_individual_time = np.mean(individual_times)

        results.append(
            {
                "model": name,
                "batch_prediction_time": batch_time,
                "avg_individual_prediction_time": avg_individual_time,
                "speedup_factor": (
                    avg_individual_time * len(X_test) / batch_time
                    if batch_time > 0
                    else float("inf")
                ),
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Plot timing comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(
        x - width / 2, df["batch_prediction_time"], width, label="Batch Prediction Time"
    )
    plt.bar(
        x + width / 2,
        df["avg_individual_prediction_time"] * len(X_test),
        width,
        label="Total Individual Prediction Time",
    )

    plt.xticks(x, model_names, rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title(f"Prediction Time Comparison ({embedding_type} Embeddings)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR, "performance", f"{embedding_type}_prediction_time.png"
        ),
        dpi=300,
    )
    plt.close()

    # Plot speedup factors
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, df["speedup_factor"], color="teal")
    plt.yscale("log")
    plt.xlabel("Model")
    plt.ylabel("Speedup Factor (log scale)")
    plt.title("Batch vs. Individual Prediction Speedup")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR, "performance", f"{embedding_type}_speedup_factors.png"
        ),
        dpi=300,
    )
    plt.close()

    return df


def generate_dashboard_from_files(data_dir=".", pattern="*_model.pkl"):
    """
    Generate a dashboard from saved model files.

    Args:
        data_dir: Directory to search for model files
        pattern: Glob pattern for model files

    Returns:
        Summary of processed models
    """
    create_dirs()

    # Find model files
    model_files = glob.glob(os.path.join(data_dir, pattern))

    if not model_files:
        print(f"No model files found matching pattern {pattern} in {data_dir}")
        return None

    # Group by embedding type and model type
    models_by_type = {}

    for model_file in model_files:
        filename = os.path.basename(model_file)
        match = re.match(r"(.+?)_(.+?)_model\.pkl", filename)

        if not match:
            continue

        model_type, prediction_mode = match.groups()

        # Find corresponding feature files
        for embedding_type in ["tfidf", "bert", "llama"]:
            feature_file = os.path.join(
                data_dir, f"{embedding_type}_{prediction_mode}_features_checkpoint.pkl"
            )

            if os.path.exists(feature_file):
                key = (embedding_type, model_type, prediction_mode)

                if key not in models_by_type:
                    models_by_type[key] = {
                        "model_file": model_file,
                        "feature_file": feature_file,
                    }

    # Process each model
    results = []

    for (embedding_type, model_type, prediction_mode), files in models_by_type.items():
        print(f"Processing {embedding_type} + {model_type} ({prediction_mode})...")

        try:
            # Load model
            model = joblib.load(files["model_file"])

            # Load features
            features = joblib.load(files["feature_file"])
            X_test = features["X_test"]
            y_test = features["y_test"]

            # Analyze predictions
            metrics = prediction_error_analysis(
                model, X_test, y_test, embedding_type, model_type
            )

            # Store results
            results.append(
                {
                    "embedding_type": embedding_type,
                    "model_type": model_type,
                    "prediction_mode": prediction_mode,
                    **metrics,
                }
            )

        except Exception as e:
            print(f"Error processing {embedding_type} + {model_type}: {e}")

    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)

        # Save summary
        df.to_csv(os.path.join(FIGURE_DIR, "model_summary.csv"), index=False)

        # Create performance comparison chart
        plt.figure(figsize=(12, 6))

        # Group by embedding and model type
        df["model_config"] = df.apply(
            lambda x: f"{x['embedding_type']} + {x['model_type']}", axis=1
        )

        # Plot mean squared error
        if "mse" in df.columns:
            if isinstance(df["mse"].iloc[0], list):
                df["mse_value"] = df["mse"].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )
            else:
                df["mse_value"] = df["mse"]

            plt.figure(figsize=(12, 6))
            sns.barplot(x="model_config", y="mse_value", data=df, palette="viridis")
            plt.title("MSE by Model Configuration")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Mean Squared Error")
            plt.tight_layout()

            plt.savefig(
                os.path.join(FIGURE_DIR, "performance", "model_mse_comparison.png"),
                dpi=300,
            )
            plt.close()

        return df

    return None


def visualize_performance_context(
    run_id=None, output_dir="figures/performance_context"
):
    """
    Visualize performance context metrics from MLflow for a specific run.

    Args:
        run_id: Specific MLflow run ID (if None, uses latest run)
        output_dir: Directory to save visualizations

    Returns:
        Path to the performance summary markdown file
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)

    # Connect to MLflow
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name("dsl-performance-prediction")

    if experiment is None:
        print("No MLflow experiment found.")
        return None

    # Get run
    if run_id is None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
        )
        if not runs:
            print("No runs found")
            return None
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    # Check if performance summary exists
    artifacts = client.list_artifacts(run_id)
    summary_path = None
    for artifact in artifacts:
        if artifact.path == "performance_summary.md":
            # Download artifact
            temp_path = client.download_artifacts(run_id, artifact.path)

            # Copy to output directory
            output_path = os.path.join(output_dir, "performance_summary.md")
            shutil.copy(temp_path, output_path)
            summary_path = output_path

            # Also print contents to console
            with open(output_path, "r") as f:
                summary_content = f.read()
                print("\n" + "=" * 40)
                print("PERFORMANCE SUMMARY")
                print("=" * 40)
                print(summary_content)
                print("=" * 40 + "\n")

            break

    # Copy visualization images if available
    for image_name in [
        "mse_baseline_comparison.png",
        "error_distribution_context.png",
        "performance_quadrant.png",
    ]:
        try:
            temp_path = client.download_artifacts(run_id, image_name)
            output_path = os.path.join(output_dir, image_name)
            shutil.copy(temp_path, output_path)
        except:
            pass

    return summary_path


if __name__ == "__main__":
    # Example usage
    create_dirs()

    # Create experiment summary from MLflow
    print("Creating experiment summary...")
    summary_df = create_experiment_summary()

    # Generate dashboard from saved models
    print("Generating dashboard from saved models...")
    model_summary = generate_dashboard_from_files()

    if model_summary is not None:
        print(f"Processed {len(model_summary)} models.")
    else:
        print("No models processed. Try training some models first.")

    print("Visualization complete. Check the 'figures' directory for results.")
