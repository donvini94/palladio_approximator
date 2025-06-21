#!/usr/bin/env python3
"""
Experiment Summary Tool

This script connects to MLflow, fetches all experiment runs, and generates 
a summary report showing the performance of different models, embeddings, 
and hyperparameter configurations.
"""

import os
import sys
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import argparse
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Summarize MLflow experiment results")
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_summary.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "markdown", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--sort", type=str, default="val_r2_score", help="Metric to sort results by"
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        help="Filter by model type (rf, ridge, lasso, torch)",
    )
    parser.add_argument(
        "--filter-embedding",
        type=str,
        help="Filter by embedding type (tfidf, bert, llama)",
    )
    parser.add_argument("--min-date", type=str, help="Min date filter (YYYY-MM-DD)")
    parser.add_argument("--max-date", type=str, help="Max date filter (YYYY-MM-DD)")

    return parser.parse_args()


def get_all_runs():
    """Retrieve all runs from MLflow."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("thesis")

    if not experiment:
        print("No experiments found in MLflow")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_r2_score DESC"],
    )
    return runs


def create_summary_dataframe(runs, sort_by="val_r2_score"):
    """Create a summary DataFrame from MLflow runs."""
    data = []

    for run in runs:
        run_data = {}

        # Get basic info
        run_data["run_id"] = run.info.run_id
        run_data["start_time"] = datetime.fromtimestamp(
            run.info.start_time / 1000.0
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Get parameters
        params = run.data.params
        run_data["model"] = params.get("model", "unknown")
        run_data["embedding"] = params.get("embedding", "unknown")
        run_data["prediction_mode"] = params.get("prediction_mode", "unknown")

        # Add model-specific hyperparameters
        if run_data["model"] == "rf":
            run_data["n_estimators"] = params.get("n_estimators", "")
        elif run_data["model"] in ["ridge", "lasso"]:
            run_data["alpha"] = params.get("alpha", "")
        elif run_data["model"] == "torch":
            run_data["batch_size"] = params.get("batch_size", "")
            run_data["epochs"] = params.get("epochs", "")

        # Get metrics
        metrics = run.data.metrics
        for metric_name, metric_value in metrics.items():
            run_data[metric_name] = metric_value

        data.append(run_data)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by the specified metric if it exists
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)

    return df


def apply_filters(df, args):
    """Apply filters to the DataFrame based on command line arguments."""
    filtered_df = df.copy()

    if args.filter_model:
        filtered_df = filtered_df[filtered_df["model"] == args.filter_model]

    if args.filter_embedding:
        filtered_df = filtered_df[filtered_df["embedding"] == args.filter_embedding]

    if args.min_date:
        filtered_df = filtered_df[filtered_df["start_time"] >= args.min_date]

    if args.max_date:
        filtered_df = filtered_df[filtered_df["start_time"] <= args.max_date]

    return filtered_df


def generate_summary_tables(df):
    """Generate summary tables by model and embedding types."""
    # Define metrics to aggregate (if available)
    metrics_to_agg = {"run_id": "count"}

    # Add validation metrics if available
    if "val_r2_score" in df.columns:
        metrics_to_agg["val_r2_score"] = "mean"
    if "val_mse" in df.columns:
        metrics_to_agg["val_mse"] = "mean"
    if "val_mae" in df.columns:
        metrics_to_agg["val_mae"] = "mean"

    # Add test metrics if available
    if "test_r2_score" in df.columns:
        metrics_to_agg["test_r2_score"] = "mean"
    if "test_mse" in df.columns:
        metrics_to_agg["test_mse"] = "mean"
    if "test_mae" in df.columns:
        metrics_to_agg["test_mae"] = "mean"

    # Average metrics by model type
    model_summary = df.groupby("model").agg(metrics_to_agg).reset_index()
    model_summary.rename(columns={"run_id": "count"}, inplace=True)

    # Average metrics by embedding type
    embedding_summary = df.groupby("embedding").agg(metrics_to_agg).reset_index()
    embedding_summary.rename(columns={"run_id": "count"}, inplace=True)

    # Combined model+embedding summary
    combined_summary = (
        df.groupby(["model", "embedding"]).agg(metrics_to_agg).reset_index()
    )
    combined_summary.rename(columns={"run_id": "count"}, inplace=True)

    return model_summary, embedding_summary, combined_summary


def save_output(df, model_summary, embedding_summary, combined_summary, args):
    """Save output in the specified format(s)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ["csv", "both"]:
        # Save detailed results
        output_path = (
            args.output if args.output.endswith(".csv") else f"{args.output}.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")

        # Save summary tables
        model_summary.to_csv(f"model_summary_{timestamp}.csv", index=False)
        embedding_summary.to_csv(f"embedding_summary_{timestamp}.csv", index=False)
        combined_summary.to_csv(f"combined_summary_{timestamp}.csv", index=False)

    if args.format in ["markdown", "both"]:
        # Save markdown tables
        with open(f"experiment_summary_{timestamp}.md", "w") as f:
            f.write("# Experiment Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Try to use to_markdown if tabulate is available, otherwise use to_string
            try:
                # Model summary
                f.write("## Model Performance Summary\n\n")
                f.write(model_summary.to_markdown(index=False))
                f.write("\n\n")

                # Embedding summary
                f.write("## Embedding Performance Summary\n\n")
                f.write(embedding_summary.to_markdown(index=False))
                f.write("\n\n")

                # Combined summary
                f.write("## Combined Model+Embedding Performance\n\n")
                f.write(combined_summary.to_markdown(index=False))
                f.write("\n\n")
            except ImportError:
                # Fallback to to_string if tabulate is not available
                print("Warning: tabulate package not found, using plain text format")

                f.write("## Model Performance Summary\n\n")
                f.write("```\n")
                f.write(model_summary.to_string(index=False))
                f.write("\n```\n\n")

                f.write("## Embedding Performance Summary\n\n")
                f.write("```\n")
                f.write(embedding_summary.to_string(index=False))
                f.write("\n```\n\n")

                f.write("## Combined Model+Embedding Performance\n\n")
                f.write("```\n")
                f.write(combined_summary.to_string(index=False))
                f.write("\n```\n\n")

            f.write("## Top 10 Performing Configurations\n\n")
            # Handle sorting like in console output
            sort_col = args.sort if args.sort in df.columns else "val_mse"
            if sort_col in df.columns:
                sort_ascending = (
                    True if "mse" in sort_col or "mae" in sort_col else False
                )
                top_configs = df.sort_values(
                    by=sort_col, ascending=sort_ascending
                ).head(10)
                f.write(
                    f"Sorted by: {sort_col} ({'ascending' if sort_ascending else 'descending'})\n\n"
                )
            else:
                # If preferred sort column not available, try alternatives
                for alt_col in ["val_mse", "val_mae", "val_r2_score", "start_time"]:
                    if alt_col in df.columns:
                        sort_ascending = (
                            True if "mse" in alt_col or "mae" in alt_col else False
                        )
                        top_configs = df.sort_values(
                            by=alt_col, ascending=sort_ascending
                        ).head(10)
                        f.write(
                            f"Note: Sorting by {alt_col} instead of {args.sort} (not available)\n\n"
                        )
                        break
                else:
                    # If no sorting column is available, just take the first 10
                    top_configs = df.head(10)
                    f.write(
                        f"Note: No metrics available for sorting, showing first 10 runs\n\n"
                    )

            try:
                f.write(top_configs.to_markdown(index=False))
            except ImportError:
                f.write("```\n")
                f.write(top_configs.to_string(index=False))
                f.write("\n```")

        print(f"Markdown summary saved to experiment_summary_{timestamp}.md")


def print_summary_to_console(
    df, model_summary, embedding_summary, combined_summary, args
):
    """Print a summary to the console."""
    print("\n=== EXPERIMENT SUMMARY ===\n")

    print("Model Performance Summary:")
    print(model_summary.to_string(index=False))
    print("\n")

    print("Embedding Performance Summary:")
    print(embedding_summary.to_string(index=False))
    print("\n")

    # Determine sort column for combined summary
    sort_col = args.sort if args.sort in combined_summary.columns else "val_mse"
    sort_ascending = True if "mse" in sort_col or "mae" in sort_col else False

    print(f"Combined Model+Embedding Performance (Top 5, sorted by {sort_col}):")
    try:
        print(
            combined_summary.sort_values(by=sort_col, ascending=sort_ascending)
            .head(5)
            .to_string(index=False)
        )
    except KeyError:
        print(combined_summary.head(5).to_string(index=False))
    print("\n")

    print("Top 5 Performing Configurations:")
    # Get columns available in the dataframe
    cols_to_show = ["model", "embedding"]

    # Add metrics if available
    for metric in [
        "val_r2_score",
        "val_mse",
        "val_mae",
        "test_r2_score",
        "test_mse",
        "test_mae",
    ]:
        if metric in df.columns:
            cols_to_show.append(metric)

    # Add model-specific hyperparameters
    if args.filter_model == "rf":
        cols_to_show.append("n_estimators")
    elif args.filter_model in ["ridge", "lasso"]:
        cols_to_show.append("alpha")
    elif args.filter_model == "svm":
        cols_to_show.extend(["C", "epsilon", "kernel", "gamma"])
    elif args.filter_model == "torch":
        cols_to_show.extend(["batch_size", "epochs"])

    # Sort by available metrics
    sort_col = args.sort if args.sort in df.columns else "val_mse"
    if sort_col in df.columns:
        sort_ascending = True if "mse" in sort_col or "mae" in sort_col else False
        top_configs = df.sort_values(by=sort_col, ascending=sort_ascending).head(5)
    else:
        # If preferred sort column not available, try alternatives
        for alt_col in ["val_mse", "val_mae", "val_r2_score", "start_time"]:
            if alt_col in df.columns:
                sort_ascending = True if "mse" in alt_col or "mae" in alt_col else False
                top_configs = df.sort_values(by=alt_col, ascending=sort_ascending).head(
                    5
                )
                print(
                    f"Note: Sorting by {alt_col} instead of {args.sort} (not available)"
                )
                break
        else:
            # If no sorting column is available, just take the first 5
            top_configs = df.head(5)
            print(f"Note: No metrics available for sorting, showing first 5 runs")

    # Filter columns to only include those in dataframe
    filtered_cols = [col for col in cols_to_show if col in df.columns]
    if filtered_cols:
        print(top_configs[filtered_cols].to_string(index=False))
    else:
        print("No relevant columns available to display")


def main():
    """Main execution function."""
    args = parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Get runs
    runs = get_all_runs()
    if not runs:
        print("No runs found in MLflow")
        return 1

    # Create summary DataFrame
    df = create_summary_dataframe(runs, sort_by=args.sort)

    # Apply filters
    df = apply_filters(df, args)
    if df.empty:
        print("No runs match the specified filters")
        return 1

    # Generate summary tables
    model_summary, embedding_summary, combined_summary = generate_summary_tables(df)

    # Print summary to console
    print_summary_to_console(
        df, model_summary, embedding_summary, combined_summary, args
    )

    # Save output
    save_output(df, model_summary, embedding_summary, combined_summary, args)

    print(f"\nTotal runs analyzed: {len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
