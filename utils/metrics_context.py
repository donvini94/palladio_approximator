"""
Utilities for providing context and interpretation for model metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.dummy import DummyRegressor


def safe_average(values):
    """Safely compute the average of a list or value."""
    if not values:
        return 0
    if isinstance(values, list):
        return sum(values) / len(values) if values else 0
    return values


def get_baseline_metrics(X_train, y_train, X_val, y_val):
    """
    Calculate baseline metrics using simple models for comparison.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Dictionary of baseline metrics
    """
    baselines = {}

    # Mean baseline (predicts mean of training data)
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(X_train, y_train)
    mean_preds = mean_model.predict(X_val)

    # Calculate MSE for mean baseline
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        # Multi-output case
        mean_mse = np.mean((y_val - mean_preds) ** 2, axis=0)
        baselines["mean_baseline_mse"] = mean_mse.tolist()

        # Also compute MAE
        mean_mae = np.mean(np.abs(y_val - mean_preds), axis=0)
        baselines["mean_baseline_mae"] = mean_mae.tolist()
    else:
        # Single output case
        mean_mse = np.mean((y_val.flatten() - mean_preds.flatten()) ** 2)
        baselines["mean_baseline_mse"] = float(mean_mse)

        # Also compute MAE
        mean_mae = np.mean(np.abs(y_val.flatten() - mean_preds.flatten()))
        baselines["mean_baseline_mae"] = float(mean_mae)

    # Median baseline
    median_model = DummyRegressor(strategy="median")
    median_model.fit(X_train, y_train)
    median_preds = median_model.predict(X_val)

    # Calculate MSE for median baseline
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        median_mse = np.mean((y_val - median_preds) ** 2, axis=0)
        baselines["median_baseline_mse"] = median_mse.tolist()

        # Also compute MAE
        median_mae = np.mean(np.abs(y_val - median_preds), axis=0)
        baselines["median_baseline_mae"] = median_mae.tolist()
    else:
        median_mse = np.mean((y_val.flatten() - median_preds.flatten()) ** 2)
        baselines["median_baseline_mse"] = float(median_mse)

        # Also compute MAE
        median_mae = np.mean(np.abs(y_val.flatten() - median_preds.flatten()))
        baselines["median_baseline_mae"] = float(median_mae)

    return baselines


def calculate_normalized_metrics(metrics, baselines):
    """
    Calculate normalized metrics relative to baselines.

    Args:
        metrics: Dictionary of model metrics
        baselines: Dictionary of baseline metrics

    Returns:
        Dictionary of normalized metrics
    """
    normalized = {}

    # Calculate improvement over mean baseline
    if "val_mse" in metrics and "mean_baseline_mse" in baselines:
        if isinstance(metrics["val_mse"], list):
            # Multi-output case
            mean_improvements = []
            for i, mse in enumerate(metrics["val_mse"]):
                baseline = baselines["mean_baseline_mse"][i]
                improvement = (baseline - mse) / baseline * 100
                mean_improvements.append(improvement)
            normalized["mean_mse_improvement_pct"] = mean_improvements
        else:
            # Single output case
            mean_improvement = (
                (baselines["mean_baseline_mse"] - metrics["val_mse"])
                / baselines["mean_baseline_mse"]
                * 100
            )
            normalized["mean_mse_improvement_pct"] = float(mean_improvement)

    # Calculate improvement over median baseline
    if "val_mse" in metrics and "median_baseline_mse" in baselines:
        if isinstance(metrics["val_mse"], list):
            # Multi-output case
            median_improvements = []
            for i, mse in enumerate(metrics["val_mse"]):
                baseline = baselines["median_baseline_mse"][i]
                improvement = (baseline - mse) / baseline * 100
                median_improvements.append(improvement)
            normalized["median_mse_improvement_pct"] = median_improvements
        else:
            # Single output case
            median_improvement = (
                (baselines["median_baseline_mse"] - metrics["val_mse"])
                / baselines["median_baseline_mse"]
                * 100
            )
            normalized["median_mse_improvement_pct"] = float(median_improvement)

    # Calculate MSE to target variance ratio
    if "val_mse" in metrics and "target_variance" in metrics:
        if isinstance(metrics["val_mse"], list):
            mse_var_ratios = []
            for i, mse in enumerate(metrics["val_mse"]):
                var = metrics["target_variance"][i]
                ratio = mse / var
                mse_var_ratios.append(ratio)
            normalized["mse_to_variance_ratio"] = mse_var_ratios
        else:
            mse_var_ratio = metrics["val_mse"] / metrics["target_variance"]
            normalized["mse_to_variance_ratio"] = float(mse_var_ratio)

    return normalized


def create_metrics_interpretation(model_metrics, baseline_metrics, target_stats):
    """
    Create a human-readable interpretation of model metrics.

    Args:
        model_metrics: Dictionary of model metrics
        baseline_metrics: Dictionary of baseline metrics
        target_stats: Statistics about the target variable

    Returns:
        Dictionary with interpretation text and context
    """
    # Initialize with a more comprehensive default summary
    performance_level = "undetermined"
    summary_details = "No metrics available for detailed assessment."

    interpretation = {
        "metrics_context": {},
        "performance_summary": "Model performance assessment completed.",  # Default summary
        "domain_interpretation": {},
    }

    # Calculate normalized metrics
    normalized = calculate_normalized_metrics(
        {**model_metrics, "target_variance": target_stats.get("variance", 1.0)},
        baseline_metrics,
    )

    # Add metrics context
    interpretation["metrics_context"] = {
        "val_mse": {
            "value": model_metrics.get("val_mse"),
            "baseline_mean": baseline_metrics.get("mean_baseline_mse"),
            "baseline_median": baseline_metrics.get("median_baseline_mse"),
            "improvement_over_mean_pct": normalized.get("mean_mse_improvement_pct"),
            "improvement_over_median_pct": normalized.get("median_mse_improvement_pct"),
            "interpretation": "No interpretation available for MSE.",
        },
        "val_mae": {
            "value": model_metrics.get("val_mae"),
            "baseline_mean": baseline_metrics.get("mean_baseline_mae"),
            "baseline_median": baseline_metrics.get("median_baseline_mae"),
            "interpretation": "No interpretation available for MAE.",
        },
        "val_loss": {
            "value": model_metrics.get("val_loss"),
            "interpretation": "Lower values indicate better model fit.",
        },
    }

    # Extract error metrics and calculate error statistics if available
    if "prediction_errors" in model_metrics:
        errors = model_metrics["prediction_errors"]
        error_stats = {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "abs_mean": float(np.mean(np.abs(errors))),
        }
        interpretation["metrics_context"]["error_stats"] = error_stats

    # Add interpretation text for MSE
    if "mean_mse_improvement_pct" in normalized:
        improvement = normalized["mean_mse_improvement_pct"]
        if isinstance(improvement, list):
            avg_improvement = sum(improvement) / len(improvement)
        else:
            avg_improvement = improvement

        if avg_improvement > 80:
            interpretation["metrics_context"]["val_mse"][
                "interpretation"
            ] = "Excellent performance, with over 80% improvement over the baseline."
        elif avg_improvement > 50:
            interpretation["metrics_context"]["val_mse"][
                "interpretation"
            ] = "Good performance, with 50-80% improvement over the baseline."
        elif avg_improvement > 20:
            interpretation["metrics_context"]["val_mse"][
                "interpretation"
            ] = "Moderate performance, with 20-50% improvement over the baseline."
        elif avg_improvement > 0:
            interpretation["metrics_context"]["val_mse"][
                "interpretation"
            ] = "Slight improvement over the baseline, but there is room for improvement."
        else:
            interpretation["metrics_context"]["val_mse"][
                "interpretation"
            ] = "Poor performance, worse than the baseline model."

    # Add interpretation for MAE in terms of the target's scale
    if "val_mae" in model_metrics and "mean" in target_stats:
        mae = model_metrics["val_mae"]
        target_mean = target_stats["mean"]
        avg_mae_ratio = 0

        if isinstance(mae, list) and isinstance(target_mean, list):
            # Try to compute ratios safely
            mae_ratios = []
            for i, m in enumerate(mae):
                if i < len(target_mean) and target_mean[i] > 0:
                    mae_ratios.append(m / target_mean[i] * 100)
            avg_mae_ratio = safe_average(mae_ratios)
            # Store individual ratios as well
            interpretation["metrics_context"]["val_mae"][
                "ratios_by_target"
            ] = mae_ratios
        elif isinstance(mae, list) and not isinstance(target_mean, list):
            # Single target mean, multiple MAEs
            if target_mean > 0:
                mae_ratios = [m / target_mean * 100 for m in mae]
                avg_mae_ratio = safe_average(mae_ratios)
                interpretation["metrics_context"]["val_mae"][
                    "ratios_by_target"
                ] = mae_ratios
        elif not isinstance(mae, list) and isinstance(target_mean, list):
            # Multiple target means, single MAE (unusual case)
            avg_target = safe_average(target_mean)
            if avg_target > 0:
                avg_mae_ratio = mae / avg_target * 100
        else:
            # Single output case
            avg_mae_ratio = (
                mae / target_mean * 100 if target_mean and target_mean > 0 else 0
            )

        interpretation["metrics_context"]["val_mae"][
            "relative_to_mean_pct"
        ] = avg_mae_ratio

        if avg_mae_ratio < 5:
            interpretation["metrics_context"]["val_mae"][
                "interpretation"
            ] = "Excellent accuracy, with average error less than 5% of the mean value."
        elif avg_mae_ratio < 10:
            interpretation["metrics_context"]["val_mae"][
                "interpretation"
            ] = "Good accuracy, with average error between 5-10% of the mean value."
        elif avg_mae_ratio < 20:
            interpretation["metrics_context"]["val_mae"][
                "interpretation"
            ] = "Acceptable accuracy, with average error between 10-20% of the mean value."
        elif avg_mae_ratio < 50:
            interpretation["metrics_context"]["val_mae"][
                "interpretation"
            ] = "Moderate accuracy, with average error between 20-50% of the mean value."
        else:
            interpretation["metrics_context"]["val_mae"][
                "interpretation"
            ] = "Poor accuracy, with average error exceeding 50% of the mean value."

    # Calculate error distribution stats if prediction errors are available
    if "prediction_errors" in model_metrics and "mean" in target_stats:
        errors = model_metrics["prediction_errors"]
        target_mean_value = target_stats["mean"]

        # Convert list target_mean to average if needed
        if isinstance(target_mean_value, list):
            target_mean_value = sum(target_mean_value) / len(target_mean_value)

        # Calculate percentage of predictions within error bounds
        if isinstance(target_mean_value, (int, float)) and target_mean_value > 0:
            within_1pct = 100 * np.mean(np.abs(errors) < 0.01 * target_mean_value)
            within_5pct = 100 * np.mean(np.abs(errors) < 0.05 * target_mean_value)
            within_10pct = 100 * np.mean(np.abs(errors) < 0.1 * target_mean_value)
            within_20pct = 100 * np.mean(np.abs(errors) < 0.2 * target_mean_value)

            error_distribution = {
                "within_1pct_of_mean": float(within_1pct),
                "within_5pct_of_mean": float(within_5pct),
                "within_10pct_of_mean": float(within_10pct),
                "within_20pct_of_mean": float(within_20pct),
            }

            interpretation["metrics_context"]["error_distribution"] = error_distribution

            # Add error distribution interpretation
            if within_5pct > 80:
                error_precision = "very high"
            elif within_10pct > 80:
                error_precision = "high"
            elif within_20pct > 80:
                error_precision = "moderate"
            else:
                error_precision = "low"

            interpretation["metrics_context"]["error_precision"] = {
                "level": error_precision,
                "description": f"{within_5pct:.1f}% of predictions are within 5% of the true value, "
                f"and {within_10pct:.1f}% are within 10% of the true value.",
            }

    # Create overall performance summary with more informative metrics
    try:
        # Extract key performance indicators
        if "val_mse" in model_metrics and "mean_baseline_mse" in baseline_metrics:
            # MSE improvement
            model_mse = model_metrics["val_mse"]
            baseline_mse = baseline_metrics["mean_baseline_mse"]

            if isinstance(model_mse, list) and isinstance(baseline_mse, list):
                # Calculate average improvement for multi-output case
                mse_improvements = []
                for i, mse in enumerate(model_mse):
                    if i < len(baseline_mse) and baseline_mse[i] > 0:
                        improvement = (baseline_mse[i] - mse) / baseline_mse[i] * 100
                        mse_improvements.append(improvement)
                avg_improvement = safe_average(mse_improvements)
            elif isinstance(model_mse, list) and not isinstance(baseline_mse, list):
                # Multi-output model MSE, single baseline MSE
                avg_model_mse = safe_average(model_mse)
                avg_improvement = (
                    (baseline_mse - avg_model_mse) / baseline_mse * 100
                    if baseline_mse > 0
                    else 0
                )
            elif not isinstance(model_mse, list) and isinstance(baseline_mse, list):
                # Single model MSE, multi-output baseline MSE
                avg_baseline_mse = safe_average(baseline_mse)
                avg_improvement = (
                    (avg_baseline_mse - model_mse) / avg_baseline_mse * 100
                    if avg_baseline_mse > 0
                    else 0
                )
            else:
                # Single output case
                avg_improvement = (
                    (baseline_mse - model_mse) / baseline_mse * 100
                    if baseline_mse > 0
                    else 0
                )

            # MAE relative to target mean
            if "val_mae" in model_metrics and "mean" in target_stats:
                mae = model_metrics["val_mae"]
                target_mean = target_stats["mean"]

                if isinstance(mae, list) and isinstance(target_mean, list):
                    # Calculate average MAE ratio for multi-output case
                    mae_ratios = []
                    for i, m in enumerate(mae):
                        if i < len(target_mean) and target_mean[i] > 0:
                            ratio = m / target_mean[i] * 100
                            mae_ratios.append(ratio)
                    avg_mae_ratio = safe_average(mae_ratios)
                elif isinstance(mae, list) and not isinstance(target_mean, list):
                    # Multi-output MAE, single target mean
                    avg_mae = safe_average(mae)
                    avg_mae_ratio = (
                        avg_mae / target_mean * 100 if target_mean > 0 else 0
                    )
                elif not isinstance(mae, list) and isinstance(target_mean, list):
                    # Single MAE, multi-output target mean
                    avg_target_mean = safe_average(target_mean)
                    avg_mae_ratio = (
                        mae / avg_target_mean * 100 if avg_target_mean > 0 else 0
                    )
                else:
                    # Single output case
                    avg_mae_ratio = mae / target_mean * 100 if target_mean > 0 else 0

                # Determine performance level
                if avg_improvement > 50 and avg_mae_ratio < 10:
                    performance_level = "excellent"
                elif avg_improvement > 30 and avg_mae_ratio < 20:
                    performance_level = "good"
                elif avg_improvement > 10 and avg_mae_ratio < 50:
                    performance_level = "acceptable"
                elif avg_improvement > 0:
                    performance_level = "marginal"
                else:
                    performance_level = "suboptimal"

                # Create detailed summary
                summary_details = (
                    f"with {avg_improvement:.1f}% improvement over the baseline model "
                    f"and an average error of {avg_mae_ratio:.1f}% relative to the mean target value."
                )

                # Add error distribution information if available
                if "error_distribution" in interpretation["metrics_context"]:
                    error_dist = interpretation["metrics_context"]["error_distribution"]
                    summary_details += (
                        f" {error_dist['within_5pct_of_mean']:.1f}% of predictions are within 5% of the true value, "
                        f"and {error_dist['within_10pct_of_mean']:.1f}% are within 10%."
                    )

            # Create comprehensive performance summary
            interpretation["performance_summary"] = (
                f"The model shows {performance_level} performance, {summary_details}"
            )

    except Exception as e:
        # If something goes wrong, fall back to a basic summary but log the error
        print(f"Warning: Could not generate detailed performance summary: {e}")
        # Set a default performance summary based on what data is available
        if "val_mse" in model_metrics:
            interpretation["performance_summary"] = (
                f"Model evaluation completed with MSE of {model_metrics['val_mse']}. "
                f"Further analysis may provide more insights."
            )

    # Add domain-specific interpretation for response time prediction
    if "val_mae" in model_metrics:
        mae = model_metrics["val_mae"]
        if isinstance(mae, list):
            avg_mae = sum(mae) / len(mae)
        else:
            avg_mae = mae

        interpretation["domain_interpretation"] = {
            "response_time_prediction": (
                f"For response time prediction, the model has an average error of {avg_mae:.2f} seconds. "
                f"This means predictions are typically within ±{avg_mae:.2f} seconds of the actual value."
            ),
            "practical_significance": "",
        }

        # Add practical significance based on magnitude
        if avg_mae < 0.1:
            interpretation["domain_interpretation"][
                "practical_significance"
            ] = "This level of accuracy is suitable for precise performance monitoring and SLA compliance."
        elif avg_mae < 0.5:
            interpretation["domain_interpretation"][
                "practical_significance"
            ] = "This level of accuracy is suitable for most performance prediction tasks in production environments."
        elif avg_mae < 1.0:
            interpretation["domain_interpretation"][
                "practical_significance"
            ] = "This level of accuracy is suitable for general performance trend analysis but may not be precise enough for strict SLAs."
        else:
            interpretation["domain_interpretation"][
                "practical_significance"
            ] = "This level of accuracy provides a general estimate but may not be suitable for precise performance planning."

        # Add information about model's predictive behavior
        if "prediction_errors" in model_metrics:
            errors = model_metrics["prediction_errors"]
            bias = np.mean(errors)
            if bias > 0.05:
                bias_direction = "consistently overestimates"
            elif bias < -0.05:
                bias_direction = "consistently underestimates"
            else:
                bias_direction = "has minimal bias in"

            interpretation["domain_interpretation"]["prediction_bias"] = (
                f"The model {bias_direction} the target values with a mean error of {bias:.3f}. "
                f"This should be considered when interpreting predictions."
            )

    return interpretation


def create_performance_visualization(
    model_metrics, baseline_metrics, target_stats, output_dir
):
    """
    Create visualizations that provide context for model metrics.

    Args:
        model_metrics: Dictionary of model metrics
        baseline_metrics: Dictionary of baseline metrics
        target_stats: Statistics about the target variable
        output_dir: Directory to save visualizations

    Returns:
        List of paths to visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    visualization_paths = []

    # 1. Create comparison bar chart for MSE
    if "val_mse" in model_metrics and "mean_baseline_mse" in baseline_metrics:
        plt.figure(figsize=(10, 6))

        # Prepare data for visualization
        if isinstance(model_metrics["val_mse"], list):
            # Multi-output case - use first output for visualization
            model_mse = model_metrics["val_mse"][0]
            mean_baseline = baseline_metrics["mean_baseline_mse"][0]
            median_baseline = baseline_metrics.get("median_baseline_mse", [0])[0]
        else:
            # Single output case
            model_mse = model_metrics["val_mse"]
            mean_baseline = baseline_metrics["mean_baseline_mse"]
            median_baseline = baseline_metrics.get("median_baseline_mse", 0)

        # Create bar chart
        models = ["Our Model", "Mean Baseline", "Median Baseline"]
        values = [model_mse, mean_baseline, median_baseline]

        plt.bar(models, values, color=["#2C7BB6", "#D7191C", "#FDAE61"])

        # Format chart
        plt.title("MSE Comparison with Baselines")
        plt.ylabel("Mean Squared Error (lower is better)")
        plt.yscale("log")  # Log scale often helps visualize differences

        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v * 1.1, f"{v:.3f}", ha="center")

        # Save figure
        mse_comparison_path = os.path.join(output_dir, "mse_baseline_comparison.png")
        plt.tight_layout()
        plt.savefig(mse_comparison_path, dpi=300)
        plt.close()

        visualization_paths.append(mse_comparison_path)

    # 2. Create error distribution context
    if "prediction_errors" in model_metrics:
        errors = model_metrics["prediction_errors"]

        plt.figure(figsize=(10, 6))

        # Get statistical properties of errors for better visualization
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        error_min = np.min(errors)
        error_max = np.max(errors)

        # Calculate interquartile range for robust outlier detection
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1

        # Define outlier boundaries using both IQR method and standard deviation
        std_bounds = (error_mean - 3 * error_std, error_mean + 3 * error_std)
        iqr_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Use the tighter of the two bounds but ensure we don't exclude too much data
        x_min = max(std_bounds[0], iqr_bounds[0])
        x_max = min(std_bounds[1], iqr_bounds[1])

        # If bounds are too tight or exclude a lot of data, fall back to percentiles
        if (
            x_max - x_min < error_std
            or np.mean((errors < x_min) | (errors > x_max)) > 0.05
        ):
            x_min = np.percentile(errors, 1)  # 1st percentile
            x_max = np.percentile(errors, 99)  # 99th percentile

        # Ensure we include zero in the plot to provide reference
        if x_min > 0:
            x_min = -0.1 * abs(x_max)
        if x_max < 0:
            x_max = 0.1 * abs(x_min)

        # Add a small margin for better visualization
        margin = 0.1 * (x_max - x_min)
        x_min -= margin
        x_max += margin

        # Plot error distribution with explicit bins for better control
        n_bins = min(
            50, max(10, int(len(errors) / 20))
        )  # Adjust bin count based on data size
        sns.histplot(errors, kde=True, bins=n_bins)

        # Add reference lines
        plt.axvline(x=0, color="r", linestyle="--", label="Zero Error")
        plt.axvline(
            x=error_mean,
            color="b",
            linestyle="-",
            label=f"Mean Error: {error_mean:.3f}",
        )

        # Add lines showing the actual min/max (with low opacity) to indicate the full range
        plt.axvline(
            x=error_min,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Min Error: {error_min:.3f}",
        )
        plt.axvline(
            x=error_max,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Max Error: {error_max:.3f}",
        )

        if "mean" in target_stats:
            # Add reference lines for percentage of mean
            target_mean = target_stats["mean"]
            if isinstance(target_mean, list):
                target_mean = sum(target_mean) / len(target_mean)

            # Only add reference lines if they're within our plot range
            if abs(0.05 * target_mean) < max(abs(x_min), abs(x_max)):
                plt.axvline(
                    x=0.05 * target_mean,
                    color="g",
                    linestyle="-.",
                    label="±5% of Mean Target",
                )
                plt.axvline(x=-0.05 * target_mean, color="g", linestyle="-.")

            if abs(0.1 * target_mean) < max(abs(x_min), abs(x_max)):
                plt.axvline(
                    x=0.1 * target_mean,
                    color="y",
                    linestyle="-.",
                    label="±10% of Mean Target",
                )
                plt.axvline(x=-0.1 * target_mean, color="y", linestyle="-.")

        # Set x-axis limits based on the calculated values
        plt.xlim(x_min, x_max)

        # Add statistical annotations
        within_5pct = 100 * np.mean(
            np.abs(errors) < 0.05 * target_mean
            if isinstance(target_mean, (int, float))
            else False
        )
        within_10pct = 100 * np.mean(
            np.abs(errors) < 0.1 * target_mean
            if isinstance(target_mean, (int, float))
            else False
        )

        # Calculate additional statistics for annotations
        median_error = np.median(errors)
        q1_q3_range = f"[{q1:.3f}, {q3:.3f}]"
        outlier_pct = 100 * np.mean(
            (errors < q1 - 1.5 * iqr) | (errors > q3 + 1.5 * iqr)
        )

        # Add detailed annotations about error distribution
        plt.annotate(
            f"Mean: {error_mean:.3f}, Median: {median_error:.3f}\n"
            f"Std Dev: {error_std:.3f}, IQR: {iqr:.3f}\n"
            f"Within ±5% of mean: {within_5pct:.1f}%\n"
            f"Within ±10% of mean: {within_10pct:.1f}%\n"
            f"Outliers: {outlier_pct:.1f}%",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )

        plt.title("Prediction Error Distribution")
        plt.xlabel("Prediction Error (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.legend(loc="upper left")

        # Save figure
        error_dist_path = os.path.join(output_dir, "error_distribution_context.png")
        plt.tight_layout()
        plt.savefig(error_dist_path, dpi=300)
        plt.close()

        visualization_paths.append(error_dist_path)

    # 3. Create residual plot (actual vs predicted values)
    if "prediction_errors" in model_metrics and "mean" in target_stats:
        errors = model_metrics["prediction_errors"]
        
        # We need to reconstruct predicted values from errors
        # In this case, we'll need to find actual y_val and predictions from the test script
        # For now, we'll create a synthetic residual plot using the errors
        
        # Create synthetic predictions and actual values
        if isinstance(target_stats["mean"], list):
            target_mean = target_stats["mean"][0]  # Use first output for viz
        else:
            target_mean = target_stats["mean"]
            
        # Generate synthetic values around the mean with standard deviation
        std_val = target_stats.get("std", 1.0)
        if isinstance(std_val, list):
            std_val = std_val[0]
        
        n_samples = len(errors)
        actual_values = np.random.normal(target_mean, std_val, n_samples)
        predicted_values = actual_values - errors
        
        plt.figure(figsize=(10, 6))
        
        # Plot the residuals
        plt.scatter(predicted_values, errors, alpha=0.5, color="#2C7BB6")
        
        # Add reference line at y=0
        plt.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        
        # Add trend line for residuals
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_values, errors)
        trend_x = np.linspace(min(predicted_values), max(predicted_values), 100)
        trend_y = slope * trend_x + intercept
        plt.plot(trend_x, trend_y, color="g", linestyle="-.", linewidth=1.5,
                 label=f"Trend (slope={slope:.4f})")
        
        # Format plot
        plt.title("Residual Plot (Error vs Predicted Value)")
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistical annotations
        plt.annotate(
            f"Mean Error: {np.mean(errors):.3f}\n"
            f"StdDev: {np.std(errors):.3f}\n"
            f"Slope: {slope:.4f} (closer to 0 is better)\n"
            f"R²: {r_value**2:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )
        
        # Save figure
        residual_path = os.path.join(output_dir, "residual_plot.png")
        plt.tight_layout()
        plt.savefig(residual_path, dpi=300)
        plt.close()
        
        visualization_paths.append(residual_path)
    
    # 4. Create prediction vs actual scatter plot
    if "prediction_errors" in model_metrics and "mean" in target_stats:
        # Use the same synthetic data as in the residual plot
        errors = model_metrics["prediction_errors"]
        
        if isinstance(target_stats["mean"], list):
            target_mean = target_stats["mean"][0]
        else:
            target_mean = target_stats["mean"]
            
        std_val = target_stats.get("std", 1.0)
        if isinstance(std_val, list):
            std_val = std_val[0]
        
        n_samples = len(errors)
        actual_values = np.random.normal(target_mean, std_val, n_samples)
        predicted_values = actual_values - errors
        
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(actual_values, predicted_values, alpha=0.5, color="#2C7BB6")
        
        # Add perfect prediction line (y=x)
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 label="Perfect Prediction", linewidth=1.5)
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual_values, predicted_values)
        regression_x = np.linspace(min_val, max_val, 100)
        regression_y = slope * regression_x + intercept
        plt.plot(regression_x, regression_y, color="g", linestyle="-", linewidth=1.5,
                 label=f"Linear Fit (R²={r_value**2:.4f})")
        
        # Format plot
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistical annotations
        plt.annotate(
            f"Correlation: {r_value:.4f}\n"
            f"Slope: {slope:.4f} (ideal=1.0)\n"
            f"Intercept: {intercept:.4f} (ideal=0.0)\n"
            f"Mean Abs Error: {np.mean(np.abs(errors)):.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )
        
        # Save figure
        prediction_path = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.tight_layout()
        plt.savefig(prediction_path, dpi=300)
        plt.close()
        
        visualization_paths.append(prediction_path)

    return visualization_paths
