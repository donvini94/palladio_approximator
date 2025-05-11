"""
Script to test metrics visualization with extreme outliers.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from utils.metrics_context import (
    get_baseline_metrics,
    calculate_normalized_metrics,
    create_metrics_interpretation,
    create_performance_visualization,
)

def generate_data_with_outliers(n_samples=200, outlier_ratio=0.05, extreme_ratio=0.01):
    """Generate synthetic data with controlled outliers for testing."""
    # Create input features
    X_train = np.random.randn(1000, 10)
    X_val = np.random.randn(n_samples, 10)
    
    # Create synthetic target values
    y_base = np.abs(5 + 2 * np.random.randn(1000, 1))
    y_train = np.hstack([y_base, y_base * 0.7, y_base * 1.3])
    
    y_base_val = np.abs(5 + 2 * np.random.randn(n_samples, 1))
    y_val = np.hstack([y_base_val, y_base_val * 0.7, y_base_val * 1.3])
    
    # Create synthetic model predictions with some error
    predictions = y_val.copy()
    
    # Add regular errors following normal distribution
    regular_errors = np.random.randn(n_samples, 3) * 0.5
    predictions += regular_errors
    
    # Add moderate outliers
    n_outliers = int(n_samples * outlier_ratio)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_values = np.random.uniform(-5, 5, (n_outliers, 3))
    predictions[outlier_indices] = y_val[outlier_indices] + outlier_values
    
    # Add extreme outliers
    n_extreme = int(n_samples * extreme_ratio)
    extreme_indices = np.random.choice(n_samples, n_extreme, replace=False)
    extreme_values = np.random.uniform(-50, 30, (n_extreme, 3))
    predictions[extreme_indices] = y_val[extreme_indices] + extreme_values
    
    # Create prediction errors
    errors = y_val[:, 0] - predictions[:, 0]  # Use first column for visualization
    
    return X_train, y_train, X_val, y_val, predictions, errors

def main():
    print("Generating data with extreme outliers...")
    X_train, y_train, X_val, y_val, predictions, errors = generate_data_with_outliers()
    
    # Calculate model metrics
    mse = np.mean((y_val - predictions) ** 2, axis=0)
    mae = np.mean(np.abs(y_val - predictions), axis=0)
    
    model_metrics = {
        "val_mse": mse.tolist(),
        "val_mae": mae.tolist(),
        "prediction_errors": errors,
    }
    
    # Get baseline metrics
    baseline_metrics = get_baseline_metrics(X_train, y_train, X_val, y_val)
    
    # Calculate target statistics
    target_stats = {
        "mean": np.mean(y_val, axis=0).tolist(),
        "median": np.median(y_val, axis=0).tolist(),
        "std": np.std(y_val, axis=0).tolist(),
        "variance": np.var(y_val, axis=0).tolist(),
    }
    
    print("Creating metrics interpretation...")
    interpretation = create_metrics_interpretation(model_metrics, baseline_metrics, target_stats)
    
    # Create output directory
    output_dir = "figures/test_viz_outliers"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating visualizations in {output_dir}...")
    viz_paths = create_performance_visualization(
        model_metrics, baseline_metrics, target_stats, output_dir
    )
    
    # Create a markdown summary
    summary_path = os.path.join(output_dir, "performance_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Model Performance Summary (Outlier Test)\n\n")
        
        f.write("## Overall Assessment\n\n")
        f.write(interpretation["performance_summary"])
        f.write("\n\n")
        
        f.write("## Metrics Context\n\n")
        for metric, info in interpretation["metrics_context"].items():
            # Skip complex nested objects
            if isinstance(info, dict):
                f.write(f"### {metric}\n")
                for key, val in info.items():
                    if isinstance(val, (str, int, float)) or (isinstance(val, list) and len(val) < 10):
                        f.write(f"- {key}: {val}\n")
                f.write("\n")
        
        f.write("## Domain-Specific Interpretation\n\n")
        for key, val in interpretation["domain_interpretation"].items():
            f.write(f"### {key}\n")
            f.write(f"{val}\n\n")
            
    print(f"Created performance summary at {summary_path}")
    
    # Save interpretation as JSON for reference
    json_path = os.path.join(output_dir, "interpretation.json")
    with open(json_path, "w") as f:
        json.dump(interpretation, f, indent=2)
        
    print(f"Saved interpretation data to {json_path}")
    
    # Save the raw error data for analysis
    error_stats = {
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "percentile_1": float(np.percentile(errors, 1)),
        "percentile_5": float(np.percentile(errors, 5)),
        "percentile_25": float(np.percentile(errors, 25)),
        "percentile_75": float(np.percentile(errors, 75)),
        "percentile_95": float(np.percentile(errors, 95)),
        "percentile_99": float(np.percentile(errors, 99)),
    }
    
    stats_path = os.path.join(output_dir, "error_stats.json")
    with open(stats_path, "w") as f:
        json.dump(error_stats, f, indent=2)
    
    print(f"Saved error statistics to {stats_path}")
    print("Completed outlier visualization test.")

if __name__ == "__main__":
    main()