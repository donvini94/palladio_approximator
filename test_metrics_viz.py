"""
Script to test metrics visualization and interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import json
from utils.metrics_context import (
    get_baseline_metrics,
    calculate_normalized_metrics,
    create_metrics_interpretation,
    create_performance_visualization,
)

# Create some synthetic data to test the visualization
def generate_synthetic_data():
    # Create input features
    X_train = np.random.randn(1000, 10)
    X_val = np.random.randn(200, 10)
    
    # Create synthetic target values with three outputs (avg, min, max)
    y_base = np.abs(5 + 2 * np.random.randn(1000, 1))
    y_train = np.hstack([
        y_base,                     # avg
        y_base * 0.7,               # min
        y_base * 1.3                # max
    ])
    
    y_base_val = np.abs(5 + 2 * np.random.randn(200, 1))
    y_val = np.hstack([
        y_base_val,                 # avg
        y_base_val * 0.7,           # min
        y_base_val * 1.3            # max
    ])
    
    # Create synthetic model predictions with some error
    predictions = y_val + np.random.randn(200, 3) * 0.5
    
    # Create prediction errors
    errors = y_val[:, 0] - predictions[:, 0]  # Use first column for visualization
    
    return X_train, y_train, X_val, y_val, predictions, errors

def main():
    print("Generating synthetic data...")
    X_train, y_train, X_val, y_val, predictions, errors = generate_synthetic_data()
    
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
    
    # Create a temporary directory for visualizations
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating visualizations in {tmpdir}...")
        viz_paths = create_performance_visualization(
            model_metrics, baseline_metrics, target_stats, tmpdir
        )
        
        # Create a proper output directory
        output_dir = "figures/test_viz"
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy visualizations to the output directory
        for viz_path in viz_paths:
            filename = os.path.basename(viz_path)
            dest_path = os.path.join(output_dir, filename)
            
            # Use matplotlib to read and save the figure to ensure it's properly saved
            img = plt.imread(viz_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(dest_path, dpi=300)
            plt.close()
            
            print(f"Saved visualization to {dest_path}")
        
        # Create a markdown summary
        summary_path = os.path.join(output_dir, "performance_summary.md")
        with open(summary_path, "w") as f:
            f.write("# Model Performance Summary\n\n")
            
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
    
    print("Completed metrics visualization test.")
    
if __name__ == "__main__":
    main()