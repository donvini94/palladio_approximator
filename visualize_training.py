#!/usr/bin/env python3
"""
Script to visualize neural network training metrics from MLflow logs.
Creates line charts showing the progression of metrics (loss, MSE, MAE) over epochs.
"""

import argparse
import os
import sys
from utils.visualize import visualize_training_metrics

def main():
    """Main function to parse arguments and visualize training metrics"""
    parser = argparse.ArgumentParser(description="Visualize neural network training metrics from MLflow")
    
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Specific MLflow run ID to visualize (if None, uses the latest run)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metric names to visualize (if None, uses default metrics)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/training",
        help="Directory to save the visualizations"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Visualizing training metrics...")
    print("Searching for MLflow experiments with PyTorch model runs...")
    
    try:
        df = visualize_training_metrics(
            run_id=args.run_id,
            metric_names=args.metrics,
            output_dir=args.output_dir
        )
        
        if df is not None:
            print(f"Generated training metrics visualization with {len(df)} epochs.")
            print(f"Available metrics: {', '.join(df.columns[1:])}")
            print(f"Visualizations saved to {args.output_dir}")
        else:
            print("No training metrics found or no MLflow experiment available.")
            print("Please run training with --use_mlflow first.")
    except Exception as e:
        print(f"Error visualizing training metrics: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())