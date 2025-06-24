#!/usr/bin/env python3
"""
Script to generate a performance summary with interpretable metrics.
"""
import argparse
import os
import sys
from utils.visualize import visualize_performance_context


def main():
    """Main function to parse arguments and generate performance summary"""
    parser = argparse.ArgumentParser(
        description="Generate interpretable performance metrics summary"
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Specific MLflow run ID to analyze (if None, uses the latest run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/performance_context",
        help="Directory to save the visualizations",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating performance context summary...")

    try:
        summary_path = visualize_performance_context(
            run_id=args.run_id, output_dir=args.output_dir
        )

        if summary_path:
            print(f"Performance summary generated at {summary_path}")
            print(f"Visualizations saved to {args.output_dir}")
        else:
            print("No performance summary data found.")
            print(
                "Please run training with --use_mlflow and ensure the metrics_context module is implemented."
            )
    except Exception as e:
        print(f"Error generating performance summary: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
