#!/usr/bin/env python3
"""
Test script to verify the structured feature extraction is working correctly.
This script loads a sample DSL file, extracts structured features, and displays them.
"""

import os
import sys
import glob
import pandas as pd
from structured_features import extract_structured_features


def main():
    """Test structured feature extraction on a sample DSL file"""
    # Find DSL files
    dsl_dir = "data/dsl_models"
    if not os.path.exists(dsl_dir):
        print(
            f"Error: DSL directory '{dsl_dir}' not found. Please check your project structure."
        )
        return 1

    dsl_files = glob.glob(os.path.join(dsl_dir, "*.tpcm"))

    if not dsl_files:
        print(f"No DSL files found in {dsl_dir}. Please add some .tpcm files first.")
        return 1

    print(f"Found {len(dsl_files)} DSL files. Testing on the first file.")

    # Load the first DSL file
    with open(dsl_files[0], "r", encoding="utf-8") as f:
        dsl_content = f.read()

    # Extract structured features
    print(f"Extracting structured features from: {os.path.basename(dsl_files[0])}")
    features = extract_structured_features(dsl_content)

    # Display features
    df = pd.DataFrame([features]).T.reset_index()
    df.columns = ["Feature", "Value"]

    print("\nExtracted Structured Features:")
    print("============================")
    print(df.to_string(index=False))

    print(f"\nTotal number of features extracted: {len(features)}")

    # Identify potentially valuable features
    numeric_features = {
        k: v for k, v in features.items() if isinstance(v, (int, float)) and v != 0
    }
    print(f"\nNon-zero numeric features: {len(numeric_features)}/{len(features)}")

    # Display top features by value
    top_features = sorted(
        numeric_features.items(), key=lambda x: abs(x[1]), reverse=True
    )[:10]
    print("\nTop 10 features by magnitude:")
    for feature, value in top_features:
        print(f"  {feature}: {value}")

    print("\nStructured feature extraction test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
