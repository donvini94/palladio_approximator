#!/usr/bin/env python3
"""
Test script to demonstrate visualization capabilities for thesis figures.
"""

import os
import argparse
import torch
import numpy as np
from utils.visualize import create_dirs, prediction_error_analysis, visualize_embedding_space
from utils.attention import visualize_attention_patterns

def main():
    """Main function to demonstrate visualization capabilities"""
    parser = argparse.ArgumentParser(description="Test visualization capabilities")
    parser.add_argument('--attention', action='store_true', help="Test attention visualization")
    parser.add_argument('--embedding', action='store_true', help="Test embedding visualization")
    parser.add_argument('--prediction', action='store_true', help="Test prediction error visualization")
    parser.add_argument('--all', action='store_true', help="Run all tests")
    
    args = parser.parse_args()
    
    # Create directories
    create_dirs()
    
    if args.all or args.attention:
        # Test attention visualization
        print("Testing attention visualization...")
        
        # Find a small file to test
        sample_file = "PCMs/generated__AAAJW.tpcm/generated__AAAJW.tpcm"
        
        if os.path.exists(sample_file):
            result = visualize_attention_patterns(
                sample_file,
                model_name="microsoft/codebert-base",
                output_dir="figures/attention/test",
                max_length=256  # Use smaller length for testing
            )
            print(f"Attention visualization complete: {result}")
        else:
            print(f"Sample file not found: {sample_file}")
    
    if args.all or args.embedding:
        # Test embedding visualization with random data
        print("Testing embedding visualization...")
        
        # Create synthetic embeddings for visualization
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 100 samples, 50 dimensions
        y = np.random.randn(100)      # Random target values
        
        # Visualize with PCA
        visualize_embedding_space(
            X, y, method='pca', embedding_type='test_embedding', n_components=2
        )
        
        # Visualize with t-SNE
        visualize_embedding_space(
            X, y, method='tsne', embedding_type='test_embedding', n_components=2
        )
        
        print("Embedding visualization complete.")
    
    if args.all or args.prediction:
        # Test prediction error visualization
        print("Testing prediction error visualization...")
        
        # Create a simple mock model for testing
        class MockModel:
            def predict(self, X):
                # Add some noise to the true values for realistic predictions
                return y + np.random.randn(len(y)) * 0.2
        
        model = MockModel()
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 100 samples, 50 dimensions
        y = np.random.randn(100)      # Random target values
        
        # Run prediction error analysis
        results = prediction_error_analysis(
            model, X, y, embedding_type='test_embedding', model_type='mock_model'
        )
        
        print(f"Prediction error analysis complete: {results}")
    
    print("Visualization tests complete. Check the 'figures' directory for outputs.")

if __name__ == "__main__":
    main()