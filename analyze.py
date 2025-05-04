#!/usr/bin/env python3
"""
Comprehensive analysis and visualization script for the palladio_approximator project.
This script produces visualizations and insights for thesis presentation.
"""

import os
import argparse
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
from tqdm import tqdm
import mlflow
from pathlib import Path
import sys

# Import project modules
from utils.visualize import (
    create_dirs,
    performance_comparison_dashboard,
    prediction_error_analysis,
    generate_learning_curves,
    visualize_embedding_space,
    create_experiment_summary,
    prediction_time_analysis,
    generate_dashboard_from_files,
    visualize_training_metrics
)

from utils.attention import (
    visualize_attention_patterns,
    analyze_model_representations,
    compare_model_predictions,
    extract_token_attention_patterns,
    batch_process_files
)


def setup_analysis_dirs():
    """Create all necessary directories for analysis outputs"""
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/performance", exist_ok=True)
    os.makedirs("figures/embeddings", exist_ok=True)
    os.makedirs("figures/attention", exist_ok=True)
    os.makedirs("figures/predictions", exist_ok=True)
    os.makedirs("figures/learning_curves", exist_ok=True)
    os.makedirs("figures/model_comparison", exist_ok=True)
    os.makedirs("figures/token_patterns", exist_ok=True)
    os.makedirs("figures/batch_analysis", exist_ok=True)


def find_models_and_embeddings(data_dir="."):
    """
    Find all model and embedding files in the specified directory.
    
    Args:
        data_dir: Directory to search in
        
    Returns:
        Tuple of (model_paths, embedding_paths, pairs)
    """
    # Find model files
    model_files = glob.glob(os.path.join(data_dir, "*_model.pkl"))
    embedding_files = glob.glob(os.path.join(data_dir, "*_features_checkpoint.pkl"))
    
    # Group by embedding and model types
    pairs = []
    
    for model_file in model_files:
        model_basename = os.path.basename(model_file)
        model_match = re.match(r'(.+?)_(.+?)_model\.pkl', model_basename)
        
        if not model_match:
            continue
            
        model_type, prediction_mode = model_match.groups()
        
        # Find matching embedding files
        for embedding_file in embedding_files:
            embedding_basename = os.path.basename(embedding_file)
            embedding_match = re.match(r'(.+?)_(.+?)_features_checkpoint\.pkl', embedding_basename)
            
            if not embedding_match:
                continue
                
            embedding_type, emb_prediction_mode = embedding_match.groups()
            
            # If prediction mode matches, add as a pair
            if prediction_mode == emb_prediction_mode:
                pairs.append((model_file, embedding_file, model_type, embedding_type, prediction_mode))
    
    return model_files, embedding_files, pairs


def analyze_mlflow_experiments():
    """Analyze all MLflow experiments and generate visualizations"""
    print("Analyzing MLflow experiments...")
    
    # Create experiment summary
    summary_df = create_experiment_summary()
    
    if summary_df is None:
        print("No MLflow experiments found. Skipping MLflow analysis.")
        return
        
    # Create performance dashboard
    print("Generating performance comparison dashboard...")
    
    # For MSE (lower is better)
    performance_comparison_dashboard(metric="test_mse")
    
    # For R² (higher is better)
    performance_comparison_dashboard(metric="test_r2")
    
    # Visualize training metrics for PyTorch models
    print("Generating training metric visualizations for PyTorch models...")
    try:
        # Process the latest PyTorch run
        df = visualize_training_metrics()
        if df is not None:
            print(f"Generated training metrics visualization with {len(df)} epochs.")
        else:
            print("No PyTorch training metrics found.")
    except Exception as e:
        print(f"Error visualizing training metrics: {e}")
    
    print("MLflow analysis complete. Check 'figures/performance' and 'figures/training' directories for results.")


def generate_model_predictions_analysis(pairs):
    """
    Generate prediction error analysis for each model/embedding pair.
    
    Args:
        pairs: List of (model_file, embedding_file, model_type, embedding_type, prediction_mode) tuples
    """
    print("Generating prediction error analysis...")
    
    results = []
    for model_file, embedding_file, model_type, embedding_type, prediction_mode in tqdm(pairs):
        try:
            # Load model and features
            model = joblib.load(model_file)
            features = joblib.load(embedding_file)
            
            X_test = features["X_test"]
            y_test = features["y_test"]
            
            # Generate prediction error analysis
            metrics = prediction_error_analysis(
                model, X_test, y_test, embedding_type, model_type
            )
            
            results.append({
                "model_type": model_type,
                "embedding_type": embedding_type,
                "prediction_mode": prediction_mode,
                **metrics
            })
            
        except Exception as e:
            print(f"Error analyzing {model_type} + {embedding_type}: {e}")
    
    # Create summary dataframe
    if results:
        df = pd.DataFrame(results)
        df.to_csv("figures/predictions/prediction_summary.csv", index=False)
        
        # Create comparative bar chart of metrics
        plt.figure(figsize=(12, 6))
        
        # Create a compound label for each model/embedding combination
        df["model_label"] = df.apply(lambda x: f"{x['embedding_type']} + {x['model_type']}", axis=1)
        
        # Check if mse is a list or scalar
        if isinstance(df["mse"].iloc[0], list):
            # Use first dimension for multi-output
            df["mse_value"] = df["mse"].apply(lambda x: x[0] if len(x) > 0 else 0)
        else:
            df["mse_value"] = df["mse"]
        
        # Plot MSE comparison
        plt.figure(figsize=(12, 6))
        plt.bar(df["model_label"], df["mse_value"])
        plt.title("Mean Squared Error by Model Configuration")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig("figures/predictions/mse_comparison.png", dpi=300)
        plt.close()
        
        print(f"Prediction analysis complete. Analyzed {len(results)} model configurations.")
    else:
        print("No valid model/embedding pairs found for prediction analysis.")


def generate_embedding_visualizations(pairs):
    """
    Generate visualizations of embedding spaces.
    
    Args:
        pairs: List of (model_file, embedding_file, model_type, embedding_type, prediction_mode) tuples
    """
    print("Generating embedding space visualizations...")
    
    # Group by embedding type to avoid repeated visualization of the same embedding
    embedding_types_processed = set()
    
    for model_file, embedding_file, model_type, embedding_type, prediction_mode in tqdm(pairs):
        # Skip if we've already processed this embedding type
        if embedding_type in embedding_types_processed:
            continue
            
        try:
            # Load embeddings
            features = joblib.load(embedding_file)
            
            X_train = features["X_train"]
            y_train = features["y_train"]
            
            # Visualize embedding space with PCA
            visualize_embedding_space(
                X_train, 
                y_train, 
                method="pca", 
                embedding_type=embedding_type,
                n_components=2
            )
            
            # Visualize embedding space with t-SNE (if dataset is not too large)
            if X_train.shape[0] <= 5000:  # t-SNE can be slow on large datasets
                visualize_embedding_space(
                    X_train, 
                    y_train, 
                    method="tsne", 
                    embedding_type=embedding_type,
                    n_components=2
                )
            
            # Add to processed set
            embedding_types_processed.add(embedding_type)
            
        except Exception as e:
            print(f"Error visualizing embedding space for {embedding_type}: {e}")
    
    print(f"Embedding visualization complete. Visualized {len(embedding_types_processed)} embedding types.")


def generate_model_representations(pairs):
    """
    Analyze and visualize model representations.
    
    Args:
        pairs: List of (model_file, embedding_file, model_type, embedding_type, prediction_mode) tuples
    """
    print("Analyzing model representations...")
    
    results = []
    for model_file, embedding_file, model_type, embedding_type, prediction_mode in tqdm(pairs):
        try:
            # Analyze model representations
            result = analyze_model_representations(
                model_file,
                embedding_file,
                output_dir=f"figures/embeddings/{embedding_type}_{model_type}",
                n_components=2
            )
            
            results.append({
                "model_type": model_type,
                "embedding_type": embedding_type,
                "prediction_mode": prediction_mode,
                **result
            })
            
        except Exception as e:
            print(f"Error analyzing representations for {model_type} + {embedding_type}: {e}")
    
    print(f"Model representation analysis complete. Analyzed {len(results)} model configurations.")


def compare_models(pairs):
    """
    Compare predictions from multiple models on the same test data.
    
    Args:
        pairs: List of (model_file, embedding_file, model_type, embedding_type, prediction_mode) tuples
    """
    print("Comparing model predictions...")
    
    # Group pairs by prediction mode
    pairs_by_mode = {}
    for model_file, embedding_file, model_type, embedding_type, prediction_mode in pairs:
        if prediction_mode not in pairs_by_mode:
            pairs_by_mode[prediction_mode] = []
        pairs_by_mode[prediction_mode].append((model_file, embedding_file, model_type, embedding_type))
    
    # Compare models within each prediction mode
    for mode, mode_pairs in pairs_by_mode.items():
        print(f"Comparing models for {mode} prediction mode...")
        
        model_paths = [p[0] for p in mode_pairs]
        embedding_paths = [p[1] for p in mode_pairs]
        
        try:
            # Compare predictions
            result = compare_model_predictions(
                model_paths,
                embedding_paths,
                output_dir=f"figures/model_comparison/{mode}"
            )
            
            print(f"Compared {len(result.get('models_compared', []))} models for {mode} prediction mode.")
            
        except Exception as e:
            print(f"Error comparing models for {mode} prediction mode: {e}")
    
    print("Model comparison complete.")


def analyze_attention_patterns(code_files_dir, model_name=None):
    """
    Analyze attention patterns in code files using the specified model.
    
    Args:
        code_files_dir: Directory containing code files to analyze
        model_name: Name of the model to use (default: CodeBERT)
    """
    print("Analyzing attention patterns in code files...")
    
    if model_name is None:
        model_name = "microsoft/codebert-base"
        
    # Check if directory exists
    if not os.path.exists(code_files_dir):
        print(f"Directory {code_files_dir} does not exist. Skipping attention analysis.")
        return
        
    # Find code files
    code_files = []
    for ext in [".tpcm", ".psl", ".pcm"]:
        code_files.extend(glob.glob(os.path.join(code_files_dir, f"*{ext}")))
    
    if not code_files:
        print(f"No code files found in {code_files_dir}. Skipping attention analysis.")
        return
        
    # Limit to 10 files to avoid excessive computation
    if len(code_files) > 10:
        code_files = code_files[:10]
        print(f"Limiting attention analysis to first 10 files.")
    
    # Process each file
    for file_path in tqdm(code_files, desc="Processing files"):
        try:
            file_name = Path(file_path).stem
            
            # Visualize attention patterns
            visualize_attention_patterns(
                file_path,
                model_name=model_name,
                output_dir=f"figures/attention/{file_name}"
            )
            
            # Extract token patterns
            extract_token_attention_patterns(
                file_path,
                model_name=model_name,
                output_dir=f"figures/token_patterns/{file_name}"
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Attention analysis complete. Processed {len(code_files)} files.")


def generate_learning_curve_analysis(model_types, embedding_types):
    """
    Generate learning curves for specified model and embedding types.
    
    Args:
        model_types: List of model types to analyze
        embedding_types: List of embedding types to analyze
    """
    print("Generating learning curves...")
    
    # Use smaller subsets to speed up the process
    train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for model_type in model_types:
        for embedding_type in embedding_types:
            try:
                print(f"Generating learning curves for {embedding_type} + {model_type}...")
                
                # Generate learning curves
                df = generate_learning_curves(
                    model_type,
                    embedding_type,
                    train_sizes=train_sizes
                )
                
                print(f"Learning curves generated for {embedding_type} + {model_type}.")
                
            except Exception as e:
                print(f"Error generating learning curves for {embedding_type} + {model_type}: {e}")
    
    print("Learning curve analysis complete.")


def performance_analysis():
    """Run comprehensive performance analysis"""
    print("Running comprehensive performance analysis...")
    
    # Find all models and embeddings
    model_files, embedding_files, pairs = find_models_and_embeddings()
    
    if not pairs:
        print("No valid model/embedding pairs found. Skipping performance analysis.")
        return
        
    print(f"Found {len(pairs)} model/embedding pairs for analysis.")
    
    # Generate dashboard from files
    generate_dashboard_from_files()
    
    # Generate prediction error analysis
    generate_model_predictions_analysis(pairs)
    
    # Compare model predictions
    compare_models(pairs)
    
    print("Comprehensive performance analysis complete.")


def embedding_analysis():
    """Run comprehensive embedding analysis"""
    print("Running comprehensive embedding analysis...")
    
    # Find all models and embeddings
    model_files, embedding_files, pairs = find_models_and_embeddings()
    
    if not pairs:
        print("No valid model/embedding pairs found. Skipping embedding analysis.")
        return
        
    # Generate embedding visualizations
    generate_embedding_visualizations(pairs)
    
    # Generate model representation analysis
    generate_model_representations(pairs)
    
    print("Comprehensive embedding analysis complete.")


def generate_comprehensive_thesis_visuals():
    """Generate all visualizations for thesis presentation"""
    print("Generating comprehensive thesis visualizations...")
    
    # Set up all directories
    setup_analysis_dirs()
    
    # Analyze MLflow experiments
    analyze_mlflow_experiments()
    
    # Run performance analysis
    performance_analysis()
    
    # Run embedding analysis
    embedding_analysis()
    
    # Analyze attention patterns
    analyze_attention_patterns("PCMs")
    
    # Generate learning curves for key configurations
    generate_learning_curve_analysis(
        model_types=["rf", "torch"], 
        embedding_types=["tfidf", "bert"]
    )
    
    print("Comprehensive thesis visualization generation complete. Check the 'figures' directory.")


def main():
    """Main function to parse arguments and run analysis"""
    parser = argparse.ArgumentParser(description="Generate visualizations and analysis for thesis")
    
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Generate all thesis visualizations"
    )
    parser.add_argument(
        "--mlflow", 
        action="store_true", 
        help="Analyze MLflow experiments"
    )
    parser.add_argument(
        "--training_metrics", 
        action="store_true", 
        help="Visualize training metrics for PyTorch models"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help="Specific MLflow run ID to visualize training metrics for"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="Run performance analysis"
    )
    parser.add_argument(
        "--embeddings", 
        action="store_true", 
        help="Run embedding analysis"
    )
    parser.add_argument(
        "--attention", 
        action="store_true", 
        help="Analyze attention patterns"
    )
    parser.add_argument(
        "--code_dir", 
        type=str, 
        default="PCMs",
        help="Directory containing code files for attention analysis"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/codebert-base",
        help="Model to use for attention analysis"
    )
    parser.add_argument(
        "--learning_curves", 
        action="store_true", 
        help="Generate learning curves"
    )
    
    args = parser.parse_args()
    
    # Set up directories
    setup_analysis_dirs()
    
    # If no specific arguments provided, show help
    if not any([args.all, args.mlflow, args.performance, args.embeddings, 
                args.attention, args.learning_curves, args.training_metrics]):
        parser.print_help()
        return
    
    # Run requested analyses
    if args.all:
        generate_comprehensive_thesis_visuals()
    else:
        if args.mlflow:
            analyze_mlflow_experiments()
        
        if args.training_metrics:
            print("Visualizing training metrics...")
            try:
                df = visualize_training_metrics(run_id=args.run_id)
                if df is not None:
                    print(f"Generated training metrics visualization with {len(df)} epochs.")
                else:
                    print("No PyTorch training metrics found.")
            except Exception as e:
                print(f"Error visualizing training metrics: {e}")
        
        if args.performance:
            performance_analysis()
        
        if args.embeddings:
            embedding_analysis()
        
        if args.attention:
            analyze_attention_patterns(args.code_dir, args.model_name)
        
        if args.learning_curves:
            model_types = ["rf", "torch"]
            embedding_types = ["tfidf", "bert"]
            generate_learning_curve_analysis(model_types, embedding_types)
    
    print("Analysis complete. Check 'figures' directory for results.")


if __name__ == "__main__":
    main()