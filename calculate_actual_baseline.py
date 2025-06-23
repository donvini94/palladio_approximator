#!/usr/bin/env python3
"""
Calculate actual baseline performance by training and evaluating a dummy model.

This script loads your actual dataset, trains a dummy model that always predicts
the mean of the training targets, and evaluates it on test/validation sets.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import argparse

# Add the project root to the path to import data loading utilities
sys.path.append('.')

def load_actual_dataset():
    """Load the actual dataset used in your experiments."""
    try:
        # Try to load the dataset using your existing data loading pipeline
        # This should match how you load data in train.py
        
        # Option 1: Load from saved splits if they exist
        data_files = {
            'train': None,
            'test': None, 
            'val': None
        }
        
        data_dir = Path('data')
        if data_dir.exists():
            # Look for saved dataset splits
            for split in ['train', 'test', 'val']:
                for ext in ['.pkl', '.csv', '.parquet']:
                    file_path = data_dir / f'{split}_data{ext}'
                    if file_path.exists():
                        print(f"Found {split} data: {file_path}")
                        if ext == '.pkl':
                            data_files[split] = pd.read_pickle(file_path)
                        elif ext == '.csv':
                            data_files[split] = pd.read_csv(file_path)
                        elif ext == '.parquet':
                            data_files[split] = pd.read_parquet(file_path)
                        break
        
        # Option 2: Try to import and use your data loading functions
        if not any(data_files.values()):
            try:
                from utils.data_loader import load_data
                print("Using your data loading pipeline...")
                data_files = load_data()
            except ImportError:
                print("Could not import data loading utilities")
        
        # Option 3: Look for any dataset files and split manually
        if not any(data_files.values()):
            dataset_files = []
            for pattern in ['*.csv', '*.pkl', '*.parquet']:
                dataset_files.extend(list(data_dir.glob(pattern)))
                dataset_files.extend(list(data_dir.glob(f'**/{pattern}')))
            
            if dataset_files:
                print(f"Found dataset files: {[f.name for f in dataset_files]}")
                # Load the first suitable file
                main_file = dataset_files[0]
                print(f"Loading from {main_file}")
                
                if main_file.suffix == '.pkl':
                    full_data = pd.read_pickle(main_file)
                elif main_file.suffix == '.csv':
                    full_data = pd.read_csv(main_file)
                elif main_file.suffix == '.parquet':
                    full_data = pd.read_parquet(main_file)
                
                # Split the data manually (80/10/10 split)
                train_data, temp_data = train_test_split(full_data, test_size=0.2, random_state=42)
                test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)
                
                data_files = {
                    'train': train_data,
                    'test': test_data,
                    'val': val_data
                }
        
        return data_files
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def extract_features_and_targets(data_files):
    """Extract features and targets from the loaded data."""
    extracted_data = {}
    
    for split_name, df in data_files.items():
        if df is not None and not df.empty:
            print(f"Processing {split_name} data: {len(df)} samples")
            
            # Look for target columns (response times)
            target_cols = []
            
            # Common patterns for target columns
            target_patterns = [
                'response_time', 'response', 'time', 'latency', 'duration',
                'y', 'target', 'label', 'output'
            ]
            
            for col in df.columns:
                if any(pattern in col.lower() for pattern in target_patterns):
                    target_cols.append(col)
            
            if not target_cols:
                print(f"Could not identify target columns in {split_name} data")
                print(f"Available columns: {list(df.columns)}")
                # Use the last column as target (common convention)
                target_cols = [df.columns[-1]]
                print(f"Using last column as target: {target_cols[0]}")
            
            # Extract targets (assume single target for now)
            targets = df[target_cols[0]].values
            
            # Extract features (everything except targets)
            feature_cols = [col for col in df.columns if col not in target_cols]
            features = df[feature_cols].values if feature_cols else None
            
            extracted_data[split_name] = {
                'features': features,
                'targets': targets,
                'target_column': target_cols[0]
            }
            
            print(f"  Target column: {target_cols[0]}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Target stats: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
            if features is not None:
                print(f"  Features shape: {features.shape}")
    
    return extracted_data


def train_and_evaluate_dummy_model(data, normalize=False):
    """Train a dummy model and evaluate it properly."""
    print(f"\nTraining dummy model (normalize={normalize})...")
    
    if 'train' not in data or 'test' not in data:
        print("Need both train and test data")
        return None
    
    # Get the raw targets
    y_train = data['train']['targets']
    y_test = data['test']['targets']
    y_val = data['val']['targets'] if 'val' in data else y_test
    
    print(f"Train targets: {y_train.shape}, Test targets: {y_test.shape}")
    
    if normalize:
        # Apply target normalization (as done in your experiments)
        print("Applying target normalization...")
        scaler = StandardScaler()
        
        # Fit scaler on training data only
        y_train_norm = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_norm = scaler.transform(y_test.reshape(-1, 1)).flatten()
        y_val_norm = scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Use normalized targets
        train_targets = y_train_norm
        test_targets = y_test_norm
        val_targets = y_val_norm
        
        print(f"Normalized train targets: mean={train_targets.mean():.4f}, std={train_targets.std():.4f}")
        print(f"Normalized test targets: mean={test_targets.mean():.4f}, std={test_targets.std():.4f}")
    else:
        # Use raw targets
        train_targets = y_train
        test_targets = y_test
        val_targets = y_val
    
    # Create and train dummy model
    dummy_model = DummyRegressor(strategy='mean')
    
    # Dummy features (not used by DummyRegressor, but needed for interface)
    X_train_dummy = np.zeros((len(train_targets), 1))
    X_test_dummy = np.zeros((len(test_targets), 1))
    X_val_dummy = np.zeros((len(val_targets), 1))
    
    # Fit dummy model on training data
    dummy_model.fit(X_train_dummy, train_targets)
    
    # Get predictions (will be the mean of training targets)
    pred_test = dummy_model.predict(X_test_dummy)
    pred_val = dummy_model.predict(X_val_dummy)
    
    print(f"Dummy model prediction (constant): {pred_test[0]:.6f}")
    print(f"Training targets mean: {train_targets.mean():.6f}")
    
    # Calculate metrics
    baseline_metrics = {
        'metrics.test_mse_avg': mean_squared_error(test_targets, pred_test),
        'metrics.test_mae_avg': mean_absolute_error(test_targets, pred_test),
        'metrics.test_r2_avg': r2_score(test_targets, pred_test),
        'metrics.test_rmse_avg': np.sqrt(mean_squared_error(test_targets, pred_test)),
        'metrics.val_mse_avg': mean_squared_error(val_targets, pred_val),
        'metrics.val_mae_avg': mean_absolute_error(val_targets, pred_val),
        'metrics.val_r2_avg': r2_score(val_targets, pred_val),
        'metrics.val_rmse_avg': np.sqrt(mean_squared_error(val_targets, pred_val)),
    }
    
    print("Baseline metrics:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return baseline_metrics


def main():
    parser = argparse.ArgumentParser(description='Calculate actual baseline performance')
    parser.add_argument('--output', type=str, default='actual_baseline_metrics.py', 
                       help='Output file for baseline code')
    
    args = parser.parse_args()
    
    print("=== Calculating Actual Baseline Performance ===")
    print("This will train a dummy model that predicts the mean and evaluate it properly.")
    
    # Load the actual dataset
    print("\n1. Loading dataset...")
    data_files = load_actual_dataset()
    
    if not data_files or not any(data_files.values()):
        print("Could not load dataset. Please check your data directory structure.")
        return
    
    # Extract features and targets
    print("\n2. Extracting features and targets...")
    data = extract_features_and_targets(data_files)
    
    if not data:
        print("Could not extract data.")
        return
    
    # Calculate baseline for both normalization conditions
    print("\n3. Training and evaluating dummy models...")
    
    baseline_norm = train_and_evaluate_dummy_model(data, normalize=True)
    baseline_no_norm = train_and_evaluate_dummy_model(data, normalize=False)
    
    if baseline_norm and baseline_no_norm:
        print("\n=== RESULTS ===")
        print("\nBaseline Performance (Dummy Model Predicting Training Mean):")
        print("\nWith Target Normalization:")
        for metric, value in baseline_norm.items():
            print(f"  {metric}: {value:.6f}")
        
        print("\nWithout Target Normalization:")  
        for metric, value in baseline_no_norm.items():
            print(f"  {metric}: {value:.6f}")
        
        # Generate updated function code
        code_template = f'''def calculate_baseline_performance(embedding: str, normalize: bool) -> Dict[str, float]:
    """Calculate baseline performance (dummy model predicting mean) for comparison.
    
    These values were calculated by actually training a DummyRegressor(strategy='mean')
    on your dataset and evaluating it on the test set.
    """
    
    if normalize:
        # Baseline performance with target normalization
        baseline_values = {{
            'metrics.test_mse_avg': {baseline_norm['metrics.test_mse_avg']:.6f},
            'metrics.test_mae_avg': {baseline_norm['metrics.test_mae_avg']:.6f},
            'metrics.test_r2_avg': {baseline_norm['metrics.test_r2_avg']:.6f},
            'metrics.test_rmse_avg': {baseline_norm['metrics.test_rmse_avg']:.6f},
            'metrics.val_mse_avg': {baseline_norm['metrics.val_mse_avg']:.6f},
            'metrics.val_mae_avg': {baseline_norm['metrics.val_mae_avg']:.6f},
            'metrics.val_r2_avg': {baseline_norm['metrics.val_r2_avg']:.6f},
            'metrics.val_rmse_avg': {baseline_norm['metrics.val_rmse_avg']:.6f},
        }}
    else:
        # Baseline performance without target normalization
        baseline_values = {{
            'metrics.test_mse_avg': {baseline_no_norm['metrics.test_mse_avg']:.6f},
            'metrics.test_mae_avg': {baseline_no_norm['metrics.test_mae_avg']:.6f},
            'metrics.test_r2_avg': {baseline_no_norm['metrics.test_r2_avg']:.6f},
            'metrics.test_rmse_avg': {baseline_no_norm['metrics.test_rmse_avg']:.6f},
            'metrics.val_mse_avg': {baseline_no_norm['metrics.val_mse_avg']:.6f},
            'metrics.val_mae_avg': {baseline_no_norm['metrics.val_mae_avg']:.6f},
            'metrics.val_r2_avg': {baseline_no_norm['metrics.val_r2_avg']:.6f},
            'metrics.val_rmse_avg': {baseline_no_norm['metrics.val_rmse_avg']:.6f},
        }}
    
    return baseline_values
'''
        
        with open(args.output, 'w') as f:
            f.write(code_template)
        
        print(f"\n✅ Generated actual baseline function in {args.output}")
        print("Replace the calculate_baseline_performance function in your plotting scripts with this code.")
        
        # Show improvement analysis
        print("\n=== Model Improvement Analysis ===")
        print("Compare your model performance against these baselines to show improvement:")
        print(f"- Any MSE below {baseline_norm['metrics.test_mse_avg']:.3f} (norm) or {baseline_no_norm['metrics.test_mse_avg']:.3f} (no norm) beats baseline")
        print(f"- Any R² above {baseline_norm['metrics.test_r2_avg']:.3f} shows explained variance")
        print(f"- Baseline R² ≈ 0 confirms dummy model predicts mean correctly")
        
    else:
        print("Failed to calculate baseline metrics.")


if __name__ == "__main__":
    main()