"""
Hyperparameter optimization module using Optuna.

This module provides functions for optimizing hyperparameters of different model types
(Random Forest, Ridge/Lasso Regression, PyTorch Neural Networks) using Optuna.
"""

import os
import time
import json
import optuna
import mlflow
import logging
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# Import model training functions
from models.rf_model import train_random_forest
from models.linear_model import train_linear_model
from models.torch_model import train_torch_regressor, EmbeddingRegressorNet
from utils.config import get_device
from evaluate import evaluate_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_optuna_pruner(args):
    """Create an appropriate pruner for Optuna trials based on args.

    Args:
        args: Command line arguments

    Returns:
        optuna.pruners.BasePruner: The pruner to use
    """
    if args.model == "torch":
        # Use more aggressive pruning for expensive PyTorch model training
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    else:
        # Use more conservative pruning for faster models
        return optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0,
            interval_steps=1
        )


def get_metric_value(results, metric_name):
    """Extract a specific metric value from evaluation results.

    Args:
        results (dict): Dictionary of evaluation results
        metric_name (str): Name of the metric to extract (e.g., 'val_mse')

    Returns:
        float: The value of the specified metric
    """
    # Handle case where metric might have avg/min/max variants
    if metric_name in results:
        return results[metric_name]
    
    # For multi-output metrics, default to average target
    if f"{metric_name}_avg" in results:
        return results[f"{metric_name}_avg"]
    
    # If still not found, log error and return a high value
    logger.error(f"Metric {metric_name} not found in results: {results}")
    return float('inf')


def optimize_rf_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30, metric='val_mse'):
    """
    Optimize Random Forest hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        metric: Metric to optimize ('val_mse', 'val_mae', 'val_r2')
        
    Returns:
        tuple: (best_params, study)
    """
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        # Train model with suggested hyperparameters
        model = train_random_forest(X_train, y_train, **params)
        
        # Evaluate on validation set
        val_results = evaluate_model(model, X_val, y_val, split_name="val", verbose=False)
        
        # Get the appropriate metric value (lower is better for MSE and MAE, higher is better for R²)
        metric_value = get_metric_value(val_results, metric)
        
        # Invert R² score since Optuna minimizes by default
        if 'r2' in metric:
            metric_value = -metric_value
            
        return metric_value
    
    # Create study object
    study = optuna.create_study(direction='minimize', 
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    
    # Optimize
    logger.info(f"Starting Random Forest hyperparameter optimization with {n_trials} trials")
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Return best parameters and study for analysis
    return best_params, study


def optimize_linear_hyperparameters(X_train, y_train, X_val, y_val, model_type, n_trials=30, metric='val_mse'):
    """
    Optimize Ridge or Lasso hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: 'ridge' or 'lasso'
        n_trials: Number of optimization trials
        metric: Metric to optimize ('val_mse', 'val_mae', 'val_r2')
        
    Returns:
        tuple: (best_params, study)
    """
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 10, log=True),
        }
        
        if model_type == 'ridge':
            params['solver'] = trial.suggest_categorical('solver', 
                ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            
        # Max iterations only for iterative solvers
        if model_type == 'lasso' or params.get('solver') in ['sag', 'saga']:
            params['max_iter'] = trial.suggest_int('max_iter', 1000, 10000)
        
        # Train model with suggested hyperparameters
        model = train_linear_model(X_train, y_train, model_type=model_type, **params)
        
        # Evaluate on validation set
        val_results = evaluate_model(model, X_val, y_val, split_name="val", verbose=False)
        
        # Get the appropriate metric value (lower is better for MSE and MAE, higher is better for R²)
        metric_value = get_metric_value(val_results, metric)
        
        # Invert R² score since Optuna minimizes by default
        if 'r2' in metric:
            metric_value = -metric_value
            
        return metric_value
    
    # Create study object
    study = optuna.create_study(direction='minimize', 
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    
    # Optimize
    logger.info(f"Starting {model_type.capitalize()} hyperparameter optimization with {n_trials} trials")
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Return best parameters and study for analysis
    return best_params, study


def optimize_torch_hyperparameters(X_train, y_train, X_val, y_val, device, n_trials=30, metric='val_mse'):
    """
    Optimize PyTorch neural network hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        device: Device to use (cuda or cpu)
        n_trials: Number of optimization trials
        metric: Metric to optimize ('val_mse', 'val_mae', 'val_r2')
        
    Returns:
        tuple: (best_params, study)
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1
    )
    
    def objective(trial):
        # Define hyperparameters to optimize
        # Limit epochs during optimization for efficiency
        epochs = 100
        
        # Choose batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        
        # Learning rate
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Dropout rate
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Configure hidden layer dimensions
        # First, determine number of hidden layers
        n_layers = trial.suggest_int('n_layers', 2, 4)
        
        # Determine first hidden layer size based on input_dim
        input_dim = X_train.shape[1]
        hidden_dims = []
        
        # Dynamically determine layer sizes
        first_layer_size = trial.suggest_categorical('first_layer_size', 
                                                   [256, 512, 768, 1024])
        hidden_dims.append(first_layer_size)
        
        # Add remaining layers with decreasing sizes
        for i in range(1, n_layers):
            # Each subsequent layer is a fraction of the previous layer's size
            prev_size = hidden_dims[-1]
            next_size = trial.suggest_int(f'layer_{i}_size', 
                                        max(64, prev_size // 4), 
                                        prev_size)
            hidden_dims.append(next_size)
        
        # Train model with suggested hyperparameters
        try:
            model = train_torch_regressor(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                hidden_dims=hidden_dims,
                patience=20,  # Lower patience for optimization
                verbose=False  # Disable verbose output during optimization
            )
            
            # Get the internal model's parameters for tracking
            if hasattr(model, "training_metrics"):
                # Report intermediate values for pruning
                history = model.training_metrics["history"]
                for epoch, val_loss in enumerate(history["val_loss"]):
                    trial.report(val_loss, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            
            # Evaluate on validation set
            val_results = evaluate_model(model, X_val, y_val, split_name="val", verbose=False)
            
            # Get the appropriate metric value (lower is better for MSE and MAE, higher is better for R²)
            metric_value = get_metric_value(val_results, metric)
            
            # Invert R² score since Optuna minimizes by default
            if 'r2' in metric:
                metric_value = -metric_value
                
            # Clean up GPU memory
            if device == "cuda":
                if hasattr(model, "training_metrics") and "pytorch_model" in model.training_metrics:
                    del model.training_metrics["pytorch_model"]
                torch.cuda.empty_cache()
                
            return metric_value
            
        except (RuntimeError, ValueError, Exception) as e:
            logger.error(f"Trial failed with error: {e}")
            # Clean up GPU memory on error
            if device == "cuda":
                torch.cuda.empty_cache()
            return float('inf')  # Return a high error value for failed trials
    
    # Create study object with pruner
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    # Optimize
    logger.info(f"Starting PyTorch hyperparameter optimization with {n_trials} trials")
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError, ValueError))
    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Convert the hidden dims structure to a list for the model
    hidden_dims = []
    n_layers = best_params.pop('n_layers')
    hidden_dims.append(best_params.pop('first_layer_size'))
    
    for i in range(1, n_layers):
        hidden_dims.append(best_params.pop(f'layer_{i}_size'))
    
    # Add hidden_dims to best_params
    best_params['hidden_dims'] = hidden_dims
    
    # Return best parameters and study for analysis
    return best_params, study


def run_hyperparameter_optimization(args, X_train, y_train, X_val, y_val):
    """
    Main function to run hyperparameter optimization based on model type.
    
    Args:
        args: Command line arguments
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tuple: (best_params, optimization_history)
    """
    # Set metric to optimize based on args
    metric = args.optimization_metric if hasattr(args, 'optimization_metric') else 'val_mse'
    n_trials = args.n_trials if hasattr(args, 'n_trials') else 30
    
    # Create directories for storing optimization results
    os.makedirs("models/optimization", exist_ok=True)
    
    # Log hyperparameter optimization process to MLflow if enabled
    if args.use_mlflow:
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimization_metric", metric)
    
    # Run optimization based on model type
    if args.model == "rf":
        best_params, study = optimize_rf_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=n_trials, metric=metric
        )
    
    elif args.model in ["ridge", "lasso"]:
        best_params, study = optimize_linear_hyperparameters(
            X_train, y_train, X_val, y_val, model_type=args.model, 
            n_trials=n_trials, metric=metric
        )
    
    elif args.model == "torch":
        device = get_device(args)
        best_params, study = optimize_torch_hyperparameters(
            X_train, y_train, X_val, y_val, device=device,
            n_trials=n_trials, metric=metric
        )
    
    else:
        raise ValueError(f"Unsupported model type for optimization: {args.model}")
    
    # Format optimization history for visualization
    optimization_history = {
        'best_params': best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'trials': []
    }
    
    # Add trial information to history
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_info = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value,
            }
            optimization_history['trials'].append(trial_info)
    
    # Save optimization results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"models/optimization/{args.model}_optimization_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(optimization_history, f, indent=2)
    
    logger.info(f"Saved optimization history to {results_file}")
    
    # Log results to MLflow if enabled
    if args.use_mlflow:
        # Log best parameters
        mlflow.log_params({f"opt_{k}": v for k, v in best_params.items()})
        
        # Log best value
        best_metric_name = f"optimized_{metric}"
        if 'r2' in metric:  # Correct the sign for R²
            mlflow.log_metric(best_metric_name, -study.best_value)
        else:
            mlflow.log_metric(best_metric_name, study.best_value)
            
        # Log optimization results as artifact
        mlflow.log_artifact(results_file)
        
        # Generate and log visualization plots if matplotlib is available
        try:
            from matplotlib import pyplot as plt
            
            # Optimization history plot
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            history_plot = f"models/optimization/{args.model}_history_{timestamp}.png"
            fig.savefig(history_plot)
            mlflow.log_artifact(history_plot)
            plt.close(fig)
            
            # Parameter importance plot
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            importance_plot = f"models/optimization/{args.model}_importance_{timestamp}.png"
            fig.savefig(importance_plot)
            mlflow.log_artifact(importance_plot)
            plt.close(fig)
            
            logger.info("Created and logged optimization visualizations")
        except Exception as e:
            logger.warning(f"Could not generate optimization visualizations: {e}")
    
    return best_params, optimization_history