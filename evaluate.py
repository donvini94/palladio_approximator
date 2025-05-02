from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(model, X, y, split_name="val", verbose=True):
    """
    Evaluate model predictions with MSE, MAE, and R² per output dimension.

    Parameters:
        model: Trained regression model
        X: Input features
        y: True targets
        split_name: Label for this dataset split (e.g., 'val', 'test')
        verbose: Whether to print results

    Returns:
        dict: Dictionary of metric names and values
    """
    predictions = model.predict(X)
    
    # Handle potential shape differences gracefully
    if len(predictions.shape) == 1 and len(y.shape) > 1:
        predictions = predictions.reshape(-1, 1)
    
    # Calculate metrics with multioutput mode
    mse = mean_squared_error(y, predictions, multioutput="raw_values")
    mae = mean_absolute_error(y, predictions, multioutput="raw_values")
    r2 = r2_score(y, predictions, multioutput="raw_values")
    
    # Handle both single target and multi-target cases
    if len(mse.shape) == 0:
        # Single target case
        results = {
            f"{split_name}_mse": float(mse),
            f"{split_name}_mae": float(mae),
            f"{split_name}_r2": float(r2),
        }
    else:
        # Multi-target case (avg, min, max)
        results = {
            f"{split_name}_mse_avg": float(mse[0]),
            f"{split_name}_mse_min": float(mse[1]) if len(mse) > 1 else float(mse[0]),
            f"{split_name}_mse_max": float(mse[2]) if len(mse) > 2 else float(mse[0]),
            f"{split_name}_mae_avg": float(mae[0]),
            f"{split_name}_mae_min": float(mae[1]) if len(mae) > 1 else float(mae[0]),
            f"{split_name}_mae_max": float(mae[2]) if len(mae) > 2 else float(mae[0]),
            f"{split_name}_r2_avg": float(r2[0]),
            f"{split_name}_r2_min": float(r2[1]) if len(r2) > 1 else float(r2[0]),
            f"{split_name}_r2_max": float(r2[2]) if len(r2) > 2 else float(r2[0]),
        }

    if verbose:
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

    return results