"""
GPU-accelerated Random Forest using RAPIDS cuML.
This file implements a Random Forest regressor that runs on NVIDIA GPUs.
"""

import numpy as np
import time
import joblib


def train_gpu_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=16,  # RAPIDS RF typically needs a max_depth limit
    n_bins=128,  # Number of bins used for finding splits
    max_features=None,
    random_state=42,
    verbose=True,
):
    """
    Train a random forest regressor using GPU acceleration via RAPIDS cuML.

    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        n_bins: Number of bins used by the split-finding algorithm
        max_features: Number of features to consider for best split
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Returns:
        Trained RandomForestRegressor model from cuML
    """
    try:
        import cudf
        import cuml
        from cuml.ensemble import RandomForestRegressor

        has_rapids = True
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

        has_rapids = False
        print(
            "WARNING: RAPIDS not found. Using scikit-learn CPU implementation instead."
        )

    start_time = time.time()

    if verbose:
        print(
            f"Starting {'GPU' if has_rapids else 'CPU'} random forest training with {n_estimators} trees..."
        )
        print(f"Input data shape: X_train={X_train.shape}, y_train={y_train.shape}")

    if has_rapids:
        # RAPIDS implementation
        try:
            # Convert data to GPU if needed
            if isinstance(X_train, np.ndarray):
                if verbose:
                    print("Converting numpy arrays to cuDF DataFrame/Series...")
                X_train_gpu = cudf.DataFrame(X_train)

                # Handle multi-output regression
                if y_train.ndim > 1 and y_train.shape[1] > 1:
                    if verbose:
                        print(
                            f"Multi-output regression detected with {y_train.shape[1]} targets"
                        )
                    # RAPIDS doesn't support multi-output directly, so we'll train separate models
                    models = []
                    for i in range(y_train.shape[1]):
                        if verbose:
                            print(
                                f"Training model for output {i+1}/{y_train.shape[1]}..."
                            )
                        y_train_gpu = cudf.Series(y_train[:, i])

                        # Create and train the model
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            n_bins=n_bins,
                            max_features=max_features,
                            random_state=random_state,
                        )
                        model.fit(X_train_gpu, y_train_gpu)
                        models.append(model)

                    # Create a wrapper for multi-output prediction
                    class MultiOutputRF:
                        def __init__(self, models):
                            self.models = models

                        def predict(self, X):
                            # Convert to GPU if needed
                            if isinstance(X, np.ndarray):
                                X_gpu = cudf.DataFrame(X)
                            else:
                                X_gpu = X

                            # Get predictions from each model
                            preds = [
                                model.predict(X_gpu).values for model in self.models
                            ]

                            # Combine predictions
                            return np.column_stack(preds)

                    final_model = MultiOutputRF(models)
                else:
                    # Single output regression
                    y_train_gpu = cudf.Series(y_train)

                    # Create and train the model
                    final_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        n_bins=n_bins,
                        max_features=max_features,
                        random_state=random_state,
                    )
                    final_model.fit(X_train_gpu, y_train_gpu)
            else:
                # If data is already on GPU (unlikely with your pipeline)
                final_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_bins=n_bins,
                    max_features=max_features,
                    random_state=random_state,
                )
                final_model.fit(X_train, y_train)

        except Exception as e:
            # Fall back to scikit-learn if RAPIDS fails
            print(f"ERROR using RAPIDS: {e}")
            print("Falling back to scikit-learn CPU implementation")
            from sklearn.ensemble import RandomForestRegressor as SklearnRF

            final_model = SklearnRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,  # Use all CPU cores
                verbose=1 if verbose else 0,
            )
            final_model.fit(X_train, y_train)

    else:
        # scikit-learn fallback
        from sklearn.ensemble import RandomForestRegressor as SklearnRF

        final_model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1 if verbose else 0,
        )
        final_model.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Random forest training completed in {elapsed_time:.2f} seconds")

    return final_model
