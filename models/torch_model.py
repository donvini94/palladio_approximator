"""
PyTorch-based regression model for embeddings.
This file implements a neural network that can be used as a substitute for Random Forest when GPU acceleration is desired.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import joblib
from sklearn.base import BaseEstimator, RegressorMixin


class EmbeddingRegressorNet(nn.Module):
    """Neural network for regression on embeddings."""

    def __init__(self, input_dim, output_dim, hidden_dims=[400, 200, 100]):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(0.3))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.layers.append(nn.Dropout(0.2))

        # Output layer
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)


class TorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for PyTorch regression model.
    Makes it compatible with existing code for evaluation.
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def predict(self, X):
        # Convert to torch tensor if necessary
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Set model to eval mode
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)

        # Return as numpy array
        return y_pred.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def train_torch_regressor(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    device=None,
    hidden_dims=[400, 200, 100],
    patience=10,  # For early stopping
    verbose=True,
):
    """
    Train a PyTorch regression model on embeddings, leveraging GPU if available.

    Args:
        X_train: Training features (numpy array)
        y_train: Training targets (numpy array)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        hidden_dims: List of hidden layer dimensions
        patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        Trained model wrapped in a scikit-learn compatible wrapper
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()
    if verbose:
        print(f"Starting PyTorch regressor training on {device}...")
        print(f"Input data: X_train={X_train.shape}, y_train={y_train.shape}")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    # Create validation set (10% of training data)
    val_size = int(0.1 * len(X_train))
    train_size = len(X_train) - val_size

    # Split indices
    indices = torch.randperm(len(X_train))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create validation dataset and loader
    val_dataset = TensorDataset(
        X_train_tensor[val_indices], y_train_tensor[val_indices]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device == "cuda")
    )

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = EmbeddingRegressorNet(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=verbose
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            if output_dim == 1:
                outputs = outputs.squeeze()

            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                if output_dim == 1:
                    outputs = outputs.squeeze()

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"PyTorch model training completed in {elapsed_time:.2f} seconds")

    # Create sklearn-compatible wrapper
    wrapped_model = TorchRegressorWrapper(model, device)

    # Save the model
    torch.save(model.state_dict(), "torch_regressor_model.pt")
    if verbose:
        print("Saved PyTorch model to torch_regressor_model.pt")

    # Return the wrapped model
    return wrapped_model
