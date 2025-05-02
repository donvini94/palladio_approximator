"""
PyTorch-based regression model for embeddings.
This file implements a neural network that can be used as a substitute for Random Forest when GPU acceleration is desired.
Optimized for RTX 3090 GPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm


class EmbeddingRegressorNet(nn.Module):
    """Neural network for regression on embeddings, optimized for RTX 3090."""

    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
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

        # For large datasets, predict in batches to avoid OOM errors
        if X.shape[0] > 10000:
            batch_size = 1024  # Larger batch size for RTX 3090
            outputs = []
            
            # Predict in batches
            for i in range(0, X.shape[0], batch_size):
                # Set model to eval mode
                self.model.eval()
                with torch.no_grad():
                    batch_X = X[i:i+batch_size]
                    batch_pred = self.model(batch_X)
                    outputs.append(batch_pred.cpu().numpy())
                    
            # Combine all batches
            return np.vstack(outputs)
        else:
            # For smaller datasets, predict all at once
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X)
            return y_pred.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def train_torch_regressor(
    X_train,
    y_train,
    epochs=150,               # Increased epochs for better convergence
    batch_size=128,           # Larger batch size for RTX 3090
    learning_rate=0.001,
    device=None,
    hidden_dims=[512, 256, 128],  # Larger model for RTX 3090
    patience=15,              # Increased patience
    verbose=True,
):
    """
    Train a PyTorch regression model on embeddings, optimized for RTX 3090 GPU.

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

    # Check if we're using CUDA and optimize settings
    if device == "cuda" and torch.cuda.is_available():
        # Set optimal CUDA settings for RTX 3090
        torch.backends.cudnn.benchmark = True  # Speed up training
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN
        
        # Check available memory and adjust batch size if needed
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_mem_gb = free_mem / (1024**3)
        print(f"Available CUDA memory: {free_mem_gb:.2f} GB")
        
        # If memory is constrained, reduce batch size
        if free_mem_gb < 6 and batch_size > 64:
            print(f"Reducing batch size from {batch_size} to 64 due to memory constraints")
            batch_size = 64

    start_time = time.time()
    if verbose:
        print(f"Starting PyTorch regressor training on {device}...")
        print(f"Input data: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Model architecture: hidden_dims={hidden_dims}")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create dataset and dataloader with appropriate workers for RTX 3090
    num_workers = 4 if device == "cuda" else 0  # Use multiple workers on GPU
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=num_workers,
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
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=(device == "cuda"),
        num_workers=num_workers
    )

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = EmbeddingRegressorNet(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Use OneCycleLR scheduler for improved convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000,
    )

    # Enable automatic mixed precision for faster training on RTX 3090
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Progress bar for training
    progress_bar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Use mixed precision training if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(batch_X)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                
                # Backward and optimize with scaled gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                # Forward pass
                outputs = model(batch_X)
                if output_dim == 1:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
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

                # Use mixed precision if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        if output_dim == 1:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Update progress bar
        if verbose:
            progress_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"PyTorch model training completed in {elapsed_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Create sklearn-compatible wrapper
    wrapped_model = TorchRegressorWrapper(model, device)

    # Save the model
    model_filename = "torch_regressor_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dims': hidden_dims,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'best_val_loss': best_val_loss,
    }, model_filename)
    
    if verbose:
        print(f"Saved PyTorch model to {model_filename}")

    # Return the wrapped model
    return wrapped_model