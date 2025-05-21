"""
PyTorch-based regression model for embeddings.
This file implements a neural network that can be used as a substitute for Random Forest when GPU acceleration is desired.
Optimized for RTX 3090 GPU.
"""

import os

# Set tokenizers parallelism explicitly to prevent warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    """Enhanced neural network for regression on embeddings, with advanced architecture."""

    def __init__(self, input_dim, output_dim, hidden_dims=[768, 512, 256, 128]):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.LeakyReLU(0.1)  # LeakyReLU for better gradient flow
        self.input_dropout = nn.Dropout(0.2)  # Initial dropout to reduce overfitting

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(self.activation)
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(0.4))  # Higher dropout for first layer

        # Hidden layers with residual connections when possible
        for i in range(len(hidden_dims) - 1):
            # Add main layer
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(self.activation)
            self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

            # Gradually decrease dropout as we go deeper
            dropout_rate = max(0.1, 0.4 - 0.05 * i)
            self.layers.append(nn.Dropout(dropout_rate))

        # Final hidden layer with reduced dropout
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            self.activation,
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Dropout(0.1),
        )

        # Output layer
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Initialize weights for better training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Apply input batch normalization and dropout
        x = self.input_bn(x)
        x = self.input_dropout(x)

        # Forward through main layers
        for layer in self.layers:
            x = layer(x)

        # Final pre-output block
        x = self.pre_output(x)

        # Output layer
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
        # Check if this is a large sparse matrix (memory-efficient handling)
        is_sparse = hasattr(X, "toarray")
        is_large_sparse = is_sparse and X.shape[1] > 10000

        # For large sparse matrices, use batched prediction without full conversion
        if is_large_sparse:
            from scipy import sparse

            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()  # Convert to CSR for efficient row slicing

            # Use smaller batch size for very large matrices
            batch_size = 64 if X.shape[1] > 100000 else 128
            outputs = []
            n_samples = X.shape[0]

            # Set model to eval mode
            self.model.eval()

            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_indices = list(range(i, end_idx))

                # Extract batch and convert to dense tensor
                with torch.no_grad():
                    batch_X = torch.tensor(
                        X[batch_indices].toarray(), dtype=torch.float32
                    ).to(self.device)
                    batch_pred = self.model(batch_X)
                    outputs.append(batch_pred.cpu().numpy())

                # Clear memory periodically
                if i % (batch_size * 10) == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

            # Combine all batches
            return np.vstack(outputs)

        # For dense arrays or smaller sparse matrices, use standard approach
        if not isinstance(X, torch.Tensor):
            # Handle sparse matrices
            if is_sparse:
                X = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
            else:
                X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # For large (but not sparse) datasets, predict in batches to avoid OOM errors
        if X.shape[0] > 1000:
            batch_size = 1024  # Larger batch size for RTX 3090
            outputs = []

            # Predict in batches
            for i in range(0, X.shape[0], batch_size):
                # Set model to eval mode
                self.model.eval()
                with torch.no_grad():
                    batch_X = X[i : i + batch_size]
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
    epochs=300,  # Many epochs for thorough training
    batch_size=128,  # Moderate batch size for better generalization
    learning_rate=0.0008,  # Smaller learning rate for more stable training
    device=None,
    hidden_dims=[768, 512, 256, 128],  # Deeper network for better feature learning
    patience=30,  # Higher patience to avoid early stopping
    verbose=True,
):
    """
    Train a PyTorch regression model on embeddings, optimized for RTX 3090 GPU.

    Args:
        X_train: Training features (numpy array or sparse matrix)
        y_train: Training targets (numpy array)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        hidden_dims: List of hidden layer dimensions
        patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        Trained model wrapped in a scikit-learn compatible wrapper, with
        training metrics (train_loss, val_loss, etc.) stored as attributes
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
        free_mem = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        free_mem_gb = free_mem / (1024**3)
        print(f"Available CUDA memory: {free_mem_gb:.2f} GB")

        # If memory is constrained, reduce batch size
        if free_mem_gb < 6 and batch_size > 64:
            print(
                f"Reducing batch size from {batch_size} to 64 due to memory constraints"
            )
            batch_size = 64

    start_time = time.time()
    if verbose:
        print(f"Starting PyTorch regressor training on {device}...")
        print(f"Input data: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Model architecture: hidden_dims={hidden_dims}")

    # Check if input is sparse matrix
    is_sparse = hasattr(X_train, "toarray")

    # Handle sparse matrices properly
    if is_sparse:
        print("Detected sparse matrix input. Using optimized sparse training.")
        from scipy import sparse

        if not sparse.isspmatrix_csr(X_train):
            print("Converting to CSR format for efficient row slicing...")
            X_train = X_train.tocsr()

        # Create validation split manually
        n_samples = X_train.shape[0]
        val_size = int(0.1 * n_samples)
        train_size = n_samples - val_size

        # Create indices with permutation
        all_indices = np.random.permutation(n_samples)
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        # Prepare target tensor
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Function to extract batch from sparse matrix
        def get_batch(indices, sparse_matrix, target_tensor):
            X_batch = torch.tensor(
                sparse_matrix[indices].toarray(), dtype=torch.float32
            ).to(device)
            y_batch = target_tensor[indices].to(device)
            return X_batch, y_batch

    else:
        # Dense input - use TensorDataset
        print("Using standard tensor-based training approach.")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Create dataset and dataloader
        full_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        # Split into train/val
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Define worker initialization function for DataLoader
        def worker_init_fn(worker_id):
            # Each worker should have different but deterministic random seed
            torch.manual_seed(42 + worker_id)
            np.random.seed(42 + worker_id)
            # Ensure tokenizers parallelism is properly set in subprocesses
            if "TOKENIZERS_PARALLELISM" not in os.environ:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create dataloaders with worker initialization
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device == "cuda"),
            num_workers=4 if device == "cuda" else 0,
            worker_init_fn=worker_init_fn,
            persistent_workers=(
                True if device == "cuda" and torch.cuda.is_available() else False
            ),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(device == "cuda"),
            num_workers=4 if device == "cuda" else 0,
            worker_init_fn=worker_init_fn,
            persistent_workers=(
                True if device == "cuda" and torch.cuda.is_available() else False
            ),
        )

    # Create model with input dim from data
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # Initialize model
    model = EmbeddingRegressorNet(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    ).to(device)

    # Define loss functions
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    # Combined loss function for robustness
    def combined_loss(pred, target, alpha=0.8):
        """Combined MSE and L1 loss for robust regression"""
        mse = mse_criterion(pred, target)
        l1 = l1_criterion(pred, target)
        return alpha * mse + (1 - alpha) * l1

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Learning rate scheduler with plateau detection
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,  # Halve the learning rate when plateauing
        patience=10,  # Wait 10 epochs before reducing LR
        threshold=0.001,
        min_lr=learning_rate / 100,  # Don't go below this LR
    )

    # Initialize metric history tracking
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "train_mae": [],
        "val_mse": [],
        "val_mae": [],
        "learning_rate": [],
    }

    # Add batch-level tracking
    batch_history = {"train_loss_batches": []}

    # Initialize best metrics tracking
    best_metrics = {
        "val_loss": float("inf"),
        "val_mse": float("inf"),
        "val_mae": float("inf"),
        "epoch": 0,
    }

    # Enable mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Progress bar for epochs
    progress_bar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    # Batch generator for sparse matrices
    def batch_generator(indices, batch_size):
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            yield indices[i : i + batch_size]

    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Handle sparse vs dense training differently
        if is_sparse:
            # Sparse matrix training
            for batch_indices in batch_generator(train_indices, batch_size):
                # Get batch data
                X_batch, y_batch = get_batch(batch_indices, X_train, y_train_tensor)

                # Mixed precision training if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        if output_dim == 1:
                            outputs = outputs.squeeze()
                        loss = combined_loss(outputs, y_batch)

                    # Gradient optimization with scaling
                    optimizer.zero_grad(set_to_none=True)  # More efficient zeroing
                    scaler.scale(loss).backward()

                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    outputs = model(X_batch)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    loss = combined_loss(outputs, y_batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Store batch loss for both running total and batch history
                loss_val = loss.item()
                train_loss += loss_val
                train_batches += 1

                # Store batch loss for finer-grained tracking
                batch_history["train_loss_batches"].append(loss_val)

                # Clear GPU memory periodically
                if train_batches % 10 == 0 and device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Dense tensor training
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Mixed precision training if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        if output_dim == 1:
                            outputs = outputs.squeeze()
                        loss = combined_loss(outputs, y_batch)

                    # Gradient optimization with scaling
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()

                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    outputs = model(X_batch)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    loss = combined_loss(outputs, y_batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Store batch loss for both running total and batch history
                loss_val = loss.item()
                train_loss += loss_val
                train_batches += 1

                # Store batch loss for finer-grained tracking
                batch_history["train_loss_batches"].append(loss_val)

        # Calculate average training loss
        train_loss /= train_batches if train_batches > 0 else 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            # Handle sparse vs dense validation differently
            if is_sparse:
                # Sparse matrix validation
                for batch_indices in batch_generator(val_indices, batch_size):
                    # Get batch data
                    X_batch, y_batch = get_batch(batch_indices, X_train, y_train_tensor)

                    # Use mixed precision for validation too
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(X_batch)
                            if output_dim == 1:
                                outputs = outputs.squeeze()
                            loss = combined_loss(outputs, y_batch)
                    else:
                        outputs = model(X_batch)
                        if output_dim == 1:
                            outputs = outputs.squeeze()
                        loss = combined_loss(outputs, y_batch)

                    val_loss += loss.item()
                    val_batches += 1
            else:
                # Dense tensor validation
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    # Use mixed precision for validation too
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(X_batch)
                            if output_dim == 1:
                                outputs = outputs.squeeze()
                            loss = combined_loss(outputs, y_batch)
                    else:
                        outputs = model(X_batch)
                        if output_dim == 1:
                            outputs = outputs.squeeze()
                        loss = combined_loss(outputs, y_batch)

                    val_loss += loss.item()
                    val_batches += 1

        # Calculate average validation loss
        val_loss /= val_batches if val_batches > 0 else 1

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update progress bar
        if verbose:
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

        # Calculate additional metrics
        model.eval()
        with torch.no_grad():
            # Calculate MSE and MAE on validation set
            val_mse = 0.0
            val_mae = 0.0
            train_mse = 0.0
            train_mae = 0.0

            # Calculate metrics on a validation subset to save time
            if is_sparse:
                # Use a subset of val_indices to save time
                subset_size = min(1000, len(val_indices))
                subset_indices = val_indices[:subset_size]
                X_val_subset, y_val_subset = get_batch(
                    subset_indices, X_train, y_train_tensor
                )
                outputs = model(X_val_subset)
                if output_dim == 1:
                    outputs = outputs.squeeze()
                val_mse = mse_criterion(outputs, y_val_subset).item()
                val_mae = l1_criterion(outputs, y_val_subset).item()

                # Also calculate training metrics on a subset
                train_subset_size = min(1000, len(train_indices))
                train_subset_indices = train_indices[:train_subset_size]
                X_train_subset, y_train_subset = get_batch(
                    train_subset_indices, X_train, y_train_tensor
                )
                train_outputs = model(X_train_subset)
                if output_dim == 1:
                    train_outputs = train_outputs.squeeze()
                train_mse = mse_criterion(train_outputs, y_train_subset).item()
                train_mae = l1_criterion(train_outputs, y_train_subset).item()
            else:
                # Try to use a full validation batch
                for X_batch, y_batch in [next(iter(val_loader))]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    val_mse = mse_criterion(outputs, y_batch).item()
                    val_mae = l1_criterion(outputs, y_batch).item()
                    break  # Just use one batch

                # Calculate training metrics on one batch
                for X_batch, y_batch in [next(iter(train_loader))]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    if output_dim == 1:
                        outputs = outputs.squeeze()
                    train_mse = mse_criterion(outputs, y_batch).item()
                    train_mae = l1_criterion(outputs, y_batch).item()
                    break  # Just use one batch

        # Store metrics for history
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_mse"].append(val_mse)
        metrics_history["val_mae"].append(val_mae)
        metrics_history["train_mse"].append(train_mse)
        metrics_history["train_mae"].append(train_mae)
        metrics_history["learning_rate"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            # Update best metrics
            best_metrics["val_loss"] = best_val_loss
            best_metrics["val_mse"] = val_mse
            best_metrics["val_mae"] = val_mae
            best_metrics["epoch"] = epoch + 1
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Report training time
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"PyTorch model training completed in {elapsed_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Create sklearn-compatible wrapper
    wrapped_model = TorchRegressorWrapper(model, device)

    # Save the model
    model_filename = "torch_regressor_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": hidden_dims,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "best_val_loss": best_val_loss,
        },
        model_filename,
    )

    # Add training metadata to the wrapped model
    wrapped_model.training_metrics = {
        "history": metrics_history,
        "batch_history": batch_history,
        "best_metrics": best_metrics,
        "training_time_seconds": elapsed_time,
        "final_val_loss": best_val_loss,
        "final_train_loss": train_loss,
        "total_epochs": epoch + 1,
        "total_steps": train_batches * (epoch + 1),  # Total number of batch updates
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": hidden_dims,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "early_stopping": patience,
        # Store pytorch model for access later
        "pytorch_model": model,
    }

    if verbose:
        print(f"Saved PyTorch model to {model_filename}")

    # Return the wrapped model
    return wrapped_model
