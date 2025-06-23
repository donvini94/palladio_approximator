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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import joblib
import psutil
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm


class EmbeddingRegressorNet(nn.Module):
    """Enhanced neural network for regression on embeddings, with advanced architecture."""

    def __init__(self, input_dim, output_dim, hidden_dims=[768, 512, 256, 128], dropout_rate=0.3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.LeakyReLU(0.1)  # LeakyReLU for better gradient flow
        self.input_dropout = nn.Dropout(dropout_rate * 0.6)  # Initial dropout to reduce overfitting

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(self.activation)
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(dropout_rate))  # Higher dropout for first layer

        # Hidden layers with residual connections when possible
        for i in range(len(hidden_dims) - 1):
            # Add main layer
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(self.activation)
            self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

            # Gradually decrease dropout as we go deeper
            layer_dropout_rate = max(dropout_rate * 0.3, dropout_rate - 0.05 * i)
            self.layers.append(nn.Dropout(layer_dropout_rate))

        # Final hidden layer with reduced dropout
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            self.activation,
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Dropout(dropout_rate * 0.3),
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
    Enhanced sklearn-compatible wrapper for PyTorch regression models.
    Supports multiple architectures including uncertainty estimation.
    """

    def __init__(self, model, device="cpu", architecture_type="embedding_regressor"):
        self.model = model
        self.device = device
        self.architecture_type = architecture_type

    def predict(self, X):
        """Make predictions with special handling for different architectures."""
        # Handle sparse matrices
        is_sparse = hasattr(X, "toarray")
        is_large_sparse = is_sparse and X.shape[1] > 10000

        # For large sparse matrices, use batched prediction
        if is_large_sparse:
            from scipy import sparse

            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()

            batch_size = 64 if X.shape[1] > 100000 else 128
            outputs = []
            n_samples = X.shape[0]

            self.model.eval()
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_indices = list(range(i, end_idx))

                with torch.no_grad():
                    batch_X = torch.tensor(
                        X[batch_indices].toarray(), dtype=torch.float32
                    ).to(self.device)
                    batch_pred = self.model(batch_X)
                    outputs.append(batch_pred.cpu().numpy())

                if i % (batch_size * 10) == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

            return np.vstack(outputs)

        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            if is_sparse:
                X = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
            else:
                X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        # Handle different architecture types
        if self.architecture_type == "variational":
            # For variational models, we can return uncertainty estimates
            with torch.no_grad():
                if (
                    hasattr(self.model, "forward")
                    and "return_uncertainty" in self.model.forward.__code__.co_varnames
                ):
                    predictions, uncertainty = self.model(X, return_uncertainty=True)
                    # Store uncertainty for later analysis
                    self.last_uncertainty = uncertainty.cpu().numpy()
                    return predictions.cpu().numpy()
                else:
                    return self.model(X).cpu().numpy()
        else:
            # Standard prediction for other architectures
            # Handle batching for large datasets
            if X.shape[0] > 1000:
                batch_size = 1024
                outputs = []

                for i in range(0, X.shape[0], batch_size):
                    with torch.no_grad():
                        batch_X = X[i : i + batch_size]
                        batch_pred = self.model(batch_X)
                        outputs.append(batch_pred.cpu().numpy())

                return np.vstack(outputs)
            else:
                with torch.no_grad():
                    predictions = self.model(X)
                    return predictions.cpu().numpy()

    def predict_with_uncertainty(self, X):
        """Special method for variational models to return uncertainty."""
        if self.architecture_type != "variational":
            raise ValueError(
                "Uncertainty prediction only available for variational models"
            )

        if not isinstance(X, torch.Tensor):
            if hasattr(X, "toarray"):
                X = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
            else:
                X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions, uncertainty = self.model(X, return_uncertainty=True)
            return predictions.cpu().numpy(), uncertainty.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def train_torch_regressor(
    X_train,
    y_train,
    epochs=300,
    batch_size=256,  # Increased default batch size
    learning_rate=0.0008,
    architecture_type="embedding_regressor",  # Default to original architecture
    device=None,
    hidden_dims=[768, 512, 256, 128],
    patience=30,
    verbose=True,
    **arch_params,  # Accept additional architecture parameters
):
    """
    Train a PyTorch regression model with support for multiple architectures.

    Args:
        X_train: Training features (numpy array or sparse matrix)
        y_train: Training targets (numpy array)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        architecture_type: Type of architecture to use
        device: Device to use ('cuda' or 'cpu')
        hidden_dims: List of hidden layer dimensions
        patience: Patience for early stopping
        verbose: Whether to print progress
        **arch_params: Additional architecture-specific parameters

    Returns:
        Trained model wrapped in TorchRegressorWrapper
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get architecture-specific recommended parameters if using new architectures
    if architecture_type != "embedding_regressor":
        recommended_params = get_architecture_specific_params(architecture_type)

        # Use recommended parameters if defaults weren't overridden
        if learning_rate == 0.0008:  # Default value
            learning_rate = recommended_params["learning_rate"]
        if batch_size == 256:  # Updated default value
            batch_size = recommended_params["batch_size"]
        if epochs == 300:  # Default value
            epochs = recommended_params["epochs"]
        if patience == 30:  # Default value
            patience = recommended_params["patience"]

    # GPU optimizations
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        free_mem = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        free_mem_gb = free_mem / (1024**3)
        print(f"Available CUDA memory: {free_mem_gb:.2f} GB")

        if free_mem_gb < 6 and batch_size > 64:
            print(
                f"Reducing batch size from {batch_size} to 64 due to memory constraints"
            )
            batch_size = 64

    start_time = time.time()
    if verbose:
        print(f"Starting {architecture_type} training on {device}...")
        print(f"Input data: X_train={X_train.shape}, y_train={y_train.shape}")

    # Data preparation
    is_sparse = hasattr(X_train, "toarray")
    if is_sparse:
        print(
            "Detected sparse matrix input. Converting to dense for new architectures."
        )
        X_train = X_train.toarray()

    # Create tensors and data loaders with optimizations
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    full_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Optimized DataLoader settings
    num_workers = min(8, psutil.cpu_count(logical=False))  # Use physical cores
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger validation batches
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Create model based on architecture type
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    if architecture_type == "embedding_regressor":
        # Use original EmbeddingRegressorNet
        dropout_rate = arch_params.get('dropout_rate', 0.3)
        model = EmbeddingRegressorNet(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate
        ).to(device)
    else:
        # Use new architectures
        model = create_architecture(
            architecture_type, input_dim, output_dim, **arch_params
        ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    # Loss function - architecture specific
    if architecture_type == "variational":

        def criterion_fn(outputs, targets):
            mse_loss = nn.MSELoss()(outputs, targets)
            # Add KL divergence for variational layers
            kl_loss = 0.0
            for module in model.modules():
                if hasattr(module, "weight_mu") and hasattr(module, "weight_logvar"):
                    kl = -0.5 * torch.sum(
                        1
                        + module.weight_logvar
                        - module.weight_mu.pow(2)
                        - module.weight_logvar.exp()
                    )
                    kl_loss += kl
            return mse_loss + 1e-6 * kl_loss

    else:
        # Standard combined loss
        mse_criterion = nn.MSELoss()
        l1_criterion = nn.L1Loss()

        def criterion_fn(outputs, targets):
            mse = mse_criterion(outputs, targets)
            l1 = l1_criterion(outputs, targets)
            return 0.8 * mse + 0.2 * l1

    # Optimizer - architecture specific
    if architecture_type == "attention":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # Learning rate scheduler
    if architecture_type == "residual":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=patience // 3,
            min_lr=learning_rate / 100,
        )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "architecture_type": architecture_type,
    }

    progress_bar = (
        tqdm(range(epochs), desc=f"Training {architecture_type}")
        if verbose
        else range(epochs)
    )

    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Optimized GPU transfer
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient
            outputs = model(X_batch)
            if output_dim == 1:
                outputs = outputs.squeeze()

            loss = criterion_fn(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Clear cache periodically
            if device == "cuda" and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        train_loss /= train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                if output_dim == 1:
                    outputs = outputs.squeeze()
                loss = criterion_fn(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Store metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["learning_rate"].append(current_lr)

        # Update progress bar
        if verbose:
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

        # Early stopping
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

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Report training time
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"{architecture_type} training completed in {elapsed_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Create wrapper with architecture type
    wrapped_model = TorchRegressorWrapper(model, device, architecture_type)

    # Add training metadata
    wrapped_model.training_metrics = {
        "history": metrics_history,
        "architecture_type": architecture_type,
        "training_time_seconds": elapsed_time,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "pytorch_model": model,
    }

    return wrapped_model


class BaseRegressorNet(nn.Module):
    """Base class for all regressor architectures with common functionality."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _init_weights(self, init_type="kaiming"):
        """Initialize weights using specified initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="leaky_relu"
                    )
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_type == "he":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class StandardMLP(BaseRegressorNet):
    """Standard Multi-Layer Perceptron (your current architecture, refined)."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[768, 512, 256, 128],
        activation="leaky_relu",
        dropout_rate=0.3,
        use_batch_norm=True,
    ):
        super().__init__(input_dim, output_dim)

        self.activation_fn = self._get_activation(activation)
        self.use_batch_norm = use_batch_norm

        # Input processing
        self.input_bn = nn.BatchNorm1d(input_dim) if use_batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate * 0.7)  # Lower dropout for input

        # Build hidden layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layer_block = nn.ModuleList(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    self.activation_fn,
                    nn.BatchNorm1d(dims[i + 1]) if use_batch_norm else nn.Identity(),
                    nn.Dropout(dropout_rate * (1 - 0.1 * i)),  # Decreasing dropout
                ]
            )
            self.layers.append(layer_block)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self._init_weights()

    def _get_activation(self, activation):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish activation
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(activation, nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_dropout(x)

        for layer_block in self.layers:
            for layer in layer_block:
                x = layer(x)

        return self.output_layer(x)


class ResidualMLP(BaseRegressorNet):
    """MLP with residual connections for better gradient flow."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[768, 512, 512, 256],
        activation="leaky_relu",
        dropout_rate=0.3,
    ):
        super().__init__(input_dim, output_dim)

        self.activation_fn = self._get_activation(activation)

        # Input projection to first hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(
                hidden_dims[i], hidden_dims[i + 1], activation, dropout_rate
            )
            self.residual_blocks.append(block)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self._init_weights()

    def _get_activation(self, activation):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }
        return activations.get(activation, nn.LeakyReLU(0.1))

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.activation_fn(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block with optional dimension change."""

    def __init__(self, in_dim, out_dim, activation="leaky_relu", dropout_rate=0.3):
        super().__init__()

        self.activation_fn = self._get_activation(activation)

        # Main path
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection (if dimensions don't match)
        self.skip_connection = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def _get_activation(self, activation):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }
        return activations.get(activation, nn.LeakyReLU(0.1))

    def forward(self, x):
        residual = self.skip_connection(x)

        # Main path
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)

        # Add residual connection
        out += residual
        out = self.activation_fn(out)
        out = self.dropout2(out)

        return out


class AttentionMLP(BaseRegressorNet):
    """MLP with simplified attention mechanism for feature importance."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[768, 512, 256],
        dropout_rate=0.3,
    ):
        super().__init__(input_dim, output_dim)

        # Feature attention mechanism - learns which features are important
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid(),  # Attention weights between 0 and 1
        )

        # Feature transformation after attention
        self.feature_transform = nn.Linear(input_dim, hidden_dims[0])
        self.transform_norm = nn.BatchNorm1d(hidden_dims[0])

        # Standard MLP layers
        self.mlp_layers = nn.ModuleList()
        dims = hidden_dims
        for i in range(len(dims) - 1):
            self.mlp_layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.Dropout(
                        dropout_rate * (1 - 0.1 * i)
                    ),  # Gradually decrease dropout
                ]
            )

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self._init_weights()

    def forward(self, x):
        # Compute feature attention weights
        attention_weights = self.feature_attention(x)  # [batch_size, input_dim]

        # Apply attention to input features (element-wise multiplication)
        attended_features = x * attention_weights

        # Transform attended features
        x_out = self.feature_transform(attended_features)
        x_out = self.transform_norm(x_out)
        x_out = F.relu(x_out)

        # Standard MLP processing
        for layer in self.mlp_layers:
            x_out = layer(x_out)

        return self.output_layer(x_out)


class EnsembleMLP(BaseRegressorNet):
    """Ensemble of multiple smaller networks."""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts=4,
        expert_hidden_dims=[256, 128],
        dropout_rate=0.3,
    ):
        super().__init__(input_dim, output_dim)

        # Multiple expert networks
        self.experts = nn.ModuleList(
            [
                self._create_expert(input_dim, expert_hidden_dims, dropout_rate)
                for _ in range(num_experts)
            ]
        )

        # Gating network to weight expert outputs
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_experts * 2, num_experts),
            nn.Softmax(dim=1),
        )

        # Final combination layer
        self.output_layer = nn.Linear(num_experts * expert_hidden_dims[-1], output_dim)

        self._init_weights()

    def _create_expert(self, input_dim, hidden_dims, dropout_rate):
        """Create a single expert network."""
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.Dropout(dropout_rate),
                ]
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        # Get gating weights
        gates = self.gating_network(x)  # [batch_size, num_experts]

        # Weighted combination of expert outputs
        weighted_outputs = []
        for i, expert_out in enumerate(expert_outputs):
            weight = gates[:, i : i + 1]  # [batch_size, 1]
            weighted_outputs.append(weight * expert_out)

        # Concatenate weighted outputs
        combined = torch.cat(weighted_outputs, dim=1)

        return self.output_layer(combined)


class VariationalMLP(BaseRegressorNet):
    """MLP with variational (Bayesian) layers for uncertainty estimation."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[768, 512, 256],
        dropout_rate=0.3,
        num_samples=10,
    ):
        super().__init__(input_dim, output_dim)

        self.num_samples = num_samples

        # Variational layers
        self.var_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            var_layer = VariationalLinear(dims[i], dims[i + 1])
            self.var_layers.append(var_layer)

        # Activation and normalization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer (also variational)
        self.output_layer = VariationalLinear(hidden_dims[-1], output_dim)

    def forward(self, x, return_uncertainty=False):
        if return_uncertainty:
            # Sample multiple times for uncertainty estimation
            predictions = []
            for _ in range(self.num_samples):
                pred = self._forward_single(x)
                predictions.append(pred)

            predictions = torch.stack(
                predictions
            )  # [num_samples, batch_size, output_dim]
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)

            return mean_pred, uncertainty
        else:
            return self._forward_single(x)

    def _forward_single(self, x):
        for var_layer in self.var_layers:
            x = var_layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        return self.output_layer(x)


class VariationalLinear(nn.Module):
    """Linear layer with variational weights."""

    def __init__(self, in_features, out_features):
        super().__init__()

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1 - 5
        )

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features) - 5)

    def forward(self, x):
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps

        # Sample bias
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_std * bias_eps

        return F.linear(x, weight, bias)


class AdaptiveMLP(BaseRegressorNet):
    """MLP with adaptive architecture based on input complexity."""

    def __init__(
        self,
        input_dim,
        output_dim,
        base_hidden_dims=[512, 256, 128],
        complexity_threshold=0.5,
        dropout_rate=0.3,
    ):
        super().__init__(input_dim, output_dim)

        self.complexity_threshold = complexity_threshold

        # Complexity estimation network
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Simple path (for low complexity inputs)
        self.simple_path = StandardMLP(
            input_dim,
            output_dim,
            hidden_dims=base_hidden_dims[:2],  # Fewer layers
            dropout_rate=dropout_rate,
        )

        # Complex path (for high complexity inputs)
        self.complex_path = StandardMLP(
            input_dim,
            output_dim,
            hidden_dims=base_hidden_dims + [64],  # More layers
            dropout_rate=dropout_rate,
        )

        self._init_weights()

    def forward(self, x):
        # Estimate input complexity
        complexity = self.complexity_estimator(x)  # [batch_size, 1]

        # Route through appropriate path
        simple_out = self.simple_path(x)
        complex_out = self.complex_path(x)

        # Weighted combination based on complexity
        weight = (complexity > self.complexity_threshold).float()
        output = weight * complex_out + (1 - weight) * simple_out

        return output


def create_architecture(arch_type, input_dim, output_dim, **kwargs):
    """Factory function to create different architectures."""

    architectures = {
        "standard": StandardMLP,
        "residual": ResidualMLP,
        "attention": AttentionMLP,
        "ensemble": EnsembleMLP,
        "variational": VariationalMLP,
        "adaptive": AdaptiveMLP,
    }

    if arch_type not in architectures:
        raise ValueError(
            f"Unknown architecture: {arch_type}. Available: {list(architectures.keys())}"
        )

    return architectures[arch_type](input_dim, output_dim, **kwargs)


# Architecture-specific training modifications
def get_architecture_specific_params(arch_type):
    """Get recommended training parameters for each architecture."""

    params = {
        "standard": {
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 300,
            "patience": 30,
            "weight_decay": 1e-4,
        },
        "residual": {
            "learning_rate": 0.0008,
            "batch_size": 256,
            "epochs": 400,
            "patience": 40,
            "weight_decay": 1e-4,
        },
        "attention": {
            "learning_rate": 0.0008,
            "batch_size": 256,
            "epochs": 350,
            "patience": 35,
            "weight_decay": 1e-5,
        },
        "ensemble": {
            "learning_rate": 0.001,
            "batch_size": 192,
            "epochs": 300,
            "patience": 30,
            "weight_decay": 1e-4,
        },
        "variational": {
            "learning_rate": 0.0008,
            "batch_size": 256,
            "epochs": 500,
            "patience": 50,
            "weight_decay": 1e-5,
        },
        "adaptive": {
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 350,
            "patience": 35,
            "weight_decay": 1e-4,
        },
    }

    return params.get(arch_type, params["standard"])


def compare_architectures(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    architectures=None,
    device=None,
    verbose=True,
):
    """Compare multiple architectures on the same data."""
    if architectures is None:
        architectures = ["standard", "residual", "attention", "ensemble"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for arch_type in architectures:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training {arch_type} architecture")
            print(f"{'='*50}")

        try:
            model = train_torch_regressor(
                X_train,
                y_train,
                architecture_type=arch_type,
                device=device,
                verbose=verbose,
            )

            from evaluate import evaluate_model

            val_results = evaluate_model(
                model, X_val, y_val, split_name="val", verbose=False
            )
            test_results = evaluate_model(
                model, X_test, y_test, split_name="test", verbose=False
            )

            results[arch_type] = {
                "model": model,
                "val_results": val_results,
                "test_results": test_results,
                "training_metrics": model.training_metrics,
            }

            if verbose:
                print(f"Test MSE: {test_results.metrics.get('test_mse', 'N/A')}")
                print(f"Parameters: {model.training_metrics['total_parameters']:,}")

        except Exception as e:
            print(f"Error training {arch_type}: {e}")
            results[arch_type] = {"error": str(e)}

        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# Integration with existing hyperparameter optimization
def optimize_architecture_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    architecture_type="standard",
    n_trials=30,
    metric="val_mse",
):
    """
    Optimize hyperparameters for a specific architecture using Optuna.

    This extends the existing hyperparameter optimization to work with different architectures.
    """
    import optuna

    def objective(trial):
        # Architecture-specific parameter suggestions
        if architecture_type == "standard":
            params = {
                "hidden_dims": [
                    trial.suggest_categorical("layer1_size", [256, 512, 768, 1024]),
                    trial.suggest_categorical("layer2_size", [128, 256, 512]),
                    trial.suggest_categorical("layer3_size", [64, 128, 256]),
                ],
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "leaky_relu", "gelu", "swish"]
                ),
            }
        elif architecture_type == "attention":
            params = {
                "hidden_dims": [
                    trial.suggest_categorical("layer1_size", [512, 768, 1024]),
                    trial.suggest_categorical("layer2_size", [256, 512]),
                    trial.suggest_categorical("layer3_size", [128, 256]),
                ],
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
            }
        elif architecture_type == "ensemble":
            params = {
                "num_experts": trial.suggest_int("num_experts", 2, 8),
                "expert_hidden_dims": [
                    trial.suggest_categorical("expert_layer1", [128, 256, 512]),
                    trial.suggest_categorical("expert_layer2", [64, 128, 256]),
                ],
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            }
        else:
            # Default parameters for other architectures
            params = {
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            }

        # Training parameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        try:
            # Train model
            model = train_torch_regressor(
                X_train,
                y_train,
                architecture_type=architecture_type,
                arch_params=params,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=100,  # Reduced for optimization
                verbose=False,
            )

            # Evaluate
            from evaluate import evaluate_model

            val_results = evaluate_model(
                model, X_val, y_val, split_name="val", verbose=False
            )

            metric_value = val_results.metrics.get(metric, float("inf"))

            # For R² metrics, negate since Optuna minimizes
            if "r2" in metric:
                metric_value = -metric_value

            return metric_value

        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study
