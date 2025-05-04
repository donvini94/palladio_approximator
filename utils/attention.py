import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import joblib
from pathlib import Path
import re
from tqdm import tqdm

# Import visualization utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize import create_dirs


def normalize_attention(attention_matrix):
    """
    Normalize attention weights to make them easier to visualize.
    
    Args:
        attention_matrix: Attention matrix to normalize
        
    Returns:
        Normalized attention matrix
    """
    # Avoid division by zero
    if attention_matrix.max() == attention_matrix.min():
        return np.ones_like(attention_matrix)
    
    normalized = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    return normalized


def visualize_attention_heads(attention_data, tokens, layer_idx, output_dir="figures/attention", model_name="bert", filename_prefix=""):
    """
    Visualize attention patterns for each head in a specific layer.
    
    Args:
        attention_data: Tensor of attention weights [num_heads, seq_len, seq_len]
        tokens: List of tokens corresponding to the sequence
        layer_idx: Index of the layer to visualize
        output_dir: Directory to save attention visualizations
        model_name: Name of the model for figure titles
        filename_prefix: Prefix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_heads = attention_data.shape[0]
    
    # Create subplots for all heads
    fig, axes = plt.subplots(
        nrows=int(np.ceil(num_heads / 2)), 
        ncols=2, 
        figsize=(16, 4 * int(np.ceil(num_heads / 2)))
    )
    
    # Ensure axes is always 2D for consistent indexing
    if num_heads == 1:
        axes = np.array([[axes]])
    elif num_heads == 2:
        axes = np.array([axes])
    
    # Plot each attention head
    for head_idx in range(num_heads):
        i, j = divmod(head_idx, 2)
        ax = axes[i, j]
        
        attention = attention_data[head_idx].cpu().numpy()
        
        # Only show a manageable number of tokens (first 30)
        max_tokens = min(30, len(tokens))
        window_tokens = tokens[:max_tokens]
        window_attention = attention[:max_tokens, :max_tokens]
        
        # Plot heatmap
        sns.heatmap(
            window_attention,
            ax=ax,
            cmap="viridis",
            xticklabels=window_tokens,
            yticklabels=window_tokens,
            annot=False
        )
        
        ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    
    # Hide empty subplots if any
    for i in range(num_heads, axes.shape[0] * axes.shape[1]):
        row, col = divmod(i, 2)
        axes[row, col].axis('off')
    
    plt.suptitle(f"{model_name} Attention Heads (Layer {layer_idx})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    filename = f"{filename_prefix}layer_{layer_idx}_heads.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def visualize_attention_patterns(file_path, model_name=None, output_dir="figures/attention", max_length=512, layer_indices=None):
    """
    Visualize attention patterns for a given file (code document) and model.
    
    Args:
        file_path: Path to the file to analyze
        model_name: HuggingFace model name (if None, uses CodeBERT)
        output_dir: Directory to save attention visualizations
        max_length: Maximum sequence length
        layer_indices: List of layer indices to visualize (if None, uses last 2 layers)
        
    Returns:
        Dictionary with attention statistics
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        model_name = "microsoft/codebert-base"
    
    # Determine model type (BERT-like or Causal LM like Llama)
    is_causal_lm = any(name in model_name.lower() for name in ["llama", "gpt", "falcon", "mistral"])
    
    try:
        # Load file content
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Set up tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Special handling for causal LMs (Llama, etc.)
        if is_causal_lm:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                output_attentions=True
            ).to(device)
        else:
            # BERT-like models
            model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
        
        model.eval()
        
        # Tokenize the input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract attention matrices
        if is_causal_lm:
            # For causal LMs, attention is directly in outputs
            attentions = outputs.attentions
        else:
            # For BERT-like models
            attentions = outputs.attentions
        
        # Get tokens
        token_ids = inputs.input_ids[0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Determine which layers to visualize
        if layer_indices is None:
            # Default to last two layers
            total_layers = len(attentions)
            layer_indices = [total_layers - 2, total_layers - 1]
        
        # File name prefix based on input file
        file_name = Path(file_path).stem
        model_short_name = model_name.split("/")[-1]
        prefix = f"{file_name}_{model_short_name}_"
        
        # Visualize attention for specified layers
        for layer_idx in layer_indices:
            if layer_idx < len(attentions):
                layer_attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
                
                # Visualize all attention heads for this layer
                visualize_attention_heads(
                    layer_attention, 
                    tokens, 
                    layer_idx,
                    output_dir=output_dir,
                    model_name=model_short_name,
                    filename_prefix=prefix
                )
                
                # Also create combined attention visualization for the layer
                avg_attention = layer_attention.mean(dim=0).cpu().numpy()
                
                # Only show a manageable number of tokens
                max_tokens = min(40, len(tokens))
                window_tokens = tokens[:max_tokens]
                window_attention = avg_attention[:max_tokens, :max_tokens]
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    window_attention,
                    cmap="viridis",
                    xticklabels=window_tokens,
                    yticklabels=window_tokens,
                    annot=False
                )
                
                plt.title(f"{model_short_name} Layer {layer_idx} Average Attention")
                plt.xticks(rotation=90, ha="right", fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                
                # Save the combined attention figure
                plt.savefig(os.path.join(output_dir, f"{prefix}layer_{layer_idx}_avg.png"), dpi=300, bbox_inches="tight")
                plt.close()
            
        # Find important tokens by attention
        # Use the last layer for this analysis
        last_layer_idx = len(attentions) - 1
        last_layer_attention = attentions[last_layer_idx][0].mean(dim=0).cpu().numpy()  # Average across heads
        
        # Get attention importance for each token
        token_importance = last_layer_attention.mean(axis=0)  # Average attention received by each token
        
        # Create a dataframe for the tokens and their importance
        token_df = pd.DataFrame({
            "Token": tokens,
            "Importance": token_importance,
        })
        
        # Sort by importance
        token_df = token_df.sort_values("Importance", ascending=False)
        
        # Plot top 30 important tokens
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=token_df.head(30),
            x="Importance",
            y="Token",
            palette="viridis"
        )
        
        plt.title(f"Top 30 Important Tokens by Attention ({model_short_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}token_importance.png"), dpi=300)
        plt.close()
        
        # Save token importance data
        token_df.to_csv(os.path.join(output_dir, f"{prefix}token_importance.csv"), index=False)
        
        return {
            "model_name": model_name,
            "file_name": file_path,
            "token_count": len(tokens),
            "layer_count": len(attentions),
            "top_tokens": token_df.head(10)["Token"].tolist(),
        }
    
    except Exception as e:
        print(f"Error visualizing attention for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def analyze_model_representations(model_path, embedding_path, output_dir="figures/embeddings", n_components=2):
    """
    Analyze the learned representations of a trained model by comparing 
    input embeddings and model's internal representations.
    
    Args:
        model_path: Path to the trained model
        embedding_path: Path to the embedding checkpoint
        output_dir: Directory to save visualizations
        n_components: Number of components for dimensionality reduction
        
    Returns:
        Dictionary with analysis results
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model and embeddings
        model = joblib.load(model_path)
        embeddings = joblib.load(embedding_path)
        
        X_test = embeddings["X_test"]
        y_test = embeddings["y_test"]
        
        # Extract model type and embedding type from filenames
        model_file = os.path.basename(model_path)
        embedding_file = os.path.basename(embedding_path)
        
        match_model = re.match(r'(.+?)_(.+?)_model\.pkl', model_file)
        match_embedding = re.match(r'(.+?)_(.+?)_features_checkpoint\.pkl', embedding_file)
        
        if match_model and match_embedding:
            model_type = match_model.group(1)
            embedding_type = match_embedding.group(1)
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Ensure predictions have the right shape
            if len(predictions.shape) == 1 and len(y_test.shape) > 1:
                predictions = predictions.reshape(-1, 1)
            
            # Calculate prediction errors
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                # Multi-output case - use first dimension
                errors = np.abs(predictions[:, 0] - y_test[:, 0])
            else:
                # Single output case
                errors = np.abs(predictions.flatten() - y_test.flatten())
            
            # Apply dimensionality reduction to visualize embeddings
            if hasattr(X_test, 'toarray'):
                X_dense = X_test.toarray()
            else:
                X_dense = X_test
            
            # PCA for linear dimensionality reduction
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_dense)
            
            # t-SNE for non-linear dimensionality reduction
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_dense)
            
            # Visualize PCA embeddings colored by prediction error
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=errors, cmap="viridis", alpha=0.7)
            plt.colorbar(scatter, label="Prediction Error")
            plt.title(f"PCA of {embedding_type.upper()} Embeddings (Colored by {model_type.upper()} Model Error)")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{embedding_type}_{model_type}_pca_error.png"), dpi=300)
            plt.close()
            
            # Visualize t-SNE embeddings colored by prediction error
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=errors, cmap="viridis", alpha=0.7)
            plt.colorbar(scatter, label="Prediction Error")
            plt.title(f"t-SNE of {embedding_type.upper()} Embeddings (Colored by {model_type.upper()} Model Error)")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{embedding_type}_{model_type}_tsne_error.png"), dpi=300)
            plt.close()
            
            # Also color by actual values
            if len(y_test.shape) > 1:
                y_values = y_test[:, 0]  # Use first output dimension
            else:
                y_values = y_test
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_values, cmap="plasma", alpha=0.7)
            plt.colorbar(scatter, label="Actual Value")
            plt.title(f"PCA of {embedding_type.upper()} Embeddings (Colored by Actual Value)")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{embedding_type}_{model_type}_pca_actual.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_values, cmap="plasma", alpha=0.7)
            plt.colorbar(scatter, label="Actual Value")
            plt.title(f"t-SNE of {embedding_type.upper()} Embeddings (Colored by Actual Value)")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{embedding_type}_{model_type}_tsne_actual.png"), dpi=300)
            plt.close()
            
            # For random forest, visualize feature importance
            if model_type == "rf" and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                
                # Plot top N important features
                plt.figure(figsize=(12, 6))
                
                # For high-dimensional features, show only top 50
                if len(importances) > 50:
                    top_indices = np.argsort(importances)[-50:]
                    top_importances = importances[top_indices]
                    
                    plt.barh(range(len(top_indices)), top_importances, color="teal")
                    plt.yticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices])
                    plt.title(f"Top 50 Important Features ({embedding_type.upper()} with {model_type.upper()})")
                else:
                    plt.barh(range(len(importances)), importances, color="teal")
                    plt.yticks(range(len(importances)), [f"Feature {i}" for i in range(len(importances))])
                    plt.title(f"Feature Importances ({embedding_type.upper()} with {model_type.upper()})")
                
                plt.xlabel("Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{embedding_type}_{model_type}_feature_importance.png"), dpi=300)
                plt.close()
                
            return {
                "model_type": model_type,
                "embedding_type": embedding_type,
                "pca_explained_variance": pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else None,
                "data_shape": X_test.shape,
                "error_stats": {
                    "min": float(errors.min()),
                    "max": float(errors.max()),
                    "mean": float(errors.mean()),
                    "median": float(np.median(errors))
                }
            }
    
    except Exception as e:
        print(f"Error analyzing model representations: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def compare_model_predictions(model_paths, embedding_paths, output_dir="figures/model_comparison"):
    """
    Compare predictions from multiple models trained on the same test data.
    
    Args:
        model_paths: List of paths to trained models
        embedding_paths: List of paths to embedding checkpoints
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with comparison results
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and make predictions
    models = []
    predictions = []
    model_names = []
    embedding_types = []
    
    # Use the first embedding path for reference test data
    reference_embeddings = joblib.load(embedding_paths[0])
    X_test = reference_embeddings["X_test"]
    y_test = reference_embeddings["y_test"]
    
    # Process each model
    for i, (model_path, embedding_path) in enumerate(zip(model_paths, embedding_paths)):
        try:
            model = joblib.load(model_path)
            
            # Extract model type and embedding type from filenames
            model_file = os.path.basename(model_path)
            embedding_file = os.path.basename(embedding_path)
            
            match_model = re.match(r'(.+?)_(.+?)_model\.pkl', model_file)
            match_embedding = re.match(r'(.+?)_(.+?)_features_checkpoint\.pkl', embedding_file)
            
            if match_model and match_embedding:
                model_type = match_model.group(1)
                embedding_type = match_embedding.group(1)
                
                # Load embeddings if not the reference
                if i > 0:
                    embeddings = joblib.load(embedding_path)
                    if embeddings["X_test"].shape != X_test.shape:
                        print(f"Skipping {model_type} + {embedding_type} due to shape mismatch")
                        continue
                
                # Generate predictions
                pred = model.predict(X_test)
                
                # Ensure predictions have the right shape
                if len(pred.shape) == 1 and len(y_test.shape) > 1:
                    pred = pred.reshape(-1, 1)
                
                models.append(model)
                predictions.append(pred)
                model_names.append(f"{embedding_type} + {model_type}")
                embedding_types.append(embedding_type)
                
        except Exception as e:
            print(f"Error processing {model_path}: {e}")
    
    if not models:
        return {"error": "No valid models found"}
    
    # Compare predictions
    # Use first dimension for multi-output
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_actual = y_test[:, 0]
        preds = [p[:, 0] for p in predictions]
    else:
        y_actual = y_test.flatten()
        preds = [p.flatten() for p in predictions]
    
    # Create a figure for all model predictions vs actual
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.plot(range(len(y_actual)), y_actual, 'k-', label='Actual', alpha=0.7)
    
    # Plot predictions from each model
    for i, (pred, name) in enumerate(zip(preds, model_names)):
        plt.plot(range(len(pred)), pred, '-', label=name, alpha=0.7)
    
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_model_predictions.png"), dpi=300)
    plt.close()
    
    # Create correlation heatmap between model predictions
    pred_df = pd.DataFrame({name: p for name, p in zip(model_names, preds)})
    pred_df['Actual'] = y_actual
    
    correlation = pred_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="viridis", vmin=-1, vmax=1, linewidths=.5)
    plt.title("Correlation Between Model Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_correlation.png"), dpi=300)
    plt.close()
    
    # Calculate error metrics for each model
    metrics = []
    for i, (pred, name) in enumerate(zip(preds, model_names)):
        mse = ((pred - y_actual) ** 2).mean()
        mae = np.abs(pred - y_actual).mean()
        
        metrics.append({
            'model': name,
            'mse': mse,
            'mae': mae
        })
    
    # Create a dataframe for metrics
    metrics_df = pd.DataFrame(metrics)
    
    # Plot MSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='mse', data=metrics_df, palette='viridis')
    plt.title('MSE Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_comparison.png"), dpi=300)
    plt.close()
    
    # Plot MAE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='mae', data=metrics_df, palette='viridis')
    plt.title('MAE Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_comparison.png"), dpi=300)
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, "model_comparison_metrics.csv"), index=False)
    
    return {
        "models_compared": model_names,
        "metrics": metrics_df.to_dict(orient="records")
    }


def extract_token_attention_patterns(file_path, model_name=None, keywords=None, output_dir="figures/token_patterns"):
    """
    Extract and visualize attention patterns for specific code tokens/keywords.
    
    Args:
        file_path: Path to the file to analyze
        model_name: HuggingFace model name
        keywords: List of keywords to analyze (if None, auto-detects)
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with token pattern statistics
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        model_name = "microsoft/codebert-base"
    
    # Default keywords for DSL files if none provided
    if keywords is None:
        keywords = [
            "component", "interface", "service", "operation", "provides", "requires", 
            "repository", "system", "allocation", "resource", "container", "parameter",
            "from", "to", "RETURN"
        ]
    
    try:
        # Load file content
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set up tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        is_causal_lm = any(name in model_name.lower() for name in ["llama", "gpt", "falcon", "mistral"])
        
        if is_causal_lm:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                output_attentions=True
            ).to(device)
        else:
            model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
        
        model.eval()
        
        # Tokenize the input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract attention matrices
        if is_causal_lm:
            attentions = outputs.attentions
        else:
            attentions = outputs.attentions
        
        # Get tokens
        token_ids = inputs.input_ids[0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Find indices of keywords in token list
        keyword_indices = {}
        for keyword in keywords:
            # Find tokens that match the keyword (exact match or subword)
            indices = [i for i, token in enumerate(tokens) if keyword.lower() in token.lower()]
            if indices:
                keyword_indices[keyword] = indices
        
        # Create a visualization for each keyword
        results = {}
        
        for keyword, indices in keyword_indices.items():
            if not indices:
                continue
            
            # Use the first occurrence of the keyword
            token_idx = indices[0]
            file_name = Path(file_path).stem
            
            # Analyze attention patterns across layers
            layer_attentions = []
            
            for layer_idx, layer_attention in enumerate(attentions):
                # Average across heads
                avg_attention = layer_attention[0].mean(dim=0).cpu().numpy()
                
                # Get attention FROM this token TO other tokens
                attention_from = avg_attention[token_idx, :]
                
                # Get attention TO this token FROM other tokens
                attention_to = avg_attention[:, token_idx]
                
                layer_attentions.append({
                    "layer": layer_idx,
                    "attention_from": attention_from,
                    "attention_to": attention_to
                })
            
            # Create a DataFrame for the tokens and their attention relationship
            token_attention_df = pd.DataFrame({
                "Token": tokens,
            })
            
            # Add attention for each layer
            for layer_data in layer_attentions:
                layer_idx = layer_data["layer"]
                token_attention_df[f"From_{layer_idx}"] = layer_data["attention_from"]
                token_attention_df[f"To_{layer_idx}"] = layer_data["attention_to"]
            
            # Add token positions
            token_attention_df["Position"] = range(len(tokens))
            
            # Save token attention data
            model_short_name = model_name.split("/")[-1]
            token_attention_df.to_csv(
                os.path.join(output_dir, f"{file_name}_{model_short_name}_{keyword}_attention.csv"), 
                index=False
            )
            
            # Create visualization for this keyword
            # Use the last layer for most informative attention
            last_layer_idx = len(attentions) - 1
            last_layer_data = layer_attentions[last_layer_idx]
            
            # Sort tokens by attention FROM the keyword
            from_df = token_attention_df.sort_values(f"From_{last_layer_idx}", ascending=False)
            
            # Plot top 20 tokens receiving attention FROM the keyword
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=from_df.head(20),
                x=f"From_{last_layer_idx}",
                y="Token",
                palette="viridis"
            )
            
            plt.title(f"Top 20 Tokens Receiving Attention FROM '{keyword}'")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{file_name}_{model_short_name}_{keyword}_from.png"), 
                dpi=300
            )
            plt.close()
            
            # Sort tokens by attention TO the keyword
            to_df = token_attention_df.sort_values(f"To_{last_layer_idx}", ascending=False)
            
            # Plot top 20 tokens sending attention TO the keyword
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=to_df.head(20),
                x=f"To_{last_layer_idx}",
                y="Token",
                palette="viridis"
            )
            
            plt.title(f"Top 20 Tokens Sending Attention TO '{keyword}'")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{file_name}_{model_short_name}_{keyword}_to.png"), 
                dpi=300
            )
            plt.close()
            
            # Visualize attention across layers
            # Plot attention FROM keyword across layers
            plt.figure(figsize=(12, 8))
            
            # Create data for this visualization
            layer_data = []
            for layer_idx in range(len(attentions)):
                layer_data.append({
                    "Layer": layer_idx,
                    "Attention": token_attention_df[f"From_{layer_idx}"].mean()
                })
            
            # Create DataFrame for layer progression
            layer_df = pd.DataFrame(layer_data)
            
            # Plot
            sns.lineplot(data=layer_df, x="Layer", y="Attention", marker="o")
            plt.title(f"Average Attention FROM '{keyword}' Across Layers")
            plt.xticks(range(len(attentions)))
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{file_name}_{model_short_name}_{keyword}_layer_progression.png"), 
                dpi=300
            )
            plt.close()
            
            # Store results
            results[keyword] = {
                "token_idx": token_idx,
                "occurrence_count": len(indices),
                "top_attention_from": from_df.head(5)["Token"].tolist(),
                "top_attention_to": to_df.head(5)["Token"].tolist(),
            }
        
        return {
            "model_name": model_name,
            "file_name": file_path,
            "keywords_found": list(keyword_indices.keys()),
            "keywords_analyzed": results
        }
    
    except Exception as e:
        print(f"Error extracting token patterns: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def batch_process_files(directory, pattern="*.tpcm", model_name=None, output_dir="figures/batch_analysis"):
    """
    Process multiple files in batch to generate attention visualizations.
    
    Args:
        directory: Directory containing files to process
        pattern: File pattern to match
        model_name: Model to use for attention analysis
        output_dir: Directory to save outputs
        
    Returns:
        Summary of processed files
    """
    create_dirs()
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files found matching pattern {pattern} in {directory}")
        return {"error": "No files found"}
    
    # Process each file
    results = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Create a subdirectory for each file
            file_name = Path(file_path).stem
            file_output_dir = os.path.join(output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze attention patterns
            result = visualize_attention_patterns(
                file_path, 
                model_name=model_name,
                output_dir=file_output_dir
            )
            
            # Also extract token patterns
            token_result = extract_token_attention_patterns(
                file_path,
                model_name=model_name,
                output_dir=file_output_dir
            )
            
            results.append({
                "file": file_path,
                "attention_result": result,
                "token_result": token_result
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append({
                "file": file_path,
                "error": str(e)
            })
    
    # Create summary report
    summary = {
        "total_files": len(files),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "model_used": model_name
    }
    
    # Save summary
    with open(os.path.join(output_dir, "batch_summary.txt"), "w") as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"----------------------\n")
        f.write(f"Total files: {summary['total_files']}\n")
        f.write(f"Successfully processed: {summary['successful']}\n")
        f.write(f"Failed: {summary['failed']}\n")
        f.write(f"Model used: {summary['model_used']}\n")
    
    return summary


if __name__ == "__main__":
    # Example usage
    create_dirs()
    
    # Process a single file
    print("Visualizing attention patterns for a sample file...")
    
    # Find a sample file
    sample_files = glob.glob("data/dsl_models/*.tpcm")
    
    if sample_files:
        sample_file = sample_files[0]
        print(f"Using sample file: {sample_file}")
        
        # Visualize patterns
        result = visualize_attention_patterns(
            sample_file,
            model_name="microsoft/codebert-base"
        )
        
        print(f"Attention visualization complete. Check the 'figures/attention' directory.")
    else:
        print("No sample files found. Please provide a file path to analyze.")