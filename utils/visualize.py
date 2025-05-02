import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# SETTINGS
data_dir = "data/dsl_models/"
models_to_compare = {
    "bert-base-uncased": "BERT",
    "microsoft/codebert-base": "CodeBERT",
}

max_samples = 5  # DSL files to load
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dsl_files(data_dir, limit=10):
    """
    Load up to 'limit' DSL model files from a directory.
    """
    dsl_files = glob.glob(os.path.join(data_dir, "*.tpcm"))
    dsl_files = dsl_files[:limit]
    texts = []
    for file_path in dsl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def embed_text_and_attentions(model_name, texts, max_length=512):
    """
    Embed list of DSL texts using a given model.
    Also optionally returns attentions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    all_embeddings = []
    all_attentions = []

    for text in texts:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding.squeeze())
            all_attentions.append(outputs.attentions)  # list of layer attentions

    return np.vstack(all_embeddings), all_attentions, tokenizer


def compare_embeddings(data_dir="data/dsl_models/", output_dir="figures/", limit=5):
    """
    Compare embeddings from different models and generate visualizations.
    
    Args:
        data_dir: Directory containing .tpcm files
        output_dir: Directory to save visualization outputs
        limit: Maximum number of DSL files to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load DSL texts
    dsl_texts = load_dsl_files(data_dir, limit=limit)
    print(f"Loaded {len(dsl_texts)} DSL model files.")
    
    if not dsl_texts:
        print("No DSL files found. Please check the data directory.")
        return
    
    # 2. Embed with all models
    all_embeddings = {}
    all_attentions = {}
    
    for model_name, label in models_to_compare.items():
        print(f"Embedding with {label}...")
        embeddings, attentions, tokenizer = embed_text_and_attentions(
            model_name, dsl_texts
        )
        all_embeddings[label] = embeddings
        all_attentions[label] = attentions
        del embeddings, attentions, tokenizer
        torch.cuda.empty_cache()
    
    # Build cosine similarity matrix across models for the first DSL file
    first_embeddings = [embeddings[0] for embeddings in all_embeddings.values()]
    labels = list(models_to_compare.values())
    
    similarity_matrix = cosine_similarity(first_embeddings)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap="viridis",
        fmt=".2f",
    )
    plt.title("Cosine Similarity between Models (First DSL File)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_similarity.png"))
    print(f"Saved embedding similarity heatmap to {output_dir}/embedding_similarity.png")
    
    # Plot histograms for first DSL file
    plt.figure(figsize=(12, 6))
    for label, embeddings in all_embeddings.items():
        plt.hist(embeddings[0], bins=100, alpha=0.5, label=label, density=True)
    
    plt.legend()
    plt.title("Distribution of Embedding Values per Model (First DSL)")
    plt.xlabel("Embedding Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_value_distributions.png"))
    print(f"Saved embedding distributions to {output_dir}/embedding_value_distributions.png")
    
    # Visualize attention heatmap for CodeBERT on first DSL file
    model_to_plot = "CodeBERT"  # Using CodeBERT for attention visualization
    
    if model_to_plot in all_embeddings:
        attentions = all_attentions[model_to_plot][0]  # First DSL file
        layer_to_plot = 0  # Layer 0
        head_to_plot = 0  # Head 0
        
        # Get attention for first head of first layer
        attention_matrix = (
            attentions[layer_to_plot][0, head_to_plot].cpu().numpy()
        )  # shape (seq_len, seq_len)
        
        # Plot attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, cmap="viridis")
        plt.title(
            f"Attention Heatmap: {model_to_plot} - Layer {layer_to_plot}, Head {head_to_plot}"
        )
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention_heatmap_{model_to_plot}.png"))
        print(f"Saved attention heatmap to {output_dir}/attention_heatmap_{model_to_plot}.png")
    
    return {
        "models": list(models_to_compare.values()),
        "n_samples": len(dsl_texts),
        "similarity_matrix": similarity_matrix,
    }


if __name__ == "__main__":
    # Example usage
    compare_embeddings(limit=2)