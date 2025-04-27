import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

with open("data/dsl_models/generated__AAAJW.tpcm") as f:
    dsl_text = f.read()

models_to_compare = {
    "bert-base-uncased": "BERT",
    "microsoft/codebert-base": "CodeBERT",
    "microsoft/graphcodebert-base": "GraphCodeBERT",
    "allenai/longformer-base-4096": "Longformer",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def embed_text(model_name, text, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

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

    return cls_embedding.squeeze()


# Create embeddings
embeddings = []
labels = []

for model_name, label in models_to_compare.items():
    max_len = 4096 if "longformer" in model_name else 512
    print(f"Embedding using {label} (max length {max_len})...")
    emb = embed_text(model_name, dsl_text, max_len)
    embeddings.append(emb)
    labels.append(label)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Plot as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    similarity_matrix,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    cmap="viridis",
    fmt=".2f",
)
plt.title("Cosine Similarity between Embeddings")
plt.tight_layout()
plt.savefig("embedding_similarity.png")


# Create embeddings
embeddings = {}
for model_name, label in models_to_compare.items():
    max_len = 4096 if "longformer" in model_name else 512
    print(f"Embedding using {label} (max length {max_len})...")
    emb = embed_text(model_name, dsl_text, max_len)
    embeddings[label] = emb

# Plot histograms of embedding values
plt.figure(figsize=(12, 6))
for label, emb in embeddings.items():
    plt.hist(emb, bins=100, alpha=0.5, label=label, density=True)

plt.legend()
plt.title("Distribution of Embedding Values per Model")
plt.xlabel("Embedding Value")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("embedding_value_distributions.png")
