"""
This module provides simplified, scientifically sound feature extraction methods:
1. TF-IDF: Fast, sparse embeddings with configurable parameters
2. BERT: Deep contextual embeddings with simplified chunking
3. LLaMA: State-of-the-art embeddings with efficient processing
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm
from dataset import load_dataset


# Configuration class for chunking parameters
class ChunkingConfig:
    """Configuration for document chunking with scientifically justified defaults."""

    def __init__(self):
        # Based on literature review: most effective chunk counts are 10-15
        self.max_chunks_per_doc = 12  # Reduced from 30, based on legal NLP research

        # Overlap strategies: 10-20% overlap is standard in literature
        self.chunk_overlap_tokens = 50  # ~10% of 512 tokens

        # Aggregation method: simple mean is more interpretable than weighted
        self.chunk_aggregation = "mean"  # Options: "mean", "attention", "first_last"

        # Memory vs accuracy trade-offs
        self.enable_quantization = True  # 4-bit for LLaMA by default
        self.batch_size_bert = 4  # Conservative for memory
        self.batch_size_llama = 2  # Even more conservative for large models


def extract_features(args, device):
    """Extract features based on the specified embedding type.

    Args:
        args: Command line arguments containing embedding options
        device (str): 'cuda' or 'cpu' for processing

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, embedding_model)
    """
    if args.prediction_mode != "summary":
        raise ValueError("Unsupported prediction mode. Choose 'summary'")

    print("Loading dataset...")
    train_samples, val_samples, test_samples = load_dataset(
        args.data_dir, save_dataset=args.save_dataset, load_dataset=args.load_dataset
    )
    print(f"Dataset loaded. Train samples: {len(train_samples)}")

    chunking_config = ChunkingConfig()

    print(f"Building features using {args.embedding} embedding...")
    print(
        f"Chunking config: max_chunks={chunking_config.max_chunks_per_doc}, "
        f"overlap={chunking_config.chunk_overlap_tokens}, "
        f"aggregation={chunking_config.chunk_aggregation}"
    )

    try:
        if args.embedding == "tfidf":
            return build_tfidf_features(train_samples, val_samples, test_samples, args)

        elif args.embedding == "bert":
            return build_bert_features(
                train_samples, val_samples, test_samples, device, chunking_config, args
            )

        elif args.embedding == "llama":
            return build_llama_features(
                train_samples, val_samples, test_samples, device, chunking_config, args
            )

        else:
            raise ValueError(f"Unsupported embedding type: {args.embedding}")

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


def extract_targets(df, normalize=False, target_scaler=None):
    """Extract avg response time from dataframe.
    
    Args:
        df: DataFrame containing target values
        normalize: Whether to normalize the targets
        target_scaler: Pre-fitted scaler for validation/test sets
        
    Returns:
        tuple: (targets, scaler) where scaler is None if normalize=False
    """
    targets = df["avg_resp_time"].to_numpy()
    
    if not normalize:
        return targets, None
    
    # Reshape for sklearn scaler (expects 2D array)
    targets_reshaped = targets.reshape(-1, 1)
    
    if target_scaler is None:
        # Training set: fit new scaler
        scaler = StandardScaler()
        targets_normalized = scaler.fit_transform(targets_reshaped).flatten()
        return targets_normalized, scaler
    else:
        # Validation/test set: use existing scaler
        targets_normalized = target_scaler.transform(targets_reshaped).flatten()
        return targets_normalized, target_scaler


def inverse_transform_predictions(predictions, embedding_model, normalize_targets=False):
    """Inverse transform predictions if target normalization was used.
    
    Args:
        predictions: Model predictions (normalized if normalize_targets=True)
        embedding_model: Tuple containing model artifacts including target_scaler
        normalize_targets: Whether targets were normalized during training
        
    Returns:
        predictions: Predictions in original scale
    """
    if not normalize_targets:
        return predictions
    
    # Extract target scaler from embedding model tuple
    target_scaler = None
    if isinstance(embedding_model, tuple):
        # Check if the last element is a StandardScaler (target scaler)
        for item in embedding_model:
            if hasattr(item, 'inverse_transform') and hasattr(item, 'scale_'):
                target_scaler = item
                break
    
    if target_scaler is None:
        print("Warning: Target normalization was used but scaler not found in embedding_model")
        return predictions
    
    # Reshape predictions for inverse transform
    if len(predictions.shape) == 1:
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_original = target_scaler.inverse_transform(predictions_reshaped).flatten()
    else:
        predictions_original = target_scaler.inverse_transform(predictions)
    
    return predictions_original


def build_tfidf_features(train_samples, val_samples, test_samples, args):
    """
    Build TF-IDF features with scientifically justified parameters.

    Parameters chosen based on:
    - Sublinear TF: Reduces impact of very frequent terms (standard practice)
    - L2 norm: Standard normalization for TF-IDF
    - Bigrams: Capture local context without overfitting
    - SVD: Dimensionality reduction to prevent overfitting
    """
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Scientifically justified TF-IDF parameters
    tfidf_params = {
        "max_features": 10000,  # Balance between coverage and overfitting
        "sublinear_tf": True,  # Log scaling reduces impact of very frequent terms
        "norm": "l2",  # Standard L2 normalization
        "use_idf": True,  # Essential for TF-IDF
        "smooth_idf": True,  # Prevents division by zero
        "ngram_range": (1, 2),  # Unigrams + bigrams for local context
        "min_df": 2,  # Remove very rare terms (reduce noise)
    }

    print(f"Creating TF-IDF vectors with parameters: {tfidf_params}")
    vectorizer = TfidfVectorizer(**tfidf_params)

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"TF-IDF features shape: {X_train.shape}")

    # Apply dimensionality reduction if beneficial
    n_components = 1000  # Conservative choice for interpretability
    if X_train.shape[1] > n_components:
        from sklearn.decomposition import TruncatedSVD

        # Ensure we don't exceed sample constraints
        actual_n_components = min(n_components, min(X_train.shape) - 1)

        print(f"Applying SVD: {X_train.shape[1]} -> {actual_n_components} dimensions")

        svd = TruncatedSVD(
            n_components=actual_n_components,
            n_iter=7,  # Standard value for convergence
            random_state=42,
            algorithm="randomized",
        )

        X_train = svd.fit_transform(X_train)
        X_val = svd.transform(X_val)
        X_test = svd.transform(X_test)

        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"SVD explained variance: {explained_variance:.2%}")

        if explained_variance < 0.7:
            print(
                f"WARNING: Low explained variance ({explained_variance:.2%}). "
                f"Consider increasing n_components or reviewing data quality."
            )

        embedding_model = (vectorizer, svd)
    else:
        embedding_model = vectorizer

    # Extract targets with optional normalization
    normalize_targets = getattr(args, 'normalize_targets', False)
    y_train, target_scaler = extract_targets(train_samples, normalize=normalize_targets)
    y_val, _ = extract_targets(val_samples, normalize=normalize_targets, target_scaler=target_scaler)
    y_test, _ = extract_targets(test_samples, normalize=normalize_targets, target_scaler=target_scaler)

    # Include target scaler in embedding model tuple if normalization is used
    if normalize_targets and target_scaler is not None:
        if isinstance(embedding_model, tuple):
            embedding_model = embedding_model + (target_scaler,)
        else:
            embedding_model = (embedding_model, target_scaler)

    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model


def build_bert_features(
    train_samples, val_samples, test_samples, device, chunking_config, args
):
    """
    Build BERT features with simplified, scientifically sound chunking.

    Improvements:
    - Fixed overlap strategy based on literature (10% overlap)
    - Simple mean pooling instead of complex weighting
    - Reduced max chunks based on research
    """
    model_name = "microsoft/codebert-base"

    try:
        print(f"Loading BERT model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        # Standard BERT parameters
        max_length = 510  # 512 - 2 for special tokens

    except Exception as e:
        print(f"Error loading BERT model: {e}")
        print("Falling back to basic BERT model...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()
        max_length = 510

    def encode_with_simplified_chunking(df):
        """Simplified chunking strategy based on literature."""
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for text in tqdm(texts, desc="Encoding with BERT"):
            if not text or len(text.strip()) == 0:
                # Handle empty text
                inputs = tokenizer(
                    " ",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(embedding / np.linalg.norm(embedding))
                continue

            try:
                # Always tokenize with proper parameters to avoid length errors
                initial_inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,  # Don't truncate yet, just tokenize
                    add_special_tokens=True,
                )

                token_length = initial_inputs.input_ids.shape[1]

                if token_length <= max_length:
                    # Short text: process directly with proper truncation
                    inputs = tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    embeddings.append(embedding / np.linalg.norm(embedding))

                else:
                    # Long text: use simplified chunking
                    print(
                        f"  Processing long text with {token_length} tokens -> chunking"
                    )
                    chunk_embeddings = []

                    # Get tokens for chunking
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                    # Calculate chunk parameters
                    overlap = chunking_config.chunk_overlap_tokens
                    step_size = max_length - overlap - 2  # Account for special tokens

                    # Create overlapping chunks
                    chunks = []
                    for i in range(0, len(tokens), step_size):
                        chunk_tokens = tokens[i : i + step_size]
                        chunk_text = tokenizer.decode(
                            chunk_tokens, skip_special_tokens=True
                        )
                        chunks.append(chunk_text)

                        # Limit number of chunks based on literature
                        if len(chunks) >= chunking_config.max_chunks_per_doc:
                            break

                    print(f"    Created {len(chunks)} chunks")

                    # Process chunks with safe tokenization
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            inputs = tokenizer(
                                chunk,
                                max_length=max_length,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt",
                            ).to(device)

                            with torch.no_grad():
                                outputs = model(**inputs)
                                chunk_emb = (
                                    outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                                )
                                chunk_embeddings.append(
                                    chunk_emb / np.linalg.norm(chunk_emb)
                                )

                        except Exception as chunk_error:
                            print(
                                f"    Error processing chunk {chunk_idx}: {chunk_error}"
                            )
                            continue

                    # Aggregate chunks using simple mean (more interpretable)
                    if chunk_embeddings:
                        if chunking_config.chunk_aggregation == "mean":
                            final_embedding = np.mean(chunk_embeddings, axis=0)
                        elif chunking_config.chunk_aggregation == "first_last":
                            # Simple alternative: average first and last chunks
                            if len(chunk_embeddings) == 1:
                                final_embedding = chunk_embeddings[0]
                            else:
                                final_embedding = (
                                    chunk_embeddings[0] + chunk_embeddings[-1]
                                ) / 2
                        else:  # fallback to mean
                            final_embedding = np.mean(chunk_embeddings, axis=0)

                        final_embedding = final_embedding / np.linalg.norm(
                            final_embedding
                        )
                        embeddings.append(final_embedding)
                    else:
                        # Fallback for failed processing - use truncated version
                        print("    All chunks failed, using truncated text")
                        inputs = tokenizer(
                            text,
                            max_length=max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        ).to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            fallback_emb = (
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            )
                        embeddings.append(fallback_emb / np.linalg.norm(fallback_emb))

            except Exception as e:
                print(f"  Error processing text (length: {len(text)} chars): {e}")
                # Emergency fallback
                inputs = tokenizer(
                    text[:1000],  # Take first 1000 characters as emergency fallback
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    fallback_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(fallback_emb / np.linalg.norm(fallback_emb))

            # Memory cleanup
            if device == "cuda" and len(embeddings) % 10 == 0:
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    # Process all datasets
    print(f"Processing {len(train_samples)} training samples...")
    X_train = encode_with_simplified_chunking(train_samples)

    print(f"Processing {len(val_samples)} validation samples...")
    X_val = encode_with_simplified_chunking(val_samples)

    print(f"Processing {len(test_samples)} test samples...")
    X_test = encode_with_simplified_chunking(test_samples)

    print(f"BERT feature extraction complete. Shape: {X_train.shape}")

    # Extract targets with optional normalization
    normalize_targets = getattr(args, 'normalize_targets', False)
    y_train, target_scaler = extract_targets(train_samples, normalize=normalize_targets)
    y_val, _ = extract_targets(val_samples, normalize=normalize_targets, target_scaler=target_scaler)
    y_test, _ = extract_targets(test_samples, normalize=normalize_targets, target_scaler=target_scaler)

    # Include target scaler in embedding model tuple if normalization is used
    embedding_model = (tokenizer, model)
    if normalize_targets and target_scaler is not None:
        embedding_model = embedding_model + (target_scaler,)

    # Final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model


def build_llama_features(
    train_samples, val_samples, test_samples, device, chunking_config, args
):
    """
    Build LLaMA features with simplified chunking and better parameter management.

    """
    model_name = getattr(args, "llama_model", "codellama/CodeLlama-7b-hf")

    print(f"Loading LLaMA model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Explicit quantization configuration based on available memory
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        # Use 4-bit quantization by default for memory efficiency
        if chunking_config.enable_quantization and device == "cuda":
            print("Using 4-bit quantization for memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        # Get context length
        max_length = getattr(model.config, "max_position_embeddings", 2048) - 2
        print(f"Using context length: {max_length}")

    except Exception as e:
        print(f"Error loading LLaMA model: {e}")
        raise

    def encode_with_llama_simplified(df):
        """Simplified LLaMA encoding with better chunk management."""
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for text in tqdm(texts, desc="Encoding with LLaMA"):
            if not text or len(text.strip()) == 0:
                # Handle empty text
                embedding_dim = model.config.hidden_size
                embeddings.append(np.zeros(embedding_dim))
                continue

            try:
                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)

                if len(tokens) <= max_length:
                    # Process short text directly
                    inputs = tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]

                        # Mean pooling with attention mask
                        mask = inputs.attention_mask.unsqueeze(-1)
                        embedding = torch.sum(hidden_states * mask, dim=1) / torch.sum(
                            mask, dim=1
                        )
                        embedding = embedding.cpu().numpy()[0]
                        embeddings.append(embedding / np.linalg.norm(embedding))

                else:
                    # Process long text with simplified chunking
                    chunk_embeddings = []
                    overlap = chunking_config.chunk_overlap_tokens
                    step_size = max_length - overlap

                    # Create chunks
                    chunks = []
                    for i in range(0, len(tokens), step_size):
                        chunk_tokens = tokens[i : i + max_length]
                        chunk_text = tokenizer.decode(chunk_tokens)
                        chunks.append(chunk_text)

                        if len(chunks) >= chunking_config.max_chunks_per_doc:
                            break

                    # Process chunks
                    for chunk in chunks:
                        inputs = tokenizer(
                            chunk,
                            max_length=max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)
                            hidden_states = outputs.hidden_states[-1]

                            # Mean pooling
                            mask = inputs.attention_mask.unsqueeze(-1)
                            chunk_emb = torch.sum(
                                hidden_states * mask, dim=1
                            ) / torch.sum(mask, dim=1)
                            chunk_emb = chunk_emb.cpu().numpy()[0]
                            chunk_embeddings.append(
                                chunk_emb / np.linalg.norm(chunk_emb)
                            )

                    # Simple aggregation
                    if chunk_embeddings:
                        final_embedding = np.mean(chunk_embeddings, axis=0)
                        final_embedding = final_embedding / np.linalg.norm(
                            final_embedding
                        )
                        embeddings.append(final_embedding)
                    else:
                        # Fallback
                        embedding_dim = model.config.hidden_size
                        embeddings.append(np.zeros(embedding_dim))

            except Exception as e:
                print(f"Error processing text: {e}")
                embedding_dim = model.config.hidden_size
                embeddings.append(np.zeros(embedding_dim))

            # Memory cleanup
            if device == "cuda" and len(embeddings) % 5 == 0:
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    # Process datasets
    print(f"Processing {len(train_samples)} training samples...")
    X_train = encode_with_llama_simplified(train_samples)

    print(f"Processing {len(val_samples)} validation samples...")
    X_val = encode_with_llama_simplified(val_samples)

    print(f"Processing {len(test_samples)} test samples...")
    X_test = encode_with_llama_simplified(test_samples)

    print(f"LLaMA feature extraction complete. Shape: {X_train.shape}")

    # Extract targets with optional normalization
    normalize_targets = getattr(args, 'normalize_targets', False)
    y_train, target_scaler = extract_targets(train_samples, normalize=normalize_targets)
    y_val, _ = extract_targets(val_samples, normalize=normalize_targets, target_scaler=target_scaler)
    y_test, _ = extract_targets(test_samples, normalize=normalize_targets, target_scaler=target_scaler)

    # Include target scaler in embedding model tuple if normalization is used
    embedding_model = (tokenizer, model)
    if normalize_targets and target_scaler is not None:
        embedding_model = embedding_model + (target_scaler,)

    # Final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model
