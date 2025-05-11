from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm
import warnings
import os
import re
import gc
import json
import glob
from pathlib import Path
from dataset import load_dataset

# Set tokenizers parallelism explicitly before import/usage
# This prevents the warning when using DataLoader with num_workers > 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
Feature extraction module for Palladio Approximator.

This module provides functions for generating feature vectors from DSL models using 
various embedding techniques:

1. TF-IDF: Fast, sparse embeddings suitable for many machine learning models
2. BERT: Deep contextual embeddings using a pre-trained transformer model
3. LLaMA: State-of-the-art embeddings using large language models

The module now supports pre-computed LLaMA embeddings to address memory issues
with large models. When using LLaMA embeddings, the module will:

1. Check for pre-computed embeddings in the specified directory
2. Use pre-computed embeddings when available
3. Fall back to on-the-fly generation for samples without pre-computed embeddings
4. Maintain API compatibility with the rest of the codebase

Pre-computed embeddings are generated using the precompute_llama_embeddings.py script, 
which efficiently generates and saves embeddings for multiple DSL files in batches.

To use pre-computed embeddings:
1. Run `python precompute_llama_embeddings.py` to generate the embeddings
2. Run training normally with `python train.py --embedding llama ...`
3. The code will automatically detect and use the pre-computed embeddings

Pre-computed embeddings are stored in the 'features/llama_embeddings' directory by default
and can be configured using the 'precomputed_embeddings_dir' parameter.
"""

# Import the structured feature extraction module
# Add this at the top of your feature_extraction.py file
from structured_features import (
    extract_structured_features,
    combine_structured_and_embedded_features,
)


# Then modify the existing build_tfidf_features function to include structured features
def build_hybrid_tfidf_features(
    train_samples,
    val_samples,
    test_samples,
    max_features=None,
    apply_truncated_svd=True,
    n_components=2000,
    use_structured_features=True,
):
    """
    Builds hybrid features combining TF-IDF and structured architectural features.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        max_features: Maximum number of features to extract with TF-IDF
        apply_truncated_svd: Whether to apply dimensionality reduction
        n_components: Number of components to keep if using SVD
        use_structured_features: Whether to include structured features

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model
    """
    # First build the TF-IDF features using the existing function
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = (
        build_tfidf_features(
            train_samples,
            val_samples,
            test_samples,
            max_features=max_features,
            apply_truncated_svd=apply_truncated_svd,
            n_components=n_components,
        )
    )

    if not use_structured_features:
        return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model

    print("Adding structured architectural features...")

    # Extract train texts
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Extract structured features for each set
    train_structured = [extract_structured_features(text) for text in train_texts]
    val_structured = [extract_structured_features(text) for text in val_texts]
    test_structured = [extract_structured_features(text) for text in test_texts]

    # Ensure all structured feature dictionaries have the same keys
    all_keys = set()
    for features in train_structured + val_structured + test_structured:
        all_keys.update(features.keys())

    feature_keys = sorted(list(all_keys))

    # Convert to array format
    def dict_to_array(feature_dicts, keys):
        arrays = []
        for features in feature_dicts:
            # Ensure all features have the same keys in the same order
            feat_array = [features.get(key, 0) for key in keys]
            arrays.append(feat_array)
        return np.array(arrays)

    train_struct_array = dict_to_array(train_structured, feature_keys)
    val_struct_array = dict_to_array(val_structured, feature_keys)
    test_struct_array = dict_to_array(test_structured, feature_keys)

    print(f"Structured features shape: {train_struct_array.shape}")

    # Combine with TF-IDF features
    from scipy import sparse

    # Check if TF-IDF features are sparse
    if hasattr(X_train, "toarray"):
        # Convert structured to sparse and combine
        train_struct_sparse = sparse.csr_matrix(train_struct_array)
        val_struct_sparse = sparse.csr_matrix(val_struct_array)
        test_struct_sparse = sparse.csr_matrix(test_struct_array)

        X_train_combined = sparse.hstack([train_struct_sparse, X_train])
        X_val_combined = sparse.hstack([val_struct_sparse, X_val])
        X_test_combined = sparse.hstack([test_struct_sparse, X_test])
    else:
        # For dense arrays, use numpy concatenation
        X_train_combined = np.concatenate([train_struct_array, X_train], axis=1)
        X_val_combined = np.concatenate([val_struct_array, X_val], axis=1)
        X_test_combined = np.concatenate([test_struct_array, X_test], axis=1)

    print(f"Combined features shape: {X_train_combined.shape}")

    # Create a combined embedding model that includes structured feature extraction
    combined_model = {
        "tfidf_model": embedding_model,
        "structured_keys": feature_keys,
        "combined": True,
    }

    return (
        X_train_combined,
        y_train,
        X_val_combined,
        y_val,
        X_test_combined,
        y_test,
        combined_model,
    )


# Also modify the BERT features function to include structured features
def build_hybrid_bert_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="microsoft/codebert-base",
    device="cpu",
    batch_size=32,
    use_structured_features=True,
):
    """
    Builds hybrid features combining BERT embeddings and structured architectural features.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        model_name: Name of the pretrained model to use
        device: Device to run inference on
        batch_size: Batch size for processing
        use_structured_features: Whether to include structured features

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    # First build the BERT features using the existing function
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
        build_bert_features(
            train_samples,
            val_samples,
            test_samples,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )

    if not use_structured_features:
        return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model

    print("Adding structured architectural features...")

    # Extract train texts
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Extract structured features for each set
    train_structured = [extract_structured_features(text) for text in train_texts]
    val_structured = [extract_structured_features(text) for text in val_texts]
    test_structured = [extract_structured_features(text) for text in test_texts]

    # Ensure all structured feature dictionaries have the same keys
    all_keys = set()
    for features in train_structured + val_structured + test_structured:
        all_keys.update(features.keys())

    feature_keys = sorted(list(all_keys))

    # Convert to array format
    def dict_to_array(feature_dicts, keys):
        arrays = []
        for features in feature_dicts:
            # Ensure all features have the same keys in the same order
            feat_array = [features.get(key, 0) for key in keys]
            arrays.append(feat_array)
        return np.array(arrays)

    train_struct_array = dict_to_array(train_structured, feature_keys)
    val_struct_array = dict_to_array(val_structured, feature_keys)
    test_struct_array = dict_to_array(test_structured, feature_keys)

    print(f"Structured features shape: {train_struct_array.shape}")

    # Combine with BERT features (which are always dense)
    X_train_combined = np.concatenate([train_struct_array, X_train], axis=1)
    X_val_combined = np.concatenate([val_struct_array, X_val], axis=1)
    X_test_combined = np.concatenate([test_struct_array, X_test], axis=1)

    print(f"Combined features shape: {X_train_combined.shape}")

    # Create a combined model wrapper that includes structured feature extraction info
    embedding_model = (
        tokenizer,
        model,
        {"structured_keys": feature_keys, "combined": True},
    )

    return (
        X_train_combined,
        y_train,
        X_val_combined,
        y_val,
        X_test_combined,
        y_test,
        embedding_model,
    )


# To make it easy to use the new hybrid features, update the extract_features function
# This modified version of extract_features should be used to replace the existing one


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

    # Feature extraction
    print(
        f"Building features using {args.embedding} embedding (with structured features)..."
    )

    # Check if we should use hybrid features with structured architecture information
    use_structured = getattr(args, "use_structured_features", True)

    # Check if we should use pre-computed embeddings
    use_precomputed_embeddings = getattr(args, "use_precomputed_embeddings", True)
    precomputed_embeddings_dir = getattr(
        args, "precomputed_embeddings_dir", "features/llama_embeddings"
    )

    if args.embedding == "tfidf":
        # Configure parameters based on model type
        if args.model == "torch":
            max_features = None
            n_components = 2000
            apply_truncated_svd = True
        else:
            max_features = 10000
            n_components = 1000
            apply_truncated_svd = True

        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = (
            build_hybrid_tfidf_features(
                train_samples,
                val_samples,
                test_samples,
                max_features=max_features,
                apply_truncated_svd=apply_truncated_svd,
                n_components=n_components,
                use_structured_features=use_structured,
            )
        )

    elif args.embedding == "bert":
        print(f"Using device: {device}")
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
            build_hybrid_bert_features(
                train_samples,
                val_samples,
                test_samples,
                device=device,
                use_structured_features=use_structured,
            )
        )
        embedding_model = (tokenizer, model)

    elif args.embedding == "llama":
        # For LLaMA, we would need a similar hybrid approach
        device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
        if device != "cuda":
            print("WARNING: Llama models require CUDA. Forcing CUDA if available.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        model_name = (
            args.llama_model if args.llama_model else "codellama/CodeLlama-7b-hf"
        )

        # First get the base LLaMA features, with potential pre-computed embeddings
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
            build_llama_features(
                train_samples,
                val_samples,
                test_samples,
                model_name=model_name,
                device=device,
                batch_size=args.llama_batch_size,
                use_half_precision=not args.no_half_precision,
                use_8bit=args.use_8bit_llama,
                use_4bit=args.use_4bit_llama,
                memory_efficient=True,
                use_precomputed_embeddings=use_precomputed_embeddings,
                precomputed_embeddings_dir=precomputed_embeddings_dir,
            )
        )

        if use_structured:
            print("Adding structured architectural features...")

            # Extract train texts
            train_texts = train_samples["tpcm_text"].tolist()
            val_texts = val_samples["tpcm_text"].tolist()
            test_texts = test_samples["tpcm_text"].tolist()

            # Extract structured features for each set
            train_structured = [
                extract_structured_features(text) for text in train_texts
            ]
            val_structured = [extract_structured_features(text) for text in val_texts]
            test_structured = [extract_structured_features(text) for text in test_texts]

            # Ensure all structured feature dictionaries have the same keys
            all_keys = set()
            for features in train_structured + val_structured + test_structured:
                all_keys.update(features.keys())

            feature_keys = sorted(list(all_keys))

            # Convert to array format
            def dict_to_array(feature_dicts, keys):
                arrays = []
                for features in feature_dicts:
                    # Ensure all features have the same keys in the same order
                    feat_array = [features.get(key, 0) for key in keys]
                    arrays.append(feat_array)
                return np.array(arrays)

            train_struct_array = dict_to_array(train_structured, feature_keys)
            val_struct_array = dict_to_array(val_structured, feature_keys)
            test_struct_array = dict_to_array(test_structured, feature_keys)

            print(f"Structured features shape: {train_struct_array.shape}")

            # Combine with LLaMA features (which are dense)
            X_train = np.concatenate([train_struct_array, X_train], axis=1)
            X_val = np.concatenate([val_struct_array, X_val], axis=1)
            X_test = np.concatenate([test_struct_array, X_test], axis=1)

            print(f"Combined features shape: {X_train.shape}")

            # Add structured feature info to embedding model
            # Handle different types of embedding models returned by build_llama_features
            if isinstance(model, dict) and model.get("precomputed", False):
                # For a pre-computed model dict, add structured feature info
                model.update({"structured_keys": feature_keys, "combined": True})
                embedding_model = (tokenizer, model)
            else:
                # For a standard model, use the tuple with structured info
                embedding_model = (
                    tokenizer,
                    model,
                    {"structured_keys": feature_keys, "combined": True},
                )
        else:
            # Just use the model as is
            if isinstance(model, dict) and model.get("precomputed", False):
                embedding_model = (tokenizer, model)
            else:
                embedding_model = (tokenizer, model)

        print("Feature extraction completed successfully")
        print(f"Feature shapes: X_train={X_train.shape}, y_train={y_train.shape}")

        # Free up memory
        if device == "cuda" and torch.cuda.is_available():
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()

    else:
        raise ValueError(
            "Unsupported embedding type. Choose 'tfidf', 'bert', or 'llama'"
        )

    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model


def extract_targets(df):
    """Extract avg/min/max response times from dataframe."""
    return df[["avg_resp_time", "min_resp_time", "max_resp_time"]].to_numpy()


def build_tfidf_features(
    train_samples,
    val_samples,
    test_samples,
    max_features=None,
    apply_truncated_svd=True,
    n_components=2000,
):
    """
    Builds TF-IDF feature matrices and target arrays from dataset samples.
    Optimized for high-quality features with optional dimensionality reduction.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        max_features: Maximum number of features to extract with TF-IDF (None for all features)
        apply_truncated_svd: Whether to apply dimensionality reduction
        n_components: Number of components to keep if using SVD

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model
    """
    # Extract DSL text
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Create TF-IDF vectorizer with enhanced parameters for better feature quality
    print(f"Creating TF-IDF vectors with max_features={max_features}...")

    # Create a high-quality TF-IDF vectorizer
    # - sublinear_tf: Apply sublinear scaling (logarithmic) to term frequencies
    # - min_df: Ignore terms that appear in less than 3 documents
    # - max_df: Ignore terms that appear in more than 95% of documents (likely boilerplate)
    # - norm: L2 normalization of vectors for better numerical stability
    # - use_idf: Apply inverse document frequency weighting
    # - smooth_idf: Add 1 to document frequencies to prevent division by zero
    # - ngram_range: Include both unigrams and bigrams for better feature representation
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        ngram_range=(1, 2),  # Include unigrams and bigrams
    )

    # Fit vectorizer and transform text
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"TF-IDF features shape: {X_train.shape}")

    # Apply dimensionality reduction if requested
    if apply_truncated_svd and n_components < X_train.shape[1]:
        from sklearn.decomposition import TruncatedSVD

        print(
            f"Applying TruncatedSVD to reduce dimensions from {X_train.shape[1]} to {n_components}..."
        )

        # Check if we have enough samples for the requested components
        actual_n_components = min(n_components, min(X_train.shape) - 1)
        if actual_n_components != n_components:
            print(
                f"Reducing n_components to {actual_n_components} due to sample size constraints"
            )
            n_components = actual_n_components

        # Apply advanced SVD for high-quality dimensionality reduction
        # - High n_iter value for better convergence
        # - Randomized algorithm for faster processing of large matrices
        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=10,  # More iterations for better convergence
            random_state=42,
            algorithm="randomized",
        )

        # Transform data
        X_train = svd.fit_transform(X_train)
        X_val = svd.transform(X_val)
        X_test = svd.transform(X_test)

        print(f"After dimensionality reduction: {X_train.shape}")
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_variance:.2%}")

        # If explained variance is too low, consider not using SVD
        if explained_variance < 0.5:
            print(
                f"WARNING: Low explained variance ({explained_variance:.2%}). Consider reducing n_components or skipping SVD."
            )

        # Create combined model for inference
        embedding_model = (vectorizer, svd)
    else:
        embedding_model = vectorizer

    # Extract targets
    y_train = extract_targets(train_samples)
    y_val = extract_targets(val_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model


def build_llama_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="codellama/CodeLlama-7b-hf",
    device="cuda",
    batch_size=1,  # Reduced batch size to avoid OOM
    use_half_precision=True,  # Use half precision for efficiency with large models
    use_8bit=False,  # Whether to use 8-bit quantization
    use_4bit=False,  # Whether to use 4-bit quantization (most memory efficient)
    memory_efficient=True,  # Use memory-efficient settings
    use_precomputed_embeddings=True,  # Whether to use pre-computed embeddings when available
    precomputed_embeddings_dir="features/llama_embeddings",  # Directory with pre-computed embeddings
):
    """
    Builds embeddings using Llama-based models (like CodeLlama) and target arrays from dataset samples.
    Optimized for handling large models and long code inputs.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        model_name: Name of the Llama model to use (e.g., "codellama/CodeLlama-7b-hf")
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for processing (smaller for large models)
        use_half_precision: Whether to use half precision (float16) to save memory
        use_8bit: Whether to use 8-bit quantization
        use_4bit: Whether to use 4-bit quantization (most memory efficient)
        memory_efficient: Whether to use memory-efficient settings
        use_precomputed_embeddings: Whether to use pre-computed embeddings when available
        precomputed_embeddings_dir: Directory containing pre-computed embeddings

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    # First, try to use pre-computed embeddings if requested
    if use_precomputed_embeddings:
        print("Checking for pre-computed LLaMA embeddings...")

        try:
            # Try to load pre-computed embeddings
            (
                all_embeddings,
                metadata,
                sample_ids_with_embeddings,
                sample_ids_without_embeddings,
            ) = load_precomputed_llama_embeddings(
                [train_samples, val_samples, test_samples],
                embeddings_dir=precomputed_embeddings_dir,
            )

            # Check if we have a good coverage of embeddings
            total_samples = len(train_samples) + len(val_samples) + len(test_samples)
            coverage = len(sample_ids_with_embeddings) / total_samples

            # Get embeddings and indices
            train_embeddings, train_indices_with_emb, train_indices_without_emb = (
                all_embeddings[0]
            )
            val_embeddings, val_indices_with_emb, val_indices_without_emb = (
                all_embeddings[1]
            )
            test_embeddings, test_indices_with_emb, test_indices_without_emb = (
                all_embeddings[2]
            )

            if coverage > 0.95:  # More than 95% coverage, just use pre-computed
                print(
                    f"Using pre-computed embeddings for all samples (coverage: {coverage:.2%})"
                )

                # Extract targets
                y_train = extract_targets(train_samples)
                y_val = extract_targets(val_samples)
                y_test = extract_targets(test_samples)

                # Create mock tokenizer and model for API compatibility
                tokenizer = None
                model = None

                # Create a mock model container with metadata
                if metadata:
                    embedding_dim = metadata.get("embedding_dim", 4096)
                    model_info = {
                        "model_name": metadata.get("model_name", model_name),
                        "embedding_dim": embedding_dim,
                        "precomputed": True,
                        "coverage": coverage,
                        "metadata": metadata,
                    }
                    model = model_info

                return (
                    train_embeddings,
                    y_train,
                    val_embeddings,
                    y_val,
                    test_embeddings,
                    y_test,
                    tokenizer,
                    model,
                )

            # Always use what's available, regardless of coverage percentage
            # Even low coverage is better than re-computing everything
            print(
                f"Using pre-computed embeddings where available (coverage: {coverage:.2%})"
            )
            print(
                f"Will generate embeddings for {len(sample_ids_without_embeddings)} remaining samples"
            )

            # Need to load model for remaining samples
            tokenizer, model = _load_llama_model(
                model_name,
                device,
                use_half_precision,
                use_8bit,
                use_4bit,
                memory_efficient,
            )

            # Function to get samples without embeddings
            def get_samples_without_embeddings(samples, indices_without_emb):
                return samples.iloc[indices_without_emb].reset_index(drop=True)

            # Get samples that need embeddings
            train_without_emb = (
                get_samples_without_embeddings(train_samples, train_indices_without_emb)
                if train_indices_without_emb
                else None
            )
            val_without_emb = (
                get_samples_without_embeddings(val_samples, val_indices_without_emb)
                if val_indices_without_emb
                else None
            )
            test_without_emb = (
                get_samples_without_embeddings(test_samples, test_indices_without_emb)
                if test_indices_without_emb
                else None
            )

            # Generate embeddings for remaining samples
            # Use the existing encoder function
            print("Generating embeddings for remaining samples...")

            def generate_missing_embeddings(samples_without_emb, embedding_dim):
                if not samples_without_emb or len(samples_without_emb) == 0:
                    return np.zeros((0, embedding_dim))
                return _encode_with_llama(
                    samples_without_emb,
                    tokenizer,
                    model,
                    device,
                    max_length=1024,  # Default for LLaMA
                    sliding_window_overlap=100,
                    max_chunks_per_doc=15,
                    use_half_precision=use_half_precision,
                )

            # Get embedding dimension
            embedding_dim = metadata.get("embedding_dim", 4096) if metadata else 4096
            if (
                not embedding_dim
                and hasattr(model, "config")
                and hasattr(model.config, "hidden_size")
            ):
                embedding_dim = model.config.hidden_size

            # Generate missing embeddings
            train_missing_embeddings = (
                generate_missing_embeddings(train_without_emb, embedding_dim)
                if train_without_emb is not None
                else None
            )
            val_missing_embeddings = (
                generate_missing_embeddings(val_without_emb, embedding_dim)
                if val_without_emb is not None
                else None
            )
            test_missing_embeddings = (
                generate_missing_embeddings(test_without_emb, embedding_dim)
                if test_without_emb is not None
                else None
            )

            # Complete the embeddings by filling in the missing values
            def complete_embeddings(
                precomputed_embeddings,
                indices_with_emb,
                indices_without_emb,
                missing_embeddings,
                total_samples,
            ):
                # Create a new array for the complete embeddings
                complete = np.zeros((total_samples, precomputed_embeddings.shape[1]))

                # Fill in the pre-computed embeddings
                for idx, emb_idx in enumerate(indices_with_emb):
                    complete[emb_idx] = precomputed_embeddings[emb_idx]

                # Fill in the newly computed embeddings
                if missing_embeddings is not None and indices_without_emb:
                    for idx, emb_idx in enumerate(indices_without_emb):
                        if idx < len(missing_embeddings):
                            complete[emb_idx] = missing_embeddings[idx]

                return complete

            # Complete all embeddings
            X_train = complete_embeddings(
                train_embeddings,
                train_indices_with_emb,
                train_indices_without_emb,
                train_missing_embeddings,
                len(train_samples),
            )
            X_val = complete_embeddings(
                val_embeddings,
                val_indices_with_emb,
                val_indices_without_emb,
                val_missing_embeddings,
                len(val_samples),
            )
            X_test = complete_embeddings(
                test_embeddings,
                test_indices_with_emb,
                test_indices_without_emb,
                test_missing_embeddings,
                len(test_samples),
            )

            # Extract targets
            y_train = extract_targets(train_samples)
            y_val = extract_targets(val_samples)
            y_test = extract_targets(test_samples)

            # Add metadata to model
            if isinstance(model, dict):
                model.update(
                    {
                        "precomputed_partial": True,
                        "coverage": coverage,
                        "metadata": metadata,
                    }
                )

            print("Feature extraction complete (hybrid pre-computed/on-the-fly)!")
            print(f"Features shape: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"X_val={X_val.shape}, X_test={X_test.shape}")

            # Free GPU memory
            if device == "cuda":
                print("Clearing GPU cache...")
                torch.cuda.empty_cache()

            return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model

        except Exception as e:
            print(f"Error using pre-computed embeddings: {e}")
            print("Falling back to on-the-fly embedding generation")

    # If we got here, we're generating embeddings without using pre-computed ones
    print(f"Setting up embeddings with {model_name} (on-the-fly generation)")

    # Load the LLaMA model
    tokenizer, model = _load_llama_model(
        model_name, device, use_half_precision, use_8bit, use_4bit, memory_efficient
    )

    # Get model context length - LLaMA models can handle much longer contexts
    try:
        # Try to get context length from model config
        if hasattr(model.config, "max_position_embeddings"):
            max_length = model.config.max_position_embeddings - 2
        else:
            # Try common attributes for max length
            config_attrs = [
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "max_seq_len",
            ]
            for attr in config_attrs:
                if hasattr(model.config, attr):
                    max_length = getattr(model.config, attr) - 2
                    break
            else:
                # Fallback values based on model name patterns
                if "13b" in model_name.lower():
                    max_length = 4096 - 2
                else:
                    max_length = 2048 - 2

        print(f"Using model's max sequence length: {max_length}")
    except:
        # Fall back to conservative default for LLaMA
        max_length = 2048 - 2
        print(f"Using default max sequence length: {max_length}")

    # For long inputs, we need specific handling strategies
    sliding_window_overlap = 100  # More overlap for better context preservation
    max_chunks_per_doc = 15  # Increased from 10 to handle very long documents better

    # Process datasets
    print(f"Processing training set ({len(train_samples)} samples)...")
    X_train = _encode_with_llama(
        train_samples,
        tokenizer,
        model,
        device,
        max_length=max_length,
        sliding_window_overlap=sliding_window_overlap,
        max_chunks_per_doc=max_chunks_per_doc,
        use_half_precision=use_half_precision,
    )
    y_train = extract_targets(train_samples)

    print(f"Processing validation set ({len(val_samples)} samples)...")
    X_val = _encode_with_llama(
        val_samples,
        tokenizer,
        model,
        device,
        max_length=max_length,
        sliding_window_overlap=sliding_window_overlap,
        max_chunks_per_doc=max_chunks_per_doc,
        use_half_precision=use_half_precision,
    )
    y_val = extract_targets(val_samples)

    print(f"Processing test set ({len(test_samples)} samples)...")
    X_test = _encode_with_llama(
        test_samples,
        tokenizer,
        model,
        device,
        max_length=max_length,
        sliding_window_overlap=sliding_window_overlap,
        max_chunks_per_doc=max_chunks_per_doc,
        use_half_precision=use_half_precision,
    )
    y_test = extract_targets(test_samples)

    print("Feature extraction complete!")
    print(f"Features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_val={X_val.shape}, X_test={X_test.shape}")

    # Free GPU memory
    if device == "cuda":
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model


def _load_llama_model(
    model_name, device, use_half_precision, use_8bit, use_4bit, memory_efficient
):
    """Helper function to load the LLaMA model with optimizations."""
    # Force device to CPU if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        device = "cpu"

    if device == "cuda":
        # Check available GPU memory and warn if low
        free_mem = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        free_mem_gb = free_mem / (1024**3)
        print(f"Available CUDA memory: {free_mem_gb:.2f} GB")

        if free_mem_gb < 10 and "7b" in model_name.lower():
            print(
                "WARNING: Less than 10GB VRAM available. Model may not fit in memory."
            )
        elif free_mem_gb < 20 and "13b" in model_name.lower():
            print(
                "WARNING: Less than 20GB VRAM available. Model may not fit in memory."
            )

    # Set PyTorch memory allocation to be more efficient
    if memory_efficient and device == "cuda":
        print("Setting up memory-efficient PyTorch configuration")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,expandable_segments:True"
        )

        # Free up memory before loading model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Try to load the model with optimizations for large models
    print(f"Loading model {model_name}...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            # Some LLaMA models don't have a pad token
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization if requested
        quantization_config = None
        if use_4bit:
            print("Using 4-bit quantization for maximum memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.float16 if use_half_precision else torch.float32
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Expected memory usage reduction: ~75% compared to FP16")
        elif use_8bit:
            print("Using 8-bit quantization for better memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=(
                    torch.float16 if use_half_precision else torch.float32
                ),
            )
            print("Expected memory usage reduction: ~50% compared to FP16")

        # Configure model loading parameters based on device and memory constraints
        model_kwargs = {
            "torch_dtype": (
                torch.float16
                if use_half_precision and device == "cuda"
                else torch.float32
            ),
            "low_cpu_mem_usage": True,
            "device_map": "auto" if memory_efficient else None,
        }

        # Add quantization config if available
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            # Device map must be 'auto' with quantization
            model_kwargs["device_map"] = "auto"

        # Try to identify model architecture and load appropriate class
        if any(
            name in model_name.lower() for name in ["llama", "mistral", "codellama"]
        ):
            # For LLaMA-based models, use CausalLM architecture
            print("Loading LLaMA-based model with Causal LM architecture")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            # For other models, use base model class
            model = AutoModel.from_pretrained(model_name, **model_kwargs)

        # Move model to appropriate device
        model = model.to(device)

        # Put model in evaluation mode
        model.eval()

        print(f"Successfully loaded {model_name}")

    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to smaller CodeLLaMA model")

        try:
            # Try smaller model as fallback
            fallback_model = "codellama/CodeLlama-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "torch_dtype": (
                    torch.float16
                    if use_half_precision and device == "cuda"
                    else torch.float32
                ),
                "low_cpu_mem_usage": True,
            }

            model = AutoModelForCausalLM.from_pretrained(fallback_model, **model_kwargs)
            model = model.to(device)
            model.eval()
            print(f"Successfully loaded fallback model {fallback_model}")

        except Exception as fallback_e:
            print(f"Error loading fallback model: {fallback_e}")
            raise RuntimeError(
                "Could not load any Llama model. Please check your transformers and torch installation."
            )

    return tokenizer, model


def _encode_with_llama(
    df,
    tokenizer,
    model,
    device,
    max_length=1024,
    sliding_window_overlap=100,
    max_chunks_per_doc=15,
    use_half_precision=True,
):
    """
    Create embeddings from text using a Llama-based model.
    Optimized for long code sequences and large models.
    """
    texts = df["tpcm_text"].tolist()
    embeddings = []

    # Use mixed precision for efficiency
    amp_enabled = device == "cuda" and use_half_precision

    # For memory monitoring
    if device == "cuda" and torch.cuda.is_available():
        peak_memory = 0

        def log_memory():
            nonlocal peak_memory
            current = torch.cuda.max_memory_allocated() / (1024**3)
            if current > peak_memory:
                peak_memory = current
                print(f"Peak GPU memory usage: {peak_memory:.2f} GB")

            # Clear memory statistics
            torch.cuda.reset_peak_memory_stats()
            # Free cache
            torch.cuda.empty_cache()

        log_memory()
    else:

        def log_memory():
            pass

    # Process texts one by one to minimize memory usage
    actual_batch_size = 1  # Force batch size to 1 for LLaMA
    for i in tqdm(range(0, len(texts), actual_batch_size), desc=f"Encoding with LLaMA"):
        batch_texts = texts[i : i + actual_batch_size]
        batch_embeddings = []

        # Aggressive memory cleanup before processing each text
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Process each text in batch
        for text in batch_texts:
            # Initialize chunks to an empty list at the beginning
            chunks = []

            # Handle empty text
            if not text or len(text.strip()) == 0:
                # Create embedding for empty text
                inputs = tokenizer(" ", return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    # For CausalLM models, use the hidden states
                    if isinstance(model, AutoModelForCausalLM):
                        outputs = model(**inputs, output_hidden_states=True)
                        # Use last layer's hidden state of the last token
                        embedding = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                    else:
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

                normalized_emb = embedding / np.linalg.norm(embedding)
                batch_embeddings.append(normalized_emb)
                continue

            try:
                # Tokenize the text to get token IDs
                tokenized_text = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_tensors=None,
                )

                token_count = len(tokenized_text)

                if token_count > max_length:
                    # For long documents, use a chunking approach appropriate for Llama models
                    # LLaMA models can handle longer contexts, but we still need chunking for very long inputs

                    # Find natural break points (focusing on code structure)
                    natural_breaks = []
                    for pattern in [
                        r"\n\s*def\s+",
                        r"\n\s*class\s+",
                        r"\n\s*if\s+__name__",
                        r"\n\s*#",
                    ]:
                        for match in re.finditer(pattern, text):
                            natural_breaks.append((match.start(), "code_structure"))

                    # Also add simple newline blocks as potential break points
                    for match in re.finditer(r"\n\n+", text):
                        natural_breaks.append((match.start(), "newline"))

                    # Sort breaks by position
                    natural_breaks.sort(key=lambda x: x[0])

                    # Handle chunking
                    if not natural_breaks:
                        # No natural breaks - use token-based chunking
                        # For LLaMA, we can use longer chunks
                        effective_length = max_length - sliding_window_overlap

                        chunks = []
                        for j in range(0, token_count, effective_length):
                            # Get token IDs for this chunk with overlap
                            end_idx = min(j + max_length, token_count)
                            chunk_tokens = tokenized_text[j:end_idx]
                            chunk_text = tokenizer.decode(chunk_tokens)
                            chunks.append(chunk_text)
                    else:
                        # Use natural breaks
                        chunks = []
                        start_pos = 0

                        # Estimate character length per token
                        char_ratio = len(text) / token_count
                        target_char_length = int(
                            max_length * char_ratio * 0.9
                        )  # 90% to be safe

                        while start_pos < len(text):
                            target_pos = start_pos + target_char_length
                            if target_pos >= len(text):
                                chunks.append(text[start_pos:])
                                break

                            # Find break points near our target position
                            valid_breaks = [
                                b
                                for b in natural_breaks
                                if b[0] > start_pos and b[0] <= target_pos + 1000
                            ]

                            if valid_breaks:
                                # Prioritize code structure breaks over simple newlines
                                code_breaks = [
                                    b for b in valid_breaks if b[1] == "code_structure"
                                ]
                                if code_breaks:
                                    break_pos = code_breaks[-1][0]
                                else:
                                    break_pos = valid_breaks[-1][0]

                                chunks.append(text[start_pos : break_pos + 1])
                                start_pos = break_pos + 1
                            else:
                                # No good break found
                                safe_pos = min(target_pos, len(text) - 1)
                                chunks.append(text[start_pos:safe_pos])
                                start_pos = safe_pos

                            # Safety check
                            if len(chunks) >= max_chunks_per_doc:
                                if start_pos < len(text):
                                    chunks.append(text[start_pos:])
                                break
                else:
                    # Text fits within model's context window
                    chunks = [text]

                # Safety check - ensure chunks is not empty
                if not chunks:
                    print(
                        "Warning: Chunks list is empty, using whole text as single chunk"
                    )
                    chunks = [text]

                # Further limit chunks for memory efficiency with large models
                # Increased from 4/6 to 8/12 for better document coverage
                max_chunks = (
                    8
                    if "13b" in getattr(model, "config", {}).name_or_path.lower()
                    else 12
                )
                if len(chunks) > max_chunks:
                    print(
                        f"Limiting document from {len(chunks)} to {max_chunks} chunks due to memory constraints"
                    )
                    # Keep first, last, and evenly sample the middle chunks
                    if max_chunks >= 3:
                        middle_chunks = chunks[1:-1]
                        num_middle = max_chunks - 2
                        step = len(middle_chunks) / num_middle
                        middle_indices = [int(i * step) for i in range(num_middle)]
                        selected_middle = [middle_chunks[i] for i in middle_indices]
                        chunks = [chunks[0]] + selected_middle + [chunks[-1]]
                    else:
                        chunks = chunks[:max_chunks]
            except Exception as e:
                print(f"Error during document chunking: {e}")
                print("Falling back to single chunk with truncation")
                chunks = [text[: min(len(text), int(max_length * 0.9))]]

            # Process all chunks and generate embeddings
            try:
                chunk_embeddings = []

                # Process chunks one by one with memory cleanup between
                for chunk_idx, chunk in enumerate(chunks):
                    # Log memory usage
                    if chunk_idx > 0 and chunk_idx % 2 == 0:
                        log_memory()

                    # Tokenize with appropriate padding for LLaMA
                    inputs = tokenizer(
                        chunk,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(device)

                    # Clear tokenizer cache and garbage collect
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    # Generate embeddings
                    with torch.no_grad():
                        if amp_enabled:
                            with torch.amp.autocast("cuda"):
                                if isinstance(model, AutoModelForCausalLM):
                                    outputs = model(**inputs, output_hidden_states=True)
                                    # Use last hidden layer representation
                                    hidden_states = outputs.hidden_states[-1]
                                    # Average the token embeddings
                                    mask = inputs.attention_mask.unsqueeze(-1)
                                    # Use mean pooling over sequence
                                    embedding = torch.sum(
                                        hidden_states * mask, dim=1
                                    ) / torch.sum(mask, dim=1)
                                    embedding = embedding.cpu().numpy()[0]
                                else:
                                    outputs = model(**inputs)
                                    embedding = (
                                        outputs.last_hidden_state[:, 0, :]
                                        .cpu()
                                        .numpy()[0]
                                    )
                        else:
                            if isinstance(model, AutoModelForCausalLM):
                                outputs = model(**inputs, output_hidden_states=True)
                                hidden_states = outputs.hidden_states[-1]
                                mask = inputs.attention_mask.unsqueeze(-1)
                                embedding = torch.sum(
                                    hidden_states * mask, dim=1
                                ) / torch.sum(mask, dim=1)
                                embedding = embedding.cpu().numpy()[0]
                            else:
                                outputs = model(**inputs)
                                embedding = (
                                    outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                                )

                    # Normalize embedding
                    normalized_emb = embedding / np.linalg.norm(embedding)
                    chunk_embeddings.append(normalized_emb)

                # Combine chunk embeddings with smart weighting
                if len(chunk_embeddings) > 1:
                    # Use position and content-based weighting
                    num_chunks = len(chunk_embeddings)

                    # Position weights emphasize beginning and end
                    pos_weights = np.ones(num_chunks)
                    if num_chunks >= 3:
                        pos_weights[0] = 1.5  # First chunk (imports, declarations)
                        pos_weights[-1] = 1.3  # Last chunk (often main logic)

                    # Length-based weights
                    length_weights = np.array([len(c) for c in chunks])
                    length_weights = length_weights / length_weights.sum()

                    # Final weights
                    final_weights = 0.4 * pos_weights + 0.6 * length_weights
                    final_weights = final_weights / final_weights.sum()

                    # Apply weights
                    weighted_emb = np.zeros_like(chunk_embeddings[0])
                    for j, emb in enumerate(chunk_embeddings):
                        weighted_emb += emb * final_weights[j]

                    # Normalize combined embedding
                    weighted_emb_norm = weighted_emb / np.linalg.norm(weighted_emb)
                    batch_embeddings.append(weighted_emb_norm)
                elif len(chunk_embeddings) == 1:
                    # Just one chunk
                    batch_embeddings.append(chunk_embeddings[0])
                else:
                    # Fallback for empty chunks
                    print("Warning: No chunk embeddings created, using fallback")
                    inputs = tokenizer(" ", return_tensors="pt").to(device)

                    with torch.no_grad():
                        if isinstance(model, AutoModelForCausalLM):
                            outputs = model(**inputs, output_hidden_states=True)
                            embedding = (
                                outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                            )
                        else:
                            outputs = model(**inputs)
                            embedding = (
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            )

                    normalized_emb = embedding / np.linalg.norm(embedding)
                    batch_embeddings.append(normalized_emb)

            except Exception as e:
                print(f"Error processing chunks: {e}")
                # Fallback embedding (zeros)
                if isinstance(model, AutoModelForCausalLM):
                    embedding_dim = model.config.hidden_size
                else:
                    embedding_dim = model.config.hidden_size

                fallback_emb = np.zeros(embedding_dim)
                batch_embeddings.append(fallback_emb)

        # Add batch embeddings to result
        embeddings.extend(batch_embeddings)

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final GPU cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    # Stack all embeddings
    if not embeddings:
        # Handle case where no embeddings were created
        print("Warning: No embeddings were created, returning zeros")
        if isinstance(model, AutoModelForCausalLM):
            embedding_dim = model.config.hidden_size
        else:
            embedding_dim = model.config.hidden_size
        return np.zeros((1, embedding_dim))

    return np.vstack(embeddings)


def load_precomputed_llama_embeddings(
    samples_list,
    embeddings_dir="features/llama_embeddings",
    metadata_file="embedding_metadata.json",
):
    """
    Load pre-computed LLaMA embeddings from disk.

    Args:
        samples_list: List of DataFrames containing samples (train, val, test)
        embeddings_dir: Directory containing pre-computed embeddings
        metadata_file: Filename of metadata JSON in embeddings_dir

    Returns:
        tuple: (embeddings_list, metadata, sample_ids_with_embeddings, sample_ids_without_embeddings)
            embeddings_list: List of numpy arrays containing embeddings for each samples list
            metadata: Dictionary with embedding information
            sample_ids_with_embeddings: List of sample IDs that had pre-computed embeddings
            sample_ids_without_embeddings: List of sample IDs that did not have pre-computed embeddings
    """
    print(f"Loading pre-computed LLaMA embeddings from {embeddings_dir}...")
    embeddings_path = Path(embeddings_dir)
    metadata_path = embeddings_path / metadata_file

    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(
                f"Loaded embedding metadata. Model: {metadata.get('model_name', 'unknown')}, "
                f"Dimension: {metadata.get('embedding_dim', 'unknown')}"
            )
        except Exception as e:
            print(f"Warning: Could not load embedding metadata: {e}")

    # Get all available embedding files
    available_embeddings = {
        Path(path).stem: path for path in glob.glob(str(embeddings_path / "*.npy"))
    }

    print(f"Found {len(available_embeddings)} pre-computed embeddings")

    # Process each samples list (train, val, test)
    all_embeddings = []
    all_sample_ids_with_embeddings = []
    all_sample_ids_without_embeddings = []

    for samples in samples_list:
        # Extract file IDs from filenames in the dataset
        sample_ids = []
        for _, sample in samples.iterrows():
            text = sample["tpcm_text"]
            file_id = None

            # Method 1: Try to find file ID in file content
            # Start with the first 10 lines where metadata/comments usually appear
            for line in text.split("\n")[:10]:
                # Look for patterns like "model_X" or "generated__ABCDE"
                for pattern in ["model_", "generated__"]:
                    if pattern in line:
                        # Extract ID that starts with the pattern and is followed by alphanumerics
                        match = re.search(f"({pattern}[A-Za-z0-9_]+)", line)
                        if match:
                            file_id = match.group(1)
                            break
                if file_id:
                    break

            # If no ID found in first 10 lines, scan the entire text
            if not file_id:
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        # Try to match common patterns in DSL files
                        match = re.search(
                            r"(model_[A-Za-z0-9_]+|generated__[A-Za-z0-9_]+)", line
                        )
                        if match:
                            file_id = match.group(1)
                            break

            # Method 2: If index is available, try to match with filename pattern directly
            if not file_id and "filename" in sample:
                filename = sample["filename"]
                if filename:
                    # Extract base name without extension
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    # Check if it matches our expected patterns
                    if base_name.startswith("model_") or base_name.startswith(
                        "generated__"
                    ):
                        file_id = base_name

            # Method 3: Try matching any available embedding by content length
            # This is a last resort fallback for when we can't identify the file
            if not file_id:
                # Get text length as a fingerprint
                text_fingerprint = len(text.strip())
                # Use a numerical identifier based on row index + text length
                file_id = f"unknown_len_{text_fingerprint}_idx_{_}"

            sample_ids.append(file_id)

        # Find which samples have pre-computed embeddings
        sample_ids_with_embeddings = []
        sample_indices_with_embeddings = []
        sample_ids_without_embeddings = []
        sample_indices_without_embeddings = []

        for i, sample_id in enumerate(sample_ids):
            if sample_id and sample_id in available_embeddings:
                sample_ids_with_embeddings.append(sample_id)
                sample_indices_with_embeddings.append(i)
            else:
                if sample_id:
                    sample_ids_without_embeddings.append(sample_id)
                else:
                    sample_ids_without_embeddings.append(f"unknown_at_index_{i}")
                sample_indices_without_embeddings.append(i)

        # Store globally
        all_sample_ids_with_embeddings.extend(sample_ids_with_embeddings)
        all_sample_ids_without_embeddings.extend(sample_ids_without_embeddings)

        # Load embeddings for samples that have them
        embeddings = None
        if sample_indices_with_embeddings:
            # Determine embedding dimension from metadata or first file
            embedding_dim = metadata.get("embedding_dim", 4096) if metadata else 4096
            # Pre-allocate array for all samples
            embeddings = np.zeros((len(samples), embedding_dim))

            # Load each embedding
            for sample_id, idx in zip(
                sample_ids_with_embeddings, sample_indices_with_embeddings
            ):
                embedding_path = available_embeddings[sample_id]
                try:
                    embedding = np.load(embedding_path)
                    embeddings[idx] = embedding
                except Exception as e:
                    print(f"Error loading embedding for {sample_id}: {e}")
                    # Mark as not having an embedding
                    sample_indices_with_embeddings.remove(idx)
                    sample_indices_without_embeddings.append(idx)
                    sample_ids_with_embeddings.remove(sample_id)
                    sample_ids_without_embeddings.append(sample_id)

        all_embeddings.append(
            (
                embeddings,
                sample_indices_with_embeddings,
                sample_indices_without_embeddings,
            )
        )

    coverage = (
        len(all_sample_ids_with_embeddings) / sum(len(s) for s in samples_list) * 100
    )
    print(
        f"Pre-computed embeddings coverage: {coverage:.2f}% ({len(all_sample_ids_with_embeddings)} out of {sum(len(s) for s in samples_list)} samples)"
    )

    return (
        all_embeddings,
        metadata,
        all_sample_ids_with_embeddings,
        all_sample_ids_without_embeddings,
    )


def build_bert_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="microsoft/codebert-base",
    device="cpu",
    batch_size=32,  # Increased for RTX 3090
):
    """
    Builds BERT embeddings and target arrays from dataset samples.
    Optimized for RTX 3090 GPU.

    Args:
        train_samples: Training data samples (DataFrame)
        val_samples: Validation data samples (DataFrame)
        test_samples: Test data samples (DataFrame)
        model_name: Name of the pretrained model to use
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for processing (32 is optimal for RTX 3090)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    # Set torch to use higher precision operations for better embedding quality
    if device == "cuda":
        # Try to optimize CUDA operations
        if torch.cuda.is_available():
            # Set optimal CUDA settings for RTX 3090
            torch.backends.cudnn.benchmark = True  # Speed up training
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN

            # Check for available memory and adjust batch size if needed
            free_mem = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            free_mem_gb = free_mem / (1024**3)
            print(f"Available CUDA memory: {free_mem_gb:.2f} GB")

            # If less than 8GB free, reduce batch size
            if free_mem_gb < 8 and batch_size > 16:
                print(
                    f"Reducing batch size from {batch_size} to 16 due to memory constraints"
                )
                batch_size = 16

            # Enable memory efficient attention if model supports it
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                print("Using memory-efficient attention mechanism")

    # Try to load the model
    try:
        # First try with the specified model
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        # If it fails, try alternatives
        print(f"Error loading {model_name}: {e}")
        alternative_models = [
            "bert-base-uncased",  # Simplest BERT model
            "google/bert-base-uncased",
        ]

        for alt_model in alternative_models:
            try:
                print(f"Trying alternative model: {alt_model}")
                tokenizer = AutoTokenizer.from_pretrained(alt_model)
                model = AutoModel.from_pretrained(alt_model).to(device)
                model.eval()
                print(f"Successfully loaded {alt_model}")
                break
            except Exception as alt_e:
                print(f"Error loading {alt_model}: {alt_e}")
        else:
            # If we get here, all alternatives failed
            raise RuntimeError(
                "Could not load any BERT model. Please check your transformers and torch installation."
            )

    # Get model's max sequence length - BERT models typically have 512 token limit
    try:
        # Try to get max length from model config
        max_length = (
            model.config.max_position_embeddings - 2
        )  # Account for [CLS] and [SEP]
        print(f"Using model's max sequence length: {max_length}")
    except:
        # Fall back to default
        max_length = 510  # Default for most BERT models (512 - 2 special tokens)
        print(f"Using default max sequence length: {max_length}")

    # For long inputs (~7000 tokens), we need to handle a large number of chunks effectively
    sliding_window_overlap = 50  # Token overlap between chunks to maintain context
    max_chunks_per_doc = 30  # Allow more chunks for very long documents (~15000 tokens)

    # Worker initialization and multiprocess handling is now handled at the module level
    # with the os.environ["TOKENIZERS_PARALLELISM"] = "false" setting

    def encode_with_chunks(df):
        """
        Encode text with a smart chunking approach optimized for RTX 3090.
        Uses batch processing and half-precision for better performance.
        """
        texts = df["tpcm_text"].tolist()
        embeddings = []

        # Use amp for faster processing if on GPU
        amp_enabled = (
            device == "cuda"
            and hasattr(torch.cuda, "amp")
            and torch.cuda.is_available()
        )

        if amp_enabled:
            print("Using automatic mixed precision for faster processing")

        # Process in batches for efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding DSL files"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            # Process each text in the batch
            for text in batch_texts:
                # Handle empty text case
                if not text or len(text.strip()) == 0:
                    # Create embedding for empty text
                    inputs = tokenizer(
                        " ",  # Use space instead of empty string to avoid errors
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        batch_embeddings.append(
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        )
                    continue

                # New token-aware chunking approach for long documents

                # First, tokenize the entire text to get token IDs
                # We need to know actual token counts, not just character counts
                tokenized_text = tokenizer.encode(
                    text,
                    add_special_tokens=False,  # Don't add special tokens yet
                    truncation=False,  # Don't truncate here
                    return_tensors=None,  # Just get the list of token IDs
                )

                # Now create chunks based on token count rather than character count
                token_count = len(tokenized_text)

                if token_count > max_length:
                    # For long documents, create overlapping chunks based on token counts
                    # This is more reliable than character-based chunking

                    # Calculate chunk sizes based on model's max length
                    effective_chunk_length = max_length - sliding_window_overlap

                    # Pre-process text to identify natural break points
                    # This helps us maintain meaningful semantic units
                    natural_breaks = []

                    # Get character positions for typical DSL block endings
                    for delimiter in ["}", ";", "\n\n"]:
                        positions = [
                            pos
                            for pos in range(len(text))
                            if text.startswith(delimiter, pos)
                        ]
                        # Mark each position with its delimiter type for later use
                        natural_breaks.extend([(pos, delimiter) for pos in positions])

                    # Sort breaks by position
                    natural_breaks.sort(key=lambda x: x[0])

                    # If no natural breaks found or they're too far apart, fall back to token-based chunking
                    if not natural_breaks:
                        print(
                            f"No natural breaks found in document of {token_count} tokens, using token-based chunking"
                        )
                        # Create overlapping chunks of tokens
                        chunks = []
                        for i in range(0, token_count, effective_chunk_length):
                            # Get token IDs for this chunk with overlap
                            end_idx = min(i + max_length, token_count)
                            # If not the last chunk, include overlap tokens
                            if (
                                end_idx < token_count
                                and end_idx - i > sliding_window_overlap
                            ):
                                chunk_tokens = tokenized_text[i:end_idx]
                            else:
                                # For last chunk, just take remaining tokens
                                chunk_tokens = tokenized_text[i:end_idx]

                            # Decode tokens back to text to maintain exact tokenization
                            chunk_text = tokenizer.decode(chunk_tokens)
                            chunks.append(chunk_text)
                    else:
                        # Use natural breaks to create meaningful chunks
                        chunks = []
                        start_pos = 0

                        # Get character length estimate based on token count
                        # This helps us find natural break points at approximately the right positions
                        char_ratio = len(text) / token_count
                        char_length_estimate = int(max_length * char_ratio)

                        while start_pos < len(text):
                            # Find the next natural break within our token limit
                            target_pos = start_pos + char_length_estimate
                            if target_pos >= len(text):
                                # Just take the rest of the text
                                chunks.append(text[start_pos:])
                                break

                            # Find break points after our target position
                            valid_breaks = [
                                b
                                for b in natural_breaks
                                if b[0] > start_pos and b[0] <= target_pos + 500
                            ]

                            if valid_breaks:
                                # Use the last break before exceeding our max length
                                break_pos = valid_breaks[-1][0]
                                chunks.append(text[start_pos : break_pos + 1])
                                start_pos = break_pos + 1
                            else:
                                # No good break found, use character estimate with a safety margin
                                safe_pos = min(target_pos, len(text) - 1)
                                chunks.append(text[start_pos:safe_pos])
                                start_pos = safe_pos

                            # Safety check for infinite loops
                            if len(chunks) >= max_chunks_per_doc:
                                # Add remaining text as final chunk
                                if start_pos < len(text):
                                    chunks.append(text[start_pos:])
                                break

                else:
                    # Short enough to process as a single chunk
                    chunks = [text]

                # Ensure we didn't create too many chunks (memory constraint)
                if len(chunks) > max_chunks_per_doc:
                    print(
                        f"Warning: Limiting document from {len(chunks)} to {max_chunks_per_doc} chunks"
                    )
                    # Prioritize start and end chunks, evenly sample the middle
                    if max_chunks_per_doc >= 3:
                        # Keep first chunk, last chunk, and sample from middle
                        middle_chunks = chunks[1:-1]
                        num_middle_chunks = max_chunks_per_doc - 2

                        # Sample middle chunks at regular intervals
                        step = len(middle_chunks) / num_middle_chunks
                        middle_indices = [
                            int(i * step) for i in range(num_middle_chunks)
                        ]
                        selected_middle = [middle_chunks[i] for i in middle_indices]

                        # Combine with first and last chunk
                        chunks = [chunks[0]] + selected_middle + [chunks[-1]]
                    else:
                        # Just take first N chunks if max_chunks_per_doc is very small
                        chunks = chunks[:max_chunks_per_doc]

                # Process chunks in a single batch to maximize GPU utilization
                try:
                    # Token-aware batching with attention to max_length
                    all_inputs = tokenizer(
                        chunks,
                        padding="max_length",  # Pad to max_length for consistent tensor shapes
                        truncation=True,  # Truncate if needed
                        return_tensors="pt",  # PyTorch tensors
                        max_length=max_length,  # Use model's max length
                        return_attention_mask=True,  # Get attention mask for proper pooling
                    ).to(device)

                    with torch.no_grad():
                        if amp_enabled:
                            with torch.amp.autocast("cuda"):
                                outputs = model(**all_inputs)
                                # Use CLS token embedding for each chunk
                                chunk_embeddings = (
                                    outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                )
                        else:
                            outputs = model(**all_inputs)
                            chunk_embeddings = (
                                outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            )
                except Exception as e:
                    print(f"Error processing chunks: {e}")
                    # Fallback: process one at a time
                    chunk_embeddings = []
                    for chunk in chunks:
                        try:
                            inputs = tokenizer(
                                chunk,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                max_length=max_length,
                            ).to(device)

                            with torch.no_grad():
                                outputs = model(**inputs)
                                embedding = (
                                    outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                                )
                                chunk_embeddings.append(embedding)
                        except Exception as chunk_e:
                            print(f"Error with chunk: {chunk_e}")
                            # Continue with next chunk

                    # Convert to numpy array if we have any successful embeddings
                    if chunk_embeddings:
                        chunk_embeddings = np.array(chunk_embeddings)

                # Enhanced chunk embedding combination - better for long documents
                if len(chunk_embeddings) > 1:
                    # Use a more sophisticated weighting scheme for better document representation

                    # 1. Position-aware weighting (beginning and end are often more important in DSL files)
                    num_chunks = len(chunk_embeddings)
                    # Create position weights that emphasize beginning, end, and have smooth weights for middle
                    pos_weights = np.ones(num_chunks)

                    # For DSL files, the beginning typically has important imports, type declarations
                    # and the end often has the main logic or processing code
                    if num_chunks >= 5:
                        # Boost beginning and end chunks, with slight emphasis on beginning
                        pos_weights[0] = 1.5  # First chunk (declarations, imports)
                        pos_weights[1] = (
                            1.2  # Second chunk (often contains type declarations)
                        )
                        pos_weights[-1] = 1.3  # Last chunk (often contains main logic)
                        pos_weights[-2] = 1.1  # Second to last chunk

                    # 2. Content-based weighting (longer chunks often contain more information)
                    # Normalize by total length to get proportions
                    content_weights = np.array([len(c) for c in chunks])
                    content_weights = content_weights / content_weights.sum()

                    # 3. Calculate token-density weighting (chunks with more non-whitespace tokens
                    #    likely contain more information)
                    # Count non-whitespace characters as a proxy for information density
                    non_ws_counts = np.array(
                        [sum(1 for char in c if not char.isspace()) for c in chunks]
                    )
                    if non_ws_counts.sum() > 0:  # Avoid division by zero
                        density_weights = non_ws_counts / non_ws_counts.sum()
                    else:
                        density_weights = np.ones(num_chunks) / num_chunks

                    # 4. Combine the different weighting schemes
                    # We'll use a weighted average of the different schemes
                    combined_weights = (
                        0.3 * pos_weights
                        + 0.4 * content_weights
                        + 0.3 * density_weights
                    )
                    # Normalize to sum to 1
                    final_weights = combined_weights / combined_weights.sum()

                    # Log weights for the first few documents (for debugging)
                    if len(embeddings) < 2:  # Only for the first document
                        print(f"Document chunking: {num_chunks} chunks with weights:")
                        for i, w in enumerate(final_weights):
                            print(
                                f"  Chunk {i}: weight={w:.3f}, length={len(chunks[i])}"
                            )

                    # Apply the weights to create a combined embedding
                    weighted_emb = np.zeros_like(chunk_embeddings[0])
                    for j, emb in enumerate(chunk_embeddings):
                        weighted_emb += emb * final_weights[j]

                    # Add normalized version for better consistency
                    weighted_emb_norm = weighted_emb / np.linalg.norm(weighted_emb)
                    batch_embeddings.append(weighted_emb_norm)

                elif len(chunk_embeddings) == 1:
                    # Single chunk - just normalize it
                    embedding = chunk_embeddings[0]
                    normalized_emb = embedding / np.linalg.norm(embedding)
                    batch_embeddings.append(normalized_emb)

                else:
                    # Fallback for empty chunks (shouldn't happen)
                    print("Warning: No chunk embeddings created, using fallback")
                    inputs = tokenizer(
                        " ",  # Single space to avoid errors with empty string
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        fallback_emb = (
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        )
                        # Normalize
                        fallback_emb = fallback_emb / np.linalg.norm(fallback_emb)
                        batch_embeddings.append(fallback_emb)

            # Add batch embeddings to overall result
            embeddings.extend(batch_embeddings)

            # Clear cache periodically to prevent memory buildup
            if device == "cuda" and (i + batch_size) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        # Final cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

        # Stack all embeddings into a single array
        if not embeddings:
            # Handle case where no embeddings were created
            print("Warning: No embeddings were created, returning zero vector")
            dummy_embedding = (
                model(tokenizer(" ", return_tensors="pt").to(device))
                .last_hidden_state[:, 0, :]
                .cpu()
                .numpy()[0]
            )
            return np.zeros((1, dummy_embedding.shape[0]))

        return np.vstack(embeddings)

    # Process datasets with progress reporting
    print(f"Processing training set ({len(train_samples)} samples)...")
    X_train = encode_with_chunks(train_samples)
    y_train = extract_targets(train_samples)

    print(f"Processing validation set ({len(val_samples)} samples)...")
    X_val = encode_with_chunks(val_samples)
    y_val = extract_targets(val_samples)

    print(f"Processing test set ({len(test_samples)} samples)...")
    X_test = encode_with_chunks(test_samples)
    y_test = extract_targets(test_samples)

    print("Feature extraction complete!")
    print(f"Features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_val={X_val.shape}, X_test={X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
