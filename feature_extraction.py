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

# Set tokenizers parallelism explicitly before import/usage
# This prevents the warning when using DataLoader with num_workers > 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    print(f"Setting up embeddings with {model_name}")

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

    # Create a function to encode DSL files with a Llama model
    def encode_with_llama(df):
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
        for i in tqdm(
            range(0, len(texts), actual_batch_size), desc=f"Encoding with {model_name}"
        ):
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
                    inputs = tokenizer(" ", return_tensors="pt", padding=True).to(
                        device
                    )
                    with torch.no_grad():
                        # For CausalLM models, use the hidden states
                        if isinstance(model, AutoModelForCausalLM):
                            outputs = model(**inputs, output_hidden_states=True)
                            # Use last layer's hidden state of the last token
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
                                        b
                                        for b in valid_breaks
                                        if b[1] == "code_structure"
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
                    max_chunks = 8 if "13b" in model_name.lower() else 12
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
                                        outputs = model(
                                            **inputs, output_hidden_states=True
                                        )
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
                                        outputs.last_hidden_state[:, 0, :]
                                        .cpu()
                                        .numpy()[0]
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

    # Process datasets
    print(f"Processing training set ({len(train_samples)} samples)...")
    X_train = encode_with_llama(train_samples)
    y_train = extract_targets(train_samples)

    print(f"Processing validation set ({len(val_samples)} samples)...")
    X_val = encode_with_llama(val_samples)
    y_val = extract_targets(val_samples)

    print(f"Processing test set ({len(test_samples)} samples)...")
    X_test = encode_with_llama(test_samples)
    y_test = extract_targets(test_samples)

    print("Feature extraction complete!")
    print(f"Features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_val={X_val.shape}, X_test={X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model


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
